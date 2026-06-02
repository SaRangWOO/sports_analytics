from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engineering import build_features
from .game_level_features import (
    align_game_level_matrix,
    attach_player_context,
    attach_pitching_context,
    build_game_level_frame,
    build_player_team_context,
    prepare_game_level_matrix,
)
from .model_evaluation import (
    calibration_table,
    confidence_metrics,
    normalize_game_probabilities,
    pick_better_model,
    probability_scores,
)
from .run_expectancy import export_run_expectancy_dataset
from .train_win_predictor import prepare_matrix, sigmoid, standardize_train_test, train_logistic_regression


STREAK_FEATURES = {
    "team_current_streak_length",
    "team_current_streak_type",
    "opponent_current_streak_length",
    "opponent_current_streak_type",
    "streak_length_gap",
    "team_losing_streak_flag",
    "team_winning_streak_flag",
    "opponent_losing_streak_flag",
    "opponent_winning_streak_flag",
    "winning_streak_regression_risk",
    "winning_streak_with_low_run_diff",
    "winning_streak_after_close_games",
    "winning_streak_bullpen_fatigue_proxy",
    "losing_streak_with_negative_run_diff",
    "losing_streak_allowed_runs_spike",
    "losing_streak_low_scoring_offense",
    "opponent_vs_losing_streak_flag",
}


def non_streak_columns(columns):
    return [column for column in columns if column not in STREAK_FEATURES]


def align_prediction_matrix(features: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series):
    x, _ = prepare_matrix(features)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return (x - mean) / std.replace(0, 1)


def prediction_reason(row: pd.Series, predicted_team: str | None = None):
    predicted_team = predicted_team or row.get("team")
    team_perspective = predicted_team == row.get("team")
    reasons = []
    season_gap = row.get("season_win_rate_gap", 0) if team_perspective else -row.get("season_win_rate_gap", 0)
    recent_gap = row.get("recent_5_win_rate_gap", 0) if team_perspective else -row.get("recent_5_win_rate_gap", 0)
    recent_10_gap = row.get("recent_10_win_rate_gap", 0) if team_perspective else -row.get("recent_10_win_rate_gap", 0)
    h2h_gap = row.get("head_to_head_win_rate_gap", 0) if team_perspective else -row.get("head_to_head_win_rate_gap", 0)
    venue_gap = row.get("venue_win_rate_gap", 0) if team_perspective else -row.get("venue_win_rate_gap", 0)
    season_run_gap = row.get("season_avg_run_diff_gap", 0) if team_perspective else -row.get("season_avg_run_diff_gap", 0)
    own_run_diff = row.get("avg_run_diff_last_5", 0) if team_perspective else row.get("opponent_avg_run_diff_last_5", 0)
    opponent_run_diff = row.get("opponent_avg_run_diff_last_5", 0) if team_perspective else row.get("avg_run_diff_last_5", 0)
    is_home_side = row.get("is_home", 0) == 1 if team_perspective else row.get("is_home", 0) == 0

    if season_gap > 0.03:
        reasons.append(f"{predicted_team} 시즌 누적 승률 우위")
    if recent_gap > 0.15:
        reasons.append(f"{predicted_team} 최근 5경기 흐름 우위")
    if recent_10_gap > 0.12:
        reasons.append(f"{predicted_team} 최근 10경기 흐름 우위")
    if h2h_gap > 0.2:
        reasons.append(f"{predicted_team} 시즌 상대전적 우위")
    if venue_gap > 0.15:
        reasons.append(f"{predicted_team} 홈/원정 성향 우위")
    if season_run_gap > 0.5:
        reasons.append(f"{predicted_team} 시즌 득실차 우위")
    if own_run_diff > opponent_run_diff + 0.8:
        reasons.append(f"{predicted_team} 최근 득실차 우위")
    if is_home_side:
        reasons.append(f"{predicted_team} 홈 경기")

    return ", ".join(reasons[:2]) if reasons else "양 팀 지표가 비슷해 기본 전력과 최근 흐름을 종합"


def game_prediction_reason(row: pd.Series, predicted_team: str):
    home_perspective = predicted_team == row.get("home_team")
    reasons = []
    recent_gap = row.get("recent_10_win_rate_gap", 0) if home_perspective else -row.get("recent_10_win_rate_gap", 0)
    season_gap = row.get("season_win_rate_gap", 0) if home_perspective else -row.get("season_win_rate_gap", 0)
    run_gap = row.get("season_avg_run_diff_gap", 0) if home_perspective else -row.get("season_avg_run_diff_gap", 0)
    recent_run_gap = row.get("recent_run_diff_10_gap", 0) if home_perspective else -row.get("recent_run_diff_10_gap", 0)
    venue_gap = row.get("venue_win_rate_gap", 0) if home_perspective else -row.get("venue_win_rate_gap", 0)
    bullpen_gap = row.get("bullpen_fatigue_gap", 0) if home_perspective else -row.get("bullpen_fatigue_gap", 0)
    rest_gap = row.get("rest_days_gap", 0) if home_perspective else -row.get("rest_days_gap", 0)

    if bullpen_gap > 1.0:
        reasons.append(f"{predicted_team} 불펜 피로 부담 낮음")
    if rest_gap > 0:
        reasons.append(f"{predicted_team} 휴식일 우위")
    if recent_run_gap > 0.8:
        reasons.append(f"{predicted_team} 최근 득실차 우위")
    if recent_gap > 0.12:
        reasons.append(f"{predicted_team} 최근 10경기 흐름 우위")
    if season_gap > 0.03:
        reasons.append(f"{predicted_team} 시즌 승률 우위")
    if run_gap > 0.5:
        reasons.append(f"{predicted_team} 시즌 득실차 우위")
    if venue_gap > 0.15:
        reasons.append(f"{predicted_team} 홈/원정 성향 우위")
    if home_perspective:
        reasons.append(f"{predicted_team} 홈 경기")

    return ", ".join(reasons[:2]) if reasons else "양 팀 지표가 비슷해 기본 전력과 최근 흐름을 종합"


def chronological_split_index(dates: pd.Series, train_ratio: float = 0.8):
    ordered_dates = pd.to_datetime(dates)
    unique_dates = ordered_dates.drop_duplicates().sort_values()
    date_index = max(int(len(unique_dates) * train_ratio), 1)
    date_index = min(date_index, len(unique_dates) - 1)
    cutoff_date = unique_dates.iloc[date_index]
    split_index = int((ordered_dates < cutoff_date).sum())
    return max(min(split_index, len(ordered_dates) - 1), 1)


def probability_distribution(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"total_games": 0, "p50": None, "p75": None, "p90": None, "max": None, "over_53": 0, "over_55": 0, "over_58": 0, "over_60": 0}
    confidence = np.maximum(values, 1 - values)
    return {
        "total_games": int(len(confidence)),
        "p50": round(float(np.percentile(confidence, 50)), 3),
        "p75": round(float(np.percentile(confidence, 75)), 3),
        "p90": round(float(np.percentile(confidence, 90)), 3),
        "max": round(float(confidence.max()), 3),
        "over_53": int((confidence >= 0.53).sum()),
        "over_55": int((confidence >= 0.55).sum()),
        "over_58": int((confidence >= 0.58).sum()),
        "over_60": int((confidence >= 0.60).sum()),
    }


def model_probability_spread(model_name: str, y_true: np.ndarray, probability: np.ndarray, accuracy: float, score: dict):
    confidence = np.maximum(probability, 1 - probability)
    pred = (probability >= 0.5).astype(int)
    over_55_mask = confidence >= 0.55
    return {
        "model": model_name,
        "total_games": int(len(probability)),
        "accuracy": round(float(accuracy), 3),
        "brier": score["Brier Score"],
        "logloss": score["Log Loss"],
        "avg_confidence": round(float(confidence.mean()), 3) if len(confidence) else None,
        "p75_confidence": round(float(np.percentile(confidence, 75)), 3) if len(confidence) else None,
        "p90_confidence": round(float(np.percentile(confidence, 90)), 3) if len(confidence) else None,
        "max_confidence": round(float(confidence.max()), 3) if len(confidence) else None,
        "over_55": int(over_55_mask.sum()),
        "over_58": int((confidence >= 0.58).sum()),
        "over_60": int((confidence >= 0.60).sum()),
        "over_55_accuracy": round(float((pred[over_55_mask] == y_true[over_55_mask]).mean()), 3) if over_55_mask.any() else None,
    }


def streak_experiment_metrics(model_name: str, feature_set: str, frame: pd.DataFrame, y_true: np.ndarray, probability: np.ndarray, selected_candidate: bool):
    pred = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1 - probability)
    over_55_mask = confidence >= 0.55
    winning_mask = frame.get("team_winning_streak_flag", pd.Series(0, index=frame.index)).to_numpy() == 1
    losing_mask = frame.get("team_losing_streak_flag", pd.Series(0, index=frame.index)).to_numpy() == 1
    close_mask = confidence < 0.53
    score = probability_scores(y_true, probability)

    def accuracy_for(mask):
        return round(float((pred[mask] == y_true[mask]).mean()), 3) if mask.any() else None

    return {
        "model": model_name,
        "feature_set": feature_set,
        "accuracy": round(float((pred == y_true).mean()), 3),
        "brier": score["Brier Score"],
        "log_loss": score["Log Loss"],
        "over_55_games": int(over_55_mask.sum()),
        "over_55_accuracy": accuracy_for(over_55_mask),
        "winning_streak_accuracy": accuracy_for(winning_mask),
        "losing_streak_accuracy": accuracy_for(losing_mask),
        "close_game_accuracy": accuracy_for(close_mask),
        "selected_candidate": selected_candidate,
    }


def write_streak_feature_experiment_report(results_dir: Path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(results_dir / "streak_feature_experiment_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_model_probability_spread_report(results_dir: Path, rows: list[dict]):
    output = results_dir / "model_probability_spread_report.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return rows


def confidence_bucket_policy(y_true: np.ndarray, probability: np.ndarray):
    confidence = np.maximum(probability, 1 - probability)
    pred = (probability >= 0.5).astype(int)
    correct = pred == y_true
    overall_accuracy = float(correct.mean()) if len(correct) else 0.0
    top20_threshold = float(np.percentile(confidence, 80)) if len(confidence) else 1.0
    top20_mask = confidence >= top20_threshold
    top20_accuracy = float(correct[top20_mask].mean()) if top20_mask.any() else 0.0
    return {
        "confidence_thresholds": {
            "top_20_percent_confidence": round(top20_threshold, 3),
            "recommendation_enabled": bool(top20_mask.any() and top20_accuracy > overall_accuracy),
            "recommendation_rule": "confidence가 백테스트 상위 20% 구간이고 해당 구간 적중률이 전체 적중률보다 높을 때 추천 후보로 표시",
        },
        "confidence_bucket_performance": {
            "overall_accuracy": round(overall_accuracy, 3),
            "top_20_percent_accuracy": round(top20_accuracy, 3),
            "top_20_percent_games": int(top20_mask.sum()),
        },
    }


def write_probability_distribution_report(results_dir: Path, backtest_probability, today_probability):
    rows = []
    for split, values in [("backtest", backtest_probability), ("today", today_probability)]:
        row = {"split": split}
        row.update(probability_distribution(values))
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "probability_distribution_report.csv", index=False, encoding="utf-8-sig")
    return rows


def sklearn_candidate_specs(recency_weight):
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return []

    return [
        ("LogisticRegression 보수 모델", LogisticRegression(C=0.35, class_weight="balanced", max_iter=1500, random_state=42), None),
        ("RandomForest 비선형 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), None),
        ("RandomForest 시간가중 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), recency_weight),
        ("RandomForest 보수 모델", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), None),
        ("RandomForest 보수 시간가중 모델", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), recency_weight),
        ("ExtraTrees 보수 모델", ExtraTreesClassifier(n_estimators=700, max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1), None),
        ("GradientBoosting 비선형 모델", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42), None),
        ("GradientBoosting 시간가중 모델", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42), recency_weight),
        ("GradientBoosting 보수 모델", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=10, l2_regularization=0.15, random_state=42), None),
        ("GradientBoosting 보수 시간가중 모델", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=10, l2_regularization=0.15, random_state=42), recency_weight),
        (
            "GradientBoosting 확률보정(sigmoid)",
            CalibratedClassifierCV(
                estimator=HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42),
                method="sigmoid",
                cv=3,
            ),
            None,
        ),
        (
            "GradientBoosting 확률보정(isotonic)",
            CalibratedClassifierCV(
                estimator=HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42),
                method="isotonic",
                cv=3,
            ),
            None,
        ),
    ]


def compact_feature_columns(x: pd.DataFrame):
    return [
        col
        for col in [
            "is_home",
            "rest_days",
            "recent_5_win_rate",
            "recent_10_win_rate",
            "avg_run_diff_last_5",
            "avg_run_diff_last_10",
            "season_win_rate_prior",
            "opponent_recent_5_win_rate",
            "opponent_recent_10_win_rate",
            "opponent_avg_run_diff_last_5",
            "opponent_avg_run_diff_last_10",
            "season_win_rate_gap",
            "recent_5_win_rate_gap",
            "recent_10_win_rate_gap",
            "season_avg_run_diff_gap",
            "recent_run_diff_10_gap",
            "venue_win_rate_gap",
            "head_to_head_win_rate_gap",
            "elo_diff",
            "games_last_7_days",
            "back_to_back",
        ]
        if col in x.columns
    ]


def streak_feature_columns(x: pd.DataFrame):
    columns = compact_feature_columns(x) + list(STREAK_FEATURES)
    return [column for column in columns if column in x.columns]


def compact_sklearn_candidate_specs(recency_weight):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError:
        return []

    return [
        ("핵심 수치 RandomForest 보수 모델", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), None),
        ("핵심 수치 RandomForest 보수 시간가중 모델", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), recency_weight),
        ("핵심 수치 GradientBoosting 보수 모델", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=10, l2_regularization=0.15, random_state=42), None),
        ("핵심 수치 GradientBoosting 보수 시간가중 모델", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=10, l2_regularization=0.15, random_state=42), recency_weight),
    ]


def streak_candidate_specs(recency_weight):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError:
        return []

    return [
        ("Streak 피처 RandomForest 실험", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), None),
        ("Streak 피처 RandomForest 시간가중 실험", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), recency_weight),
        ("Streak 피처 GradientBoosting 실험", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=10, l2_regularization=0.15, random_state=42), None),
        ("Streak 피처 GradientBoosting 시간가중 실험", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=10, l2_regularization=0.15, random_state=42), recency_weight),
    ]


def baseball_feature_columns(x: pd.DataFrame):
    candidates = [
        "recent_10_win_rate_gap",
        "season_win_rate_gap",
        "season_avg_run_diff_gap",
        "recent_run_diff_10_gap",
        "venue_win_rate_gap",
        "rest_days_gap",
        "games_last_7_days_gap",
        "home_recent_3day_games",
        "away_recent_3day_games",
        "recent_3day_games_gap",
        "home_bullpen_fatigue_score",
        "away_bullpen_fatigue_score",
        "bullpen_fatigue_score_gap",
        "home_recent_5_runs_avg",
        "away_recent_5_runs_avg",
        "recent_5_runs_avg_gap",
        "home_recent_5_allowed_avg",
        "away_recent_5_allowed_avg",
        "recent_5_allowed_avg_gap",
        "recent_5_run_creation_gap",
        "recent_10_run_creation_gap",
        "home_starter_era",
        "away_starter_era",
        "starter_era_gap",
        "home_starter_whip",
        "away_starter_whip",
        "starter_whip_gap",
        "home_starter_info_quality",
        "away_starter_info_quality",
        "home_starter_quality_score",
        "away_starter_quality_score",
        "starter_quality_gap",
        "both_starters_confirmed",
        "partial_starter_confirmed",
    ]
    return [column for column in candidates if column in x.columns]


def baseball_candidate_specs():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError:
        return []
    return [
        ("야구 핵심 피처 강화 모델", RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ("야구 핵심 피처 GradientBoosting 모델", HistGradientBoostingClassifier(max_iter=260, learning_rate=0.035, max_leaf_nodes=12, l2_regularization=0.12, random_state=42)),
        ("야구 핵심 피처 RandomForest 모델", RandomForestClassifier(n_estimators=700, max_depth=7, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ]


def evaluate_model(training_games: pd.DataFrame, current_games: pd.DataFrame, cutoff: date, prediction_date: date, data_dir: Path, results_dir: Path):
    training_games = training_games.copy()
    training_games["date"] = pd.to_datetime(training_games["date"])
    completed = training_games[
        (training_games["status"] == "Final")
        & ((training_games["date"].dt.year < cutoff.year) | (training_games["date"].dt.date <= cutoff))
    ].copy()
    model_input = data_dir / "model_training_games.csv"
    completed.to_csv(model_input, index=False, encoding="utf-8-sig")
    features = build_features(model_input)
    results_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(results_dir / "features.csv", index=False, encoding="utf-8-sig")
    pitching_context_path = data_dir / "pitching_context.csv"
    pitching_context = pd.read_csv(pitching_context_path) if pitching_context_path.exists() else pd.DataFrame()
    game_level_features = attach_pitching_context(build_game_level_frame(features), pitching_context)
    game_level_features.to_csv(results_dir / "game_level_features.csv", index=False, encoding="utf-8-sig")
    run_expectancy_frame = export_run_expectancy_dataset(features, completed, results_dir / "run_expectancy_features.csv")
    player_feature_note = ""
    player_game_frame = pd.DataFrame()
    hitter_stats_path = data_dir / "hitter_stats.csv"
    pitcher_stats_path = data_dir / "pitcher_stats.csv"
    if hitter_stats_path.exists() and pitcher_stats_path.exists():
        hitters = pd.read_csv(hitter_stats_path)
        pitchers = pd.read_csv(pitcher_stats_path)
        player_context = build_player_team_context(hitters, pitchers)
        player_context.to_csv(results_dir / "player_team_context.csv", index=False, encoding="utf-8-sig")
        player_game_frame = attach_player_context(game_level_features, player_context)
        player_game_frame.to_csv(results_dir / "game_level_player_features.csv", index=False, encoding="utf-8-sig")
        player_feature_note = (
            "선수 영향도 피처는 현재 공식 기록 스냅샷 기반 참고 피처입니다. "
            "과거 시점별 선수 스냅샷이 쌓이면 최종 모델 선택 후보로 승격할 수 있습니다."
        )

    if len(features) < 20:
        return {"available": False, "reason": "학습 가능한 완료 경기가 부족합니다.", "training_cutoff": cutoff.isoformat()}

    x, y = prepare_matrix(features)
    split_index = chronological_split_index(features["date"])
    y_train, y_test = y[:split_index], y[split_index:]
    train_years = pd.to_datetime(features.iloc[:split_index]["date"]).dt.year
    max_train_year = int(train_years.max())
    recency_weight = (0.85 ** (max_train_year - train_years)).clip(lower=0.35).to_numpy(dtype=float)

    candidate_columns = {
        "기본 흐름 모델": [col for col in non_streak_columns(x.columns) if col not in {"team_elo_pre", "opponent_elo_pre", "elo_diff", "games_last_7_days", "back_to_back"}],
        "전력/일정 피로도 포함 모델": non_streak_columns(x.columns),
        "핵심 수치 모델": compact_feature_columns(x),
    }
    best = None
    candidate_results = []
    probability_spread_rows = []
    streak_experiment_rows = []

    for name, columns in candidate_columns.items():
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        weights, bias = train_logistic_regression(train_scaled.to_numpy(), y_train, lr=0.05, epochs=3500)
        probability = normalize_game_probabilities(features.iloc[split_index:], sigmoid(test_scaled.to_numpy() @ weights + bias))
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "weights": weights, "bias": bias, "model_type": "from_scratch_logistic_regression", "prediction_unit": "team", "test_scaled": test_scaled, "test_frame": features.iloc[split_index:].copy(), "y_test": y_test}
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        probability_spread_rows.append(model_probability_spread(name, y_test, probability, accuracy, score))
        best = pick_better_model(best, result)

    for name, model, sample_weight in compact_sklearn_candidate_specs(recency_weight):
        columns = compact_feature_columns(x)
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(train_scaled, y_train, **fit_kwargs)
        probability = normalize_game_probabilities(features.iloc[split_index:], model.predict_proba(test_scaled)[:, 1])
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "model": model, "model_type": model.__class__.__name__, "prediction_unit": "team", "test_scaled": test_scaled, "test_frame": features.iloc[split_index:].copy(), "y_test": y_test}
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        probability_spread_rows.append(model_probability_spread(name, y_test, probability, accuracy, score))
        streak_experiment_rows.append(streak_experiment_metrics(name, "compact_baseline", features.iloc[split_index:].copy(), y_test, probability, False))
        best = pick_better_model(best, result)

    for name, model, sample_weight in streak_candidate_specs(recency_weight):
        columns = streak_feature_columns(x)
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(train_scaled, y_train, **fit_kwargs)
        probability = normalize_game_probabilities(features.iloc[split_index:], model.predict_proba(test_scaled)[:, 1])
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        probability_spread_rows.append(model_probability_spread(name, y_test, probability, accuracy, score))
        streak_experiment_rows.append(streak_experiment_metrics(name, "compact_plus_streak", features.iloc[split_index:].copy(), y_test, probability, False))

    sklearn_candidates = sklearn_candidate_specs(recency_weight)
    for name, model, sample_weight in sklearn_candidates:
        columns = non_streak_columns(x.columns)
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(train_scaled, y_train, **fit_kwargs)
        probability = normalize_game_probabilities(features.iloc[split_index:], model.predict_proba(test_scaled)[:, 1])
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "model": model, "model_type": model.__class__.__name__, "prediction_unit": "team", "test_scaled": test_scaled, "test_frame": features.iloc[split_index:].copy(), "y_test": y_test}
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        probability_spread_rows.append(model_probability_spread(name, y_test, probability, accuracy, score))
        best = pick_better_model(best, result)

    game_frame = game_level_features.dropna(subset=["target_home_win"]).copy()
    if sklearn_candidates and len(game_frame) >= 20:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

        gx, gy = prepare_game_level_matrix(game_frame)
        game_split = chronological_split_index(game_frame["date"])
        gy_train, gy_test = gy[:game_split], gy[game_split:]
        game_years = pd.to_datetime(game_frame.iloc[:game_split]["date"]).dt.year
        max_game_year = int(game_years.max())
        game_recency_weight = (0.85 ** (max_game_year - game_years)).clip(lower=0.35).to_numpy(dtype=float)
        game_candidates = [
            ("경기 단위 RandomForest 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), None),
            ("경기 단위 RandomForest 시간가중 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), game_recency_weight),
            ("경기 단위 GradientBoosting 모델", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42), None),
        ]
        for name, model, sample_weight in game_candidates:
            columns = list(gx.columns)
            gx_train, gx_test = gx.iloc[:game_split][columns], gx.iloc[game_split:][columns]
            train_scaled, test_scaled, mean, std = standardize_train_test(gx_train, gx_test)
            fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            model.fit(train_scaled, gy_train, **fit_kwargs)
            probability = model.predict_proba(test_scaled)[:, 1]
            pred = (probability >= 0.5).astype(int)
            accuracy = round(float((pred == gy_test).mean()), 3)
            score = probability_scores(gy_test, probability)
            result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "model": model, "model_type": model.__class__.__name__, "prediction_unit": "game", "test_scaled": test_scaled, "test_frame": game_frame.iloc[game_split:].copy(), "y_test": gy_test}
            candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
            probability_spread_rows.append(model_probability_spread(name, gy_test, probability, accuracy, score))
            best = pick_better_model(best, result)

        baseball_columns = baseball_feature_columns(gx)
        for name, model in baseball_candidate_specs():
            gx_train, gx_test = gx.iloc[:game_split][baseball_columns], gx.iloc[game_split:][baseball_columns]
            train_scaled, test_scaled, mean, std = standardize_train_test(gx_train, gx_test)
            model.fit(train_scaled, gy_train)
            probability = model.predict_proba(test_scaled)[:, 1]
            pred = (probability >= 0.5).astype(int)
            accuracy = round(float((pred == gy_test).mean()), 3)
            score = probability_scores(gy_test, probability)
            result = {"name": name, "columns": baseball_columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "model": model, "model_type": model.__class__.__name__, "prediction_unit": "game", "test_scaled": test_scaled, "test_frame": game_frame.iloc[game_split:].copy(), "y_test": gy_test}
            candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(baseball_columns), **score})
            probability_spread_rows.append(model_probability_spread(name, gy_test, probability, accuracy, score))
            best = pick_better_model(best, result)

    if sklearn_candidates and not player_game_frame.empty:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

        player_frame = player_game_frame.dropna(subset=["target_home_win"]).copy()
        if len(player_frame) >= 20:
            px, py = prepare_game_level_matrix(player_frame)
            player_split = chronological_split_index(player_frame["date"])
            py_train, py_test = py[:player_split], py[player_split:]
            player_candidates = [
                ("경기 단위 선수영향도 참고 RandomForest", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1)),
                ("경기 단위 선수영향도 참고 GradientBoosting", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42)),
            ]
            for name, model in player_candidates:
                columns = list(px.columns)
                px_train, px_test = px.iloc[:player_split][columns], px.iloc[player_split:][columns]
                train_scaled, test_scaled, _, _ = standardize_train_test(px_train, px_test)
                model.fit(train_scaled, py_train)
                probability = model.predict_proba(test_scaled)[:, 1]
                pred = (probability >= 0.5).astype(int)
                accuracy = round(float((pred == py_test).mean()), 3)
                score = probability_scores(py_test, probability)
                candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
                probability_spread_rows.append(model_probability_spread(name, py_test, probability, accuracy, score))

    for row in streak_experiment_rows:
        row["selected_candidate"] = row["model"] == best["name"]
    streak_report = write_streak_feature_experiment_report(results_dir, streak_experiment_rows)
    spread_report = write_model_probability_spread_report(results_dir, probability_spread_rows)
    payload = build_payload(best, candidate_results, features, split_index, y_test, current_games, cutoff, prediction_date, data_dir, results_dir, completed, training_games, spread_report, streak_experiment_rows)
    payload["streak_feature_experiment_report"] = "modeling/results/streak_feature_experiment_report.csv"
    payload["streak_feature_experiment_rows"] = len(streak_report)
    payload.setdefault("diagnostic_reports", {})["streak_feature_experiment_report"] = "modeling/results/streak_feature_experiment_report.csv"
    if payload.get("feature_importance"):
        pd.DataFrame(
            [{"feature": feature, "importance": importance} for feature, importance in payload["feature_importance"].items()]
        ).to_csv(results_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    if player_feature_note:
        payload["player_feature_note"] = player_feature_note
        payload["player_feature_files"] = [
            "modeling/results/player_team_context.csv",
            "modeling/results/game_level_player_features.csv",
        ]
    payload["run_expectancy_note"] = (
        "득점 예측용 데이터셋은 완료 경기의 홈/원정 득점 목표값과 경기 전 팀 흐름 피처를 한 경기 1행으로 저장합니다. "
        "현재 단계에서는 모델 학습 전 데이터셋 준비용입니다."
    )
    payload["run_expectancy_files"] = ["modeling/results/run_expectancy_features.csv"]
    payload["run_expectancy_rows"] = int(len(run_expectancy_frame))
    (results_dir / "win_predictor_model.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def summarize_team_backtest_by_game(recent: pd.DataFrame):
    rows = []
    if recent.empty:
        return rows

    recent = recent.copy()
    if "game_id" in recent.columns:
        recent["_game_key"] = recent["game_id"].astype(str).str.replace(r"_[^_]+$", "", regex=True)
        group_key = "_game_key"
    else:
        recent["_game_key"] = recent.apply(lambda row: f'{row["경기일"]}_{"_".join(sorted([str(row["team"]), str(row["opponent"])]))}', axis=1)
        group_key = "_game_key"
    for _, game in recent.groupby(group_key, sort=False):
        game = game.copy()
        home_rows = game[game.get("is_home", 0) == 1] if "is_home" in game.columns else pd.DataFrame()
        home_team = home_rows.iloc[0]["team"] if not home_rows.empty else game.iloc[0]["team"]
        away_rows = game[game.get("is_home", 0) == 0] if "is_home" in game.columns else pd.DataFrame()
        away_team = away_rows.iloc[0]["team"] if not away_rows.empty else game.iloc[0]["opponent"]
        pick_row = game.sort_values("_prediction_probability", ascending=False).iloc[0]
        actual_rows = game[game["target_win"] == 1]
        actual_team = actual_rows.iloc[0]["team"] if not actual_rows.empty else "무승부"
        predicted_team = pick_row["예측 구단"]
        rows.append(
            {
                "경기일": pick_row["경기일"],
                "경기": f"{away_team} vs {home_team}",
                "예측 구단": predicted_team,
                "예측승률": f'{float(pick_row["_prediction_probability"]):.1%}',
                "실제 승리 구단": actual_team,
                "결과": "적중" if predicted_team == actual_team else "오답",
                "예측 근거": pick_row["예측 근거"],
            }
        )
    return rows


def selected_model_probability(best: dict, matrix: pd.DataFrame):
    if best["model_type"] == "from_scratch_logistic_regression":
        raw_probability = sigmoid(matrix.to_numpy() @ best["weights"] + best["bias"])
    else:
        raw_probability = best["model"].predict_proba(matrix)[:, 1]
    if best.get("prediction_unit", "team") == "team":
        return normalize_game_probabilities(best["test_frame"], raw_probability)
    return raw_probability


def permutation_importance(best: dict, y_eval: np.ndarray):
    matrix = best.get("test_scaled")
    if matrix is None or len(matrix) == 0:
        return {}
    baseline_probability = best["probability"]
    baseline_brier = probability_scores(y_eval, baseline_probability)["Brier Score"]
    rng = np.random.default_rng(42)
    importances = []
    for feature in best["columns"]:
        if feature not in matrix.columns:
            continue
        shuffled = matrix.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        permuted_probability = selected_model_probability(best, shuffled)
        permuted_brier = probability_scores(y_eval, permuted_probability)["Brier Score"]
        importances.append((feature, max(0.0, float(permuted_brier - baseline_brier))))
    return {name: round(value, 6) for name, value in sorted(importances, key=lambda x: x[1], reverse=True)}


def segment_metrics(segment: str, y_true: np.ndarray, probability: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return {
            "segment": segment,
            "total_games": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "avg_confidence": None,
            "over_55_games": 0,
            "over_55_accuracy": None,
        }
    y_segment = y_true[mask]
    p_segment = probability[mask]
    pred = (p_segment >= 0.5).astype(int)
    confidence = np.maximum(p_segment, 1 - p_segment)
    over_55 = confidence >= 0.55
    score = probability_scores(y_segment, p_segment)
    return {
        "segment": segment,
        "total_games": int(mask.sum()),
        "accuracy": round(float((pred == y_segment).mean()), 3),
        "brier": score["Brier Score"],
        "log_loss": score["Log Loss"],
        "avg_confidence": round(float(confidence.mean()), 3),
        "over_55_games": int(over_55.sum()),
        "over_55_accuracy": round(float((pred[over_55] == y_segment[over_55]).mean()), 3) if over_55.any() else None,
    }


def write_game_type_performance_report(results_dir: Path, frame: pd.DataFrame, y_true: np.ndarray, probability: np.ndarray):
    confidence = np.maximum(probability, 1 - probability)
    masks = {
        "전체": np.ones(len(frame), dtype=bool),
        "홈팀 관점 행": frame.get("is_home", pd.Series(0, index=frame.index)).to_numpy() == 1,
        "원정팀 관점 행": frame.get("is_home", pd.Series(0, index=frame.index)).to_numpy() == 0,
        "최근 5경기 흐름 차이 큼": frame.get("recent_5_win_rate_gap", pd.Series(0, index=frame.index)).abs().to_numpy() >= 0.2,
        "시즌 승률 차이 큼": frame.get("season_win_rate_gap", pd.Series(0, index=frame.index)).abs().to_numpy() >= 0.08,
        "득실차 차이 큼": frame.get("season_avg_run_diff_gap", pd.Series(0, index=frame.index)).abs().to_numpy() >= 1.0,
        "휴식일 차이 큼": frame.get("rest_days_gap", pd.Series(0, index=frame.index)).abs().to_numpy() >= 1.0,
        "불펜 피로 차이 큼": frame.get("bullpen_fatigue_score_gap", pd.Series(0, index=frame.index)).abs().to_numpy() >= 1.0,
        "박빙 경기": confidence < 0.53,
        "강팀 vs 약팀": frame.get("season_win_rate_gap", pd.Series(0, index=frame.index)).abs().to_numpy() >= 0.12,
        "연승 흐름": frame.get("recent_5_win_rate", pd.Series(0.5, index=frame.index)).to_numpy() >= 0.8,
        "연패 흐름": frame.get("recent_5_win_rate", pd.Series(0.5, index=frame.index)).to_numpy() <= 0.2,
        "명시 연승 streak": frame.get("team_winning_streak_flag", pd.Series(0, index=frame.index)).to_numpy() == 1,
        "명시 연패 streak": frame.get("team_losing_streak_flag", pd.Series(0, index=frame.index)).to_numpy() == 1,
        "연승 회귀 위험": frame.get("winning_streak_regression_risk", pd.Series(0, index=frame.index)).to_numpy() > 0,
        "연패 득실 악화": frame.get("losing_streak_with_negative_run_diff", pd.Series(0, index=frame.index)).to_numpy() == 1,
    }
    rows = [segment_metrics(segment, y_true, probability, np.asarray(mask, dtype=bool)) for segment, mask in masks.items()]
    pd.DataFrame(rows).to_csv(results_dir / "game_type_performance_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_seasonal_performance_report(results_dir: Path, frame: pd.DataFrame, y_true: np.ndarray, probability: np.ndarray):
    dates = pd.to_datetime(frame["date"])
    years = dates.dt.year
    rows = []
    periods = {
        "전체 기간": np.ones(len(frame), dtype=bool),
        "최근 3년": years >= years.max() - 2,
        "최근 2년": years >= years.max() - 1,
        "2026 시즌": years == 2026,
    }
    for label, mask in periods.items():
        rows.append(segment_metrics(label, y_true, probability, np.asarray(mask, dtype=bool)))
    month_periods = dates.dt.to_period("M")
    for month in sorted(month_periods.dropna().unique()):
        mask = month_periods.eq(month).to_numpy()
        rows.append(segment_metrics(str(month), y_true, probability, mask))
    pd.DataFrame(rows).to_csv(results_dir / "seasonal_performance_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_feature_diagnostic_report(results_dir: Path, payload: dict):
    importance = payload.get("feature_importance", {})
    selected_columns = payload.get("feature_columns", [])
    rows = []
    for rank, feature in enumerate(selected_columns, start=1):
        value = float(importance.get(feature, 0.0))
        if value > 0.01:
            diagnostic = "strong_signal"
        elif value > 0.002:
            diagnostic = "usable_signal"
        elif value > 0:
            diagnostic = "weak_signal"
        else:
            diagnostic = "no_positive_permutation_signal"
        rows.append(
            {
                "feature": feature,
                "selected_importance": round(value, 6),
                "selected_rank": rank,
                "diagnostic": diagnostic,
                "note": "Permutation/Built-in importance 기준 진단이며 단독 인과 효과가 아닙니다.",
            }
        )
    rows = sorted(rows, key=lambda row: row["selected_importance"], reverse=True)
    pd.DataFrame(rows).to_csv(results_dir / "feature_diagnostic_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_model_selection_report(results_dir: Path, candidate_results: list[dict], probability_spread_rows: list[dict], selected_model: str):
    spread_by_model = {row["model"]: row for row in probability_spread_rows}
    rows = []
    for row in candidate_results:
        model = row["모델"]
        spread = spread_by_model.get(model, {})
        rows.append(
            {
                "model": model,
                "selected": model == selected_model,
                "accuracy": row.get("검증 정확도"),
                "brier": row.get("Brier Score"),
                "log_loss": row.get("Log Loss"),
                "feature_count": row.get("피처 수"),
                "over_55_games": spread.get("over_55"),
                "over_55_accuracy": spread.get("over_55_accuracy"),
                "over_58_games": spread.get("over_58"),
                "over_60_games": spread.get("over_60"),
                "avg_confidence": spread.get("avg_confidence"),
                "p90_confidence": spread.get("p90_confidence"),
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "model_selection_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_data_gap_analysis(results_dir: Path):
    rows = [
        ["확정 선발투수", "부분 확보", "가능", "경기 전 발표 후 가능", "낮음", "높음", "중간"],
        ["선발 최근 3경기 성적", "미흡", "가능", "경기 전 가능", "중간", "높음", "중간"],
        ["선발 휴식일", "부분 확보", "가능", "경기 전 가능", "낮음", "중간", "낮음"],
        ["불펜 최근 3일 실제 투구 수", "미확보", "수집기 필요", "경기 전 가능", "중간", "높음", "높음"],
        ["전날 선발 이닝", "미확보", "가능", "경기 전 가능", "중간", "중간", "중간"],
        ["라인업 확정 여부", "표시용 확보", "가능", "경기 직전 가능", "중간", "중간", "중간"],
        ["팀 OPS / wRC+ 유사 지표", "부분 확보", "가능", "경기 전 가능", "낮음", "중간", "중간"],
        ["구장별 득점 환경", "미확보", "가능", "경기 전 가능", "낮음", "중간", "중간"],
        ["날씨", "미확보", "외부 API 필요", "경기 전 가능", "낮음", "중간", "높음"],
        ["상대 선발 유형", "미흡", "가능", "경기 전 가능", "중간", "중간", "중간"],
        ["좌우 투수/타자 매치업", "미확보", "수집기 필요", "경기 전 가능", "중간", "높음", "높음"],
    ]
    frame = pd.DataFrame(
        rows,
        columns=["data_candidate", "current_availability", "automatic_collection", "known_before_prediction", "leakage_risk", "expected_effect", "implementation_difficulty"],
    )
    frame.to_csv(results_dir / "data_gap_analysis.csv", index=False, encoding="utf-8-sig")
    return frame.to_dict(orient="records")


def write_starter_data_availability_report(results_dir: Path, data_dir: Path):
    pitching_context_path = data_dir / "pitching_context.csv"
    pitcher_stats_path = data_dir / "pitcher_stats.csv"
    pitching_snapshot_path = data_dir / "pitching_daily_snapshot.csv"
    has_pitching_context = pitching_context_path.exists()
    has_pitcher_stats = pitcher_stats_path.exists()
    has_pitching_snapshot = pitching_snapshot_path.exists()
    rows = [
        {
            "data_item": "예측 시점 투수 스냅샷",
            "current_source": "pitching_daily_snapshot.csv",
            "available_now": "started" if has_pitching_snapshot else "no",
            "collection_method": "official_kbo_dashboard.py 실행 시점의 선발/불펜 context 누적 저장",
            "known_before_game": "yes",
            "leakage_risk": "low",
            "expected_effect": "high_after_accumulation",
            "implementation_difficulty": "low",
            "next_action": "충분한 기간 누적 후 날짜 기준 shift(1) 피처 실험",
        },
        {
            "data_item": "확정 선발투수",
            "current_source": "pitching_context.csv / KBO GameCenter",
            "available_now": "partial" if has_pitching_context else "no",
            "collection_method": "경기 전 GameCenter START_PIT_CK와 선발명 수집",
            "known_before_game": "yes",
            "leakage_risk": "low",
            "expected_effect": "high",
            "implementation_difficulty": "medium",
            "next_action": "시점별 confirmed starter 히스토리 저장 후 백테스트 피처로 승격",
        },
        {
            "data_item": "선발 최근 3경기 ERA",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "투수별 경기 로그와 등판일 기준 rolling 집계 필요",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "medium",
            "next_action": "투수별 game log 수집기 추가 전까지 모델 피처화 보류",
        },
        {
            "data_item": "선발 최근 3경기 평균 이닝",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "투수별 경기 로그에서 해당 경기 이전 등판만 shift(1) 집계",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "medium",
            "next_action": "투수별 선발 등판 로그 확보 후 생성",
        },
        {
            "data_item": "선발 최근 3경기 실점/자책점",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "투수별 등판 로그의 실점/자책 rolling 집계",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "medium",
            "next_action": "당일 경기 성적 제외 검증 로직과 함께 수집",
        },
        {
            "data_item": "선발 최근 3경기 WHIP 유사 지표",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "최근 등판 피안타/볼넷/이닝 기반 rolling 계산",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "medium",
            "next_action": "투수별 피안타·볼넷·이닝 로그 확보 후 생성",
        },
        {
            "data_item": "선발 휴식일",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "확정/추정 선발명과 이전 등판일 차이 계산",
            "known_before_game": "yes",
            "leakage_risk": "low",
            "expected_effect": "medium",
            "implementation_difficulty": "medium",
            "next_action": "선발명 히스토리와 투수 등판 로그를 연결한 뒤 추가",
        },
        {
            "data_item": "선발 시즌 ERA/WHIP",
            "current_source": "pitcher_stats.csv / pitching_context.csv",
            "available_now": "partial" if has_pitcher_stats and has_pitching_context else "no",
            "collection_method": "현재 스냅샷은 표시용으로 사용, 과거 시점별 스냅샷 필요",
            "known_before_game": "yes",
            "leakage_risk": "high_without_snapshots",
            "expected_effect": "medium",
            "implementation_difficulty": "medium",
            "next_action": "과거 경기에는 최신 스냅샷을 붙이지 않고 날짜별 누적 기록 저장",
        },
    ]
    pd.DataFrame(rows).to_csv(results_dir / "starter_data_availability_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_bullpen_data_availability_report(results_dir: Path, data_dir: Path):
    model_training_path = data_dir / "model_training_games.csv"
    pitching_snapshot_path = data_dir / "pitching_daily_snapshot.csv"
    has_team_games = model_training_path.exists()
    has_pitching_snapshot = pitching_snapshot_path.exists()
    rows = [
        {
            "data_item": "불펜 피로 proxy 스냅샷",
            "current_source": "pitching_daily_snapshot.csv",
            "available_now": "proxy_snapshot_started" if has_pitching_snapshot else "no",
            "collection_method": "최근 3일 경기 수와 불펜 피로 라벨을 예측 시점 기준으로 누적",
            "known_before_game": "yes",
            "leakage_risk": "low",
            "expected_effect": "medium_after_accumulation",
            "implementation_difficulty": "low",
            "next_action": "실제 불펜 투구 수 로그 확보 전까지 proxy 히스토리로만 보관",
        },
        {
            "data_item": "최근 3일 팀 불펜 등판 수",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "투수별 등판 로그에서 선발 제외 투수 수 집계",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "high",
            "next_action": "투수별 box score 수집기 필요",
        },
        {
            "data_item": "최근 3일 불펜 총 이닝",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "팀별 투수 등판 로그에서 불펜 이닝 rolling 집계",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "high",
            "next_action": "전날 경기까지의 불펜 이닝만 사용하도록 수집",
        },
        {
            "data_item": "최근 3일 불펜 투구 수",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "투수별 투구수 로그 수집",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "high",
            "next_action": "KBO box score 또는 GameCenter 세부 투구 기록 파싱 검토",
        },
        {
            "data_item": "전날 선발 이닝",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "전날 경기 선발투수 이닝 추출",
            "known_before_game": "yes",
            "leakage_risk": "low",
            "expected_effect": "medium",
            "implementation_difficulty": "medium",
            "next_action": "선발 등판 로그 확보 후 불펜 부담 proxy로 사용",
        },
        {
            "data_item": "전날 불펜 소모량",
            "current_source": "games_last_7_days/back_to_back proxy",
            "available_now": "proxy_only" if has_team_games else "no",
            "collection_method": "현재는 일정 기반 proxy, 실제 불펜 이닝으로 대체 필요",
            "known_before_game": "yes",
            "leakage_risk": "low",
            "expected_effect": "high",
            "implementation_difficulty": "high",
            "next_action": "proxy는 유지하되 모델 교체 근거로 사용하지 않음",
        },
        {
            "data_item": "핵심 불펜 연투 여부",
            "current_source": "not_available",
            "available_now": "no",
            "collection_method": "세이브/홀드 상위 투수의 최근 2~3일 등판 여부 계산",
            "known_before_game": "yes",
            "leakage_risk": "medium",
            "expected_effect": "high",
            "implementation_difficulty": "high",
            "next_action": "핵심 불펜 정의와 등판 로그 수집 후 실험",
        },
    ]
    pd.DataFrame(rows).to_csv(results_dir / "bullpen_data_availability_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_pitching_feature_experiment_report(results_dir: Path, streak_rows: list[dict], selected_model: str):
    def row_from_streak(feature_set: str):
        candidates = [row for row in streak_rows if row.get("feature_set") == feature_set]
        if not candidates:
            return None
        return max(candidates, key=lambda row: row.get("accuracy") or 0)

    baseline = row_from_streak("compact_baseline")
    streak = row_from_streak("compact_plus_streak")
    rows = []
    for feature_set, source in [("baseline_core", baseline), ("baseline_plus_streak", streak)]:
        if source:
            rows.append(
                {
                    "model": source["model"],
                    "feature_set": feature_set,
                    "accuracy": source["accuracy"],
                    "brier": source["brier"],
                    "log_loss": source["log_loss"],
                    "over_55_games": source["over_55_games"],
                    "over_55_accuracy": source["over_55_accuracy"],
                    "winning_streak_accuracy": source["winning_streak_accuracy"],
                    "losing_streak_accuracy": source["losing_streak_accuracy"],
                    "close_game_accuracy": source["close_game_accuracy"],
                    "recent_3year_accuracy": None,
                    "selected_candidate": source["model"] == selected_model,
                    "status": "evaluated",
                }
            )

    for feature_set, reason in [
        ("baseline_plus_starter", "과거 시점별 선발 최근 3경기/휴식일 데이터가 없어 누수 없이 백테스트 불가"),
        ("baseline_plus_bullpen", "실제 불펜 이닝/투구 수 로그가 없어 일정 proxy만 존재"),
        ("baseline_plus_starter_bullpen", "선발 히스토리와 불펜 사용량 원천 데이터가 모두 필요"),
    ]:
        rows.append(
            {
                "model": "not_available",
                "feature_set": feature_set,
                "accuracy": None,
                "brier": None,
                "log_loss": None,
                "over_55_games": None,
                "over_55_accuracy": None,
                "winning_streak_accuracy": None,
                "losing_streak_accuracy": None,
                "close_game_accuracy": None,
                "recent_3year_accuracy": None,
                "selected_candidate": False,
                "status": reason,
            }
        )

    pd.DataFrame(rows).to_csv(results_dir / "pitching_feature_experiment_report.csv", index=False, encoding="utf-8-sig")
    return rows


def write_model_insight_summary(
    results_dir: Path,
    payload: dict,
    feature_rows: list[dict],
    segment_rows: list[dict],
    data_gaps: list[dict],
    starter_availability: list[dict] | None = None,
    bullpen_availability: list[dict] | None = None,
    pitching_experiment_rows: list[dict] | None = None,
):
    sorted_segments = [row for row in segment_rows if row["total_games"]]
    best_segments = sorted(sorted_segments, key=lambda row: row["accuracy"] or 0, reverse=True)[:5]
    worst_segments = sorted(sorted_segments, key=lambda row: row["accuracy"] or 1)[:5]
    strongest = [row["feature"] for row in feature_rows[:8]]
    weak = [row["feature"] for row in feature_rows if row["diagnostic"] in {"weak_signal", "no_positive_permutation_signal"}][-12:]
    selected = payload.get("selected_model")
    candidate_rows = payload.get("candidate_results", [])
    selected_row = next((row for row in candidate_rows if row["모델"] == selected), {})
    best_accuracy_row = max(candidate_rows, key=lambda row: row.get("검증 정확도", 0)) if candidate_rows else {}
    safe_to_replace = bool(best_accuracy_row and best_accuracy_row.get("모델") != selected and best_accuracy_row.get("검증 정확도", 0) > selected_row.get("검증 정확도", 0) + 0.005)
    summary = {
        "current_baseline": {
            "selected_model": selected,
            "accuracy": payload.get("accuracy"),
            "brier": selected_row.get("Brier Score"),
            "log_loss": selected_row.get("Log Loss"),
        },
        "strongest_features": strongest,
        "weak_features": weak,
        "best_performing_segments": best_segments,
        "worst_performing_segments": worst_segments,
        "data_gaps": data_gaps,
        "pitching_data_availability_summary": starter_availability or [],
        "bullpen_data_availability_summary": bullpen_availability or [],
        "pitching_feature_experiment_summary": pitching_experiment_rows or [],
        "pitching_snapshot_collection_status": {
            "status": "pending_dashboard_snapshot_step",
            "note": "official_kbo_dashboard.py의 pitching context 생성 이후 현재 실행 기준 스냅샷 상태로 갱신됩니다.",
        },
        "leakage_safe_pitching_data_policy": "투수 스냅샷은 예측 시점에 알고 있던 정보만 누적 저장하며, 현재 운영 모델 학습 피처로 바로 사용하지 않습니다.",
        "next_step_after_snapshot_accumulation": "스냅샷이 충분히 쌓이면 선발 최근 성적, 선발 정보 품질, 불펜 피로 proxy를 날짜 기준 shift(1) 피처로 별도 후보 모델에서 검증합니다.",
        "recommended_next_steps": [
            "확정 선발 최근 3경기 성적과 휴식일을 과거 시점 스냅샷으로 저장",
            "불펜 최근 3일 실제 투구 수와 전날 선발 이닝 수집",
            "구장별 득점 환경과 라인업 확정 정보를 예측 시점 기준으로 축적",
            "후보 모델은 Brier/Log Loss와 확신 구간 적중률 개선이 확인될 때만 교체",
        ],
        "recommended_next_modeling_step": "투수별 경기 로그를 수집해 선발 최근 3경기 성적과 실제 불펜 소모량을 날짜 기준 shift(1) 피처로 검증",
        "safe_to_replace_model": safe_to_replace,
        "reason_not_to_replace_if_false": "" if safe_to_replace else "후보 모델이 정확도와 확률 품질을 동시에 안정적으로 개선했다는 근거가 부족합니다.",
    }
    (results_dir / "model_insight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def train_prediction_bundle(best, training_games, prediction_training_cutoff, data_dir, results_dir):
    prediction_completed = training_games[
        (training_games["status"] == "Final")
        & (pd.to_datetime(training_games["date"]).dt.date <= prediction_training_cutoff)
    ].copy()
    prediction_model_input = results_dir / "_prediction_model_training_games.tmp.csv"
    prediction_completed.to_csv(prediction_model_input, index=False, encoding="utf-8-sig")
    prediction_features = build_features(prediction_model_input)
    prediction_model_input.unlink(missing_ok=True)
    columns = best["columns"]
    if best.get("prediction_unit", "team") == "game":
        pitching_context_path = data_dir / "pitching_context.csv"
        pitching_context = pd.read_csv(pitching_context_path) if pitching_context_path.exists() else pd.DataFrame()
        frame = attach_pitching_context(build_game_level_frame(prediction_features), pitching_context).dropna(subset=["target_home_win"]).copy()
        px, py = prepare_game_level_matrix(frame)
        weight_dates = pd.to_datetime(frame["date"])
    else:
        px, py = prepare_matrix(prediction_features)
        weight_dates = pd.to_datetime(prediction_features["date"])
    px = px[columns]
    scaled, _, mean, std = standardize_train_test(px, px)
    if best["model_type"] == "from_scratch_logistic_regression":
        weights, bias = train_logistic_regression(scaled.to_numpy(), py, lr=0.05, epochs=3500)
        return {"model_type": "from_scratch_logistic_regression", "weights": weights, "bias": bias, "mean": mean, "std": std}
    from sklearn.base import clone
    model = clone(best["model"])
    fit_kwargs = {}
    if "시간가중" in best["name"]:
        years = weight_dates.dt.year
        max_year = int(years.max())
        fit_kwargs["sample_weight"] = (0.85 ** (max_year - years)).clip(lower=0.35).to_numpy(dtype=float)
    model.fit(scaled, py, **fit_kwargs)
    return {"model_type": best["model_type"], "model": model, "mean": mean, "std": std}


def build_payload(best, candidate_results, features, split_index, y_test, current_games, cutoff, prediction_date, data_dir, results_dir, completed, training_games, probability_spread_rows, streak_experiment_rows):
    columns = best["columns"]
    probability = best["probability"]
    pred = best["pred"]
    mean = best["mean"]
    std = best["std"]
    weights = best.get("weights")
    bias = best.get("bias")
    prediction_unit = best.get("prediction_unit", "team")
    y_eval = best.get("y_test", y_test)
    prediction_training_cutoff = prediction_date - timedelta(days=1)
    latest_completed = training_games[
        (training_games["status"] == "Final")
        & (pd.to_datetime(training_games["date"]).dt.date <= prediction_training_cutoff)
    ].copy()
    latest_completed_game_date = pd.to_datetime(latest_completed["date"]).dt.date.max().isoformat() if not latest_completed.empty else ""
    current_week_games_included = bool(
        not latest_completed.empty
        and pd.to_datetime(latest_completed["date"]).dt.date.gt(cutoff).any()
    )
    prediction_bundle = train_prediction_bundle(best, training_games, prediction_training_cutoff, data_dir, results_dir)

    if prediction_unit == "game":
        recent = best["test_frame"].copy()
        recent["경기일"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
        recent["기준팀"] = recent["home_team"]
        recent["상대팀"] = recent["away_team"]
        recent["예측승률"] = [f"{max(p, 1 - p):.1%}" for p in probability]
        recent["예측 구단"] = np.where(probability >= 0.5, recent["home_team"], recent["away_team"])
        recent["예측"] = np.where(probability >= 0.5, "승리 예측", "패배 예측")
        recent["실제 승리 구단"] = np.where(y_eval == 1, recent["home_team"], recent["away_team"])
        recent["예측 근거"] = recent.apply(lambda row: game_prediction_reason(row, row["예측 구단"]), axis=1)
        recent["경기"] = recent["away_team"] + " vs " + recent["home_team"]
        recent["결과"] = np.where(recent["예측 구단"] == recent["실제 승리 구단"], "적중", "오답")
        recent_backtest = recent[["경기일", "경기", "예측 구단", "예측승률", "실제 승리 구단", "결과", "예측 근거"]].tail(12).to_dict(orient="records")
        train_rows = int(len(build_game_level_frame(features).dropna(subset=["target_home_win"])) - len(recent))
        test_rows = int(len(recent))
    else:
        recent = features.iloc[split_index:].copy()
        recent["_prediction_probability"] = probability
        recent["경기일"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
        recent["기준팀"] = recent["team"]
        recent["상대팀"] = recent["opponent"]
        recent["예측승률"] = [f"{p:.1%}" for p in probability]
        recent["예측"] = np.where(pred == 1, "승리 예측", "패배 예측")
        recent["예측 구단"] = np.where(pred == 1, recent["team"], recent["opponent"])
        recent["실제 승리 구단"] = np.where(y_test == 1, recent["team"], recent["opponent"])
        recent["예측 근거"] = recent.apply(lambda row: prediction_reason(row, row["예측 구단"]), axis=1)
        recent_backtest = summarize_team_backtest_by_game(recent)[-12:]
        train_rows = int(split_index)
        test_rows = int(len(features) - split_index)

    prediction_input = data_dir / "prediction_games.csv"
    current_games.to_csv(prediction_input, index=False, encoding="utf-8-sig")
    prediction_features = build_features(prediction_input, include_unlabeled=True)
    prediction_features["date_obj"] = pd.to_datetime(prediction_features["date"]).dt.date
    today_features = prediction_features[(prediction_features["date_obj"] == prediction_date) & (prediction_features["target_win"].isna())].copy()
    today_predictions = []
    if not today_features.empty and prediction_unit == "game":
        pitching_context_path = data_dir / "pitching_context.csv"
        pitching_context = pd.read_csv(pitching_context_path) if pitching_context_path.exists() else pd.DataFrame()
        game_prediction_frame = attach_pitching_context(build_game_level_frame(prediction_features), pitching_context)
        game_prediction_frame["date_obj"] = pd.to_datetime(game_prediction_frame["date"]).dt.date
        today_games = game_prediction_frame[(game_prediction_frame["date_obj"] == prediction_date) & (game_prediction_frame["target_home_win"].isna())].copy()
        if not today_games.empty:
            prediction_scaled = align_game_level_matrix(today_games.drop(columns=["date_obj"]), columns, prediction_bundle["mean"], prediction_bundle["std"])
            game_probability = prediction_bundle["model"].predict_proba(prediction_scaled)[:, 1]
            for (_, row), home_prob in zip(today_games.iterrows(), game_probability):
                home_pick = home_prob >= 0.5
                predicted_team = row["home_team"] if home_pick else row["away_team"]
                reason = game_prediction_reason(row, predicted_team)
                today_predictions.extend([
                    {"경기일": row["date"], "기준팀": row["home_team"], "상대팀": row["away_team"], "예측 구단": predicted_team, "예측승률": f"{home_prob:.1%}", "예측": "승리 예측" if home_pick else "패배 예측", "예측 근거": reason},
                    {"경기일": row["date"], "기준팀": row["away_team"], "상대팀": row["home_team"], "예측 구단": predicted_team, "예측승률": f"{1 - home_prob:.1%}", "예측": "승리 예측" if not home_pick else "패배 예측", "예측 근거": reason},
                ])
    elif not today_features.empty:
        prediction_scaled = align_prediction_matrix(today_features.drop(columns=["date_obj"]), columns, prediction_bundle["mean"], prediction_bundle["std"])
        if prediction_bundle["model_type"] == "from_scratch_logistic_regression":
            raw_today_probability = sigmoid(prediction_scaled.to_numpy() @ prediction_bundle["weights"] + prediction_bundle["bias"])
        else:
            raw_today_probability = prediction_bundle["model"].predict_proba(prediction_scaled)[:, 1]
        today_probability = normalize_game_probabilities(today_features, raw_today_probability)
        today_features["경기일"] = pd.to_datetime(today_features["date"]).dt.strftime("%Y-%m-%d")
        today_features["기준팀"] = today_features["team"]
        today_features["상대팀"] = today_features["opponent"]
        today_features["예측승률"] = [f"{p:.1%}" for p in today_probability]
        today_features["예측"] = np.where(today_probability >= 0.5, "승리 예측", "패배 예측")
        today_features["예측 구단"] = np.where(today_probability >= 0.5, today_features["team"], today_features["opponent"])
        today_features["예측 근거"] = today_features.apply(lambda row: prediction_reason(row, row["예측 구단"]), axis=1)
        today_predictions = today_features[["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "예측 근거"]].to_dict(orient="records")

    today_probability_values = []
    for row in today_predictions:
        try:
            today_probability_values.append(float(str(row["예측승률"]).replace("%", "")) / 100)
        except (KeyError, ValueError):
            continue
    distribution_rows = write_probability_distribution_report(results_dir, probability, today_probability_values)
    today_distribution = next((row for row in distribution_rows if row["split"] == "today"), {})
    policy = confidence_bucket_policy(y_eval, probability)
    selected_spread = next((row for row in probability_spread_rows if row["model"] == best["name"]), {})
    high_confidence_summary = {
        "selected_model": best["name"],
        "overall_accuracy": selected_spread.get("accuracy"),
        "over_55_games": selected_spread.get("over_55"),
        "over_55_accuracy": selected_spread.get("over_55_accuracy"),
        "avg_confidence": selected_spread.get("avg_confidence"),
        "p90_confidence": selected_spread.get("p90_confidence"),
    }

    payload = {
        "available": True,
        "training_cutoff": cutoff.isoformat(),
        "validation_cutoff": cutoff.isoformat(),
        "prediction_training_cutoff": prediction_training_cutoff.isoformat(),
        "latest_completed_game_date_used": latest_completed_game_date,
        "current_week_games_included_for_prediction": current_week_games_included,
        "training_start_year": int(completed["date"].dt.year.min()),
        "training_end_year": int(completed["date"].dt.year.max()),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": best["accuracy"],
        "selected_model": best["name"],
        "candidate_results": candidate_results,
        "confidence_metrics": confidence_metrics(y_eval, probability),
        "calibration_table": calibration_table(y_eval, probability),
        "confidence_thresholds": policy["confidence_thresholds"],
        "confidence_bucket_performance": policy["confidence_bucket_performance"],
        "today_probability_distribution": today_distribution,
        "confidence_policy_note": "예측승률 자체는 보정하지 않고, 백테스트 상위 확신 구간과 정보 품질을 표시용 신뢰도 판단에 사용합니다.",
        "model_probability_spread_report": probability_spread_rows,
        "selected_model_probability_spread": selected_spread,
        "high_confidence_backtest_summary": high_confidence_summary,
        "baseball_feature_policy_note": "선발투수, 불펜 피로도, 최근 득점/실점 흐름 피처를 별도 후보 모델에 추가해 비교합니다. 확률을 강제로 키우지 않고 Accuracy, Brier Score, Log Loss, 55% 이상 구간 성능과 확률 분포를 함께 확인합니다.",
        "recent_backtest": recent_backtest,
        "today_predictions": today_predictions,
        "source_note": "현재 주 경기는 적중/오답 집계에 포함하지 않습니다.",
        "feature_columns": columns,
        "model_type": best["model_type"],
        "prediction_unit": prediction_unit,
    }
    if best["model_type"] == "from_scratch_logistic_regression":
        payload["bias"] = round(float(bias), 6)
        payload["coefficients"] = {name: round(float(value), 6) for name, value in sorted(zip(columns, weights), key=lambda x: abs(x[1]), reverse=True)}
    elif hasattr(best.get("model"), "feature_importances_"):
        importances = best["model"].feature_importances_
        payload["feature_importance"] = {name: round(float(value), 6) for name, value in sorted(zip(columns, importances), key=lambda x: x[1], reverse=True)}
    else:
        payload["feature_importance"] = permutation_importance(best, y_eval)
    eval_frame = best["test_frame"].copy()
    feature_diagnostics = write_feature_diagnostic_report(results_dir, payload)
    game_type_segments = write_game_type_performance_report(results_dir, eval_frame, np.asarray(y_eval), np.asarray(probability))
    seasonal_segments = write_seasonal_performance_report(results_dir, eval_frame, np.asarray(y_eval), np.asarray(probability))
    model_selection_rows = write_model_selection_report(results_dir, candidate_results, probability_spread_rows, best["name"])
    data_gaps = write_data_gap_analysis(results_dir)
    starter_availability = write_starter_data_availability_report(results_dir, data_dir)
    bullpen_availability = write_bullpen_data_availability_report(results_dir, data_dir)
    pitching_experiment_rows = write_pitching_feature_experiment_report(results_dir, streak_experiment_rows, best["name"])
    insight_summary = write_model_insight_summary(
        results_dir,
        payload,
        feature_diagnostics,
        game_type_segments,
        data_gaps,
        starter_availability,
        bullpen_availability,
        pitching_experiment_rows,
    )
    payload["diagnostic_reports"] = {
        "feature_diagnostic_report": "modeling/results/feature_diagnostic_report.csv",
        "game_type_performance_report": "modeling/results/game_type_performance_report.csv",
        "data_gap_analysis": "modeling/results/data_gap_analysis.csv",
        "starter_data_availability_report": "modeling/results/starter_data_availability_report.csv",
        "bullpen_data_availability_report": "modeling/results/bullpen_data_availability_report.csv",
        "pitching_feature_experiment_report": "modeling/results/pitching_feature_experiment_report.csv",
        "model_selection_report": "modeling/results/model_selection_report.csv",
        "seasonal_performance_report": "modeling/results/seasonal_performance_report.csv",
        "model_insight_summary": "modeling/results/model_insight_summary.json",
    }
    payload["diagnostic_report_rows"] = {
        "feature_diagnostics": len(feature_diagnostics),
        "game_type_segments": len(game_type_segments),
        "seasonal_segments": len(seasonal_segments),
        "model_selection_rows": len(model_selection_rows),
        "data_gaps": len(data_gaps),
        "starter_data_availability": len(starter_availability),
        "bullpen_data_availability": len(bullpen_availability),
        "pitching_feature_experiment_rows": len(pitching_experiment_rows),
    }
    payload["model_insight_summary"] = insight_summary
    return payload
