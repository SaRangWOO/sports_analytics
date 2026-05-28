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
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError:
        return []

    return [
        ("RandomForest 비선형 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), None),
        ("RandomForest 시간가중 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), recency_weight),
        ("RandomForest 보수 모델", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), None),
        ("RandomForest 보수 시간가중 모델", RandomForestClassifier(n_estimators=800, max_depth=5, min_samples_leaf=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1), recency_weight),
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
        "기본 흐름 모델": [col for col in x.columns if col not in {"team_elo_pre", "opponent_elo_pre", "elo_diff", "games_last_7_days", "back_to_back"}],
        "전력/일정 피로도 포함 모델": list(x.columns),
        "핵심 수치 모델": compact_feature_columns(x),
    }
    best = None
    candidate_results = []
    probability_spread_rows = []

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
        best = pick_better_model(best, result)

    sklearn_candidates = sklearn_candidate_specs(recency_weight)
    for name, model, sample_weight in sklearn_candidates:
        columns = list(x.columns)
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

    spread_report = write_model_probability_spread_report(results_dir, probability_spread_rows)
    payload = build_payload(best, candidate_results, features, split_index, y_test, current_games, cutoff, prediction_date, data_dir, results_dir, completed, training_games, spread_report)
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


def build_payload(best, candidate_results, features, split_index, y_test, current_games, cutoff, prediction_date, data_dir, results_dir, completed, training_games, probability_spread_rows):
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
    return payload
