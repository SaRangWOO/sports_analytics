from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engineering import build_features
from .game_level_features import (
    align_game_level_matrix,
    build_game_level_frame,
    export_game_level_dataset,
    prepare_game_level_matrix,
)
from .model_evaluation import (
    calibration_table,
    confidence_metrics,
    normalize_game_probabilities,
    pick_better_model,
    probability_scores,
)
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


def sklearn_candidate_specs(recency_weight):
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError:
        return []

    return [
        ("RandomForest 비선형 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), None),
        ("RandomForest 시간가중 모델", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), recency_weight),
        ("GradientBoosting 비선형 모델", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42), None),
        ("GradientBoosting 시간가중 모델", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42), recency_weight),
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
    export_game_level_dataset(features, results_dir / "game_level_features.csv")

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
        "핵심 수치 모델": [col for col in ["is_home", "rest_days", "recent_5_win_rate", "recent_10_win_rate", "avg_run_diff_last_5", "avg_run_diff_last_10", "season_win_rate_prior", "opponent_recent_5_win_rate", "opponent_recent_10_win_rate", "opponent_avg_run_diff_last_5", "opponent_avg_run_diff_last_10", "season_win_rate_gap", "recent_5_win_rate_gap", "recent_10_win_rate_gap", "season_avg_run_diff_gap", "recent_run_diff_10_gap", "venue_win_rate_gap", "head_to_head_win_rate_gap", "elo_diff", "games_last_7_days", "back_to_back"] if col in x.columns],
    }
    best = None
    candidate_results = []

    for name, columns in candidate_columns.items():
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        weights, bias = train_logistic_regression(train_scaled.to_numpy(), y_train, lr=0.05, epochs=3500)
        probability = normalize_game_probabilities(features.iloc[split_index:], sigmoid(test_scaled.to_numpy() @ weights + bias))
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "weights": weights, "bias": bias, "model_type": "from_scratch_logistic_regression", "prediction_unit": "team"}
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
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
        result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "model": model, "model_type": model.__class__.__name__, "prediction_unit": "team"}
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        best = pick_better_model(best, result)

    game_frame = build_game_level_frame(features).dropna(subset=["target_home_win"]).copy()
    if sklearn_candidates and len(game_frame) >= 20:
        gx, gy = prepare_game_level_matrix(game_frame)
        game_split = chronological_split_index(game_frame["date"])
        gy_train, gy_test = gy[:game_split], gy[game_split:]
        game_years = pd.to_datetime(game_frame.iloc[:game_split]["date"]).dt.year
        max_game_year = int(game_years.max())
        game_recency_weight = (0.85 ** (max_game_year - game_years)).clip(lower=0.35).to_numpy(dtype=float)
        game_candidates = [
            ("경기 단위 RandomForest 모델", sklearn_candidates[0][1].__class__(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), None),
            ("경기 단위 RandomForest 시간가중 모델", sklearn_candidates[0][1].__class__(n_estimators=500, max_depth=7, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1), game_recency_weight),
            ("경기 단위 GradientBoosting 모델", sklearn_candidates[2][1].__class__(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42), None),
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
            result = {"name": name, "columns": columns, "accuracy": accuracy, "score": score, "probability": probability, "pred": pred, "mean": mean, "std": std, "model": model, "model_type": model.__class__.__name__, "prediction_unit": "game", "test_frame": game_frame.iloc[game_split:].copy(), "y_test": gy_test}
            candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
            best = pick_better_model(best, result)

    payload = build_payload(best, candidate_results, features, split_index, y_test, current_games, cutoff, prediction_date, data_dir, completed)
    (results_dir / "win_predictor_model.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_payload(best, candidate_results, features, split_index, y_test, current_games, cutoff, prediction_date, data_dir, completed):
    columns = best["columns"]
    probability = best["probability"]
    pred = best["pred"]
    mean = best["mean"]
    std = best["std"]
    weights = best.get("weights")
    bias = best.get("bias")
    prediction_unit = best.get("prediction_unit", "team")
    y_eval = best.get("y_test", y_test)

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
        train_rows = int(len(build_game_level_frame(features).dropna(subset=["target_home_win"])) - len(recent))
        test_rows = int(len(recent))
    else:
        recent = features.iloc[split_index:].copy()
        recent["경기일"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
        recent["기준팀"] = recent["team"]
        recent["상대팀"] = recent["opponent"]
        recent["예측승률"] = [f"{p:.1%}" for p in probability]
        recent["예측"] = np.where(pred == 1, "승리 예측", "패배 예측")
        recent["예측 구단"] = np.where(pred == 1, recent["team"], recent["opponent"])
        recent["실제 승리 구단"] = np.where(y_test == 1, recent["team"], recent["opponent"])
        recent["예측 근거"] = recent.apply(lambda row: prediction_reason(row, row["예측 구단"]), axis=1)
        train_rows = int(split_index)
        test_rows = int(len(features) - split_index)

    prediction_input = data_dir / "prediction_games.csv"
    current_games.to_csv(prediction_input, index=False, encoding="utf-8-sig")
    prediction_features = build_features(prediction_input, include_unlabeled=True)
    prediction_features["date_obj"] = pd.to_datetime(prediction_features["date"]).dt.date
    today_features = prediction_features[(prediction_features["date_obj"] == prediction_date) & (prediction_features["target_win"].isna())].copy()
    today_predictions = []
    if not today_features.empty and prediction_unit == "game":
        game_prediction_frame = build_game_level_frame(prediction_features)
        game_prediction_frame["date_obj"] = pd.to_datetime(game_prediction_frame["date"]).dt.date
        today_games = game_prediction_frame[(game_prediction_frame["date_obj"] == prediction_date) & (game_prediction_frame["target_home_win"].isna())].copy()
        if not today_games.empty:
            prediction_scaled = align_game_level_matrix(today_games.drop(columns=["date_obj"]), columns, mean, std)
            game_probability = best["model"].predict_proba(prediction_scaled)[:, 1]
            for (_, row), home_prob in zip(today_games.iterrows(), game_probability):
                home_pick = home_prob >= 0.5
                predicted_team = row["home_team"] if home_pick else row["away_team"]
                reason = game_prediction_reason(row, predicted_team)
                today_predictions.extend([
                    {"경기일": row["date"], "기준팀": row["home_team"], "상대팀": row["away_team"], "예측 구단": predicted_team, "예측승률": f"{home_prob:.1%}", "예측": "승리 예측" if home_pick else "패배 예측", "예측 근거": reason},
                    {"경기일": row["date"], "기준팀": row["away_team"], "상대팀": row["home_team"], "예측 구단": predicted_team, "예측승률": f"{1 - home_prob:.1%}", "예측": "승리 예측" if not home_pick else "패배 예측", "예측 근거": reason},
                ])
    elif not today_features.empty:
        prediction_scaled = align_prediction_matrix(today_features.drop(columns=["date_obj"]), columns, mean, std)
        if best["model_type"] == "from_scratch_logistic_regression":
            raw_today_probability = sigmoid(prediction_scaled.to_numpy() @ weights + bias)
        else:
            raw_today_probability = best["model"].predict_proba(prediction_scaled)[:, 1]
        today_probability = normalize_game_probabilities(today_features, raw_today_probability)
        today_features["경기일"] = pd.to_datetime(today_features["date"]).dt.strftime("%Y-%m-%d")
        today_features["기준팀"] = today_features["team"]
        today_features["상대팀"] = today_features["opponent"]
        today_features["예측승률"] = [f"{p:.1%}" for p in today_probability]
        today_features["예측"] = np.where(today_probability >= 0.5, "승리 예측", "패배 예측")
        today_features["예측 구단"] = np.where(today_probability >= 0.5, today_features["team"], today_features["opponent"])
        today_features["예측 근거"] = today_features.apply(lambda row: prediction_reason(row, row["예측 구단"]), axis=1)
        today_predictions = today_features[["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "예측 근거"]].to_dict(orient="records")

    payload = {
        "available": True,
        "training_cutoff": cutoff.isoformat(),
        "training_start_year": int(completed["date"].dt.year.min()),
        "training_end_year": int(completed["date"].dt.year.max()),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": best["accuracy"],
        "selected_model": best["name"],
        "candidate_results": candidate_results,
        "confidence_metrics": confidence_metrics(y_eval, probability),
        "calibration_table": calibration_table(y_eval, probability),
        "recent_backtest": recent[["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "실제 승리 구단", "예측 근거"]].tail(12).to_dict(orient="records"),
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
    return payload
