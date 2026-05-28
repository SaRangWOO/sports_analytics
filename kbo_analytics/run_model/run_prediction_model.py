from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, TweedieRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "official" / "model_training_games.csv"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def _base_game_id(game_id: str):
    return str(game_id).rsplit("_", 1)[0]


def load_completed_team_games(input_path: Path):
    df = pd.read_csv(input_path)
    required = {"game_id", "date", "team", "opponent", "home_away", "status", "score_team", "score_opp"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["status"].eq("Final")].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["base_game_id"] = df["game_id"].map(_base_game_id)
    df["score_team"] = pd.to_numeric(df["score_team"], errors="coerce")
    df["score_opp"] = pd.to_numeric(df["score_opp"], errors="coerce")
    df = df.dropna(subset=["score_team", "score_opp"])
    return df.sort_values(["date", "base_game_id", "home_away", "team"]).reset_index(drop=True)


def build_base_run_df(team_games: pd.DataFrame):
    """Convert official team-perspective rows into one row per team-game."""
    df = team_games.copy()
    df["season"] = df["date"].dt.year
    df["is_home"] = df["home_away"].eq("H").astype(int)
    df["target_runs"] = df["score_team"]
    df["runs_allowed"] = df["score_opp"]
    df["target_win"] = (df["score_team"] > df["score_opp"]).astype(int)
    df["month"] = df["date"].dt.month
    df["game_key"] = df["base_game_id"]

    cols = [
        "game_key",
        "game_id",
        "date",
        "season",
        "month",
        "team",
        "opponent",
        "is_home",
        "target_runs",
        "runs_allowed",
        "target_win",
    ]
    if "ballpark" in df.columns:
        cols.append("ballpark")
    return df[cols].sort_values(["season", "date", "game_key", "is_home"]).reset_index(drop=True)


def create_rolling_run_features(run_df: pd.DataFrame):
    """Create pre-game rolling run features without using the current game."""
    df = run_df.copy().sort_values(["season", "team", "date", "game_key"]).reset_index(drop=True)
    grouped = df.groupby(["season", "team"], sort=False)
    df["shifted_runs"] = grouped["target_runs"].shift(1)
    df["shifted_allowed"] = grouped["runs_allowed"].shift(1)
    df["shifted_win"] = grouped["target_win"].shift(1)

    df["team_recent_3g_runs"] = grouped["shifted_runs"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["team_recent_5g_runs"] = grouped["shifted_runs"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["team_recent_10g_runs"] = grouped["shifted_runs"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["team_season_runs"] = grouped["shifted_runs"].transform(lambda x: x.expanding(min_periods=1).mean())
    df["team_recent_5g_allowed"] = grouped["shifted_allowed"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["team_recent_10g_allowed"] = grouped["shifted_allowed"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["team_season_allowed"] = grouped["shifted_allowed"].transform(lambda x: x.expanding(min_periods=1).mean())
    df["team_recent_5g_win_rate"] = grouped["shifted_win"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["team_recent_10g_win_rate"] = grouped["shifted_win"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["team_season_win_rate"] = grouped["shifted_win"].transform(lambda x: x.expanding(min_periods=1).mean())
    df["team_recent_5g_run_diff"] = df["team_recent_5g_runs"] - df["team_recent_5g_allowed"]
    df["team_recent_10g_run_diff"] = df["team_recent_10g_runs"] - df["team_recent_10g_allowed"]

    df["previous_game_date"] = grouped["date"].shift(1)
    df["rest_days"] = (df["date"] - df["previous_game_date"]).dt.days.fillna(1).clip(lower=0, upper=14)
    df["back_to_back"] = (df["rest_days"] <= 1).astype(int)

    run_feature_cols = [
        "team_recent_3g_runs",
        "team_recent_5g_runs",
        "team_recent_10g_runs",
        "team_season_runs",
        "team_recent_5g_allowed",
        "team_recent_10g_allowed",
        "team_season_allowed",
        "team_recent_5g_win_rate",
        "team_recent_10g_win_rate",
        "team_season_win_rate",
        "team_recent_5g_run_diff",
        "team_recent_10g_run_diff",
    ]
    league_avg_runs = float(df["target_runs"].mean())
    fill_values = {
        "team_recent_5g_win_rate": 0.5,
        "team_recent_10g_win_rate": 0.5,
        "team_season_win_rate": 0.5,
        "team_recent_5g_run_diff": 0.0,
        "team_recent_10g_run_diff": 0.0,
    }
    for col in run_feature_cols:
        df[col] = df[col].fillna(fill_values.get(col, league_avg_runs))

    return df.drop(columns=["shifted_runs", "shifted_allowed", "shifted_win", "previous_game_date"])


def merge_opponent_features(df: pd.DataFrame):
    opponent_cols = [
        "game_key",
        "team",
        "team_recent_5g_runs",
        "team_recent_10g_runs",
        "team_season_runs",
        "team_recent_5g_allowed",
        "team_recent_10g_allowed",
        "team_season_allowed",
        "team_recent_5g_win_rate",
        "team_recent_10g_win_rate",
        "team_season_win_rate",
        "team_recent_5g_run_diff",
        "team_recent_10g_run_diff",
        "rest_days",
        "back_to_back",
    ]
    opp = df[opponent_cols].copy().rename(
        columns={
            "team": "opponent",
            "team_recent_5g_runs": "opponent_recent_5g_runs",
            "team_recent_10g_runs": "opponent_recent_10g_runs",
            "team_season_runs": "opponent_season_runs",
            "team_recent_5g_allowed": "opponent_recent_5g_allowed",
            "team_recent_10g_allowed": "opponent_recent_10g_allowed",
            "team_season_allowed": "opponent_season_allowed",
            "team_recent_5g_win_rate": "opponent_recent_5g_win_rate",
            "team_recent_10g_win_rate": "opponent_recent_10g_win_rate",
            "team_season_win_rate": "opponent_season_win_rate",
            "team_recent_5g_run_diff": "opponent_recent_5g_run_diff",
            "team_recent_10g_run_diff": "opponent_recent_10g_run_diff",
            "rest_days": "opponent_rest_days",
            "back_to_back": "opponent_back_to_back",
        }
    )
    merged = pd.merge(df, opp, on=["game_key", "opponent"], how="left")
    merged["recent_5g_runs_gap"] = merged["team_recent_5g_runs"] - merged["opponent_recent_5g_runs"]
    merged["recent_10g_runs_gap"] = merged["team_recent_10g_runs"] - merged["opponent_recent_10g_runs"]
    merged["season_runs_gap"] = merged["team_season_runs"] - merged["opponent_season_runs"]
    merged["recent_5g_allowed_gap"] = merged["opponent_recent_5g_allowed"] - merged["team_recent_5g_allowed"]
    merged["recent_10g_allowed_gap"] = merged["opponent_recent_10g_allowed"] - merged["team_recent_10g_allowed"]
    merged["season_allowed_gap"] = merged["opponent_season_allowed"] - merged["team_season_allowed"]
    merged["recent_5g_run_diff_gap"] = merged["team_recent_5g_run_diff"] - merged["opponent_recent_5g_run_diff"]
    merged["recent_10g_run_diff_gap"] = merged["team_recent_10g_run_diff"] - merged["opponent_recent_10g_run_diff"]
    merged["recent_10g_win_rate_gap"] = merged["team_recent_10g_win_rate"] - merged["opponent_recent_10g_win_rate"]
    merged["season_win_rate_gap"] = merged["team_season_win_rate"] - merged["opponent_season_win_rate"]
    merged["rest_days_gap"] = merged["rest_days"] - merged["opponent_rest_days"]
    return merged.sort_values(["date", "game_key", "is_home"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame):
    cols = [
        "is_home",
        "month",
        "rest_days",
        "back_to_back",
        "team_recent_3g_runs",
        "team_recent_5g_runs",
        "team_recent_10g_runs",
        "team_season_runs",
        "team_recent_5g_allowed",
        "team_recent_10g_allowed",
        "team_season_allowed",
        "team_recent_5g_win_rate",
        "team_recent_10g_win_rate",
        "team_season_win_rate",
        "team_recent_5g_run_diff",
        "team_recent_10g_run_diff",
        "opponent_recent_5g_runs",
        "opponent_recent_10g_runs",
        "opponent_season_runs",
        "opponent_recent_5g_allowed",
        "opponent_recent_10g_allowed",
        "opponent_season_allowed",
        "opponent_recent_5g_win_rate",
        "opponent_recent_10g_win_rate",
        "opponent_season_win_rate",
        "opponent_recent_5g_run_diff",
        "opponent_recent_10g_run_diff",
        "opponent_rest_days",
        "opponent_back_to_back",
        "recent_5g_runs_gap",
        "recent_10g_runs_gap",
        "season_runs_gap",
        "recent_5g_allowed_gap",
        "recent_10g_allowed_gap",
        "season_allowed_gap",
        "recent_5g_run_diff_gap",
        "recent_10g_run_diff_gap",
        "recent_10g_win_rate_gap",
        "season_win_rate_gap",
        "rest_days_gap",
    ]
    return [col for col in cols if col in df.columns]


def chronological_split(df: pd.DataFrame, train_ratio: float):
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    split_pos = max(1, min(int(len(dates) * train_ratio), len(dates) - 1))
    cutoff = dates.iloc[split_pos]
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy(), cutoff


def train_run_regressors(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]):
    x_train, y_train = train_df[features], train_df["target_runs"]
    x_val, y_val = val_df[features], val_df["target_runs"]
    models = {
        "Poisson": PoissonRegressor(max_iter=500, alpha=0.1),
        "Ridge": Ridge(alpha=2.0),
        "Tweedie": TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=500),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=42),
    }
    trained = {}
    run_scores = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = np.clip(model.predict(x_val), 0, None)
        trained[name] = model
        run_scores.append(
            {
                "model": name,
                "mae": round(float(mean_absolute_error(y_val, preds)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_val, preds))), 4),
            }
        )
    return trained, run_scores


def to_game_level_prediction(frame: pd.DataFrame, pred_col: str):
    home = frame[frame["is_home"].eq(1)][["date", "game_key", "team", "opponent", "target_win", "target_runs", pred_col]].rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "target_win": "target_home_win",
            "target_runs": "home_actual_runs",
            pred_col: "home_expected_runs",
        }
    )
    away = frame[frame["is_home"].eq(0)][["game_key", "target_runs", pred_col]].rename(
        columns={"target_runs": "away_actual_runs", pred_col: "away_expected_runs"}
    )
    games = pd.merge(home, away, on="game_key", how="inner")
    games["expected_run_diff"] = games["home_expected_runs"] - games["away_expected_runs"]
    games["actual_run_diff"] = games["home_actual_runs"] - games["away_actual_runs"]
    return games


def evaluate_win_conversion(trained_models: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]):
    scores = []
    predictions = {}
    for name, model in trained_models.items():
        train_scored = train_df.copy()
        val_scored = val_df.copy()
        pred_col = f"pred_runs_{name}"
        train_scored[pred_col] = np.clip(model.predict(train_scored[features]), 0, None)
        val_scored[pred_col] = np.clip(model.predict(val_scored[features]), 0, None)
        train_games = to_game_level_prediction(train_scored, pred_col)
        val_games = to_game_level_prediction(val_scored, pred_col)

        clf = LogisticRegression(fit_intercept=True)
        clf.fit(train_games[["expected_run_diff"]], train_games["target_home_win"])
        val_games["home_win_probability"] = clf.predict_proba(val_games[["expected_run_diff"]])[:, 1]
        val_games["pred_home_win"] = (val_games["home_win_probability"] >= 0.5).astype(int)
        val_games["predicted_winner"] = np.where(val_games["pred_home_win"].eq(1), val_games["home_team"], val_games["away_team"])
        val_games["actual_winner"] = np.where(val_games["target_home_win"].eq(1), val_games["home_team"], val_games["away_team"])
        val_games["prediction_result"] = np.where(val_games["predicted_winner"].eq(val_games["actual_winner"]), "correct", "wrong")

        score = {
            "model": name,
            "accuracy": round(float(accuracy_score(val_games["target_home_win"], val_games["pred_home_win"])), 4),
            "brier_score": round(float(brier_score_loss(val_games["target_home_win"], val_games["home_win_probability"])), 4),
            "log_loss": round(float(log_loss(val_games["target_home_win"], val_games["home_win_probability"])), 4),
            "run_diff_direction_accuracy": round(float(((val_games["expected_run_diff"] > 0) == (val_games["target_home_win"] == 1)).mean()), 4),
        }
        scores.append(score)
        predictions[name] = val_games
    return scores, predictions


def select_model(run_scores: list[dict], win_scores: list[dict]):
    by_model = {row["model"]: dict(row) for row in run_scores}
    for row in win_scores:
        by_model[row["model"]].update(row)
    candidates = list(by_model.values())
    return sorted(candidates, key=lambda row: (row["mae"], row["brier_score"], -row["accuracy"]))[0], candidates


def error_tags(row: pd.Series):
    tags = []
    wrong = row["prediction_result"] == "wrong"
    actual_total = row["home_actual_runs"] + row["away_actual_runs"]
    actual_diff_abs = abs(row["actual_run_diff"])
    expected_diff_abs = abs(row["expected_run_diff"])
    direction_miss = (row["expected_run_diff"] > 0) != (row["actual_run_diff"] > 0)
    if wrong and actual_total <= 6:
        tags.append("LOW_SCORING_MISS")
    if wrong and actual_total >= 12:
        tags.append("HIGH_SCORING_MISS")
    if direction_miss:
        tags.append("RUN_DIFF_DIRECTION_MISS")
    if actual_diff_abs >= 5 and expected_diff_abs < 1:
        tags.append("BLOWOUT_UNDERPREDICTED")
    if wrong and actual_diff_abs <= 1:
        tags.append("CLOSE_GAME_NOISE")
    return "|".join(tags) if tags else "NORMAL"


def scoring_bucket(total_runs: float):
    if total_runs <= 6:
        return "LOW_0_6"
    if total_runs >= 12:
        return "HIGH_12_PLUS"
    return "MID_7_11"


def build_error_analysis(predictions: pd.DataFrame):
    frame = predictions.copy()
    frame["actual_total_runs"] = frame["home_actual_runs"] + frame["away_actual_runs"]
    frame["expected_total_runs"] = frame["home_expected_runs"] + frame["away_expected_runs"]
    frame["home_run_error"] = frame["home_expected_runs"] - frame["home_actual_runs"]
    frame["away_run_error"] = frame["away_expected_runs"] - frame["away_actual_runs"]
    frame["home_abs_error"] = frame["home_run_error"].abs()
    frame["away_abs_error"] = frame["away_run_error"].abs()
    frame["total_run_error"] = frame["expected_total_runs"] - frame["actual_total_runs"]
    frame["total_abs_error"] = frame["total_run_error"].abs()
    frame["run_diff_error"] = frame["expected_run_diff"] - frame["actual_run_diff"]
    frame["run_diff_abs_error"] = frame["run_diff_error"].abs()
    frame["actual_total_bucket"] = frame["actual_total_runs"].apply(scoring_bucket)
    frame["error_tags"] = frame.apply(error_tags, axis=1)
    return frame


def error_summary(error_frame: pd.DataFrame):
    bucket_rows = []
    for bucket, subset in error_frame.groupby("actual_total_bucket", sort=False):
        bucket_rows.append(
            {
                "bucket": bucket,
                "games": int(len(subset)),
                "home_mae": round(float(subset["home_abs_error"].mean()), 4),
                "away_mae": round(float(subset["away_abs_error"].mean()), 4),
                "total_mae": round(float(subset["total_abs_error"].mean()), 4),
                "run_diff_mae": round(float(subset["run_diff_abs_error"].mean()), 4),
                "accuracy": round(float((subset["prediction_result"] == "correct").mean()), 4),
            }
        )
    tag_rows = []
    exploded = error_frame.assign(error_tag=error_frame["error_tags"].str.split("|")).explode("error_tag")
    for tag, subset in exploded.groupby("error_tag", sort=False):
        tag_rows.append({"tag": tag, "games": int(len(subset))})
    return {"score_bucket_error": bucket_rows, "error_tag_counts": tag_rows}


def selected_feature_importance(model, val_df: pd.DataFrame, features: list[str]):
    result = permutation_importance(
        model,
        val_df[features],
        val_df["target_runs"],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    rows = []
    for feature, mean_value, std_value in zip(features, result.importances_mean, result.importances_std):
        rows.append(
            {
                "feature": feature,
                "importance_mean": round(float(max(mean_value, 0.0)), 6),
                "importance_std": round(float(std_value), 6),
            }
        )
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)


def run_pipeline(input_path: Path, output_dir: Path, train_ratio: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    team_games = load_completed_team_games(input_path)
    run_df = build_base_run_df(team_games)
    run_df = create_rolling_run_features(run_df)
    run_df = merge_opponent_features(run_df)
    features = feature_columns(run_df)
    run_df.to_csv(output_dir / "run_model_features.csv", index=False, encoding="utf-8-sig")

    train_df, val_df, cutoff = chronological_split(run_df, train_ratio)
    trained_models, run_scores = train_run_regressors(train_df, val_df, features)
    win_scores, prediction_map = evaluate_win_conversion(trained_models, train_df, val_df, features)
    selected, candidate_scores = select_model(run_scores, win_scores)

    selected_predictions = prediction_map[selected["model"]].copy()
    selected_error_analysis = build_error_analysis(selected_predictions)
    selected_error_analysis["date"] = pd.to_datetime(selected_error_analysis["date"]).dt.strftime("%Y-%m-%d")
    error_columns = [
        "date",
        "game_key",
        "home_team",
        "away_team",
        "home_expected_runs",
        "away_expected_runs",
        "home_actual_runs",
        "away_actual_runs",
        "expected_run_diff",
        "actual_run_diff",
        "home_win_probability",
        "predicted_winner",
        "actual_winner",
        "prediction_result",
        "actual_total_bucket",
        "home_abs_error",
        "away_abs_error",
        "total_abs_error",
        "run_diff_abs_error",
        "error_tags",
    ]
    selected_error_analysis[error_columns].to_csv(output_dir / "run_model_error_analysis.csv", index=False, encoding="utf-8-sig")
    importance_rows = selected_feature_importance(trained_models[selected["model"]], val_df, features)
    pd.DataFrame(importance_rows).to_csv(output_dir / "run_model_feature_importance.csv", index=False, encoding="utf-8-sig")

    selected_predictions["date"] = pd.to_datetime(selected_predictions["date"]).dt.strftime("%Y-%m-%d")
    selected_predictions = selected_predictions[
        [
            "date",
            "game_key",
            "home_team",
            "away_team",
            "home_expected_runs",
            "away_expected_runs",
            "expected_run_diff",
            "home_win_probability",
            "predicted_winner",
            "actual_winner",
            "prediction_result",
            "home_actual_runs",
            "away_actual_runs",
        ]
    ]
    for col in ["home_expected_runs", "away_expected_runs", "expected_run_diff", "home_win_probability"]:
        selected_predictions[col] = selected_predictions[col].round(4)
    selected_predictions.to_csv(output_dir / "expected_runs_predictions.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "model_scope": "independent_run_model",
        "notes": [
            "기존 대시보드/승패 모델과 분리된 득점 기반 모델입니다.",
            "기존 modeling/results 산출물은 사용하지 않습니다.",
            "완료 경기 원천 CSV만 읽기 전용으로 사용합니다.",
        ],
        "train_ratio": train_ratio,
        "training_cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "feature_rows": int(len(run_df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "validation_games": int(len(selected_predictions)),
        "feature_columns": features,
        "run_regression_scores": run_scores,
        "win_conversion_scores": win_scores,
        "candidate_scores": candidate_scores,
        "selected_model": selected,
        "error_analysis_summary": error_summary(selected_error_analysis),
        "feature_importance_top20": importance_rows[:20],
        "output_files": [
            "results/run_model_features.csv",
            "results/expected_runs_predictions.csv",
            "results/expected_runs_model.json",
            "results/run_model_error_analysis.csv",
            "results/run_model_feature_importance.csv",
        ],
    }
    (output_dir / "expected_runs_model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Independent KBO expected-runs prediction model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Completed team-game CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS, help="Directory for run model outputs")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Chronological train split ratio")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run_pipeline(args.input, args.output_dir, args.train_ratio)
    selected = payload["selected_model"]
    print("Independent KBO run model completed")
    print(f"selected_model={selected['model']}")
    print(f"mae={selected['mae']}")
    print(f"rmse={selected['rmse']}")
    print(f"accuracy={selected['accuracy']}")
    print(f"brier_score={selected['brier_score']}")
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
