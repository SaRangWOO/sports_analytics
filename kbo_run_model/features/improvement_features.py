from __future__ import annotations

import numpy as np
import pandas as pd


PARK_FEATURE_COLUMNS = [
    "ballpark_avg_total_runs",
    "ballpark_run_factor",
    "ballpark_home_run_factor",
    "ballpark_sample_games",
    "ballpark_total_runs_recent",
    "ballpark_total_runs_season",
]

BIAS_FEATURE_COLUMNS = [
    "team_recent_prediction_error",
    "team_recent_prediction_bias_5g",
    "team_recent_prediction_bias_10g",
    "opponent_recent_allowed_prediction_bias_5g",
    "opponent_recent_allowed_prediction_bias_10g",
]

IMPROVEMENT_FEATURE_COLUMNS = PARK_FEATURE_COLUMNS + BIAS_FEATURE_COLUMNS


def _league_prior(home_games: pd.DataFrame, season: int, column: str, default: float) -> float:
    previous = home_games[home_games["season"].eq(season - 1)]
    if not previous.empty:
        return float(previous[column].mean())
    history = home_games[home_games["season"].lt(season)]
    if not history.empty:
        return float(history[column].mean())
    return default


def add_park_factor_features(feature_df: pd.DataFrame, shrinkage_games: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = feature_df.copy()
    if "ballpark" not in df.columns:
        for column in PARK_FEATURE_COLUMNS:
            df[column] = 0 if column == "ballpark_sample_games" else 1.0
        return df, pd.DataFrame(columns=["ballpark", "games", "avg_total_runs", "run_factor", "sample_note"])

    home_games = df[df["is_home"].eq(1)].copy().sort_values(["season", "date", "game_key"])
    home_games["actual_total_runs"] = home_games["target_runs"] + home_games["runs_allowed"]
    home_games["actual_home_runs"] = home_games["target_runs"]
    league_total_default = float(home_games["actual_total_runs"].mean())
    league_home_default = float(home_games["actual_home_runs"].mean())
    records: list[dict] = []
    park_history: dict[tuple[int, str], list[dict[str, float]]] = {}

    for (season, current_date), date_group in home_games.groupby(["season", "date"], sort=True):
        league_past = home_games[
            home_games["season"].eq(season)
            & (
                (home_games["date"].lt(current_date))
            )
        ]
        league_total = float(league_past["actual_total_runs"].mean()) if not league_past.empty else _league_prior(
            home_games, int(season), "actual_total_runs", league_total_default
        )
        league_home = float(league_past["actual_home_runs"].mean()) if not league_past.empty else _league_prior(
            home_games, int(season), "actual_home_runs", league_home_default
        )

        for row in date_group.itertuples(index=False):
            key = (int(season), str(row.ballpark))
            history = park_history.get(key, [])
            total_values = [item["total_runs"] for item in history]
            home_values = [item["home_runs"] for item in history]
            sample_games = len(history)
            prior_total = _league_prior(home_games, int(season), "actual_total_runs", league_total)
            prior_home = _league_prior(home_games, int(season), "actual_home_runs", league_home)
            total_mean = float(np.mean(total_values)) if total_values else prior_total
            home_mean = float(np.mean(home_values)) if home_values else prior_home
            recent_total = float(np.mean(total_values[-10:])) if total_values else prior_total
            shrunk_total = ((total_mean * sample_games) + (league_total * shrinkage_games)) / (sample_games + shrinkage_games)
            shrunk_home = ((home_mean * sample_games) + (league_home * shrinkage_games)) / (sample_games + shrinkage_games)
            records.append(
                {
                    "game_key": row.game_key,
                    "ballpark_avg_total_runs": shrunk_total,
                    "ballpark_run_factor": shrunk_total / league_total if league_total else 1.0,
                    "ballpark_home_run_factor": shrunk_home / league_home if league_home else 1.0,
                    "ballpark_sample_games": sample_games,
                    "ballpark_total_runs_recent": recent_total,
                    "ballpark_total_runs_season": total_mean,
                }
            )

        for row in date_group.itertuples(index=False):
            key = (int(season), str(row.ballpark))
            park_history.setdefault(key, []).append({"total_runs": float(row.actual_total_runs), "home_runs": float(row.actual_home_runs)})

    park_features = pd.DataFrame(records)
    df = df.merge(park_features, on="game_key", how="left")
    for column in PARK_FEATURE_COLUMNS:
        if column == "ballpark_sample_games":
            df[column] = df[column].fillna(0)
        else:
            df[column] = df[column].fillna(1.0)

    metrics = (
        home_games.groupby("ballpark", as_index=False)
        .agg(
            games=("game_key", "size"),
            avg_total_runs=("actual_total_runs", "mean"),
            avg_home_runs=("actual_home_runs", "mean"),
        )
        .sort_values("games")
    )
    metrics["run_factor"] = metrics["avg_total_runs"] / league_total_default if league_total_default else 1.0
    sample_warning_games = max(shrinkage_games, 40)
    metrics["sample_note"] = np.where(metrics["games"].le(sample_warning_games), "표본 주의", "일반")
    for column in ["avg_total_runs", "avg_home_runs", "run_factor"]:
        metrics[column] = metrics[column].round(4)
    return df, metrics


def add_prediction_bias_features(feature_df: pd.DataFrame, baseline_predicted_runs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = feature_df.copy().sort_values(["season", "date", "game_key", "is_home"]).reset_index(drop=True)
    df["baseline_predicted_runs"] = np.clip(baseline_predicted_runs, 0, None)
    opponent_prediction = df[["game_key", "team", "baseline_predicted_runs"]].rename(
        columns={"team": "opponent", "baseline_predicted_runs": "opponent_baseline_predicted_runs"}
    )
    df = df.merge(opponent_prediction, on=["game_key", "opponent"], how="left")
    df["prediction_error"] = df["baseline_predicted_runs"] - df["target_runs"]
    df["allowed_prediction_error"] = df["opponent_baseline_predicted_runs"] - df["runs_allowed"]

    records: list[dict[str, float | str]] = []
    for (season, team), group in df.sort_values(["season", "team", "date", "game_key"]).groupby(["season", "team"], sort=False):
        run_errors: list[float] = []
        allowed_errors: list[float] = []
        for current_date, date_group in group.groupby("date", sort=True):
            recent_error = float(np.mean(run_errors[-10:])) if run_errors else 0.0
            bias_5 = float(np.mean(run_errors[-5:])) if run_errors else 0.0
            bias_10 = float(np.mean(run_errors[-10:])) if run_errors else 0.0
            allowed_bias_5 = float(np.mean(allowed_errors[-5:])) if allowed_errors else 0.0
            allowed_bias_10 = float(np.mean(allowed_errors[-10:])) if allowed_errors else 0.0
            for row in date_group.itertuples(index=False):
                records.append(
                    {
                        "game_id": row.game_id,
                        "team_recent_prediction_error": recent_error,
                        "team_recent_prediction_bias_5g": bias_5,
                        "team_recent_prediction_bias_10g": bias_10,
                        "opponent_recent_allowed_prediction_bias_5g": allowed_bias_5,
                        "opponent_recent_allowed_prediction_bias_10g": allowed_bias_10,
                    }
                )
            run_errors.extend(date_group["prediction_error"].astype(float).tolist())
            allowed_errors.extend(date_group["allowed_prediction_error"].astype(float).tolist())

    bias_features = pd.DataFrame(records)
    df = df.merge(bias_features, on="game_id", how="left")
    for column in BIAS_FEATURE_COLUMNS:
        df[column] = df[column].fillna(0.0)

    metrics = (
        df.groupby("team", as_index=False)
        .agg(
            games=("game_id", "size"),
            avg_prediction_error=("prediction_error", "mean"),
            avg_abs_prediction_error=("prediction_error", lambda value: value.abs().mean()),
            avg_allowed_prediction_error=("allowed_prediction_error", "mean"),
            avg_abs_allowed_prediction_error=("allowed_prediction_error", lambda value: value.abs().mean()),
        )
        .sort_values("avg_abs_prediction_error", ascending=False)
    )
    metrics["bias_direction"] = np.select(
        [metrics["avg_prediction_error"].gt(0.25), metrics["avg_prediction_error"].lt(-0.25)],
        ["과대예측 보정 후보", "과소예측 보정 후보"],
        default="중립",
    )
    for column in [
        "avg_prediction_error",
        "avg_abs_prediction_error",
        "avg_allowed_prediction_error",
        "avg_abs_allowed_prediction_error",
    ]:
        metrics[column] = metrics[column].round(4)
    return df.drop(columns=["baseline_predicted_runs", "opponent_baseline_predicted_runs"]), metrics


def build_improvement_feature_matrix(
    feature_df: pd.DataFrame,
    baseline_model: object,
    baseline_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    park_df, park_metrics = add_park_factor_features(feature_df)
    baseline_predictions = baseline_model.predict(park_df[baseline_feature_columns])
    improved_df, bias_metrics = add_prediction_bias_features(park_df, baseline_predictions)
    improved_columns = baseline_feature_columns + IMPROVEMENT_FEATURE_COLUMNS
    return improved_df, improved_columns, park_metrics, bias_metrics
