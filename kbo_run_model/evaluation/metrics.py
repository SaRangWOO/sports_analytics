from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss


RUN_DIFF_BINS = [-np.inf, -3, -1, 1, 3, np.inf]
RUN_DIFF_LABELS = ["-3 이하", "-3 초과 -1 이하", "-1 초과 1 이하", "1 초과 3 미만", "3 이상"]
ABS_RUN_DIFF_BINS = [0, 0.5, 1.0, 1.5, np.inf]
ABS_RUN_DIFF_LABELS = ["0.0 ~ 0.5점", "0.5 ~ 1.0점", "1.0 ~ 1.5점", "1.5점 이상"]


def to_game_predictions(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    home = frame[frame["is_home"].eq(1)][
        ["date", "season", "game_key", "team", "opponent", "target_win", "target_runs", prediction_column]
    ].rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "target_win": "home_actual_win",
            "target_runs": "home_actual_runs",
            prediction_column: "home_expected_runs",
        }
    )
    away = frame[frame["is_home"].eq(0)][["game_key", "target_runs", prediction_column]].rename(
        columns={"target_runs": "away_actual_runs", prediction_column: "away_expected_runs"}
    )
    games = home.merge(away, on="game_key", how="inner")
    games["expected_run_diff"] = games["home_expected_runs"] - games["away_expected_runs"]
    games["actual_run_diff"] = games["home_actual_runs"] - games["away_actual_runs"]
    games["expected_total_runs"] = games["home_expected_runs"] + games["away_expected_runs"]
    return games


def add_win_probability(train_games: pd.DataFrame, validation_games: pd.DataFrame) -> pd.DataFrame:
    converter = LogisticRegression()
    converter.fit(train_games[["expected_run_diff"]], train_games["home_actual_win"])
    scored = validation_games.copy()
    scored["home_win_probability"] = converter.predict_proba(scored[["expected_run_diff"]])[:, 1]
    scored["predicted_home_win"] = (scored["home_win_probability"] >= 0.5).astype(int)
    scored["predicted_winner"] = np.where(scored["predicted_home_win"].eq(1), scored["home_team"], scored["away_team"])
    scored["actual_winner"] = np.where(scored["home_actual_win"].eq(1), scored["home_team"], scored["away_team"])
    scored["confidence"] = (scored["home_win_probability"] - 0.5).abs() * 2
    scored["brier_component"] = (scored["home_win_probability"] - scored["home_actual_win"]) ** 2
    return scored


def run_diff_bucket_accuracy(scored_games: pd.DataFrame) -> list[dict[str, float | str | int]]:
    bucketed = scored_games.copy()
    bucketed["run_diff_bucket"] = pd.cut(bucketed["expected_run_diff"], bins=RUN_DIFF_BINS, labels=RUN_DIFF_LABELS)
    rows: list[dict[str, float | str | int]] = []
    for bucket, group in bucketed.groupby("run_diff_bucket", observed=False):
        if group.empty:
            rows.append({"bucket": str(bucket), "games": 0, "accuracy": ""})
            continue
        correct = group["predicted_home_win"].eq(group["home_actual_win"])
        rows.append({"bucket": str(bucket), "games": int(len(group)), "accuracy": round(float(correct.mean()), 4)})
    return rows


def abs_run_diff_bucket_metrics(scored_games: pd.DataFrame) -> list[dict[str, float | str | int]]:
    bucketed = scored_games.copy()
    bucketed["abs_run_diff_bucket"] = pd.cut(
        bucketed["expected_run_diff"].abs(),
        bins=ABS_RUN_DIFF_BINS,
        labels=ABS_RUN_DIFF_LABELS,
        include_lowest=True,
    )
    rows: list[dict[str, float | str | int]] = []
    for bucket, group in bucketed.groupby("abs_run_diff_bucket", observed=False):
        if group.empty:
            rows.append({"bucket": str(bucket), "games": 0, "accuracy": "", "avg_brier_score": ""})
            continue
        correct = group["predicted_home_win"].eq(group["home_actual_win"])
        rows.append(
            {
                "bucket": str(bucket),
                "games": int(len(group)),
                "accuracy": round(float(correct.mean()), 4),
                "avg_brier_score": round(float(group["brier_component"].mean()), 4),
            }
        )
    return rows


def season_metrics(scored_games: pd.DataFrame) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for season, group in scored_games.groupby("season", sort=True):
        run_actual = pd.concat([group["home_actual_runs"], group["away_actual_runs"]], ignore_index=True)
        run_predicted = pd.concat([group["home_expected_runs"], group["away_expected_runs"]], ignore_index=True)
        run_errors = run_actual - run_predicted
        rows.append(
            {
                "season": int(season),
                "games": int(len(group)),
                "run_mae": round(float(run_errors.abs().mean()), 4),
                "run_rmse": round(float(np.sqrt((run_errors**2).mean())), 4),
                "home_win_accuracy": round(float(group["predicted_home_win"].eq(group["home_actual_win"]).mean()), 4),
                "brier_score": round(float(group["brier_component"].mean()), 4),
            }
        )
    return rows


def team_bias_metrics(scored_games: pd.DataFrame) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str]], list[dict[str, float | str]]]:
    home = scored_games[["home_team", "home_actual_runs", "home_expected_runs"]].rename(
        columns={"home_team": "team", "home_actual_runs": "actual_runs", "home_expected_runs": "predicted_runs"}
    )
    away = scored_games[["away_team", "away_actual_runs", "away_expected_runs"]].rename(
        columns={"away_team": "team", "away_actual_runs": "actual_runs", "away_expected_runs": "predicted_runs"}
    )
    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows["error"] = team_rows["predicted_runs"] - team_rows["actual_runs"]
    team_rows["abs_error"] = team_rows["error"].abs()
    metrics = (
        team_rows.groupby("team", as_index=False)
        .agg(
            games=("team", "size"),
            actual_avg_runs=("actual_runs", "mean"),
            predicted_avg_runs=("predicted_runs", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
        )
        .sort_values("bias", ascending=False)
    )
    for column in ["actual_avg_runs", "predicted_avg_runs", "mae", "bias"]:
        metrics[column] = metrics[column].round(4)
    over = metrics[metrics["bias"] > 0].head(5)[["team", "bias", "mae"]].to_dict("records")
    under = metrics[metrics["bias"] < 0].sort_values("bias").head(5)[["team", "bias", "mae"]].to_dict("records")
    return metrics.to_dict("records"), over, under


def evaluate_win_conversion(
    trained_models: dict[str, object],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    list[dict[str, float | str]],
    dict[str, pd.DataFrame],
    dict[str, list[dict[str, float | str | int]]],
    dict[str, list[dict[str, float | str | int]]],
]:
    scores: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}
    buckets: dict[str, list[dict[str, float | str | int]]] = {}
    abs_buckets: dict[str, list[dict[str, float | str | int]]] = {}

    for name, model in trained_models.items():
        prediction_column = f"predicted_runs_{name}"
        train_scored = train_df.copy()
        validation_scored = validation_df.copy()
        train_scored[prediction_column] = np.clip(model.predict(train_scored[feature_columns]), 0, None)
        validation_scored[prediction_column] = np.clip(model.predict(validation_scored[feature_columns]), 0, None)

        train_games = to_game_predictions(train_scored, prediction_column)
        validation_games = to_game_predictions(validation_scored, prediction_column)
        scored_games = add_win_probability(train_games, validation_games)

        scores.append(
            {
                "model": name,
                "home_win_accuracy": round(float(accuracy_score(scored_games["home_actual_win"], scored_games["predicted_home_win"])), 4),
                "brier_score": round(float(brier_score_loss(scored_games["home_actual_win"], scored_games["home_win_probability"])), 4),
            }
        )
        predictions[name] = scored_games
        buckets[name] = run_diff_bucket_accuracy(scored_games)
        abs_buckets[name] = abs_run_diff_bucket_metrics(scored_games)

    return scores, predictions, buckets, abs_buckets


def select_model(run_scores: list[dict[str, float | str]], win_scores: list[dict[str, float | str]]) -> dict[str, float | str]:
    combined = {row["model"]: dict(row) for row in run_scores}
    for row in win_scores:
        combined[row["model"]].update(row)
    return sorted(combined.values(), key=lambda row: (row["run_mae"], row["brier_score"], -row["home_win_accuracy"]))[0]
