from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


STAGES = ("morning", "pregame")
METRIC_COLUMNS = [
    "dimension",
    "value",
    "games",
    "accuracy",
    "brier",
    "log_loss",
    "over_55_games",
    "over_55_accuracy",
    "avg_probability",
]
HISTORY_COLUMNS = {
    "run_time",
    "reference_date",
    "update_stage",
    "game_id",
    "away_team",
    "home_team",
    "predicted_team",
    "win_probability",
    "starter_status",
    "lineup_status",
}
GAME_COLUMNS = {"game_id", "status", "result", "team"}


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")


def completed_game_results(
    games: pd.DataFrame, expected_game_ids: set[str] | None = None
) -> pd.DataFrame:
    _require_columns(games, GAME_COLUMNS, "games")
    completed = games[
        games["status"].eq("Final") & games["result"].isin(["Win", "Loss"])
    ].copy()
    completed["actual_game_id"] = (
        completed["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    )
    if expected_game_ids is not None:
        completed = completed[completed["actual_game_id"].isin(expected_game_ids)]
    winners = completed[completed["result"].eq("Win")][
        ["actual_game_id", "team"]
    ].rename(columns={"team": "actual_winner"})
    winner_counts = winners.groupby("actual_game_id")["actual_winner"].nunique()
    invalid = winner_counts[winner_counts.ne(1)]
    if not invalid.empty:
        raise ValueError(f"completed games have ambiguous winners: {invalid.index.tolist()}")
    return winners.drop_duplicates("actual_game_id").reset_index(drop=True)


def latest_stage_predictions(history: pd.DataFrame) -> pd.DataFrame:
    _require_columns(history, HISTORY_COLUMNS, "prediction history")
    frame = history[history["update_stage"].isin(STAGES)].copy()
    frame["run_time"] = pd.to_datetime(frame["run_time"], errors="raise")
    frame["win_probability"] = pd.to_numeric(
        frame["win_probability"], errors="raise"
    )
    invalid = ~frame["win_probability"].between(0.5, 1.0, inclusive="both")
    if invalid.any():
        raise ValueError("win_probability must be the predicted-team probability in [0.5, 1]")
    return (
        frame.sort_values("run_time")
        .groupby(["game_id", "update_stage"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _metric_row(dimension: str, value: str, frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "dimension": dimension,
            "value": value,
            "games": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "over_55_games": 0,
            "over_55_accuracy": None,
            "avg_probability": None,
        }
    target = frame["correct"].astype(int).to_numpy()
    probability = frame["win_probability"].astype(float).to_numpy()
    over_55 = probability >= 0.55
    return {
        "dimension": dimension,
        "value": value,
        "games": int(len(frame)),
        "accuracy": round(float(target.mean()), 4),
        "brier": round(float(brier_score_loss(target, probability)), 4),
        "log_loss": round(
            float(
                log_loss(
                    target,
                    np.clip(probability, 1e-6, 1 - 1e-6),
                    labels=[0, 1],
                )
            ),
            4,
        ),
        "over_55_games": int(over_55.sum()),
        "over_55_accuracy": (
            round(float(target[over_55].mean()), 4) if over_55.any() else None
        ),
        "avg_probability": round(float(probability.mean()), 4),
    }


def evaluate_stage_predictions(
    history: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    latest = latest_stage_predictions(history)
    results = completed_game_results(games, set(latest["game_id"].astype(str)))
    evaluated = latest.merge(
        results, left_on="game_id", right_on="actual_game_id", how="inner"
    )
    evaluated["correct"] = evaluated["predicted_team"].eq(
        evaluated["actual_winner"]
    )

    metric_rows = []
    for stage, group in evaluated.groupby("update_stage", sort=False):
        metric_rows.append(_metric_row("update_stage", str(stage), group))
    for column in ["starter_status", "lineup_status"]:
        for (stage, status), group in evaluated.groupby(
            ["update_stage", column], sort=False
        ):
            metric_rows.append(
                _metric_row(f"{stage}_{column}", str(status), group)
            )

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    paired = _paired_predictions(evaluated)
    summary = _stage_summary(metrics, paired)
    return metrics, paired, summary


def _paired_predictions(evaluated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "reference_date",
        "game_id",
        "away_team",
        "home_team",
        "update_stage",
        "predicted_team",
        "win_probability",
        "correct",
        "actual_winner",
        "starter_status",
        "lineup_status",
    ]
    paired = evaluated[columns].pivot(
        index=["reference_date", "game_id", "away_team", "home_team", "actual_winner"],
        columns="update_stage",
        values=[
            "predicted_team",
            "win_probability",
            "correct",
            "starter_status",
            "lineup_status",
        ],
    )
    required = [(field, stage) for field in ["predicted_team", "win_probability", "correct"] for stage in STAGES]
    if any(column not in paired.columns for column in required):
        return pd.DataFrame()
    paired = paired.dropna(subset=required).reset_index()
    paired.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in paired.columns
    ]
    paired["direction_changed"] = paired["predicted_team_morning"].ne(
        paired["predicted_team_pregame"]
    )
    paired["probability_delta"] = (
        paired["win_probability_pregame"].astype(float)
        - paired["win_probability_morning"].astype(float)
    )
    paired["absolute_probability_delta"] = paired["probability_delta"].abs()
    paired["pregame_improved_result"] = (
        ~paired["correct_morning"].astype(bool)
        & paired["correct_pregame"].astype(bool)
    )
    paired["pregame_harmed_result"] = (
        paired["correct_morning"].astype(bool)
        & ~paired["correct_pregame"].astype(bool)
    )
    return paired


def _stage_summary(metrics: pd.DataFrame, paired: pd.DataFrame) -> dict:
    stage_rows = metrics[metrics["dimension"].eq("update_stage")].set_index("value")
    morning = stage_rows.loc["morning"].to_dict() if "morning" in stage_rows.index else {}
    pregame = stage_rows.loc["pregame"].to_dict() if "pregame" in stage_rows.index else {}
    paired_games = int(len(paired))
    changed = int(paired["direction_changed"].sum()) if paired_games else 0
    probability_updates = (
        int(paired["absolute_probability_delta"].ge(0.005).sum())
        if paired_games
        else 0
    )
    morning_accuracy = float(paired["correct_morning"].mean()) if paired_games else None
    pregame_accuracy = float(paired["correct_pregame"].mean()) if paired_games else None
    accuracy_lift = (
        round(pregame_accuracy - morning_accuracy, 4)
        if paired_games
        else None
    )
    gates = {
        "minimum_paired_games": paired_games >= 100,
        "probability_updates_are_material": (
            probability_updates / paired_games >= 0.2 if paired_games else False
        ),
        "pregame_accuracy_lift_at_least_1pp": (
            accuracy_lift is not None and accuracy_lift >= 0.01
        ),
        "pregame_brier_not_worse": (
            bool(morning)
            and bool(pregame)
            and pregame.get("brier") is not None
            and morning.get("brier") is not None
            and pregame["brier"] <= morning["brier"]
        ),
    }
    return {
        "paired_games": paired_games,
        "direction_changes": changed,
        "direction_change_rate": round(changed / paired_games, 4) if paired_games else 0.0,
        "probability_updates_ge_0_5pp": probability_updates,
        "probability_update_rate": (
            round(probability_updates / paired_games, 4) if paired_games else 0.0
        ),
        "mean_absolute_probability_delta": (
            round(float(paired["absolute_probability_delta"].mean()), 4)
            if paired_games
            else None
        ),
        "paired_morning_accuracy": (
            round(morning_accuracy, 4) if morning_accuracy is not None else None
        ),
        "paired_pregame_accuracy": (
            round(pregame_accuracy, 4) if pregame_accuracy is not None else None
        ),
        "paired_accuracy_lift": accuracy_lift,
        "pregame_improved_games": (
            int(paired["pregame_improved_result"].sum()) if paired_games else 0
        ),
        "pregame_harmed_games": (
            int(paired["pregame_harmed_result"].sum()) if paired_games else 0
        ),
        "stage_metrics": {"morning": morning, "pregame": pregame},
        "validation_gates": gates,
        "pregame_value_gate_passed": all(gates.values()),
        "decision": (
            "pregame_model_value_demonstrated"
            if all(gates.values())
            else "pregame_display_updates_do_not_yet_demonstrate_model_value"
        ),
    }


def write_stage_validation_reports(
    history_path: Path, games_path: Path, output_dir: Path
) -> dict:
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    games = pd.read_csv(games_path, encoding="utf-8-sig")
    metrics, paired, summary = evaluate_stage_predictions(history, games)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "pregame_stage_performance_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "pregame_stage_transition_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "pregame_stage_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
