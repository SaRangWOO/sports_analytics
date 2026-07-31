from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .atomic_io import atomic_write_json
from .runner import run_managed_task, stable_checksum


def evaluate_postgame_rows(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"actual_home_win", "home_win_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing postgame columns: {sorted(missing)}")
    completed = frame.dropna(subset=list(required)).copy()
    if completed.empty:
        return {
            "completed_games": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "calibration": [],
        }
    y = completed["actual_home_win"].astype(int)
    probability = completed["home_win_probability"].astype(float).clip(1e-9, 1 - 1e-9)
    prediction = (probability >= 0.5).astype(int)
    bins = pd.cut(
        probability,
        bins=[0, 0.45, 0.5, 0.55, 1],
        include_lowest=True,
    )
    calibration = []
    for interval, group in completed.assign(_bin=bins).groupby("_bin", observed=False):
        if group.empty:
            continue
        calibration.append(
            {
                "bucket": str(interval),
                "games": int(len(group)),
                "average_probability": round(float(group["home_win_probability"].mean()), 4),
                "actual_rate": round(float(group["actual_home_win"].mean()), 4),
            }
        )
    return {
        "completed_games": int(len(completed)),
        "accuracy": round(float((prediction == y).mean()), 4),
        "brier_score": round(float(((probability - y) ** 2).mean()), 4),
        "log_loss": round(
            float(-(y * probability.map(math.log) + (1 - y) * (1 - probability).map(math.log)).mean()),
            4,
        ),
        "calibration": calibration,
    }


def run_postgame(
    config,
    reference_date: date,
    reference_datetime: datetime,
    audit_frame: pd.DataFrame,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    if dry_run:
        return {
            "dry_run": True,
            "report_path": str(config.report_root / f"postgame-{reference_date}.json"),
            "input_rows": len(audit_frame),
        }

    def action():
        metrics = evaluate_postgame_rows(audit_frame)
        report_path = config.report_root / f"postgame-{reference_date}.json"
        atomic_write_json(report_path, metrics)
        return {
            "rows_before": len(audit_frame),
            "rows_after": metrics["completed_games"],
            "report_path": str(report_path),
            "metrics": metrics,
        }

    return run_managed_task(
        config,
        "postgame",
        reference_date,
        reference_datetime,
        action,
        run_id=run_id,
        update_stage="postgame",
        input_checksum=stable_checksum(audit_frame.to_dict(orient="records")),
        force=force,
    )
