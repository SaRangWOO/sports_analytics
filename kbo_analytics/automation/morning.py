from __future__ import annotations

from datetime import date, datetime

from .prediction import run_prediction_update
from .runner import run_managed_task, stable_checksum


def run_morning(
    config,
    reference_date: date,
    reference_datetime: datetime,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    if dry_run:
        return run_prediction_update(
            config,
            run_id or "dry-run",
            reference_date,
            reference_datetime,
            "morning",
            dry_run=True,
        )
    return run_managed_task(
        config,
        "morning",
        reference_date,
        reference_datetime,
        lambda: run_prediction_update(
            config,
            run_id or "morning",
            reference_date,
            reference_datetime,
            "morning",
        ),
        run_id=run_id,
        update_stage="morning",
        input_checksum=stable_checksum(
            {
                "reference_date": reference_date.isoformat(),
                "stage": "morning",
            }
        ),
        force=force,
    )
