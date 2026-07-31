from __future__ import annotations

from datetime import date, datetime

from .prediction import run_prediction_update
from .runner import run_managed_task, stable_checksum


def run_pregame(
    config,
    reference_date: date,
    reference_datetime: datetime,
    *,
    game_id: str,
    source_checksum: str,
    run_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    if not game_id:
        raise ValueError("pregame update requires official gameId")
    if dry_run:
        return run_prediction_update(
            config,
            run_id or "dry-run",
            reference_date,
            reference_datetime,
            "pregame",
            dry_run=True,
        )
    return run_managed_task(
        config,
        "pregame",
        reference_date,
        reference_datetime,
        lambda: run_prediction_update(
            config,
            run_id or f"pregame-{game_id}",
            reference_date,
            reference_datetime,
            "pregame",
        ),
        run_id=run_id,
        game_id=game_id,
        update_stage="pregame",
        input_checksum=stable_checksum(
            {
                "game_id": game_id,
                "source_checksum": source_checksum,
                "stage": "pregame",
            }
        ),
        force=force,
    )
