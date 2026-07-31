from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


REQUIRED_SCHEDULE_COLUMNS = {
    "official_game_id",
    "scheduled_start_datetime",
    "status",
}


def eligible_pregame_games(
    schedule: pd.DataFrame,
    reference_datetime: datetime,
    window_start_minutes: int,
    window_end_minutes: int,
) -> list[dict[str, Any]]:
    missing = REQUIRED_SCHEDULE_COLUMNS - set(schedule.columns)
    if missing:
        raise ValueError(f"missing schedule columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for _, row in schedule.iterrows():
        if str(row["status"]) != "Scheduled":
            continue
        game_id = str(row["official_game_id"]).strip()
        if not game_id:
            raise ValueError("official gameId is required")
        start = pd.to_datetime(row["scheduled_start_datetime"], errors="coerce")
        if pd.isna(start):
            raise ValueError(f"scheduled start is required: {game_id}")
        start_datetime = start.to_pydatetime()
        if reference_datetime.tzinfo and start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=reference_datetime.tzinfo)
        minutes_until = (start_datetime - reference_datetime).total_seconds() / 60
        if window_end_minutes <= minutes_until <= window_start_minutes:
            rows.append(
                {
                    **row.to_dict(),
                    "minutes_until_start": minutes_until,
                }
            )
    return rows


def dispatcher_decision(
    schedule: pd.DataFrame,
    reference_datetime: datetime,
    window_start_minutes: int,
    window_end_minutes: int,
) -> dict[str, Any]:
    eligible = eligible_pregame_games(
        schedule,
        reference_datetime,
        window_start_minutes,
        window_end_minutes,
    )
    return {
        "reference_datetime": reference_datetime.isoformat(),
        "eligible_games": eligible,
        "eligible_count": len(eligible),
    }
