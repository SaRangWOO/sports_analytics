from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.pitching_snapshot_storage import (
    build_pitching_schedule_frame,
    save_pitching_schedule,
)
from official_kbo_dashboard import fetch_kbo_game_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill official start times for canonical pitching snapshots."
    )
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        default=PROJECT_DIR / "data" / "official" / "pitching_daily_snapshot.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "official" / "pitching_snapshot_schedule.csv",
    )
    args = parser.parse_args()

    snapshots = pd.read_csv(args.snapshot_file)
    dates = sorted(pd.to_datetime(snapshots["reference_date"], errors="raise").dt.date.unique())
    frames = []
    for game_date in dates:
        frame = build_pitching_schedule_frame(fetch_kbo_game_list(game_date))
        if frame.empty:
            raise RuntimeError(f"공식 경기 시작 시각을 찾을 수 없습니다: {game_date}")
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True)
    save_pitching_schedule(schedule, args.output)
    print(
        f"[Success] pitching snapshot schedule backfilled: "
        f"dates={len(dates)}, games={len(schedule)}, generated_at={datetime.now().isoformat()}"
    )


if __name__ == "__main__":
    main()
