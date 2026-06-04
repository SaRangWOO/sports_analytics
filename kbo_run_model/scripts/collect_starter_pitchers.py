from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from collectors.schedule import load_schedule
from collectors.starter_pitcher_collector import collect_and_maybe_apply


DEFAULT_SCHEDULE = REPO_DIR / "kbo_analytics" / "data" / "official" / "prediction_games.csv"
DEFAULT_STARTERS = PROJECT_DIR / "data" / "starter_pitchers.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect KBO starter pitcher mappings.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--date", type=_parse_date, default=None)
    parser.add_argument("--start-date", type=_parse_date, default=None)
    parser.add_argument("--end-date", type=_parse_date, default=None)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--starter-path", type=Path, default=DEFAULT_STARTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = args.date or args.start_date
    end_date = args.date or args.end_date
    schedule = load_schedule(args.schedule)
    report = collect_and_maybe_apply(
        schedule,
        args.starter_path,
        args.output_dir,
        start_date=start_date,
        end_date=end_date,
        apply=args.apply,
        probe_external=True,
    )
    print(f"starter_pitcher_collection_completed=True")
    print(f"starter_pitcher_source_available={report['source_accessible'] and report['starter_fields_available']}")
    print(f"starter_pitcher_rows_collected={report['rows_collected']}")
    print(f"starter_pitcher_schedule_match_rate={report['schedule_match_rate']}")
    print(f"starter_pitcher_full_match_rate={report['full_match_rate']}")
    print(f"starter_pitcher_partial_match_count={report['partial_match_count']}")
    print(f"starter_pitcher_id_missing_count={report['pitcher_id_missing_count']}")
    print(f"starter_pitcher_data_ready_to_train={report['data_ready_to_train']}")
    print(f"starter_pitcher_collection_applied={report['collection_applied']}")
    print(f"starter_pitcher_collection_blocker={report['blocker']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
