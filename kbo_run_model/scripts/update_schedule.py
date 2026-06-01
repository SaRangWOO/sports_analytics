from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from collectors.schedule_update import build_schedule_status, update_schedule_file, write_schedule_update_report


DEFAULT_SCHEDULE = REPO_DIR / "kbo_analytics" / "data" / "official" / "prediction_games.csv"
DEFAULT_REPORT = PROJECT_DIR / "results" / "schedule_update_report.csv"
DEFAULT_BACKUP_DIR = PROJECT_DIR / "data" / "backups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check or update the KBO prediction schedule file.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--source-csv", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        report = build_schedule_status(args.schedule)
        report["update_attempted"] = False
        report["update_applied"] = False
    else:
        if args.source_csv is None:
            report = build_schedule_status(args.schedule)
            report["update_attempted"] = True
            report["update_applied"] = False
            report["schedule_update_blocker"] = "실제 일정 수집 원천 또는 --source-csv 없음"
        else:
            report = update_schedule_file(args.source_csv, args.schedule, args.backup_dir)

    write_schedule_update_report(report, args.report_path)
    for key in [
        "current_date_kst",
        "schedule_min_date",
        "schedule_max_date",
        "total_rows",
        "total_games",
        "future_games",
        "today_games",
        "schedule_is_stale",
        "stale_schedule_days",
        "schedule_update_needed",
        "schedule_update_blocker",
        "update_attempted",
        "update_applied",
    ]:
        print(f"{key}={report.get(key, '')}")
    print(f"report_path={args.report_path}")


if __name__ == "__main__":
    main()
