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
from collectors.starter_source_research import write_source_research_outputs


DEFAULT_SCHEDULE = REPO_DIR / "kbo_analytics" / "data" / "official" / "prediction_games.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"
SOURCES = ["all", "kbo", "naver", "daum", "statiz", "manual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research starter pitcher data sources.")
    parser.add_argument("--date", required=True, type=lambda value: datetime.strptime(value, "%Y-%m-%d").date())
    parser.add_argument("--source", choices=SOURCES, default="all")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = load_schedule(args.schedule)
    summary = write_source_research_outputs(schedule, args.date, args.output_dir, args.source)
    print(f"starter_pitcher_source_research_completed={summary['starter_pitcher_source_research_completed']}")
    print(f"starter_pitcher_sources_checked={summary['starter_pitcher_sources_checked']}")
    print(f"viable_starter_pitcher_sources={summary['viable_starter_pitcher_sources']}")
    print(f"recommended_starter_pitcher_source={summary['recommended_starter_pitcher_source']}")
    print(f"recommended_starter_pitcher_source_rank={summary['recommended_starter_pitcher_source_rank']}")
    print(f"starter_pitcher_source_blocker={summary['starter_pitcher_source_blocker']}")
    print(f"next_recommended_starter_collection_step={summary['next_recommended_starter_collection_step']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
