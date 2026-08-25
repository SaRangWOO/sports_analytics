from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pitcher_game_log_collector import collect_pitcher_game_logs
from pregame_feature_store import apply_feature_store_schema, sync_feature_store, upsert_pitcher_game_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect official KBO pitcher box scores incrementally.")
    parser.add_argument("--games", type=Path, default=PROJECT_DIR / "data" / "official" / "model_training_games.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "data" / "official" / "pitcher_game_logs.csv")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default=(date.today() - timedelta(days=1)).isoformat())
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    games = pd.read_csv(args.games)
    logs, status = collect_pitcher_game_logs(
        games,
        args.output,
        date.fromisoformat(args.start_date),
        date.fromisoformat(args.end_date),
    )
    db_url = os.getenv("DB_URL", "")
    if db_url and not args.no_db:
        schema = PROJECT_DIR / "sql" / "002_feature_store.sql"
        apply_feature_store_schema(db_url, schema)
        status["db_pitcher_rows"] = upsert_pitcher_game_logs(db_url, logs)
        status["db_snapshot_rows"] = sync_feature_store(db_url, PROJECT_DIR / "data" / "official", schema)
    report = PROJECT_DIR / "modeling" / "results" / "pitcher_game_log_collection_status.json"
    report.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
