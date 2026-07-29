from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.predict_only import run_predict_only
from official_kbo_dashboard import (
    DATA_DIR,
    RESULTS_DIR,
    build_dashboard,
    build_team_analysis_pages,
    export_sources,
    fetch_player_stats,
    fetch_registered_rosters,
    fetch_schedule,
    fetch_team_standings,
    load_official_tables_to_db,
)


ARTIFACT_ROOT = PROJECT_DIR / "modeling" / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the KBO dashboard with an approved production model artifact.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--reference-datetime", default="")
    parser.add_argument("--update-stage", choices=["morning", "pregame"], default="morning")
    args = parser.parse_args()
    reference_datetime = (
        datetime.strptime(args.reference_datetime, "%Y-%m-%d %H:%M")
        if args.reference_datetime
        else datetime.now()
    )
    reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    standings, vs_table = fetch_team_standings()
    games = fetch_schedule(reference_date.year, reference_date.month)
    hitters, pitchers = fetch_player_stats()
    rosters = fetch_registered_rosters()
    export_sources(standings, vs_table, games, hitters, pitchers, rosters)
    db_status = load_official_tables_to_db(standings, vs_table, games, hitters, pitchers, rosters)
    model_payload = run_predict_only(games, reference_date, DATA_DIR, RESULTS_DIR, ARTIFACT_ROOT)
    team_pages = build_team_analysis_pages(
        standings,
        vs_table,
        games,
        hitters,
        pitchers,
        rosters,
        reference_date,
    )
    build_dashboard(
        standings,
        vs_table,
        games,
        hitters,
        pitchers,
        model_payload,
        reference_date,
        team_pages,
        reference_datetime,
        args.update_stage,
        db_status,
    )
    print(
        f"[Success] predict-only KBO dashboard generated: "
        f"artifact={model_payload['production_artifact_id']}, current_game_rows={len(games)}"
    )


if __name__ == "__main__":
    main()
