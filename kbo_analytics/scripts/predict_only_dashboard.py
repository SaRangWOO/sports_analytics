from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.predict_only import run_predict_only
import official_kbo_dashboard as dashboard


ARTIFACT_ROOT = PROJECT_DIR / "modeling" / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the KBO dashboard with an approved production model artifact.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--reference-datetime", default="")
    parser.add_argument("--update-stage", choices=["morning", "pregame"], default="morning")
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT))
    parser.add_argument("--data-dir", default=str(dashboard.DATA_DIR))
    parser.add_argument("--results-dir", default=str(dashboard.RESULTS_DIR))
    parser.add_argument("--dashboard-dir", default=str(dashboard.DASHBOARD_DIR))
    parser.add_argument("--public-dir", default=str(dashboard.PUBLIC_DIR))
    parser.add_argument("--run-model-input", default=str(dashboard.RUN_MODEL_INPUT))
    parser.add_argument("--run-model-schedule-input", default=str(dashboard.RUN_MODEL_SCHEDULE_INPUT))
    parser.add_argument("--run-model-results-dir", default=str(dashboard.RUN_MODEL_RESULTS))
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()
    reference_datetime = (
        datetime.strptime(args.reference_datetime, "%Y-%m-%d %H:%M")
        if args.reference_datetime
        else datetime.now()
    )
    reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    dashboard.DATA_DIR = Path(args.data_dir).resolve()
    dashboard.RESULTS_DIR = Path(args.results_dir).resolve()
    dashboard.DASHBOARD_DIR = Path(args.dashboard_dir).resolve()
    dashboard.PUBLIC_DIR = Path(args.public_dir).resolve()
    dashboard.RUN_MODEL_INPUT = Path(args.run_model_input).resolve()
    dashboard.RUN_MODEL_SCHEDULE_INPUT = Path(args.run_model_schedule_input).resolve()
    dashboard.RUN_MODEL_RESULTS = Path(args.run_model_results_dir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    standings, vs_table = dashboard.fetch_team_standings()
    games = dashboard.fetch_schedule(reference_date.year, reference_date.month)
    hitters, pitchers = dashboard.fetch_player_stats()
    rosters = dashboard.fetch_registered_rosters()
    dashboard.export_sources(standings, vs_table, games, hitters, pitchers, rosters)
    db_status = (
        {"status": "skipped", "reason": "predict-only isolated run"}
        if args.skip_db
        else dashboard.load_official_tables_to_db(
            standings,
            vs_table,
            games,
            hitters,
            pitchers,
            rosters,
        )
    )
    model_payload = run_predict_only(
        games,
        reference_date,
        dashboard.DATA_DIR,
        dashboard.RESULTS_DIR,
        artifact_root,
    )
    team_pages = dashboard.build_team_analysis_pages(
        standings,
        vs_table,
        games,
        hitters,
        pitchers,
        rosters,
        reference_date,
    )
    dashboard.build_dashboard(
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
