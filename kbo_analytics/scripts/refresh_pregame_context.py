from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from official_kbo_dashboard import (
    DATA_DIR,
    RESULTS_DIR,
    build_dashboard,
    build_team_analysis_pages,
    fetch_schedule,
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh official pregame starter and lineup context without retraining.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--reference-datetime", default="")
    args = parser.parse_args()
    reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    reference_datetime = (
        datetime.strptime(args.reference_datetime, "%Y-%m-%d %H:%M")
        if args.reference_datetime
        else datetime.now()
    )

    standings = read_csv("team_standings.csv")
    vs_table = read_csv("team_vs_team.csv")
    games = fetch_schedule(reference_date.year, reference_date.month)
    hitters = read_csv("hitter_stats.csv")
    pitchers = read_csv("pitcher_stats.csv")
    rosters = read_csv("registered_rosters.csv")
    model_payload = json.loads((RESULTS_DIR / "win_predictor_model.json").read_text(encoding="utf-8"))
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
        "pregame",
        {"status": "not_run", "warning": "lightweight pregame context refresh"},
    )
    print(f"[Success] pregame context refreshed: reference_date={reference_date.isoformat()}")


if __name__ == "__main__":
    main()
