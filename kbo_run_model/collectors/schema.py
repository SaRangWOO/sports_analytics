from __future__ import annotations

from pathlib import Path

import pandas as pd


STARTER_CANDIDATE_COLUMNS = [
    "home_starter",
    "away_starter",
    "home_starting_pitcher",
    "away_starting_pitcher",
    "home_pitcher",
    "away_pitcher",
    "starter",
    "pitcher",
    "pitcher_id",
    "player_id",
    "era",
    "whip",
    "innings_pitched",
    "earned_runs",
    "strikeouts",
    "walks",
]

REQUIRED_GAME_STARTER_COLUMNS = [
    "season",
    "date",
    "game_id",
    "home_team",
    "away_team",
    "home_starter_id 또는 home_starter_name",
    "away_starter_id 또는 away_starter_name",
]

REQUIRED_PITCHER_LOG_COLUMNS = [
    "season",
    "date",
    "game_id",
    "pitcher_id 또는 pitcher_name",
    "team",
    "opponent",
    "innings_pitched",
    "earned_runs",
    "hits_allowed",
    "walks",
    "strikeouts",
    "home_runs_allowed",
    "pitches",
]

FUTURE_STARTER_FEATURES = [
    "starter_era",
    "starter_whip",
    "starter_recent_3g_era",
    "starter_rest_days",
    "starter_avg_ip",
]


def inspect_starter_schema(input_path: Path) -> dict:
    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    matched = [column for column in STARTER_CANDIDATE_COLUMNS if column in columns]
    return {
        "has_starter_source_data": bool(matched),
        "matched_columns": matched,
        "available_columns": columns,
        "candidate_columns": STARTER_CANDIDATE_COLUMNS,
        "required_game_starter_columns": REQUIRED_GAME_STARTER_COLUMNS,
        "required_pitcher_log_columns": REQUIRED_PITCHER_LOG_COLUMNS,
        "future_starter_features": FUTURE_STARTER_FEATURES,
        "status": "선발투수 원천 데이터 확인됨" if matched else "선발투수 원천 데이터 없음",
    }
