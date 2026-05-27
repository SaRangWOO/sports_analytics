from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "game_id",
    "date",
    "team",
    "opponent",
    "home_away",
    "status",
    "score_team",
    "score_opp",
}


def base_game_id(game_id: str) -> str:
    return str(game_id).rsplit("_", 1)[0]


def load_completed_team_games(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["status"].eq("Final")].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year
    df["game_key"] = df["game_id"].map(base_game_id)
    df["score_team"] = pd.to_numeric(df["score_team"], errors="coerce")
    df["score_opp"] = pd.to_numeric(df["score_opp"], errors="coerce")
    df = df.dropna(subset=["score_team", "score_opp"])
    return df.sort_values(["season", "date", "game_key", "team"]).reset_index(drop=True)
