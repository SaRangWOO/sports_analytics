from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


def apply_feature_store_schema(db_url: str, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    engine = create_engine(db_url)
    with engine.begin() as connection:
        connection.exec_driver_sql(sql)


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def upsert_pitcher_game_logs(db_url: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    columns = list(frame.columns)
    assignments = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in columns
        if column not in {"game_id", "team", "pitcher_index"}
    )
    statement = text(
        f"""
        INSERT INTO pitcher_game_logs ({', '.join(columns)})
        VALUES ({', '.join(':' + column for column in columns)})
        ON CONFLICT (game_id, team, pitcher_index)
        DO UPDATE SET {assignments}
        """
    )
    engine = create_engine(db_url)
    with engine.begin() as connection:
        connection.execute(statement, _records(frame))
    return len(frame)


def _snapshot_frame(path: Path, table: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rename = {"scheduled_game_id": "game_id", "player": "player_name"}
    frame = frame.rename(columns=rename)
    if table == "pregame_pitching_snapshots":
        columns = [
            "snapshot_time", "reference_date", "game_id", "team", "opponent", "home_away",
            "starter_name", "starter_source", "starter_info_quality", "starter_era", "starter_whip",
            "bullpen_fatigue_label", "recent_3day_games", "data_source", "note",
        ]
    else:
        columns = [
            "snapshot_time", "reference_date", "game_id", "team", "home_away", "lineup_source",
            "lineup_info_quality", "batting_order", "position", "player_name", "war", "data_source",
        ]
    return frame.reindex(columns=columns)


def _upsert_snapshot(db_url: str, table: str, frame: pd.DataFrame, conflict_columns: list[str]) -> int:
    if frame.empty:
        return 0
    columns = list(frame.columns)
    assignments = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in columns if column not in conflict_columns
    )
    statement = text(
        f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(':' + column for column in columns)})
        ON CONFLICT ({', '.join(conflict_columns)})
        DO UPDATE SET {assignments}
        """
    )
    engine = create_engine(db_url)
    with engine.begin() as connection:
        connection.execute(statement, _records(frame))
    return len(frame)


def sync_feature_store(db_url: str, data_dir: Path, schema_path: Path) -> dict:
    apply_feature_store_schema(db_url, schema_path)
    pitching = _snapshot_frame(data_dir / "pitching_daily_snapshot.csv", "pregame_pitching_snapshots")
    lineup = _snapshot_frame(data_dir / "lineup_daily_snapshot.csv", "pregame_lineup_snapshots")
    return {
        "pitching_snapshot_rows": _upsert_snapshot(
            db_url,
            "pregame_pitching_snapshots",
            pitching,
            ["reference_date", "game_id", "team", "snapshot_time"],
        ),
        "lineup_snapshot_rows": _upsert_snapshot(
            db_url,
            "pregame_lineup_snapshots",
            lineup,
            ["reference_date", "game_id", "team", "snapshot_time", "batting_order"],
        ),
    }
