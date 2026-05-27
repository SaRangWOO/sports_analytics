from __future__ import annotations

from pathlib import Path

import pandas as pd


STARTER_COLUMNS = [
    "season",
    "date",
    "game_id",
    "home_team",
    "away_team",
    "home_starter_name",
    "away_starter_name",
    "home_starter_id",
    "away_starter_id",
]

PITCHER_LOG_COLUMNS = [
    "season",
    "date",
    "game_id",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "is_starter",
    "innings_pitched",
    "earned_runs",
    "hits_allowed",
    "walks",
    "strikeouts",
    "home_runs_allowed",
    "pitches",
]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _load_csv(path: Path, required_columns: list[str], label: str) -> tuple[pd.DataFrame, dict]:
    status = {
        "file": str(path),
        "label": label,
        "available": False,
        "phase": "Data collection phase",
        "v2_training_status": "V2 not trained",
        "message": "starter data unavailable",
        "missing_columns": [],
        "row_count": 0,
    }
    if not path.exists() or path.stat().st_size == 0:
        status["message"] = f"{label} file unavailable"
        status["missing_columns"] = required_columns
        return _empty_frame(required_columns), status

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        status["message"] = f"{label} missing required columns"
        status["missing_columns"] = missing
        return _empty_frame(required_columns), status

    df = df[required_columns].copy()
    status["row_count"] = int(len(df))
    if df.empty:
        status["message"] = f"{label} schema ready but no rows"
        return df, status

    joined = " ".join(df.astype(str).head(5).to_numpy().ravel()).upper()
    if "SAMPLE_" in joined or "MOCK_" in joined:
        status["message"] = f"{label} contains sample/mock rows only"
        status["sample_only"] = True
        return _empty_frame(required_columns), status

    status["available"] = True
    status["sample_only"] = False
    status["message"] = f"{label} available"
    return df, status


def load_starter_pitchers(path: Path) -> tuple[pd.DataFrame, dict]:
    return _load_csv(path, STARTER_COLUMNS, "starter_pitchers")


def load_pitcher_game_logs(path: Path) -> tuple[pd.DataFrame, dict]:
    return _load_csv(path, PITCHER_LOG_COLUMNS, "pitcher_game_logs")


def load_starter_inputs(starter_path: Path, pitcher_log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    starters, starter_status = load_starter_pitchers(starter_path)
    pitcher_logs, pitcher_log_status = load_pitcher_game_logs(pitcher_log_path)
    available = bool(starter_status["available"] and pitcher_log_status["available"])
    status = {
        "starter_data_available": bool(starter_status["available"]),
        "pitcher_logs_available": bool(pitcher_log_status["available"]),
        "starter_schema_ready": not starter_status["missing_columns"],
        "pitcher_log_schema_ready": not pitcher_log_status["missing_columns"],
        "v3_bullpen_schema_reusable": True,
        "v2_status": "ready_for_feature_generation" if available else "data_collection_phase",
        "status_label": "Starter data available" if available else "Starter data unavailable (Data collection phase)",
        "training_status": "V2 ready" if available else "V2 not trained",
        "reason": "" if available else "선발투수/투수 등판 기록 데이터 미수집",
        "starter_pitchers": starter_status,
        "pitcher_game_logs": pitcher_log_status,
    }
    return starters, pitcher_logs, status
