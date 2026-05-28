from __future__ import annotations

from pathlib import Path

import pandas as pd

from collectors.load_starter_data import PITCHER_LOG_COLUMNS, STARTER_COLUMNS
from features.starter_features import ip_to_outs


def _base_game_id(value: object) -> str:
    return str(value).rsplit("_", 1)[0]


def _empty_result(path: Path, required_columns: list[str], label: str) -> tuple[pd.DataFrame, dict]:
    return pd.DataFrame(columns=required_columns), {
        "label": label,
        "file": str(path),
        "file_exists": path.exists(),
        "schema_valid": False,
        "row_count": 0,
        "missing_columns": required_columns,
        "duplicate_rows": 0,
        "game_match_rate": 0.0,
        "valid": False,
        "message": f"{label} 데이터 미수집",
    }


def _read_csv(path: Path, required_columns: list[str], label: str) -> tuple[pd.DataFrame, dict]:
    if not path.exists() or path.stat().st_size == 0:
        return _empty_result(path, required_columns, label)
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    status = {
        "label": label,
        "file": str(path),
        "file_exists": True,
        "schema_valid": not missing,
        "row_count": int(len(frame)),
        "missing_columns": missing,
        "duplicate_rows": 0,
        "game_match_rate": 0.0,
        "valid": False,
        "message": "",
    }
    if missing:
        status["message"] = f"{label} 필수 컬럼 누락"
        return pd.DataFrame(columns=required_columns), status
    if frame.empty:
        status["message"] = f"{label} 스키마 준비, 실제 데이터 없음"
        return frame[required_columns].copy(), status
    return frame[required_columns].copy(), status


def _schedule_game_keys(schedule: pd.DataFrame) -> set[str]:
    if schedule.empty or "game_id" not in schedule.columns:
        return set()
    return set(schedule["game_id"].map(_base_game_id).astype(str))


def _match_rate(game_ids: pd.Series, schedule_keys: set[str]) -> float:
    if game_ids.empty or not schedule_keys:
        return 0.0
    keys = set(game_ids.map(_base_game_id).astype(str))
    return round(len(keys & schedule_keys) / len(keys), 4) if keys else 0.0


def validate_starter_pitchers(path: Path, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame, status = _read_csv(path, STARTER_COLUMNS, "starter_pitchers.csv")
    if frame.empty or not status["schema_valid"]:
        return frame, status

    schedule_keys = _schedule_game_keys(schedule)
    duplicate_rows = int(frame.duplicated(subset=["season", "date", "game_id"]).sum())
    missing_ids = int(frame[["home_starter_id", "away_starter_id"]].isna().any(axis=1).sum())
    match_rate = _match_rate(frame["game_id"], schedule_keys)
    valid = duplicate_rows == 0 and missing_ids == 0 and match_rate > 0
    status.update(
        {
            "duplicate_rows": duplicate_rows,
            "missing_pitcher_id_rows": missing_ids,
            "game_match_rate": match_rate,
            "valid": valid,
            "message": "선발투수 매핑 학습 가능" if valid else "선발투수 매핑 검증 실패",
        }
    )
    return frame, status


def _valid_innings(value: object) -> bool:
    try:
        ip_to_outs(value)
    except ValueError:
        return False
    return True


def validate_pitcher_game_logs(path: Path, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame, status = _read_csv(path, PITCHER_LOG_COLUMNS, "pitcher_game_logs.csv")
    if frame.empty or not status["schema_valid"]:
        return frame, status

    schedule_keys = _schedule_game_keys(schedule)
    duplicate_rows = int(frame.duplicated(subset=["season", "date", "game_id", "pitcher_id", "team"]).sum())
    missing_pitcher_ids = int(frame["pitcher_id"].isna().sum())
    valid_is_starter = frame["is_starter"].astype(str).str.lower().isin(["true", "false", "1", "0", "yes", "no", "y", "n"]).all()
    valid_innings = frame["innings_pitched"].map(_valid_innings).all()
    match_rate = _match_rate(frame["game_id"], schedule_keys)
    valid = duplicate_rows == 0 and missing_pitcher_ids == 0 and bool(valid_is_starter) and bool(valid_innings) and match_rate > 0
    status.update(
        {
            "duplicate_rows": duplicate_rows,
            "missing_pitcher_id_rows": missing_pitcher_ids,
            "invalid_is_starter_rows": 0 if valid_is_starter else int((~frame["is_starter"].astype(str).str.lower().isin(["true", "false", "1", "0", "yes", "no", "y", "n"])).sum()),
            "invalid_innings_rows": 0 if valid_innings else int((~frame["innings_pitched"].map(_valid_innings)).sum()),
            "game_match_rate": match_rate,
            "valid": valid,
            "message": "투수 등판 로그 학습 가능" if valid else "투수 등판 로그 검증 실패",
        }
    )
    return frame, status


def validate_pitcher_data_pipeline(starter_path: Path, pitcher_log_path: Path, schedule: pd.DataFrame) -> dict:
    _, starter_status = validate_starter_pitchers(starter_path, schedule)
    _, log_status = validate_pitcher_game_logs(pitcher_log_path, schedule)
    ready = bool(starter_status["valid"] and log_status["valid"])
    if ready:
        blocker = ""
    elif not starter_status["file_exists"] or not log_status["file_exists"]:
        blocker = "실제 투수 데이터 파일 없음"
    elif starter_status["row_count"] == 0 or log_status["row_count"] == 0:
        blocker = "실제 투수 데이터 미수집"
    elif not starter_status["schema_valid"] or not log_status["schema_valid"]:
        blocker = "필수 컬럼 누락"
    else:
        blocker = "경기 매칭률, 중복, pitcher_id, innings_pitched 검증 실패"
    return {
        "pitcher_data_validation_completed": True,
        "pitcher_data_pipeline_ready": True,
        "starter_pitchers_file_exists": bool(starter_status["file_exists"]),
        "pitcher_game_logs_file_exists": bool(log_status["file_exists"]),
        "starter_pitchers_schema_valid": bool(starter_status["schema_valid"]),
        "pitcher_game_logs_schema_valid": bool(log_status["schema_valid"]),
        "starter_schedule_match_rate": float(starter_status["game_match_rate"]),
        "pitcher_logs_game_match_rate": float(log_status["game_match_rate"]),
        "pitcher_data_ready_to_train": ready,
        "pitcher_data_blocker": blocker,
        "starter_pitchers_validation": starter_status,
        "pitcher_game_logs_validation": log_status,
    }
