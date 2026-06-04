from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


KST = ZoneInfo("Asia/Seoul")
OFFICIAL_SOURCE_URLS = [
    "https://www.koreabaseball.com/Schedule/ScoreBoard.aspx",
    "https://www.koreabaseball.com/Schedule/Schedule.aspx",
    "https://eng.koreabaseball.com/Schedule/DailySchedule.aspx",
]
OUTPUT_COLUMNS = [
    "season",
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_starter_id",
    "away_starter_id",
    "home_starter_name",
    "away_starter_name",
    "source",
    "collected_at",
]
REPORT_COLUMNS = [
    "source",
    "source_url",
    "source_accessible",
    "starter_fields_available",
    "target_games",
    "rows_collected",
    "schedule_match_rate",
    "full_match_rate",
    "partial_match_count",
    "pitcher_id_missing_count",
    "data_ready_to_train",
    "collection_applied",
    "blocker",
]


def _now() -> datetime:
    return datetime.now(KST)


def _base_game_id(value: object) -> str:
    return str(value).rsplit("_", 1)[0]


def _game_schedule(schedule: pd.DataFrame, start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
    frame = schedule.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if start_date:
        frame = frame[frame["date"].ge(start_date)]
    if end_date:
        frame = frame[frame["date"].le(end_date)]
    frame["game_key"] = frame["game_id"].map(_base_game_id)
    grouped = []
    for game_key, group in frame.groupby("game_key", sort=True):
        home = group[group["home_away"].eq("H")]
        away = group[group["home_away"].eq("A")]
        if home.empty or away.empty:
            continue
        grouped.append(
            {
                "season": int(pd.to_datetime(home.iloc[0]["date"]).year),
                "game_id": game_key,
                "date": home.iloc[0]["date"],
                "home_team": home.iloc[0]["team"],
                "away_team": away.iloc[0]["team"],
            }
        )
    return pd.DataFrame(grouped)


def probe_official_sources(timeout: int = 15) -> dict:
    checked = []
    accessible = False
    starter_fields_available = False
    for url in OFFICIAL_SOURCE_URLS:
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            ok = response.status_code == 200
            text = response.text.lower()
            has_starter = any(term in text for term in ["선발투수", "probable pitcher", "starting pitcher", "starter"])
            accessible = accessible or ok
            starter_fields_available = starter_fields_available or (ok and has_starter)
            checked.append(f"{url} status={response.status_code} starter_field={has_starter}")
        except requests.RequestException as exc:
            checked.append(f"{url} error={type(exc).__name__}")
    return {
        "source": "KBO official schedule pages",
        "source_url": " | ".join(OFFICIAL_SOURCE_URLS),
        "source_accessible": accessible,
        "starter_fields_available": starter_fields_available,
        "source_detail": "; ".join(checked),
    }


def _empty_starter_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def collect_from_official(schedule: pd.DataFrame, start_date: date | None = None, end_date: date | None = None, probe_external: bool = True) -> tuple[pd.DataFrame, dict]:
    source_status = probe_official_sources() if probe_external else {
        "source": "KBO official schedule pages",
        "source_url": " | ".join(OFFICIAL_SOURCE_URLS),
        "source_accessible": False,
        "starter_fields_available": False,
        "source_detail": "pipeline summary check does not access external source",
    }
    games = _game_schedule(schedule, start_date, end_date)
    collected = _empty_starter_frame()
    blocker = ""
    if not source_status["source_accessible"]:
        blocker = "공식 일정 원천 접근 실패"
    elif not source_status["starter_fields_available"]:
        blocker = "공식 일정/스코어보드 페이지에서 선발투수 필드 확인 불가"
    return collected, {
        **source_status,
        "target_games": int(len(games)),
        "blocker": blocker,
    }


def validate_starter_collection(collected: pd.DataFrame, schedule: pd.DataFrame, source_status: dict, applied: bool = False) -> tuple[pd.DataFrame, dict]:
    schedule_games = _game_schedule(schedule)
    schedule_keys = set(schedule_games["game_id"].astype(str)) if not schedule_games.empty else set()
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in collected.columns]
    if missing_columns:
        collected = _empty_starter_frame()
    if collected.empty:
        report = {
            "source": source_status["source"],
            "source_url": source_status["source_url"],
            "source_accessible": bool(source_status["source_accessible"]),
            "starter_fields_available": bool(source_status["starter_fields_available"]),
            "target_games": int(source_status.get("target_games", 0)),
            "rows_collected": 0,
            "schedule_match_rate": 0.0,
            "full_match_rate": 0.0,
            "partial_match_count": 0,
            "pitcher_id_missing_count": 0,
            "data_ready_to_train": False,
            "collection_applied": applied,
            "blocker": source_status.get("blocker", "") or "수집된 선발투수 row 없음",
        }
        return collected, report

    frame = collected[OUTPUT_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    matched = frame["game_id"].astype(str).isin(schedule_keys)
    schedule_match_rate = round(float(matched.mean()), 4) if len(frame) else 0.0
    has_home_name = frame["home_starter_name"].fillna("").astype(str).str.strip().ne("")
    has_away_name = frame["away_starter_name"].fillna("").astype(str).str.strip().ne("")
    full = has_home_name & has_away_name
    partial = has_home_name ^ has_away_name
    id_missing = frame[["home_starter_id", "away_starter_id"]].fillna("").astype(str).apply(lambda col: col.str.strip().eq("")).any(axis=1)
    duplicate_games = int(frame["game_id"].duplicated().sum())
    source_ok = frame["source"].fillna("").astype(str).str.strip().ne("").all()
    collected_ok = frame["collected_at"].fillna("").astype(str).str.strip().ne("").all()
    ready = bool(schedule_match_rate > 0 and full.all() and duplicate_games == 0 and not id_missing.any() and source_ok and collected_ok)
    blockers = []
    if schedule_match_rate == 0:
        blockers.append("prediction_games.csv와 game_id 매칭 실패")
    if not full.all():
        blockers.append("홈/원정 선발 이름 누락")
    if id_missing.any():
        blockers.append("pitcher_id 누락")
    if duplicate_games:
        blockers.append("같은 game_id 중복")
    if not source_ok:
        blockers.append("source 누락")
    if not collected_ok:
        blockers.append("collected_at 누락")
    report = {
        "source": source_status["source"],
        "source_url": source_status["source_url"],
        "source_accessible": bool(source_status["source_accessible"]),
        "starter_fields_available": bool(source_status["starter_fields_available"]),
        "target_games": int(source_status.get("target_games", 0)),
        "rows_collected": int(len(frame)),
        "schedule_match_rate": schedule_match_rate,
        "full_match_rate": round(float(full.mean()), 4) if len(frame) else 0.0,
        "partial_match_count": int(partial.sum()),
        "pitcher_id_missing_count": int(id_missing.sum()),
        "data_ready_to_train": ready,
        "collection_applied": applied,
        "blocker": "; ".join(blockers),
    }
    return frame, report


def write_reports(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report])[REPORT_COLUMNS].to_csv(output_dir / "starter_pitcher_collection_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([report])[REPORT_COLUMNS].to_csv(output_dir / "starter_pitcher_validation.csv", index=False, encoding="utf-8-sig")


def collect_and_maybe_apply(
    schedule: pd.DataFrame,
    starter_path: Path,
    output_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    apply: bool = False,
    probe_external: bool = True,
) -> dict:
    collected, source_status = collect_from_official(schedule, start_date, end_date, probe_external)
    validated, report = validate_starter_collection(collected, schedule, source_status, applied=False)
    if apply and report["data_ready_to_train"]:
        backup_dir = starter_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{starter_path.stem}_{_now().strftime('%Y%m%d_%H%M%S')}{starter_path.suffix}"
        if starter_path.exists():
            shutil.copy2(starter_path, backup_path)
        validated.to_csv(starter_path, index=False, encoding="utf-8-sig")
        report["collection_applied"] = True
    elif apply and not report["data_ready_to_train"]:
        report["collection_applied"] = False
        report["blocker"] = report["blocker"] or "검증 실패로 starter_pitchers.csv 유지"
    write_reports(report, output_dir)
    return report


def load_collection_summary(report_path: Path) -> dict:
    if not report_path.exists():
        return {
            "starter_pitcher_collection_completed": False,
            "starter_pitcher_source_available": False,
            "starter_pitcher_rows_collected": 0,
            "starter_pitcher_schedule_match_rate": 0.0,
            "starter_pitcher_full_match_rate": 0.0,
            "starter_pitcher_partial_match_count": 0,
            "starter_pitcher_id_missing_count": 0,
            "starter_pitcher_data_ready_to_train": False,
            "starter_pitcher_collection_blocker": "선발투수 수집 미실행",
        }
    row = pd.read_csv(report_path).iloc[0].to_dict()
    return {
        "starter_pitcher_collection_completed": True,
        "starter_pitcher_source_available": bool(row.get("source_accessible")) and bool(row.get("starter_fields_available")),
        "starter_pitcher_rows_collected": int(row.get("rows_collected", 0)),
        "starter_pitcher_schedule_match_rate": float(row.get("schedule_match_rate", 0.0)),
        "starter_pitcher_full_match_rate": float(row.get("full_match_rate", 0.0)),
        "starter_pitcher_partial_match_count": int(row.get("partial_match_count", 0)),
        "starter_pitcher_id_missing_count": int(row.get("pitcher_id_missing_count", 0)),
        "starter_pitcher_data_ready_to_train": bool(row.get("data_ready_to_train")),
        "starter_pitcher_collection_blocker": str(row.get("blocker", "")),
    }
