from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
REQUIRED_COLUMNS = ["game_id", "date", "team", "opponent", "home_away", "status", "ballpark"]


def current_kst_date() -> date:
    return datetime.now(KST).date()


def _empty_report(schedule_path: Path, current_date) -> dict:
    return {
        "schedule_path": str(schedule_path),
        "current_date_kst": current_date.isoformat(),
        "schedule_min_date": "",
        "schedule_max_date": "",
        "total_rows": 0,
        "total_games": 0,
        "future_games": 0,
        "today_games": 0,
        "schedule_is_stale": True,
        "stale_schedule_days": 0,
        "required_columns_present": False,
        "date_parse_ok": False,
        "team_values_ok": False,
        "home_away_values_ok": False,
        "home_away_pairing_ok": False,
        "same_game_duplicate_count": 0,
        "row_duplicate_count": 0,
        "latest_date_not_older_than_before": True,
        "schedule_update_needed": True,
        "schedule_update_check_completed": True,
        "schedule_update_blocker": "일정 파일 없음",
    }


def read_schedule_csv(schedule_path: Path) -> pd.DataFrame:
    return pd.read_csv(schedule_path)


def build_schedule_status(schedule_path: Path, current_date=None, previous_max_date=None) -> dict:
    current_date = current_date or current_kst_date()
    if not schedule_path.exists():
        return _empty_report(schedule_path, current_date)

    df = read_schedule_csv(schedule_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    report = {
        "schedule_path": str(schedule_path),
        "current_date_kst": current_date.isoformat(),
        "total_rows": int(len(df)),
        "required_columns_present": not missing_columns,
        "missing_columns": ", ".join(missing_columns),
        "schedule_update_check_completed": True,
    }
    if missing_columns:
        report.update(
            {
                "schedule_min_date": "",
                "schedule_max_date": "",
                "total_games": 0,
                "future_games": 0,
                "today_games": 0,
                "schedule_is_stale": True,
                "stale_schedule_days": 0,
                "date_parse_ok": False,
                "team_values_ok": False,
                "home_away_values_ok": False,
                "home_away_pairing_ok": False,
                "same_game_duplicate_count": 0,
                "row_duplicate_count": 0,
                "latest_date_not_older_than_before": False,
                "schedule_update_needed": True,
                "schedule_update_blocker": f"필수 컬럼 누락: {', '.join(missing_columns)}",
            }
        )
        return report

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    date_parse_ok = bool(parsed_dates.notna().all())
    dated = df.copy()
    dated["parsed_date"] = parsed_dates.dt.date
    dated["game_key"] = dated["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    valid_dates = dated["parsed_date"].dropna()
    min_date = valid_dates.min() if not valid_dates.empty else None
    max_date = valid_dates.max() if not valid_dates.empty else None
    games = dated.dropna(subset=["parsed_date"]).drop_duplicates(["game_key", "parsed_date"])
    today_games = int(games[games["parsed_date"].eq(current_date)]["game_key"].nunique())
    future_games = int(games[games["parsed_date"].gt(current_date)]["game_key"].nunique())
    total_games = int(games["game_key"].nunique())
    schedule_is_stale = bool(max_date and max_date < current_date and future_games == 0)
    stale_schedule_days = int((current_date - max_date).days) if schedule_is_stale and max_date else 0
    grouped = dated.groupby("game_key", dropna=False)
    pairing_ok = bool(
        not dated.empty
        and grouped.size().eq(2).all()
        and grouped["home_away"].apply(lambda values: set(values) == {"A", "H"}).all()
    )
    duplicate_games = int(dated.duplicated(["game_key", "team", "home_away"]).sum())
    row_duplicates = int(dated.duplicated().sum())
    team_values_ok = bool(dated["team"].notna().all() and dated["opponent"].notna().all() and dated["team"].astype(str).str.strip().ne("").all())
    home_away_values_ok = bool(dated["home_away"].isin(["A", "H"]).all())
    latest_ok = True
    if previous_max_date and max_date:
        latest_ok = max_date >= previous_max_date
    blocker = ""
    if not date_parse_ok:
        blocker = "date 파싱 실패"
    elif not team_values_ok:
        blocker = "팀명 결측 또는 공백"
    elif not home_away_values_ok:
        blocker = "home_away 값 오류"
    elif not pairing_ok:
        blocker = "home/away pairing 오류"
    elif duplicate_games or row_duplicates:
        blocker = "중복 경기 행 존재"
    elif not latest_ok:
        blocker = "새 일정 최신 날짜가 기존보다 오래됨"
    elif schedule_is_stale:
        blocker = "오늘 이후 일정 없음"

    report.update(
        {
            "schedule_min_date": min_date.isoformat() if min_date else "",
            "schedule_max_date": max_date.isoformat() if max_date else "",
            "total_games": total_games,
            "future_games": future_games,
            "today_games": today_games,
            "schedule_is_stale": schedule_is_stale,
            "stale_schedule_days": stale_schedule_days,
            "date_parse_ok": date_parse_ok,
            "team_values_ok": team_values_ok,
            "home_away_values_ok": home_away_values_ok,
            "home_away_pairing_ok": pairing_ok,
            "same_game_duplicate_count": duplicate_games,
            "row_duplicate_count": row_duplicates,
            "latest_date_not_older_than_before": latest_ok,
            "schedule_update_needed": schedule_is_stale,
            "schedule_update_blocker": blocker,
        }
    )
    return report


def write_schedule_update_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(output_path, index=False, encoding="utf-8-sig")


def validate_schedule_for_update(source_path: Path, current_path: Path) -> tuple[bool, dict]:
    current_status = build_schedule_status(current_path)
    previous_max_date = None
    if current_status.get("schedule_max_date"):
        previous_max_date = datetime.strptime(str(current_status["schedule_max_date"]), "%Y-%m-%d").date()
    source_status = build_schedule_status(source_path, previous_max_date=previous_max_date)
    valid = bool(
        source_status["required_columns_present"]
        and source_status["date_parse_ok"]
        and source_status["team_values_ok"]
        and source_status["home_away_values_ok"]
        and source_status["home_away_pairing_ok"]
        and source_status["same_game_duplicate_count"] == 0
        and source_status["row_duplicate_count"] == 0
        and source_status["latest_date_not_older_than_before"]
    )
    return valid, source_status


def update_schedule_file(source_path: Path, target_path: Path, backup_dir: Path) -> dict:
    if not source_path.exists():
        report = build_schedule_status(target_path)
        report["update_attempted"] = True
        report["update_applied"] = False
        report["schedule_update_blocker"] = "실제 일정 수집 원천 또는 source CSV 없음"
        return report

    valid, report = validate_schedule_for_update(source_path, target_path)
    report["update_attempted"] = True
    report["update_applied"] = False
    if not valid:
        report["schedule_update_blocker"] = report["schedule_update_blocker"] or "새 일정 검증 실패"
        return report

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{target_path.stem}_{timestamp}{target_path.suffix}"
    shutil.copy2(target_path, backup_path)
    shutil.copy2(source_path, target_path)
    report["update_applied"] = True
    report["backup_path"] = str(backup_path)
    report["schedule_update_blocker"] = ""
    return report
