from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def load_schedule(schedule_path: Path) -> pd.DataFrame:
    df = pd.read_csv(schedule_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id", "home_away", "team"]).reset_index(drop=True)


def select_target_games(schedule: pd.DataFrame, requested_date: date | None, today: date) -> tuple[pd.DataFrame, dict]:
    unique_dates = schedule["date"].dt.date.drop_duplicates().sort_values().tolist()
    if requested_date is not None:
        target_date = requested_date
        mode = "지정 날짜 기준"
    elif today in unique_dates:
        target_date = today
        mode = "오늘 경기 기준"
    else:
        future_dates = [value for value in unique_dates if value > today]
        if future_dates:
            target_date = future_dates[0]
            mode = "다음 예정 경기 기준"
        elif unique_dates:
            target_date = unique_dates[-1]
            mode = "과거 경기 기준 리포트"
        else:
            target_date = today
            mode = "일정 데이터 없음"

    games = schedule[schedule["date"].dt.date.eq(target_date)].copy()
    if games.empty:
        mode = "해당 날짜에 예정된 KBO 경기가 없습니다"
    return games, {"target_date": target_date.isoformat(), "report_mode": mode}


def validate_schedule_selection(target_games: pd.DataFrame, match_predictions: pd.DataFrame, target_context: dict) -> dict:
    if target_games.empty:
        return {
            "target_date": target_context["target_date"],
            "report_mode": target_context["report_mode"],
            "schedule_rows": 0,
            "expected_games": 0,
            "predicted_games": int(len(match_predictions)),
            "duplicate_games": 0,
            "home_away_pairing_ok": True,
            "game_count_ok": len(match_predictions) == 0,
            "status": "해당 날짜 경기 없음",
        }

    checked_games = target_games.copy()
    checked_games["game_key"] = checked_games["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    grouped = checked_games.groupby("game_key")
    duplicate_games = int(match_predictions["game_id"].duplicated().sum()) if "game_id" in match_predictions.columns else 0
    pairing_ok = bool(
        grouped.size().eq(2).all()
        and grouped["home_away"].apply(lambda values: set(values) == {"A", "H"}).all()
    )
    expected_games = int(checked_games["game_key"].nunique())
    predicted_games = int(len(match_predictions))
    game_count_ok = expected_games == predicted_games
    status = "통과" if pairing_ok and game_count_ok and duplicate_games == 0 else "확인 필요"
    return {
        "target_date": target_context["target_date"],
        "report_mode": target_context["report_mode"],
        "schedule_rows": int(len(target_games)),
        "expected_games": expected_games,
        "predicted_games": predicted_games,
        "duplicate_games": duplicate_games,
        "home_away_pairing_ok": pairing_ok,
        "game_count_ok": game_count_ok,
        "status": status,
    }
