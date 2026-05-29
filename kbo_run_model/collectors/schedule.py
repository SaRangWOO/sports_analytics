from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def load_schedule(schedule_path: Path) -> pd.DataFrame:
    df = pd.read_csv(schedule_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id", "home_away", "team"]).reset_index(drop=True)


def select_target_games(
    schedule: pd.DataFrame,
    requested_date: date | None,
    today: date,
    allow_past_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    unique_dates = schedule["date"].dt.date.drop_duplicates().sort_values().tolist()
    latest_schedule_date = unique_dates[-1] if unique_dates else None
    future_dates = [value for value in unique_dates if value > today]
    has_today_games = today in unique_dates
    has_future_games = bool(future_dates)
    schedule_is_stale = bool(latest_schedule_date and latest_schedule_date < today and not has_future_games)
    stale_schedule_days = (today - latest_schedule_date).days if schedule_is_stale and latest_schedule_date else 0
    target_date: date | None

    if requested_date is not None:
        target_date = requested_date
        mode = "지정 날짜 기준"
        reason = "사용자가 지정한 과거 경기 기준" if requested_date < today else "사용자가 지정한 날짜 기준"
        user_prediction_available = True
    elif has_today_games:
        target_date = today
        mode = "오늘 경기 기준"
        reason = "오늘 경기 기준"
        user_prediction_available = True
    elif has_future_games:
        target_date = future_dates[0]
        mode = "다음 예정 경기 기준"
        reason = "오늘 이후 가장 가까운 예정 경기 기준"
        user_prediction_available = True
    elif allow_past_fallback and latest_schedule_date is not None:
        target_date = latest_schedule_date
        mode = "과거 경기 기준 리포트"
        reason = "최신 예정 일정이 없어 과거 경기 기준으로 표시 중"
        user_prediction_available = True
    else:
        target_date = None
        mode = "일정 데이터 없음"
        if latest_schedule_date is None:
            reason = "일정 데이터 없음"
        else:
            reason = f"최신 경기 일정 데이터가 {latest_schedule_date.isoformat()}에서 멈춰 있음"
        user_prediction_available = False

    games = schedule[schedule["date"].dt.date.eq(target_date)].copy() if target_date else schedule.iloc[0:0].copy()
    if games.empty:
        user_prediction_available = False
        if requested_date is not None:
            mode = "해당 날짜에 예정된 KBO 경기가 없습니다"
            reason = "사용자가 지정한 날짜에 경기 없음"
    return games, {
        "target_date": target_date.isoformat() if target_date else "",
        "selected_target_date": target_date.isoformat() if target_date else "",
        "report_mode": mode,
        "current_date_kst": today.isoformat(),
        "schedule_latest_date": latest_schedule_date.isoformat() if latest_schedule_date else "",
        "latest_schedule_date": latest_schedule_date.isoformat() if latest_schedule_date else "",
        "schedule_is_stale": schedule_is_stale,
        "stale_schedule_days": int(stale_schedule_days),
        "has_today_games": has_today_games,
        "has_future_schedule": has_future_games,
        "has_future_games": has_future_games,
        "allow_past_fallback": allow_past_fallback,
        "schedule_selection_reason": reason,
        "selection_reason": reason,
        "user_prediction_available": user_prediction_available,
    }


def validate_schedule_selection(target_games: pd.DataFrame, match_predictions: pd.DataFrame, target_context: dict) -> dict:
    if target_games.empty:
        game_count_ok = len(match_predictions) == 0
        return {
            "target_date": target_context["target_date"],
            "selected_target_date": target_context["selected_target_date"],
            "report_mode": target_context["report_mode"],
            "current_date_kst": target_context["current_date_kst"],
            "latest_schedule_date": target_context["latest_schedule_date"],
            "has_today_games": target_context["has_today_games"],
            "has_future_games": target_context["has_future_games"],
            "schedule_is_stale": target_context["schedule_is_stale"],
            "user_prediction_available": target_context["user_prediction_available"],
            "selection_reason": target_context["selection_reason"],
            "schedule_rows": 0,
            "expected_games": 0,
            "predicted_games": int(len(match_predictions)),
            "duplicate_games": 0,
            "home_away_pairing_ok": True,
            "game_count_ok": game_count_ok,
            "status": "통과" if game_count_ok else "확인 필요",
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
        "selected_target_date": target_context["selected_target_date"],
        "report_mode": target_context["report_mode"],
        "current_date_kst": target_context["current_date_kst"],
        "latest_schedule_date": target_context["latest_schedule_date"],
        "has_today_games": target_context["has_today_games"],
        "has_future_games": target_context["has_future_games"],
        "schedule_is_stale": target_context["schedule_is_stale"],
        "user_prediction_available": target_context["user_prediction_available"],
        "selection_reason": target_context["selection_reason"],
        "schedule_rows": int(len(target_games)),
        "expected_games": expected_games,
        "predicted_games": predicted_games,
        "duplicate_games": duplicate_games,
        "home_away_pairing_ok": pairing_ok,
        "game_count_ok": game_count_ok,
        "status": status,
    }
