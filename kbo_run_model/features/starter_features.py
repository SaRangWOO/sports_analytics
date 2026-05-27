from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


DEFAULT_STARTER_ERA = 4.50
DEFAULT_STARTER_WHIP = 1.35
DEFAULT_STARTER_IP = 5.0
DEFAULT_STARTER_REST_DAYS = 5


def ip_to_outs(value: object) -> int:
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return 0
        decimal = Decimal(text)
    except (InvalidOperation, ValueError):
        return 0

    whole = int(decimal)
    fraction = int((decimal - whole) * 10)
    if fraction not in {0, 1, 2}:
        raise ValueError(f"Invalid baseball innings value: {value}")
    return whole * 3 + fraction


def outs_to_decimal_ip(outs: int | float) -> float:
    return round(float(outs) / 3.0, 4)


def normalize_innings_pitched(value: object) -> float:
    return outs_to_decimal_ip(ip_to_outs(value))


def _base_game_id(game_id: object) -> str:
    return str(game_id).rsplit("_", 1)[0]


def _league_baselines(logs: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    starter_logs = logs[logs["is_starter"].astype(str).str.lower().isin(["true", "1", "yes", "y"])].copy()
    if starter_logs.empty:
        return {}, {}
    grouped = starter_logs.groupby("season")
    current = {}
    previous = {}
    for season, group in grouped:
        outs = group["outs"].sum()
        innings = outs_to_decimal_ip(outs)
        current[int(season)] = {
            "starter_era": float(group["earned_runs"].sum() * 9 / innings) if innings else DEFAULT_STARTER_ERA,
            "starter_whip": float((group["hits_allowed"].sum() + group["walks"].sum()) / innings) if innings else DEFAULT_STARTER_WHIP,
            "starter_avg_ip": float(group["outs"].mean() / 3) if len(group) else DEFAULT_STARTER_IP,
        }
    for season in current:
        if season - 1 in current:
            previous[season] = current[season - 1]
    return current, previous


def _baseline(season: int, current: dict[int, dict[str, float]], previous: dict[int, dict[str, float]], key: str, default: float) -> float:
    if season in current:
        return current[season][key]
    if season in previous:
        return previous[season][key]
    return default


def build_pitcher_pre_game_features(pitcher_logs: pd.DataFrame) -> pd.DataFrame:
    logs = pitcher_logs.copy()
    if logs.empty:
        return pd.DataFrame()

    logs["date"] = pd.to_datetime(logs["date"])
    logs["outs"] = logs["innings_pitched"].map(ip_to_outs)
    logs["season"] = logs["season"].astype(int)
    logs["pitcher_id"] = logs["pitcher_id"].astype(str)
    logs["game_key"] = logs["game_id"].map(_base_game_id)
    logs = logs.sort_values(["season", "pitcher_id", "date", "game_id"]).reset_index(drop=True)
    league_current, league_previous = _league_baselines(logs)
    frames: list[pd.DataFrame] = []

    for (season, pitcher_id), group in logs.groupby(["season", "pitcher_id"], sort=False):
        previous_logs: list[dict] = []
        previous_date: pd.Timestamp | None = None
        records: list[dict[str, float | int | str]] = []

        for current_date, date_group in group.groupby("date", sort=True):
            if previous_logs:
                history = pd.DataFrame(previous_logs)
                outs = history["outs"].sum()
                innings = outs_to_decimal_ip(outs)
                era = float(history["earned_runs"].sum() * 9 / innings) if innings else DEFAULT_STARTER_ERA
                whip = float((history["hits_allowed"].sum() + history["walks"].sum()) / innings) if innings else DEFAULT_STARTER_WHIP
                recent = history.tail(3)
                recent_ip = outs_to_decimal_ip(recent["outs"].sum())
                recent_era = float(recent["earned_runs"].sum() * 9 / recent_ip) if recent_ip else era
                avg_ip = float(history["outs"].mean() / 3)
            else:
                era = _baseline(int(season), league_current, league_previous, "starter_era", DEFAULT_STARTER_ERA)
                whip = _baseline(int(season), league_current, league_previous, "starter_whip", DEFAULT_STARTER_WHIP)
                recent_era = era
                avg_ip = _baseline(int(season), league_current, league_previous, "starter_avg_ip", DEFAULT_STARTER_IP)

            rest_days = DEFAULT_STARTER_REST_DAYS if previous_date is None else max(1, min((current_date - previous_date).days, 30))
            for _, row in date_group.iterrows():
                records.append(
                    {
                        "season": int(season),
                        "date": current_date,
                        "game_key": row["game_key"],
                        "pitcher_id": pitcher_id,
                        "starter_era": round(era, 4),
                        "starter_whip": round(whip, 4),
                        "starter_recent_3g_era": round(recent_era, 4),
                        "starter_rest_days": int(rest_days),
                        "starter_avg_ip": round(avg_ip, 4),
                    }
                )

            previous_logs.extend(date_group.to_dict("records"))
            previous_date = current_date

        frames.append(pd.DataFrame(records))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_game_starter_rows(starters: pd.DataFrame) -> pd.DataFrame:
    if starters.empty:
        return pd.DataFrame()
    home = starters[
        ["season", "date", "game_id", "home_team", "away_team", "home_starter_name", "home_starter_id"]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_starter_name": "starter_name",
            "home_starter_id": "pitcher_id",
        }
    )
    home["is_home"] = 1
    away = starters[
        ["season", "date", "game_id", "away_team", "home_team", "away_starter_name", "away_starter_id"]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_starter_name": "starter_name",
            "away_starter_id": "pitcher_id",
        }
    )
    away["is_home"] = 0
    rows = pd.concat([home, away], ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["season"] = rows["season"].astype(int)
    rows["pitcher_id"] = rows["pitcher_id"].astype(str)
    rows["game_key"] = rows["game_id"].map(_base_game_id)
    return rows


def add_starter_features(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    starters: pd.DataFrame,
    pitcher_logs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict]:
    status = {
        "starter_features_added": False,
        "message": "Starter data unavailable (Data collection phase)",
        "feature_columns": [],
    }
    if starters.empty or pitcher_logs.empty:
        return feature_df, feature_columns, status

    starter_rows = build_game_starter_rows(starters)
    pitcher_features = build_pitcher_pre_game_features(pitcher_logs)
    if starter_rows.empty or pitcher_features.empty:
        return feature_df, feature_columns, status

    starter_feature_rows = starter_rows.merge(
        pitcher_features,
        on=["season", "date", "game_key", "pitcher_id"],
        how="left",
    )
    starter_feature_columns = [
        "starter_era",
        "starter_whip",
        "starter_recent_3g_era",
        "starter_rest_days",
        "starter_avg_ip",
    ]
    merged = feature_df.merge(
        starter_feature_rows[["season", "date", "game_key", "team", *starter_feature_columns]],
        on=["season", "date", "game_key", "team"],
        how="left",
    )
    if merged[starter_feature_columns].isna().any().any():
        return feature_df, feature_columns, status

    status["starter_features_added"] = True
    status["message"] = "Starter features ready"
    status["feature_columns"] = starter_feature_columns
    return merged, [*feature_columns, *starter_feature_columns], status
