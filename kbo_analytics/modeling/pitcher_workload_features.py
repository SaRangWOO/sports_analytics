from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


WORKLOAD_COLUMNS = [
    "starter_name",
    "starter_id",
    "starter_recent3_era",
    "starter_recent3_whip",
    "starter_recent3_ip",
    "starter_recent3_pitch_count",
    "starter_rest_days",
    "bullpen_pitch_count_last1d",
    "bullpen_pitch_count_last3d",
    "bullpen_appearances_last3d",
    "previous_game_bullpen_pitch_count",
]


def _ratio(numerator: float, denominator: float, multiplier: float) -> float:
    return float(numerator * multiplier / denominator) if denominator else np.nan


def build_pitcher_workload_features(logs: pd.DataFrame) -> pd.DataFrame:
    if logs is None or logs.empty:
        return pd.DataFrame(columns=["game_id", "team", *WORKLOAD_COLUMNS])
    frame = logs.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    numeric = [
        "pitcher_index", "pitcher_id", "innings_outs", "pitch_count", "hits_allowed",
        "walks_hbp", "earned_runs",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_starter"] = frame["is_starter"].astype(str).str.lower().isin({"true", "1"})

    starters = frame[frame["is_starter"]].copy()
    starters["pitcher_key"] = np.where(
        starters["pitcher_id"].notna(),
        "id:" + starters["pitcher_id"].astype("Int64").astype(str),
        "name:" + starters["pitcher_name"].astype(str).str.replace(" ", "", regex=False),
    )
    starters = starters.sort_values(["pitcher_key", "game_date", "game_id"])
    starter_rows = []
    for _, group in starters.groupby("pitcher_key", sort=False):
        group = group.reset_index(drop=True)
        for index, row in group.iterrows():
            prior = group.iloc[max(0, index - 3):index]
            outs = float(prior["innings_outs"].sum())
            starter_rows.append(
                {
                    "game_id": row["game_id"],
                    "team": row["team"],
                    "starter_name": row["pitcher_name"],
                    "starter_id": row["pitcher_id"],
                    "starter_recent3_era": _ratio(float(prior["earned_runs"].sum()), outs, 27),
                    "starter_recent3_whip": _ratio(float(prior["hits_allowed"].sum() + prior["walks_hbp"].sum()), outs, 3),
                    "starter_recent3_ip": float(prior["innings_outs"].mean() / 3) if len(prior) else np.nan,
                    "starter_recent3_pitch_count": float(prior["pitch_count"].mean()) if len(prior) else np.nan,
                    "starter_rest_days": int((row["game_date"] - group.iloc[index - 1]["game_date"]).days) if index else np.nan,
                }
            )
    starter_features = pd.DataFrame(starter_rows)

    bullpen = frame[~frame["is_starter"]].copy()
    daily = (
        bullpen.groupby(["team", "game_date"], as_index=False)
        .agg(bullpen_pitch_count=("pitch_count", "sum"), bullpen_appearances=("pitcher_index", "count"))
        .sort_values(["team", "game_date"])
    )
    workload_rows = []
    game_teams = frame[["game_id", "game_date", "team"]].drop_duplicates().sort_values(["team", "game_date", "game_id"])
    for team, group in game_teams.groupby("team", sort=False):
        team_daily = daily[daily["team"].eq(team)]
        for _, row in group.iterrows():
            game_date = row["game_date"]
            last1 = team_daily[team_daily["game_date"].eq(game_date - timedelta(days=1))]
            last3 = team_daily[
                team_daily["game_date"].lt(game_date)
                & team_daily["game_date"].ge(game_date - timedelta(days=3))
            ]
            previous = team_daily[team_daily["game_date"].lt(game_date)].tail(1)
            workload_rows.append(
                {
                    "game_id": row["game_id"],
                    "team": team,
                    "bullpen_pitch_count_last1d": float(last1["bullpen_pitch_count"].sum()),
                    "bullpen_pitch_count_last3d": float(last3["bullpen_pitch_count"].sum()),
                    "bullpen_appearances_last3d": int(last3["bullpen_appearances"].sum()),
                    "previous_game_bullpen_pitch_count": float(previous["bullpen_pitch_count"].iloc[0]) if len(previous) else np.nan,
                }
            )
    workload = pd.DataFrame(workload_rows)
    return starter_features.merge(workload, on=["game_id", "team"], how="outer")


def attach_pitcher_workload_features(game_frame: pd.DataFrame, workload: pd.DataFrame) -> pd.DataFrame:
    output = game_frame.copy()
    if workload is None or workload.empty:
        return output
    workload = workload.reindex(columns=["game_id", "team", *WORKLOAD_COLUMNS])
    workload_columns = [f"{side}_{column}" for side in ["home", "away"] for column in WORKLOAD_COLUMNS]
    output = output.drop(columns=[column for column in workload_columns if column in output.columns])
    for side in ["home", "away"]:
        side_frame = workload.rename(
            columns={
                "team": f"{side}_team",
                **{column: f"{side}_{column}" for column in WORKLOAD_COLUMNS},
            }
        )
        output = output.merge(side_frame, on=["game_id", f"{side}_team"], how="left")
    output["home_starter"] = output["home_starter_name"].fillna("")
    output["away_starter"] = output["away_starter_name"].fillna("")
    output["both_starters_known"] = output["home_starter_name"].notna() & output["away_starter_name"].notna()
    output["starter_info_quality"] = output["both_starters_known"].astype(float)
    for feature in [
        "starter_recent3_era", "starter_recent3_whip", "starter_recent3_ip",
        "starter_recent3_pitch_count", "starter_rest_days", "bullpen_pitch_count_last1d",
        "bullpen_pitch_count_last3d", "bullpen_appearances_last3d", "previous_game_bullpen_pitch_count",
    ]:
        output[f"{feature}_gap"] = output[f"home_{feature}"].fillna(0) - output[f"away_{feature}"].fillna(0)
    output["home_starter_era_prior"] = output["home_starter_recent3_era"]
    output["away_starter_era_prior"] = output["away_starter_recent3_era"]
    output["starter_era_gap"] = output["starter_recent3_era_gap"]
    output["home_starter_whip_prior"] = output["home_starter_recent3_whip"]
    output["away_starter_whip_prior"] = output["away_starter_recent3_whip"]
    output["starter_whip_gap"] = output["starter_recent3_whip_gap"]
    output["previous_game_bullpen_usage_gap"] = output["previous_game_bullpen_pitch_count_gap"]
    return output
