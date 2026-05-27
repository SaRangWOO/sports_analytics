from __future__ import annotations

import pandas as pd
import numpy as np

from features.team_features import add_opponent_features, add_pre_game_team_features, build_team_game_rows


def build_prediction_feature_matrix(completed_games: pd.DataFrame, target_games: pd.DataFrame) -> pd.DataFrame:
    completed_rows = build_team_game_rows(completed_games)
    target = target_games.copy()
    target["season"] = target["date"].dt.year
    target["game_key"] = target["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    target["score_team"] = np.nan
    target["score_opp"] = np.nan
    target["is_home"] = target["home_away"].eq("H").astype(int)
    target["target_runs"] = np.nan
    target["runs_allowed"] = np.nan
    target["target_win"] = 0
    target["month"] = target["date"].dt.month
    columns = [
        "season",
        "date",
        "game_key",
        "game_id",
        "team",
        "opponent",
        "is_home",
        "month",
        "target_runs",
        "runs_allowed",
        "target_win",
    ]
    if "ballpark" in target.columns:
        columns.append("ballpark")
        if "ballpark" not in completed_rows.columns:
            completed_rows["ballpark"] = ""
    target_rows = target[columns]
    combined = pd.concat([completed_rows, target_rows], ignore_index=True)
    combined = combined.sort_values(["season", "date", "game_key", "is_home"]).reset_index(drop=True)
    featured = add_pre_game_team_features(combined)
    featured = add_opponent_features(featured)
    target_keys = set(target_rows["game_id"])
    return featured[featured["game_id"].isin(target_keys)].copy().reset_index(drop=True)
