from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_game_level_frame(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    working = features.copy()
    working["actual_game_id"] = working["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    for game_id, game_rows in working.groupby("actual_game_id", sort=False):
        if len(game_rows) != 2:
            continue
        home_rows = game_rows[game_rows["is_home"] == 1]
        away_rows = game_rows[game_rows["is_home"] == 0]
        if home_rows.empty or away_rows.empty:
            continue
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]
        rows.append(
            {
                "game_id": game_id,
                "date": pd.to_datetime(home["date"]).strftime("%Y-%m-%d"),
                "home_team": home["team"],
                "away_team": away["team"],
                "target_home_win": np.nan if pd.isna(home["target_win"]) else int(home["target_win"]),
                "home_recent_10_win_rate": round(float(home["recent_10_win_rate"]), 4),
                "away_recent_10_win_rate": round(float(away["recent_10_win_rate"]), 4),
                "recent_10_win_rate_gap": round(float(home["recent_10_win_rate"] - away["recent_10_win_rate"]), 4),
                "season_win_rate_gap": round(float(home["season_win_rate_prior"] - away["season_win_rate_prior"]), 4),
                "season_avg_run_diff_gap": round(float(home["season_avg_run_diff_prior"] - away["season_avg_run_diff_prior"]), 4),
                "recent_run_diff_10_gap": round(float(home["avg_run_diff_last_10"] - away["avg_run_diff_last_10"]), 4),
                "home_venue_win_rate": round(float(home["venue_win_rate_prior"]), 4),
                "away_venue_win_rate": round(float(away["venue_win_rate_prior"]), 4),
                "venue_win_rate_gap": round(float(home["venue_win_rate_prior"] - away["venue_win_rate_prior"]), 4),
                "home_games_last_7_days": int(home["games_last_7_days"]),
                "away_games_last_7_days": int(away["games_last_7_days"]),
                "games_last_7_days_gap": int(away["games_last_7_days"] - home["games_last_7_days"]),
                "home_rest_days": round(float(home["rest_days"]), 2),
                "away_rest_days": round(float(away["rest_days"]), 2),
                "rest_days_gap": round(float(home["rest_days"] - away["rest_days"]), 2),
                "home_bullpen_fatigue_proxy": round(float(home["games_last_7_days"] + home["back_to_back"] * 1.5), 2),
                "away_bullpen_fatigue_proxy": round(float(away["games_last_7_days"] + away["back_to_back"] * 1.5), 2),
                "bullpen_fatigue_gap": round(float((away["games_last_7_days"] + away["back_to_back"] * 1.5) - (home["games_last_7_days"] + home["back_to_back"] * 1.5)), 2),
            }
        )
    return pd.DataFrame(rows)


def export_game_level_dataset(features: pd.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_game_level_frame(features).to_csv(output_path, index=False, encoding="utf-8-sig")


def prepare_game_level_matrix(frame: pd.DataFrame):
    x = frame.drop(columns=["date", "game_id", "target_home_win"])
    x = pd.get_dummies(x, columns=["home_team", "away_team"], drop_first=False, dtype=float)
    y = frame["target_home_win"].to_numpy(dtype=float)
    return x, y


def align_game_level_matrix(frame: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series):
    x, _ = prepare_game_level_matrix(frame)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return (x - mean) / std.replace(0, 1)
