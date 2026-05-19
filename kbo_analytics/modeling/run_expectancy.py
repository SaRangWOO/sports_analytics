from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _actual_game_id(value) -> str:
    return str(value).rsplit("_", 1)[0]


def _safe_float(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def build_run_expectancy_frame(features: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-game targets and pregame run-scoring features."""
    feature_rows = features.copy()
    feature_rows["actual_game_id"] = feature_rows["game_id"].apply(_actual_game_id)

    game_rows = games.copy()
    game_rows["actual_game_id"] = game_rows["game_id"].apply(_actual_game_id)
    game_rows = game_rows[(game_rows["status"] == "Final") & game_rows["score_team"].notna() & game_rows["score_opp"].notna()]

    rows = []
    for game_id, grouped_features in feature_rows.groupby("actual_game_id", sort=False):
        if len(grouped_features) != 2:
            continue
        grouped_scores = game_rows[game_rows["actual_game_id"] == game_id]
        if len(grouped_scores) != 2:
            continue

        home_features = grouped_features[grouped_features["is_home"] == 1]
        away_features = grouped_features[grouped_features["is_home"] == 0]
        home_scores = grouped_scores[grouped_scores["home_away"] == "H"]
        away_scores = grouped_scores[grouped_scores["home_away"] == "A"]
        if home_features.empty or away_features.empty or home_scores.empty or away_scores.empty:
            continue

        home = home_features.iloc[0]
        away = away_features.iloc[0]
        home_score = float(home_scores.iloc[0]["score_team"])
        away_score = float(away_scores.iloc[0]["score_team"])

        rows.append(
            {
                "game_id": game_id,
                "date": pd.to_datetime(home["date"]).strftime("%Y-%m-%d"),
                "home_team": home["team"],
                "away_team": away["team"],
                "home_score": home_score,
                "away_score": away_score,
                "run_diff": home_score - away_score,
                "total_runs": home_score + away_score,
                "target_home_win": np.nan if pd.isna(home["target_win"]) else int(home["target_win"]),
                "home_recent_runs_avg": round(_safe_float(home, "avg_score_last_5"), 4),
                "away_recent_runs_avg": round(_safe_float(away, "avg_score_last_5"), 4),
                "home_recent_allowed_avg": round(_safe_float(home, "avg_allowed_last_5"), 4),
                "away_recent_allowed_avg": round(_safe_float(away, "avg_allowed_last_5"), 4),
                "home_season_runs_avg": round(_safe_float(home, "season_avg_score_prior"), 4),
                "away_season_runs_avg": round(_safe_float(away, "season_avg_score_prior"), 4),
                "home_season_allowed_avg": round(_safe_float(home, "season_avg_allowed_prior"), 4),
                "away_season_allowed_avg": round(_safe_float(away, "season_avg_allowed_prior"), 4),
                "recent_runs_avg_gap": round(_safe_float(home, "avg_score_last_5") - _safe_float(away, "avg_score_last_5"), 4),
                "recent_allowed_avg_gap": round(_safe_float(away, "avg_allowed_last_5") - _safe_float(home, "avg_allowed_last_5"), 4),
                "season_runs_avg_gap": round(_safe_float(home, "season_avg_score_prior") - _safe_float(away, "season_avg_score_prior"), 4),
                "season_allowed_avg_gap": round(_safe_float(away, "season_avg_allowed_prior") - _safe_float(home, "season_avg_allowed_prior"), 4),
                "recent_run_diff_gap": round(_safe_float(home, "avg_run_diff_last_5") - _safe_float(away, "avg_run_diff_last_5"), 4),
                "season_run_diff_gap": round(_safe_float(home, "season_avg_run_diff_prior") - _safe_float(away, "season_avg_run_diff_prior"), 4),
                "home_recent_10_win_rate": round(_safe_float(home, "recent_10_win_rate"), 4),
                "away_recent_10_win_rate": round(_safe_float(away, "recent_10_win_rate"), 4),
                "recent_10_win_rate_gap": round(_safe_float(home, "recent_10_win_rate") - _safe_float(away, "recent_10_win_rate"), 4),
                "home_rest_days": round(_safe_float(home, "rest_days"), 2),
                "away_rest_days": round(_safe_float(away, "rest_days"), 2),
                "rest_days_gap": round(_safe_float(home, "rest_days") - _safe_float(away, "rest_days"), 2),
                "venue_win_rate_gap": round(_safe_float(home, "venue_win_rate_prior") - _safe_float(away, "venue_win_rate_prior"), 4),
                "elo_diff": round(_safe_float(home, "team_elo_pre") - _safe_float(away, "team_elo_pre"), 4),
            }
        )
    return pd.DataFrame(rows)


def export_run_expectancy_dataset(features: pd.DataFrame, games: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_run_expectancy_frame(features, games)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame
