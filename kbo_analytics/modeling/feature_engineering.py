from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_elo_features(df: pd.DataFrame, k_factor: float = 20.0) -> pd.DataFrame:
    df = df.copy()
    df["actual_game_id"] = df["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    df["team_elo_pre"] = 1500.0
    df["opponent_elo_pre"] = 1500.0
    df["elo_diff"] = 0.0
    ratings: dict[str, float] = {}

    for _, game_rows in df.groupby("actual_game_id", sort=False):
        if len(game_rows) < 2:
            continue
        first = game_rows.iloc[0]
        second = game_rows.iloc[1]
        team_a = first["team"]
        team_b = second["team"]
        rating_a = ratings.get(team_a, 1500.0)
        rating_b = ratings.get(team_b, 1500.0)

        a_mask = game_rows.index[game_rows["team"] == team_a]
        b_mask = game_rows.index[game_rows["team"] == team_b]
        df.loc[a_mask, ["team_elo_pre", "opponent_elo_pre", "elo_diff"]] = [rating_a, rating_b, rating_a - rating_b]
        df.loc[b_mask, ["team_elo_pre", "opponent_elo_pre", "elo_diff"]] = [rating_b, rating_a, rating_b - rating_a]

        if first.get("result") not in {"Win", "Loss"}:
            continue
        score_a = 1.0 if first["result"] == "Win" else 0.0
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        delta = k_factor * (score_a - expected_a)
        ratings[team_a] = rating_a + delta
        ratings[team_b] = rating_b - delta

    return df


def build_features(input_path: str | Path, include_unlabeled: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if include_unlabeled:
        df = df[df["status"].isin(["Final", "Scheduled"])].copy()
    else:
        df = df[df["status"] == "Final"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year
    sort_columns = ["date"]
    if "team" in df.columns:
        sort_columns.append("team")
    df = df.sort_values(sort_columns).reset_index(drop=True)

    df["target_win"] = df["result"].map({"Win": 1, "Loss": 0})
    df["run_diff"] = df["score_team"] - df["score_opp"]
    df["month"] = df["date"].dt.month
    df["is_home"] = (df["home_away"] == "H").astype(int)
    if "series_id" in df.columns and "team" in df.columns:
        df["series_game_no"] = df.groupby(["season", "team", "series_id"]).cumcount() + 1
    elif "series_id" in df.columns:
        df["series_game_no"] = df.groupby("series_id").cumcount() + 1
    elif "team" in df.columns:
        df["series_game_no"] = df.groupby(["season", "team", "opponent", "home_away"]).cumcount() + 1
    else:
        df["series_game_no"] = df.groupby(["opponent", "home_away"]).cumcount() + 1

    if "team" in df.columns:
        df = add_elo_features(df)
        group_keys = ["season", "team"]
        grouped = df.groupby(group_keys, group_keys=False)
        df["rest_days"] = grouped["date"].diff().dt.days.fillna(1).clip(lower=1)
        shifted_win = grouped["target_win"].shift(1)
        shifted_score = grouped["score_team"].shift(1)
        shifted_allowed = grouped["score_opp"].shift(1)
        shifted_diff = grouped["run_diff"].shift(1)
        prior_games = grouped.cumcount()
        prior_wins = grouped["target_win"].cumsum().shift(1)
        first_team_row = grouped.cumcount() == 0
        prior_wins = prior_wins.mask(first_team_row, 0)
        df["season_win_rate_prior"] = (prior_wins / prior_games.where(prior_games > 0)).fillna(0.5).astype(float)
        df["recent_5_win_rate"] = shifted_win.groupby([df["season"], df["team"]]).rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.5)
        df["avg_score_last_5"] = shifted_score.groupby([df["season"], df["team"]]).rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(df["score_team"].mean())
        df["avg_allowed_last_5"] = shifted_allowed.groupby([df["season"], df["team"]]).rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(df["score_opp"].mean())
        df["avg_run_diff_last_5"] = shifted_diff.groupby([df["season"], df["team"]]).rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0)

        opponent_context = df[
            [
                "date",
                "team",
                "recent_5_win_rate",
                "avg_run_diff_last_5",
                "season_win_rate_prior",
            ]
        ].rename(
            columns={
                "team": "opponent",
                "recent_5_win_rate": "opponent_recent_5_win_rate",
                "avg_run_diff_last_5": "opponent_avg_run_diff_last_5",
                "season_win_rate_prior": "opponent_season_win_rate_prior",
            }
        )
        df = df.merge(opponent_context, on=["date", "opponent"], how="left")
        df["opponent_recent_5_win_rate"] = df["opponent_recent_5_win_rate"].fillna(0.5)
        df["opponent_avg_run_diff_last_5"] = df["opponent_avg_run_diff_last_5"].fillna(0)
        df["opponent_season_win_rate_prior"] = df["opponent_season_win_rate_prior"].fillna(0.5)
        df["season_win_rate_gap"] = df["season_win_rate_prior"] - df["opponent_season_win_rate_prior"]
        df["recent_5_win_rate_gap"] = df["recent_5_win_rate"] - df["opponent_recent_5_win_rate"]
        games_last_7 = pd.Series(0, index=df.index, dtype=float)
        for _, team_dates in df.groupby(group_keys)["date"]:
            for row_index, current_date in team_dates.items():
                games_last_7.loc[row_index] = (
                    (team_dates < current_date)
                    & (team_dates >= current_date - pd.Timedelta(days=7))
                ).sum()
        df["games_last_7_days"] = games_last_7
        df["back_to_back"] = (df["rest_days"] <= 1).astype(int)
    else:
        df["rest_days"] = df["date"].diff().dt.days.fillna(1).clip(lower=1)
        shifted_win = df["target_win"].shift(1)
        shifted_score = df["score_team"].shift(1)
        shifted_allowed = df["score_opp"].shift(1)
        shifted_diff = df["run_diff"].shift(1)

        df["recent_5_win_rate"] = shifted_win.rolling(5, min_periods=1).mean().fillna(0.5)
        df["avg_score_last_5"] = shifted_score.rolling(5, min_periods=1).mean().fillna(df["score_team"].mean())
        df["avg_allowed_last_5"] = shifted_allowed.rolling(5, min_periods=1).mean().fillna(df["score_opp"].mean())
        df["avg_run_diff_last_5"] = shifted_diff.rolling(5, min_periods=1).mean().fillna(0)
        df["season_win_rate_prior"] = 0.5
        df["opponent_recent_5_win_rate"] = 0.5
        df["opponent_avg_run_diff_last_5"] = 0
        df["opponent_season_win_rate_prior"] = 0.5
        df["season_win_rate_gap"] = 0
        df["recent_5_win_rate_gap"] = 0
        df["team_elo_pre"] = 1500.0
        df["opponent_elo_pre"] = 1500.0
        df["elo_diff"] = 0.0
        df["games_last_7_days"] = 0
        df["back_to_back"] = 0

    if include_unlabeled:
        df = df[df["result"].isin(["Win", "Loss", ""]) | df["result"].isna()].copy()
    else:
        # Draw games do not provide a clean binary target for this first model.
        df = df[df["result"].isin(["Win", "Loss"])].copy()

    feature_columns = [
        "date",
        "game_id",
    ]
    if "team" in df.columns:
        feature_columns.append("team")
    feature_columns.extend([
        "opponent",
        "is_home",
        "month",
        "series_game_no",
        "rest_days",
        "recent_5_win_rate",
        "avg_score_last_5",
        "avg_allowed_last_5",
        "avg_run_diff_last_5",
        "season_win_rate_prior",
        "opponent_recent_5_win_rate",
        "opponent_avg_run_diff_last_5",
        "opponent_season_win_rate_prior",
        "season_win_rate_gap",
        "recent_5_win_rate_gap",
        "team_elo_pre",
        "opponent_elo_pre",
        "elo_diff",
        "games_last_7_days",
        "back_to_back",
        "target_win",
    ])
    return df[feature_columns]


def main():
    parser = argparse.ArgumentParser(description="Build KBO win prediction features.")
    parser.add_argument("--input", default="../data/weekly/game_results.csv")
    parser.add_argument("--output", default="features.csv")
    args = parser.parse_args()

    features = build_features(args.input)
    features.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved features: {args.output} ({len(features)} rows)")


if __name__ == "__main__":
    main()
