from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_features(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[df["status"] == "Final"].copy()
    df["date"] = pd.to_datetime(df["date"])
    sort_columns = ["date"]
    if "team" in df.columns:
        sort_columns.append("team")
    df = df.sort_values(sort_columns).reset_index(drop=True)

    df["target_win"] = (df["result"] == "Win").astype(int)
    df["run_diff"] = df["score_team"] - df["score_opp"]
    df["month"] = df["date"].dt.month
    df["is_home"] = (df["home_away"] == "H").astype(int)
    if "series_id" in df.columns and "team" in df.columns:
        df["series_game_no"] = df.groupby(["team", "series_id"]).cumcount() + 1
    elif "series_id" in df.columns:
        df["series_game_no"] = df.groupby("series_id").cumcount() + 1
    elif "team" in df.columns:
        df["series_game_no"] = df.groupby(["team", "opponent", "home_away"]).cumcount() + 1
    else:
        df["series_game_no"] = df.groupby(["opponent", "home_away"]).cumcount() + 1

    if "team" in df.columns:
        df["rest_days"] = df.groupby("team")["date"].diff().dt.days.fillna(1).clip(lower=1)
        shifted_win = df.groupby("team")["target_win"].shift(1)
        shifted_score = df.groupby("team")["score_team"].shift(1)
        shifted_allowed = df.groupby("team")["score_opp"].shift(1)
        shifted_diff = df.groupby("team")["run_diff"].shift(1)
        df["recent_5_win_rate"] = shifted_win.groupby(df["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.5)
        df["avg_score_last_5"] = shifted_score.groupby(df["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).fillna(df["score_team"].mean())
        df["avg_allowed_last_5"] = shifted_allowed.groupby(df["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).fillna(df["score_opp"].mean())
        df["avg_run_diff_last_5"] = shifted_diff.groupby(df["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
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
