from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_features(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[df["status"] == "Final"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["target_win"] = (df["result"] == "Win").astype(int)
    df["run_diff"] = df["score_team"] - df["score_opp"]
    df["month"] = df["date"].dt.month
    df["is_home"] = (df["home_away"] == "H").astype(int)
    if "series_id" in df.columns:
        df["series_game_no"] = df.groupby("series_id").cumcount() + 1
    else:
        df["series_game_no"] = df.groupby(["opponent", "home_away"]).cumcount() + 1
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
    ]
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
