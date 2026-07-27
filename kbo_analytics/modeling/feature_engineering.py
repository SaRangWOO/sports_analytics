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


def add_prior_rate(numerator: pd.Series, denominator: pd.Series, default: float) -> pd.Series:
    return (numerator / denominator.where(denominator > 0)).fillna(default).astype(float)


def assign_prior_streak_features(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    streak_length = pd.Series(0, index=df.index, dtype=int)
    streak_type = pd.Series(0, index=df.index, dtype=int)
    for _, group in df.groupby(group_keys, sort=False):
        current_type = 0
        current_length = 0
        for row_index, result in group["target_win"].items():
            streak_length.loc[row_index] = current_length
            streak_type.loc[row_index] = current_type
            if pd.isna(result):
                continue
            result_type = 1 if int(result) == 1 else -1
            if result_type == current_type:
                current_length += 1
            else:
                current_type = result_type
                current_length = 1
    df["team_current_streak_length"] = streak_length
    df["team_current_streak_type"] = streak_type
    return df


def rolling_group_mean(values: pd.Series, season: pd.Series, team: pd.Series, window: int, default: float) -> pd.Series:
    return values.groupby([season, team]).rolling(window, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(default)


def rolling_group_std(values: pd.Series, season: pd.Series, team: pd.Series, window: int) -> pd.Series:
    return values.groupby([season, team]).rolling(window, min_periods=2).std().reset_index(level=[0, 1], drop=True).fillna(0.0)


def prior_group_cumsum(df: pd.DataFrame, group_keys: list[str], value_column: str) -> pd.Series:
    return df.groupby(group_keys, sort=False)[value_column].transform(
        lambda values: values.fillna(0).cumsum().shift(1)
    )


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
    df["actual_run_margin"] = df["run_diff"].abs()
    df["actual_close_game"] = (df["actual_run_margin"] <= 1).astype(int)
    df["actual_blowout_game"] = (df["actual_run_margin"] >= 5).astype(int)
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
        shifted_abs_diff = shifted_diff.abs()
        shifted_close = (shifted_abs_diff <= 1).astype(float)
        shifted_blowout_win = ((grouped["run_diff"].shift(1) >= 5) & (grouped["target_win"].shift(1) == 1)).astype(float)
        shifted_blowout_loss = ((grouped["run_diff"].shift(1) <= -5) & (grouped["target_win"].shift(1) == 0)).astype(float)
        df["previous_game_win"] = shifted_win.fillna(0.5)
        df["previous_game_score"] = shifted_score.fillna(df["score_team"].mean())
        df["previous_game_allowed"] = shifted_allowed.fillna(df["score_opp"].mean())
        df["previous_game_run_diff"] = shifted_diff.fillna(0.0)
        prior_games = grouped.cumcount()
        prior_wins = prior_group_cumsum(df, group_keys, "target_win")
        prior_runs = prior_group_cumsum(df, group_keys, "score_team")
        prior_allowed = prior_group_cumsum(df, group_keys, "score_opp")
        prior_run_diff = prior_group_cumsum(df, group_keys, "run_diff")
        first_team_row = grouped.cumcount() == 0
        prior_wins = prior_wins.mask(first_team_row, 0)
        prior_runs = prior_runs.mask(first_team_row, 0)
        prior_allowed = prior_allowed.mask(first_team_row, 0)
        prior_run_diff = prior_run_diff.mask(first_team_row, 0)
        df["season_win_rate_prior"] = add_prior_rate(prior_wins, prior_games, 0.5)
        df["season_avg_score_prior"] = add_prior_rate(prior_runs, prior_games, df["score_team"].mean())
        df["season_avg_allowed_prior"] = add_prior_rate(prior_allowed, prior_games, df["score_opp"].mean())
        df["season_avg_run_diff_prior"] = add_prior_rate(prior_run_diff, prior_games, 0.0)
        df["recent_5_win_rate"] = rolling_group_mean(shifted_win, df["season"], df["team"], 5, 0.5)
        df["recent_10_win_rate"] = rolling_group_mean(shifted_win, df["season"], df["team"], 10, 0.5)
        df["avg_score_last_3"] = rolling_group_mean(shifted_score, df["season"], df["team"], 3, df["score_team"].mean())
        df["avg_score_last_5"] = rolling_group_mean(shifted_score, df["season"], df["team"], 5, df["score_team"].mean())
        df["avg_score_last_10"] = rolling_group_mean(shifted_score, df["season"], df["team"], 10, df["score_team"].mean())
        df["avg_allowed_last_3"] = rolling_group_mean(shifted_allowed, df["season"], df["team"], 3, df["score_opp"].mean())
        df["avg_allowed_last_5"] = rolling_group_mean(shifted_allowed, df["season"], df["team"], 5, df["score_opp"].mean())
        df["avg_allowed_last_10"] = rolling_group_mean(shifted_allowed, df["season"], df["team"], 10, df["score_opp"].mean())
        df["avg_run_diff_last_5"] = rolling_group_mean(shifted_diff, df["season"], df["team"], 5, 0)
        df["avg_run_diff_last_10"] = rolling_group_mean(shifted_diff, df["season"], df["team"], 10, 0)
        df["team_recent_5_run_std"] = rolling_group_std(shifted_score, df["season"], df["team"], 5)
        df["team_recent_10_run_std"] = rolling_group_std(shifted_score, df["season"], df["team"], 10)
        df["team_recent_5_allowed_std"] = rolling_group_std(shifted_allowed, df["season"], df["team"], 5)
        df["team_recent_10_allowed_std"] = rolling_group_std(shifted_allowed, df["season"], df["team"], 10)
        df["team_recent_5_close_game_rate"] = rolling_group_mean(shifted_close, df["season"], df["team"], 5, 0.0)
        df["team_recent_10_blowout_win_rate"] = rolling_group_mean(shifted_blowout_win, df["season"], df["team"], 10, 0.0)
        df["team_recent_10_blowout_loss_rate"] = rolling_group_mean(shifted_blowout_loss, df["season"], df["team"], 10, 0.0)
        df["team_run_momentum_3_vs_10"] = df["avg_score_last_3"] - df["avg_score_last_10"]
        df["team_allowed_momentum_3_vs_10"] = df["avg_allowed_last_3"] - df["avg_allowed_last_10"]
        df = assign_prior_streak_features(df, group_keys)
        df["team_winning_streak_flag"] = ((df["team_current_streak_type"] == 1) & (df["team_current_streak_length"] >= 2)).astype(int)
        df["team_losing_streak_flag"] = ((df["team_current_streak_type"] == -1) & (df["team_current_streak_length"] >= 2)).astype(int)
        df["recent_5_close_game_rate"] = df["team_recent_5_close_game_rate"]
        venue_grouped = df.groupby(["season", "team", "home_away"], group_keys=False)
        venue_prior_games = venue_grouped.cumcount()
        venue_prior_wins = prior_group_cumsum(df, ["season", "team", "home_away"], "target_win")
        venue_shifted_win = venue_grouped["target_win"].shift(1)
        venue_shifted_diff = venue_grouped["run_diff"].shift(1)
        venue_first_row = venue_grouped.cumcount() == 0
        venue_prior_wins = venue_prior_wins.mask(venue_first_row, 0)
        df["venue_win_rate_prior"] = add_prior_rate(venue_prior_wins, venue_prior_games, 0.5)
        df["team_recent_same_venue_win_rate_prior"] = venue_shifted_win.groupby([df["season"], df["team"], df["home_away"]]).rolling(5, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).fillna(0.5)
        df["team_recent_same_venue_run_diff"] = venue_shifted_diff.groupby([df["season"], df["team"], df["home_away"]]).rolling(5, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).fillna(0.0)
        df["team_recent_home_win_rate_prior"] = df["team_recent_same_venue_win_rate_prior"].where(df["home_away"] == "H", 0.5)
        df["team_recent_away_win_rate_prior"] = df["team_recent_same_venue_win_rate_prior"].where(df["home_away"] == "A", 0.5)
        month_grouped = df.groupby(["season", "team", "month"], group_keys=False)
        month_prior_games = month_grouped.cumcount()
        month_prior_wins = prior_group_cumsum(df, ["season", "team", "month"], "target_win")
        month_prior_diff = prior_group_cumsum(df, ["season", "team", "month"], "run_diff")
        month_first_row = month_grouped.cumcount() == 0
        month_prior_wins = month_prior_wins.mask(month_first_row, 0)
        month_prior_diff = month_prior_diff.mask(month_first_row, 0)
        df["team_month_win_rate_prior"] = add_prior_rate(month_prior_wins, month_prior_games, 0.5)
        df["team_month_run_diff_prior"] = add_prior_rate(month_prior_diff, month_prior_games, 0.0)
        df["season_phase"] = pd.cut(df["month"], bins=[0, 4, 7, 12], labels=[0, 1, 2]).astype(int)
        h2h_grouped = df.groupby(["season", "team", "opponent"], group_keys=False)
        h2h_prior_games = h2h_grouped.cumcount()
        h2h_prior_wins = prior_group_cumsum(df, ["season", "team", "opponent"], "target_win")
        h2h_first_row = h2h_grouped.cumcount() == 0
        h2h_prior_wins = h2h_prior_wins.mask(h2h_first_row, 0)
        df["head_to_head_win_rate_prior"] = add_prior_rate(h2h_prior_wins, h2h_prior_games, 0.5)

        opponent_context = df[
            [
                "date",
                "team",
                "recent_5_win_rate",
                "recent_10_win_rate",
                "avg_run_diff_last_5",
                "avg_run_diff_last_10",
                "team_recent_5_run_std",
                "team_recent_10_run_std",
                "team_recent_5_allowed_std",
                "team_recent_10_allowed_std",
                "team_recent_5_close_game_rate",
                "team_recent_10_blowout_win_rate",
                "team_recent_10_blowout_loss_rate",
                "team_run_momentum_3_vs_10",
                "team_allowed_momentum_3_vs_10",
                "season_win_rate_prior",
                "season_avg_score_prior",
                "season_avg_allowed_prior",
                "season_avg_run_diff_prior",
                "team_current_streak_length",
                "team_current_streak_type",
                "team_winning_streak_flag",
                "team_losing_streak_flag",
                "recent_5_close_game_rate",
                "venue_win_rate_prior",
                "team_recent_home_win_rate_prior",
                "team_recent_away_win_rate_prior",
                "team_recent_same_venue_run_diff",
                "team_month_win_rate_prior",
                "team_month_run_diff_prior",
                "head_to_head_win_rate_prior",
                "previous_game_win",
                "previous_game_score",
                "previous_game_allowed",
                "previous_game_run_diff",
            ]
        ].rename(
            columns={
                "team": "opponent",
                "recent_5_win_rate": "opponent_recent_5_win_rate",
                "recent_10_win_rate": "opponent_recent_10_win_rate",
                "avg_run_diff_last_5": "opponent_avg_run_diff_last_5",
                "avg_run_diff_last_10": "opponent_avg_run_diff_last_10",
                "team_recent_5_run_std": "opponent_recent_5_run_std",
                "team_recent_10_run_std": "opponent_recent_10_run_std",
                "team_recent_5_allowed_std": "opponent_recent_5_allowed_std",
                "team_recent_10_allowed_std": "opponent_recent_10_allowed_std",
                "team_recent_5_close_game_rate": "opponent_recent_5_close_game_rate_exact",
                "team_recent_10_blowout_win_rate": "opponent_recent_10_blowout_win_rate",
                "team_recent_10_blowout_loss_rate": "opponent_recent_10_blowout_loss_rate",
                "team_run_momentum_3_vs_10": "opponent_run_momentum_3_vs_10",
                "team_allowed_momentum_3_vs_10": "opponent_allowed_momentum_3_vs_10",
                "season_win_rate_prior": "opponent_season_win_rate_prior",
                "season_avg_score_prior": "opponent_season_avg_score_prior",
                "season_avg_allowed_prior": "opponent_season_avg_allowed_prior",
                "season_avg_run_diff_prior": "opponent_season_avg_run_diff_prior",
                "team_current_streak_length": "opponent_current_streak_length",
                "team_current_streak_type": "opponent_current_streak_type",
                "team_winning_streak_flag": "opponent_winning_streak_flag",
                "team_losing_streak_flag": "opponent_losing_streak_flag",
                "recent_5_close_game_rate": "opponent_recent_5_close_game_rate",
                "venue_win_rate_prior": "opponent_venue_win_rate_prior",
                "team_recent_home_win_rate_prior": "opponent_recent_home_win_rate_prior",
                "team_recent_away_win_rate_prior": "opponent_recent_away_win_rate_prior",
                "team_recent_same_venue_run_diff": "opponent_recent_same_venue_run_diff",
                "team_month_win_rate_prior": "opponent_month_win_rate_prior",
                "team_month_run_diff_prior": "opponent_month_run_diff_prior",
                "head_to_head_win_rate_prior": "opponent_head_to_head_win_rate_prior",
                "previous_game_win": "opponent_previous_game_win",
                "previous_game_score": "opponent_previous_game_score",
                "previous_game_allowed": "opponent_previous_game_allowed",
                "previous_game_run_diff": "opponent_previous_game_run_diff",
            }
        )
        df = df.merge(opponent_context, on=["date", "opponent"], how="left")
        df["opponent_recent_5_win_rate"] = df["opponent_recent_5_win_rate"].fillna(0.5)
        df["opponent_recent_10_win_rate"] = df["opponent_recent_10_win_rate"].fillna(0.5)
        df["opponent_avg_run_diff_last_5"] = df["opponent_avg_run_diff_last_5"].fillna(0)
        df["opponent_avg_run_diff_last_10"] = df["opponent_avg_run_diff_last_10"].fillna(0)
        for column in [
            "opponent_recent_5_run_std",
            "opponent_recent_10_run_std",
            "opponent_recent_5_allowed_std",
            "opponent_recent_10_allowed_std",
            "opponent_recent_5_close_game_rate_exact",
            "opponent_recent_10_blowout_win_rate",
            "opponent_recent_10_blowout_loss_rate",
            "opponent_run_momentum_3_vs_10",
            "opponent_allowed_momentum_3_vs_10",
            "opponent_recent_same_venue_run_diff",
            "opponent_month_run_diff_prior",
        ]:
            df[column] = df[column].fillna(0.0)
        df["opponent_season_win_rate_prior"] = df["opponent_season_win_rate_prior"].fillna(0.5)
        df["opponent_season_avg_score_prior"] = df["opponent_season_avg_score_prior"].fillna(df["score_opp"].mean())
        df["opponent_season_avg_allowed_prior"] = df["opponent_season_avg_allowed_prior"].fillna(df["score_team"].mean())
        df["opponent_season_avg_run_diff_prior"] = df["opponent_season_avg_run_diff_prior"].fillna(0)
        df["opponent_current_streak_length"] = df["opponent_current_streak_length"].fillna(0)
        df["opponent_current_streak_type"] = df["opponent_current_streak_type"].fillna(0)
        df["opponent_winning_streak_flag"] = df["opponent_winning_streak_flag"].fillna(0)
        df["opponent_losing_streak_flag"] = df["opponent_losing_streak_flag"].fillna(0)
        df["opponent_recent_5_close_game_rate"] = df["opponent_recent_5_close_game_rate"].fillna(0)
        df["opponent_venue_win_rate_prior"] = df["opponent_venue_win_rate_prior"].fillna(0.5)
        df["opponent_recent_home_win_rate_prior"] = df["opponent_recent_home_win_rate_prior"].fillna(0.5)
        df["opponent_recent_away_win_rate_prior"] = df["opponent_recent_away_win_rate_prior"].fillna(0.5)
        df["opponent_month_win_rate_prior"] = df["opponent_month_win_rate_prior"].fillna(0.5)
        df["opponent_head_to_head_win_rate_prior"] = df["opponent_head_to_head_win_rate_prior"].fillna(0.5)
        df["season_win_rate_gap"] = df["season_win_rate_prior"] - df["opponent_season_win_rate_prior"]
        df["recent_10_win_rate_gap"] = df["recent_10_win_rate"] - df["opponent_recent_10_win_rate"]
        df["recent_5_win_rate_gap"] = df["recent_5_win_rate"] - df["opponent_recent_5_win_rate"]
        df["season_avg_score_gap"] = df["season_avg_score_prior"] - df["opponent_season_avg_score_prior"]
        df["season_avg_allowed_gap"] = df["opponent_season_avg_allowed_prior"] - df["season_avg_allowed_prior"]
        df["season_avg_run_diff_gap"] = df["season_avg_run_diff_prior"] - df["opponent_season_avg_run_diff_prior"]
        df["recent_run_diff_10_gap"] = df["avg_run_diff_last_10"] - df["opponent_avg_run_diff_last_10"]
        df["run_std_gap"] = df["team_recent_5_run_std"] - df["opponent_recent_5_run_std"]
        df["allowed_std_gap"] = df["opponent_recent_5_allowed_std"] - df["team_recent_5_allowed_std"]
        df["close_game_rate_gap"] = df["team_recent_5_close_game_rate"] - df["opponent_recent_5_close_game_rate_exact"]
        df["blowout_tendency_gap"] = (
            df["team_recent_10_blowout_win_rate"]
            - df["team_recent_10_blowout_loss_rate"]
            - df["opponent_recent_10_blowout_win_rate"]
            + df["opponent_recent_10_blowout_loss_rate"]
        )
        df["run_momentum_gap"] = df["team_run_momentum_3_vs_10"] - df["opponent_run_momentum_3_vs_10"]
        df["allowed_momentum_gap"] = df["opponent_allowed_momentum_3_vs_10"] - df["team_allowed_momentum_3_vs_10"]
        df["venue_context_win_rate_gap"] = df["team_recent_same_venue_win_rate_prior"] - df["opponent_venue_win_rate_prior"]
        df["month_win_rate_gap"] = df["team_month_win_rate_prior"] - df["opponent_month_win_rate_prior"]
        df["month_run_diff_gap"] = df["team_month_run_diff_prior"] - df["opponent_month_run_diff_prior"]
        df["streak_length_gap"] = (df["team_current_streak_length"] * df["team_current_streak_type"]) - (df["opponent_current_streak_length"] * df["opponent_current_streak_type"])
        df["winning_streak_regression_risk"] = df["team_winning_streak_flag"] * ((df["avg_run_diff_last_5"] < 0.8).astype(int) + (df["recent_5_close_game_rate"] >= 0.6).astype(int))
        df["winning_streak_with_low_run_diff"] = df["team_winning_streak_flag"] * (df["avg_run_diff_last_5"] < 0.8).astype(int)
        df["winning_streak_after_close_games"] = df["team_winning_streak_flag"] * (df["recent_5_close_game_rate"] >= 0.6).astype(int)
        df["venue_win_rate_gap"] = df["venue_win_rate_prior"] - df["opponent_venue_win_rate_prior"]
        df["head_to_head_win_rate_gap"] = df["head_to_head_win_rate_prior"] - df["opponent_head_to_head_win_rate_prior"]
        df["previous_game_win_gap"] = df["previous_game_win"] - df["opponent_previous_game_win"]
        df["previous_game_score_gap"] = df["previous_game_score"] - df["opponent_previous_game_score"]
        df["previous_game_allowed_gap"] = df["opponent_previous_game_allowed"] - df["previous_game_allowed"]
        df["previous_game_run_diff_gap"] = df["previous_game_run_diff"] - df["opponent_previous_game_run_diff"]
        games_last_7 = pd.Series(0, index=df.index, dtype=float)
        for _, team_dates in df.groupby(group_keys)["date"]:
            for row_index, current_date in team_dates.items():
                games_last_7.loc[row_index] = (
                    (team_dates < current_date)
                    & (team_dates >= current_date - pd.Timedelta(days=7))
                ).sum()
        df["games_last_7_days"] = games_last_7
        df["back_to_back"] = (df["rest_days"] <= 1).astype(int)
        df["winning_streak_bullpen_fatigue_proxy"] = df["team_winning_streak_flag"] * df["games_last_7_days"]
        df["losing_streak_with_negative_run_diff"] = df["team_losing_streak_flag"] * (df["avg_run_diff_last_5"] < -0.8).astype(int)
        df["losing_streak_allowed_runs_spike"] = df["team_losing_streak_flag"] * (df["avg_allowed_last_5"] > df["season_avg_allowed_prior"] + 1.0).astype(int)
        df["losing_streak_low_scoring_offense"] = df["team_losing_streak_flag"] * (df["avg_score_last_5"] < df["season_avg_score_prior"] - 1.0).astype(int)
        df["opponent_vs_losing_streak_flag"] = df["opponent_losing_streak_flag"].astype(int)
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
        df["avg_score_last_3"] = shifted_score.rolling(3, min_periods=1).mean().fillna(df["score_team"].mean())
        df["avg_score_last_10"] = shifted_score.rolling(10, min_periods=1).mean().fillna(df["score_team"].mean())
        df["avg_allowed_last_3"] = shifted_allowed.rolling(3, min_periods=1).mean().fillna(df["score_opp"].mean())
        df["avg_allowed_last_10"] = shifted_allowed.rolling(10, min_periods=1).mean().fillna(df["score_opp"].mean())
        for column in [
            "team_recent_5_run_std",
            "team_recent_10_run_std",
            "opponent_recent_5_run_std",
            "opponent_recent_10_run_std",
            "run_std_gap",
            "team_recent_5_allowed_std",
            "opponent_recent_5_allowed_std",
            "allowed_std_gap",
            "team_recent_5_close_game_rate",
            "opponent_recent_5_close_game_rate_exact",
            "close_game_rate_gap",
            "team_recent_10_blowout_win_rate",
            "team_recent_10_blowout_loss_rate",
            "opponent_recent_10_blowout_win_rate",
            "opponent_recent_10_blowout_loss_rate",
            "blowout_tendency_gap",
            "team_run_momentum_3_vs_10",
            "opponent_run_momentum_3_vs_10",
            "run_momentum_gap",
            "team_allowed_momentum_3_vs_10",
            "opponent_allowed_momentum_3_vs_10",
            "allowed_momentum_gap",
            "team_recent_home_win_rate_prior",
            "team_recent_away_win_rate_prior",
            "opponent_recent_home_win_rate_prior",
            "opponent_recent_away_win_rate_prior",
            "venue_context_win_rate_gap",
            "team_recent_same_venue_run_diff",
            "season_phase",
            "team_month_win_rate_prior",
            "opponent_month_win_rate_prior",
            "month_win_rate_gap",
            "team_month_run_diff_prior",
            "opponent_month_run_diff_prior",
            "month_run_diff_gap",
        ]:
            df[column] = 0
        df["season_win_rate_prior"] = 0.5
        df["opponent_recent_5_win_rate"] = 0.5
        df["opponent_avg_run_diff_last_5"] = 0
        df["opponent_season_win_rate_prior"] = 0.5
        df["season_win_rate_gap"] = 0
        df["recent_5_win_rate_gap"] = 0
        df["recent_10_win_rate"] = 0.5
        df["opponent_recent_10_win_rate"] = 0.5
        df["recent_10_win_rate_gap"] = 0
        df["avg_run_diff_last_10"] = 0
        df["opponent_avg_run_diff_last_10"] = 0
        df["recent_run_diff_10_gap"] = 0
        df["team_current_streak_length"] = 0
        df["team_current_streak_type"] = 0
        df["opponent_current_streak_length"] = 0
        df["opponent_current_streak_type"] = 0
        df["streak_length_gap"] = 0
        df["team_losing_streak_flag"] = 0
        df["team_winning_streak_flag"] = 0
        df["opponent_losing_streak_flag"] = 0
        df["opponent_winning_streak_flag"] = 0
        df["winning_streak_regression_risk"] = 0
        df["winning_streak_with_low_run_diff"] = 0
        df["winning_streak_after_close_games"] = 0
        df["winning_streak_bullpen_fatigue_proxy"] = 0
        df["losing_streak_with_negative_run_diff"] = 0
        df["losing_streak_allowed_runs_spike"] = 0
        df["losing_streak_low_scoring_offense"] = 0
        df["opponent_vs_losing_streak_flag"] = 0
        df["season_avg_score_prior"] = df["score_team"].mean()
        df["opponent_season_avg_score_prior"] = df["score_opp"].mean()
        df["season_avg_allowed_prior"] = df["score_opp"].mean()
        df["opponent_season_avg_allowed_prior"] = df["score_team"].mean()
        df["season_avg_run_diff_prior"] = 0
        df["opponent_season_avg_run_diff_prior"] = 0
        df["season_avg_score_gap"] = 0
        df["season_avg_allowed_gap"] = 0
        df["season_avg_run_diff_gap"] = 0
        df["venue_win_rate_prior"] = 0.5
        df["opponent_venue_win_rate_prior"] = 0.5
        df["venue_win_rate_gap"] = 0
        df["head_to_head_win_rate_prior"] = 0.5
        df["opponent_head_to_head_win_rate_prior"] = 0.5
        df["head_to_head_win_rate_gap"] = 0
        df["previous_game_win"] = 0.5
        df["previous_game_score"] = df["score_team"].mean()
        df["previous_game_allowed"] = df["score_opp"].mean()
        df["previous_game_run_diff"] = 0
        df["opponent_previous_game_win"] = 0.5
        df["opponent_previous_game_score"] = df["score_opp"].mean()
        df["opponent_previous_game_allowed"] = df["score_team"].mean()
        df["opponent_previous_game_run_diff"] = 0
        df["previous_game_win_gap"] = 0
        df["previous_game_score_gap"] = 0
        df["previous_game_allowed_gap"] = 0
        df["previous_game_run_diff_gap"] = 0
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
        "actual_run_margin",
        "actual_close_game",
        "actual_blowout_game",
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
        "recent_10_win_rate",
        "avg_score_last_5",
        "avg_allowed_last_5",
        "avg_score_last_3",
        "avg_score_last_10",
        "avg_allowed_last_3",
        "avg_allowed_last_10",
        "avg_run_diff_last_5",
        "avg_run_diff_last_10",
        "team_recent_5_run_std",
        "team_recent_10_run_std",
        "opponent_recent_5_run_std",
        "opponent_recent_10_run_std",
        "run_std_gap",
        "team_recent_5_allowed_std",
        "opponent_recent_5_allowed_std",
        "allowed_std_gap",
        "team_recent_5_close_game_rate",
        "opponent_recent_5_close_game_rate_exact",
        "close_game_rate_gap",
        "team_recent_10_blowout_win_rate",
        "team_recent_10_blowout_loss_rate",
        "opponent_recent_10_blowout_win_rate",
        "opponent_recent_10_blowout_loss_rate",
        "blowout_tendency_gap",
        "team_run_momentum_3_vs_10",
        "opponent_run_momentum_3_vs_10",
        "run_momentum_gap",
        "team_allowed_momentum_3_vs_10",
        "opponent_allowed_momentum_3_vs_10",
        "allowed_momentum_gap",
        "season_win_rate_prior",
        "season_avg_score_prior",
        "season_avg_allowed_prior",
        "season_avg_run_diff_prior",
        "opponent_recent_5_win_rate",
        "opponent_recent_10_win_rate",
        "opponent_avg_run_diff_last_5",
        "opponent_avg_run_diff_last_10",
        "opponent_season_win_rate_prior",
        "opponent_season_avg_score_prior",
        "opponent_season_avg_allowed_prior",
        "opponent_season_avg_run_diff_prior",
        "season_win_rate_gap",
        "recent_5_win_rate_gap",
        "recent_10_win_rate_gap",
        "season_avg_score_gap",
        "season_avg_allowed_gap",
        "season_avg_run_diff_gap",
        "recent_run_diff_10_gap",
        "team_current_streak_length",
        "team_current_streak_type",
        "opponent_current_streak_length",
        "opponent_current_streak_type",
        "streak_length_gap",
        "team_losing_streak_flag",
        "team_winning_streak_flag",
        "opponent_losing_streak_flag",
        "opponent_winning_streak_flag",
        "winning_streak_regression_risk",
        "winning_streak_with_low_run_diff",
        "winning_streak_after_close_games",
        "winning_streak_bullpen_fatigue_proxy",
        "losing_streak_with_negative_run_diff",
        "losing_streak_allowed_runs_spike",
        "losing_streak_low_scoring_offense",
        "opponent_vs_losing_streak_flag",
        "venue_win_rate_prior",
        "opponent_venue_win_rate_prior",
        "venue_win_rate_gap",
        "team_recent_home_win_rate_prior",
        "team_recent_away_win_rate_prior",
        "opponent_recent_home_win_rate_prior",
        "opponent_recent_away_win_rate_prior",
        "venue_context_win_rate_gap",
        "team_recent_same_venue_run_diff",
        "season_phase",
        "team_month_win_rate_prior",
        "opponent_month_win_rate_prior",
        "month_win_rate_gap",
        "team_month_run_diff_prior",
        "opponent_month_run_diff_prior",
        "month_run_diff_gap",
        "head_to_head_win_rate_prior",
        "opponent_head_to_head_win_rate_prior",
        "head_to_head_win_rate_gap",
        "previous_game_win",
        "previous_game_score",
        "previous_game_allowed",
        "previous_game_run_diff",
        "opponent_previous_game_win",
        "opponent_previous_game_score",
        "opponent_previous_game_allowed",
        "opponent_previous_game_run_diff",
        "previous_game_win_gap",
        "previous_game_score_gap",
        "previous_game_allowed_gap",
        "previous_game_run_diff_gap",
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
