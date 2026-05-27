from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LEAGUE_RUNS = 4.5


def build_team_game_rows(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy()
    df["is_home"] = df["home_away"].eq("H").astype(int)
    df["target_runs"] = df["score_team"].astype(float)
    df["runs_allowed"] = df["score_opp"].astype(float)
    df["target_win"] = (df["score_team"] > df["score_opp"]).astype(int)
    df["month"] = df["date"].dt.month
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
    if "ballpark" in df.columns:
        columns.append("ballpark")
    return df[columns].sort_values(["season", "date", "game_key", "is_home"]).reset_index(drop=True)


def _prior_season_averages(df: pd.DataFrame) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float], dict[int, float]]:
    team_runs = df.groupby(["team", "season"])["target_runs"].mean().to_dict()
    team_allowed = df.groupby(["team", "season"])["runs_allowed"].mean().to_dict()
    league_runs = df.groupby("season")["target_runs"].mean().to_dict()
    return team_runs, team_allowed, league_runs


def _run_prior(team: str, season: int, team_prior: dict[tuple[str, int], float], league_prior: dict[int, float]) -> float:
    previous_season = season - 1
    if (team, previous_season) in team_prior:
        return float(team_prior[(team, previous_season)])
    if previous_season in league_prior:
        return float(league_prior[previous_season])
    return DEFAULT_LEAGUE_RUNS


def add_pre_game_team_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    df = team_rows.copy().sort_values(["season", "team", "date", "game_key"]).reset_index(drop=True)
    team_run_prior, team_allowed_prior, league_run_prior = _prior_season_averages(df)
    feature_frames: list[pd.DataFrame] = []

    for (season, team), group in df.groupby(["season", "team"], sort=False):
        history_runs: list[float] = []
        history_allowed: list[float] = []
        history_wins: list[int] = []
        previous_date: pd.Timestamp | None = None
        records: list[dict[str, float | int]] = []

        for current_date, date_group in group.groupby("date", sort=True):
            runs_prior = _run_prior(team, int(season), team_run_prior, league_run_prior)
            allowed_prior = _run_prior(team, int(season), team_allowed_prior, league_run_prior)
            rest_days = 7 if previous_date is None else max(1, min((current_date - previous_date).days, 14))

            for _ in date_group.index:
                records.append(
                    {
                        "team_recent_5g_runs": float(np.mean(history_runs[-5:])) if history_runs else runs_prior,
                        "team_recent_10g_runs": float(np.mean(history_runs[-10:])) if history_runs else runs_prior,
                        "team_season_runs": float(np.mean(history_runs)) if history_runs else runs_prior,
                        "team_recent_5g_allowed": float(np.mean(history_allowed[-5:])) if history_allowed else allowed_prior,
                        "team_recent_10g_allowed": float(np.mean(history_allowed[-10:])) if history_allowed else allowed_prior,
                        "team_season_allowed": float(np.mean(history_allowed)) if history_allowed else allowed_prior,
                        "team_recent_win_rate": float(np.mean(history_wins[-10:])) if history_wins else 0.5,
                        "rest_days": rest_days,
                        "back_to_back": int(rest_days <= 1),
                    }
                )

            history_runs.extend(date_group["target_runs"].astype(float).tolist())
            history_allowed.extend(date_group["runs_allowed"].astype(float).tolist())
            history_wins.extend(date_group["target_win"].astype(int).tolist())
            previous_date = current_date

        enriched = group.copy()
        for column in records[0]:
            enriched[column] = [record[column] for record in records]
        feature_frames.append(enriched)

    return pd.concat(feature_frames, ignore_index=True).sort_values(["season", "date", "game_key", "is_home"]).reset_index(drop=True)


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    opponent = df[
        [
            "game_key",
            "team",
            "team_recent_5g_runs",
            "team_recent_10g_runs",
            "team_season_runs",
            "team_recent_5g_allowed",
            "team_recent_10g_allowed",
            "team_season_allowed",
            "team_recent_win_rate",
            "rest_days",
            "back_to_back",
        ]
    ].rename(
        columns={
            "team": "opponent",
            "team_recent_5g_runs": "opponent_recent_5g_runs",
            "team_recent_10g_runs": "opponent_recent_10g_runs",
            "team_season_runs": "opponent_season_runs",
            "team_recent_5g_allowed": "opponent_recent_5g_allowed",
            "team_recent_10g_allowed": "opponent_recent_10g_allowed",
            "team_season_allowed": "opponent_season_allowed",
            "team_recent_win_rate": "opponent_recent_win_rate",
            "rest_days": "opponent_rest_days",
            "back_to_back": "opponent_back_to_back",
        }
    )
    merged = df.merge(opponent, on=["game_key", "opponent"], how="inner")
    merged["recent_5g_runs_gap"] = merged["team_recent_5g_runs"] - merged["opponent_recent_5g_runs"]
    merged["recent_10g_runs_gap"] = merged["team_recent_10g_runs"] - merged["opponent_recent_10g_runs"]
    merged["season_runs_gap"] = merged["team_season_runs"] - merged["opponent_season_runs"]
    merged["recent_5g_allowed_gap"] = merged["opponent_recent_5g_allowed"] - merged["team_recent_5g_allowed"]
    merged["recent_10g_allowed_gap"] = merged["opponent_recent_10g_allowed"] - merged["team_recent_10g_allowed"]
    merged["season_allowed_gap"] = merged["opponent_season_allowed"] - merged["team_season_allowed"]
    merged["recent_win_rate_gap"] = merged["team_recent_win_rate"] - merged["opponent_recent_win_rate"]
    merged["rest_days_gap"] = merged["rest_days"] - merged["opponent_rest_days"]
    return merged.sort_values(["season", "date", "game_key", "is_home"]).reset_index(drop=True)


def build_feature_matrix(team_games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    team_rows = build_team_game_rows(team_games)
    features = add_pre_game_team_features(team_rows)
    features = add_opponent_features(features)
    feature_columns = [
        "is_home",
        "month",
        "team_recent_5g_runs",
        "team_recent_10g_runs",
        "team_season_runs",
        "team_recent_5g_allowed",
        "team_recent_10g_allowed",
        "team_season_allowed",
        "team_recent_win_rate",
        "rest_days",
        "back_to_back",
        "opponent_recent_5g_runs",
        "opponent_recent_10g_runs",
        "opponent_season_runs",
        "opponent_recent_5g_allowed",
        "opponent_recent_10g_allowed",
        "opponent_season_allowed",
        "opponent_recent_win_rate",
        "opponent_rest_days",
        "opponent_back_to_back",
        "recent_5g_runs_gap",
        "recent_10g_runs_gap",
        "season_runs_gap",
        "recent_5g_allowed_gap",
        "recent_10g_allowed_gap",
        "season_allowed_gap",
        "recent_win_rate_gap",
        "rest_days_gap",
    ]
    return features, feature_columns
