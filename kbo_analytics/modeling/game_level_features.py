from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FATIGUE_SCORE = {"낮음": 0.0, "보통": 1.0, "높음": 2.0}
NEUTRAL_STARTER_ERA = 4.5
NEUTRAL_STARTER_WHIP = 1.35


def _recent_team_games(working: pd.DataFrame, days: int) -> pd.Series:
    counts = pd.Series(0, index=working.index, dtype=float)
    for (_, _), team_rows in working.sort_values(["date_dt", "game_id"]).groupby(["season", "team"], sort=False):
        dates = team_rows["date_dt"]
        for idx, current_date in dates.items():
            counts.loc[idx] = int(((dates < current_date) & (dates >= current_date - pd.Timedelta(days=days))).sum())
    return counts


def build_game_level_frame(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    working = features.copy()
    working["date_dt"] = pd.to_datetime(working["date"])
    working["season"] = working["date_dt"].dt.year
    working["recent_3day_games"] = _recent_team_games(working, 3)
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
                "home_recent_5_runs_avg": round(float(home["avg_score_last_5"]), 4),
                "away_recent_5_runs_avg": round(float(away["avg_score_last_5"]), 4),
                "recent_5_runs_avg_gap": round(float(home["avg_score_last_5"] - away["avg_score_last_5"]), 4),
                "home_recent_5_allowed_avg": round(float(home["avg_allowed_last_5"]), 4),
                "away_recent_5_allowed_avg": round(float(away["avg_allowed_last_5"]), 4),
                "recent_5_allowed_avg_gap": round(float(away["avg_allowed_last_5"] - home["avg_allowed_last_5"]), 4),
                "recent_5_run_creation_gap": round(float(home["avg_score_last_5"] - away["avg_allowed_last_5"]), 4),
                "recent_10_run_creation_gap": round(float(home["avg_run_diff_last_10"] - away["avg_run_diff_last_10"]), 4),
                "recent_run_diff_10_gap": round(float(home["avg_run_diff_last_10"] - away["avg_run_diff_last_10"]), 4),
                "home_venue_win_rate": round(float(home["venue_win_rate_prior"]), 4),
                "away_venue_win_rate": round(float(away["venue_win_rate_prior"]), 4),
                "venue_win_rate_gap": round(float(home["venue_win_rate_prior"] - away["venue_win_rate_prior"]), 4),
                "home_games_last_7_days": int(home["games_last_7_days"]),
                "away_games_last_7_days": int(away["games_last_7_days"]),
                "games_last_7_days_gap": int(away["games_last_7_days"] - home["games_last_7_days"]),
                "home_recent_3day_games": int(home["recent_3day_games"]),
                "away_recent_3day_games": int(away["recent_3day_games"]),
                "recent_3day_games_gap": int(away["recent_3day_games"] - home["recent_3day_games"]),
                "home_rest_days": round(float(home["rest_days"]), 2),
                "away_rest_days": round(float(away["rest_days"]), 2),
                "rest_days_gap": round(float(home["rest_days"] - away["rest_days"]), 2),
                "home_bullpen_fatigue_proxy": round(float(home["games_last_7_days"] + home["back_to_back"] * 1.5), 2),
                "away_bullpen_fatigue_proxy": round(float(away["games_last_7_days"] + away["back_to_back"] * 1.5), 2),
                "home_bullpen_fatigue_score": round(float(home["recent_3day_games"] + home["back_to_back"] * 1.5), 2),
                "away_bullpen_fatigue_score": round(float(away["recent_3day_games"] + away["back_to_back"] * 1.5), 2),
                "bullpen_fatigue_score_gap": round(float((away["recent_3day_games"] + away["back_to_back"] * 1.5) - (home["recent_3day_games"] + home["back_to_back"] * 1.5)), 2),
                "bullpen_fatigue_gap": round(float((away["games_last_7_days"] + away["back_to_back"] * 1.5) - (home["games_last_7_days"] + home["back_to_back"] * 1.5)), 2),
            }
        )
    return pd.DataFrame(rows)


def _starter_quality(era: float, whip: float, info_quality: float) -> float:
    era_score = max(0.0, (6.0 - era) / 6.0) * 55.0
    whip_score = max(0.0, (2.0 - whip) / 2.0) * 45.0
    return round((era_score + whip_score) * max(info_quality, 0.0), 4)


def attach_pitching_context(frame: pd.DataFrame, pitching_context: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for side in ["home", "away"]:
        enriched[f"{side}_starter_era"] = NEUTRAL_STARTER_ERA
        enriched[f"{side}_starter_whip"] = NEUTRAL_STARTER_WHIP
        enriched[f"{side}_starter_info_quality"] = 0.0
        enriched[f"{side}_starter_source"] = "unknown"
        enriched[f"{side}_starter_quality_score"] = _starter_quality(NEUTRAL_STARTER_ERA, NEUTRAL_STARTER_WHIP, 0.0)

    if not pitching_context.empty:
        context = pitching_context.copy()
        context["date"] = pd.to_datetime(context["date"]).dt.strftime("%Y-%m-%d")
        context["starter_era"] = context["starter_era"].apply(lambda value: _to_float(value, NEUTRAL_STARTER_ERA))
        context["starter_whip"] = context["starter_whip"].apply(lambda value: _to_float(value, NEUTRAL_STARTER_WHIP))
        context["starter_info_quality"] = context["starter_info_quality"].apply(lambda value: _to_float(value, 0.0))
        context["bullpen_fatigue_score_from_label"] = context["bullpen_fatigue"].map(FATIGUE_SCORE).fillna(1.0)

        for side in ["home", "away"]:
            key = f"{side}_team"
            side_context = context.rename(
                columns={
                    "team": key,
                    "starter_era": f"{side}_starter_era",
                    "starter_whip": f"{side}_starter_whip",
                    "starter_info_quality": f"{side}_starter_info_quality",
                    "starter_source": f"{side}_starter_source",
                    "recent_3day_games": f"{side}_recent_3day_games",
                    "bullpen_fatigue_score_from_label": f"{side}_bullpen_fatigue_score",
                }
            )[
                [
                    "date",
                    key,
                    f"{side}_starter_era",
                    f"{side}_starter_whip",
                    f"{side}_starter_info_quality",
                    f"{side}_starter_source",
                    f"{side}_recent_3day_games",
                    f"{side}_bullpen_fatigue_score",
                ]
            ]
            enriched = enriched.merge(side_context, on=["date", key], how="left", suffixes=("", "_ctx"))
            for column in ["starter_era", "starter_whip", "starter_info_quality", "starter_source", "recent_3day_games", "bullpen_fatigue_score"]:
                base = f"{side}_{column}"
                ctx = f"{base}_ctx"
                if ctx in enriched:
                    enriched[base] = enriched[ctx].where(enriched[ctx].notna(), enriched[base])
                    enriched = enriched.drop(columns=[ctx])

    for side in ["home", "away"]:
        enriched[f"{side}_starter_quality_score"] = enriched.apply(
            lambda row: _starter_quality(float(row[f"{side}_starter_era"]), float(row[f"{side}_starter_whip"]), float(row[f"{side}_starter_info_quality"])),
            axis=1,
        )
    enriched["starter_era_gap"] = enriched["away_starter_era"] - enriched["home_starter_era"]
    enriched["starter_whip_gap"] = enriched["away_starter_whip"] - enriched["home_starter_whip"]
    enriched["starter_quality_gap"] = enriched["home_starter_quality_score"] - enriched["away_starter_quality_score"]
    enriched["both_starters_confirmed"] = ((enriched["home_starter_source"] == "confirmed") & (enriched["away_starter_source"] == "confirmed")).astype(int)
    enriched["partial_starter_confirmed"] = (
        ((enriched["home_starter_source"] == "confirmed") | (enriched["away_starter_source"] == "confirmed"))
        & (enriched["both_starters_confirmed"] == 0)
    ).astype(int)
    enriched["bullpen_fatigue_score_gap"] = enriched["away_bullpen_fatigue_score"] - enriched["home_bullpen_fatigue_score"]
    return enriched.fillna(
        {
            "home_starter_era": NEUTRAL_STARTER_ERA,
            "away_starter_era": NEUTRAL_STARTER_ERA,
            "home_starter_whip": NEUTRAL_STARTER_WHIP,
            "away_starter_whip": NEUTRAL_STARTER_WHIP,
            "home_starter_info_quality": 0.0,
            "away_starter_info_quality": 0.0,
            "home_starter_source": "unknown",
            "away_starter_source": "unknown",
        }
    )


def _to_float(value, default=0.0):
    try:
        text = str(value).replace(",", "").strip()
        if text in {"", "-", "nan"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _parse_innings(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == "-":
        return 0.0
    total = 0.0
    for part in text.split():
        if "/" in part:
            numerator, denominator = part.split("/", 1)
            total += float(numerator) / float(denominator)
        else:
            total += float(part)
    return total


def build_player_team_context(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    teams = sorted(set(hitters.get("팀", pd.Series(dtype=str))).union(set(pitchers.get("팀", pd.Series(dtype=str)))))
    for team in teams:
        team_hitters = hitters[hitters["팀"] == team].copy() if "팀" in hitters else pd.DataFrame()
        team_pitchers = pitchers[pitchers["팀"] == team].copy() if "팀" in pitchers else pd.DataFrame()

        if not team_hitters.empty:
            for column in ["OPS", "출루율", "장타율", "득점권타율", "타석"]:
                team_hitters[f"{column}_num"] = team_hitters[column].apply(_to_float) if column in team_hitters else 0.0
            eligible_hitters = team_hitters[team_hitters["타석_num"] >= 30].copy()
            if eligible_hitters.empty:
                eligible_hitters = team_hitters.copy()
            eligible_hitters["hitter_impact_score"] = (
                eligible_hitters["OPS_num"] * 55
                + eligible_hitters["출루율_num"] * 25
                + eligible_hitters["장타율_num"] * 20
                + eligible_hitters["득점권타율_num"] * 10
                + eligible_hitters["타석_num"].clip(upper=150) / 150 * 10
            )
            top_hitters = eligible_hitters.sort_values("hitter_impact_score", ascending=False).head(3)
            team_ops = team_hitters["OPS_num"].replace(0, np.nan).mean()
            top3_ops = top_hitters["OPS_num"].mean() if not top_hitters.empty else 0.0
            hitter_dependency = max(float(top3_ops - (team_ops or 0.0)), 0.0)
            top3_hitter_score = top_hitters["hitter_impact_score"].mean() if not top_hitters.empty else 0.0
        else:
            team_ops = top3_ops = hitter_dependency = top3_hitter_score = 0.0

        if not team_pitchers.empty:
            for column in ["ERA", "WHIP", "이닝", "탈삼진", "볼넷", "세이브", "홀드"]:
                if column == "이닝":
                    team_pitchers[f"{column}_num"] = team_pitchers[column].apply(_parse_innings) if column in team_pitchers else 0.0
                else:
                    team_pitchers[f"{column}_num"] = team_pitchers[column].apply(_to_float) if column in team_pitchers else 0.0
            eligible_pitchers = team_pitchers[team_pitchers["이닝_num"] >= 5].copy()
            if eligible_pitchers.empty:
                eligible_pitchers = team_pitchers.copy()
            kbb = eligible_pitchers["탈삼진_num"] / eligible_pitchers["볼넷_num"].replace(0, 1)
            eligible_pitchers["pitcher_impact_score"] = (
                (6 - eligible_pitchers["ERA_num"]).clip(lower=0) / 6 * 35
                + (2 - eligible_pitchers["WHIP_num"]).clip(lower=0) / 2 * 25
                + eligible_pitchers["이닝_num"].clip(upper=50) / 50 * 25
                + kbb.clip(upper=4) / 4 * 10
                + (eligible_pitchers["세이브_num"] + eligible_pitchers["홀드_num"]).clip(upper=15) / 15 * 5
            )
            top_pitchers = eligible_pitchers.sort_values("pitcher_impact_score", ascending=False).head(3)
            total_innings = team_pitchers["이닝_num"].sum()
            top3_innings_share = top_pitchers["이닝_num"].sum() / max(total_innings, 1)
            pitcher_core_score = top_pitchers["pitcher_impact_score"].mean() if not top_pitchers.empty else 0.0
        else:
            top3_innings_share = pitcher_core_score = 0.0

        rows.append(
            {
                "team": team,
                "top3_hitter_ops_avg": round(float(top3_ops), 4),
                "team_ops_avg": round(float(team_ops if not pd.isna(team_ops) else 0.0), 4),
                "hitter_dependency": round(float(hitter_dependency), 4),
                "top3_hitter_impact_score": round(float(top3_hitter_score), 4),
                "pitcher_core_score": round(float(pitcher_core_score), 4),
                "pitcher_dependency": round(float(top3_innings_share), 4),
            }
        )
    return pd.DataFrame(rows)


def attach_player_context(frame: pd.DataFrame, player_context: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or player_context.empty:
        return frame.copy()
    home_context = player_context.add_prefix("home_").rename(columns={"home_team": "home_team"})
    away_context = player_context.add_prefix("away_").rename(columns={"away_team": "away_team"})
    enriched = frame.merge(home_context, on="home_team", how="left").merge(away_context, on="away_team", how="left")
    for column in [
        "top3_hitter_ops_avg",
        "top3_hitter_impact_score",
        "hitter_dependency",
        "pitcher_core_score",
        "pitcher_dependency",
    ]:
        enriched[f"{column}_gap"] = enriched[f"home_{column}"].fillna(0) - enriched[f"away_{column}"].fillna(0)
    return enriched.fillna(0)


def export_game_level_dataset(features: pd.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_game_level_frame(features).to_csv(output_path, index=False, encoding="utf-8-sig")


def prepare_game_level_matrix(frame: pd.DataFrame):
    x = frame.drop(columns=["date", "game_id", "target_home_win"])
    dummy_columns = [column for column in ["home_team", "away_team", "home_starter_source", "away_starter_source"] if column in x.columns]
    x = pd.get_dummies(x, columns=dummy_columns, drop_first=False, dtype=float)
    y = frame["target_home_win"].to_numpy(dtype=float)
    return x, y


def align_game_level_matrix(frame: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series):
    x, _ = prepare_game_level_matrix(frame)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return (x - mean) / std.replace(0, 1)
