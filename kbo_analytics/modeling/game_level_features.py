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
    x = pd.get_dummies(x, columns=["home_team", "away_team"], drop_first=False, dtype=float)
    y = frame["target_home_win"].to_numpy(dtype=float)
    return x, y


def align_game_level_matrix(frame: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series):
    x, _ = prepare_game_level_matrix(frame)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return (x - mean) / std.replace(0, 1)
