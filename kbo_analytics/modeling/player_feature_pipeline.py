from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "player_feature_model.yaml"

TEAM_ALIASES = {
    "Samsung": "삼성",
    "Doosan": "두산",
    "Hanwha": "한화",
    "Lotte": "롯데",
    "Kiwoom": "키움",
    "KIA": "KIA",
    "KT": "KT",
    "LG": "LG",
    "NC": "NC",
    "SSG": "SSG",
}

FATIGUE_SCORE = {"낮음": 0.0, "보통": 1.0, "높음": 2.0}

BASELINE_FEATURES = [
    "season_win_rate_gap",
    "recent_5_win_rate_gap",
    "recent_run_diff_gap",
    "rest_days_gap",
]

PITCHING_FEATURES = [
    "starter_era_gap",
    "starter_whip_gap",
    "starter_info_quality_gap",
    "starter_info_quality_min",
    "starter_confirmed_gap",
    "bullpen_fatigue_gap",
    "recent_3day_games_gap",
    "starter_era_available",
    "starter_whip_available",
    "starter_recent_3_era_gap",
    "starter_recent_5_era_gap",
    "starter_k_per_bb_gap",
    "starter_rest_days_gap",
    "starter_recent_data_available",
    "bullpen_innings_last_1d_gap",
    "bullpen_innings_last_3d_gap",
    "bullpen_pitch_count_last_1d_gap",
    "bullpen_pitch_count_last_3d_gap",
    "bullpen_consecutive_usage_gap",
    "bullpen_available_arms_gap",
    "bullpen_usage_data_available",
]

LINEUP_FEATURES = [
    "lineup_weighted_war_gap",
    "lineup_confirmation_quality_gap",
    "lineup_data_available",
    "player_id_mapping_coverage_min",
    "top_order_strength_gap",
    "middle_order_power_gap",
    "bottom_order_strength_gap",
    "lineup_ops_gap",
    "lineup_obp_gap",
    "lineup_slg_gap",
    "lineup_recent_7_ops_gap",
    "lineup_recent_14_ops_gap",
    "lineup_recent_30_ops_gap",
    "lineup_player_stat_coverage_min",
]

GAME_FEATURE_COLUMNS = [
    "date",
    "official_game_id",
    "home_team",
    "away_team",
    "target_home_win",
    *BASELINE_FEATURES,
    *PITCHING_FEATURES,
    *LINEUP_FEATURES,
    "home_starter_name",
    "away_starter_name",
    "home_lineup_source",
    "away_lineup_source",
    "home_lineup_players",
    "away_lineup_players",
    "home_starter_era",
    "away_starter_era",
    "home_starter_whip",
    "away_starter_whip",
    "home_starter_recent_3_era",
    "away_starter_recent_3_era",
    "home_starter_rest_days",
    "away_starter_rest_days",
    "home_starter_info_quality",
    "away_starter_info_quality",
]


@dataclass(frozen=True)
class PlayerFeatureConfig:
    values: dict[str, Any]

    @property
    def minimum_comparable_games(self) -> int:
        return int(
            self.values["challenger_gate_thresholds"]["minimum_comparable_games"]
        )

    @property
    def lineup_weights(self) -> dict[int, float]:
        return {
            int(key): float(value)
            for key, value in self.values["lineup_position_weights"].items()
        }


def load_player_feature_config(path: str | Path = DEFAULT_CONFIG) -> PlayerFeatureConfig:
    return PlayerFeatureConfig(json.loads(Path(path).read_text(encoding="utf-8")))


def normalize_team(value: Any) -> str:
    text = str(value).strip()
    return TEAM_ALIASES.get(text, text)


def official_game_key(value: Any) -> str:
    text = str(value)
    return text.rsplit("_", 1)[0] if "_" in text else text


def local_naive_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp


def shrink_rate(observed: float, sample: float, prior: float, prior_sample: float) -> float:
    if sample < 0 or prior_sample <= 0:
        raise ValueError("sample sizes must be non-negative and prior_sample positive")
    return float((observed * sample + prior * prior_sample) / (sample + prior_sample))


def select_as_of_player_rows(
    player_stats: pd.DataFrame,
    prediction_reference_datetime: datetime,
) -> pd.DataFrame:
    frame = player_stats.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("player game rows contain invalid dates")
    cutoff = local_naive_timestamp(prediction_reference_datetime).normalize()
    selected = frame[frame["date"] < cutoff].copy()
    selected["team"] = selected["team"].map(normalize_team)
    selected["opponent"] = selected["opponent"].map(normalize_team)
    return selected.sort_values(["date", "game_id", "player_id"])


def validate_player_ids(
    player_stats: pd.DataFrame,
    roster_mapping: pd.DataFrame,
) -> dict[str, int | float]:
    duplicated = int(
        roster_mapping.duplicated(["player_id"], keep=False).sum()
        + player_stats.duplicated(["game_id", "player_id"], keep=False).sum()
    )
    known = set(roster_mapping["player_id"].dropna().astype(str))
    ids = player_stats["player_id"].dropna().astype(str)
    failures = int((~ids.isin(known)).sum())
    coverage = float(1 - failures / len(ids)) if len(ids) else 0.0
    return {
        "player_id_mapping_failure": failures,
        "duplicate_player_rows": duplicated,
        "player_id_mapping_coverage": round(coverage, 4),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def batter_rates(rows: pd.DataFrame, prior_pa: float = 60.0) -> dict[str, float]:
    plate_appearances = float(_numeric(rows, "plate_appearances").sum())
    at_bats = float(_numeric(rows, "at_bats").sum())
    hits = float(_numeric(rows, "hits").sum())
    doubles = float(_numeric(rows, "doubles").sum())
    triples = float(_numeric(rows, "triples").sum())
    home_runs = float(_numeric(rows, "home_runs").sum())
    walks = float(_numeric(rows, "walks").sum())
    strikeouts = float(_numeric(rows, "strikeouts").sum())
    total_bases = hits + doubles + 2 * triples + 3 * home_runs
    observed_obp = (hits + walks) / max(at_bats + walks, 1.0)
    observed_slg = total_bases / max(at_bats, 1.0)
    observed_avg = hits / max(at_bats, 1.0)
    obp = shrink_rate(observed_obp, plate_appearances, 0.330, prior_pa)
    slg = shrink_rate(observed_slg, plate_appearances, 0.400, prior_pa)
    return {
        "plate_appearances": plate_appearances,
        "avg": shrink_rate(observed_avg, plate_appearances, 0.260, prior_pa),
        "obp": obp,
        "slg": slg,
        "ops": obp + slg,
        "walk_rate": shrink_rate(
            walks / max(plate_appearances, 1.0), plate_appearances, 0.08, prior_pa
        ),
        "strikeout_rate": shrink_rate(
            strikeouts / max(plate_appearances, 1.0),
            plate_appearances,
            0.19,
            prior_pa,
        ),
    }


def pitcher_rates(rows: pd.DataFrame, prior_innings: float = 20.0) -> dict[str, float]:
    innings = float(_numeric(rows, "innings_pitched").sum())
    earned_runs = float(_numeric(rows, "earned_runs").sum())
    walks = float(_numeric(rows, "walks_allowed").sum())
    hits = float(_numeric(rows, "hits_allowed").sum())
    strikeouts = float(_numeric(rows, "strikeouts_pitched").sum())
    era = shrink_rate(earned_runs * 9 / max(innings, 1.0), innings, 4.50, prior_innings)
    whip = shrink_rate((walks + hits) / max(innings, 1.0), innings, 1.40, prior_innings)
    return {
        "innings": innings,
        "era": era,
        "whip": whip,
        "k_per_bb": strikeouts / max(walks, 1.0),
        "strikeout_rate": strikeouts / max(innings * 3, 1.0),
        "walk_rate": walks / max(innings * 3, 1.0),
    }


def recent_player_rates(
    rows: pd.DataFrame,
    player_id: str,
    reference_datetime: datetime,
    *,
    recent_games: int | None = None,
    recent_days: int | None = None,
) -> pd.DataFrame:
    selected = select_as_of_player_rows(rows, reference_datetime)
    selected = selected[selected["player_id"].astype(str).eq(str(player_id))]
    if recent_days is not None:
        start = local_naive_timestamp(reference_datetime).normalize() - pd.Timedelta(days=recent_days)
        selected = selected[selected["date"] >= start]
    if recent_games is not None:
        game_ids = selected[["date", "game_id"]].drop_duplicates().tail(recent_games)["game_id"]
        selected = selected[selected["game_id"].isin(game_ids)]
    return selected


def _team_prior_metrics(game_rows: pd.DataFrame) -> pd.DataFrame:
    frame = game_rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["team"] = frame["team"].map(normalize_team)
    frame["opponent"] = frame["opponent"].map(normalize_team)
    frame = frame[frame["status"].eq("Final")].sort_values(["date", "team", "game_id"])
    frame["win"] = frame["result"].eq("Win").astype(float)
    frame["run_diff"] = _numeric(frame, "score_team") - _numeric(frame, "score_opp")
    group = frame.groupby("team", group_keys=False)
    frame["season_win_rate_prior"] = (
        group["win"].transform(lambda values: values.shift(1).expanding().mean()).fillna(0.5)
    )
    frame["recent_5_win_rate"] = (
        group["win"].transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    )
    frame["recent_run_diff"] = (
        group["run_diff"].transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
    )
    frame["rest_days"] = group["date"].diff().dt.days.fillna(1).clip(lower=1)
    return frame


def _latest_snapshot_rows(
    frame: pd.DataFrame,
    reference_datetime: datetime,
    key: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["snapshot_time"] = pd.to_datetime(result["snapshot_time"], errors="coerce")
    if result["snapshot_time"].isna().any():
        raise ValueError("snapshot rows contain invalid snapshot_time")
    result = result[result["snapshot_time"] <= local_naive_timestamp(reference_datetime)].copy()
    return result.sort_values("snapshot_time").drop_duplicates(key, keep="last")


def _latest_lineup_rows(
    frame: pd.DataFrame,
    reference_datetime: datetime,
) -> pd.DataFrame:
    result = frame.copy()
    result["snapshot_time"] = pd.to_datetime(result["snapshot_time"], errors="coerce")
    if result["snapshot_time"].isna().any():
        raise ValueError("lineup rows contain invalid snapshot_time")
    result = result[result["snapshot_time"] <= local_naive_timestamp(reference_datetime)].copy()
    result["source_priority"] = result["lineup_source"].map(
        {"confirmed": 3, "manual": 3, "estimated": 2, "recent": 1}
    ).fillna(0)
    key = ["reference_date", "scheduled_game_id", "team", "batting_order"]
    result = result.sort_values([*key, "source_priority", "snapshot_time"])
    return result.drop_duplicates(key, keep="last").drop(columns="source_priority")


def _lineup_team_summary(
    lineup: pd.DataFrame,
    mapping: pd.DataFrame,
    player_stats: pd.DataFrame,
    weights: dict[int, float],
) -> pd.DataFrame:
    frame = lineup.copy()
    frame["team"] = frame["team"].map(normalize_team)
    map_frame = mapping.copy()
    map_frame["team_ko"] = map_frame["team_ko"].map(normalize_team)
    map_frame = map_frame[["team_ko", "player_name", "player_id"]].drop_duplicates()
    frame = frame.merge(
        map_frame,
        left_on=["team", "player"],
        right_on=["team_ko", "player_name"],
        how="left",
    )
    frame["batting_order"] = pd.to_numeric(frame["batting_order"], errors="coerce")
    frame["war"] = pd.to_numeric(frame["war"], errors="coerce")
    frame["order_weight"] = frame["batting_order"].map(weights).fillna(1.0)
    frame["weighted_war"] = frame["war"].fillna(0.0) * frame["order_weight"]
    frame["mapped"] = frame["player_id"].notna().astype(float)
    player_history = player_stats.copy()
    player_history["date"] = pd.to_datetime(player_history["date"], errors="coerce")
    ability_rows = []
    for lineup_row in frame.itertuples():
        game_date = pd.Timestamp(lineup_row.reference_date).normalize()
        history = player_history[
            player_history["player_id"].astype(str).eq(str(lineup_row.player_id))
            & player_history["date"].lt(game_date)
        ]
        season = batter_rates(history)
        values = {
            "lineup_ops": season["ops"],
            "lineup_obp": season["obp"],
            "lineup_slg": season["slg"],
            "lineup_player_stat_available": float(not history.empty),
        }
        for days in [7, 14, 30]:
            recent = history[history["date"] >= game_date - pd.Timedelta(days=days)]
            values[f"lineup_recent_{days}_ops"] = batter_rates(recent)["ops"]
        ability_rows.append(values)
    ability = pd.DataFrame(ability_rows, index=frame.index)
    frame = pd.concat([frame, ability], axis=1)
    for column in [
        "lineup_ops",
        "lineup_obp",
        "lineup_slg",
        "lineup_recent_7_ops",
        "lineup_recent_14_ops",
        "lineup_recent_30_ops",
    ]:
        frame[column] = frame[column] * frame["order_weight"]
    detail_rows = []
    for (game_id, team), group in frame.groupby(["scheduled_game_id", "team"]):
        players = []
        for row in group.sort_values("batting_order").itertuples():
            players.append(
                {
                    "player_name": row.player,
                    "player_id": row.player_id if pd.notna(row.player_id) else None,
                    "batting_order": int(row.batting_order),
                    "ops": float(row.lineup_ops / row.order_weight),
                    "obp": float(row.lineup_obp / row.order_weight),
                    "slg": float(row.lineup_slg / row.order_weight),
                    "recent_7_ops": float(row.lineup_recent_7_ops / row.order_weight),
                    "recent_14_ops": float(row.lineup_recent_14_ops / row.order_weight),
                    "recent_30_ops": float(row.lineup_recent_30_ops / row.order_weight),
                    "info_quality": float(row.lineup_info_quality),
                    "lineup_source": row.lineup_source,
                    "order_weight": float(row.order_weight),
                }
            )
        detail_rows.append(
            {
                "scheduled_game_id": game_id,
                "team": team,
                "lineup_players": json.dumps(players, ensure_ascii=False),
                "lineup_source": group.sort_values("snapshot_time").iloc[-1]["lineup_source"],
            }
        )
    frame["top"] = frame["weighted_war"].where(frame["batting_order"] <= 3, 0.0)
    frame["middle"] = frame["weighted_war"].where(frame["batting_order"].between(4, 6), 0.0)
    frame["bottom"] = frame["weighted_war"].where(frame["batting_order"] >= 7, 0.0)
    summary = frame.groupby(["scheduled_game_id", "team"], as_index=False).agg(
        lineup_weighted_war=("weighted_war", "sum"),
        lineup_confirmation_quality=("lineup_info_quality", "mean"),
        confirmed_starter_count=("batting_order", "count"),
        player_id_mapping_coverage=("mapped", "mean"),
        top_order_strength=("top", "sum"),
        middle_order_power=("middle", "sum"),
        bottom_order_strength=("bottom", "sum"),
        lineup_ops=("lineup_ops", "sum"),
        lineup_obp=("lineup_obp", "sum"),
        lineup_slg=("lineup_slg", "sum"),
        lineup_recent_7_ops=("lineup_recent_7_ops", "sum"),
        lineup_recent_14_ops=("lineup_recent_14_ops", "sum"),
        lineup_recent_30_ops=("lineup_recent_30_ops", "sum"),
        lineup_player_stat_coverage=("lineup_player_stat_available", "mean"),
        lineup_weight_total=("order_weight", "sum"),
    )
    for column in [
        "lineup_ops",
        "lineup_obp",
        "lineup_slg",
        "lineup_recent_7_ops",
        "lineup_recent_14_ops",
        "lineup_recent_30_ops",
    ]:
        summary[column] = summary[column] / summary["lineup_weight_total"]
    summary = summary.drop(columns="lineup_weight_total")
    summary = summary.merge(pd.DataFrame(detail_rows), on=["scheduled_game_id", "team"], how="left")
    summary["lineup_data_available"] = summary["confirmed_starter_count"].ge(9).astype(float)
    return summary


def _starter_history(
    player_stats: pd.DataFrame,
    roster_mapping: pd.DataFrame,
    team: str,
    starter_name: str,
    game_date: pd.Timestamp,
) -> dict[str, float]:
    mapping = roster_mapping.copy()
    mapping["team_ko"] = mapping["team_ko"].map(normalize_team)
    matched = mapping[
        mapping["team_ko"].eq(team)
        & mapping["player_name"].astype(str).eq(str(starter_name))
    ]
    if len(matched) != 1:
        return {
            "recent_3_era": 4.5,
            "recent_5_era": 4.5,
            "k_per_bb": 2.0,
            "rest_days": 5.0,
            "available": 0.0,
        }
    history = player_stats[
        player_stats["player_id"].astype(str).eq(str(matched.iloc[0]["player_id"]))
        & player_stats["date"].lt(game_date)
        & pd.to_numeric(player_stats["innings_pitched"], errors="coerce").gt(0)
    ].sort_values(["date", "game_id"])
    if history.empty:
        return {
            "recent_3_era": 4.5,
            "recent_5_era": 4.5,
            "k_per_bb": 2.0,
            "rest_days": 5.0,
            "available": 0.0,
        }
    recent_3 = history[history["game_id"].isin(history["game_id"].drop_duplicates().tail(3))]
    recent_5 = history[history["game_id"].isin(history["game_id"].drop_duplicates().tail(5))]
    season = pitcher_rates(history)
    return {
        "recent_3_era": pitcher_rates(recent_3)["era"],
        "recent_5_era": pitcher_rates(recent_5)["era"],
        "k_per_bb": season["k_per_bb"],
        "rest_days": float((game_date - history["date"].max()).days),
        "available": 1.0,
    }


def _bullpen_history(
    player_stats: pd.DataFrame,
    roster_mapping: pd.DataFrame,
    team: str,
    starter_name: str,
    game_date: pd.Timestamp,
) -> dict[str, float]:
    mapping = roster_mapping.copy()
    mapping["team_ko"] = mapping["team_ko"].map(normalize_team)
    pitcher_mapping = mapping[
        mapping["team_ko"].eq(team) & mapping["role"].eq("pitcher")
    ]
    pitcher_ids = set(pitcher_mapping["player_id"].astype(str))
    starter_ids = set(
        pitcher_mapping[pitcher_mapping["player_name"].astype(str).eq(str(starter_name))]["player_id"].astype(str)
    )
    bullpen_ids = pitcher_ids - starter_ids
    rows = player_stats[
        player_stats["player_id"].astype(str).isin(bullpen_ids)
        & player_stats["date"].lt(game_date)
        & player_stats["date"].ge(game_date - pd.Timedelta(days=3))
    ].copy()
    if rows.empty:
        return {
            "innings_1d": 0.0,
            "innings_3d": 0.0,
            "pitches_1d": 0.0,
            "pitches_3d": 0.0,
            "consecutive_usage": 0.0,
            "available_arms": float(len(bullpen_ids)),
            "available": 0.0,
        }
    yesterday = game_date - pd.Timedelta(days=1)
    previous = game_date - pd.Timedelta(days=2)
    one_day = rows[rows["date"].eq(yesterday)]
    used_yesterday = set(one_day["player_id"].astype(str))
    used_previous = set(rows[rows["date"].eq(previous)]["player_id"].astype(str))
    pitches_yesterday = one_day.groupby("player_id")["pitches"].sum()
    unavailable = set(pitches_yesterday[pitches_yesterday > 30].index.astype(str)) | (used_yesterday & used_previous)
    return {
        "innings_1d": float(pd.to_numeric(one_day["innings_pitched"], errors="coerce").sum()),
        "innings_3d": float(pd.to_numeric(rows["innings_pitched"], errors="coerce").sum()),
        "pitches_1d": float(pd.to_numeric(one_day["pitches"], errors="coerce").sum()),
        "pitches_3d": float(pd.to_numeric(rows["pitches"], errors="coerce").sum()),
        "consecutive_usage": float(len(used_yesterday & used_previous)),
        "available_arms": float(len(bullpen_ids - unavailable)),
        "available": 1.0,
    }


def build_snapshot_game_features(
    game_results: pd.DataFrame,
    pitching_snapshots: pd.DataFrame,
    lineup_snapshots: pd.DataFrame,
    roster_mapping: pd.DataFrame,
    reference_datetime: datetime,
    config: PlayerFeatureConfig,
    player_stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    games = _team_prior_metrics(game_results)
    games["official_game_id"] = games["game_id"].map(official_game_key)
    pitching = _latest_snapshot_rows(
        pitching_snapshots,
        reference_datetime,
        ["reference_date", "scheduled_game_id", "team"],
    )
    pitching["official_game_id"] = pitching["scheduled_game_id"].map(official_game_key)
    pitching["team"] = pitching["team"].map(normalize_team)
    pitching["starter_era"] = pd.to_numeric(pitching["starter_era"], errors="coerce")
    pitching["starter_whip"] = pd.to_numeric(pitching["starter_whip"], errors="coerce")
    pitching["starter_info_quality"] = pd.to_numeric(
        pitching["starter_info_quality"], errors="coerce"
    )
    pitching["bullpen_fatigue_score"] = pitching["bullpen_fatigue_label"].map(FATIGUE_SCORE)

    lineup = _latest_lineup_rows(lineup_snapshots, reference_datetime)
    stats = player_stats.copy() if player_stats is not None else pd.DataFrame(
        columns=["date", "game_id", "player_id", "innings_pitched", "pitches"]
    )
    stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
    lineup_summary = _lineup_team_summary(lineup, roster_mapping, stats, config.lineup_weights)
    lineup_summary["official_game_id"] = lineup_summary["scheduled_game_id"].map(official_game_key)

    rows: list[dict[str, Any]] = []
    for official_id, game in games.groupby("official_game_id"):
        if len(game) != 2:
            continue
        home = game[game["home_away"].eq("H")]
        away = game[game["home_away"].eq("A")]
        if len(home) != 1 or len(away) != 1:
            continue
        home_row, away_row = home.iloc[0], away.iloc[0]
        home_pitch = pitching[
            pitching["official_game_id"].eq(official_id)
            & pitching["team"].eq(home_row["team"])
        ]
        away_pitch = pitching[
            pitching["official_game_id"].eq(official_id)
            & pitching["team"].eq(away_row["team"])
        ]
        home_lineup = lineup_summary[
            lineup_summary["official_game_id"].eq(official_id)
            & lineup_summary["team"].eq(home_row["team"])
        ]
        away_lineup = lineup_summary[
            lineup_summary["official_game_id"].eq(official_id)
            & lineup_summary["team"].eq(away_row["team"])
        ]
        if any(frame.empty for frame in [home_pitch, away_pitch, home_lineup, away_lineup]):
            continue
        hp, ap = home_pitch.iloc[0], away_pitch.iloc[0]
        hl, al = home_lineup.iloc[0], away_lineup.iloc[0]
        game_date = pd.Timestamp(home_row["date"]).normalize()
        home_starter = _starter_history(stats, roster_mapping, home_row["team"], hp["starter_name"], game_date)
        away_starter = _starter_history(stats, roster_mapping, away_row["team"], ap["starter_name"], game_date)
        home_bullpen = _bullpen_history(stats, roster_mapping, home_row["team"], hp["starter_name"], game_date)
        away_bullpen = _bullpen_history(stats, roster_mapping, away_row["team"], ap["starter_name"], game_date)
        era_available = float(pd.notna(hp["starter_era"]) and pd.notna(ap["starter_era"]))
        whip_available = float(pd.notna(hp["starter_whip"]) and pd.notna(ap["starter_whip"]))
        rows.append(
            {
                "date": home_row["date"],
                "official_game_id": official_id,
                "home_team": home_row["team"],
                "away_team": away_row["team"],
                "target_home_win": int(home_row["result"] == "Win"),
                "season_win_rate_gap": home_row["season_win_rate_prior"] - away_row["season_win_rate_prior"],
                "recent_5_win_rate_gap": home_row["recent_5_win_rate"] - away_row["recent_5_win_rate"],
                "recent_run_diff_gap": home_row["recent_run_diff"] - away_row["recent_run_diff"],
                "rest_days_gap": home_row["rest_days"] - away_row["rest_days"],
                "starter_era_gap": (ap["starter_era"] - hp["starter_era"]) if era_available else 0.0,
                "starter_whip_gap": (ap["starter_whip"] - hp["starter_whip"]) if whip_available else 0.0,
                "starter_info_quality_gap": hp["starter_info_quality"] - ap["starter_info_quality"],
                "starter_info_quality_min": min(hp["starter_info_quality"], ap["starter_info_quality"]),
                "starter_confirmed_gap": float(hp["starter_source"] == "confirmed") - float(ap["starter_source"] == "confirmed"),
                "bullpen_fatigue_gap": ap["bullpen_fatigue_score"] - hp["bullpen_fatigue_score"],
                "recent_3day_games_gap": float(ap["recent_3day_games"]) - float(hp["recent_3day_games"]),
                "starter_era_available": era_available,
                "starter_whip_available": whip_available,
                "starter_recent_3_era_gap": away_starter["recent_3_era"] - home_starter["recent_3_era"],
                "starter_recent_5_era_gap": away_starter["recent_5_era"] - home_starter["recent_5_era"],
                "starter_k_per_bb_gap": home_starter["k_per_bb"] - away_starter["k_per_bb"],
                "starter_rest_days_gap": home_starter["rest_days"] - away_starter["rest_days"],
                "starter_recent_data_available": min(home_starter["available"], away_starter["available"]),
                "bullpen_innings_last_1d_gap": away_bullpen["innings_1d"] - home_bullpen["innings_1d"],
                "bullpen_innings_last_3d_gap": away_bullpen["innings_3d"] - home_bullpen["innings_3d"],
                "bullpen_pitch_count_last_1d_gap": away_bullpen["pitches_1d"] - home_bullpen["pitches_1d"],
                "bullpen_pitch_count_last_3d_gap": away_bullpen["pitches_3d"] - home_bullpen["pitches_3d"],
                "bullpen_consecutive_usage_gap": away_bullpen["consecutive_usage"] - home_bullpen["consecutive_usage"],
                "bullpen_available_arms_gap": home_bullpen["available_arms"] - away_bullpen["available_arms"],
                "bullpen_usage_data_available": min(home_bullpen["available"], away_bullpen["available"]),
                "lineup_weighted_war_gap": hl["lineup_weighted_war"] - al["lineup_weighted_war"],
                "lineup_confirmation_quality_gap": hl["lineup_confirmation_quality"] - al["lineup_confirmation_quality"],
                "lineup_data_available": min(hl["lineup_data_available"], al["lineup_data_available"]),
                "player_id_mapping_coverage_min": min(hl["player_id_mapping_coverage"], al["player_id_mapping_coverage"]),
                "top_order_strength_gap": hl["top_order_strength"] - al["top_order_strength"],
                "middle_order_power_gap": hl["middle_order_power"] - al["middle_order_power"],
                "bottom_order_strength_gap": hl["bottom_order_strength"] - al["bottom_order_strength"],
                "lineup_ops_gap": hl["lineup_ops"] - al["lineup_ops"],
                "lineup_obp_gap": hl["lineup_obp"] - al["lineup_obp"],
                "lineup_slg_gap": hl["lineup_slg"] - al["lineup_slg"],
                "lineup_recent_7_ops_gap": hl["lineup_recent_7_ops"] - al["lineup_recent_7_ops"],
                "lineup_recent_14_ops_gap": hl["lineup_recent_14_ops"] - al["lineup_recent_14_ops"],
                "lineup_recent_30_ops_gap": hl["lineup_recent_30_ops"] - al["lineup_recent_30_ops"],
                "lineup_player_stat_coverage_min": min(hl["lineup_player_stat_coverage"], al["lineup_player_stat_coverage"]),
                "home_starter_name": hp["starter_name"],
                "away_starter_name": ap["starter_name"],
                "home_lineup_source": home_lineup.iloc[0].get("lineup_source", "confirmed"),
                "away_lineup_source": away_lineup.iloc[0].get("lineup_source", "confirmed"),
                "home_lineup_players": hl["lineup_players"],
                "away_lineup_players": al["lineup_players"],
                "home_starter_era": hp["starter_era"],
                "away_starter_era": ap["starter_era"],
                "home_starter_whip": hp["starter_whip"],
                "away_starter_whip": ap["starter_whip"],
                "home_starter_recent_3_era": home_starter["recent_3_era"],
                "away_starter_recent_3_era": away_starter["recent_3_era"],
                "home_starter_rest_days": home_starter["rest_days"],
                "away_starter_rest_days": away_starter["rest_days"],
                "home_starter_info_quality": hp["starter_info_quality"],
                "away_starter_info_quality": ap["starter_info_quality"],
            }
        )
    features = pd.DataFrame(rows, columns=GAME_FEATURE_COLUMNS)
    target_game_ids = set(games["official_game_id"])
    all_game_ids = set(pitching.groupby("official_game_id").filter(lambda rows: rows["team"].nunique() == 2)["official_game_id"])
    lineup_game_ids = set(lineup_summary.groupby("official_game_id").filter(lambda rows: rows["team"].nunique() == 2)["official_game_id"])
    comparable = set(features.get("official_game_id", pd.Series(dtype=str)))
    coverage = {
        "pitching_games": len(all_game_ids),
        "lineup_games": len(lineup_game_ids),
        "comparable_games": len(comparable),
        "target_games": len(target_game_ids),
        "starter_coverage": round(len(target_game_ids & all_game_ids) / max(len(target_game_ids), 1), 4),
        "lineup_coverage": round(len(target_game_ids & lineup_game_ids) / max(len(target_game_ids), 1), 4),
        "bullpen_coverage": round(len(target_game_ids & all_game_ids) / max(len(target_game_ids), 1), 4),
        "starter_recent_data_coverage": round(float(features["starter_recent_data_available"].mean()), 4) if not features.empty else 0.0,
        "bullpen_usage_data_coverage": round(float(features["bullpen_usage_data_available"].mean()), 4) if not features.empty else 0.0,
        "lineup_player_stat_coverage": round(float(features["lineup_player_stat_coverage_min"].mean()), 4) if not features.empty else 0.0,
        "player_id_mapping_coverage": (
            round(float(features["player_id_mapping_coverage_min"].mean()), 4)
            if not features.empty
            else 0.0
        ),
        "lineup_confirmation_coverage": (
            round(float(lineup["lineup_source"].eq("confirmed").mean()), 4)
            if not lineup.empty
            else 0.0
        ),
    }
    return features.sort_values(["date", "official_game_id"]), coverage


def player_feature_leakage_audit(
    player_stats: pd.DataFrame,
    pitching_snapshots: pd.DataFrame,
    lineup_snapshots: pd.DataFrame,
    reference_datetime: datetime,
) -> dict[str, Any]:
    stat_dates = pd.to_datetime(player_stats["date"], errors="coerce")
    pitching_times = pd.to_datetime(pitching_snapshots["snapshot_time"], errors="coerce")
    lineup_times = pd.to_datetime(lineup_snapshots["snapshot_time"], errors="coerce")
    reference = local_naive_timestamp(reference_datetime)
    future_stats = int((stat_dates >= reference.normalize()).sum())
    future_pitching = int((pitching_times > reference).sum())
    future_lineup = int((lineup_times > reference).sum())
    duplicate_players = int(
        player_stats.duplicated(["game_id", "player_id"], keep=False).sum()
    )
    canonical_pitching_duplicates = int(
        pitching_snapshots.duplicated(
            ["reference_date", "scheduled_game_id", "team"], keep=False
        ).sum()
    )
    canonical_lineup_duplicates = int(
        lineup_snapshots.duplicated(
            ["reference_date", "scheduled_game_id", "team", "batting_order", "snapshot_time"],
            keep=False,
        ).sum()
    )
    official_id_pattern = r"^\d{8}[A-Z]{4}\d$"
    official_mapping_failures = int(
        (~pitching_snapshots["scheduled_game_id"].map(official_game_key).str.match(official_id_pattern)).sum()
        + (~lineup_snapshots["scheduled_game_id"].map(official_game_key).str.match(official_id_pattern)).sum()
    )
    checks = {
        "future_stat_rows": future_stats,
        "future_pitching_snapshot_rows": future_pitching,
        "future_lineup_snapshot_rows": future_lineup,
        "duplicate_player_rows": duplicate_players,
        "canonical_duplicates": canonical_pitching_duplicates + canonical_lineup_duplicates,
        "official_game_id_mapping_failures": official_mapping_failures,
        "missing_stat_timestamp_rows": int(len(player_stats)),
        "post_start_snapshot_rows": None,
        "post_start_verification": "pitching canonical storage guard; lineup start time not available",
        "latest_season_snapshot_used_historically": False,
        "same_day_results_used": False,
        "player_name_used_as_model_feature": False,
    }
    blocking = [
        name
        for name in [
            "future_stat_rows",
            "future_pitching_snapshot_rows",
            "future_lineup_snapshot_rows",
            "duplicate_player_rows",
            "canonical_duplicates",
            "official_game_id_mapping_failures",
        ]
        if checks[name] != 0
    ]
    if checks["post_start_verification"].endswith("not available"):
        blocking.append("lineup_start_time_not_verifiable")
    return {
        "status": "pass" if not blocking else "blocked",
        "blocking_issues": blocking,
        "checks": checks,
        "policy_note": (
            "Date-only player game logs are treated as end-of-game records and become "
            "eligible starting the next calendar day. Current cumulative hitter/pitcher "
            "snapshots are excluded from historical training."
        ),
    }


def player_feature_schema() -> dict[str, Any]:
    unavailable = [
        "starter_recent_pitch_count",
        "starter_expected_innings",
        "starter_home_away_split",
        "starter_vs_opponent_split",
        "starter_platoon_matchup",
        "bullpen_available_arms",
        "closer_available",
        "setup_available",
        "lineup_vs_lhp_strength",
        "lineup_vs_rhp_strength",
        "missing_regular_count",
        "replacement_quality_delta",
        "WPA",
    ]
    return {
        "schema_version": 1,
        "baseline_features": BASELINE_FEATURES,
        "pitching_features": PITCHING_FEATURES,
        "lineup_features": LINEUP_FEATURES,
        "model_input_excludes": ["player_name", "starter_name", "player_id"],
        "unavailable_features": unavailable,
        "as_of_contract": {
            "player_game_stats": "date < prediction_reference_date",
            "pitching_snapshot": "snapshot_time <= prediction_reference_datetime and canonical pre-start guard",
            "lineup_snapshot": "snapshot_time <= prediction_reference_datetime",
        },
    }
