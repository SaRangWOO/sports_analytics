from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .player_feature_pipeline import (
    BASELINE_FEATURES,
    LINEUP_FEATURES,
    PITCHING_FEATURES,
    PlayerFeatureConfig,
)


STARTER_GROUP = [
    "starter_era_gap",
    "starter_whip_gap",
    "starter_info_quality_gap",
    "starter_confirmed_gap",
    "starter_recent_3_era_gap",
    "starter_recent_5_era_gap",
    "starter_k_per_bb_gap",
    "starter_rest_days_gap",
]
BULLPEN_GROUP = [
    "bullpen_fatigue_gap",
    "recent_3day_games_gap",
    "bullpen_innings_last_1d_gap",
    "bullpen_innings_last_3d_gap",
    "bullpen_pitch_count_last_1d_gap",
    "bullpen_pitch_count_last_3d_gap",
    "bullpen_consecutive_usage_gap",
    "bullpen_available_arms_gap",
]
AVAILABILITY_GROUP = [
    "starter_info_quality_min",
    "starter_era_available",
    "starter_whip_available",
    "lineup_data_available",
    "player_id_mapping_coverage_min",
    "starter_recent_data_available",
    "bullpen_usage_data_available",
    "lineup_player_stat_coverage_min",
    "lineup_confirmation_quality_gap",
]

METRIC_COLUMNS = [
    "model",
    "feature_set",
    "games",
    "accuracy",
    "brier_score",
    "log_loss",
    "calibration_error",
    "high_confidence_games",
    "high_confidence_accuracy",
    "recent_30_accuracy",
    "recent_60_accuracy",
    "home_win_accuracy",
    "prediction_failure_count",
]

CONTRIBUTION_COLUMNS = [
    "date",
    "official_game_id",
    "home_team",
    "away_team",
    "baseline_win_probability",
    "starter_adjusted_probability",
    "pitcher_adjusted_probability",
    "lineup_adjusted_probability",
    "final_challenger_probability",
    "starter_contribution_pct_point",
    "bullpen_contribution_pct_point",
    "lineup_contribution_pct_point",
    "availability_contribution_pct_point",
    "contribution_sum_pct_point",
    "probability_delta_pct_point",
    "contribution_method",
    "causal_effect",
    "home_starter_name",
    "away_starter_name",
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
    "top_positive_players",
    "top_negative_players",
    "player_contribution_method",
]


@dataclass
class FittedChallenger:
    name: str
    feature_set: str
    columns: list[str]
    model: Any
    neutral_values: dict[str, float]


def _models() -> dict[str, Any]:
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=10,
            l2_regularization=0.15,
            random_state=42,
        ),
    }


def feature_sets() -> dict[str, list[str]]:
    return {
        "production_baseline_proxy": BASELINE_FEATURES,
        "baseline_plus_pitching": BASELINE_FEATURES + PITCHING_FEATURES,
        "baseline_plus_full_player": BASELINE_FEATURES + PITCHING_FEATURES + LINEUP_FEATURES,
    }


def expanding_splits(frame: pd.DataFrame, minimum_train_games: int = 20) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = frame.sort_values(["date", "official_game_id"]).reset_index()
    count = len(ordered)
    if count <= minimum_train_games:
        return []
    remaining = count - minimum_train_games
    block = max(1, remaining // 3)
    splits = []
    start = minimum_train_games
    while start < count:
        stop = min(count, start + block)
        train_index = ordered.iloc[:start]["index"].to_numpy()
        test_index = ordered.iloc[start:stop]["index"].to_numpy()
        if len(test_index):
            splits.append((train_index, test_index))
        start = stop
    return splits


def _matrix(frame: pd.DataFrame, columns: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    matrix = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    values = medians if medians is not None else matrix.median().fillna(0.0)
    return matrix.fillna(values).fillna(0.0), values


def calibration_error(y_true: np.ndarray, probability: np.ndarray) -> float:
    bins = pd.cut(probability, bins=np.linspace(0, 1, 6), include_lowest=True)
    total = len(y_true)
    error = 0.0
    for bucket in pd.Series(range(total)).groupby(bins, observed=False).groups.values():
        indexes = np.asarray(list(bucket), dtype=int)
        if len(indexes):
            error += len(indexes) / total * abs(
                float(probability[indexes].mean()) - float(y_true[indexes].mean())
            )
    return float(error)


def metric_row(
    model: str,
    feature_set: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1 - probability)
    high = confidence >= 0.55
    latest = pd.to_datetime(frame["date"]).max()

    def window_accuracy(days: int) -> float | None:
        mask = pd.to_datetime(frame["date"]) >= latest - pd.Timedelta(days=days)
        return round(float((prediction[mask] == y_true[mask]).mean()), 4) if mask.any() else None

    return {
        "model": model,
        "feature_set": feature_set,
        "games": int(len(frame)),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 4),
        "log_loss": round(float(log_loss(y_true, probability, labels=[0, 1])), 4),
        "calibration_error": round(calibration_error(y_true, probability), 4),
        "high_confidence_games": int(high.sum()),
        "high_confidence_accuracy": (
            round(float((prediction[high] == y_true[high]).mean()), 4)
            if high.any()
            else None
        ),
        "recent_30_accuracy": window_accuracy(30),
        "recent_60_accuracy": window_accuracy(60),
        "home_win_accuracy": round(float((prediction == y_true).mean()), 4),
        "prediction_failure_count": 0,
    }


def evaluate_challengers(
    frame: pd.DataFrame,
    minimum_train_games: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "official_game_id", "target_home_win"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing challenger columns: {sorted(missing)}")
    clean = frame.sort_values(["date", "official_game_id"]).reset_index(drop=True)
    splits = expanding_splits(clean, minimum_train_games)
    if not splits:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for feature_set, columns in feature_sets().items():
        for model_name in _models():
            probabilities = np.full(len(clean), np.nan)
            for train_index, test_index in splits:
                train = clean.loc[train_index]
                test = clean.loc[test_index]
                if train["target_home_win"].nunique() < 2:
                    continue
                train_matrix, medians = _matrix(train, columns)
                test_matrix, _ = _matrix(test, columns, medians)
                model = _models()[model_name]
                model.fit(train_matrix, train["target_home_win"].astype(int))
                probabilities[test_index] = model.predict_proba(test_matrix)[:, 1]
            mask = np.isfinite(probabilities)
            if not mask.any():
                continue
            evaluated = clean.loc[mask].copy()
            values = probabilities[mask]
            rows.append(
                metric_row(
                    model_name,
                    feature_set,
                    evaluated,
                    evaluated["target_home_win"].astype(int).to_numpy(),
                    values,
                )
            )
            for (_, game), probability in zip(evaluated.iterrows(), values):
                prediction_rows.append(
                    {
                        "date": game["date"],
                        "official_game_id": game["official_game_id"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "target_home_win": int(game["target_home_win"]),
                        "model": model_name,
                        "feature_set": feature_set,
                        "home_win_probability": float(probability),
                    }
                )
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    predictions = pd.DataFrame(prediction_rows)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["feature_set", "brier_score", "log_loss", "accuracy"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
    return metrics, predictions


def fit_full_challenger(frame: pd.DataFrame) -> FittedChallenger:
    columns = feature_sets()["baseline_plus_full_player"]
    matrix, medians = _matrix(frame, columns)
    target = frame["target_home_win"].astype(int)
    if len(frame) < 20 or target.nunique() < 2:
        raise ValueError("at least 20 games and both classes are required")
    model = _models()["LogisticRegression"]
    model.fit(matrix, target)
    return FittedChallenger(
        name="LogisticRegression",
        feature_set="baseline_plus_full_player",
        columns=columns,
        model=model,
        neutral_values={key: float(value) for key, value in medians.items()},
    )


def _probability(fitted: FittedChallenger, frame: pd.DataFrame) -> np.ndarray:
    matrix, _ = _matrix(frame, fitted.columns, pd.Series(fitted.neutral_values))
    probability = fitted.model.predict_proba(matrix)[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("challenger probability is outside [0, 1]")
    return probability


def contribution_report(fitted: FittedChallenger, frame: pd.DataFrame) -> pd.DataFrame:
    full = frame.copy()
    neutral = full.copy()
    player_columns = PITCHING_FEATURES + LINEUP_FEATURES
    for column in player_columns:
        neutral[column] = fitted.neutral_values.get(column, 0.0)
    starter = neutral.copy()
    for column in STARTER_GROUP:
        if column in full:
            starter[column] = full[column]
    bullpen = starter.copy()
    for column in BULLPEN_GROUP:
        if column in full:
            bullpen[column] = full[column]
    lineup = bullpen.copy()
    for column in [value for value in LINEUP_FEATURES if value not in AVAILABILITY_GROUP]:
        if column in full:
            lineup[column] = full[column]
    availability = lineup.copy()
    for column in AVAILABILITY_GROUP:
        if column in full:
            availability[column] = full[column]
    baseline_probability = _probability(fitted, neutral)
    starter_probability = _probability(fitted, starter)
    pitcher_probability = _probability(fitted, bullpen)
    lineup_probability = _probability(fitted, lineup)
    final_probability = _probability(fitted, availability)
    report = frame[["date", "official_game_id", "home_team", "away_team"]].copy()
    report["baseline_win_probability"] = baseline_probability
    report["starter_adjusted_probability"] = starter_probability
    report["pitcher_adjusted_probability"] = pitcher_probability
    report["lineup_adjusted_probability"] = lineup_probability
    report["final_challenger_probability"] = final_probability
    report["starter_contribution_pct_point"] = (starter_probability - baseline_probability) * 100
    report["bullpen_contribution_pct_point"] = (pitcher_probability - starter_probability) * 100
    report["lineup_contribution_pct_point"] = (lineup_probability - pitcher_probability) * 100
    report["availability_contribution_pct_point"] = (final_probability - lineup_probability) * 100
    report["contribution_sum_pct_point"] = (
        report["starter_contribution_pct_point"]
        + report["bullpen_contribution_pct_point"]
        + report["lineup_contribution_pct_point"]
        + report["availability_contribution_pct_point"]
    )
    report["probability_delta_pct_point"] = (
        report["final_challenger_probability"] - report["baseline_win_probability"]
    ) * 100
    report["contribution_method"] = "sequential_feature_group_ablation"
    report["causal_effect"] = False
    detail_columns = [
        "home_starter_name",
        "away_starter_name",
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
    for column in detail_columns:
        report[column] = frame[column].values if column in frame else None
    positive, negative = _player_contribution_rankings(fitted, frame, final_probability)
    report["top_positive_players"] = positive
    report["top_negative_players"] = negative
    report["player_contribution_method"] = "lineup_player_leave_one_out_to_league_prior"
    return report


def _player_contribution_rankings(
    fitted: FittedChallenger,
    frame: pd.DataFrame,
    full_probability: np.ndarray,
) -> tuple[list[str], list[str]]:
    priors = {
        "ops": 0.730,
        "obp": 0.330,
        "slg": 0.400,
        "recent_7_ops": 0.730,
        "recent_14_ops": 0.730,
        "recent_30_ops": 0.730,
    }
    feature_by_metric = {
        "ops": "lineup_ops_gap",
        "obp": "lineup_obp_gap",
        "slg": "lineup_slg_gap",
        "recent_7_ops": "lineup_recent_7_ops_gap",
        "recent_14_ops": "lineup_recent_14_ops_gap",
        "recent_30_ops": "lineup_recent_30_ops_gap",
    }
    positive_rows: list[str] = []
    negative_rows: list[str] = []
    for position, (_, game) in enumerate(frame.iterrows()):
        impacts = []
        for side, sign in [("home", 1.0), ("away", -1.0)]:
            raw = game.get(f"{side}_lineup_players")
            players = json.loads(raw) if isinstance(raw, str) and raw else []
            total_weight = sum(float(player["order_weight"]) for player in players) or 1.0
            for player in players:
                counterfactual = frame.iloc[[position]].copy()
                weight = float(player["order_weight"]) / total_weight
                for metric, feature in feature_by_metric.items():
                    if feature in counterfactual:
                        delta = (float(player[metric]) - priors[metric]) * weight
                        counterfactual.loc[:, feature] = float(game[feature]) - sign * delta
                probability = float(_probability(fitted, counterfactual)[0])
                impacts.append(
                    {
                        "player_name": player["player_name"],
                        "team": game[f"{side}_team"],
                        "batting_order": player["batting_order"],
                        "recent_ops": player["recent_7_ops"],
                        "lineup_source": player["lineup_source"],
                        "contribution_pct_point": round((full_probability[position] - probability) * 100, 4),
                    }
                )
        ordered = sorted(impacts, key=lambda value: value["contribution_pct_point"], reverse=True)
        positive_rows.append(json.dumps(ordered[:3], ensure_ascii=False))
        negative_rows.append(json.dumps(list(reversed(ordered[-3:])), ensure_ascii=False))
    return positive_rows, negative_rows


def challenger_gate(
    metrics: pd.DataFrame,
    coverage: dict[str, Any],
    leakage: dict[str, Any],
    config: PlayerFeatureConfig,
    shadow_pass_count: int = 0,
    predictions: pd.DataFrame | None = None,
    production_parity_verified: bool = False,
) -> dict[str, Any]:
    thresholds = config.values["challenger_gate_thresholds"]
    coverage_thresholds = config.values["coverage_thresholds"]
    checks = {
        "minimum_comparable_games": coverage.get("comparable_games", 0)
        >= int(thresholds["minimum_comparable_games"]),
        "starter_coverage": coverage.get("starter_coverage", 0)
        >= float(coverage_thresholds["starter"]),
        "bullpen_coverage": coverage.get("bullpen_coverage", 0)
        >= float(coverage_thresholds["bullpen"]),
        "lineup_coverage": coverage.get("lineup_coverage", 0)
        >= float(coverage_thresholds["lineup"]),
        "player_id_mapping_coverage": coverage.get("player_id_mapping_coverage", 0)
        >= float(coverage_thresholds["player_id_mapping"]),
        "leakage_audit": leakage.get("status") == "pass",
        "shadow_passes": shadow_pass_count
        >= int(thresholds["minimum_shadow_passes"]),
        "same_game_set": False,
        "performance_improved": False,
        "production_parity_verified": production_parity_verified,
    }
    if not metrics.empty:
        best = metrics.sort_values(["brier_score", "log_loss"]).groupby("feature_set").first()
        if {"production_baseline_proxy", "baseline_plus_full_player"}.issubset(best.index):
            baseline = best.loc["production_baseline_proxy"]
            full = best.loc["baseline_plus_full_player"]
            checks["same_game_set"] = int(baseline["games"]) == int(full["games"])
            checks["performance_improved"] = bool(
                full["accuracy"] - baseline["accuracy"]
                >= float(thresholds["minimum_accuracy_delta"])
                and full["brier_score"] < baseline["brier_score"]
                and full["log_loss"] < baseline["log_loss"]
                and full["calibration_error"] <= baseline["calibration_error"]
            )
    if predictions is not None and not predictions.empty:
        game_sets = [
            set(group["official_game_id"].astype(str))
            for _, group in predictions.groupby("feature_set")
        ]
        checks["same_game_set"] = bool(game_sets) and all(
            game_set == game_sets[0] for game_set in game_sets[1:]
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "decision": "shadow_candidate_eligible" if all(checks.values()) else "blocked",
        "production_model_changed": False,
        "auto_promotion_enabled": False,
    }


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        os.close(descriptor)
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        pd.read_csv(temporary, nrows=1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
