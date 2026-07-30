from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from modeling.model_evaluation import normalize_game_probabilities
from modeling.model_training import compact_feature_columns
from modeling.train_win_predictor import prepare_matrix, standardize_train_test


PITCHING_FEATURES = [
    "team_starter_info_quality",
    "opponent_starter_info_quality",
    "both_starters_confirmed",
    "team_starter_era_snapshot",
    "opponent_starter_era_snapshot",
    "starter_era_gap",
    "team_starter_whip_snapshot",
    "opponent_starter_whip_snapshot",
    "starter_whip_gap",
    "team_bullpen_fatigue_label_encoded",
    "opponent_bullpen_fatigue_label_encoded",
    "bullpen_fatigue_gap",
    "team_recent_3day_games_snapshot",
    "opponent_recent_3day_games_snapshot",
    "recent_3day_games_snapshot_gap",
]

FATIGUE_SCORE = {"낮음": 0.0, "보통": 1.0, "높음": 2.0}


def conservative_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.025,
        max_leaf_nodes=10,
        l2_regularization=0.15,
        random_state=42,
    )


def attach_snapshot_features(features: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    snapshots = snapshots.copy()
    snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], errors="coerce")
    snapshots["reference_date"] = pd.to_datetime(snapshots["reference_date"], errors="coerce")
    snapshots = snapshots[
        snapshots["scheduled_game_id"].astype(str).str.match(r"^\d{8}[A-Z]{4}0_.+$")
        & snapshots["snapshot_time"].notna()
        & (snapshots["snapshot_time"].dt.hour < 18)
    ].copy()
    snapshots = snapshots.sort_values("snapshot_time").drop_duplicates(
        ["reference_date", "scheduled_game_id", "team"],
        keep="last",
    )
    snapshots["starter_era"] = pd.to_numeric(snapshots["starter_era"], errors="coerce")
    snapshots["starter_whip"] = pd.to_numeric(snapshots["starter_whip"], errors="coerce")
    snapshots["starter_info_quality"] = pd.to_numeric(
        snapshots["starter_info_quality"], errors="coerce"
    )
    snapshots["recent_3day_games"] = pd.to_numeric(
        snapshots["recent_3day_games"], errors="coerce"
    )
    snapshots["bullpen_fatigue_score"] = (
        snapshots["bullpen_fatigue_label"].map(FATIGUE_SCORE).astype(float)
    )

    snapshot_columns = [
        "reference_date",
        "scheduled_game_id",
        "team",
        "starter_source",
        "starter_info_quality",
        "starter_era",
        "starter_whip",
        "bullpen_fatigue_score",
        "recent_3day_games",
    ]
    own = snapshots[snapshot_columns].rename(
        columns={
            "reference_date": "snapshot_reference_date",
            "scheduled_game_id": "game_id",
            "starter_source": "team_starter_source",
            "starter_info_quality": "team_starter_info_quality",
            "starter_era": "team_starter_era_snapshot",
            "starter_whip": "team_starter_whip_snapshot",
            "bullpen_fatigue_score": "team_bullpen_fatigue_label_encoded",
            "recent_3day_games": "team_recent_3day_games_snapshot",
        }
    )
    enriched = features.merge(own, on=["game_id", "team"], how="inner")
    enriched["actual_game_id"] = (
        enriched["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    )

    opponent = own.rename(
        columns={
            "team": "opponent",
            "team_starter_source": "opponent_starter_source",
            "team_starter_info_quality": "opponent_starter_info_quality",
            "team_starter_era_snapshot": "opponent_starter_era_snapshot",
            "team_starter_whip_snapshot": "opponent_starter_whip_snapshot",
            "team_bullpen_fatigue_label_encoded": "opponent_bullpen_fatigue_label_encoded",
            "team_recent_3day_games_snapshot": "opponent_recent_3day_games_snapshot",
        }
    ).drop(columns=["snapshot_reference_date"])
    opponent["actual_game_id"] = (
        opponent["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    )
    opponent = opponent.drop(columns=["game_id"])
    enriched = enriched.merge(opponent, on=["actual_game_id", "opponent"], how="inner")
    enriched["starter_era_gap"] = (
        enriched["opponent_starter_era_snapshot"]
        - enriched["team_starter_era_snapshot"]
    )
    enriched["starter_whip_gap"] = (
        enriched["opponent_starter_whip_snapshot"]
        - enriched["team_starter_whip_snapshot"]
    )
    enriched["both_starters_confirmed"] = (
        enriched["team_starter_source"].eq("confirmed")
        & enriched["opponent_starter_source"].eq("confirmed")
    ).astype(int)
    enriched["bullpen_fatigue_gap"] = (
        enriched["opponent_bullpen_fatigue_label_encoded"]
        - enriched["team_bullpen_fatigue_label_encoded"]
    )
    enriched["recent_3day_games_snapshot_gap"] = (
        enriched["opponent_recent_3day_games_snapshot"]
        - enriched["team_recent_3day_games_snapshot"]
    )
    complete_games = enriched.groupby("actual_game_id").size()
    complete_games = complete_games[complete_games.eq(2)].index
    return enriched[enriched["actual_game_id"].isin(complete_games)].sort_values(
        ["date", "actual_game_id", "is_home"]
    )


def metric_row(
    name: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probability: np.ndarray,
    feature_count: int,
) -> dict:
    prediction = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1 - probability)
    high = confidence >= 0.55
    return {
        "model": name,
        "feature_count": feature_count,
        "test_team_rows": int(len(frame)),
        "test_games": int(frame["actual_game_id"].nunique()),
        "accuracy": round(float((prediction == y_true).mean()), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 4),
        "log_loss": round(float(log_loss(y_true, probability, labels=[0, 1])), 4),
        "over_55_games": int(high.sum() // 2),
        "over_55_accuracy": (
            round(float((prediction[high] == y_true[high]).mean()), 4)
            if high.any()
            else None
        ),
        "average_confidence": round(float(confidence.mean()), 4),
    }


def bootstrap_accuracy_delta(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    iterations: int = 1000,
) -> dict:
    game_ids = frame["actual_game_id"].drop_duplicates().to_numpy()
    indexes_by_game = {
        game_id: np.flatnonzero(frame["actual_game_id"].to_numpy() == game_id)
        for game_id in game_ids
    }
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(game_ids, size=len(game_ids), replace=True)
        indexes = np.concatenate([indexes_by_game[game_id] for game_id in sampled])
        baseline_accuracy = (
            (baseline_probability[indexes] >= 0.5).astype(int) == y_true[indexes]
        ).mean()
        candidate_accuracy = (
            (candidate_probability[indexes] >= 0.5).astype(int) == y_true[indexes]
        ).mean()
        deltas.append(float(candidate_accuracy - baseline_accuracy))
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "iterations": iterations,
        "mean_accuracy_delta": round(float(np.mean(deltas)), 4),
        "ci_lower_95": round(float(lower), 4),
        "ci_upper_95": round(float(upper), 4),
        "stable_positive_delta": bool(lower > 0),
    }


def run_validation(base_dir: Path) -> dict:
    results_dir = base_dir / "modeling" / "results"
    features = pd.read_csv(results_dir / "features.csv")
    snapshots = pd.read_csv(base_dir / "data" / "official" / "pitching_daily_snapshot.csv")
    frame = attach_snapshot_features(features, snapshots)
    if frame["actual_game_id"].nunique() < 20:
        raise RuntimeError("투수 스냅샷과 완료 경기의 매칭 표본이 부족합니다.")

    dates = pd.to_datetime(frame["date"])
    unique_dates = dates.drop_duplicates().sort_values().to_list()
    cutoff = unique_dates[max(int(len(unique_dates) * 0.7), 1)]
    train_mask = dates < cutoff
    test_mask = dates >= cutoff
    x, y = prepare_matrix(frame)
    baseline_columns = compact_feature_columns(x)
    candidate_columns = baseline_columns + [
        feature for feature in PITCHING_FEATURES if feature in x.columns
    ]
    probabilities = {}
    report_rows = []
    for name, columns in [
        ("baseline_core_snapshot_window", baseline_columns),
        ("baseline_plus_pitching_snapshot", candidate_columns),
    ]:
        x_train = x.loc[train_mask, columns]
        x_test = x.loc[test_mask, columns]
        train_scaled, test_scaled, _, _ = standardize_train_test(x_train, x_test)
        model = conservative_model()
        model.fit(train_scaled, y[train_mask])
        raw_probability = model.predict_proba(test_scaled)[:, 1]
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        probability = normalize_game_probabilities(test_frame, raw_probability)
        probabilities[name] = probability
        report_rows.append(
            metric_row(name, test_frame, y[test_mask], probability, len(columns))
        )

    report = pd.DataFrame(report_rows)
    report.to_csv(
        results_dir / "pitching_snapshot_candidate_validation_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline = report_rows[0]
    candidate = report_rows[1]
    bootstrap = bootstrap_accuracy_delta(
        frame.loc[test_mask].reset_index(drop=True),
        y[test_mask],
        probabilities["baseline_core_snapshot_window"],
        probabilities["baseline_plus_pitching_snapshot"],
    )
    accuracy_delta = round(candidate["accuracy"] - baseline["accuracy"], 4)
    brier_delta = round(candidate["brier_score"] - baseline["brier_score"], 4)
    log_loss_delta = round(candidate["log_loss"] - baseline["log_loss"], 4)
    gates = {
        "minimum_snapshot_days_30": snapshots["snapshot_date"].nunique() >= 30,
        "minimum_matched_games_300": frame["actual_game_id"].nunique() >= 300,
        "minimum_test_games_100": candidate["test_games"] >= 100,
        "accuracy_delta_greater_than_0_005": accuracy_delta > 0.005,
        "brier_not_worse": brier_delta <= 0.0,
        "log_loss_not_worse": log_loss_delta <= 0.0,
        "bootstrap_ci_stable": bootstrap["stable_positive_delta"],
    }
    payload = {
        "experiment": "baseline_plus_pitching_snapshot",
        "status": "validated_candidate_not_promoted",
        "snapshot_days": int(snapshots["snapshot_date"].nunique()),
        "canonical_snapshot_rows": int(len(snapshots)),
        "matched_games": int(frame["actual_game_id"].nunique()),
        "matched_team_rows": int(len(frame)),
        "train_games": int(frame.loc[train_mask, "actual_game_id"].nunique()),
        "test_games": int(frame.loc[test_mask, "actual_game_id"].nunique()),
        "train_start_date": str(frame.loc[train_mask, "date"].min()),
        "train_end_date": str(frame.loc[train_mask, "date"].max()),
        "test_start_date": str(frame.loc[test_mask, "date"].min()),
        "test_end_date": str(frame.loc[test_mask, "date"].max()),
        "baseline": baseline,
        "candidate": candidate,
        "accuracy_delta": accuracy_delta,
        "brier_delta": brier_delta,
        "log_loss_delta": log_loss_delta,
        "bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "production_promotion_allowed": bool(all(gates.values())),
        "safe_to_replace_model": False,
        "safe_to_use_pitching_snapshot_as_features": False,
        "snapshot_eligible_for_experiment": True,
        "leakage_policy": (
            "표준 KBO game_id와 18시 이전 예측 시점 스냅샷만 완료 경기의 동일 팀 행에 "
            "연결했으며 현재 경기 결과는 피처에서 제외했습니다."
        ),
        "decision": (
            "후보가 모든 승격 게이트를 통과하지 않았으므로 운영 모델과 운영 확률을 변경하지 않습니다."
        ),
    }
    (results_dir / "pitching_snapshot_candidate_gate_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(json.dumps(run_validation(args.base_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
