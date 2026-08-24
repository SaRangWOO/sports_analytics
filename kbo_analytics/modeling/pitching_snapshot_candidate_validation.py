from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


def attach_snapshot_features(
    features: pd.DataFrame,
    snapshots: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    snapshots = snapshots.copy()
    snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], errors="coerce")
    snapshots["reference_date"] = pd.to_datetime(snapshots["reference_date"], errors="coerce")
    snapshots["official_game_id"] = (
        snapshots["scheduled_game_id"].astype(str).str.rsplit("_", n=1).str[0]
    )
    schedule = schedule.copy()
    schedule["reference_date"] = pd.to_datetime(
        schedule["reference_date"], errors="coerce"
    )
    schedule["scheduled_start_datetime"] = pd.to_datetime(
        schedule["scheduled_start_datetime"], errors="coerce"
    )
    snapshots = snapshots.merge(
        schedule[
            [
                "reference_date",
                "official_game_id",
                "away_team",
                "home_team",
                "scheduled_start_datetime",
            ]
        ],
        on=["reference_date", "official_game_id"],
        how="left",
        validate="many_to_one",
    )
    valid_id = snapshots["official_game_id"].astype(str).str.match(
        r"^\d{8}[A-Z]{4}\d+$"
    )
    valid_time = snapshots["snapshot_time"].notna()
    schedule_mapped = snapshots["scheduled_start_datetime"].notna()
    before_start = snapshots["snapshot_time"] < snapshots["scheduled_start_datetime"]
    team_mapping = (
        (
            snapshots["home_away"].eq("A")
            & snapshots["team"].eq(snapshots["away_team"])
            & snapshots["opponent"].eq(snapshots["home_team"])
        )
        | (
            snapshots["home_away"].eq("H")
            & snapshots["team"].eq(snapshots["home_team"])
            & snapshots["opponent"].eq(snapshots["away_team"])
        )
    )
    audit = {
        "snapshot_rows": int(len(snapshots)),
        "invalid_official_game_id_rows": int((~valid_id).sum()),
        "invalid_snapshot_time_rows": int((~valid_time).sum()),
        "schedule_unmapped_rows": int((~schedule_mapped).sum()),
        "snapshot_at_or_after_start_rows": int(
            (schedule_mapped & valid_time & ~before_start).sum()
        ),
        "team_mapping_failed_rows": int((schedule_mapped & ~team_mapping).sum()),
    }
    eligible = valid_id & valid_time & schedule_mapped & before_start & team_mapping
    snapshots = snapshots[eligible].copy()
    audit["eligible_snapshot_rows"] = int(len(snapshots))
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
    complete = enriched[enriched["actual_game_id"].isin(complete_games)].sort_values(
        ["date", "actual_game_id", "is_home"]
    )
    audit["matched_team_rows"] = int(len(complete))
    audit["matched_games"] = int(complete["actual_game_id"].nunique())
    return complete.reset_index(drop=True), audit


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


def historical_pitcher_data_audit(base_dir: Path, model_features: pd.DataFrame) -> dict:
    player_path = base_dir / "data" / "weekly" / "player_game_stats.csv"
    log_path = base_dir.parent / "kbo_run_model" / "data" / "pitcher_game_logs.csv"
    player_rows = pd.read_csv(player_path) if player_path.exists() else pd.DataFrame()
    pitcher_rows = (
        player_rows[
            pd.to_numeric(player_rows.get("innings_pitched"), errors="coerce").fillna(0) > 0
        ].copy()
        if not player_rows.empty and "innings_pitched" in player_rows
        else pd.DataFrame()
    )
    explicit_logs = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    model_game_ids = (
        model_features["game_id"].astype(str).str.rsplit("_", n=1).str[0].nunique()
    )
    audit = {
        "source": str(player_path.relative_to(base_dir)),
        "player_rows": int(len(player_rows)),
        "pitcher_rows": int(len(pitcher_rows)),
        "covered_games": int(pitcher_rows["game_id"].nunique()) if not pitcher_rows.empty else 0,
        "first_date": str(pitcher_rows["date"].min()) if not pitcher_rows.empty else "",
        "last_date": str(pitcher_rows["date"].max()) if not pitcher_rows.empty else "",
        "model_games": int(model_game_ids),
        "explicit_pitcher_log_rows": int(len(explicit_logs)),
        "has_explicit_starter_flag": bool(
            not explicit_logs.empty and "is_starter" in explicit_logs
        ),
        "usable_for_full_historical_training": False,
        "decision": (
            "2026년 일부 경기만 포함하고 명시적 선발 구분이 없어 전체 역사 학습 피처로 사용하지 않습니다."
        ),
    }
    pd.DataFrame([audit]).to_csv(
        base_dir / "modeling" / "results" / "historical_pitcher_data_availability_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit


def fit_baseline(
    all_features: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    train_end: pd.Timestamp,
    columns: list[str],
) -> np.ndarray:
    all_x, all_y = prepare_matrix(all_features)
    prediction_x, _ = prepare_matrix(prediction_frame)
    train_mask = pd.to_datetime(all_features["date"]) < train_end
    x_train = all_x.loc[train_mask, columns]
    x_prediction = prediction_x[columns]
    train_scaled, prediction_scaled, _, _ = standardize_train_test(
        x_train, x_prediction
    )
    model = conservative_model()
    model.fit(train_scaled, all_y[train_mask])
    raw_probability = model.predict_proba(prediction_scaled)[:, 1]
    return normalize_game_probabilities(
        prediction_frame.reset_index(drop=True), raw_probability
    )


def fit_snapshot_adjustment(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    train_baseline_probability: np.ndarray,
    test_frame: pd.DataFrame,
    test_baseline_probability: np.ndarray,
    features: list[str],
    regularization: float,
    blend: float,
) -> np.ndarray:
    available = [
        feature
        for feature in features
        if feature in train_frame and feature in test_frame
    ]
    train = train_frame[available].apply(pd.to_numeric, errors="coerce")
    test = test_frame[available].apply(pd.to_numeric, errors="coerce")
    medians = train.median()
    train = train.fillna(medians).fillna(0.0)
    test = test.fillna(medians).fillna(0.0)
    train.insert(
        0,
        "baseline_logit",
        np.log(
            np.clip(train_baseline_probability, 1e-6, 1 - 1e-6)
            / (1 - np.clip(train_baseline_probability, 1e-6, 1 - 1e-6))
        ),
    )
    test.insert(
        0,
        "baseline_logit",
        np.log(
            np.clip(test_baseline_probability, 1e-6, 1 - 1e-6)
            / (1 - np.clip(test_baseline_probability, 1e-6, 1 - 1e-6))
        ),
    )
    mean = train.mean()
    std = train.std(ddof=0).replace(0, 1)
    model = LogisticRegression(
        C=regularization,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit((train - mean) / std, train_y)
    adjusted_probability = model.predict_proba((test - mean) / std)[:, 1]
    baseline_logit = np.log(
        np.clip(test_baseline_probability, 1e-6, 1 - 1e-6)
        / (1 - np.clip(test_baseline_probability, 1e-6, 1 - 1e-6))
    )
    adjusted_logit = np.log(
        np.clip(adjusted_probability, 1e-6, 1 - 1e-6)
        / (1 - np.clip(adjusted_probability, 1e-6, 1 - 1e-6))
    )
    blended_probability = 1 / (
        1 + np.exp(-((1 - blend) * baseline_logit + blend * adjusted_logit))
    )
    return normalize_game_probabilities(
        test_frame.reset_index(drop=True), blended_probability
    )


def rolling_date_folds(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.Series(pd.to_datetime(frame["date"]).drop_duplicates().sort_values())
    initial_days = max(15, int(len(dates) * 0.35))
    validation_days = max(7, int(len(dates) * 0.15))
    folds = []
    start = initial_days
    while start < len(dates):
        stop = min(start + validation_days, len(dates))
        folds.append((pd.Timestamp(dates.iloc[start]), pd.Timestamp(dates.iloc[stop - 1])))
        start = stop
    return folds


def run_validation(base_dir: Path) -> dict:
    results_dir = base_dir / "modeling" / "results"
    features = pd.read_csv(results_dir / "features.csv")
    snapshots = pd.read_csv(base_dir / "data" / "official" / "pitching_daily_snapshot.csv")
    schedule_path = base_dir / "data" / "official" / "pitching_snapshot_schedule.csv"
    if not schedule_path.exists():
        raise RuntimeError(
            "pitching_snapshot_schedule.csv가 없습니다. 공식 경기 시작 시각을 먼저 수집해야 합니다."
        )
    schedule = pd.read_csv(schedule_path)
    frame, leakage_audit = attach_snapshot_features(features, snapshots, schedule)
    if frame["actual_game_id"].nunique() < 20:
        raise RuntimeError("투수 스냅샷과 완료 경기의 매칭 표본이 부족합니다.")

    all_x, _ = prepare_matrix(features)
    _, y = prepare_matrix(frame)
    baseline_columns = compact_feature_columns(all_x)
    first_snapshot_date = pd.to_datetime(frame["date"]).min()
    prior_baseline_probability = fit_baseline(
        features,
        frame,
        first_snapshot_date,
        baseline_columns,
    )
    candidate_specs = [
        (
            "historical_baseline_plus_starter_conservative",
            ["starter_era_gap", "starter_whip_gap", "both_starters_confirmed"],
            0.03,
            0.25,
        ),
        (
            "historical_baseline_plus_bullpen_proxy_conservative",
            ["bullpen_fatigue_gap", "recent_3day_games_snapshot_gap"],
            0.03,
            0.25,
        ),
        (
            "historical_baseline_plus_all_snapshot_conservative",
            PITCHING_FEATURES,
            0.01,
            0.25,
        ),
    ]
    folds = rolling_date_folds(frame)
    if not folds:
        raise RuntimeError("시간순 rolling 검증 구간을 생성할 수 없습니다.")
    fold_rows = []
    test_frames = []
    test_targets = []
    baseline_probabilities = []
    candidate_probabilities = {spec[0]: [] for spec in candidate_specs}
    dates = pd.to_datetime(frame["date"])
    for fold_index, (test_start, test_end) in enumerate(folds, start=1):
        train_mask = dates < test_start
        test_mask = dates.between(test_start, test_end)
        train_frame = frame.loc[train_mask].reset_index(drop=True)
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        if train_frame["actual_game_id"].nunique() < 50 or test_frame.empty:
            continue
        train_y = y[train_mask]
        test_y = y[test_mask]
        train_baseline_probability = prior_baseline_probability[train_mask]
        test_baseline_probability = fit_baseline(
            features,
            test_frame,
            test_start,
            baseline_columns,
        )
        test_frames.append(test_frame)
        test_targets.append(test_y)
        baseline_probabilities.append(test_baseline_probability)
        baseline_fold = metric_row(
            "historical_baseline",
            test_frame,
            test_y,
            test_baseline_probability,
            len(baseline_columns),
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "train_end": str(test_start.date() - pd.Timedelta(days=1)),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                **baseline_fold,
            }
        )
        for name, candidate_features, regularization, blend in candidate_specs:
            probability = fit_snapshot_adjustment(
                train_frame,
                train_y,
                train_baseline_probability,
                test_frame,
                test_baseline_probability,
                candidate_features,
                regularization,
                blend,
            )
            candidate_probabilities[name].append(probability)
            candidate_fold = metric_row(
                name,
                test_frame,
                test_y,
                probability,
                len(candidate_features) + 1,
            )
            fold_rows.append(
                {
                    "fold": fold_index,
                    "train_end": str(test_start.date() - pd.Timedelta(days=1)),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    **candidate_fold,
                }
            )
    if not test_frames:
        raise RuntimeError("유효한 시간순 rolling 검증 구간이 없습니다.")

    test_frame = pd.concat(test_frames, ignore_index=True)
    test_y = np.concatenate(test_targets)
    test_baseline_probability = np.concatenate(baseline_probabilities)
    report_rows = [
        metric_row(
            "historical_baseline",
            test_frame,
            test_y,
            test_baseline_probability,
            len(baseline_columns),
        )
    ]
    aggregated_candidate_probabilities = {}
    for name, candidate_features, regularization, blend in candidate_specs:
        probability = np.concatenate(candidate_probabilities[name])
        aggregated_candidate_probabilities[name] = probability
        row = metric_row(
            name,
            test_frame,
            test_y,
            probability,
            len(candidate_features) + 1,
        )
        row["regularization_c"] = regularization
        row["snapshot_blend"] = blend
        report_rows.append(row)

    report = pd.DataFrame(report_rows)
    report.to_csv(
        results_dir / "pitching_snapshot_candidate_validation_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        results_dir / "pitching_snapshot_candidate_rolling_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline = report_rows[0]
    candidate = report_rows[1]
    selected_probability = aggregated_candidate_probabilities[candidate["model"]]
    bootstrap = bootstrap_accuracy_delta(
        test_frame,
        test_y,
        test_baseline_probability,
        selected_probability,
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
    historical_pitcher_audit = historical_pitcher_data_audit(base_dir, features)
    leakage_rows = [
        {
            "check": name,
            "failed_rows": value,
            "status": "pass" if value == 0 else "excluded",
        }
        for name, value in leakage_audit.items()
        if name.endswith("_rows") and name not in {"snapshot_rows", "eligible_snapshot_rows", "matched_team_rows"}
    ]
    pd.DataFrame(leakage_rows).to_csv(
        results_dir / "pitching_snapshot_feature_leakage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "experiment": "historical_baseline_plus_snapshot_adjustment",
        "status": "validated_candidate_not_promoted",
        "training_policy": (
            "2016~2026 기존 완료 경기로 기준 모델을 학습하고, 초기 스냅샷 구간에서 "
            "정규화된 확률 보정층을 학습한 뒤 후기 스냅샷 구간을 검증합니다."
        ),
        "snapshot_days": int(snapshots["snapshot_date"].nunique()),
        "canonical_snapshot_rows": int(len(snapshots)),
        "schedule_rows": int(len(schedule)),
        "matched_games": int(frame["actual_game_id"].nunique()),
        "matched_team_rows": int(len(frame)),
        "rolling_folds": int(pd.DataFrame(fold_rows)["fold"].nunique()),
        "test_games": int(test_frame["actual_game_id"].nunique()),
        "test_start_date": str(test_frame["date"].min()),
        "test_end_date": str(test_frame["date"].max()),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_comparison": report_rows[1:],
        "primary_candidate_policy": (
            "선발 ERA/WHIP 차이와 선발 확정 여부를 사용하는 보수 후보를 사전 지정하며, "
            "나머지 스냅샷 조합은 진단용으로만 비교한다."
        ),
        "historical_pitcher_data_audit": historical_pitcher_audit,
        "accuracy_delta": accuracy_delta,
        "brier_delta": brier_delta,
        "log_loss_delta": log_loss_delta,
        "bootstrap": bootstrap,
        "leakage_audit": leakage_audit,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "production_promotion_allowed": bool(all(gates.values())),
        "safe_to_replace_model": False,
        "safe_to_use_pitching_snapshot_as_features": False,
        "snapshot_eligible_for_experiment": True,
        "leakage_policy": (
            "표준 KBO game_id와 공식 scheduled_start_datetime 이전 스냅샷만 완료 경기의 "
            "동일 팀 행에 연결하고, 날짜 순 rolling 검증으로 현재 경기 결과를 피처에서 제외했습니다."
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
