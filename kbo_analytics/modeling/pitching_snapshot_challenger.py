from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FATIGUE_SCORE = {"낮음": 0.0, "보통": 1.0, "높음": 2.0}
BASELINE_COLUMNS = [
    "recent_10_win_rate_gap",
    "season_win_rate_gap",
    "season_avg_run_diff_gap",
    "recent_5_runs_avg_gap",
    "recent_5_allowed_avg_gap",
    "recent_5_run_creation_gap",
    "recent_10_run_creation_gap",
    "recent_run_diff_10_gap",
    "venue_win_rate_gap",
    "games_last_7_days_gap",
    "recent_3day_games_gap",
    "rest_days_gap",
    "bullpen_fatigue_score_gap",
    "bullpen_fatigue_gap",
]
PITCHING_COLUMNS = [
    "starter_era_gap_snapshot",
    "starter_whip_gap_snapshot",
    "starter_info_quality_gap",
    "both_starters_confirmed_snapshot",
    "bullpen_fatigue_label_gap_snapshot",
    "recent_3day_games_gap_snapshot",
]
SNAPSHOT_COLUMNS = {
    "snapshot_time",
    "reference_date",
    "team",
    "starter_name",
    "starter_source",
    "starter_info_quality",
    "starter_era",
    "starter_whip",
    "bullpen_fatigue_label",
    "recent_3day_games",
    "scheduled_game_id",
    "home_away",
}
SCHEDULE_COLUMNS = {
    "reference_date",
    "official_game_id",
    "scheduled_start_datetime",
    "status",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def canonical_snapshot_rows(
    snapshot: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    _require_columns(snapshot, SNAPSHOT_COLUMNS, "pitching snapshot")
    _require_columns(schedule, SCHEDULE_COLUMNS, "official schedule")
    rows = snapshot.copy()
    rows["snapshot_time"] = pd.to_datetime(rows["snapshot_time"], errors="raise")
    rows["reference_date"] = pd.to_datetime(rows["reference_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    rows["official_game_id"] = (
        rows["scheduled_game_id"].astype(str).str.rsplit("_", n=1).str[0]
    )
    official = schedule.copy()
    official["reference_date"] = pd.to_datetime(
        official["reference_date"]
    ).dt.strftime("%Y-%m-%d")
    official["scheduled_start_datetime"] = pd.to_datetime(
        official["scheduled_start_datetime"], errors="raise"
    )
    official = official.drop_duplicates(["reference_date", "official_game_id"])
    merged = rows.merge(
        official[
            [
                "reference_date",
                "official_game_id",
                "scheduled_start_datetime",
                "status",
            ]
        ],
        on=["reference_date", "official_game_id"],
        how="left",
        validate="many_to_one",
    )
    mapping_failures = int(merged["scheduled_start_datetime"].isna().sum())
    post_start = merged["snapshot_time"].ge(merged["scheduled_start_datetime"])
    non_final = ~merged["status"].eq("Final")
    eligible = merged[
        merged["scheduled_start_datetime"].notna() & ~post_start & ~non_final
    ].copy()
    eligible = (
        eligible.sort_values("snapshot_time")
        .groupby(["official_game_id", "team"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    audit = {
        "snapshot_rows": int(len(snapshot)),
        "schedule_rows": int(len(schedule)),
        "mapping_failures": mapping_failures,
        "post_start_rows_excluded": int(post_start.sum()),
        "non_final_rows_excluded": int(non_final.sum()),
        "eligible_team_rows": int(len(eligible)),
        "eligible_games": int(eligible["official_game_id"].nunique()),
    }
    return eligible, audit


def build_pitching_game_features(snapshot_rows: pd.DataFrame) -> pd.DataFrame:
    rows = snapshot_rows.copy()
    rows["starter_era"] = pd.to_numeric(rows["starter_era"], errors="coerce")
    rows["starter_whip"] = pd.to_numeric(rows["starter_whip"], errors="coerce")
    rows["starter_info_quality"] = pd.to_numeric(
        rows["starter_info_quality"], errors="coerce"
    )
    rows["recent_3day_games"] = pd.to_numeric(
        rows["recent_3day_games"], errors="coerce"
    )
    rows["fatigue_score"] = rows["bullpen_fatigue_label"].map(FATIGUE_SCORE)
    home = rows[rows["home_away"].eq("H")].copy()
    away = rows[rows["home_away"].eq("A")].copy()
    features = home.merge(
        away,
        on="official_game_id",
        suffixes=("_home", "_away"),
        how="inner",
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "game_id": features["official_game_id"],
            "snapshot_time_home": features["snapshot_time_home"],
            "snapshot_time_away": features["snapshot_time_away"],
            "home_starter_name_snapshot": features["starter_name_home"],
            "away_starter_name_snapshot": features["starter_name_away"],
            "starter_era_gap_snapshot": (
                features["starter_era_away"] - features["starter_era_home"]
            ),
            "starter_whip_gap_snapshot": (
                features["starter_whip_away"] - features["starter_whip_home"]
            ),
            "starter_info_quality_gap": (
                features["starter_info_quality_home"]
                - features["starter_info_quality_away"]
            ),
            "both_starters_confirmed_snapshot": (
                features["starter_source_home"].isin(["confirmed", "manual"])
                & features["starter_source_away"].isin(["confirmed", "manual"])
            ).astype(int),
            "bullpen_fatigue_label_gap_snapshot": (
                features["fatigue_score_away"] - features["fatigue_score_home"]
            ),
            "recent_3day_games_gap_snapshot": (
                features["recent_3day_games_away"]
                - features["recent_3day_games_home"]
            ),
        }
    )


def _model(family: str):
    if family == "LogisticRegression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5),
        )
    if family == "HistGradientBoosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.035,
                max_leaf_nodes=10,
                l2_regularization=0.2,
                random_state=42,
            ),
        )
    raise ValueError(f"unknown model family: {family}")


def expanding_date_predictions(
    frame: pd.DataFrame,
    columns: list[str],
    family: str,
    minimum_train_games: int = 80,
) -> pd.DataFrame:
    work = frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    work["game_date"] = pd.to_datetime(work["game_date"])
    predictions = []
    for prediction_date in sorted(work["game_date"].unique()):
        train = work[work["game_date"].lt(prediction_date)]
        test = work[work["game_date"].eq(prediction_date)]
        if len(train) < minimum_train_games or test.empty:
            continue
        model = _model(family)
        model.fit(train[columns], train["home_win"].astype(int))
        probability = model.predict_proba(test[columns])[:, 1]
        result = test[["game_date", "game_id", "away_team", "home_team", "home_win"]].copy()
        result["home_probability"] = probability
        result["pred_home_win"] = (probability >= 0.5).astype(int)
        result["correct"] = result["pred_home_win"].eq(result["home_win"])
        predictions.append(result)
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)


def _metrics(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {
            "games": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "over_55_games": 0,
            "over_55_accuracy": None,
        }
    y = predictions["home_win"].astype(int).to_numpy()
    p = predictions["home_probability"].astype(float).to_numpy()
    confidence = np.maximum(p, 1 - p)
    over_55 = confidence >= 0.55
    return {
        "games": int(len(predictions)),
        "accuracy": round(float(predictions["correct"].mean()), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(
            float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), 4
        ),
        "over_55_games": int(over_55.sum()),
        "over_55_accuracy": (
            round(float(predictions.loc[over_55, "correct"].mean()), 4)
            if over_55.any()
            else None
        ),
    }


def _bootstrap_accuracy_delta(
    baseline: pd.DataFrame, candidate: pd.DataFrame, repeats: int = 2000
) -> dict:
    paired = baseline[["game_id", "correct"]].merge(
        candidate[["game_id", "correct"]],
        on="game_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if paired.empty:
        return {"paired_games": 0, "delta": None, "ci_low": None, "ci_high": None}
    differences = (
        paired["correct_candidate"].astype(float)
        - paired["correct_baseline"].astype(float)
    ).to_numpy()
    rng = np.random.default_rng(42)
    samples = np.array(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(repeats)]
    )
    return {
        "paired_games": int(len(paired)),
        "delta": round(float(differences.mean()), 4),
        "ci_low": round(float(np.quantile(samples, 0.025)), 4),
        "ci_high": round(float(np.quantile(samples, 0.975)), 4),
    }


def compare_with_production_history(
    candidate: pd.DataFrame, history: pd.DataFrame
) -> dict:
    required = {
        "run_time",
        "update_stage",
        "game_id",
        "home_team",
        "predicted_team",
        "win_probability",
    }
    _require_columns(history, required, "production prediction history")
    production = history[history["update_stage"].eq("pregame")].copy()
    production["run_time"] = pd.to_datetime(production["run_time"], errors="raise")
    production["win_probability"] = pd.to_numeric(
        production["win_probability"], errors="raise"
    )
    production = production.sort_values("run_time").groupby("game_id").tail(1)
    production["home_probability"] = np.where(
        production["predicted_team"].eq(production["home_team"]),
        production["win_probability"],
        1 - production["win_probability"],
    )
    paired = candidate.merge(
        production[["game_id", "home_probability"]],
        on="game_id",
        how="inner",
        suffixes=("_candidate", "_production"),
        validate="one_to_one",
    )
    if paired.empty:
        return {"paired_games": 0, "sufficient_sample": False}
    target = paired["home_win"].astype(int)
    candidate_probability = paired["home_probability_candidate"].astype(float)
    production_probability = paired["home_probability_production"].astype(float)

    def metrics(probability: pd.Series) -> dict:
        correct = probability.ge(0.5).astype(int).eq(target)
        return {
            "accuracy": round(float(correct.mean()), 4),
            "brier": round(float(brier_score_loss(target, probability)), 4),
            "log_loss": round(
                float(
                    log_loss(
                        target,
                        probability.clip(1e-6, 1 - 1e-6),
                        labels=[0, 1],
                    )
                ),
                4,
            ),
        }

    return {
        "paired_games": int(len(paired)),
        "sufficient_sample": len(paired) >= 100,
        "production": metrics(production_probability),
        "candidate": metrics(candidate_probability),
    }


def evaluate_pitching_snapshot_challenger(
    pregame_store: pd.DataFrame,
    snapshot: pd.DataFrame,
    schedule: pd.DataFrame,
    production_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    canonical, audit = canonical_snapshot_rows(snapshot, schedule)
    pitching = build_pitching_game_features(canonical)
    frame = pregame_store.merge(pitching, on="game_id", how="inner", validate="one_to_one")
    frame = frame.dropna(subset=["home_win"]).copy()
    baseline_columns = [column for column in BASELINE_COLUMNS if column in frame.columns]
    pitching_columns = [column for column in PITCHING_COLUMNS if column in frame.columns]
    report_rows = []
    prediction_rows = []
    family_results = {}
    for family in ["LogisticRegression", "HistGradientBoosting"]:
        baseline = expanding_date_predictions(frame, baseline_columns, family)
        candidate = expanding_date_predictions(
            frame, baseline_columns + pitching_columns, family
        )
        baseline_metrics = _metrics(baseline)
        candidate_metrics = _metrics(candidate)
        bootstrap = _bootstrap_accuracy_delta(baseline, candidate)
        for feature_set, metrics in [
            ("baseline_same_games", baseline_metrics),
            ("baseline_plus_pitching_snapshot", candidate_metrics),
        ]:
            report_rows.append(
                {
                    "model_family": family,
                    "feature_set": feature_set,
                    **metrics,
                    "accuracy_delta_vs_baseline": (
                        bootstrap["delta"]
                        if feature_set == "baseline_plus_pitching_snapshot"
                        else 0.0
                    ),
                    "bootstrap_ci_low": (
                        bootstrap["ci_low"]
                        if feature_set == "baseline_plus_pitching_snapshot"
                        else None
                    ),
                    "bootstrap_ci_high": (
                        bootstrap["ci_high"]
                        if feature_set == "baseline_plus_pitching_snapshot"
                        else None
                    ),
                }
            )
        for feature_set, predictions in [
            ("baseline_same_games", baseline),
            ("baseline_plus_pitching_snapshot", candidate),
        ]:
            if not predictions.empty:
                tagged = predictions.copy()
                tagged["model_family"] = family
                tagged["feature_set"] = feature_set
                prediction_rows.append(tagged)
        family_results[family] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "bootstrap": bootstrap,
        }
    best_family = max(
        family_results,
        key=lambda name: family_results[name]["candidate"].get("accuracy") or 0,
    )
    best = family_results[best_family]
    baseline = best["baseline"]
    candidate = best["candidate"]
    bootstrap = best["bootstrap"]
    production_comparison = None
    if production_history is not None:
        best_candidate_predictions = next(
            tagged
            for tagged in prediction_rows
            if tagged["model_family"].iloc[0] == best_family
            and tagged["feature_set"].iloc[0]
            == "baseline_plus_pitching_snapshot"
        )
        production_comparison = compare_with_production_history(
            best_candidate_predictions, production_history
        )
    gates = {
        "minimum_100_out_of_sample_games": candidate["games"] >= 100,
        "accuracy_improves_more_than_0_5pp": (
            candidate["accuracy"] is not None
            and baseline["accuracy"] is not None
            and candidate["accuracy"] - baseline["accuracy"] > 0.005
        ),
        "brier_not_worse": (
            candidate["brier"] is not None
            and baseline["brier"] is not None
            and candidate["brier"] <= baseline["brier"] + 0.001
        ),
        "log_loss_not_worse": (
            candidate["log_loss"] is not None
            and baseline["log_loss"] is not None
            and candidate["log_loss"] <= baseline["log_loss"] + 0.002
        ),
        "bootstrap_ci_stable": (
            bootstrap["ci_low"] is not None and bootstrap["ci_low"] > 0
        ),
        "no_schedule_mapping_failures": audit["mapping_failures"] == 0,
    }
    if production_comparison is not None:
        comparison_candidate = production_comparison.get("candidate", {})
        comparison_production = production_comparison.get("production", {})
        gates.update(
            {
                "minimum_100_production_paired_games": production_comparison.get(
                    "sufficient_sample", False
                ),
                "candidate_accuracy_beats_production_same_games": (
                    comparison_candidate.get("accuracy") is not None
                    and comparison_production.get("accuracy") is not None
                    and comparison_candidate["accuracy"]
                    - comparison_production["accuracy"]
                    > 0.005
                ),
                "candidate_probability_quality_not_worse_than_production": (
                    comparison_candidate.get("brier") is not None
                    and comparison_production.get("brier") is not None
                    and comparison_candidate.get("log_loss") is not None
                    and comparison_production.get("log_loss") is not None
                    and comparison_candidate["brier"]
                    <= comparison_production["brier"] + 0.001
                    and comparison_candidate["log_loss"]
                    <= comparison_production["log_loss"] + 0.002
                ),
            }
        )
    offline_gates_passed = all(gates.values())
    summary = {
        "snapshot_audit": audit,
        "eligible_model_games": int(len(frame)),
        "baseline_features": baseline_columns,
        "pitching_snapshot_features": pitching_columns,
        "best_candidate_family": best_family,
        "best_family_results": best,
        "production_same_game_comparison": production_comparison,
        "production_replacement_gates": gates,
        "offline_gates_passed": offline_gates_passed,
        "safe_to_replace_model": False,
        "safe_to_use_pitching_snapshot_as_production_features": False,
        "decision": (
            "eligible_for_full_production_gate_review"
            if offline_gates_passed
            else "offline_candidate_only"
        ),
        "policy_note": "Only snapshots captured before official game start are used. Cancelled, postponed, scheduled, unmapped, and post-start rows are excluded. Passing this offline experiment only permits a separate production gate review; it never promotes the model or changes production probabilities.",
    }
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return pd.DataFrame(report_rows), predictions, summary


def write_pitching_snapshot_challenger_reports(
    pregame_store_path: Path,
    snapshot_path: Path,
    schedule_path: Path,
    output_dir: Path,
    production_history_path: Path | None = None,
) -> dict:
    pregame_store = pd.read_csv(pregame_store_path, encoding="utf-8-sig")
    snapshot = pd.read_csv(snapshot_path, encoding="utf-8-sig")
    schedule = pd.read_csv(schedule_path, encoding="utf-8-sig")
    production_history = (
        pd.read_csv(production_history_path, encoding="utf-8-sig")
        if production_history_path is not None
        else None
    )
    report, predictions, summary = evaluate_pitching_snapshot_challenger(
        pregame_store, snapshot, schedule, production_history
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        output_dir / "pitching_snapshot_challenger_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        output_dir / "pitching_snapshot_challenger_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "pitching_snapshot_challenger_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
