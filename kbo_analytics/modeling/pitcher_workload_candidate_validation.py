from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_training import _fit_predict_pregame_model, _pregame_metrics, pregame_feature_sets


def _model_specs():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=12,
            l2_regularization=0.1,
            random_state=42,
        ),
    }


def _paired_bootstrap_delta(y_true, baseline_probability, candidate_probability, dates, iterations=2000):
    baseline_correct = (np.asarray(baseline_probability) >= 0.5) == np.asarray(y_true)
    candidate_correct = (np.asarray(candidate_probability) >= 0.5) == np.asarray(y_true)
    rng = np.random.default_rng(42)
    deltas = np.empty(iterations)
    dates = np.asarray(dates)
    blocks = [np.flatnonzero(dates == day) for day in pd.unique(dates)]
    for index in range(iterations):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        deltas[index] = candidate_correct[sample].mean() - baseline_correct[sample].mean()
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _chronological_current_season_split(frame: pd.DataFrame, train_ratio: float = 0.7):
    ordered_dates = sorted(pd.to_datetime(frame["game_date"]).dt.date.unique())
    split_index = max(1, min(len(ordered_dates) - 1, int(len(ordered_dates) * train_ratio)))
    cutoff = ordered_dates[split_index]
    train = frame[pd.to_datetime(frame["game_date"]).dt.date < cutoff].copy()
    test = frame[pd.to_datetime(frame["game_date"]).dt.date >= cutoff].copy()
    return train, test, cutoff


def validate_pitcher_workload_candidate(store: pd.DataFrame, results_dir: Path):
    frame = store.dropna(subset=["home_win"]).copy()
    latest_year = pd.to_datetime(frame["game_date"]).dt.year.max()
    frame = frame[pd.to_datetime(frame["game_date"]).dt.year.eq(latest_year)]
    frame = frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    development, test, cutoff = _chronological_current_season_split(frame, 0.8)
    train, selection, selection_cutoff = _chronological_current_season_split(development, 0.75)
    if len(train) < 250 or len(selection) < 50 or len(test) < 100:
        raise ValueError("현재 시즌 투수 challenger 검증에 필요한 시간순 표본이 부족합니다.")

    sets = pregame_feature_sets(train)
    evaluated_sets = [
        "production_features_only",
        "production_plus_starter",
        "production_plus_bullpen",
        "production_plus_starter_bullpen",
        "production_plus_pregame_safe_features",
    ]
    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    y_selection = selection["home_win"].astype(int).to_numpy()
    rows = []
    for model_name, factory in _model_specs().items():
        for feature_set in evaluated_sets:
            columns = sets.get(feature_set, [])
            if not columns:
                continue
            probability = _fit_predict_pregame_model(
                model_name,
                factory(),
                train[columns],
                y_train,
                selection[columns],
            )
            metrics = _pregame_metrics(selection, y_selection, probability)
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "season": int(latest_year),
                    "evaluation_split": "candidate_selection_only",
                    "train_cutoff_exclusive": selection_cutoff.isoformat(),
                    "train_games": int(len(train)),
                    "test_games": int(len(selection)),
                    "accuracy": metrics["accuracy"],
                    "brier": metrics["brier"],
                    "log_loss": metrics["log_loss"],
                    "over_55_games": metrics["over_55_games"],
                    "over_55_accuracy": metrics["over_55_accuracy"],
                    "daily_top1_accuracy": metrics["daily_top1_accuracy"],
                    "selected_candidate": False,
                }
            )

    candidates = [row for row in rows if row["feature_set"] != "production_features_only"]
    best = max(candidates, key=lambda row: (row["accuracy"], -row["brier"], row["over_55_accuracy"] or 0))
    best["selected_candidate"] = True
    factory = _model_specs()[best["model"]]
    holdout = []
    probabilities = []
    for feature_set in ("production_features_only", best["feature_set"]):
        columns = sets[feature_set]
        probability = _fit_predict_pregame_model(
            best["model"], factory(), development[columns],
            development["home_win"].astype(int).to_numpy(), test[columns],
        )
        metrics = _pregame_metrics(test, y_test, probability)
        holdout.append({"model": best["model"], "feature_set": feature_set,
                        "evaluation_split": "untouched_final_holdout", "test_games": len(test),
                        **{key: metrics[key] for key in ("accuracy", "brier", "log_loss", "over_55_games", "over_55_accuracy")}})
        probabilities.append(probability)
    baseline, best_holdout = holdout
    baseline_probability, candidate_probability = probabilities
    best = best_holdout
    ci_low, ci_high = _paired_bootstrap_delta(y_test, baseline_probability, candidate_probability, test["game_date"].to_numpy())
    accuracy_delta = round(best["accuracy"] - baseline["accuracy"], 4)
    checks = {
        "minimum_test_games": len(test) >= 100,
        "accuracy_delta_greater_than_0_005": accuracy_delta > 0.005,
        "bootstrap_ci_stable": ci_low > 0,
        "brier_not_worse": best["brier"] <= baseline["brier"] + 0.001,
        "log_loss_not_worse": best["log_loss"] <= baseline["log_loss"] + 0.002,
        "over_55_accuracy_not_worse": (
            best["over_55_accuracy"] is not None
            and baseline["over_55_accuracy"] is not None
            and best["over_55_accuracy"] >= baseline["over_55_accuracy"]
        ),
    }
    gate = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "season": int(latest_year),
        "selection_cutoff": selection_cutoff.isoformat(),
        "final_holdout_start": cutoff.isoformat(),
        "final_holdout_end": str(test["game_date"].max()),
        "baseline_type": "refit_feature_ablation_not_deployed_production_model",
        "production_comparison_verified": False,
        "starter_identity_asof_verified": False,
        "bootstrap_unit": "game_date_block",
        "train_cutoff_exclusive": cutoff.isoformat(),
        "baseline": baseline,
        "best_candidate": best,
        "accuracy_delta": accuracy_delta,
        "bootstrap_accuracy_delta_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "gate_checks": checks,
        "failed_gate_items": [name for name, passed in checks.items() if not passed],
        "safe_to_replace_model": False,
        "safe_to_use_pitching_snapshot_as_features": False,
        "eligible_for_production_comparison": all(checks.values()),
        "decision": "offline_candidate_only",
        "promotion_blockers": ["deployed_model_same_game_comparison_missing", "starter_identity_asof_not_verified", "full_production_gate_not_run"],
        "leakage_policy": "Prior-date workload only. Historical boxscore starter identity is not proof of pregame availability. No pitching snapshots are used.",
        "feature_semantics": {"starter_recent3_whip": "H+BB+HBP per inning proxy, not official WHIP", "starter_rest_days": "calendar days since prior start, not all pitching appearances"},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows + holdout).to_csv(
        results_dir / "pitcher_workload_candidate_validation_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (results_dir / "pitcher_workload_candidate_gate_audit.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gate
