from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .feature_engineering import build_features
from .model_artifacts import load_production_artifact, predict_bundle_probabilities
from .model_evaluation import normalize_game_probabilities
from .model_training import _fit_predict_pregame_model, pregame_feature_sets
from .pitcher_workload_candidate_validation import _model_specs, _paired_bootstrap_delta
from .pitcher_workload_features import attach_pitcher_workload_features, build_pitcher_workload_features
from .prediction_runtime import _team_matrix


def probability_metrics(target, probability):
    target = np.asarray(target)
    probability = np.asarray(probability)
    confidence = np.maximum(probability, 1 - probability)
    correct = (probability >= 0.5) == target
    bins = np.minimum((probability * 10).astype(int), 9)
    calibration = sum(
        np.mean(bins == bucket) * abs(target[bins == bucket].mean() - probability[bins == bucket].mean())
        for bucket in np.unique(bins)
    )
    return {"games": len(target), "accuracy": float(accuracy_score(target, probability >= 0.5)),
            "brier": float(brier_score_loss(target, probability)),
            "log_loss": float(log_loss(target, probability, labels=[0, 1])),
            "calibration_error_10_bins": float(calibration),
            "over_55_games": int((confidence >= 0.55).sum()),
            "over_55_accuracy": float(correct[confidence >= 0.55].mean()) if (confidence >= 0.55).any() else None}


def audit_starter_identity(games, snapshots, schedule, history):
    snapshots = snapshots.copy()
    snapshots["canonical_id"] = snapshots["scheduled_game_id"].str.rsplit("_", n=1).str[0]
    snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], errors="coerce")
    schedule = schedule.copy()
    schedule["scheduled_start_datetime"] = pd.to_datetime(schedule["scheduled_start_datetime"], errors="coerce")
    history = history.copy()
    history["run_time"] = pd.to_datetime(history["run_time"], errors="coerce")
    rows = []
    for game in games.to_dict("records"):
        official = schedule[schedule.official_game_id.eq(game["game_id"]) & schedule.reference_date.eq(game["game_date"])]
        start = official.iloc[0].scheduled_start_datetime if len(official) == 1 else pd.NaT
        predictions = history[history.game_id.eq(game["game_id"]) & history.reference_date.eq(game["game_date"])]
        pregame = predictions[predictions.run_time.lt(start)]
        prediction_time = pregame.run_time.max()
        for side in ("home", "away"):
            team = game[f"{side}_team"]
            candidates = snapshots[snapshots.canonical_id.eq(game["game_id"]) & snapshots.team.eq(team)
                                   & snapshots.reference_date.eq(game["game_date"])]
            eligible = candidates[candidates.snapshot_time.lt(start) & candidates.snapshot_time.le(prediction_time)]
            latest = eligible.sort_values("snapshot_time").tail(1)
            expected = str(game[f"{side}_starter_name"]).replace(" ", "")
            name = str(latest.iloc[0].starter_name).replace(" ", "") if len(latest) else ""
            reason = ("schedule_unverified" if pd.isna(start) else
                      "pregame_prediction_time_missing" if pd.isna(prediction_time) else
                      "no_asof_snapshot" if latest.empty else
                      "starter_not_confirmed" if latest.iloc[0].starter_source not in {"confirmed", "manual"} else
                      "starter_identity_mismatch" if not expected or expected == "nan" or name != expected else
                      "verified")
            rows.append({"game_id": game["game_id"], "date": game["game_date"], "team": team,
                         "official_start": start, "prediction_reference_datetime": prediction_time,
                         "snapshot_time": latest.iloc[0].snapshot_time if len(latest) else None,
                         "boxscore_starter": expected, "snapshot_starter": name,
                         "status": reason, "verified": reason == "verified"})
    return pd.DataFrame(rows)


def replay_approved_model(raw, test, artifact, scratch):
    if artifact["metadata"]["prediction_unit"] != "team":
        raise ValueError("This audit implements the deployed team-model contract only")
    cutoff = pd.Timestamp(artifact["metadata"]["training_cutoff_date"])
    if pd.to_datetime(test.game_date).min() <= cutoff:
        raise ValueError("Holdout overlaps approved model training")
    rows = []
    available_columns = set()
    year = int(test.game_date.str[:4].max())
    raw = raw[pd.to_datetime(raw.date).dt.year.eq(year)]
    for day in sorted(test.game_date.unique()):
        history = raw[pd.to_datetime(raw.date).le(pd.Timestamp(day))].copy()
        today = pd.to_datetime(history.date).eq(pd.Timestamp(day))
        history.loc[today, "status"] = "Scheduled"
        history.loc[today, "result"] = ""
        history.loc[today, ["score_team", "score_opp"]] = np.nan
        path = scratch / "replay_input.csv"
        history.to_csv(path, index=False)
        features = build_features(path, include_unlabeled=True)
        features = features[pd.to_datetime(features.date).eq(pd.Timestamp(day))].copy()
        available_columns.update(features.columns)
        matrix = _team_matrix(features, artifact["schema"]["feature_order"], artifact["bundle"], artifact["schema"])
        probability = predict_bundle_probabilities(artifact["bundle"], matrix, artifact["schema"])
        features["production_probability"] = normalize_game_probabilities(features, probability)
        features["canonical_id"] = features.game_id.str.rsplit("_", n=1).str[0]
        for game in test[test.game_date.eq(day)].to_dict("records"):
            home = features[features.canonical_id.eq(game["game_id"]) & features.team.eq(game["home_team"])]
            if len(home) != 1:
                raise ValueError(f"Ambiguous production replay: {game['game_id']}")
            rows.append({"game_id": game["game_id"], "production_probability": float(home.iloc[0].production_probability)})
    return pd.DataFrame(rows), available_columns


def run_review(source: Path, artifact_root: Path, candidate_audit: Path, output: Path):
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Audit output must be outside the live source directory")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    files = {"games": "data/official/model_training_games.csv",
             "logs": "data/official/pitcher_game_logs.csv",
             "snapshots": "data/official/pitching_daily_snapshot.csv",
             "schedule": "data/official/pitching_snapshot_schedule.csv",
             "history": "modeling/results/pregame_prediction_history.csv",
             "store": "modeling/results/pregame_matchup_feature_store.csv"}
    source_hashes = {}
    frames = {}
    for name, relative in files.items():
        destination = inputs / Path(relative).name
        shutil.copy2(source / relative, destination)
        source_hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
        frames[name] = pd.read_csv(destination)
    shutil.copy2(candidate_audit, inputs / "selection_audit.json")
    artifact_copy = output / "artifact/production/current"
    shutil.copytree(artifact_root / "production/current", artifact_copy)
    artifact = load_production_artifact(output / "artifact")
    frozen = json.loads((inputs / "selection_audit.json").read_text())
    store = attach_pitcher_workload_features(frames["store"], build_pitcher_workload_features(frames["logs"]))
    store = store.dropna(subset=["home_win"])
    store = store[store.game_date.str.startswith(str(frozen["season"]))].sort_values(["game_date", "game_id"])
    train = store[store.game_date.lt(frozen["selection_cutoff"])]
    development = store[store.game_date.lt(frozen["final_holdout_start"])]
    test = store[store.game_date.ge(frozen["final_holdout_start"]) & store.game_date.le(frozen["final_holdout_end"])].copy()
    candidate = frozen["best_candidate"]
    columns = pregame_feature_sets(train)[candidate["feature_set"]]
    probabilities = _fit_predict_pregame_model(candidate["model"], _model_specs()[candidate["model"]](),
                                              development[columns], development.home_win.astype(int), test[columns])
    test["candidate_probability"] = probabilities
    production, available = replay_approved_model(frames["games"], test, artifact, output)
    comparison = test.merge(production, on="game_id", validate="one_to_one")
    identities = audit_starter_identity(test, frames["snapshots"], frames["schedule"], frames["history"])
    verified_games = identities.groupby("game_id").verified.agg(["all", "count"])
    verified_games = verified_games.index[verified_games["all"] & verified_games["count"].eq(2)]
    comparison["starter_asof_verified"] = comparison.game_id.isin(verified_games)
    baseline = probability_metrics(comparison.home_win, comparison.production_probability)
    challenger = probability_metrics(comparison.home_win, comparison.candidate_probability)
    ci = _paired_bootstrap_delta(comparison.home_win, comparison.production_probability,
                                comparison.candidate_probability, comparison.game_date)
    checks = {
        "minimum_final_holdout_100": len(test) >= 100,
        "production_same_games_compared": len(comparison) == len(test),
        "accuracy_delta_greater_than_0_005": challenger["accuracy"] - baseline["accuracy"] > 0.005,
        "brier_not_worse": challenger["brier"] <= baseline["brier"] + 0.001,
        "log_loss_not_worse": challenger["log_loss"] <= baseline["log_loss"] + 0.001,
        "bootstrap_ci_stable": ci[0] > 0,
        "over_55_accuracy_improved": challenger["over_55_accuracy"] is not None and baseline["over_55_accuracy"] is not None and challenger["over_55_accuracy"] > baseline["over_55_accuracy"],
        "calibration_not_worse": challenger["calibration_error_10_bins"] <= baseline["calibration_error_10_bins"] + 0.005,
        "all_holdout_starter_identities_asof_verified": bool(comparison.starter_asof_verified.all()),
        "candidate_features_supported_by_runtime": set(columns).issubset(available),
    }
    eligible = comparison[comparison.starter_asof_verified]
    audit = {"generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
             "production_artifact_id": artifact["metadata"]["artifact_id"],
             "production_training_cutoff": artifact["metadata"]["training_cutoff_date"],
             "comparison_type": "frozen_candidate_vs_approved_artifact_point_in_time_replay",
             "selection_cutoff": frozen["selection_cutoff"], "holdout_start": frozen["final_holdout_start"],
             "holdout_end": frozen["final_holdout_end"], "production_metrics": baseline, "candidate_metrics": challenger,
             "accuracy_delta": challenger["accuracy"] - baseline["accuracy"], "bootstrap_ci95": list(ci),
             "starter_identity_status_counts": identities.status.value_counts().to_dict(),
             "both_starters_asof_verified_games": len(eligible),
             "verified_subset_metrics": {"production": probability_metrics(eligible.home_win, eligible.production_probability),
                                         "candidate": probability_metrics(eligible.home_win, eligible.candidate_probability)} if len(eligible) else None,
             "missing_runtime_features": sorted(set(columns) - available),
             "gate_checks": {key: bool(value) for key, value in checks.items()},
             "failed_gates": [key for key, passed in checks.items() if not passed],
             "safe_to_replace_model": False, "safe_to_use_pitching_snapshot_as_features": False,
             "decision": "blocked" if not all(checks.values()) else "requires_full_segment_and_runtime_validation",
             "source_sha256": source_hashes,
             "limitations": ["This is a reconstructed replay, not an archived live probability log.",
                             "Approved win classifier only; the final hybrid_50_50 policy still needs shadow validation.",
                             "Post-hoc starter identity is not accepted without a pre-prediction snapshot.",
                             "Development data and historical core features still require full leakage/segment audits.",
                             "No promotion is performed by this audit command."]}
    comparison[["game_date", "game_id", "home_team", "away_team", "home_win", "production_probability", "candidate_probability", "starter_asof_verified"]].to_csv(output / "same_game_comparison.csv", index=False)
    identities.to_csv(output / "starter_identity_asof_audit.csv", index=False)
    (output / "promotion_review.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description="Read-only promotion review; never promotes a model")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_review(args.source_dir, args.artifact_root, args.candidate_audit, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
