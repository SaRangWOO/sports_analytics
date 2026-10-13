from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .model_artifacts import load_production_artifact
from .model_training import game_prediction_reason, prediction_reason
from .prediction_runtime import generate_today_predictions


def run_predict_only(
    current_games: pd.DataFrame,
    prediction_date: date,
    data_dir: str | Path,
    results_dir: str | Path,
    artifact_root: str | Path,
) -> dict:
    artifact = load_production_artifact(artifact_root)
    metadata = artifact["metadata"]
    schema = artifact["schema"]
    bundle = artifact["bundle"]
    today_predictions = generate_today_predictions(
        current_games=current_games,
        prediction_date=prediction_date,
        data_dir=data_dir,
        feature_order=schema["feature_order"],
        prediction_unit=metadata["prediction_unit"],
        prediction_bundle=bundle,
        team_reason=prediction_reason,
        game_reason=game_prediction_reason,
        feature_schema=schema,
    )

    results_dir = Path(results_dir)
    payload_path = results_dir / "win_predictor_model.json"
    # Never inherit evaluation fields from a different training run.
    payload = dict(artifact["metrics"].get("evaluation_context", {}))
    selected_metrics = artifact["metrics"].get("selected_candidate_metrics", {})
    completed_dates = pd.Series(dtype="datetime64[ns]")
    if {"status", "date"}.issubset(current_games.columns):
        completed_dates = pd.to_datetime(current_games.loc[current_games["status"].eq("Final"), "date"])
        completed_dates = completed_dates[completed_dates.dt.date < prediction_date]
    for name in ("confidence_metrics", "calibration_table", "high_confidence_backtest_summary",
                 "selected_model_probability_spread", "confidence_thresholds", "confidence_bucket_performance"):
        value = artifact["metrics"].get(name)
        if value is not None:
            payload[name] = value
    payload.update(
        {
            "available": True,
            "selected_model": metadata["model_name"],
            "model_type": metadata["model_family"],
            "prediction_unit": metadata["prediction_unit"],
            "model_id": metadata["artifact_id"],
            "metrics_model_id": metadata["artifact_id"],
            "metrics_scope": "approved_artifact_validation_not_live_hybrid_accuracy",
            "latest_completed_game_date_used": completed_dates.max().date().isoformat() if len(completed_dates) else None,
            "prediction_reference_date": prediction_date.isoformat(),
            "source_note": "Validation metrics belong to the approved artifact. Current-game history updates features, not fitted weights. Hybrid live performance requires a separate outcome audit.",
            "validation_period_status": "recorded" if payload.get("validation_start_date") and payload.get("validation_end_date") else "not_recorded_in_legacy_artifact",
            "training_cutoff": metadata["training_cutoff_date"],
            "prediction_training_cutoff": metadata["training_cutoff_date"],
            "training_start_year": metadata.get("training_start_year"),
            "feature_columns": schema["feature_order"],
            "accuracy": selected_metrics.get("검증 정확도", selected_metrics.get("accuracy")),
            "brier": selected_metrics.get("Brier Score", selected_metrics.get("brier")),
            "log_loss": selected_metrics.get("Log Loss", selected_metrics.get("log_loss")),
            "today_predictions": today_predictions,
            "production_artifact_id": metadata["artifact_id"],
            "production_artifact_loaded": True,
            "prediction_mode": "predict_only",
        }
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(payload_path)
    return payload
