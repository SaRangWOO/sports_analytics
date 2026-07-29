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
    payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.exists() else {}
    selected_metrics = artifact["metrics"].get("selected_candidate_metrics", {})
    payload.update(
        {
            "available": True,
            "selected_model": metadata["model_name"],
            "model_type": metadata["model_family"],
            "prediction_unit": metadata["prediction_unit"],
            "prediction_training_cutoff": metadata["training_cutoff_date"],
            "training_start_year": metadata.get("training_start_year"),
            "feature_columns": schema["feature_order"],
            "accuracy": selected_metrics.get("검증 정확도", selected_metrics.get("accuracy")),
            "today_predictions": today_predictions,
            "production_artifact_id": metadata["artifact_id"],
            "production_artifact_loaded": True,
            "prediction_mode": "predict_only",
        }
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
