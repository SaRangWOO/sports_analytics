from datetime import date


def production_health_checks(payload: dict, reference_date: date) -> list[dict]:
    model_id = payload.get("production_artifact_id")
    spread = payload.get("selected_model_probability_spread", {})
    consistent = bool(model_id) and payload.get("metrics_model_id") == model_id
    consistent = consistent and payload.get("model_id") == model_id
    consistent = consistent and spread.get("model", payload.get("selected_model")) == payload.get("selected_model")
    consistent = consistent and spread.get("accuracy", payload.get("accuracy")) == payload.get("accuracy")
    cutoff = payload.get("prediction_training_cutoff")
    try:
        age = (reference_date - date.fromisoformat(cutoff)).days
    except (TypeError, ValueError):
        age = None
    return [
        {"check_name": "production_metric_identity_check", "status": "pass" if consistent else "fail",
         "severity": "blocking", "detail": f"artifact={model_id}, metrics={payload.get('metrics_model_id')}",
         "recommended_action": "Use evaluation metrics from the approved artifact only."},
        {"check_name": "production_artifact_age_check", "status": "pass" if age is not None and 0 <= age <= 30 else "warning",
         "severity": "warning", "detail": f"training_cutoff={cutoff}, age_days={age}, warning_after_days=30",
         "recommended_action": "Review recent-season evidence; do not automatically promote a model."},
        {"check_name": "production_validation_period_check", "status": "pass" if payload.get("validation_period_status") == "recorded" else "warning",
         "severity": "warning", "detail": str(payload.get("validation_period_status", "missing")),
         "recommended_action": "Legacy artifacts need verified validation dates; never infer them from another run."},
    ]
