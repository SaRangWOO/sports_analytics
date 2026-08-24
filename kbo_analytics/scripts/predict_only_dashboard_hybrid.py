from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import official_kbo_dashboard as dashboard
from modeling.hybrid_probability_policy import apply_probability_policy
from scripts import predict_only_dashboard


BASE_RUN_PREDICT_ONLY = predict_only_dashboard.run_predict_only


def operational_policy() -> str:
    configured = os.environ.get("KBO_WIN_PROBABILITY_POLICY")
    if configured:
        return configured
    config_path = PROJECT_DIR / "config" / "probability_policy.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["policy"]


def run_predict_only_with_policy(*args, **kwargs):
    payload = BASE_RUN_PREDICT_ONLY(*args, **kwargs)
    policy = operational_policy()
    if policy == "production_only":
        return payload
    prediction_date = args[1] if len(args) > 1 else kwargs["prediction_date"]
    dashboard.run_expected_runs_pipeline(
        dashboard.RUN_MODEL_INPUT,
        dashboard.RUN_MODEL_RESULTS,
        0.8,
        prediction_date.isoformat(),
        dashboard.RUN_MODEL_SCHEDULE_INPUT,
    )
    payload = apply_probability_policy(
        payload,
        dashboard.RUN_MODEL_RESULTS / "today_expected_runs_predictions.csv",
        policy,
    )
    (dashboard.RESULTS_DIR / "win_predictor_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    predict_only_dashboard.run_predict_only = run_predict_only_with_policy
    predict_only_dashboard.main()
