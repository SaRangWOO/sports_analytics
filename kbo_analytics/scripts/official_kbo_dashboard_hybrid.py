from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import official_kbo_dashboard as dashboard
from modeling.hybrid_probability_policy import apply_probability_policy


BASE_EVALUATE_MODEL = dashboard.run_model_evaluation
BASE_RUN_EXPECTED_RUNS = dashboard.run_expected_runs_pipeline


def operational_policy() -> str:
    configured = os.environ.get("KBO_WIN_PROBABILITY_POLICY")
    if configured:
        return configured
    config_path = PROJECT_DIR / "config/probability_policy.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["policy"]


def evaluate_with_operational_policy(
    training_games,
    current_games,
    cutoff,
    prediction_date,
    data_dir,
    results_dir,
):
    payload = BASE_EVALUATE_MODEL(
        training_games,
        current_games,
        cutoff,
        prediction_date,
        data_dir,
        results_dir,
    )
    BASE_RUN_EXPECTED_RUNS(
        dashboard.RUN_MODEL_INPUT,
        dashboard.RUN_MODEL_RESULTS,
        0.8,
        prediction_date.isoformat(),
        dashboard.RUN_MODEL_SCHEDULE_INPUT,
    )
    policy = operational_policy()
    payload = apply_probability_policy(
        payload,
        dashboard.RUN_MODEL_RESULTS / "today_expected_runs_predictions.csv",
        policy,
    )
    model_path = dashboard.RESULTS_DIR / "win_predictor_model.json"
    model_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    if operational_policy() != "production_only":
        dashboard.run_model_evaluation = evaluate_with_operational_policy
        dashboard.run_expected_runs_pipeline = lambda *args, **kwargs: None
    dashboard.main()
