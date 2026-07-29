from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from modeling.model_artifacts import create_candidate_artifact, load_production_artifact, promote_candidate
from modeling.predict_only import run_predict_only
from modeling.prediction_runtime import PREDICTION_OUTPUT_COLUMNS, generate_today_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PredictOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.artifacts = self.base / "artifacts"
        frame = pd.DataFrame({"feature_a": [-1.0, -0.5, 0.5, 1.0]})
        target = np.array([0, 0, 1, 1])
        mean = frame.mean()
        std = frame.std().replace(0, 1)
        model = LogisticRegression(max_iter=500).fit((frame - mean) / std, target)
        bundle = {"model_type": model.__class__.__name__, "model": model, "mean": mean, "std": std}
        metadata = {
            "model_name": "fixture",
            "model_family": model.__class__.__name__,
            "selected_candidate": "fixture",
            "prediction_unit": "team",
            "training_start_year": 2021,
            "training_cutoff_date": "2026-07-28",
            "target_name": "target_win",
            "feature_count": 1,
        }
        schema = {
            "feature_names": ["feature_a"],
            "feature_order": ["feature_a"],
            "feature_count": 1,
            "required_features": ["feature_a"],
            "optional_features": [],
            "expected_dtype": "float64",
            "missing_value_policy": "none",
        }
        path = create_candidate_artifact(
            self.artifacts,
            bundle,
            metadata,
            schema,
            {"selected_candidate_metrics": {"검증 정확도": 0.5}},
            PROJECT_ROOT,
        )
        promote_candidate(self.artifacts, path.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_predict_only_does_not_call_training_and_preserves_output_contract(self):
        row = {
            "경기일": "2026-07-29",
            "기준팀": "KT",
            "상대팀": "LG",
            "예측 구단": "KT",
            "예측승률": "52.0%",
            "예측": "승리 예측",
            "예측 근거": "fixture",
        }
        results_dir = self.base / "results"
        results_dir.mkdir()
        with (
            patch("modeling.model_training.train_prediction_bundle") as training,
            patch("modeling.predict_only.generate_today_predictions", return_value=[row]),
        ):
            payload = run_predict_only(
                pd.DataFrame(),
                date(2026, 7, 29),
                self.base / "data",
                results_dir,
                self.artifacts,
            )
        training.assert_not_called()
        self.assertEqual(list(payload["today_predictions"][0]), PREDICTION_OUTPUT_COLUMNS)
        self.assertEqual(payload["prediction_mode"], "predict_only")
        self.assertTrue(payload["production_artifact_loaded"])

    def test_shared_runtime_keeps_team_probability_normalization_and_output_shape(self):
        features = pd.DataFrame(
            [
                {
                    "date": "2026-07-29",
                    "game_id": "20260729_KT_LG_KT",
                    "team": "KT",
                    "opponent": "LG",
                    "target_win": np.nan,
                    "feature_a": 1.0,
                },
                {
                    "date": "2026-07-29",
                    "game_id": "20260729_KT_LG_LG",
                    "team": "LG",
                    "opponent": "KT",
                    "target_win": np.nan,
                    "feature_a": -1.0,
                },
            ]
        )
        production = load_production_artifact(self.artifacts)
        schema = dict(production["schema"])
        schema["optional_features"] = [
            "team_KT",
            "team_LG",
            "opponent_KT",
            "opponent_LG",
        ]
        runtime_data = self.base / "runtime-data"
        runtime_data.mkdir()
        with patch("modeling.prediction_runtime.build_features", return_value=features):
            rows = generate_today_predictions(
                current_games=pd.DataFrame({"placeholder": []}),
                prediction_date=date(2026, 7, 29),
                data_dir=runtime_data,
                feature_order=["feature_a"],
                prediction_unit="team",
                prediction_bundle=production["bundle"],
                team_reason=lambda row, team: f"{team} fixture",
                game_reason=lambda row, team: f"{team} fixture",
                feature_schema=schema,
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), PREDICTION_OUTPUT_COLUMNS)
        probabilities = [float(row["예측승률"].rstrip("%")) / 100 for row in rows]
        self.assertAlmostEqual(sum(probabilities), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
