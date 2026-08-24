from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from modeling.model_artifacts import (
    ArtifactValidationError,
    create_candidate_artifact,
    create_evaluation_candidate,
    load_production_artifact,
    predict_bundle_probabilities,
    promote_candidate,
    rollback_production,
    validate_available_features,
    validate_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "artifacts"
        self.frame = pd.DataFrame(
            {
                "feature_a": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 2.5, -2.5],
                "feature_b": [0.0, 0.5, -0.5, 1.0, 0.5, -1.0, 1.5, -1.5],
            }
        )
        self.target = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    def tearDown(self):
        self.temp.cleanup()

    def _candidate(self, name="candidate", calibrated=False):
        mean = self.frame.mean()
        std = self.frame.std().replace(0, 1)
        scaled = (self.frame - mean) / std
        if calibrated:
            model = CalibratedClassifierCV(LogisticRegression(max_iter=500), method="sigmoid", cv=2)
        else:
            model = LogisticRegression(max_iter=500)
        model.fit(scaled, self.target)
        bundle = {"model_type": model.__class__.__name__, "model": model, "mean": mean, "std": std}
        metadata = {
            "model_name": name,
            "model_family": model.__class__.__name__,
            "selected_candidate": name,
            "prediction_unit": "team",
            "training_start_year": 2021,
            "training_cutoff_date": "2026-07-28",
            "target_name": "target_win",
            "feature_count": 2,
        }
        schema = {
            "feature_names": list(self.frame.columns),
            "feature_order": list(self.frame.columns),
            "feature_count": 2,
            "required_features": list(self.frame.columns),
            "optional_features": ["team_A", "team_B"],
            "expected_dtype": "float64",
            "missing_value_policy": "none",
        }
        metrics = {"accuracy": 0.5, "Brier Score": 0.25, "Log Loss": 0.69}
        path = create_candidate_artifact(self.root, bundle, metadata, schema, metrics, PROJECT_ROOT)
        return path, bundle, schema

    @staticmethod
    def _rewrite_manifest_checksum(path: Path, filename: str):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256((path / filename).read_bytes()).hexdigest()
        for item in manifest["files"]:
            if item["name"] == filename:
                item["sha256"] = digest
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def test_candidate_round_trip_preserves_calibrated_probabilities_classes_and_shape(self):
        path, bundle, schema = self._candidate(calibrated=True)
        scaled = (self.frame - bundle["mean"]) / bundle["std"]
        before = bundle["model"].predict_proba(scaled)
        loaded = validate_artifact(self.root, path, expected_approval="candidate")
        after = loaded["bundle"]["model"].predict_proba(scaled)

        np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
        self.assertEqual(before.shape, after.shape)
        self.assertEqual(list(loaded["bundle"]["class_order"]), [0, 1])
        np.testing.assert_allclose(
            predict_bundle_probabilities(loaded["bundle"], scaled, schema),
            after[:, 1],
            rtol=0,
            atol=1e-12,
        )

    def test_identical_candidate_is_not_duplicated(self):
        first, _, _ = self._candidate()
        second, _, _ = self._candidate()
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.root / "candidate").iterdir())), 1)

    def test_feature_order_and_required_features_are_enforced(self):
        path, bundle, schema = self._candidate()
        loaded = validate_artifact(self.root, path, expected_approval="candidate")
        scaled = (self.frame - bundle["mean"]) / bundle["std"]
        with self.assertRaises(ArtifactValidationError):
            predict_bundle_probabilities(loaded["bundle"], scaled[["feature_b", "feature_a"]], schema)
        with self.assertRaises(ArtifactValidationError):
            predict_bundle_probabilities(loaded["bundle"], scaled[["feature_a"]], schema)
        with self.assertRaises(ArtifactValidationError):
            validate_available_features(["feature_a", "feature_b", "unknown_feature"], schema)

    def test_evaluation_schema_treats_sparse_team_categories_as_optional(self):
        mean = pd.Series(
            {"recent_5_win_rate": 0.5, "team_SK": 0.1, "opponent_넥센": 0.1}
        )
        std = pd.Series(
            {"recent_5_win_rate": 0.1, "team_SK": 0.3, "opponent_넥센": 0.3}
        )
        model = LogisticRegression(max_iter=500).fit(
            pd.DataFrame(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.5, 1.0, 0.0],
                ],
                columns=mean.index,
            ),
            [0, 1, 0, 1],
        )
        bundle = {
            "model_type": "LogisticRegression",
            "model": model,
            "mean": mean,
            "std": std,
            "available_feature_columns": list(mean.index),
            "feature_order": list(mean.index),
            "prediction_unit": "team",
            "class_order": [0, 1],
        }
        payload = {
            "feature_columns": list(mean.index),
            "selected_model": "schema-test",
            "model_type": "LogisticRegression",
            "prediction_unit": "team",
            "training_start_year": 2016,
            "prediction_training_cutoff": "2026-08-23",
        }

        path = create_evaluation_candidate(
            self.root,
            PROJECT_ROOT,
            bundle,
            payload,
            {},
            {},
            [],
        )
        schema = json.loads((path / "feature_schema.json").read_text(encoding="utf-8"))

        self.assertEqual(schema["required_features"], ["recent_5_win_rate"])
        self.assertIn("team_SK", schema["optional_features"])
        self.assertIn("opponent_넥센", schema["optional_features"])
        validate_available_features(["recent_5_win_rate"], schema)

    def test_schema_version_checksum_complete_and_approval_fail_closed(self):
        path, _, _ = self._candidate()
        schema_path = path / "feature_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["schema_version"] = "999"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        self._rewrite_manifest_checksum(path, "feature_schema.json")
        with self.assertRaises(ArtifactValidationError):
            validate_artifact(self.root, path, expected_approval="candidate")

        path, _, _ = self._candidate("checksum")
        with (path / "model.joblib").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaises(ArtifactValidationError):
            validate_artifact(self.root, path, expected_approval="candidate")

        path, _, _ = self._candidate("incomplete")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        manifest["complete"] = False
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ArtifactValidationError):
            validate_artifact(self.root, path, expected_approval="candidate")

        path, _, _ = self._candidate("unapproved")
        with self.assertRaises(ArtifactValidationError):
            validate_artifact(self.root, path, expected_approval="production")

    def test_untrusted_path_is_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        with self.assertRaises(ArtifactValidationError):
            validate_artifact(self.root, outside)

    def test_candidate_failure_keeps_existing_production(self):
        first, _, _ = self._candidate("production-one")
        promote_candidate(self.root, first.name)
        current_model = (self.root / "production" / "current" / "model.joblib").read_bytes()
        with patch("modeling.model_artifacts.joblib.dump", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self._candidate("failed-candidate")
        self.assertEqual((self.root / "production" / "current" / "model.joblib").read_bytes(), current_model)

    def test_promotion_failure_restores_existing_production(self):
        first, _, _ = self._candidate("production-one")
        promote_candidate(self.root, first.name)
        second, _, _ = self._candidate("production-two")
        current_id = load_production_artifact(self.root)["metadata"]["artifact_id"]
        real_replace = __import__("os").replace
        calls = {"count": 0}

        def fail_second(source, target):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("promotion failed")
            return real_replace(source, target)

        with patch("modeling.model_artifacts.os.replace", side_effect=fail_second):
            with self.assertRaises(OSError):
                promote_candidate(self.root, second.name)
        self.assertEqual(load_production_artifact(self.root)["metadata"]["artifact_id"], current_id)

    def test_previous_artifact_can_be_rolled_back(self):
        first, _, _ = self._candidate("production-one")
        promote_candidate(self.root, first.name)
        time.sleep(0.01)
        second, _, _ = self._candidate("production-two")
        promote_candidate(self.root, second.name)
        self.assertEqual(load_production_artifact(self.root)["metadata"]["artifact_id"], second.name)
        rollback_production(self.root)
        self.assertEqual(load_production_artifact(self.root)["metadata"]["artifact_id"], first.name)

    def test_model_binary_is_joblib_loadable_only_after_validation(self):
        path, _, _ = self._candidate()
        validated = validate_artifact(self.root, path, expected_approval="candidate")
        direct = joblib.load(path / "model.joblib")
        self.assertEqual(validated["bundle"]["feature_order"], direct["feature_order"])


if __name__ == "__main__":
    unittest.main()
