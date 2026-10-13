import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from modeling.pitcher_workload_features import build_pitcher_workload_features, attach_pitcher_workload_features
from modeling.pitcher_workload_candidate_validation import _chronological_current_season_split, _paired_bootstrap_delta
from modeling.model_training import pregame_feature_sets
from modeling.production_health import production_health_checks
from run_model.run_prediction_model import chronological_oof_runs, run_pipeline, DEFAULT_RESULTS


class PriorOnlyRegressor(RegressorMixin, BaseEstimator):
    def fit(self, x, y):
        self.last_date = x["day"].max()
        return self

    def predict(self, x):
        assert x["day"].min() > self.last_date
        return np.repeat(4.0, len(x))


class ConsistencyTests(unittest.TestCase):
    def logs(self):
        rows = []
        for day in range(1, 5):
            rows.append(dict(game_id=f"g{day}", game_date=f"2026-01-0{day}", team="KT",
                             pitcher_index=1, pitcher_id=1, pitcher_name="starter", is_starter=True,
                             innings_outs=27, pitch_count=90, hits_allowed=3, walks_hbp=1, earned_runs=2))
            if day == 1:
                rows.append({**rows[-1], "pitcher_index": 2, "pitcher_id": 2,
                             "is_starter": False, "pitch_count": 30})
        return pd.DataFrame(rows)

    def test_complete_game_has_zero_previous_bullpen(self):
        result = build_pitcher_workload_features(self.logs()).set_index("game_id")
        self.assertEqual(result.loc["g3", "previous_game_bullpen_pitch_count"], 0)

    def test_current_and_same_day_results_are_not_prior(self):
        logs = self.logs()
        duplicate_day = logs.iloc[-1].copy()
        duplicate_day["game_id"] = "g4b"
        duplicate_day["earned_runs"] = 100
        result = build_pitcher_workload_features(pd.concat([logs, duplicate_day.to_frame().T])).set_index("game_id")
        self.assertEqual(result.loc["g4", "starter_recent3_era"], result.loc["g4b", "starter_recent3_era"])

    def test_missing_opponent_is_not_zero_pitching(self):
        result = attach_pitcher_workload_features(pd.DataFrame([dict(game_id="g3", home_team="KT", away_team="LG")]), build_pitcher_workload_features(self.logs()))
        self.assertTrue(np.isnan(result.loc[0, "starter_recent3_era_gap"]))
        self.assertEqual(result.loc[0, "away_starter_recent3_era_missing"], 1)
        self.assertFalse(result.loc[0, "starter_identity_asof_verified"])

    def test_feature_name_contract(self):
        sets = pregame_feature_sets(pd.DataFrame({"starter_recent3_pitch_count_gap": [1, 2, 3]}))
        self.assertIn("starter_recent3_pitch_count_gap", sets["production_plus_starter"])

    def test_temporal_selection_holdout_disjoint(self):
        frame = pd.DataFrame({"game_date": pd.date_range("2026-01-01", periods=100).repeat(2)})
        development, test, _ = _chronological_current_season_split(frame, 0.8)
        train, selection, _ = _chronological_current_season_split(development, 0.75)
        self.assertLess(train.game_date.max(), selection.game_date.min())
        self.assertLess(selection.game_date.max(), test.game_date.min())

    def test_date_block_bootstrap(self):
        low, high = _paired_bootstrap_delta([1, 0], [0.9, 0.9], [0.1, 0.1], ["same", "same"], iterations=30)
        self.assertEqual((low, high), (0, 0))

    def test_oof_uses_only_prior_dates(self):
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=20).repeat(2),
                              "day": np.repeat(np.arange(20), 2), "target_runs": 4})
        result = chronological_oof_runs(PriorOnlyRegressor(), frame, ["day"])
        self.assertEqual(result.isna().sum(), 10)
        self.assertEqual(result.dropna().unique().tolist(), [4])

    def test_oof_cannot_write_production_results(self):
        with self.assertRaises(ValueError):
            run_pipeline(Path("missing"), DEFAULT_RESULTS, 0.8, temporal_oof=True)

    def test_health_catches_mixed_identity_and_old_artifact(self):
        checks = production_health_checks({"production_artifact_id": "a", "metrics_model_id": "b",
                                          "prediction_training_cutoff": "2026-08-23"}, date(2026, 10, 13))
        self.assertEqual([item["status"] for item in checks], ["fail", "warning", "warning"])

    def test_scripts_share_lock_and_require_approved_artifact(self):
        root = Path(__file__).resolve().parents[1] / "scripts"
        for name in ("daily_kbo_update.sh", "pregame_kbo_update.sh", "refresh_pregame_context.sh"):
            self.assertIn("exec 9>/tmp/kbo_pipeline_update.lock", (root / name).read_text())
        for name in ("daily_kbo_update.sh", "pregame_kbo_update.sh"):
            text = (root / name).read_text()
            self.assertIn("predict_only_dashboard_hybrid.py", text)
            self.assertNotIn("scripts/official_kbo_dashboard_hybrid.py", text)
            self.assertIn("modeling/results run_model/results", text)


if __name__ == "__main__":
    unittest.main()
