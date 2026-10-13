import unittest

import pandas as pd

from modeling.pitcher_promotion_review import audit_starter_identity, probability_metrics


class PromotionReviewTests(unittest.TestCase):
    def audit(self, snapshot_time="2026-09-04 17:00:00", start="2026-09-04 18:30:00",
              prediction_time="2026-09-04 18:00:00", source="confirmed", name="A"):
        games = pd.DataFrame([{"game_id": "20260904LGKT0", "game_date": "2026-09-04",
                               "home_team": "KT", "away_team": "LG", "home_starter_name": "A", "away_starter_name": "B"}])
        snapshots = pd.DataFrame([{"scheduled_game_id": "20260904LGKT0_KT", "team": "KT", "reference_date": "2026-09-04",
                                  "snapshot_time": snapshot_time, "starter_source": source, "starter_name": name}])
        schedule = pd.DataFrame([{"official_game_id": "20260904LGKT0", "reference_date": "2026-09-04", "scheduled_start_datetime": start}])
        history = pd.DataFrame([{"game_id": "20260904LGKT0", "reference_date": "2026-09-04", "run_time": prediction_time}])
        return audit_starter_identity(games, snapshots, schedule, history)

    def test_only_known_before_prediction_identity_is_verified(self):
        result = self.audit()
        self.assertEqual(result.status.tolist(), ["verified", "no_asof_snapshot"])

    def test_after_prediction_is_not_accepted_even_before_start(self):
        self.assertEqual(self.audit(snapshot_time="2026-09-04 18:10:00").iloc[0].status, "no_asof_snapshot")

    def test_at_game_start_is_not_accepted(self):
        self.assertEqual(self.audit(snapshot_time="2026-09-04 18:30:00").iloc[0].status, "no_asof_snapshot")

    def test_estimated_and_changed_starters_are_not_confirmed(self):
        self.assertEqual(self.audit(source="estimated").iloc[0].status, "starter_not_confirmed")
        self.assertEqual(self.audit(name="C").iloc[0].status, "starter_identity_mismatch")

    def test_after_start_prediction_cannot_prove_asof_time(self):
        self.assertEqual(self.audit(prediction_time="2026-09-04 19:00:00").iloc[0].status, "pregame_prediction_time_missing")

    def test_missing_start_fails_closed(self):
        self.assertEqual(self.audit(start=None).iloc[0].status, "schedule_unverified")

    def test_accuracy_and_probability_metrics_use_same_games(self):
        metrics = probability_metrics([1, 0], [0.6, 0.4])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["brier"], 0.16)
        self.assertEqual(metrics["over_55_games"], 2)


if __name__ == "__main__":
    unittest.main()
