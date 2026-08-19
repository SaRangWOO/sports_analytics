from __future__ import annotations

import unittest

import pandas as pd

from modeling.pitching_snapshot_challenger import (
    build_pitching_game_features,
    canonical_snapshot_rows,
    compare_with_production_history,
)


def snapshot_row(team: str, side: str, time: str, game_id: str = "g1") -> dict:
    return {
        "snapshot_time": time,
        "reference_date": "2026-08-01",
        "team": team,
        "starter_name": f"{team} 선발",
        "starter_source": "confirmed",
        "starter_info_quality": 1.0,
        "starter_era": 3.0 if side == "H" else 4.0,
        "starter_whip": 1.1 if side == "H" else 1.3,
        "bullpen_fatigue_label": "낮음" if side == "H" else "높음",
        "recent_3day_games": 1 if side == "H" else 3,
        "scheduled_game_id": f"{game_id}_{team}",
        "home_away": side,
    }


def schedule_row(status: str = "Final", game_id: str = "g1") -> dict:
    return {
        "reference_date": "2026-08-01",
        "official_game_id": game_id,
        "scheduled_start_datetime": "2026-08-01 18:00:00",
        "status": status,
    }


class PitchingSnapshotChallengerTest(unittest.TestCase):
    def test_latest_pre_start_rows_are_selected(self):
        snapshot = pd.DataFrame(
            [
                snapshot_row("LG", "H", "2026-08-01 10:00:00"),
                snapshot_row("LG", "H", "2026-08-01 17:00:00"),
                snapshot_row("KT", "A", "2026-08-01 17:00:00"),
            ]
        )
        rows, audit = canonical_snapshot_rows(snapshot, pd.DataFrame([schedule_row()]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[rows["team"].eq("LG")].iloc[0]["snapshot_time"].hour, 17)
        self.assertEqual(audit["post_start_rows_excluded"], 0)

    def test_at_or_after_start_and_cancelled_rows_are_excluded(self):
        snapshot = pd.DataFrame(
            [
                snapshot_row("LG", "H", "2026-08-01 18:00:00"),
                snapshot_row("KT", "A", "2026-08-01 18:01:00"),
            ]
        )
        rows, audit = canonical_snapshot_rows(snapshot, pd.DataFrame([schedule_row()]))
        self.assertTrue(rows.empty)
        self.assertEqual(audit["post_start_rows_excluded"], 2)
        cancelled, audit = canonical_snapshot_rows(
            snapshot.assign(snapshot_time="2026-08-01 17:00:00"),
            pd.DataFrame([schedule_row("Cancelled")]),
        )
        self.assertTrue(cancelled.empty)
        self.assertEqual(audit["non_final_rows_excluded"], 2)

    def test_home_away_snapshot_features_have_expected_direction(self):
        snapshot = pd.DataFrame(
            [
                snapshot_row("LG", "H", "2026-08-01 17:00:00"),
                snapshot_row("KT", "A", "2026-08-01 17:00:00"),
            ]
        )
        rows, _ = canonical_snapshot_rows(snapshot, pd.DataFrame([schedule_row()]))
        features = build_pitching_game_features(rows)
        self.assertEqual(len(features), 1)
        self.assertAlmostEqual(features.iloc[0]["starter_era_gap_snapshot"], 1.0)
        self.assertAlmostEqual(features.iloc[0]["starter_whip_gap_snapshot"], 0.2)
        self.assertEqual(features.iloc[0]["both_starters_confirmed_snapshot"], 1)
        self.assertEqual(features.iloc[0]["bullpen_fatigue_label_gap_snapshot"], 2.0)

    def test_production_probability_is_converted_to_home_probability(self):
        candidate = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "home_win": 1,
                    "home_probability": 0.6,
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "run_time": "2026-08-01 17:00:00",
                    "update_stage": "pregame",
                    "game_id": "g1",
                    "home_team": "LG",
                    "predicted_team": "KT",
                    "win_probability": 0.55,
                }
            ]
        )
        comparison = compare_with_production_history(candidate, history)
        self.assertEqual(comparison["paired_games"], 1)
        self.assertEqual(comparison["production"]["accuracy"], 0.0)
        self.assertEqual(comparison["candidate"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
