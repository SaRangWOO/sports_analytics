from __future__ import annotations

import unittest

import pandas as pd

from modeling.pitching_snapshot_candidate_validation import attach_snapshot_features


def feature_rows(game_id: str, date: str, away_team: str, home_team: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "game_id": f"{game_id}_{away_team}",
                "team": away_team,
                "opponent": home_team,
                "is_home": 0,
                "target_win": 1,
            },
            {
                "date": date,
                "game_id": f"{game_id}_{home_team}",
                "team": home_team,
                "opponent": away_team,
                "is_home": 1,
                "target_win": 0,
            },
        ]
    )


def snapshot_rows(
    game_id: str,
    date: str,
    snapshot_time: str,
    away_team: str,
    home_team: str,
) -> pd.DataFrame:
    rows = []
    for team, opponent, home_away in (
        (away_team, home_team, "A"),
        (home_team, away_team, "H"),
    ):
        rows.append(
            {
                "snapshot_date": date,
                "snapshot_time": snapshot_time,
                "reference_date": date,
                "team": team,
                "starter_name": "starter",
                "starter_source": "confirmed",
                "starter_info_quality": 1.0,
                "starter_era": 3.5,
                "starter_whip": 1.2,
                "bullpen_fatigue_label": "낮음",
                "recent_3day_games": 1,
                "scheduled_game_id": f"{game_id}_{team}",
                "opponent": opponent,
                "home_away": home_away,
                "data_source": "KBO GameCenter",
                "note": "경기 전 수집 스냅샷",
            }
        )
    return pd.DataFrame(rows)


def schedule_row(game_id: str, date: str, start_time: str, away_team: str, home_team: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_date": date,
                "official_game_id": game_id,
                "away_team": away_team,
                "home_team": home_team,
                "scheduled_start_datetime": f"{date} {start_time}",
            }
        ]
    )


class PitchingSnapshotCandidateValidationTest(unittest.TestCase):
    def test_1830_snapshot_is_valid_for_1900_game(self):
        game_id = "20260820KTLG0"
        frame, audit = attach_snapshot_features(
            feature_rows(game_id, "2026-08-20", "KT", "LG"),
            snapshot_rows(game_id, "2026-08-20", "2026-08-20 18:30:00", "KT", "LG"),
            schedule_row(game_id, "2026-08-20", "19:00:00", "KT", "LG"),
        )

        self.assertEqual(frame["actual_game_id"].nunique(), 1)
        self.assertEqual(audit["snapshot_at_or_after_start_rows"], 0)

    def test_snapshot_at_game_start_is_excluded(self):
        game_id = "20260823HTWO0"
        frame, audit = attach_snapshot_features(
            feature_rows(game_id, "2026-08-23", "KIA", "키움"),
            snapshot_rows(game_id, "2026-08-23", "2026-08-23 14:00:00", "KIA", "키움"),
            schedule_row(game_id, "2026-08-23", "14:00:00", "KIA", "키움"),
        )

        self.assertTrue(frame.empty)
        self.assertEqual(audit["snapshot_at_or_after_start_rows"], 2)


if __name__ == "__main__":
    unittest.main()
