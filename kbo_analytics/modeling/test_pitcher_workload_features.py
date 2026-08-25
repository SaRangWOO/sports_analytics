import unittest

import pandas as pd

from modeling.pitcher_workload_features import attach_pitcher_workload_features, build_pitcher_workload_features


class PitcherWorkloadFeaturesTest(unittest.TestCase):
    def test_features_use_only_prior_games(self):
        rows = []
        for index, day in enumerate([1, 6, 11, 16], start=1):
            game_id = f"202604{day:02d}KTSS0"
            rows.append({
                "game_date": f"2026-04-{day:02d}", "game_id": game_id, "team": "KT", "pitcher_index": 1,
                "pitcher_id": 50001, "pitcher_name": "선발A", "is_starter": True, "innings_outs": 18,
                "pitch_count": 80 + index, "hits_allowed": 3, "walks_hbp": 1, "earned_runs": index,
            })
            rows.append({
                "game_date": f"2026-04-{day:02d}", "game_id": game_id, "team": "KT", "pitcher_index": 2,
                "pitcher_id": None, "pitcher_name": "불펜A", "is_starter": False, "innings_outs": 3,
                "pitch_count": 20 + index, "hits_allowed": 1, "walks_hbp": 0, "earned_runs": 0,
            })
        features = build_pitcher_workload_features(pd.DataFrame(rows))
        fourth = features[features["game_id"].eq("20260416KTSS0")].iloc[0]
        self.assertAlmostEqual(fourth["starter_recent3_era"], 3.0)
        self.assertEqual(fourth["starter_rest_days"], 5)
        self.assertAlmostEqual(fourth["starter_recent3_pitch_count"], 82.0)
        self.assertEqual(fourth["bullpen_pitch_count_last1d"], 0.0)
        self.assertEqual(fourth["bullpen_pitch_count_last3d"], 0.0)

    def test_attach_replaces_placeholder_columns(self):
        game = pd.DataFrame([
            {
                "game_id": "20260416KTSS0",
                "home_team": "KT",
                "away_team": "삼성",
                "home_starter_recent3_era": None,
                "away_starter_recent3_era": None,
            }
        ])
        workload = pd.DataFrame([
            {"game_id": "20260416KTSS0", "team": "KT", "starter_name": "선발A", "starter_recent3_era": 3.0},
            {"game_id": "20260416KTSS0", "team": "삼성", "starter_name": "선발B", "starter_recent3_era": 4.0},
        ])
        result = attach_pitcher_workload_features(game, workload)
        self.assertEqual(result.loc[0, "home_starter"], "선발A")
        self.assertEqual(result.loc[0, "away_starter"], "선발B")
        self.assertEqual(result.loc[0, "starter_recent3_era_gap"], -1.0)


if __name__ == "__main__":
    unittest.main()
