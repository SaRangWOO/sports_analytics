import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_engineering import build_features


class FeatureEngineeringPriorTests(unittest.TestCase):
    def test_team_priors_do_not_cross_team_boundaries(self):
        rows = [
            self._row("2026-04-01_A_B_A", "2026-04-01", "A", "B", "H", "Win", 5, 1),
            self._row("2026-04-01_A_B_B", "2026-04-01", "B", "A", "A", "Loss", 1, 5),
            self._row("2026-04-02_A_C_A", "2026-04-02", "A", "C", "H", "Loss", 2, 3),
            self._row("2026-04-02_A_C_C", "2026-04-02", "C", "A", "A", "Win", 3, 2),
            self._row("2026-04-03_A_B_A", "2026-04-03", "A", "B", "H", "Win", 4, 2),
            self._row("2026-04-03_A_B_B", "2026-04-03", "B", "A", "A", "Loss", 2, 4),
        ]

        features = self._build(rows)
        a_second = features[(features["team"] == "A") & (features["date"] == pd.Timestamp("2026-04-02"))].iloc[0]
        b_second = features[(features["team"] == "B") & (features["date"] == pd.Timestamp("2026-04-03"))].iloc[0]

        self.assertEqual(a_second["season_win_rate_prior"], 1.0)
        self.assertEqual(a_second["season_avg_score_prior"], 5.0)
        self.assertEqual(b_second["season_win_rate_prior"], 0.0)
        self.assertEqual(b_second["season_avg_score_prior"], 1.0)

    def test_current_result_is_excluded_from_group_priors(self):
        rows = [
            self._row("2026-04-01_A_B_A", "2026-04-01", "A", "B", "H", "Win", 5, 1),
            self._row("2026-04-01_A_B_B", "2026-04-01", "B", "A", "A", "Loss", 1, 5),
            self._row("2026-04-02_A_B_A", "2026-04-02", "A", "B", "H", "Loss", 0, 9),
            self._row("2026-04-02_A_B_B", "2026-04-02", "B", "A", "A", "Win", 9, 0),
        ]

        features = self._build(rows)
        second = features[(features["team"] == "A") & (features["date"] == pd.Timestamp("2026-04-02"))].iloc[0]

        self.assertEqual(second["season_win_rate_prior"], 1.0)
        self.assertEqual(second["season_avg_score_prior"], 5.0)
        self.assertEqual(second["head_to_head_win_rate_prior"], 1.0)
        self.assertEqual(second["previous_game_run_diff"], 4.0)
        self.assertEqual(second["previous_game_run_diff_gap"], 8.0)

    @staticmethod
    def _row(game_id, game_date, team, opponent, home_away, result, score_team, score_opp):
        return {
            "game_id": game_id,
            "date": game_date,
            "team": team,
            "opponent": opponent,
            "home_away": home_away,
            "status": "Final",
            "result": result,
            "score_team": score_team,
            "score_opp": score_opp,
            "ballpark": "test",
        }

    @staticmethod
    def _build(rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            return build_features(path)


if __name__ == "__main__":
    unittest.main()
