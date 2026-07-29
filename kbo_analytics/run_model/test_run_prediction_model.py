import unittest

import numpy as np

from run_prediction_model import (
    PYTHAGOREAN_WIN_CONVERSION,
    pythagorean_home_win_probability,
    select_model,
)


class RunPredictionModelTest(unittest.TestCase):
    def test_pythagorean_probability_is_symmetric_and_follows_expected_runs(self):
        probability = pythagorean_home_win_probability([5.0, 4.0], [4.0, 5.0])

        self.assertGreater(probability[0], 0.5)
        self.assertLess(probability[1], 0.5)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)

    def test_model_selection_can_choose_pythagorean_conversion(self):
        run_scores = [{"model": "Tweedie", "mae": 2.7, "rmse": 3.4}]
        win_scores = [
            {
                "model": "Tweedie",
                "win_conversion": "logistic_run_diff",
                "accuracy": 0.52,
                "brier_score": 0.250,
                "log_loss": 0.693,
            },
            {
                "model": "Tweedie",
                "win_conversion": PYTHAGOREAN_WIN_CONVERSION,
                "accuracy": 0.54,
                "brier_score": 0.248,
                "log_loss": 0.690,
            },
        ]

        selected, candidates = select_model(run_scores, win_scores)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(selected["win_conversion"], PYTHAGOREAN_WIN_CONVERSION)

if __name__ == "__main__":
    unittest.main()
