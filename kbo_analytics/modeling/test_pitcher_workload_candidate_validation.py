import unittest

import numpy as np

from modeling.pitcher_workload_candidate_validation import _paired_bootstrap_delta


class PitcherWorkloadCandidateValidationTest(unittest.TestCase):
    def test_unstable_accuracy_delta_crosses_zero(self):
        target = np.array([0, 1] * 50)
        baseline = np.where(target == 1, 0.51, 0.49).astype(float)
        candidate = baseline.copy()
        candidate[:10] = 1 - candidate[:10]
        low, high = _paired_bootstrap_delta(target, baseline, candidate, iterations=500)
        self.assertLessEqual(low, 0)
        self.assertLessEqual(high, 0)


if __name__ == "__main__":
    unittest.main()
