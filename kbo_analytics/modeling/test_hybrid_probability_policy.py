from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modeling.hybrid_probability_policy import apply_probability_policy


class HybridProbabilityPolicyTest(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "today_predictions": [
                {
                    "경기일": "2026-08-19",
                    "기준팀": "KT",
                    "상대팀": "LG",
                    "예측 구단": "KT",
                    "예측승률": "54.0%",
                    "예측": "승리 예측",
                    "예측 근거": "KT 최근 흐름 우위",
                },
                {
                    "경기일": "2026-08-19",
                    "기준팀": "LG",
                    "상대팀": "KT",
                    "예측 구단": "KT",
                    "예측승률": "46.0%",
                    "예측": "패배 예측",
                    "예측 근거": "KT 최근 흐름 우위",
                },
            ]
        }
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "today.csv"
        pd.DataFrame(
            [
                {
                    "date": "2026-08-19",
                    "home_team": "LG",
                    "away_team": "KT",
                    "home_expected_runs": 4.8,
                    "away_expected_runs": 5.2,
                    "home_win_probability": 0.48,
                }
            ]
        ).to_csv(self.path, index=False, encoding="utf-8-sig")

    def tearDown(self):
        self.temp.cleanup()

    def test_hybrid_policy_blends_team_probability_and_score(self):
        result = apply_probability_policy(self.payload, self.path, "hybrid_50_50")
        kt = result["today_predictions"][0]
        lg = result["today_predictions"][1]
        self.assertEqual(kt["예측승률"], "53.0%")
        self.assertEqual(lg["예측승률"], "47.0%")
        self.assertEqual(kt["예상스코어"], "5.2 - 4.8")
        self.assertEqual(result["operational_probability_policy"], "hybrid_50_50")
        self.assertEqual(kt["최종예측팀기준_기존모델승률"], "54.0%")
        self.assertEqual(lg["최종예측팀기준_기존모델승률"], "54.0%")
        self.assertEqual(kt["최종예측팀기준_득점모델승률"], "52.0%")
        self.assertEqual(lg["최종예측팀기준_득점모델승률"], "52.0%")
        self.assertEqual(lg["최종예측팀예상스코어"], "5.2 - 4.8")
        self.assertEqual(kt["모델합의상태"], "모델 합의")
        self.assertEqual(lg["모델합의상태"], "모델 합의")

    def test_production_only_preserves_predictions(self):
        result = apply_probability_policy(self.payload, self.path, "production_only")
        self.assertEqual(result["today_predictions"], self.payload["today_predictions"])
        self.assertEqual(result["operational_probability_policy"], "production_only")

    def test_missing_matchup_is_rejected(self):
        pd.DataFrame(
            [
                {
                    "date": "2026-08-19",
                    "home_team": "KIA",
                    "away_team": "한화",
                    "home_expected_runs": 5.0,
                    "away_expected_runs": 4.0,
                    "home_win_probability": 0.55,
                }
            ]
        ).to_csv(self.path, index=False, encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "not found"):
            apply_probability_policy(self.payload, self.path, "hybrid_50_50")


if __name__ == "__main__":
    unittest.main()
