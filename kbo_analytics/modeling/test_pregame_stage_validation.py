from __future__ import annotations

import unittest

import pandas as pd

from modeling.pregame_stage_validation import (
    evaluate_stage_predictions,
    latest_stage_predictions,
)


def history_row(
    game_id: str,
    stage: str,
    run_time: str,
    predicted_team: str,
    probability: float,
) -> dict:
    return {
        "run_time": run_time,
        "reference_date": "2026-08-01",
        "update_stage": stage,
        "game_id": game_id,
        "away_team": "KT",
        "home_team": "LG",
        "predicted_team": predicted_team,
        "win_probability": probability,
        "starter_status": "estimated_only" if stage == "morning" else "both_confirmed",
        "lineup_status": "recent" if stage == "morning" else "confirmed",
    }


def completed_game(game_id: str, winner: str) -> list[dict]:
    loser = "LG" if winner == "KT" else "KT"
    return [
        {
            "game_id": f"{game_id}_{winner}",
            "status": "Final",
            "result": "Win",
            "team": winner,
        },
        {
            "game_id": f"{game_id}_{loser}",
            "status": "Final",
            "result": "Loss",
            "team": loser,
        },
    ]


class PregameStageValidationTest(unittest.TestCase):
    def test_latest_prediction_per_game_and_stage_is_used(self):
        history = pd.DataFrame(
            [
                history_row("g1", "morning", "2026-08-01 08:00", "KT", 0.52),
                history_row("g1", "pregame", "2026-08-01 17:00", "LG", 0.53),
                history_row("g1", "pregame", "2026-08-01 18:00", "KT", 0.56),
            ]
        )
        latest = latest_stage_predictions(history)
        self.assertEqual(len(latest), 2)
        pregame = latest[latest["update_stage"].eq("pregame")].iloc[0]
        self.assertEqual(pregame["predicted_team"], "KT")
        self.assertEqual(float(pregame["win_probability"]), 0.56)

    def test_stage_report_uses_one_row_per_game(self):
        history = pd.DataFrame(
            [
                history_row("g1", "morning", "2026-08-01 08:00", "LG", 0.52),
                history_row("g1", "pregame", "2026-08-01 17:00", "KT", 0.56),
                history_row("g1", "pregame", "2026-08-01 18:00", "KT", 0.57),
            ]
        )
        games = pd.DataFrame(completed_game("g1", "KT"))
        metrics, paired, summary = evaluate_stage_predictions(history, games)
        stage = metrics[metrics["dimension"].eq("update_stage")]
        self.assertEqual(stage.set_index("value").loc["pregame", "games"], 1)
        self.assertEqual(len(paired), 1)
        self.assertTrue(bool(paired.iloc[0]["direction_changed"]))
        self.assertEqual(summary["pregame_improved_games"], 1)

    def test_non_final_games_are_not_evaluated(self):
        history = pd.DataFrame(
            [history_row("g1", "morning", "2026-08-01 08:00", "KT", 0.52)]
        )
        games = pd.DataFrame(
            [
                {
                    "game_id": "g1_KT",
                    "status": "Cancelled",
                    "result": "",
                    "team": "KT",
                }
            ]
        )
        metrics, paired, summary = evaluate_stage_predictions(history, games)
        self.assertTrue(metrics.empty)
        self.assertTrue(paired.empty)
        self.assertEqual(summary["paired_games"], 0)

    def test_probability_outside_predicted_team_range_is_rejected(self):
        history = pd.DataFrame(
            [history_row("g1", "morning", "2026-08-01 08:00", "KT", 0.49)]
        )
        with self.assertRaisesRegex(ValueError, "win_probability"):
            latest_stage_predictions(history)


if __name__ == "__main__":
    unittest.main()
