import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from pitcher_game_log_collector import (
    innings_to_outs,
    merge_pitcher_logs,
    parse_pitcher_box_score,
    save_pitcher_logs,
)


def table(rows):
    return json.dumps({"rows": [{"row": [{"Text": value} for value in row]} for row in rows]})


class PitcherGameLogCollectorTest(unittest.TestCase):
    def setUp(self):
        self.meta = {
            "G_ID": "20260823LGHH0",
            "AWAY_NM": "LG",
            "HOME_NM": "한화",
            "T_PIT_P_ID": 50123,
            "B_PIT_P_ID": 50456,
        }
        self.pitcher = ["선발A", "선발", "승", "1", "0", "0", "6", "24", "91", "22", "4", "0", "2", "7", "1", "1", "2.50"]
        self.reliever = ["불펜A", "7.8", "", "0", "0", "0", "1 2/3", "6", "24", "5", "1", "0", "1", "2", "0", "0", "3.10"]
        self.payload = {"arrPitcher": [{"table": table([self.pitcher, self.reliever])}, {"table": table([self.pitcher])}]}

    def test_innings_to_outs(self):
        self.assertEqual(innings_to_outs("6"), 18)
        self.assertEqual(innings_to_outs("1 2/3"), 5)
        self.assertEqual(innings_to_outs("5.2"), 17)

    def test_parse_assigns_starter_id_and_pitch_count(self):
        frame = parse_pitcher_box_score(date(2026, 8, 23), self.meta, self.payload, datetime(2026, 8, 24, 8))
        away = frame[frame["team"].eq("LG")]
        self.assertEqual(len(frame), 3)
        self.assertEqual(int(away.iloc[0]["pitcher_id"]), 50123)
        self.assertEqual(int(away.iloc[1]["pitch_count"]), 24)
        self.assertEqual(int(away.iloc[1]["innings_outs"]), 5)

    def test_merge_and_atomic_save_keep_business_key(self):
        frame = parse_pitcher_box_score(date(2026, 8, 23), self.meta, self.payload, datetime(2026, 8, 24, 8))
        updated = frame.copy()
        updated.loc[0, "pitch_count"] = 92
        merged = merge_pitcher_logs(frame, updated)
        self.assertEqual(len(merged), len(frame))
        self.assertEqual(int(merged[merged["team"].eq("LG")].iloc[0]["pitch_count"]), 92)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pitcher_game_logs.csv"
            save_pitcher_logs(merged, path)
            self.assertEqual(len(pd.read_csv(path)), len(frame))


if __name__ == "__main__":
    unittest.main()
