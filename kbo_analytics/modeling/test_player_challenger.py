from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from automation.player_challenger import (
    PLAYER_OUTPUT_FILES,
    build_player_feature_outputs,
    source_paths,
)
from modeling.player_challenger import (
    calibration_error,
    challenger_gate,
    contribution_report,
    evaluate_challengers,
    expanding_splits,
    feature_sets,
    fit_full_challenger,
    metric_row,
    write_csv_atomic,
    write_json_atomic,
)
from modeling.player_feature_pipeline import (
    BASELINE_FEATURES,
    LINEUP_FEATURES,
    PITCHING_FEATURES,
    batter_rates,
    build_snapshot_game_features,
    load_player_feature_config,
    normalize_team,
    official_game_key,
    pitcher_rates,
    player_feature_leakage_audit,
    player_feature_schema,
    recent_player_rates,
    select_as_of_player_rows,
    shrink_rate,
    validate_player_ids,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "player_feature_model.yaml"


def player_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "date": "2026-07-01",
                "player_id": "KT01",
                "player_name": "타자1",
                "player_type": "batter",
                "team": "KT",
                "opponent": "LG",
                "plate_appearances": 4,
                "at_bats": 4,
                "hits": 2,
                "doubles": 1,
                "triples": 0,
                "home_runs": 0,
                "walks": 0,
                "strikeouts": 1,
                "innings_pitched": 0,
                "earned_runs": 0,
                "walks_allowed": 0,
                "hits_allowed": 0,
                "strikeouts_pitched": 0,
            },
            {
                "game_id": "g2",
                "date": "2026-07-02",
                "player_id": "KT01",
                "player_name": "타자1",
                "player_type": "batter",
                "team": "KT",
                "opponent": "Samsung",
                "plate_appearances": 4,
                "at_bats": 3,
                "hits": 1,
                "doubles": 0,
                "triples": 0,
                "home_runs": 1,
                "walks": 1,
                "strikeouts": 0,
                "innings_pitched": 0,
                "earned_runs": 0,
                "walks_allowed": 0,
                "hits_allowed": 0,
                "strikeouts_pitched": 0,
            },
        ]
    )


def roster() -> pd.DataFrame:
    rows = []
    for team in ["KT", "LG"]:
        for order in range(1, 10):
            rows.append(
                {
                    "player_id": f"{team}{order:02d}",
                    "team_ko": team,
                    "player_name": f"{team}타자{order}",
                    "role": "batter",
                }
            )
    return pd.DataFrame(rows)


def game_sources(game_count: int = 1):
    games = []
    pitching = []
    lineup = []
    mapping = roster()
    start = pd.Timestamp("2026-06-01")
    for index in range(game_count):
        day = start + pd.Timedelta(days=index)
        game_id = f"{day:%Y%m%d}KTLG0"
        home_win = index % 2 == 0
        games.extend(
            [
                {
                    "game_id": f"{game_id}_KT",
                    "date": day.date(),
                    "team": "KT",
                    "opponent": "LG",
                    "home_away": "H",
                    "status": "Final",
                    "result": "Win" if home_win else "Loss",
                    "score_team": 5 if home_win else 2,
                    "score_opp": 2 if home_win else 5,
                },
                {
                    "game_id": f"{game_id}_LG",
                    "date": day.date(),
                    "team": "LG",
                    "opponent": "KT",
                    "home_away": "A",
                    "status": "Final",
                    "result": "Loss" if home_win else "Win",
                    "score_team": 2 if home_win else 5,
                    "score_opp": 5 if home_win else 2,
                },
            ]
        )
        for team, opponent, side, era in [("KT", "LG", "H", 3.2), ("LG", "KT", "A", 4.1)]:
            pitching.append(
                {
                    "snapshot_time": f"{day:%Y-%m-%d} 12:00:00",
                    "reference_date": day.date(),
                    "scheduled_game_id": game_id,
                    "team": team,
                    "opponent": opponent,
                    "home_away": side,
                    "starter_name": f"{team}선발",
                    "starter_source": "confirmed",
                    "starter_info_quality": 1.0,
                    "starter_era": era,
                    "starter_whip": 1.2,
                    "bullpen_fatigue_label": "낮음" if team == "KT" else "보통",
                    "recent_3day_games": 1 if team == "KT" else 2,
                }
            )
            for order in range(1, 10):
                lineup.append(
                    {
                        "snapshot_time": f"{day:%Y-%m-%d} 15:00:00",
                        "reference_date": day.date(),
                        "scheduled_game_id": game_id,
                        "team": team,
                        "lineup_source": "confirmed",
                        "lineup_info_quality": 1.0,
                        "batting_order": order,
                        "player": f"{team}타자{order}",
                        "war": 1.0 + order / 10,
                    }
                )
    return pd.DataFrame(games), pd.DataFrame(pitching), pd.DataFrame(lineup), mapping


def challenger_frame(count: int = 36) -> pd.DataFrame:
    rows = []
    columns = BASELINE_FEATURES + PITCHING_FEATURES + LINEUP_FEATURES
    for index in range(count):
        signal = (index % 7 - 3) / 5
        row = {
            "date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=index),
            "official_game_id": f"g{index:03d}",
            "home_team": "KT",
            "away_team": "LG",
            "target_home_win": int(index % 3 != 0),
        }
        row.update({column: signal + position / 100 for position, column in enumerate(columns)})
        rows.append(row)
    return pd.DataFrame(rows)


def lineup_player_stats() -> pd.DataFrame:
    rows = []
    for team, hits in [("KT", 3), ("LG", 1)]:
        for order in range(1, 10):
            rows.append(
                {
                    "game_id": f"prior-{team}",
                    "date": "2026-05-20",
                    "player_id": f"{team}{order:02d}",
                    "player_name": f"{team}타자{order}",
                    "team": team,
                    "opponent": "LG" if team == "KT" else "KT",
                    "plate_appearances": 4,
                    "at_bats": 4,
                    "hits": hits,
                    "doubles": 1 if hits > 1 else 0,
                    "triples": 0,
                    "home_runs": 1 if hits > 1 else 0,
                    "walks": 0,
                    "strikeouts": 1,
                    "innings_pitched": 0,
                    "pitches": 0,
                    "earned_runs": 0,
                    "walks_allowed": 0,
                    "hits_allowed": 0,
                    "strikeouts_pitched": 0,
                }
            )
    return pd.DataFrame(rows)


class PlayerFeaturePipelineTest(unittest.TestCase):
    def test_01_config_loads(self):
        self.assertEqual(load_player_feature_config(CONFIG_PATH).minimum_comparable_games, 150)

    def test_02_team_alias_normalizes(self):
        self.assertEqual(normalize_team("Samsung"), "삼성")

    def test_03_korean_team_is_stable(self):
        self.assertEqual(normalize_team("한화"), "한화")

    def test_04_official_game_key_removes_team_suffix(self):
        self.assertEqual(official_game_key("20260731KTLG0_KT"), "20260731KTLG0")

    def test_05_shrink_rate_uses_prior(self):
        self.assertAlmostEqual(shrink_rate(1.0, 10, 0.0, 10), 0.5)

    def test_06_shrink_rate_rejects_invalid_sample(self):
        with self.assertRaises(ValueError):
            shrink_rate(1.0, -1, 0.0, 10)

    def test_07_as_of_excludes_same_day_results(self):
        selected = select_as_of_player_rows(player_rows(), datetime(2026, 7, 2, 9))
        self.assertEqual(selected["game_id"].tolist(), ["g1"])

    def test_08_as_of_normalizes_opponent(self):
        selected = select_as_of_player_rows(player_rows(), datetime(2026, 7, 3, 9))
        self.assertIn("삼성", selected["opponent"].tolist())

    def test_09_as_of_rejects_invalid_date(self):
        rows = player_rows()
        rows.loc[0, "date"] = "invalid"
        with self.assertRaises(ValueError):
            select_as_of_player_rows(rows, datetime(2026, 7, 3, 9))

    def test_10_player_id_mapping_coverage(self):
        rows = player_rows()
        status = validate_player_ids(rows, pd.DataFrame({"player_id": ["KT01"]}))
        self.assertEqual(status["player_id_mapping_coverage"], 1.0)

    def test_11_player_id_mapping_failure(self):
        status = validate_player_ids(player_rows(), pd.DataFrame({"player_id": ["OTHER"]}))
        self.assertEqual(status["player_id_mapping_failure"], 2)

    def test_12_duplicate_player_rows_detected(self):
        rows = pd.concat([player_rows(), player_rows().iloc[[0]]], ignore_index=True)
        status = validate_player_ids(rows, pd.DataFrame({"player_id": ["KT01"]}))
        self.assertGreater(status["duplicate_player_rows"], 0)

    def test_13_batter_rate_is_shrunk(self):
        rates = batter_rates(player_rows())
        self.assertGreater(rates["ops"], 0)
        self.assertLess(rates["avg"], 0.5)

    def test_14_pitcher_rate_is_shrunk(self):
        rows = pd.DataFrame(
            {"innings_pitched": [6], "earned_runs": [0], "walks_allowed": [1], "hits_allowed": [2], "strikeouts_pitched": [7]}
        )
        self.assertGreater(pitcher_rates(rows)["era"], 0)

    def test_15_recent_player_games_window(self):
        selected = recent_player_rates(player_rows(), "KT01", datetime(2026, 7, 3), recent_games=1)
        self.assertEqual(selected["game_id"].tolist(), ["g2"])

    def test_16_recent_player_days_window(self):
        selected = recent_player_rates(player_rows(), "KT01", datetime(2026, 7, 3), recent_days=1)
        self.assertEqual(selected["game_id"].tolist(), ["g2"])

    def test_17_snapshot_features_build_same_game_set(self):
        sources = game_sources()
        features, coverage = build_snapshot_game_features(
            *sources, datetime(2026, 6, 2), load_player_feature_config(CONFIG_PATH)
        )
        self.assertEqual(len(features), 1)
        self.assertEqual(coverage["comparable_games"], 1)

    def test_18_snapshot_features_require_both_teams(self):
        games, pitching, lineup, mapping = game_sources()
        pitching = pitching[pitching["team"].eq("KT")]
        features, _ = build_snapshot_game_features(
            games, pitching, lineup, mapping, datetime(2026, 6, 2), load_player_feature_config(CONFIG_PATH)
        )
        self.assertTrue(features.empty)

    def test_19_snapshot_features_exclude_player_names(self):
        features, _ = build_snapshot_game_features(
            *game_sources(), datetime(2026, 6, 2), load_player_feature_config(CONFIG_PATH)
        )
        self.assertFalse(any(name in features.columns for name in ["player", "player_name", "player_id"]))

    def test_20_lineup_mapping_coverage_is_exact(self):
        features, _ = build_snapshot_game_features(
            *game_sources(), datetime(2026, 6, 2), load_player_feature_config(CONFIG_PATH)
        )
        self.assertEqual(features.iloc[0]["player_id_mapping_coverage_min"], 1.0)

    def test_21_leakage_blocks_future_stats(self):
        games, pitching, lineup, _ = game_sources()
        audit = player_feature_leakage_audit(player_rows(), pitching, lineup, datetime(2026, 7, 2))
        self.assertIn("future_stat_rows", audit["blocking_issues"])

    def test_22_leakage_blocks_unverified_lineup_start(self):
        _, pitching, lineup, _ = game_sources()
        prior = select_as_of_player_rows(player_rows(), datetime(2026, 7, 3))
        audit = player_feature_leakage_audit(prior, pitching, lineup, datetime(2026, 7, 3))
        self.assertIn("lineup_start_time_not_verifiable", audit["blocking_issues"])

    def test_23_schema_excludes_identity_columns(self):
        self.assertIn("player_name", player_feature_schema()["model_input_excludes"])

    def test_24_feature_sets_are_incremental(self):
        sets = feature_sets()
        self.assertLess(len(sets["production_baseline_proxy"]), len(sets["baseline_plus_full_player"]))

    def test_25_expanding_split_is_time_ordered(self):
        splits = expanding_splits(challenger_frame(), 20)
        self.assertTrue(all(max(train) < min(test) for train, test in splits))

    def test_26_short_frame_has_no_split(self):
        self.assertEqual(expanding_splits(challenger_frame(10), 20), [])

    def test_27_calibration_error_is_bounded(self):
        value = calibration_error(np.array([0, 1]), np.array([0.2, 0.8]))
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 1)

    def test_28_metric_probability_fields(self):
        frame = challenger_frame(4)
        row = metric_row("m", "f", frame, np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.7, 0.3]))
        self.assertEqual(row["accuracy"], 1.0)

    def test_29_challenger_evaluation_uses_equal_game_counts(self):
        metrics, _ = evaluate_challengers(challenger_frame(), minimum_train_games=20)
        self.assertEqual(metrics["games"].nunique(), 1)

    def test_30_contribution_sum_matches_probability_delta(self):
        frame = challenger_frame()
        report = contribution_report(fit_full_challenger(frame), frame.tail(3))
        self.assertTrue(np.allclose(report["contribution_sum_pct_point"], report["probability_delta_pct_point"]))

    def test_31_challenger_gate_blocks_insufficient_coverage(self):
        gate = challenger_gate(pd.DataFrame(), {"comparable_games": 0}, {"status": "pass"}, load_player_feature_config(CONFIG_PATH))
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["auto_promotion_enabled"])

    def test_32_atomic_outputs_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv_atomic(root / "x.csv", pd.DataFrame({"a": [1]}))
            write_json_atomic(root / "x.json", {"a": 1})
            self.assertEqual(pd.read_csv(root / "x.csv").iloc[0]["a"], 1)
            self.assertEqual(json.loads((root / "x.json").read_text())["a"], 1)

    def test_33_source_paths_use_mock_api_root(self):
        paths = source_paths(Path("/project"))
        self.assertEqual(paths["roster"], Path("/project/mock_api/player_roster_mapping.csv"))

    def test_34_required_output_names_are_unique(self):
        self.assertEqual(len(PLAYER_OUTPUT_FILES), len(set(PLAYER_OUTPUT_FILES.values())))

    def test_35_confirmed_lineup_has_priority_over_estimated(self):
        games, pitching, lineup, mapping = game_sources()
        estimated = lineup.assign(lineup_source="estimated", lineup_info_quality=0.5, war=0.1)
        estimated["snapshot_time"] = "2026-06-01 14:00:00"
        confirmed = lineup.assign(lineup_source="confirmed", lineup_info_quality=1.0, war=2.0)
        confirmed["snapshot_time"] = "2026-06-01 15:00:00"
        features, _ = build_snapshot_game_features(
            games,
            pitching,
            pd.concat([estimated, confirmed], ignore_index=True),
            mapping,
            datetime(2026, 6, 1, 16),
            load_player_feature_config(CONFIG_PATH),
        )
        self.assertEqual(features.iloc[0]["home_lineup_source"], "confirmed")

    def test_36_estimated_lineup_used_before_confirmation(self):
        games, pitching, lineup, mapping = game_sources()
        estimated = lineup.assign(lineup_source="estimated", lineup_info_quality=0.5)
        estimated["snapshot_time"] = "2026-06-01 14:00:00"
        confirmed = lineup.assign(lineup_source="confirmed", lineup_info_quality=1.0)
        confirmed["snapshot_time"] = "2026-06-01 17:00:00"
        features, _ = build_snapshot_game_features(
            games,
            pitching,
            pd.concat([estimated, confirmed], ignore_index=True),
            mapping,
            datetime(2026, 6, 1, 16),
            load_player_feature_config(CONFIG_PATH),
        )
        self.assertEqual(features.iloc[0]["home_lineup_source"], "estimated")

    def test_37_lineup_ops_gap_uses_prior_player_games(self):
        features, _ = build_snapshot_game_features(
            *game_sources(),
            datetime(2026, 6, 2),
            load_player_feature_config(CONFIG_PATH),
            player_stats=lineup_player_stats(),
        )
        self.assertGreater(features.iloc[0]["lineup_ops_gap"], 0)

    def test_38_starter_gap_is_home_advantage_oriented(self):
        features, _ = build_snapshot_game_features(
            *game_sources(), datetime(2026, 6, 2), load_player_feature_config(CONFIG_PATH)
        )
        self.assertGreater(features.iloc[0]["starter_era_gap"], 0)

    def test_39_bullpen_fatigue_gap_is_home_advantage_oriented(self):
        features, _ = build_snapshot_game_features(
            *game_sources(), datetime(2026, 6, 2), load_player_feature_config(CONFIG_PATH)
        )
        self.assertGreater(features.iloc[0]["bullpen_fatigue_gap"], 0)

    def test_40_doubleheader_ids_remain_distinct(self):
        games, pitching, lineup, mapping = game_sources()
        second_games = games.copy()
        second_games["game_id"] = second_games["game_id"].str.replace("KTLG0", "KTLG1", regex=False)
        second_pitching = pitching.copy()
        second_pitching["scheduled_game_id"] = second_pitching["scheduled_game_id"].str.replace("KTLG0", "KTLG1", regex=False)
        second_lineup = lineup.copy()
        second_lineup["scheduled_game_id"] = second_lineup["scheduled_game_id"].str.replace("KTLG0", "KTLG1", regex=False)
        features, _ = build_snapshot_game_features(
            pd.concat([games, second_games], ignore_index=True),
            pd.concat([pitching, second_pitching], ignore_index=True),
            pd.concat([lineup, second_lineup], ignore_index=True),
            mapping,
            datetime(2026, 6, 2),
            load_player_feature_config(CONFIG_PATH),
        )
        self.assertEqual(set(features["official_game_id"]), {"20260601KTLG0", "20260601KTLG1"})

    def test_41_contribution_probabilities_are_bounded(self):
        frame = challenger_frame()
        report = contribution_report(fit_full_challenger(frame), frame.tail(4))
        for column in ["baseline_win_probability", "final_challenger_probability"]:
            self.assertTrue(report[column].between(0, 1).all())

    def test_42_gate_requires_production_parity(self):
        gate = challenger_gate(pd.DataFrame(), {"comparable_games": 999}, {"status": "pass"}, load_player_feature_config(CONFIG_PATH))
        self.assertFalse(gate["checks"]["production_parity_verified"])

    def test_43_dry_run_does_not_write_outputs(self):
        games, pitching, lineup, mapping = game_sources()
        frames = {
            "games": games,
            "pitching": pitching,
            "lineup": lineup,
            "player_stats": lineup_player_stats(),
            "roster": mapping,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with patch(
                "automation.player_challenger.load_player_sources",
                return_value=(frames, {key: key for key in frames}),
            ):
                build_player_feature_outputs(
                    Path(temporary),
                    output,
                    datetime(2026, 6, 2),
                    CONFIG_PATH,
                    dry_run=True,
                )
            self.assertFalse(output.exists())

    def test_44_unavailable_split_features_are_declared(self):
        unavailable = player_feature_schema()["unavailable_features"]
        self.assertIn("lineup_vs_lhp_strength", unavailable)
        self.assertIn("WPA", unavailable)


if __name__ == "__main__":
    unittest.main()
