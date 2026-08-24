from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from modeling.pitching_snapshot_storage import (
    PitchingSnapshotValidationError,
    SNAPSHOT_COLUMNS,
    build_pitching_schedule_frame,
    canonicalize_pitching_snapshots,
    merge_pitching_snapshots,
    save_pitching_schedule,
    save_pitching_snapshot,
    validate_pitching_schedule,
    validate_pitching_snapshot,
)


def snapshot_row(
    snapshot_date: str,
    team: str,
    game_id: str,
    opponent: str,
    *,
    snapshot_time: str | None = None,
    home_away: str = "A",
) -> dict:
    return {
        "snapshot_date": snapshot_date,
        "snapshot_time": snapshot_time or f"{snapshot_date} 10:00:00",
        "reference_date": snapshot_date,
        "team": team,
        "starter_name": "starter",
        "starter_source": "confirmed",
        "starter_info_quality": 1.0,
        "starter_era": 3.5,
        "starter_whip": 1.2,
        "bullpen_fatigue_label": "낮음",
        "recent_3day_games": 1,
        "scheduled_game_id": game_id,
        "opponent": opponent,
        "home_away": home_away,
        "data_source": "KBO GameCenter",
        "note": "경기 전 수집 스냅샷",
    }


def snapshot_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def schedule_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "reference_date",
            "away_team",
            "home_team",
            "scheduled_start_datetime",
            "official_game_id",
        ],
    )


def schedule_row(
    game_date: str,
    official_game_id: str,
    away_team: str = "KT",
    home_team: str = "NC",
    start_time: str = "18:30:00",
) -> dict:
    return {
        "reference_date": game_date,
        "away_team": away_team,
        "home_team": home_team,
        "scheduled_start_datetime": f"{game_date} {start_time}",
        "official_game_id": official_game_id,
    }


class PitchingSnapshotStorageTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output = self.root / "pitching_daily_snapshot.csv"
        self.backups = self.root / "backups"
        self.as_of_date = date(2026, 7, 30)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_existing(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.output, index=False, encoding="utf-8-sig")

    def test_new_date_is_merged_and_saved(self):
        existing = snapshot_frame(
            [snapshot_row("2026-07-29", "KT", "20260729KTLG0_KT", "LG")]
        )
        new_frame = snapshot_frame(
            [snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성")]
        )
        self.write_existing(existing)

        merged = merge_pitching_snapshots(
            existing,
            new_frame,
            as_of_date=self.as_of_date,
        )
        backup = save_pitching_snapshot(
            merged,
            self.output,
            as_of_date=self.as_of_date,
            backup_dir=self.backups,
        )

        saved = pd.read_csv(self.output)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved["snapshot_date"].nunique(), 2)
        self.assertIsNotNone(backup)
        self.assertTrue(backup.exists())

    def test_existing_date_removal_is_rejected(self):
        existing = snapshot_frame(
            [
                snapshot_row("2026-07-29", "KT", "20260729KTLG0_KT", "LG"),
                snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성"),
            ]
        )
        self.write_existing(existing)
        candidate = snapshot_frame(
            [snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성")]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "existing_dates_missing",
        ):
            save_pitching_snapshot(
                candidate,
                self.output,
                as_of_date=self.as_of_date,
                backup_dir=self.backups,
            )

    def test_row_count_decrease_is_rejected(self):
        existing = snapshot_frame(
            [
                snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성"),
                snapshot_row(
                    "2026-07-30",
                    "삼성",
                    "20260730KTSS0_삼성",
                    "KT",
                    home_away="H",
                ),
            ]
        )
        self.write_existing(existing)
        candidate = snapshot_frame(
            [snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성")]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "row_count_decrease",
        ):
            save_pitching_snapshot(
                candidate,
                self.output,
                as_of_date=self.as_of_date,
                backup_dir=self.backups,
            )

    def test_duplicate_key_is_rejected(self):
        row = snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성")
        duplicate = snapshot_frame([row, row.copy()])

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "duplicate_key",
        ):
            save_pitching_snapshot(
                duplicate,
                self.output,
                as_of_date=self.as_of_date,
                backup_dir=self.backups,
            )

    def test_future_date_is_rejected(self):
        future = snapshot_frame(
            [snapshot_row("2026-07-31", "KT", "20260731KTSS0_KT", "삼성")]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "future_snapshot_date",
        ):
            save_pitching_snapshot(
                future,
                self.output,
                as_of_date=self.as_of_date,
                backup_dir=self.backups,
            )

    def test_validation_failure_preserves_hash_and_mtime(self):
        existing = snapshot_frame(
            [snapshot_row("2026-07-30", "KT", "20260730KTSS0_KT", "삼성")]
        )
        self.write_existing(existing)
        before_hash = hashlib.sha256(self.output.read_bytes()).hexdigest()
        before_mtime = self.output.stat().st_mtime_ns
        future = snapshot_frame(
            [snapshot_row("2026-07-31", "KT", "20260731KTSS0_KT", "삼성")]
        )

        with self.assertRaises(PitchingSnapshotValidationError):
            save_pitching_snapshot(
                future,
                self.output,
                as_of_date=self.as_of_date,
                backup_dir=self.backups,
            )

        self.assertEqual(
            hashlib.sha256(self.output.read_bytes()).hexdigest(),
            before_hash,
        )
        self.assertEqual(self.output.stat().st_mtime_ns, before_mtime)
        self.assertFalse(self.backups.exists())

    def test_official_and_fallback_ids_map_to_one_canonical_row(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC0_KT",
                    "NC",
                    snapshot_time="2026-07-30 17:00:00",
                ),
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "2026-07-30_KT_NC_KT",
                    "NC",
                    snapshot_time="2026-07-30 18:00:00",
                ),
            ]
        )
        schedule = schedule_frame(
            [schedule_row("2026-07-30", "20260730KTNC0")]
        )

        canonical = canonicalize_pitching_snapshots(
            raw,
            schedule,
            prediction_reference_datetime=pd.Timestamp(
                "2026-07-30 18:00:00"
            ).to_pydatetime(),
        )

        self.assertEqual(len(canonical), 1)
        self.assertEqual(
            canonical.iloc[0]["scheduled_game_id"],
            "20260730KTNC0_KT",
        )
        self.assertEqual(
            canonical.iloc[0]["snapshot_time"],
            "2026-07-30 18:00:00",
        )

    def test_identical_pitcher_values_do_not_create_fallback_duplicate(self):
        row = snapshot_row(
            "2026-07-30",
            "KT",
            "20260730KTNC0_KT",
            "NC",
            snapshot_time="2026-07-30 17:00:00",
        )
        fallback = row.copy()
        fallback["scheduled_game_id"] = "2026-07-30_KT_NC_KT"
        fallback["snapshot_time"] = "2026-07-30 17:30:00"

        canonical = canonicalize_pitching_snapshots(
            snapshot_frame([row, fallback]),
            schedule_frame(
                [schedule_row("2026-07-30", "20260730KTNC0")]
            ),
            prediction_reference_datetime=pd.Timestamp(
                "2026-07-30 17:30:00"
            ).to_pydatetime(),
        )

        self.assertEqual(len(canonical), 1)

    def test_post_start_snapshot_is_rejected(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC0_KT",
                    "NC",
                    snapshot_time="2026-07-30 18:31:00",
                )
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "snapshot_not_before_game_start",
        ):
            canonicalize_pitching_snapshots(
                raw,
                schedule_frame(
                    [schedule_row("2026-07-30", "20260730KTNC0")]
                ),
                prediction_reference_datetime=pd.Timestamp(
                    "2026-07-30 18:31:00"
                ).to_pydatetime(),
            )

    def test_snapshot_at_start_time_is_rejected(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC0_KT",
                    "NC",
                    snapshot_time="2026-07-30 18:30:00",
                )
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "snapshot_not_before_game_start",
        ):
            canonicalize_pitching_snapshots(
                raw,
                schedule_frame(
                    [schedule_row("2026-07-30", "20260730KTNC0")]
                ),
                prediction_reference_datetime=pd.Timestamp(
                    "2026-07-30 18:30:00"
                ).to_pydatetime(),
            )

    def test_latest_pre_start_snapshot_is_selected(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC0_KT",
                    "NC",
                    snapshot_time="2026-07-30 16:00:00",
                ),
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "2026-07-30_KT_NC_KT",
                    "NC",
                    snapshot_time="2026-07-30 18:20:00",
                ),
            ]
        )

        canonical = canonicalize_pitching_snapshots(
            raw,
            schedule_frame(
                [schedule_row("2026-07-30", "20260730KTNC0")]
            ),
            prediction_reference_datetime=pd.Timestamp(
                "2026-07-30 18:20:00"
            ).to_pydatetime(),
        )

        self.assertEqual(
            canonical.iloc[0]["snapshot_time"],
            "2026-07-30 18:20:00",
        )

    def test_same_row_count_with_replaced_key_is_rejected(self):
        existing = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC0_KT",
                    "NC",
                )
            ]
        )
        replacement = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC1_KT",
                    "NC",
                )
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "existing_canonical_keys_missing",
        ):
            validate_pitching_snapshot(
                replacement,
                as_of_date=self.as_of_date,
                existing_frame=existing,
            )

    def test_official_id_replaced_by_fallback_is_rejected(self):
        fallback = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "2026-07-30_KT_NC_KT",
                    "NC",
                )
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "noncanonical_game_id",
        ):
            validate_pitching_snapshot(
                fallback,
                as_of_date=self.as_of_date,
            )

    def test_missing_schedule_mapping_is_rejected(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "2026-07-30_KT_NC_KT",
                    "NC",
                )
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "schedule_mapping_count:0:0",
        ):
            canonicalize_pitching_snapshots(
                raw,
                schedule_frame(
                    [
                        schedule_row(
                            "2026-07-30",
                            "20260730LGSS0",
                            away_team="LG",
                            home_team="삼성",
                        )
                    ]
                ),
                prediction_reference_datetime=pd.Timestamp(
                    "2026-07-30 10:00:00"
                ).to_pydatetime(),
            )

    def test_ambiguous_schedule_mapping_is_rejected(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "2026-07-30_KT_NC_KT",
                    "NC",
                )
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "schedule_mapping_count:0:2",
        ):
            canonicalize_pitching_snapshots(
                raw,
                schedule_frame(
                    [
                        schedule_row("2026-07-30", "20260730KTNC1"),
                        schedule_row("2026-07-30", "20260730KTNC2"),
                    ]
                ),
                prediction_reference_datetime=pd.Timestamp(
                    "2026-07-30 10:00:00"
                ).to_pydatetime(),
            )

    def test_doubleheader_official_ids_remain_distinct(self):
        raw = snapshot_frame(
            [
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC1_KT",
                    "NC",
                    snapshot_time="2026-07-30 10:00:00",
                ),
                snapshot_row(
                    "2026-07-30",
                    "KT",
                    "20260730KTNC2_KT",
                    "NC",
                    snapshot_time="2026-07-30 10:00:00",
                ),
            ]
        )

        canonical = canonicalize_pitching_snapshots(
            raw,
            schedule_frame(
                [
                    schedule_row(
                        "2026-07-30",
                        "20260730KTNC1",
                        start_time="14:00:00",
                    ),
                    schedule_row(
                        "2026-07-30",
                        "20260730KTNC2",
                        start_time="18:00:00",
                    ),
                ]
            ),
            prediction_reference_datetime=pd.Timestamp(
                "2026-07-30 10:00:00"
            ).to_pydatetime(),
        )

        self.assertEqual(len(canonical), 2)

    def test_schedule_frame_preserves_official_start_time(self):
        frame = build_pitching_schedule_frame(
            [
                {
                    "G_DT": "20260820",
                    "G_TM": "19:00",
                    "G_ID": "20260820KTLG0",
                    "AWAY_NM": "KT",
                    "HOME_NM": "LG",
                }
            ]
        )

        self.assertEqual(frame.loc[0, "reference_date"], "2026-08-20")
        self.assertEqual(
            frame.loc[0, "scheduled_start_datetime"],
            pd.Timestamp("2026-08-20 19:00:00"),
        )

    def test_duplicate_schedule_key_is_rejected(self):
        duplicate = schedule_frame(
            [
                schedule_row("2026-07-30", "20260730KTNC0"),
                schedule_row("2026-07-30", "20260730KTNC0"),
            ]
        )

        with self.assertRaisesRegex(
            PitchingSnapshotValidationError,
            "duplicate_schedule_key:1",
        ):
            validate_pitching_schedule(duplicate)

    def test_schedule_save_is_atomic_and_keeps_doubleheader(self):
        output = self.root / "pitching_snapshot_schedule.csv"
        first = schedule_frame(
            [schedule_row("2026-07-30", "20260730KTNC1", start_time="14:00:00")]
        )
        second = schedule_frame(
            [schedule_row("2026-07-30", "20260730KTNC2", start_time="18:00:00")]
        )

        save_pitching_schedule(first, output)
        save_pitching_schedule(second, output)
        saved = pd.read_csv(output)

        self.assertEqual(len(saved), 2)
        self.assertEqual(set(saved["official_game_id"]), {"20260730KTNC1", "20260730KTNC2"})
        self.assertFalse(list(self.root.glob(".pitching_snapshot_schedule.csv.*.tmp")))


if __name__ == "__main__":
    unittest.main()
