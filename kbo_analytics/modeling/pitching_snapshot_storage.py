from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd


SNAPSHOT_COLUMNS = [
    "snapshot_date",
    "snapshot_time",
    "reference_date",
    "team",
    "starter_name",
    "starter_source",
    "starter_info_quality",
    "starter_era",
    "starter_whip",
    "bullpen_fatigue_label",
    "recent_3day_games",
    "scheduled_game_id",
    "opponent",
    "home_away",
    "data_source",
    "note",
]
SNAPSHOT_KEY = ["reference_date", "scheduled_game_id", "team"]
SCHEDULE_COLUMNS = [
    "reference_date",
    "away_team",
    "home_team",
    "scheduled_start_datetime",
    "official_game_id",
]
OFFICIAL_GAME_ID = re.compile(r"^\d{8}[A-Z]{4}\d+$")
FALLBACK_GAME_ID = re.compile(r"^\d{4}-\d{2}-\d{2}_.+")


class PitchingSnapshotValidationError(ValueError):
    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("; ".join(failures))


def read_pitching_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"scheduled_game_id": str})


def _date_values(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.date


def _game_id_parts(value: str) -> tuple[str, str]:
    if "_" not in value:
        return value, ""
    return tuple(value.rsplit("_", 1))


def _schedule_candidates(
    row: pd.Series,
    schedule_frame: pd.DataFrame,
) -> pd.DataFrame:
    reference_date = str(row["reference_date"])
    team = str(row["team"])
    opponent = str(row["opponent"])
    home_away = str(row["home_away"]).upper()
    if home_away == "A":
        away_team, home_team = team, opponent
    elif home_away == "H":
        away_team, home_team = opponent, team
    else:
        raise PitchingSnapshotValidationError(
            [f"invalid_home_away:{home_away}"]
        )
    return schedule_frame[
        schedule_frame["reference_date"].astype(str).eq(reference_date)
        & schedule_frame["away_team"].astype(str).eq(away_team)
        & schedule_frame["home_team"].astype(str).eq(home_team)
    ]


def canonicalize_pitching_snapshots(
    frame: pd.DataFrame,
    schedule_frame: pd.DataFrame,
    *,
    prediction_reference_datetime: datetime,
) -> pd.DataFrame:
    missing_schedule_columns = [
        column for column in SCHEDULE_COLUMNS if column not in schedule_frame.columns
    ]
    if missing_schedule_columns:
        raise PitchingSnapshotValidationError(
            [f"missing_schedule_columns:{','.join(missing_schedule_columns)}"]
        )
    missing_snapshot_columns = [
        column for column in SNAPSHOT_COLUMNS if column not in frame.columns
    ]
    if missing_snapshot_columns:
        raise PitchingSnapshotValidationError(
            [f"missing_required_columns:{','.join(missing_snapshot_columns)}"]
        )

    schedule = schedule_frame.copy()
    schedule["scheduled_start_datetime"] = pd.to_datetime(
        schedule["scheduled_start_datetime"],
        errors="coerce",
    )
    invalid_start_count = int(schedule["scheduled_start_datetime"].isna().sum())
    if invalid_start_count:
        raise PitchingSnapshotValidationError(
            [f"invalid_scheduled_start_datetime:{invalid_start_count}"]
        )

    canonical_rows = []
    failures: list[str] = []
    for row_index, row in frame.iterrows():
        candidates = _schedule_candidates(row, schedule)
        raw_game_id, raw_team = _game_id_parts(str(row["scheduled_game_id"]))
        if raw_team and raw_team != str(row["team"]):
            failures.append(f"game_id_team_mismatch:{row_index}")
            continue
        if OFFICIAL_GAME_ID.fullmatch(raw_game_id):
            candidates = candidates[
                candidates["official_game_id"].astype(str).eq(raw_game_id)
            ]
        elif not FALLBACK_GAME_ID.match(raw_game_id):
            failures.append(f"invalid_game_id_format:{row_index}")
            continue
        if len(candidates) != 1:
            failures.append(f"schedule_mapping_count:{row_index}:{len(candidates)}")
            continue

        schedule_row = candidates.iloc[0]
        official_game_id = str(schedule_row["official_game_id"])
        if not OFFICIAL_GAME_ID.fullmatch(official_game_id):
            failures.append(f"invalid_official_game_id:{row_index}")
            continue
        snapshot_time = pd.to_datetime(row["snapshot_time"], errors="coerce")
        scheduled_start = schedule_row["scheduled_start_datetime"]
        if pd.isna(snapshot_time):
            failures.append(f"invalid_snapshot_time:{row_index}")
            continue
        if snapshot_time > prediction_reference_datetime:
            failures.append(f"snapshot_after_prediction_reference:{row_index}")
            continue
        if snapshot_time >= scheduled_start:
            failures.append(f"snapshot_not_before_game_start:{row_index}")
            continue

        canonical_row = row.to_dict()
        canonical_row["scheduled_game_id"] = (
            f"{official_game_id}_{canonical_row['team']}"
        )
        canonical_rows.append(canonical_row)

    if failures:
        raise PitchingSnapshotValidationError(failures)

    canonical = pd.DataFrame(canonical_rows, columns=SNAPSHOT_COLUMNS)
    canonical["_snapshot_time"] = pd.to_datetime(
        canonical["snapshot_time"],
        errors="raise",
    )
    canonical = (
        canonical.sort_values("_snapshot_time")
        .drop_duplicates(SNAPSHOT_KEY, keep="last")
        .sort_values(
            [
                "reference_date",
                "scheduled_game_id",
                "team",
                "snapshot_time",
            ]
        )
        .drop(columns="_snapshot_time")
        .reset_index(drop=True)
    )
    return canonical[SNAPSHOT_COLUMNS]


def validate_pitching_snapshot(
    frame: pd.DataFrame,
    *,
    as_of_date: date,
    existing_frame: pd.DataFrame | None = None,
) -> None:
    failures: list[str] = []
    missing_columns = [column for column in SNAPSHOT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise PitchingSnapshotValidationError(
            [f"missing_required_columns:{','.join(missing_columns)}"]
        )

    duplicate_count = int(frame.duplicated(SNAPSHOT_KEY).sum())
    if duplicate_count:
        failures.append(f"duplicate_key:{duplicate_count}")

    invalid_game_id_count = 0
    game_id_team_mismatch_count = 0
    for _, row in frame.iterrows():
        game_id, game_team = _game_id_parts(str(row["scheduled_game_id"]))
        if not OFFICIAL_GAME_ID.fullmatch(game_id):
            invalid_game_id_count += 1
        if game_team != str(row["team"]):
            game_id_team_mismatch_count += 1
    if invalid_game_id_count:
        failures.append(f"noncanonical_game_id:{invalid_game_id_count}")
    if game_id_team_mismatch_count:
        failures.append(
            f"game_id_team_mismatch:{game_id_team_mismatch_count}"
        )

    snapshot_dates = _date_values(frame, "snapshot_date")
    reference_dates = _date_values(frame, "reference_date")
    snapshot_times = pd.to_datetime(frame["snapshot_time"], errors="coerce")
    if snapshot_dates.isna().any():
        failures.append(f"invalid_snapshot_date:{int(snapshot_dates.isna().sum())}")
    if reference_dates.isna().any():
        failures.append(f"invalid_reference_date:{int(reference_dates.isna().sum())}")
    if snapshot_times.isna().any():
        failures.append(f"invalid_snapshot_time:{int(snapshot_times.isna().sum())}")

    future_snapshot_count = int(snapshot_dates.dropna().gt(as_of_date).sum())
    future_reference_count = int(reference_dates.dropna().gt(as_of_date).sum())
    if future_snapshot_count:
        failures.append(f"future_snapshot_date:{future_snapshot_count}")
    if future_reference_count:
        failures.append(f"future_reference_date:{future_reference_count}")

    for column in ("team", "scheduled_game_id", "opponent"):
        blank_count = int(
            frame[column].fillna("").astype(str).str.strip().eq("").sum()
        )
        if blank_count:
            failures.append(f"blank_{column}:{blank_count}")

    quality = pd.to_numeric(frame["starter_info_quality"], errors="coerce")
    invalid_quality_count = int((quality.isna() | ~quality.between(0, 1)).sum())
    if invalid_quality_count:
        failures.append(f"invalid_starter_info_quality:{invalid_quality_count}")

    if existing_frame is not None:
        existing_dates = set(_date_values(existing_frame, "snapshot_date").dropna())
        candidate_dates = set(snapshot_dates.dropna())
        missing_existing_dates = sorted(existing_dates - candidate_dates)
        if missing_existing_dates:
            failures.append(
                "existing_dates_missing:"
                + ",".join(value.isoformat() for value in missing_existing_dates)
            )
        if len(candidate_dates) < len(existing_dates):
            failures.append(
                f"snapshot_day_count_decrease:{len(existing_dates)}->{len(candidate_dates)}"
            )
        if len(frame) < len(existing_frame):
            failures.append(f"row_count_decrease:{len(existing_frame)}->{len(frame)}")
        existing_keys = set(
            map(tuple, existing_frame[SNAPSHOT_KEY].astype(str).values)
        )
        candidate_keys = set(map(tuple, frame[SNAPSHOT_KEY].astype(str).values))
        missing_existing_keys = existing_keys - candidate_keys
        if missing_existing_keys:
            failures.append(
                f"existing_canonical_keys_missing:{len(missing_existing_keys)}"
            )

    if failures:
        raise PitchingSnapshotValidationError(failures)


def merge_pitching_snapshots(
    existing_frame: pd.DataFrame,
    new_frame: pd.DataFrame,
    *,
    as_of_date: date,
) -> pd.DataFrame:
    validate_pitching_snapshot(existing_frame, as_of_date=as_of_date)
    validate_pitching_snapshot(new_frame, as_of_date=as_of_date)
    if existing_frame.empty:
        merged = new_frame.copy()
    elif new_frame.empty:
        merged = existing_frame.copy()
    else:
        merged = pd.concat([existing_frame, new_frame], ignore_index=True)
    merged["_snapshot_time"] = pd.to_datetime(
        merged["snapshot_time"], errors="raise"
    )
    merged = (
        merged.sort_values("_snapshot_time")
        .drop_duplicates(SNAPSHOT_KEY, keep="last")
        .sort_values(
            [
                "reference_date",
                "scheduled_game_id",
                "team",
                "snapshot_time",
            ]
        )
        .drop(columns="_snapshot_time")
        .reset_index(drop=True)
    )
    validate_pitching_snapshot(
        merged,
        as_of_date=as_of_date,
        existing_frame=existing_frame,
    )
    return merged[SNAPSHOT_COLUMNS]


def save_pitching_snapshot(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    as_of_date: date,
    backup_dir: Path | None = None,
) -> Path | None:
    existing_frame = read_pitching_snapshot(output_path) if output_path.exists() else None
    if existing_frame is not None:
        validate_pitching_snapshot(existing_frame, as_of_date=as_of_date)
    validate_pitching_snapshot(
        frame,
        as_of_date=as_of_date,
        existing_frame=existing_frame,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if output_path.exists():
        backup_root = backup_dir or (
            Path.home() / ".local" / "share" / "kbo_analytics" / "pitching_snapshot_backups"
        )
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()[:12]
        backup_path = backup_root / f"{output_path.stem}.{timestamp}.{digest}.csv"
        shutil.copy2(output_path, backup_path)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame[SNAPSHOT_COLUMNS].to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return backup_path
