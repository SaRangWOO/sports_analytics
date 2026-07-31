from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .state import StateStore


def snapshot_health(snapshot_path: Path) -> dict[str, Any]:
    if not snapshot_path.exists():
        return {
            "snapshot_days": 0,
            "snapshot_rows": 0,
            "mapping_failures": None,
            "canonical_duplicates": None,
            "feature_coverage": 0.0,
            "post_start_rows": None,
            "post_start_rows_source": "unavailable",
            "status": "missing",
        }
    frame = pd.read_csv(snapshot_path, dtype={"scheduled_game_id": str})
    key = ["reference_date", "scheduled_game_id", "team"]
    ids = frame["scheduled_game_id"].fillna("").astype(str)
    feature_columns = [
        "starter_info_quality",
        "starter_era",
        "starter_whip",
        "bullpen_fatigue_label",
        "recent_3day_games",
    ]
    available = [column for column in feature_columns if column in frame.columns]
    feature_coverage = (
        float(frame[available].notna().mean().mean())
        if available
        else 0.0
    )
    return {
        "snapshot_days": int(frame["snapshot_date"].astype(str).nunique()),
        "snapshot_rows": int(len(frame)),
        "mapping_failures": int((~ids.str.match(r"^\d{8}[A-Z]{4}\d+_.+$")).sum()),
        "canonical_duplicates": int(frame.duplicated(key).sum()),
        "feature_coverage": round(feature_coverage, 4),
        "post_start_rows": 0,
        "post_start_rows_source": "canonical_storage_guard",
        "status": "pass",
    }


def artifact_ids(artifact_root: Path) -> dict[str, str | None]:
    def metadata_id(path: Path) -> str | None:
        metadata = path / "metadata.json"
        if not metadata.exists():
            return None
        return json.loads(metadata.read_text(encoding="utf-8"))["artifact_id"]

    previous = artifact_root / "previous"
    candidate = artifact_root / "candidate"
    previous_paths = [path for path in previous.iterdir() if path.is_dir()] if previous.exists() else []
    candidate_paths = [path for path in candidate.iterdir() if path.is_dir()] if candidate.exists() else []
    return {
        "current_production_artifact_id": metadata_id(artifact_root / "production" / "current"),
        "previous_artifact_id": metadata_id(max(previous_paths, key=lambda path: path.stat().st_mtime_ns)) if previous_paths else None,
        "latest_candidate_artifact_id": metadata_id(max(candidate_paths, key=lambda path: path.stat().st_mtime_ns)) if candidate_paths else None,
    }


def scheduler_conflicts(crontab_text: str, systemd_units: list[str]) -> dict[str, Any]:
    cron_tasks = {
        "morning": bool(re.search(r"daily_kbo_update|morning-update", crontab_text)),
        "pregame": bool(re.search(r"pregame_kbo_update|refresh_pregame_context|automation-dispatch", crontab_text)),
        "postgame": bool(re.search(r"postgame-update", crontab_text)),
        "challenger": bool(re.search(r"challenger-evaluate", crontab_text)),
    }
    unit_text = "\n".join(systemd_units)
    systemd_tasks = {
        "morning": "morning-update" in unit_text,
        "pregame": (
            "pregame-update" in unit_text
            or "automation-dispatch" in unit_text
        ),
        "postgame": "postgame-update" in unit_text,
        "challenger": "challenger-evaluate" in unit_text,
    }
    conflicts = [
        task for task in cron_tasks if cron_tasks[task] and systemd_tasks[task]
    ]
    return {
        "cron_tasks": cron_tasks,
        "systemd_tasks": systemd_tasks,
        "conflicts": conflicts,
        "status": "fail" if conflicts else "pass",
    }


def inspect_scheduler() -> dict[str, Any]:
    cron = subprocess.run(
        ["crontab", "-l"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    units = subprocess.run(
        ["systemctl", "list-unit-files", "--no-legend"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return scheduler_conflicts(cron, units)


def build_automation_status(config) -> dict[str, Any]:
    store = StateStore(config.state_root)
    status = store.read_status()
    status.update(snapshot_health(config.project_root / "data" / "official" / "pitching_daily_snapshot.csv"))
    status.update(artifact_ids(config.artifact_root))
    now = datetime.now(config.tz)
    minutes = config.dispatcher_interval_minutes
    next_dispatch = now.replace(second=0, microsecond=0) + timedelta(
        minutes=minutes - (now.minute % minutes)
    )
    scheduler = inspect_scheduler()
    status.update(
        {
            "auto_promote_enabled": config.auto_promote_enabled,
            "auto_rollback_enabled": config.auto_rollback_enabled,
            "scheduler_status": scheduler,
            "next_expected_run": next_dispatch.isoformat(),
            "snapshot_quality": status.get("status", "missing"),
            "overall_status": (
                "warning"
                if scheduler["conflicts"] or status.get("last_failure")
                else "healthy"
            ),
            "generated_at": now.isoformat(),
        }
    )
    return status
