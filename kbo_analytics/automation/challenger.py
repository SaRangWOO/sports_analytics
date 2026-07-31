from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.model_artifacts import candidate_path, validate_artifact

from .artifact_lifecycle import promotion_gate, shadow_pass_count, shadow_validate
from .health import snapshot_health
from .runner import run_managed_task, stable_checksum


def challenger_readiness(config) -> dict[str, Any]:
    snapshot = snapshot_health(
        config.project_root / "data" / "official" / "pitching_daily_snapshot.csv"
    )
    games_path = config.project_root / "data" / "official" / "model_training_games.csv"
    games = pd.read_csv(games_path) if games_path.exists() else pd.DataFrame()
    completed = (
        int(games["status"].eq("Final").sum())
        if "status" in games.columns
        else 0
    )
    quality = config.snapshot_quality_thresholds
    threshold = config.challenger_thresholds
    checks = {
        "snapshot_days": snapshot["snapshot_days"] >= int(threshold["min_snapshot_days"]),
        "completed_games": completed >= int(threshold["min_completed_games"]),
        "feature_coverage": snapshot["feature_coverage"] >= float(threshold["min_feature_coverage"]),
        "canonical_duplicates": (
            snapshot["canonical_duplicates"] is not None
            and snapshot["canonical_duplicates"] <= int(quality["max_canonical_duplicates"])
        ),
        "post_start_rows": (
            snapshot["post_start_rows"] is not None
            and snapshot["post_start_rows"] <= int(quality["max_post_start_rows"])
        ),
        "mapping_failures": (
            snapshot["mapping_failures"] is not None
            and snapshot["mapping_failures"] <= int(quality["max_mapping_failures"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "snapshot": snapshot,
        "completed_games": completed,
    }


def run_challenger(
    config,
    reference_date: date,
    reference_datetime: datetime,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    readiness = challenger_readiness(config)
    command = [
        sys.executable,
        str(config.project_root / "scripts" / "model_artifact_build.py"),
        "--reference-date",
        reference_date.isoformat(),
        "--artifact-root",
        str(config.artifact_root),
    ]
    if dry_run:
        return {
            "dry_run": True,
            "readiness": readiness,
            "command": command,
        }
    if not readiness["passed"]:
        raise ValueError(f"challenger readiness gate failed: {readiness['checks']}")

    def action():
        before = {
            path.name
            for path in (config.artifact_root / "candidate").iterdir()
            if path.is_dir()
        } if (config.artifact_root / "candidate").exists() else set()
        subprocess.run(command, cwd=config.project_root, check=True)
        after = {
            path.name
            for path in (config.artifact_root / "candidate").iterdir()
            if path.is_dir()
        }
        created = sorted(after - before)
        if len(created) != 1:
            raise ValueError(f"expected one candidate artifact, found {created}")
        artifact_id = created[0]
        artifact = validate_artifact(
            config.artifact_root,
            candidate_path(config.artifact_root, artifact_id),
            expected_approval="candidate",
        )
        shadow = shadow_validate(
            config.artifact_root,
            artifact_id,
            config.report_root,
            reference_date.isoformat(),
        )
        return {
            "artifact_id": artifact_id,
            "candidate_metrics": artifact["metrics"],
            "shadow": shadow,
            "shadow_pass_count": shadow_pass_count(config.report_root, artifact_id),
        }

    return run_managed_task(
        config,
        "challenger",
        reference_date,
        reference_datetime,
        action,
        run_id=run_id,
        update_stage="challenger",
        input_checksum=stable_checksum(readiness),
        force=force,
    )
