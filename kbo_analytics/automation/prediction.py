from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from .artifact_lifecycle import load_production_or_rollback
from .atomic_io import atomic_publish, sha256_file
from .dashboard_publish import publish_dashboard, validate_html


CommandRunner = Callable[..., subprocess.CompletedProcess]


def _copy_existing(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)


def prepare_shadow(config, run_id: str) -> dict[str, Path]:
    root = config.shadow_root / run_id
    if root.exists():
        raise FileExistsError(root)
    paths = {
        "root": root,
        "data": root / "data" / "official",
        "results": root / "modeling" / "results",
        "dashboard": root / "dashboard",
        "public": root / "docs",
        "run_results": root / "run_model" / "results",
    }
    _copy_existing(config.project_root / "data" / "official", paths["data"])
    _copy_existing(config.project_root / "modeling" / "results", paths["results"])
    _copy_existing(config.project_root / "run_model" / "results", paths["run_results"])
    paths["dashboard"].mkdir(parents=True, exist_ok=True)
    paths["public"].mkdir(parents=True, exist_ok=True)
    return paths


def validate_shadow(paths: dict[str, Path], reference_date: date) -> dict[str, Any]:
    dashboard = paths["dashboard"] / "latest.html"
    validate_html(dashboard)
    payload_path = paths["results"] / "win_predictor_model.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("prediction_mode") != "predict_only":
        raise ValueError("shadow prediction did not use predict-only mode")
    probabilities = [
        float(row["예측승률"])
        for row in payload.get("today_predictions", [])
    ]
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("prediction probability is outside [0, 1]")
    snapshot_quality = json.loads(
        (paths["results"] / "pitching_snapshot_quality_status.json").read_text(
            encoding="utf-8"
        )
    )
    if snapshot_quality.get("quality_status") != "pass":
        raise ValueError("snapshot quality gate failed")
    if snapshot_quality.get("blocking_issues"):
        raise ValueError("snapshot quality has blocking issues")
    return {
        "artifact_id": payload["production_artifact_id"],
        "prediction_rows": len(payload.get("today_predictions", [])),
        "dashboard_checksum": sha256_file(dashboard),
        "snapshot_quality": snapshot_quality["quality_status"],
        "reference_date": reference_date.isoformat(),
    }


def publish_shadow(config, paths: dict[str, Path]) -> dict[str, Any]:
    published: list[dict[str, Any]] = []
    groups = [
        (paths["data"], config.project_root / "data" / "official"),
        (paths["results"], config.project_root / "modeling" / "results"),
        (paths["dashboard"], config.project_root / "dashboard"),
        (paths["public"], config.project_root.parent / "docs"),
        (paths["run_results"], config.project_root / "run_model" / "results"),
    ]
    for source_root, destination_root in groups:
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            validator = validate_html if source.suffix == ".html" else None
            result = atomic_publish(
                source,
                destination,
                config.backup_root / relative.parent,
                validator,
            )
            published.append(
                {
                    "path": str(destination),
                    "checksum": result["checksum_after"],
                }
            )
    return {
        "published_count": len(published),
        "published": published,
    }


def run_prediction_update(
    config,
    run_id: str,
    reference_date: date,
    reference_datetime: datetime,
    update_stage: str,
    *,
    dry_run: bool = False,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(config.project_root / "scripts" / "predict_only_dashboard.py"),
        "--reference-date",
        reference_date.isoformat(),
        "--reference-datetime",
        reference_datetime.strftime("%Y-%m-%d %H:%M"),
        "--update-stage",
        update_stage,
        "--artifact-root",
        str(config.artifact_root),
    ]
    if dry_run:
        return {
            "dry_run": True,
            "command": command,
            "artifact_root": str(config.artifact_root),
            "shadow_root": str(config.shadow_root / run_id),
        }
    artifact, rollback = load_production_or_rollback(config)
    paths = prepare_shadow(config, run_id)
    command.extend(
        [
            "--data-dir",
            str(paths["data"]),
            "--results-dir",
            str(paths["results"]),
            "--dashboard-dir",
            str(paths["dashboard"]),
            "--public-dir",
            str(paths["public"]),
            "--run-model-input",
            str(paths["data"] / "model_training_games.csv"),
            "--run-model-schedule-input",
            str(paths["data"] / "prediction_games.csv"),
            "--run-model-results-dir",
            str(paths["run_results"]),
            "--skip-db",
        ]
    )
    command_runner(command, cwd=config.project_root, check=True)
    shadow = validate_shadow(paths, reference_date)
    publish = publish_shadow(config, paths)
    return {
        **shadow,
        **publish,
        "output_checksum": shadow["dashboard_checksum"],
        "artifact_id": artifact["metadata"]["artifact_id"],
        "rollback": rollback,
    }
