from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json


def _older_than(path: Path, days: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return modified < cutoff


def _keep_latest(paths: list[Path], keep: int) -> set[Path]:
    return set(sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True)[:keep])


def _has_failure(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and json.loads(line).get("status") == "failed":
            return True
    return False


def cleanup_runtime(config, dry_run: bool = True) -> dict[str, Any]:
    candidates: list[Path] = []
    if config.log_root.exists():
        for path in config.log_root.glob("*.jsonl"):
            retention_days = (
                config.retention_policy["failed_logs_days"]
                if _has_failure(path)
                else config.retention_policy["normal_logs_days"]
            )
            if _older_than(path, int(retention_days)):
                candidates.append(path)
    if config.backup_root.exists():
        backups = [path for path in config.backup_root.iterdir() if path.is_file()]
        candidates.extend(set(backups) - _keep_latest(backups, int(config.retention_policy["csv_backups"])))
    artifact_candidates = config.artifact_root / "candidate"
    if artifact_candidates.exists():
        paths = [path for path in artifact_candidates.iterdir() if path.is_dir()]
        candidates.extend(
            set(paths) - _keep_latest(paths, int(config.retention_policy["candidate_artifacts"]))
        )
    previous = config.artifact_root / "previous"
    if previous.exists():
        paths = [path for path in previous.iterdir() if path.is_dir()]
        candidates.extend(
            set(paths) - _keep_latest(paths, int(config.retention_policy["previous_artifacts"]))
        )
    protected = {
        config.artifact_root / "production",
        config.artifact_root / "production" / "current",
        config.quarantine_root,
    }
    candidates = [
        path
        for path in candidates
        if not any(path == root or root in path.parents for root in protected)
    ]
    if not dry_run:
        for path in candidates:
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            else:
                path.unlink()
    return {
        "dry_run": dry_run,
        "delete_count": len(candidates),
        "paths": [str(path) for path in candidates],
    }
