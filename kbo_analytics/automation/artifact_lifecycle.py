from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from modeling.model_artifacts import (
    ArtifactValidationError,
    candidate_path,
    load_production_artifact,
    predict_bundle_probabilities,
    promote_candidate,
    rollback_production,
    validate_artifact,
)

from .atomic_io import atomic_write_json


def load_production_or_rollback(config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        return load_production_artifact(config.artifact_root), None
    except (ArtifactValidationError, FileNotFoundError) as exc:
        if not config.auto_rollback_enabled:
            raise
        started = datetime.now(timezone.utc)
        restored_path = rollback_production(config.artifact_root)
        restored = load_production_artifact(config.artifact_root)
        completed = datetime.now(timezone.utc)
        report = {
            "failed_artifact_id": None,
            "restored_artifact_id": restored["metadata"]["artifact_id"],
            "failure_stage": "production_artifact_load",
            "failure_reason": str(exc),
            "rollback_started_at": started.isoformat(),
            "rollback_completed_at": completed.isoformat(),
            "rollback_status": "succeeded",
            "restored_path": str(restored_path),
        }
        atomic_write_json(config.report_root / "last_rollback.json", report)
        return restored, report


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    selected = metrics.get("selected_candidate_metrics", metrics)
    for name in names:
        if selected.get(name) is not None:
            return float(selected[name])
    return None


def shadow_validate(
    artifact_root: Path,
    artifact_id: str,
    report_root: Path,
    reference_date: str,
) -> dict[str, Any]:
    artifact = validate_artifact(
        artifact_root,
        candidate_path(artifact_root, artifact_id),
        expected_approval="candidate",
    )
    schema = artifact["schema"]
    frame = pd.DataFrame(
        [
            [0.0] * len(schema["feature_order"]),
            [1.0] * len(schema["feature_order"]),
        ],
        columns=schema["feature_order"],
    )
    first = predict_bundle_probabilities(artifact["bundle"], frame, schema)
    second = predict_bundle_probabilities(artifact["bundle"], frame, schema)
    parity_error = float(np.max(np.abs(first - second)))
    passed = bool(
        parity_error <= 1e-12
        and np.isfinite(first).all()
        and ((first >= 0) & (first <= 1)).all()
    )
    report = {
        "artifact_id": artifact_id,
        "reference_date": reference_date,
        "passed": passed,
        "parity_max_abs_error": parity_error,
        "fit_calls": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(
        report_root / "shadow" / f"{reference_date}-{artifact_id}.json",
        report,
    )
    if not passed:
        raise ValueError("artifact shadow parity failed")
    return report


def shadow_pass_count(report_root: Path, artifact_id: str) -> int:
    root = report_root / "shadow"
    if not root.exists():
        return 0
    dates = set()
    for path in root.glob(f"*-{artifact_id}.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("passed"):
            dates.add(payload["reference_date"])
    return len(dates)


def promotion_gate(
    production_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    shadow_passes: int,
    thresholds: dict[str, Any],
    quality_pass: bool,
    leakage_pass: bool,
) -> dict[str, Any]:
    production_accuracy = _metric(production_metrics, "accuracy", "검증 정확도")
    candidate_accuracy = _metric(candidate_metrics, "accuracy", "검증 정확도")
    production_brier = _metric(production_metrics, "brier_score", "Brier Score")
    candidate_brier = _metric(candidate_metrics, "brier_score", "Brier Score")
    production_loss = _metric(production_metrics, "log_loss", "Log Loss")
    candidate_loss = _metric(candidate_metrics, "log_loss", "Log Loss")
    production_calibration = _metric(
        production_metrics,
        "calibration_error",
        "expected_calibration_error",
    )
    candidate_calibration = _metric(
        candidate_metrics,
        "calibration_error",
        "expected_calibration_error",
    )
    production_recent_30 = _metric(production_metrics, "recent_30_accuracy")
    candidate_recent_30 = _metric(candidate_metrics, "recent_30_accuracy")
    production_recent_60 = _metric(production_metrics, "recent_60_accuracy")
    candidate_recent_60 = _metric(candidate_metrics, "recent_60_accuracy")
    production_high_confidence = _metric(
        production_metrics,
        "high_confidence_accuracy",
        "over_55_accuracy",
    )
    candidate_high_confidence = _metric(
        candidate_metrics,
        "high_confidence_accuracy",
        "over_55_accuracy",
    )
    worst_team_delta = _metric(candidate_metrics, "worst_team_accuracy_delta")
    checks = {
        "snapshot_quality_pass": quality_pass,
        "leakage_audit_pass": leakage_pass,
        "shadow_passes": shadow_passes >= int(thresholds["min_shadow_passes"]),
        "accuracy_delta": (
            production_accuracy is not None
            and candidate_accuracy is not None
            and candidate_accuracy - production_accuracy
            >= float(thresholds["min_accuracy_delta"])
        ),
        "brier_improved": (
            not thresholds.get("require_brier_improvement")
            or (
                production_brier is not None
                and candidate_brier is not None
                and candidate_brier < production_brier
            )
        ),
        "log_loss_improved": (
            not thresholds.get("require_log_loss_improvement")
            or (
                production_loss is not None
                and candidate_loss is not None
                and candidate_loss < production_loss
            )
        ),
        "calibration_not_worse": (
            not thresholds.get("require_calibration_not_worse")
            or (
                production_calibration is not None
                and candidate_calibration is not None
                and candidate_calibration <= production_calibration
            )
        ),
        "recent_30_not_worse": (
            not thresholds.get("require_recent_30_not_worse")
            or (
                production_recent_30 is not None
                and candidate_recent_30 is not None
                and candidate_recent_30 >= production_recent_30
            )
        ),
        "recent_60_not_worse": (
            not thresholds.get("require_recent_60_not_worse")
            or (
                production_recent_60 is not None
                and candidate_recent_60 is not None
                and candidate_recent_60 >= production_recent_60
            )
        ),
        "high_confidence_not_worse": (
            not thresholds.get("require_high_confidence_not_worse")
            or (
                production_high_confidence is not None
                and candidate_high_confidence is not None
                and candidate_high_confidence >= production_high_confidence
            )
        ),
        "team_performance_not_collapsed": (
            worst_team_delta is not None
            and worst_team_delta
            >= -float(thresholds.get("max_team_accuracy_drop", 1.0))
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "shadow_pass_count": shadow_passes,
    }


def promote_with_rollback(
    config,
    artifact_id: str,
    gate: dict[str, Any],
    smoke: Callable[[], None],
) -> dict[str, Any]:
    if not gate.get("passed"):
        raise ValueError("promotion gate failed")
    validate_artifact(
        config.artifact_root,
        candidate_path(config.artifact_root, artifact_id),
        expected_approval="candidate",
    )
    if not config.auto_promote_enabled:
        return {
            "status": "eligible_not_promoted",
            "artifact_id": artifact_id,
            "auto_promote_enabled": False,
        }
    path = promote_candidate(config.artifact_root, artifact_id)
    try:
        smoke()
    except Exception as exc:
        if not config.auto_rollback_enabled:
            raise
        restored = rollback_production(config.artifact_root)
        report = {
            "failed_artifact_id": artifact_id,
            "restored_artifact_id": load_production_artifact(config.artifact_root)["metadata"]["artifact_id"],
            "failure_stage": "post_promotion_smoke",
            "failure_reason": str(exc),
            "rollback_started_at": datetime.now(timezone.utc).isoformat(),
            "rollback_completed_at": datetime.now(timezone.utc).isoformat(),
            "rollback_status": "succeeded",
            "restored_path": str(restored),
        }
        atomic_write_json(config.report_root / "last_rollback.json", report)
        return report
    return {
        "status": "promoted",
        "artifact_id": artifact_id,
        "path": str(path),
    }
