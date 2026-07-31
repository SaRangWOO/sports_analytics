from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .locking import FileLock
from .logging import JsonLogger
from .state import StateStore, identity_key


@dataclass
class TaskResult:
    run_id: str
    task: str
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    reference_date: str
    reference_datetime: str
    artifact_id: str | None = None
    rows_before: int | None = None
    rows_after: int | None = None
    input_checksum: str | None = None
    output_checksum: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_managed_task(
    config,
    task: str,
    reference_date: date,
    reference_datetime: datetime,
    action: Callable[[], dict[str, Any]],
    *,
    run_id: str | None = None,
    game_id: str = "",
    update_stage: str = "",
    input_checksum: str = "",
    force: bool = False,
) -> TaskResult:
    run_id = run_id or f"{task}-{reference_datetime.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    checksum = input_checksum or stable_checksum(
        {
            "task": task,
            "reference_date": reference_date.isoformat(),
            "game_id": game_id,
            "update_stage": update_stage,
        }
    )
    key = identity_key(
        task,
        reference_date.isoformat(),
        game_id,
        update_stage,
        checksum,
    )
    store = StateStore(config.state_root)
    logger = JsonLogger(config.log_root, run_id)
    start = datetime.now(timezone.utc)
    if not force and store.should_skip(key):
        completed = datetime.now(timezone.utc)
        return TaskResult(
            run_id=run_id,
            task=task,
            status="skipped",
            started_at=start.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=0.0,
            reference_date=reference_date.isoformat(),
            reference_datetime=reference_datetime.isoformat(),
            input_checksum=checksum,
            details={"reason": "identical successful execution exists"},
        )
    store.recover_stale(key, max(config.lock_timeout_seconds * 12, 60))
    lock_name = f"{task}-{game_id or 'global'}.lock"
    with FileLock(config.lock_root / lock_name, config.lock_timeout_seconds):
        store.start(
            key,
            {
                "run_id": run_id,
                "task": task,
                "reference_date": reference_date.isoformat(),
                "reference_datetime": reference_datetime.isoformat(),
                "game_id": game_id,
                "update_stage": update_stage,
                "input_checksum": checksum,
            },
        )
        logger.write(
            run_id=run_id,
            task=task,
            stage="start",
            status="running",
            reference_date=reference_date.isoformat(),
            game_id=game_id,
        )
        try:
            attempts = 0
            while True:
                try:
                    details = action()
                    break
                except Exception:
                    if attempts >= config.retry_count:
                        raise
                    attempts += 1
                    logger.write(
                        run_id=run_id,
                        task=task,
                        stage="retry",
                        status="running",
                        attempt=attempts,
                        reference_date=reference_date.isoformat(),
                    )
                    time.sleep(config.retry_delay_seconds)
            completed = datetime.now(timezone.utc)
            result = TaskResult(
                run_id=run_id,
                task=task,
                status="succeeded",
                started_at=start.isoformat(),
                completed_at=completed.isoformat(),
                duration_seconds=round((completed - start).total_seconds(), 3),
                reference_date=reference_date.isoformat(),
                reference_datetime=reference_datetime.isoformat(),
                artifact_id=details.get("artifact_id"),
                rows_before=details.get("rows_before"),
                rows_after=details.get("rows_after"),
                input_checksum=checksum,
                output_checksum=details.get("output_checksum"),
                details={**details, "attempts": attempts + 1},
            )
            store.finish(key, "succeeded", result=result.to_dict())
            status_updates = {
                "last_successful_task": task,
                f"last_{task.replace('-', '_')}_success": completed.isoformat(),
                "last_failure": None,
            }
            if task in {"morning", "pregame"}:
                status_updates["last_predict_only_success"] = completed.isoformat()
            if task == "challenger":
                status_updates["last_challenger_result"] = details
            store.update_status(
                **status_updates,
            )
            logger.write(
                run_id=run_id,
                task=task,
                stage="complete",
                status="succeeded",
                duration_seconds=result.duration_seconds,
                reference_date=reference_date.isoformat(),
                artifact_id=result.artifact_id,
                rows_before=result.rows_before,
                rows_after=result.rows_after,
                checksum_before=None,
                checksum_after=result.output_checksum,
            )
            return result
        except Exception as exc:
            completed = datetime.now(timezone.utc)
            result = TaskResult(
                run_id=run_id,
                task=task,
                status="failed",
                started_at=start.isoformat(),
                completed_at=completed.isoformat(),
                duration_seconds=round((completed - start).total_seconds(), 3),
                reference_date=reference_date.isoformat(),
                reference_datetime=reference_datetime.isoformat(),
                input_checksum=checksum,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            store.finish(
                key,
                "failed",
                error_type=result.error_type,
                error_message=result.error_message,
                result=result.to_dict(),
            )
            store.update_status(
                last_failure={
                    "task": task,
                    "stage": "execution",
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                    "at": completed.isoformat(),
                }
            )
            logger.write(
                run_id=run_id,
                task=task,
                stage="execution",
                status="failed",
                duration_seconds=result.duration_seconds,
                reference_date=reference_date.isoformat(),
                error_type=result.error_type,
                error_message=result.error_message,
            )
            raise
