from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json


FINAL_STATUSES = {"succeeded", "failed", "skipped", "rolled_back"}


def identity_key(
    task: str,
    reference_date: str,
    game_id: str,
    update_stage: str,
    input_checksum: str,
) -> str:
    value = "|".join([task, reference_date, game_id, update_stage, input_checksum])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class StateStore:
    def __init__(self, root: Path):
        self.root = root
        self.runs = root / "runs"
        self.status_path = root / "automation_status.json"

    def run_path(self, key: str) -> Path:
        return self.runs / f"{key}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        path = self.run_path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def should_skip(self, key: str) -> bool:
        state = self.read(key)
        return bool(state and state.get("status") == "succeeded")

    def recover_stale(self, key: str, stale_after_seconds: int) -> bool:
        state = self.read(key)
        if not state or state.get("status") != "running":
            return False
        started = datetime.fromisoformat(state["started_at"])
        if datetime.now(timezone.utc) - started <= timedelta(seconds=stale_after_seconds):
            return False
        state.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": "StaleRun",
                "error_message": "running state exceeded stale threshold",
            }
        )
        atomic_write_json(self.run_path(key), state)
        return True

    def start(self, key: str, payload: dict[str, Any]) -> None:
        state = {
            **payload,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "retry_count": int(payload.get("retry_count", 0)),
        }
        atomic_write_json(self.run_path(key), state)

    def finish(self, key: str, status: str, **updates: Any) -> dict[str, Any]:
        if status not in FINAL_STATUSES:
            raise ValueError(f"invalid final status: {status}")
        state = self.read(key)
        if state is None:
            raise FileNotFoundError(self.run_path(key))
        state.update(
            {
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                **updates,
            }
        )
        atomic_write_json(self.run_path(key), state)
        return state

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def update_status(self, **updates: Any) -> dict[str, Any]:
        status = self.read_status()
        status.update(updates)
        status["generated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.status_path, status)
        return status
