from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLogger:
    def __init__(self, log_root: Path, run_id: str):
        self.path = log_root / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
