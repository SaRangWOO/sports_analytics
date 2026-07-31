from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Validator = Callable[[Path], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes, validator: Validator | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if validator:
            validator(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str, validator: Validator | None = None) -> None:
    atomic_write_bytes(path, content.encode("utf-8"), validator)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    def validate_json(candidate: Path) -> None:
        json.loads(candidate.read_text(encoding="utf-8"))

    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
        validate_json,
    )


def backup_file(path: Path, backup_root: Path) -> Path | None:
    if not path.exists():
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = backup_root / f"{path.name}.{stamp}.{sha256_file(path)[:12]}.bak"
    shutil.copy2(path, destination)
    return destination


def atomic_publish(
    source: Path,
    destination: Path,
    backup_root: Path,
    validator: Validator | None = None,
) -> dict[str, str | None]:
    if not source.is_file():
        raise FileNotFoundError(source)
    before_checksum = sha256_file(destination) if destination.exists() else None
    backup = backup_file(destination, backup_root)
    atomic_write_bytes(destination, source.read_bytes(), validator)
    return {
        "checksum_before": before_checksum,
        "checksum_after": sha256_file(destination),
        "backup": str(backup) if backup else None,
    }
