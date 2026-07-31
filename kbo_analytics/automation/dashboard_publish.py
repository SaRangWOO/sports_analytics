from __future__ import annotations

from pathlib import Path

from .atomic_io import atomic_publish


def validate_html(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    if "<html" not in content.lower() or "</html>" not in content.lower():
        raise ValueError("dashboard output is not complete HTML")


def publish_dashboard(source: Path, destination: Path, backup_root: Path) -> dict:
    return atomic_publish(source, destination, backup_root, validate_html)
