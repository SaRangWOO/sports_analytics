from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "automation.yaml"


@dataclass(frozen=True)
class AutomationConfig:
    timezone: str
    dispatcher_interval_minutes: int
    morning_window: str
    pregame_window_start_minutes: int
    pregame_window_end_minutes: int
    postgame_delay_minutes: int
    retry_count: int
    retry_delay_seconds: int
    lock_timeout_seconds: int
    snapshot_quality_thresholds: dict[str, Any]
    challenger_thresholds: dict[str, Any]
    promotion_thresholds: dict[str, Any]
    rollback_thresholds: dict[str, Any]
    project_root: Path
    artifact_root: Path
    backup_root: Path
    runtime_root: Path
    log_root: Path
    shadow_root: Path
    dashboard_publish_path: Path
    retention_policy: dict[str, Any]
    auto_promote_enabled: bool
    auto_rollback_enabled: bool

    @property
    def state_root(self) -> Path:
        return self.runtime_root / "state"

    @property
    def lock_root(self) -> Path:
        return self.runtime_root / "locks"

    @property
    def quarantine_root(self) -> Path:
        return self.runtime_root / "quarantine"

    @property
    def report_root(self) -> Path:
        return self.runtime_root / "reports"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return normalized == "true"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AutomationConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = AutomationConfig(
        timezone=str(data["timezone"]),
        dispatcher_interval_minutes=int(data["dispatcher_interval_minutes"]),
        morning_window=str(data["morning_window"]),
        pregame_window_start_minutes=int(data["pregame_window_start_minutes"]),
        pregame_window_end_minutes=int(data["pregame_window_end_minutes"]),
        postgame_delay_minutes=int(data["postgame_delay_minutes"]),
        retry_count=int(data["retry_count"]),
        retry_delay_seconds=int(data["retry_delay_seconds"]),
        lock_timeout_seconds=int(data["lock_timeout_seconds"]),
        snapshot_quality_thresholds=dict(data["snapshot_quality_thresholds"]),
        challenger_thresholds=dict(data["challenger_thresholds"]),
        promotion_thresholds=dict(data["promotion_thresholds"]),
        rollback_thresholds=dict(data["rollback_thresholds"]),
        project_root=_path(data["project_root"]),
        artifact_root=_path(os.getenv("KBO_ARTIFACT_ROOT", data["artifact_root"])),
        backup_root=_path(data["backup_root"]),
        runtime_root=_path(data["runtime_root"]),
        log_root=_path(data["log_root"]),
        shadow_root=_path(data["shadow_root"]),
        dashboard_publish_path=_path(data["dashboard_publish_path"]),
        retention_policy=dict(data["retention_policy"]),
        auto_promote_enabled=_bool_env(
            "KBO_AUTO_PROMOTE_ENABLED",
            bool(data["auto_promote_enabled"]),
        ),
        auto_rollback_enabled=_bool_env(
            "KBO_AUTO_ROLLBACK_ENABLED",
            bool(data["auto_rollback_enabled"]),
        ),
    )
    if config.pregame_window_start_minutes <= config.pregame_window_end_minutes:
        raise ValueError("pregame window start must be greater than end")
    if config.retry_count < 0 or config.lock_timeout_seconds < 0:
        raise ValueError("retry and lock settings must be non-negative")
    if config.auto_promote_enabled and os.getenv("KBO_AUTO_PROMOTE_ENABLED") != "true":
        raise ValueError("auto promotion requires explicit environment opt-in")
    return config
