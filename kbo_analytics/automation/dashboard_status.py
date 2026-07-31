from __future__ import annotations

import html
import json
import os
from pathlib import Path


DEFAULT_STATUS_PATH = Path(
    "/home/wsr/1.project/runtime/kbo_automation/state/automation_status.json"
)


def render_automation_status(path: str | Path | None = None) -> str:
    status_path = Path(
        path or os.getenv("KBO_AUTOMATION_STATUS_PATH", DEFAULT_STATUS_PATH)
    )
    if not status_path.exists():
        return ""
    status = json.loads(status_path.read_text(encoding="utf-8"))
    fields = [
        ("자동화 상태", status.get("overall_status", "확인 필요")),
        ("마지막 성공 작업", status.get("last_successful_task", "-")),
        ("다음 예상 실행", status.get("next_expected_run", "-")),
        ("Production artifact", status.get("current_production_artifact_id", "-")),
        ("Candidate artifact", status.get("latest_candidate_artifact_id", "-")),
        ("Snapshot 품질", status.get("snapshot_quality", status.get("status", "-"))),
        ("최근 실패", status.get("last_failure", "-")),
        ("Rollback", status.get("last_rollback", "-")),
        (
            "Auto promote",
            "활성" if status.get("auto_promote_enabled") else "비활성",
        ),
    ]
    cards = "".join(
        f'<div class="automation-metric"><span>{html.escape(str(label))}</span>'
        f"<strong>{html.escape(str(value))}</strong></div>"
        for label, value in fields
    )
    return (
        '<section class="automation-status-panel">'
        '<div class="eyebrow">OPS · 자동화 운영 상태</div>'
        "<h2>KBO 파이프라인 상태</h2>"
        f'<div class="automation-status-grid">{cards}</div>'
        "</section>"
    )
