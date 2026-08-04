from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from automation.artifact_lifecycle import (
    promotion_gate,
    promote_with_rollback,
    shadow_validate,
)
from automation.challenger import run_challenger
from automation.config import DEFAULT_CONFIG_PATH, load_config
from automation.dashboard_publish import publish_dashboard
from automation.dispatcher import dispatcher_decision
from automation.health import build_automation_status
from automation.morning import run_morning
from automation.postgame import run_postgame
from automation.pregame import run_pregame
from automation.player_challenger import (
    build_player_contribution_report,
    build_player_feature_outputs,
    evaluate_player_challenger,
    player_feature_quality,
)
from automation.prediction import run_prediction_update
from automation.retention import cleanup_runtime
from automation.runner import stable_checksum
from automation.state import StateStore
from modeling.model_artifacts import (
    candidate_path,
    rollback_production,
    validate_artifact,
)
from modeling.player_feature_pipeline import DEFAULT_CONFIG as DEFAULT_PLAYER_CONFIG


def _add_common(parser: argparse.ArgumentParser, *, force: bool = False) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--reference-date", default="")
    parser.add_argument("--reference-datetime", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", dest="json_output", action="store_true")
    if force:
        parser.add_argument("--force", action="store_true")


def _add_player_options(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--player-config", default=str(DEFAULT_PLAYER_CONFIG))
    parser.add_argument("--output-root", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate the KBO automation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_common(subparsers.add_parser("automation-status"))
    dispatch = subparsers.add_parser("automation-dispatch")
    _add_common(dispatch, force=True)
    dispatch.add_argument("--schedule-input", default="")
    _add_common(subparsers.add_parser("morning-update"), force=True)
    pregame = subparsers.add_parser("pregame-update")
    _add_common(pregame, force=True)
    pregame.add_argument("--game-id", required=True)
    pregame.add_argument("--source-checksum", default="")
    postgame = subparsers.add_parser("postgame-update")
    _add_common(postgame, force=True)
    postgame.add_argument("--input", default="")
    _add_common(subparsers.add_parser("snapshot-quality"))
    predict = subparsers.add_parser("predict-only")
    _add_common(predict)
    predict.add_argument("--update-stage", choices=["morning", "pregame"], default="morning")
    _add_common(subparsers.add_parser("challenger-evaluate"), force=True)
    build = subparsers.add_parser("artifact-build")
    _add_common(build)
    validate = subparsers.add_parser("artifact-validate")
    _add_common(validate)
    validate.add_argument("--artifact-id", required=True)
    shadow = subparsers.add_parser("artifact-shadow")
    _add_common(shadow)
    shadow.add_argument("--artifact-id", required=True)
    promote = subparsers.add_parser("artifact-promote")
    _add_common(promote, force=True)
    promote.add_argument("--artifact-id", required=True)
    promote.add_argument("--gate-report", required=True)
    rollback = subparsers.add_parser("artifact-rollback")
    _add_common(rollback, force=True)
    publish = subparsers.add_parser("publish-dashboard")
    _add_common(publish)
    publish.add_argument("--source", required=True)
    _add_common(subparsers.add_parser("cleanup-runtime"))
    _add_common(subparsers.add_parser("automation-smoke"))
    _add_player_options(subparsers.add_parser("player-feature-build"))
    _add_player_options(subparsers.add_parser("player-feature-quality"))
    _add_player_options(subparsers.add_parser("player-challenger-evaluate"))
    _add_player_options(subparsers.add_parser("player-contribution-report"))
    return parser


def _context(args: argparse.Namespace, config) -> tuple[date, datetime, str]:
    reference_datetime = (
        datetime.fromisoformat(args.reference_datetime)
        if args.reference_datetime
        else datetime.now(config.tz)
    )
    if reference_datetime.tzinfo is None:
        reference_datetime = reference_datetime.replace(tzinfo=config.tz)
    reference_date = (
        date.fromisoformat(args.reference_date)
        if args.reference_date
        else reference_datetime.date()
    )
    run_id = args.run_id or f"{args.command}-{reference_datetime:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    return reference_date, reference_datetime, run_id


def _schedule(reference_date: date) -> pd.DataFrame:
    import official_kbo_dashboard as dashboard

    frame = dashboard.pitching_snapshot_schedule_frame(reference_date).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["official_game_id", "scheduled_start_datetime", "status"]
        )
    return frame


def _postgame_input(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["actual_home_win", "home_win_probability"])
    frame = pd.read_csv(path)
    return frame


def _artifact_build_command(config, reference_date: date) -> list[str]:
    return [
        sys.executable,
        str(config.project_root / "scripts" / "model_artifact_build.py"),
        "--reference-date",
        reference_date.isoformat(),
        "--artifact-root",
        str(config.artifact_root),
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    reference_date, reference_datetime, run_id = _context(args, config)
    command = args.command
    if command in {
        "player-feature-build",
        "player-feature-quality",
        "player-challenger-evaluate",
        "player-contribution-report",
    }:
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else config.report_root / "player_challenger"
        )
        functions = {
            "player-feature-build": build_player_feature_outputs,
            "player-feature-quality": player_feature_quality,
            "player-challenger-evaluate": evaluate_player_challenger,
            "player-contribution-report": build_player_contribution_report,
        }
        result = functions[command](
            config.project_root,
            output_root,
            reference_datetime,
            Path(args.player_config),
            dry_run=args.dry_run,
        )
        return {key: value for key, value in result.items() if key not in {"features", "config"}}
    if command == "automation-status":
        status = build_automation_status(config)
        if not args.dry_run:
            StateStore(config.state_root).update_status(**status)
        return status
    if command == "automation-dispatch":
        schedule = (
            pd.read_csv(args.schedule_input)
            if args.schedule_input
            else _schedule(reference_date)
        )
        decision = dispatcher_decision(
            schedule,
            reference_datetime,
            config.pregame_window_start_minutes,
            config.pregame_window_end_minutes,
        )
        if args.dry_run:
            return {"run_id": run_id, "task": command, "dry_run": True, **decision}
        results = []
        for game in decision["eligible_games"]:
            checksum = stable_checksum(game)
            results.append(
                run_pregame(
                    config,
                    reference_date,
                    reference_datetime,
                    game_id=str(game["official_game_id"]),
                    source_checksum=checksum,
                    run_id=f"{run_id}-{game['official_game_id']}",
                    force=args.force,
                ).to_dict()
            )
        StateStore(config.state_root).update_status(**build_automation_status(config))
        return {"run_id": run_id, "task": command, **decision, "results": results}
    if command == "morning-update":
        result = run_morning(
            config,
            reference_date,
            reference_datetime,
            run_id=run_id,
            dry_run=args.dry_run,
            force=args.force,
        )
        return result if isinstance(result, dict) else result.to_dict()
    if command == "pregame-update":
        result = run_pregame(
            config,
            reference_date,
            reference_datetime,
            game_id=args.game_id,
            source_checksum=args.source_checksum,
            run_id=run_id,
            dry_run=args.dry_run,
            force=args.force,
        )
        return result if isinstance(result, dict) else result.to_dict()
    if command == "postgame-update":
        result = run_postgame(
            config,
            reference_date,
            reference_datetime,
            _postgame_input(args.input),
            run_id=run_id,
            dry_run=args.dry_run,
            force=args.force,
        )
        return result if isinstance(result, dict) else result.to_dict()
    if command == "snapshot-quality":
        status = build_automation_status(config)
        if not args.dry_run:
            StateStore(config.state_root).update_status(**status)
        return status
    if command == "predict-only":
        return run_prediction_update(
            config,
            run_id,
            reference_date,
            reference_datetime,
            args.update_stage,
            dry_run=args.dry_run,
        )
    if command == "challenger-evaluate":
        result = run_challenger(
            config,
            reference_date,
            reference_datetime,
            run_id=run_id,
            dry_run=args.dry_run,
            force=args.force,
        )
        return result if isinstance(result, dict) else result.to_dict()
    if command == "artifact-build":
        cmd = _artifact_build_command(config, reference_date)
        if args.dry_run:
            return {"run_id": run_id, "task": command, "dry_run": True, "command": cmd}
        import subprocess

        subprocess.run(cmd, cwd=config.project_root, check=True)
        return {"run_id": run_id, "task": command, "status": "succeeded"}
    if command == "artifact-validate":
        if args.dry_run:
            return {"run_id": run_id, "task": command, "dry_run": True, "artifact_id": args.artifact_id}
        artifact = validate_artifact(
            config.artifact_root,
            candidate_path(config.artifact_root, args.artifact_id),
            expected_approval="candidate",
        )
        return {"run_id": run_id, "task": command, "status": "succeeded", "artifact_id": artifact["metadata"]["artifact_id"]}
    if command == "artifact-shadow":
        if args.dry_run:
            return {"run_id": run_id, "task": command, "dry_run": True, "artifact_id": args.artifact_id}
        return shadow_validate(config.artifact_root, args.artifact_id, config.report_root, reference_date.isoformat())
    if command == "artifact-promote":
        gate = json.loads(Path(args.gate_report).read_text(encoding="utf-8"))
        if args.dry_run:
            return {
                "run_id": run_id,
                "task": command,
                "dry_run": True,
                "gate_passed": bool(gate.get("passed")),
                "auto_promote_enabled": config.auto_promote_enabled,
            }
        return promote_with_rollback(config, args.artifact_id, gate, lambda: None)
    if command == "artifact-rollback":
        if args.dry_run:
            return {"run_id": run_id, "task": command, "dry_run": True}
        path = rollback_production(config.artifact_root)
        return {"run_id": run_id, "task": command, "status": "rolled_back", "path": str(path)}
    if command == "publish-dashboard":
        if args.dry_run:
            return {"run_id": run_id, "task": command, "dry_run": True, "source": args.source}
        return publish_dashboard(Path(args.source), config.dashboard_publish_path, config.backup_root)
    if command == "cleanup-runtime":
        return cleanup_runtime(config, dry_run=args.dry_run)
    if command == "automation-smoke":
        status = build_automation_status(config)
        return {
            "run_id": run_id,
            "task": command,
            "status": "succeeded",
            "dry_run": args.dry_run,
            "config_loaded": True,
            "state_readable": isinstance(StateStore(config.state_root).read_status(), dict),
            "scheduler_status": status["scheduler_status"],
        }
    raise ValueError(f"unsupported command: {command}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = execute(args)
    except Exception as exc:
        payload = {
            "run_id": getattr(args, "run_id", ""),
            "task": getattr(args, "command", ""),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
