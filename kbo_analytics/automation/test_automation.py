from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from automation.artifact_lifecycle import (
    load_production_or_rollback,
    promote_with_rollback,
    promotion_gate,
    shadow_validate,
)
from automation.atomic_io import atomic_publish, atomic_write_json, sha256_file
from automation.config import AutomationConfig, load_config
from automation.dashboard_publish import publish_dashboard
from automation.dashboard_status import render_automation_status
from automation.dispatcher import eligible_pregame_games
from automation.health import scheduler_conflicts
from automation.locking import FileLock, LockUnavailable
from automation.postgame import evaluate_postgame_rows, run_postgame
from automation.prediction import prepare_shadow
from automation.morning import run_morning
from automation.pregame import run_pregame
from automation.retention import cleanup_runtime
from automation.runner import run_managed_task
from automation.state import StateStore, identity_key
from scripts.kbo_automation import build_parser
from modeling.model_artifacts import ArtifactValidationError


PROJECT_DIR = Path(__file__).resolve().parents[1]


def config_for(root: Path) -> AutomationConfig:
    project = root / "project"
    (project / "data" / "official").mkdir(parents=True)
    return AutomationConfig(
        timezone="Asia/Seoul",
        dispatcher_interval_minutes=10,
        morning_window="08:00",
        pregame_window_start_minutes=120,
        pregame_window_end_minutes=15,
        postgame_delay_minutes=45,
        retry_count=1,
        retry_delay_seconds=0,
        lock_timeout_seconds=0,
        snapshot_quality_thresholds={
            "min_snapshot_days": 30,
            "max_canonical_duplicates": 0,
            "max_post_start_rows": 0,
            "max_mapping_failures": 0,
        },
        challenger_thresholds={
            "min_snapshot_days": 30,
            "min_completed_games": 150,
            "min_feature_coverage": 0.95,
        },
        promotion_thresholds={
            "min_shadow_passes": 3,
            "min_accuracy_delta": 0.005,
            "require_brier_improvement": True,
            "require_log_loss_improvement": True,
        },
        rollback_thresholds={"max_consecutive_failures": 2},
        project_root=project,
        artifact_root=root / "artifacts",
        backup_root=root / "runtime" / "backups",
        runtime_root=root / "runtime",
        log_root=root / "runtime" / "logs",
        shadow_root=root / "runtime" / "shadow",
        dashboard_publish_path=root / "published" / "latest.html",
        retention_policy={
            "normal_logs_days": 30,
            "failed_logs_days": 90,
            "csv_backups": 1,
            "candidate_artifacts": 1,
            "previous_artifacts": 1,
        },
        auto_promote_enabled=False,
        auto_rollback_enabled=True,
    )


class AutomationTest(unittest.TestCase):
    def test_default_config_loads_with_safe_promotion_default(self):
        config = load_config()
        self.assertFalse(config.auto_promote_enabled)
        self.assertTrue(config.auto_rollback_enabled)
        self.assertEqual(config.timezone, "Asia/Seoul")

    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            atomic_write_json(path, {"status": "ok"})
            self.assertEqual(json.loads(path.read_text())["status"], "ok")
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_publish_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.write_text("invalid", encoding="utf-8")
            destination.write_text("original", encoding="utf-8")
            before_hash = sha256_file(destination)
            before_mtime = destination.stat().st_mtime_ns
            with self.assertRaises(ValueError):
                atomic_publish(
                    source,
                    destination,
                    root / "backups",
                    lambda _: (_ for _ in ()).throw(ValueError("bad")),
                )
            self.assertEqual(sha256_file(destination), before_hash)
            self.assertEqual(destination.stat().st_mtime_ns, before_mtime)

    def test_state_idempotency_and_stale_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = StateStore(Path(temporary))
            key = identity_key("morning", "2026-07-31", "", "morning", "x")
            store.start(key, {"task": "morning"})
            state = store.read(key)
            state["started_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            atomic_write_json(store.run_path(key), state)
            self.assertTrue(store.recover_stale(key, 1))
            store.start(key, {"task": "morning"})
            store.finish(key, "succeeded")
            self.assertTrue(store.should_skip(key))

    def test_duplicate_lock_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            with FileLock(path):
                with self.assertRaises(LockUnavailable):
                    with FileLock(path):
                        pass

    def test_managed_task_retries_then_succeeds_and_skips_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            calls = {"count": 0}

            def action():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError("retry")
                return {"rows_after": 1}

            current = datetime(2026, 7, 31, 8, tzinfo=timezone.utc)
            result = run_managed_task(
                config,
                "morning",
                current.date(),
                current,
                action,
                input_checksum="same",
            )
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(calls["count"], 2)
            skipped = run_managed_task(
                config,
                "morning",
                current.date(),
                current,
                action,
                input_checksum="same",
            )
            self.assertEqual(skipped.status, "skipped")
            self.assertEqual(calls["count"], 2)

    def test_morning_and_pregame_use_managed_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            current = datetime(2026, 7, 31, 8, tzinfo=timezone.utc)
            result = {"artifact_id": "production-1", "output_checksum": "hash"}
            with patch("automation.morning.run_prediction_update", return_value=result):
                morning = run_morning(config, current.date(), current)
            self.assertEqual(morning.status, "succeeded")
            with patch("automation.pregame.run_prediction_update", return_value=result):
                first = run_pregame(
                    config,
                    current.date(),
                    current,
                    game_id="20260731KTLG0_LG",
                    source_checksum="first",
                )
                duplicate = run_pregame(
                    config,
                    current.date(),
                    current,
                    game_id="20260731KTLG0_LG",
                    source_checksum="first",
                )
                changed = run_pregame(
                    config,
                    current.date(),
                    current,
                    game_id="20260731KTLG0_LG",
                    source_checksum="changed",
                )
            self.assertEqual(first.status, "succeeded")
            self.assertEqual(duplicate.status, "skipped")
            self.assertEqual(changed.status, "succeeded")

    def test_managed_task_stops_after_configured_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            calls = {"count": 0}

            def fail():
                calls["count"] += 1
                raise RuntimeError("failed")

            current = datetime(2026, 7, 31, 8, tzinfo=timezone.utc)
            with self.assertRaises(RuntimeError):
                run_managed_task(
                    config,
                    "morning",
                    current.date(),
                    current,
                    fail,
                    input_checksum="failed",
                )
            self.assertEqual(calls["count"], config.retry_count + 1)

    def test_dispatcher_window_and_post_start_block(self):
        current = datetime(2026, 7, 31, 16, 0, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            [
                {
                    "official_game_id": "20260731KTLG0_LG",
                    "scheduled_start_datetime": current + timedelta(minutes=90),
                    "status": "Scheduled",
                },
                {
                    "official_game_id": "20260731OBWO0_WO",
                    "scheduled_start_datetime": current - timedelta(minutes=1),
                    "status": "Scheduled",
                },
            ]
        )
        rows = eligible_pregame_games(frame, current, 120, 15)
        self.assertEqual([row["official_game_id"] for row in rows], ["20260731KTLG0_LG"])

    def test_dispatcher_accepts_naive_schedule_with_aware_reference(self):
        current = datetime(2026, 7, 31, 16, 0, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            [{
                "official_game_id": "20260731KTLG0_LG",
                "scheduled_start_datetime": "2026-07-31 17:30:00",
                "status": "Scheduled",
            }]
        )
        self.assertEqual(len(eligible_pregame_games(frame, current, 120, 15)), 1)

    def test_postgame_metrics(self):
        frame = pd.DataFrame(
            {
                "actual_home_win": [1, 0, 1],
                "home_win_probability": [0.8, 0.4, 0.6],
            }
        )
        result = evaluate_postgame_rows(frame)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["completed_games"], 3)
        self.assertLess(result["brier_score"], 0.2)

    def test_postgame_writes_only_isolated_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            current = datetime(2026, 7, 31, 23, tzinfo=timezone.utc)
            frame = pd.DataFrame(
                {"actual_home_win": [1], "home_win_probability": [0.6]}
            )
            result = run_postgame(config, current.date(), current, frame)
            self.assertEqual(result.status, "succeeded")
            self.assertTrue(
                (config.report_root / "postgame-2026-07-31.json").exists()
            )

    def test_promotion_gate_requires_all_checks(self):
        thresholds = {
            "min_shadow_passes": 3,
            "min_accuracy_delta": 0.005,
            "require_brier_improvement": True,
            "require_log_loss_improvement": True,
            "require_calibration_not_worse": True,
            "require_recent_30_not_worse": True,
            "require_recent_60_not_worse": True,
            "require_high_confidence_not_worse": True,
            "max_team_accuracy_drop": 0.03,
        }
        gate = promotion_gate(
            {
                "accuracy": 0.54,
                "brier_score": 0.249,
                "log_loss": 0.69,
                "calibration_error": 0.03,
                "recent_30_accuracy": 0.53,
                "recent_60_accuracy": 0.54,
                "over_55_accuracy": 0.56,
            },
            {
                "accuracy": 0.55,
                "brier_score": 0.245,
                "log_loss": 0.68,
                "calibration_error": 0.025,
                "recent_30_accuracy": 0.54,
                "recent_60_accuracy": 0.55,
                "over_55_accuracy": 0.57,
                "worst_team_accuracy_delta": -0.01,
            },
            3,
            thresholds,
            True,
            True,
        )
        self.assertTrue(gate["passed"])
        gate["checks"]["shadow_passes"] = False
        self.assertFalse(all(gate["checks"].values()))

    def test_auto_promote_false_blocks_physical_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            with patch(
                "automation.artifact_lifecycle.validate_artifact",
                return_value={"metadata": {"artifact_id": "candidate-1"}},
            ), patch("automation.artifact_lifecycle.promote_candidate") as promote:
                result = promote_with_rollback(
                    config,
                    "candidate-1",
                    {"passed": True},
                    lambda: None,
                )
            self.assertEqual(result["status"], "eligible_not_promoted")
            promote.assert_not_called()

    def test_failed_gate_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                promote_with_rollback(
                    config_for(Path(temporary)),
                    "candidate-1",
                    {"passed": False},
                    lambda: None,
                )

    def test_shadow_validation_is_deterministic_and_fit_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_root = Path(temporary)
            artifact = {
                "schema": {"feature_order": ["a", "b"]},
                "bundle": {"unused": True},
            }
            with patch(
                "automation.artifact_lifecycle.validate_artifact",
                return_value=artifact,
            ), patch(
                "automation.artifact_lifecycle.predict_bundle_probabilities",
                side_effect=[
                    pd.Series([0.4, 0.6]).to_numpy(),
                    pd.Series([0.4, 0.6]).to_numpy(),
                ],
            ):
                result = shadow_validate(
                    report_root / "artifacts",
                    "candidate-1",
                    report_root,
                    "2026-07-31",
                )
            self.assertTrue(result["passed"])
            self.assertEqual(result["fit_calls"], 0)

    def test_invalid_production_artifact_rehearses_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            restored = {"metadata": {"artifact_id": "restored"}}
            with patch(
                "automation.artifact_lifecycle.load_production_artifact",
                side_effect=[ArtifactValidationError("invalid"), restored],
            ), patch(
                "automation.artifact_lifecycle.rollback_production",
                return_value=config.artifact_root / "production" / "current",
            ):
                artifact, report = load_production_or_rollback(config)
            self.assertEqual(artifact["metadata"]["artifact_id"], "restored")
            self.assertEqual(report["rollback_status"], "succeeded")
            self.assertTrue((config.report_root / "last_rollback.json").exists())

    def test_dashboard_publish_and_status_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "latest.html"
            destination = root / "public" / "latest.html"
            source.write_text("<html><body>ok</body></html>", encoding="utf-8")
            result = publish_dashboard(source, destination, root / "backups")
            self.assertEqual(result["checksum_after"], sha256_file(destination))
            status = root / "status.json"
            status.write_text(json.dumps({"overall_status": "정상"}), encoding="utf-8")
            self.assertIn("정상", render_automation_status(status))

    def test_shadow_prepares_independent_model_and_run_model_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = config_for(Path(temporary))
            (config.project_root / "modeling" / "results").mkdir(parents=True)
            (config.project_root / "run_model" / "results").mkdir(parents=True)
            paths = prepare_shadow(config, "run-1")
            self.assertTrue(paths["results"].is_relative_to(config.shadow_root))
            self.assertTrue(paths["run_results"].is_relative_to(config.shadow_root))
            self.assertNotEqual(
                paths["run_results"],
                config.project_root / "run_model" / "results",
            )

    def test_cleanup_dry_run_preserves_production_and_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_for(root)
            production = config.artifact_root / "production" / "current"
            quarantine = config.quarantine_root / "held"
            production.mkdir(parents=True)
            quarantine.mkdir(parents=True)
            result = cleanup_runtime(config, dry_run=True)
            self.assertNotIn(str(production), result["paths"])
            self.assertNotIn(str(quarantine), result["paths"])

    def test_scheduler_conflicts_are_reported(self):
        result = scheduler_conflicts(
            "0 8 * * * daily_kbo_update.sh",
            ["kbo-morning-update.timer enabled"],
        )
        self.assertEqual(result["conflicts"], ["morning"])
        self.assertEqual(result["status"], "fail")

    def test_cli_exposes_all_required_commands(self):
        choices = build_parser()._subparsers._group_actions[0].choices
        required = {
            "automation-status",
            "automation-dispatch",
            "morning-update",
            "pregame-update",
            "postgame-update",
            "snapshot-quality",
            "predict-only",
            "challenger-evaluate",
            "artifact-build",
            "artifact-validate",
            "artifact-shadow",
            "artifact-promote",
            "artifact-rollback",
            "publish-dashboard",
            "cleanup-runtime",
            "automation-smoke",
        }
        self.assertEqual(set(choices), required)

    def test_systemd_templates_do_not_enable_services(self):
        deploy = PROJECT_DIR.parent / "deploy" / "systemd"
        files = list(deploy.glob("kbo-*.service")) + list(deploy.glob("kbo-*.timer"))
        self.assertEqual(len(files), 10)
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertNotIn("systemctl enable", combined)
        self.assertNotIn("systemctl start", combined)
        self.assertIn("User=wsr", combined)


if __name__ == "__main__":
    unittest.main()
