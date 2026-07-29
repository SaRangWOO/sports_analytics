from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def add_common_reference_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--reference-date")


def build_parser():
    parser = argparse.ArgumentParser(description="Run existing KBO analytics tasks without changing their behavior.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    subparsers = parser.add_subparsers(dest="task", required=True)

    smoke = subparsers.add_parser("smoke", help="Run low-cost artifact and predict-only checks.")
    smoke.set_defaults(task_name="smoke")

    full = subparsers.add_parser("full", help="Run the expensive official collection, model evaluation, and dashboard build.")
    add_common_reference_arguments(full)
    full.add_argument("--reference-datetime")
    full.add_argument("--update-stage", choices=["morning", "pregame"], default="morning")
    full.add_argument("--training-start-year", type=int, default=2016)
    full.set_defaults(task_name="full")

    features = subparsers.add_parser("features", help="Regenerate the team-row feature CSV from an existing game CSV.")
    features.add_argument("--input", type=Path, default=PROJECT_DIR / "data" / "official" / "model_training_games.csv")
    features.add_argument("--output", type=Path, default=PROJECT_DIR / "modeling" / "results" / "features.csv")
    features.set_defaults(task_name="features")

    run_model = subparsers.add_parser("run-model", help="Run the independent expected-runs model.")
    add_common_reference_arguments(run_model)
    run_model.add_argument("--input", type=Path, default=PROJECT_DIR / "data" / "official" / "model_training_games.csv")
    run_model.add_argument("--schedule-input", type=Path, default=PROJECT_DIR / "data" / "official" / "prediction_games.csv")
    run_model.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "run_model" / "results")
    run_model.add_argument("--train-ratio", type=float, default=0.8)
    run_model.set_defaults(task_name="run-model")

    run_dashboard = subparsers.add_parser("run-dashboard", help="Render the independent run-model dashboard from existing results.")
    run_dashboard.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "run_model" / "results")
    run_dashboard.add_argument("--output", type=Path, default=PROJECT_DIR / "run_model" / "dashboard.html")
    run_dashboard.set_defaults(task_name="run-dashboard")

    artifact_build = subparsers.add_parser(
        "model-artifact-build",
        help="Run explicit model development and save the selected D-1 refit as a candidate artifact.",
    )
    add_common_reference_arguments(artifact_build)
    artifact_build.add_argument("--training-start-year", type=int, default=2016)
    artifact_build.set_defaults(task_name="model-artifact-build")

    artifact_validate = subparsers.add_parser(
        "model-artifact-validate",
        help="Validate a candidate manifest, checksums, schema, load, and smoke prediction.",
    )
    artifact_validate.add_argument("--artifact-id", required=True)
    artifact_validate.set_defaults(task_name="model-artifact-validate")

    artifact_promote = subparsers.add_parser(
        "model-artifact-promote",
        help="Explicitly promote a validated candidate and preserve the current production artifact.",
    )
    artifact_promote.add_argument("--artifact-id", required=True)
    artifact_promote.set_defaults(task_name="model-artifact-promote")

    artifact_rollback = subparsers.add_parser(
        "model-artifact-rollback",
        help="Restore the most recent valid previous production artifact.",
    )
    artifact_rollback.set_defaults(task_name="model-artifact-rollback")

    predict_only = subparsers.add_parser(
        "predict-only",
        help="Refresh current predictions and dashboard from the approved production artifact without model training.",
    )
    add_common_reference_arguments(predict_only)
    predict_only.add_argument("--reference-datetime")
    predict_only.add_argument("--update-stage", choices=["morning", "pregame"], default="morning")
    predict_only.set_defaults(task_name="predict-only")
    return parser


def commands_for(args):
    python = sys.executable
    if args.task_name == "smoke":
        files = [
            PROJECT_DIR / "modeling" / "model_artifacts.py",
            PROJECT_DIR / "modeling" / "model_training.py",
            PROJECT_DIR / "modeling" / "predict_only.py",
            PROJECT_DIR / "modeling" / "prediction_runtime.py",
            PROJECT_DIR / "modeling" / "test_model_artifacts.py",
            PROJECT_DIR / "modeling" / "test_predict_only.py",
            PROJECT_DIR / "scripts" / "kbo_tasks.py",
            PROJECT_DIR / "scripts" / "model_artifact_admin.py",
            PROJECT_DIR / "scripts" / "model_artifact_build.py",
            PROJECT_DIR / "scripts" / "predict_only_dashboard.py",
        ]
        return [
            ([python, "-m", "py_compile", *map(str, files)], PROJECT_DIR),
            (
                [
                    python,
                    "-c",
                    (
                        "from modeling import model_artifacts, model_training; "
                        "assert callable(model_artifacts.create_evaluation_candidate); "
                        "assert callable(model_training.evaluate_model)"
                    ),
                ],
                PROJECT_DIR,
            ),
            (
                [
                    python,
                    "-c",
                    (
                        "from modeling import predict_only, prediction_runtime; "
                        "assert callable(predict_only.run_predict_only); "
                        "assert callable(prediction_runtime.generate_today_predictions)"
                    ),
                ],
                PROJECT_DIR,
            ),
            (
                [
                    python,
                    "-m",
                    "unittest",
                    "modeling.test_model_artifacts",
                    "modeling.test_predict_only",
                    "-v",
                ],
                PROJECT_DIR,
            ),
            (
                [
                    python,
                    "-c",
                    "from scripts.kbo_tasks import validate_cli_contract; validate_cli_contract()",
                ],
                PROJECT_DIR,
            ),
        ]
    if args.task_name == "full":
        command = [
            python,
            str(PROJECT_DIR / "official_kbo_dashboard.py"),
            "--training-start-year",
            str(args.training_start_year),
            "--update-stage",
            args.update_stage,
        ]
        if args.reference_date:
            command.extend(["--reference-date", args.reference_date])
        if args.reference_datetime:
            command.extend(["--reference-datetime", args.reference_datetime])
        return [(command, PROJECT_DIR)]
    if args.task_name == "features":
        return [
            (
                [
                    python,
                    str(PROJECT_DIR / "modeling" / "feature_engineering.py"),
                    "--input",
                    str(args.input),
                    "--output",
                    str(args.output),
                ],
                PROJECT_DIR,
            )
        ]
    if args.task_name == "run-model":
        command = [
            python,
            str(PROJECT_DIR / "run_model" / "run_prediction_model.py"),
            "--input",
            str(args.input),
            "--schedule-input",
            str(args.schedule_input),
            "--output-dir",
            str(args.output_dir),
            "--train-ratio",
            str(args.train_ratio),
        ]
        if args.reference_date:
            command.extend(["--reference-date", args.reference_date])
        return [(command, PROJECT_DIR)]
    if args.task_name == "run-dashboard":
        return [
            (
                [
                    python,
                    str(PROJECT_DIR / "run_model" / "run_model_dashboard.py"),
                    "--results-dir",
                    str(args.results_dir),
                    "--output",
                    str(args.output),
                ],
                PROJECT_DIR,
            )
        ]
    if args.task_name == "model-artifact-build":
        command = [
            python,
            str(PROJECT_DIR / "scripts" / "model_artifact_build.py"),
            "--training-start-year",
            str(args.training_start_year),
        ]
        if args.reference_date:
            command.extend(["--reference-date", args.reference_date])
        return [(command, PROJECT_DIR)]
    if args.task_name == "model-artifact-validate":
        return [
            (
                [
                    python,
                    str(PROJECT_DIR / "scripts" / "model_artifact_admin.py"),
                    "validate",
                    "--artifact-id",
                    args.artifact_id,
                ],
                PROJECT_DIR,
            )
        ]
    if args.task_name == "model-artifact-promote":
        return [
            (
                [
                    python,
                    str(PROJECT_DIR / "scripts" / "model_artifact_admin.py"),
                    "promote",
                    "--artifact-id",
                    args.artifact_id,
                ],
                PROJECT_DIR,
            )
        ]
    if args.task_name == "model-artifact-rollback":
        return [
            (
                [python, str(PROJECT_DIR / "scripts" / "model_artifact_admin.py"), "rollback"],
                PROJECT_DIR,
            )
        ]
    command = [
        python,
        str(PROJECT_DIR / "scripts" / "predict_only_dashboard.py"),
        "--update-stage",
        args.update_stage,
    ]
    if args.reference_date:
        command.extend(["--reference-date", args.reference_date])
    if args.reference_datetime:
        command.extend(["--reference-datetime", args.reference_datetime])
    return [(command, PROJECT_DIR)]


def validate_cli_contract():
    cases = [
        ["smoke"],
        ["features"],
        ["run-model"],
        ["run-dashboard"],
        ["full"],
        ["model-artifact-build"],
        ["model-artifact-validate", "--artifact-id", "candidate-smoke"],
        ["model-artifact-promote", "--artifact-id", "candidate-smoke"],
        ["model-artifact-rollback"],
        ["predict-only"],
    ]
    parser = build_parser()
    for argv in cases:
        commands = commands_for(parser.parse_args(argv))
        if not commands or any(not command for command, _ in commands):
            raise RuntimeError(f"CLI command generation failed: {' '.join(argv)}")


def main():
    args = build_parser().parse_args()
    for command, cwd in commands_for(args):
        display = " ".join(shlex.quote(part) for part in command)
        print(f"[{cwd}] {display}")
        if not args.dry_run:
            subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
