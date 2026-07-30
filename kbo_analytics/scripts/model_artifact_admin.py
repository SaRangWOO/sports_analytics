from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.model_artifacts import (
    candidate_path,
    promote_candidate,
    rollback_production,
    validate_artifact,
)


ARTIFACT_ROOT = PROJECT_DIR / "modeling" / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, promote, or roll back KBO model artifacts.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact-id", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--artifact-id", required=True)
    subparsers.add_parser("rollback")
    args = parser.parse_args()

    if args.action == "validate":
        result = validate_artifact(
            ARTIFACT_ROOT,
            candidate_path(ARTIFACT_ROOT, args.artifact_id),
            expected_approval="candidate",
        )
        print(f"[Success] candidate artifact valid: {result['metadata']['artifact_id']}")
    elif args.action == "promote":
        path = promote_candidate(ARTIFACT_ROOT, args.artifact_id)
        print(f"[Success] production artifact promoted: {path}")
    else:
        path = rollback_production(ARTIFACT_ROOT)
        print(f"[Success] production artifact rolled back: {path}")


if __name__ == "__main__":
    main()
