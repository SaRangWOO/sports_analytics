from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from collectors.internal_pitcher_mapping import write_internal_pitcher_mapping_outputs


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze internal KBO pitcher data mapping candidates.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_internal_pitcher_mapping_outputs(REPO_DIR, args.output_dir, apply=args.apply)
    print(f"internal_pitcher_mapping_completed={summary['internal_pitcher_mapping_completed']}")
    print(f"internal_pitcher_candidate_files={summary['internal_pitcher_candidate_files']}")
    print(f"best_starter_source_file={summary['best_starter_source_file']}")
    print(f"best_pitcher_log_source_file={summary['best_pitcher_log_source_file']}")
    print(f"starter_conversion_possible={summary['starter_conversion_possible']}")
    print(f"pitcher_log_conversion_possible={summary['pitcher_log_conversion_possible']}")
    print(f"internal_pitcher_conversion_applied={summary['internal_pitcher_conversion_applied']}")
    print(f"pitcher_data_ready_to_train_after_mapping={summary['pitcher_data_ready_to_train_after_mapping']}")
    print(f"internal_pitcher_mapping_blocker={summary['internal_pitcher_mapping_blocker']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
