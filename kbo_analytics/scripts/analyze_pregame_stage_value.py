from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.pregame_stage_validation import write_stage_validation_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether KBO pregame refreshes improve completed-game predictions."
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_stage_validation_reports(
        args.history.resolve(), args.games.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
