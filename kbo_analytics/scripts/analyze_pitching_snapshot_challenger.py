from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.pitching_snapshot_challenger import (
    write_pitching_snapshot_challenger_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate leakage-safe pitching snapshots as an offline KBO challenger."
    )
    parser.add_argument("--pregame-store", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--production-history", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_pitching_snapshot_challenger_reports(
        args.pregame_store.resolve(),
        args.snapshot.resolve(),
        args.schedule.resolve(),
        args.output_dir.resolve(),
        args.production_history.resolve() if args.production_history else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
