from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.pitcher_workload_features import attach_pitcher_workload_features, build_pitcher_workload_features
from modeling.pitcher_workload_candidate_validation import validate_pitcher_workload_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated chronological pitcher workload experiment")
    parser.add_argument("--feature-store", type=Path, default=PROJECT_DIR / "modeling/results/pregame_matchup_feature_store.csv")
    parser.add_argument("--pitcher-logs", type=Path, default=PROJECT_DIR / "data/official/pitcher_game_logs.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve() == (PROJECT_DIR / "modeling/results").resolve():
        parser.error("Use an isolated directory, not operational modeling/results")
    store = pd.read_csv(args.feature_store)
    logs = pd.read_csv(args.pitcher_logs)
    store = attach_pitcher_workload_features(store, build_pitcher_workload_features(logs))
    gate = validate_pitcher_workload_candidate(store, args.output_dir)
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()
