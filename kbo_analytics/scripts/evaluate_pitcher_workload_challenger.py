from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.model_training import write_pregame_matchup_reports
from modeling.pitcher_workload_candidate_validation import validate_pitcher_workload_candidate


def main() -> None:
    results = PROJECT_DIR / "modeling" / "results"
    bundle = write_pregame_matchup_reports(
        results,
        pd.read_csv(results / "game_level_features.csv"),
        pd.read_csv(results / "features.csv"),
        PROJECT_DIR / "data" / "official",
    )
    store = pd.read_csv(results / "pregame_matchup_feature_store.csv")
    gate = validate_pitcher_workload_candidate(store, results)
    print(json.dumps({"summary": bundle["summary"], "candidate_gate": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
