from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modeling.model_training import evaluate_model
from official_kbo_dashboard import DATA_DIR, RESULTS_DIR, fetch_schedule, fetch_training_schedule, previous_sunday


ARTIFACT_ROOT = PROJECT_DIR / "modeling" / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the explicit model-development path and save the selected D-1 refit as a candidate artifact."
    )
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--training-start-year", type=int, default=2016)
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT))
    args = parser.parse_args()
    reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    games = fetch_schedule(reference_date.year, reference_date.month)
    training_games = fetch_training_schedule(args.training_start_year, reference_date)
    payload = evaluate_model(
        training_games,
        games,
        previous_sunday(reference_date),
        reference_date,
        DATA_DIR,
        RESULTS_DIR,
        artifact_root=Path(args.artifact_root).resolve(),
    )
    print(f"[Success] candidate artifact generated: {payload['candidate_artifact_id']}")


if __name__ == "__main__":
    main()
