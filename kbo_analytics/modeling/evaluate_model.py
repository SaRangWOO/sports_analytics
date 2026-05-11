from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print saved KBO win predictor metrics.")
    parser.add_argument("--model", default="win_predictor_model.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    print("Top coefficients:")
    for name, value in list(payload["coefficients"].items())[:10]:
        print(f"- {name}: {value}")


if __name__ == "__main__":
    main()
