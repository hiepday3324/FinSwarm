from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.config import load_training_config
from puppy.training.oof import generate_oof_predictions
from puppy.training.pipeline import effective_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expanding-window OOF base predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_training_config(args.config)
    result = generate_oof_predictions(config, effective_splits(config), seed=args.seed, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
