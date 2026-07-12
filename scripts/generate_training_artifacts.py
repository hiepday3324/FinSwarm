from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.artifact_generator import generate_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal FinSwarm training artifacts without training models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-llm", action="store_true", help="Reserved for injected non-bootstrap adapters")
    args = parser.parse_args()
    result = generate_training_artifacts(
        args.config,
        start=args.start,
        end=args.end,
        bootstrap=args.bootstrap,
        resume=args.resume,
        overwrite=args.overwrite,
        force_llm=args.force_llm,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
