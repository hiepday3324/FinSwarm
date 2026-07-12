from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.config import load_training_config
from puppy.training.pipeline import STAGES, run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinSwarm Stage 6 offline training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-stage", choices=STAGES)
    parser.add_argument("--to-stage", choices=STAGES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run_training_pipeline(
        load_training_config(args.config),
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        resume=args.resume,
        force=args.force,
        seed=args.seed,
        device=args.device,
        local_files_only=True if args.local_files_only else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
