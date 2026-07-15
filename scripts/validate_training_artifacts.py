from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.artifact_validation import validate_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate causal FinSwarm training artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allow-non-final",
        action="store_true",
        help="Validate temporal integration for a short/mock diagnostic rollout",
    )
    args = parser.parse_args()
    report = validate_training_artifacts(
        args.config,
        args.run_id,
        require_final=not args.allow_non_final,
    )
    print(json.dumps({"status": report["status"], "critical_metrics": report["critical_metrics"], "validated_artifact_counts": report["validated_artifact_counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
