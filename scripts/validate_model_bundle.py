from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.model_bundle import validate_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify hashes and predictions in a Stage 6 bundle.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = validate_model_bundle(args.bundle, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
