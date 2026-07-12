from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.data_engine.normalization import load_stage4_config
from puppy.data_engine.normalization.quality import validate_and_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FinSwarm Stage 4 canonical data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = validate_and_write(load_stage4_config(args.config))
    print(json.dumps({"status": report["status"], "quality_gate": report["quality_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
