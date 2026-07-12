from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.data_engine.normalization import load_stage4_config, normalize_all
from puppy.data_engine.normalization.quality import build_quality_report, write_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize FinSwarm Stage 4 raw data.")
    parser.add_argument("--config", required=True, help="Path to experiment_v1.yaml")
    args = parser.parse_args()
    config = load_stage4_config(args.config)
    result = normalize_all(config)
    report = build_quality_report(config)
    write_quality_report(config, report)
    result["quality_status"] = report["status"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
