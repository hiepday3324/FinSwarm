from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_market_database import prepare_market_database


def build_duckdb_from_parquets(
    parquet_dir: str = "data/parquet",
    db_path: str = "data/duckdb/finswarm.db",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    warnings.warn(
        "build_duckdb_from_parquets is deprecated; use prepare_market_database.py",
        DeprecationWarning,
        stacklevel=2,
    )
    if parquet_dir != "data/parquet" or db_path != "data/duckdb/finswarm.db":
        raise ValueError("Legacy builder now supports only the Stage 4 configured paths")
    return prepare_market_database("config/swarm/experiment_v1.yaml", overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated wrapper for Stage 4 DB preparation.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_duckdb_from_parquets(overwrite=args.overwrite)
    print(result)


if __name__ == "__main__":
    main()
