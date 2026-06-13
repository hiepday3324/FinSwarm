from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.parquet_loader import load_market_parquets


def build_duckdb_from_parquets(
    parquet_dir: str = "data/parquet",
    db_path: str = "data/duckdb/finswarm.db",
) -> dict[str, Any]:
    frames = load_market_parquets(parquet_dir)
    store = DuckDBMarketStore(db_path=db_path)
    store.connect()
    store.ingest_prices(frames["prices"])
    store.ingest_news(frames["news"])
    store.ingest_filings(frames["filings"])
    dates = store.list_dates({str(row["symbol"]) for row in _iter_records(frames["prices"])})
    store.close()
    return {
        "db_path": db_path,
        "parquet_dir": parquet_dir,
        "dates": [date.isoformat() for date in dates],
    }


def _iter_records(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        return [dict(item) for item in rows.to_dict(orient="records")]
    return [dict(item) for item in rows]


def main() -> None:
    parquet_dir = sys.argv[1] if len(sys.argv) > 1 else "data/parquet"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "data/duckdb/finswarm.db"
    result = build_duckdb_from_parquets(parquet_dir=parquet_dir, db_path=db_path)
    print(result)


if __name__ == "__main__":
    main()
