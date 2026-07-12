from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.normalization import load_stage4_config
from puppy.data_engine.normalization.quality import validate_and_write
from puppy.data_engine.parquet_loader import load_market_parquets


def prepare_market_database(config_path: str, *, overwrite: bool = False) -> dict[str, object]:
    config = load_stage4_config(config_path)
    frames = load_market_parquets(str(config.canonical_prices.parent))
    counts = {name: len(frame) for name, frame in frames.items()}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Canonical parquet inputs must be non-empty: {counts}")
    target = config.duckdb_path
    if target.exists() and not overwrite:
        raise FileExistsError(f"DuckDB already exists; pass --overwrite to replace it: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    store = DuckDBMarketStore(
        str(temporary), market_timezone=config.timezone.key,
        decision_cutoff=config.decision_cutoff,
    )
    try:
        store.connect()
        store.ingest_prices(frames["prices"])
        store.ingest_news(frames["news"])
        store.ingest_filings(frames["filings"])
        actual = {
            table: store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("prices", "news", "filings")
        }
        if actual != counts:
            raise ValueError(f"Parquet-to-DuckDB row count mismatch: parquet={counts}, duckdb={actual}")
        dates = store.list_common_dates(config.symbols)
        if not dates:
            raise ValueError("No common trading dates found after DuckDB ingestion")
        sample_dates = dates[:3]
        smoke = [store.get_point_in_time_market_step(date, list(config.symbols), dates) for date in sample_dates]
        news_ids = [event.event_id for step in smoke for values in step.news.values() for event in values]
        filing_ids = [event.filing_id for step in smoke for values in step.filings.values() for event in values]
        repeated_news = len(news_ids) - len(set(news_ids))
        repeated_filings = len(filing_ids) - len(set(filing_ids))
        future = sum(
            event.available_at > step.as_of
            for step in smoke
            for collection in (step.news, step.filings)
            for values in collection.values()
            for event in values
        )
        stale_substitutions = sum(bar.date != step.date for step in smoke for bar in step.prices.values())
        routed_step = next(
            (
                store.get_point_in_time_market_step(dates[index], list(config.symbols), dates)
                for index in range(1, len(dates))
                if (dates[index] - dates[index - 1]).days > 1
            ),
            None,
        )
        if routed_step is None:
            raise ValueError("No weekend boundary found in the common trading calendar")
        previous_cutoff = dt.datetime.fromisoformat(routed_step.metadata["previous_cutoff"])
        current_cutoff = routed_step.as_of
        expected_news = {
            row[0] for row in store.conn.execute(
                "SELECT event_id FROM news WHERE available_at > ? AND available_at <= ?",
                [previous_cutoff, current_cutoff],
            ).fetchall()
        }
        expected_filings = {
            row[0] for row in store.conn.execute(
                "SELECT filing_id FROM filings WHERE available_at > ? AND available_at <= ?",
                [previous_cutoff, current_cutoff],
            ).fetchall()
        }
        routed_news = {event.event_id for values in routed_step.news.values() for event in values}
        routed_filings = {event.filing_id for values in routed_step.filings.values() for event in values}
        routing_mismatches = len(expected_news ^ routed_news) + len(expected_filings ^ routed_filings)
        routed_events = [
            event
            for collection in (routed_step.news, routed_step.filings)
            for values in collection.values()
            for event in values
        ]
        weekend_events = sum(event.available_at.astimezone(config.timezone).weekday() >= 5 for event in routed_events)
        after_hours_events = sum(
            event.available_at.astimezone(config.timezone).date() == dates[dates.index(routed_step.date) - 1]
            and event.available_at.astimezone(config.timezone).time().replace(tzinfo=None) > config.decision_cutoff
            for event in routed_events
        )
        if repeated_news or repeated_filings or future or stale_substitutions:
            raise ValueError(
                "Point-in-time smoke query failed: "
                f"repeated_news={repeated_news}, repeated_filings={repeated_filings}, "
                f"future={future}, stale_substitutions={stale_substitutions}"
            )
        if routing_mismatches:
            raise ValueError(f"Weekend/after-hours routing mismatch count: {routing_mismatches}")
    except Exception:
        store.close()
        if temporary.exists():
            temporary.unlink()
        raise
    store.close()
    if target.exists():
        target.unlink()
    temporary.replace(target)
    report = validate_and_write(config)
    return {
        "database_path": config.relative(target),
        "table_row_counts": counts,
        "sample_dates": [date.isoformat() for date in sample_dates],
        "point_in_time_smoke": {
            "repeated_news_ids": repeated_news,
            "repeated_filing_ids": repeated_filings,
            "future_events": future,
            "stale_price_substitutions": stale_substitutions,
            "routing_date": routed_step.date.isoformat(),
            "routed_news": len(routed_news),
            "routed_filings": len(routed_filings),
            "weekend_events": weekend_events,
            "after_hours_events": after_hours_events,
            "routing_mismatches": routing_mismatches,
        },
        "quality_status": report["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FinSwarm Stage 4 DuckDB database.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_market_database(args.config, overwrite=args.overwrite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
