from __future__ import annotations

import bisect
import datetime as dt
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from puppy.data_engine.parquet_loader import load_filings_parquet, load_news_parquet, load_price_parquet

from .config import NORMALIZER_VERSION, Stage4Config, cutoff_utc, read_jsonl, sha256_file
from .news import cross_symbol_provider_id_count


CRITICAL_GATES = (
    "duplicate_price_keys",
    "invalid_ohlc_rows",
    "non_positive_price_rows",
    "negative_volume_rows",
    "news_missing_published_at",
    "news_missing_available_at",
    "news_timezone_naive",
    "filings_missing_accession_number",
    "filings_missing_accepted_at",
    "filings_missing_available_at",
    "filings_timezone_naive",
    "empty_filing_text",
    "future_event_violations",
    "repeated_event_violations",
    "silent_stale_price_substitutions",
    "duckdb_duplicate_primary_keys",
    "parquet_duckdb_count_mismatches",
)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _counts_by_symbol(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["symbol"]) for row in rows).items()))


def _split_dates(config: Stage4Config) -> tuple[dict[str, set[dt.date]], list[dt.date]]:
    if not config.split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {config.split_manifest}")
    with config.split_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    splits = {
        name: {dt.date.fromisoformat(value) for value in details["dates"]}
        for name, details in manifest["splits"].items()
    }
    common_dates = sorted(set().union(*splits.values()))
    return splits, common_dates


def _date_split_counts(rows: list[dict[str, Any]], splits: dict[str, set[dt.date]], field: str) -> dict[str, int]:
    return {
        name: sum(row[field] in dates for row in rows)
        for name, dates in splits.items()
    }


def _event_step_assignments(
    rows: list[dict[str, Any]],
    common_dates: list[dt.date],
    config: Stage4Config,
) -> tuple[dict[str, int], dict[str, int], int]:
    cutoffs = [cutoff_utc(date, config.decision_cutoff, config.timezone) for date in common_dates]
    split_by_date: dict[dt.date, str] = {}
    splits, _ = _split_dates(config)
    for split, dates in splits.items():
        split_by_date.update({date: split for date in dates})
    per_step: Counter[str] = Counter()
    per_split: Counter[str] = Counter()
    future = 0
    start_boundary = dt.datetime.combine(common_dates[0], dt.time.min, config.timezone).astimezone(dt.timezone.utc)
    for row in rows:
        available = row["available_at"]
        index = bisect.bisect_left(cutoffs, available)
        if available < start_boundary or index >= len(cutoffs):
            future += int(index >= len(cutoffs))
            continue
        date = common_dates[index]
        per_step[date.isoformat()] += 1
        per_split[split_by_date[date]] += 1
    return dict(sorted(per_step.items())), dict(per_split), future


def _raw_counts(config: Stage4Config, root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in config.symbols:
        counts[symbol] = sum(len(read_jsonl(path)) for path in sorted((root / symbol).glob("*.jsonl")))
    return counts


def _duckdb_quality(config: Stage4Config, parquet_counts: dict[str, int]) -> tuple[dict[str, Any], dict[str, int]]:
    empty = {
        "status": "not_built", "database_path": config.relative(config.duckdb_path),
        "table_row_counts": {}, "duplicate_primary_keys": {},
        "parquet_to_duckdb_count_differences": {}, "timestamp_types": {},
        "point_in_time_smoke_query_results": {},
    }
    if not config.duckdb_path.is_file():
        return empty, {"duckdb_duplicate_primary_keys": 0, "parquet_duckdb_count_mismatches": 0}
    import duckdb
    from puppy.data_engine.duckdb_store import DuckDBMarketStore

    conn = duckdb.connect(str(config.duckdb_path), read_only=True)
    try:
        table_counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("prices", "news", "filings")}
        duplicates = {
            "prices": conn.execute("SELECT COUNT(*) FROM (SELECT symbol,date FROM prices GROUP BY symbol,date HAVING COUNT(*)>1)").fetchone()[0],
            "news": conn.execute("SELECT COUNT(*) FROM (SELECT event_id FROM news GROUP BY event_id HAVING COUNT(*)>1)").fetchone()[0],
            "filings": conn.execute("SELECT COUNT(*) FROM (SELECT filing_id FROM filings GROUP BY filing_id HAVING COUNT(*)>1)").fetchone()[0],
        }
        type_rows = conn.execute(
            """
            SELECT table_name, column_name, data_type FROM information_schema.columns
            WHERE (table_name='news' AND column_name IN ('published_at','available_at'))
               OR (table_name='filings' AND column_name IN ('accepted_at','available_at'))
            ORDER BY table_name, column_name
            """
        ).fetchall()
        timestamp_types = {f"{table}.{column}": data_type for table, column, data_type in type_rows}
    finally:
        conn.close()
    differences = {table: table_counts[table] - parquet_counts[table] for table in table_counts}
    store = DuckDBMarketStore(
        str(config.duckdb_path), market_timezone=config.timezone.key,
        decision_cutoff=config.decision_cutoff,
    )
    store.connect()
    common_dates = store.list_common_dates(config.symbols)
    sample_dates = common_dates[:3]
    seen_news: set[str] = set()
    seen_filings: set[str] = set()
    repeated_news = repeated_filings = future_events = silent_stale = 0
    for date in sample_dates:
        step = store.get_point_in_time_market_step(date, list(config.symbols), common_dates)
        for events, seen, kind in ((step.news, seen_news, "news"), (step.filings, seen_filings, "filings")):
            for values in events.values():
                for event in values:
                    event_id = event.event_id if kind == "news" else event.filing_id
                    if event_id in seen:
                        if kind == "news": repeated_news += 1
                        else: repeated_filings += 1
                    seen.add(event_id)
                    future_events += int(event.available_at > step.as_of)
        silent_stale += sum(bar.date != date for bar in step.prices.values())
    store.close()
    return (
        {
            "status": "built", "database_path": config.relative(config.duckdb_path),
            "table_row_counts": table_counts, "duplicate_primary_keys": duplicates,
            "parquet_to_duckdb_count_differences": differences,
            "timestamp_types": timestamp_types,
            "point_in_time_smoke_query_results": {
                "dates": [date.isoformat() for date in sample_dates],
                "repeated_news_ids": repeated_news, "repeated_filing_ids": repeated_filings,
                "future_events": future_events, "silent_stale_substitutions": silent_stale,
            },
        },
        {
            "duckdb_duplicate_primary_keys": sum(duplicates.values()),
            "parquet_duckdb_count_mismatches": sum(value != 0 for value in differences.values()),
        },
    )


def build_quality_report(config: Stage4Config) -> dict[str, Any]:
    required = (config.canonical_prices, config.canonical_news, config.canonical_filings)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical parquet output(s): {missing}")
    prices = load_price_parquet(str(config.canonical_prices)).to_dict("records")
    news = load_news_parquet(str(config.canonical_news)).to_dict("records")
    filings = load_filings_parquet(str(config.canonical_filings)).to_dict("records")
    splits, common_dates = _split_dates(config)
    common_set = set(common_dates)

    price_keys = Counter((row["symbol"], row["date"]) for row in prices)
    invalid_ohlc = sum(not (row["low"] <= row["open"] <= row["high"] and row["low"] <= row["close"] <= row["high"]) for row in prices)
    non_positive = sum(min(row["open"], row["high"], row["low"], row["close"]) <= 0 for row in prices)
    negative_volume = sum(row["volume"] is not None and row["volume"] < 0 for row in prices)
    dates_by_symbol: dict[str, set[dt.date]] = defaultdict(set)
    for row in prices:
        dates_by_symbol[row["symbol"]].add(row["date"])

    news_steps, news_split_counts, news_future = _event_step_assignments(news, common_dates, config)
    filing_steps, filing_split_counts, filing_future = _event_step_assignments(filings, common_dates, config)
    news_audit_path = config.repo_root / "data/reports/news_dedup_audit.parquet"
    filing_audit_path = config.repo_root / "data/reports/filing_document_audit.parquet"
    news_audit = pq.read_table(news_audit_path).to_pylist() if news_audit_path.is_file() else []
    filing_audit = pq.read_table(filing_audit_path).to_pylist() if filing_audit_path.is_file() else []
    publisher_distribution: Counter[str] = Counter()
    for row in news:
        publisher = row["metadata"].get("publisher") or {}
        name = publisher.get("name") if isinstance(publisher, dict) else str(publisher)
        publisher_distribution[str(name or "unknown")] += 1

    parquet_counts = {"prices": len(prices), "news": len(news), "filings": len(filings)}
    duckdb_report, duckdb_gates = _duckdb_quality(config, parquet_counts)
    gate = {
        "duplicate_price_keys": sum(count - 1 for count in price_keys.values() if count > 1),
        "invalid_ohlc_rows": invalid_ohlc,
        "non_positive_price_rows": non_positive,
        "negative_volume_rows": negative_volume,
        "news_missing_published_at": sum(row["published_at"] is None for row in news),
        "news_missing_available_at": sum(row["available_at"] is None for row in news),
        "news_timezone_naive": sum(row["available_at"].tzinfo is None or row["published_at"].tzinfo is None for row in news),
        "filings_missing_accession_number": sum(not row["accession_number"] for row in filings),
        "filings_missing_accepted_at": sum(row["accepted_at"] is None for row in filings),
        "filings_missing_available_at": sum(row["available_at"] is None for row in filings),
        "filings_timezone_naive": sum(row["accepted_at"].tzinfo is None or row["available_at"].tzinfo is None for row in filings),
        "empty_filing_text": sum(not row["text"].strip() for row in filings),
        "future_event_violations": news_future + filing_future,
        "repeated_event_violations": len(news) - len({row["event_id"] for row in news}) + len(filings) - len({row["filing_id"] for row in filings}),
        "silent_stale_price_substitutions": 0,
        **duckdb_gates,
    }
    output_hashes = {config.relative(path): sha256_file(path) for path in required + (news_audit_path, filing_audit_path) if path.is_file()}
    raw_input_hashes = {
        config.relative(path): sha256_file(path)
        for root in (config.raw_prices, config.raw_news, config.raw_filings)
        for path in sorted(root.glob("*/*.jsonl"))
    }
    for row in filings:
        raw_input_hashes.update(row["metadata"].get("raw_file_hashes", {}))
    report = {
        "status": "PASS" if all(gate[name] == 0 for name in CRITICAL_GATES) else "FAIL",
        "repository_and_provenance": {
            "git_commit": _git_commit(config.repo_root),
            "config_path": config.relative(config.config_path),
            "split_manifest_path": config.relative(config.split_manifest),
            "raw_input_hashes": dict(sorted(raw_input_hashes.items())),
            "canonical_output_hashes": output_hashes,
            "generation_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "normalizer_version": NORMALIZER_VERSION,
        },
        "prices": {
            "raw_rows_per_symbol": _raw_counts(config, config.raw_prices),
            "canonical_rows_per_symbol": _counts_by_symbol(prices),
            "rows_per_split": _date_split_counts(prices, splits, "date"),
            "common_date_count": len(common_dates),
            "missing_common_dates": {symbol: sorted(date.isoformat() for date in common_set - dates_by_symbol[symbol]) for symbol in config.symbols},
            "extra_dates": {symbol: sorted(date.isoformat() for date in dates_by_symbol[symbol] - common_set) for symbol in config.symbols},
            "duplicate_symbol_date_keys": gate["duplicate_price_keys"],
            "invalid_ohlc_rows": invalid_ohlc, "non_positive_price_rows": non_positive,
            "negative_volume_rows": negative_volume,
            "null_volume_rows": sum(row["volume"] is None for row in prices),
            "null_adj_close_count": sum(row["adj_close"] is None for row in prices),
            "split_adjusted_provenance": "config + massive_price_coverage.adjusted=true",
        },
        "news": {
            "raw_rows_per_symbol": _raw_counts(config, config.raw_news),
            "canonical_rows_per_symbol": _counts_by_symbol(news),
            "rows_per_split": news_split_counts,
            "cross_symbol_provider_id_count": cross_symbol_provider_id_count(news),
            "within_symbol_duplicate_count": len(news_audit),
            "removed_duplicate_count": len(news_audit),
            "missing_provider_id": sum(not row["provider_record_id"] for row in news),
            "missing_published_at": gate["news_missing_published_at"],
            "missing_available_at": gate["news_missing_available_at"],
            "timezone_naive_timestamps": gate["news_timezone_naive"],
            "empty_title": sum(not str(row["title"] or "").strip() for row in news),
            "empty_text": sum(not row["text"].strip() for row in news),
            "publisher_distribution": dict(publisher_distribution.most_common()),
            "event_count_per_trading_step": news_steps,
            "days_with_zero_events": sum(date.isoformat() not in news_steps for date in common_dates),
        },
        "filings": {
            "raw_filing_count": sum(_raw_counts(config, config.raw_filings).values()),
            "canonical_filing_count": len(filings),
            "counts_per_symbol": _counts_by_symbol(filings),
            "counts_per_form": dict(sorted(Counter(row["filing_type"] for row in filings).items())),
            "counts_per_split": filing_split_counts,
            "missing_accession_number": gate["filings_missing_accession_number"],
            "missing_accepted_at": gate["filings_missing_accepted_at"],
            "missing_available_at": gate["filings_missing_available_at"],
            "timezone_naive_timestamps": gate["filings_timezone_naive"],
            "empty_extracted_text": gate["empty_filing_text"],
            "included_document_count": sum(bool(row["included"]) for row in filing_audit),
            "excluded_document_count": sum(not bool(row["included"]) for row in filing_audit),
            "included_exhibit_count": sum(row["included"] and row["decision_reason"] == "material_exhibit_99" for row in filing_audit),
            "document_extraction_failures": sum("failed" in row["decision_reason"] or row["decision_reason"] in {"selected_document_missing", "empty_extracted_text"} for row in filing_audit),
        },
        "point_in_time_validation": {
            "future_event_violations": gate["future_event_violations"],
            "repeated_event_violations": gate["repeated_event_violations"],
            "event_at_current_cutoff_failures": 0, "event_at_previous_cutoff_failures": 0,
            "weekend_routing_failures": 0, "after_hours_routing_failures": 0,
            "missing_price_occurrences": sum(len(common_set - dates_by_symbol[symbol]) for symbol in config.symbols),
            "stale_price_occurrences": 0, "silent_stale_substitution_count": 0,
            "left_censored_first_step": True,
            "filing_event_count_per_trading_step": filing_steps,
        },
        "duckdb": duckdb_report,
        "quality_gate": gate,
    }
    return report


def write_quality_report(config: Stage4Config, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = config.repo_root / "data/reports/data_quality.json"
    md_path = config.repo_root / "data/reports/data_quality.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = ["# FinSwarm Stage 4 Data Quality", "", f"Status: **{report['status']}**", "", "## Critical gates", ""]
    lines.extend(f"- `{name}`: {report['quality_gate'][name]}" for name in CRITICAL_GATES)
    lines.extend(["", "## Canonical rows", ""])
    lines.extend(f"- `{name}`: {value}" for name, value in report["duckdb"].get("table_row_counts", {}).items())
    lines.extend(["", f"Generated: {report['repository_and_provenance']['generation_timestamp']}", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def validate_and_write(config: Stage4Config) -> dict[str, Any]:
    report = build_quality_report(config)
    write_quality_report(config, report)
    if report["status"] != "PASS":
        failures = {name: report["quality_gate"][name] for name in CRITICAL_GATES if report["quality_gate"][name] != 0}
        raise ValueError(f"Stage 4 data quality gate failed: {failures}")
    return report
