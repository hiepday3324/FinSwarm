from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .config import Stage4Config, load_stage4_config, sha256_file
from .filings import normalize_filings
from .news import normalize_news
from .prices import normalize_prices


PRICE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("date", pa.date32()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("adj_close", pa.float64()),
        ("volume", pa.int64()),
        ("source", pa.string()),
        ("provider_record_id", pa.string()),
        ("metadata", pa.string()),
    ]
)
NEWS_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("symbol", pa.string()),
        ("published_at", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("title", pa.string()),
        ("text", pa.string()),
        ("source", pa.string()),
        ("url", pa.string()),
        ("provider_record_id", pa.string()),
        ("metadata", pa.string()),
    ]
)
FILING_SCHEMA = pa.schema(
    [
        ("filing_id", pa.string()),
        ("symbol", pa.string()),
        ("filing_type", pa.string()),
        ("filing_date", pa.date32()),
        ("accepted_at", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("text", pa.string()),
        ("source", pa.string()),
        ("url", pa.string()),
        ("accession_number", pa.string()),
        ("provider_record_id", pa.string()),
        ("metadata", pa.string()),
    ]
)
NEWS_AUDIT_SCHEMA = pa.schema(
    [
        ("kept_event_id", pa.string()),
        ("removed_raw_path", pa.string()),
        ("removed_raw_line", pa.int64()),
        ("duplicate_reason", pa.string()),
        ("duplicate_key", pa.string()),
        ("provider_record_id", pa.string()),
        ("symbol", pa.string()),
    ]
)
FILING_AUDIT_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("accession_number", pa.string()),
        ("filing_type", pa.string()),
        ("document_name", pa.string()),
        ("document_type", pa.string()),
        ("local_path", pa.string()),
        ("included", pa.bool_()),
        ("decision_reason", pa.string()),
        ("content_type", pa.string()),
        ("extracted_character_count", pa.int64()),
        ("sha256", pa.string()),
    ]
)


def write_records(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )


def _split_adjusted_provenance(config: Stage4Config) -> bool:
    report_path = config.repo_root / "data/reports/massive_price_coverage.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing Massive price coverage report: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("adjusted") is not True:
        raise ValueError("Price coverage report does not confirm adjusted=true")
    return True


def normalize_all(config: Stage4Config) -> dict[str, Any]:
    adjusted = _split_adjusted_provenance(config)
    prices, raw_price_counts, price_hashes = normalize_prices(config, adjusted)
    news, news_audit, raw_news_counts, news_hashes = normalize_news(config)
    filings, filing_audit, raw_filing_counts, filing_hashes = normalize_filings(config)

    news_audit_path = config.repo_root / "data/reports/news_dedup_audit.parquet"
    filing_audit_path = config.repo_root / "data/reports/filing_document_audit.parquet"
    write_records(config.canonical_prices, prices, PRICE_SCHEMA)
    write_records(config.canonical_news, news, NEWS_SCHEMA)
    write_records(config.canonical_filings, filings, FILING_SCHEMA)
    write_records(news_audit_path, news_audit, NEWS_AUDIT_SCHEMA)
    write_records(filing_audit_path, filing_audit, FILING_AUDIT_SCHEMA)

    outputs = {
        config.relative(path): sha256_file(path)
        for path in (
            config.canonical_prices,
            config.canonical_news,
            config.canonical_filings,
            news_audit_path,
            filing_audit_path,
        )
    }
    return {
        "raw_counts": {
            "prices": raw_price_counts,
            "news": raw_news_counts,
            "filings": raw_filing_counts,
        },
        "raw_input_hashes": {**price_hashes, **news_hashes, **filing_hashes},
        "canonical_counts": {
            "prices": len(prices),
            "news": len(news),
            "filings": len(filings),
        },
        "audit_counts": {
            "news_duplicates_removed": len(news_audit),
            "filing_documents": len(filing_audit),
            "filing_documents_included": sum(bool(row["included"]) for row in filing_audit),
            "filing_documents_excluded": sum(not bool(row["included"]) for row in filing_audit),
        },
        "output_hashes": outputs,
    }


__all__ = [
    "FILING_SCHEMA",
    "NEWS_SCHEMA",
    "PRICE_SCHEMA",
    "Stage4Config",
    "load_stage4_config",
    "normalize_all",
]
