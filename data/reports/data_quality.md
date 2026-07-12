# FinSwarm Stage 4 Data Quality

Status: **PASS**

## Critical gates

- `duplicate_price_keys`: 0
- `invalid_ohlc_rows`: 0
- `non_positive_price_rows`: 0
- `negative_volume_rows`: 0
- `news_missing_published_at`: 0
- `news_missing_available_at`: 0
- `news_timezone_naive`: 0
- `filings_missing_accession_number`: 0
- `filings_missing_accepted_at`: 0
- `filings_missing_available_at`: 0
- `filings_timezone_naive`: 0
- `empty_filing_text`: 0
- `future_event_violations`: 0
- `repeated_event_violations`: 0
- `silent_stale_price_substitutions`: 0
- `duckdb_duplicate_primary_keys`: 0
- `parquet_duckdb_count_mismatches`: 0

## Canonical rows

- `prices`: 1503
- `news`: 16911
- `filings`: 89

Generated: 2026-07-11T23:02:39.357190+00:00
