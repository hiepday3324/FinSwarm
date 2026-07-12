from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Stage4Config, parse_utc, read_jsonl, sha256_file, stable_json
from .text_extraction import extract_filing_documents


def normalize_filing_record(
    raw: dict[str, Any],
    raw_metadata_path: str,
    raw_line: int,
    raw_metadata_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbol = str(raw.get("symbol") or "").strip()
    accession = str(raw.get("accession_number") or "").strip()
    if not accession:
        raise ValueError(f"accession_number is required at {raw_metadata_path}:{raw_line}")
    accepted_at = parse_utc(raw.get("acceptance_datetime"), "acceptance_datetime")
    try:
        filing_date = dt.date.fromisoformat(str(raw["filing_date"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid filing_date at {raw_metadata_path}:{raw_line}") from exc
    filing_type = str(raw.get("form") or "").strip()
    documents = [dict(item) for item in raw.get("files", []) if isinstance(item, dict)]
    primary_path = str(raw.get("primary_document_local_path") or "")
    primary_name = str(raw.get("primary_document") or Path(primary_path).name)
    if primary_path and not any(str(item.get("local_path") or "") == primary_path for item in documents):
        documents.append({"name": primary_name, "type": filing_type, "local_path": primary_path})
    text, audit, included, excluded = extract_filing_documents(
        symbol, accession, filing_type, primary_name, documents
    )
    if not text.strip():
        raise ValueError(f"empty filing text for accession {accession}")
    metadata = {
        "cik": raw.get("cik"),
        "accession_compact": raw.get("accession_compact"),
        "form": raw.get("form"),
        "report_date": raw.get("report_date"),
        "items": raw.get("items"),
        "primary_document": raw.get("primary_document"),
        "primary_document_description": raw.get("primary_document_description"),
        "primary_document_local_path": primary_path,
        "primary_document_exists": raw.get("primary_document_exists"),
        "index_url": raw.get("index_url"),
        "index_local_path": raw.get("index_local_path"),
        "is_xbrl": raw.get("is_xbrl"),
        "is_inline_xbrl": raw.get("is_inline_xbrl"),
        "file_number": raw.get("file_number"),
        "film_number": raw.get("film_number"),
        "document_list": documents,
        "included_documents": included,
        "excluded_documents": excluded,
        "raw_metadata_path": raw_metadata_path,
        "raw_line": raw_line,
        "raw_file_hashes": {
            raw_metadata_path: raw_metadata_sha256,
            **{item["local_path"]: item["sha256"] for item in included + excluded if item.get("sha256")},
        },
        "original_acceptance_datetime": raw.get("acceptance_datetime"),
    }
    return (
        {
            "filing_id": f"sec:{accession}",
            "symbol": symbol,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "accepted_at": accepted_at,
            "available_at": accepted_at,
            "text": text,
            "source": "sec_edgar",
            "url": raw.get("index_url"),
            "accession_number": accession,
            "provider_record_id": accession,
            "metadata": stable_json(metadata),
        },
        audit,
    )


def normalize_filings(config: Stage4Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    raw_hashes: dict[str, str] = {}
    seen: set[str] = set()
    for symbol in config.symbols:
        files = sorted((config.raw_filings / symbol).glob("*_metadata.jsonl"))
        if not files:
            raise FileNotFoundError(f"Missing SEC metadata JSONL for {symbol}: {config.raw_filings / symbol}")
        for path in files:
            digest = sha256_file(path)
            relative = config.relative(path)
            raw_hashes[relative] = digest
            for raw_line, raw in enumerate(read_jsonl(path), 1):
                raw_counts[symbol] += 1
                row, document_audit = normalize_filing_record(raw, relative, raw_line, digest)
                if row["filing_id"] in seen:
                    raise ValueError(f"duplicate filing accession: {row['accession_number']}")
                seen.add(row["filing_id"])
                rows.append(row)
                audit.extend(document_audit)
    rows.sort(key=lambda row: (row["available_at"], row["symbol"], row["filing_id"]))
    audit.sort(key=lambda row: (row["symbol"], row["accession_number"], row["document_name"]))
    return rows, audit, dict(raw_counts), raw_hashes
