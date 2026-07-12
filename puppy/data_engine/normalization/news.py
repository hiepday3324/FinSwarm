from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Stage4Config, parse_utc, read_jsonl, sha256_file, stable_json


def _fallback_key(raw: dict[str, Any], symbol: str, published_at: Any) -> str:
    title = re.sub(r"\s+", " ", str(raw.get("title") or "").strip().lower())
    publisher = raw.get("publisher") or {}
    publisher_name = publisher.get("name", "") if isinstance(publisher, dict) else str(publisher)
    description_hash = hashlib.sha256(str(raw.get("description") or "").encode("utf-8")).hexdigest()
    return "|".join((symbol, title, str(publisher_name).strip().lower(), published_at.isoformat(), description_hash))


def normalize_news_rows(
    inputs: list[tuple[dict[str, Any], str, int, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for raw, raw_path, raw_line, raw_hash in inputs:
        symbol = str(raw.get("queried_symbol") or "").strip()
        if not symbol:
            raise ValueError(f"queried_symbol is required at {raw_path}:{raw_line}")
        published_at = parse_utc(raw.get("published_utc"), "published_utc")
        provider_id = str(raw.get("id") or "").strip() or None
        if provider_id:
            duplicate_key = f"id|{symbol}|{provider_id}"
            event_id = f"massive:{provider_id}:{symbol}"
            reason = "same_symbol_provider_id"
        else:
            duplicate_key = "fallback|" + _fallback_key(raw, symbol, published_at)
            digest = hashlib.sha256(duplicate_key.encode("utf-8")).hexdigest()
            event_id = f"massive:fallback:{digest}:{symbol}"
            reason = "same_symbol_fallback_key"
        if duplicate_key in seen:
            audit.append(
                {
                    "kept_event_id": seen[duplicate_key],
                    "removed_raw_path": raw_path,
                    "removed_raw_line": raw_line,
                    "duplicate_reason": reason,
                    "duplicate_key": duplicate_key,
                    "provider_record_id": provider_id,
                    "symbol": symbol,
                }
            )
            continue
        seen[duplicate_key] = event_id
        rows.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "published_at": published_at,
                "available_at": published_at,
                "title": raw.get("title"),
                "text": str(raw.get("description") or ""),
                "source": "massive",
                "url": raw.get("article_url"),
                "provider_record_id": provider_id,
                "metadata": stable_json(
                    {
                        "publisher": raw.get("publisher"),
                        "author": raw.get("author"),
                        "tickers": raw.get("tickers"),
                        "keywords": raw.get("keywords"),
                        "insights": raw.get("insights"),
                        "image_url": raw.get("image_url"),
                        "amp_url": raw.get("amp_url"),
                        "queried_symbol": symbol,
                        "original_published_utc": raw.get("published_utc"),
                        "availability_assumption": "published_utc",
                        "raw_path": raw_path,
                        "raw_line": raw_line,
                        "raw_file_sha256": raw_hash,
                    }
                ),
            }
        )
    rows.sort(key=lambda row: (row["available_at"], row["symbol"], row["event_id"]))
    audit.sort(key=lambda row: (row["symbol"], row["removed_raw_path"], row["removed_raw_line"]))
    return rows, audit


def normalize_news(config: Stage4Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, str]]:
    inputs: list[tuple[dict[str, Any], str, int, str]] = []
    raw_counts: Counter[str] = Counter()
    raw_hashes: dict[str, str] = {}
    for symbol in config.symbols:
        files = sorted((config.raw_news / symbol).glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"Missing raw news JSONL for {symbol}: {config.raw_news / symbol}")
        for path in files:
            digest = sha256_file(path)
            relative = config.relative(path)
            raw_hashes[relative] = digest
            for raw_line, raw in enumerate(read_jsonl(path), 1):
                raw_counts[symbol] += 1
                inputs.append((raw, relative, raw_line, digest))
    rows, audit = normalize_news_rows(inputs)
    return rows, audit, dict(raw_counts), raw_hashes


def cross_symbol_provider_id_count(rows: list[dict[str, Any]]) -> int:
    symbols_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        provider_id = row.get("provider_record_id")
        if provider_id:
            symbols_by_id[str(provider_id)].add(str(row["symbol"]))
    return sum(1 for symbols in symbols_by_id.values() if len(symbols) > 1)
