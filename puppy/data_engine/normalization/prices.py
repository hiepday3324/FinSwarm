from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Stage4Config, read_jsonl, sha256_file, stable_json


def normalize_price_rows(
    raw_rows: list[dict[str, Any]],
    raw_path: str,
    raw_file_sha256: str,
    *,
    adjusted: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys: Counter[tuple[str, dt.date]] = Counter()
    for raw_line, raw in enumerate(raw_rows, 1):
        try:
            symbol = str(raw["ticker"])
            date = dt.date.fromisoformat(str(raw["market_date"]))
            open_price = float(raw["o"])
            high = float(raw["h"])
            low = float(raw["l"])
            close = float(raw["c"])
            volume = int(raw["v"])
            provider_timestamp = int(raw["t"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid price row at {raw_path}:{raw_line}: {exc}") from exc
        if min(open_price, high, low, close) <= 0:
            raise ValueError(f"non-positive price at {raw_path}:{raw_line}")
        if not (low <= open_price <= high and low <= close <= high):
            raise ValueError(f"invalid OHLC at {raw_path}:{raw_line}")
        if volume < 0:
            raise ValueError(f"negative volume at {raw_path}:{raw_line}")
        key = (symbol, date)
        keys[key] += 1
        rows.append(
            {
                "symbol": symbol,
                "date": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": None,
                "volume": volume,
                "source": "massive",
                "provider_record_id": f"massive:{symbol}:{provider_timestamp}",
                "metadata": stable_json(
                    {
                        "provider_timestamp_ms": provider_timestamp,
                        "timestamp_utc": raw.get("timestamp_utc"),
                        "timestamp_market": raw.get("timestamp_market"),
                        "vwap": raw.get("vw"),
                        "transactions": raw.get("n"),
                        "otc": raw.get("otc"),
                        "adjusted": bool(adjusted),
                        "adjustment_type": "split_adjusted" if adjusted else "unknown",
                        "raw_path": raw_path,
                        "raw_line": raw_line,
                        "raw_file_sha256": raw_file_sha256,
                    }
                ),
            }
        )
    duplicates = [key for key, count in keys.items() if count > 1]
    if duplicates:
        preview = ", ".join(f"{symbol}/{date}" for symbol, date in duplicates[:5])
        raise ValueError(f"duplicate canonical price key(s): {preview}")
    return sorted(rows, key=lambda row: (row["symbol"], row["date"]))


def normalize_prices(config: Stage4Config, adjusted: bool) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    raw_hashes: dict[str, str] = {}
    for symbol in config.symbols:
        files = sorted((config.raw_prices / symbol).glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"Missing raw price JSONL for {symbol}: {config.raw_prices / symbol}")
        for path in files:
            digest = sha256_file(path)
            relative = config.relative(path)
            raw = read_jsonl(path)
            raw_counts[symbol] = raw_counts.get(symbol, 0) + len(raw)
            raw_hashes[relative] = digest
            rows.extend(normalize_price_rows(raw, relative, digest, adjusted=adjusted))
    key_counts = Counter((row["symbol"], row["date"]) for row in rows)
    if any(count > 1 for count in key_counts.values()):
        raise ValueError("duplicate canonical price key(s) across raw files")
    return sorted(rows, key=lambda row: (row["symbol"], row["date"])), raw_counts, raw_hashes
