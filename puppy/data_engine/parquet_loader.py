from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import warnings
from typing import Any

import pyarrow.parquet as pq


PRICE_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close", "adj_close", "volume",
    "source", "provider_record_id", "metadata",
]
NEWS_COLUMNS = [
    "event_id", "symbol", "published_at", "available_at", "title", "text", "source",
    "url", "provider_record_id", "metadata",
]
FILING_COLUMNS = [
    "filing_id", "symbol", "filing_type", "filing_date", "accepted_at", "available_at",
    "text", "source", "url", "accession_number", "provider_record_id", "metadata",
]


class RecordFrame:
    """Tiny records table used to avoid pandas conversion in parquet loading."""

    def __init__(self, records: list[dict[str, Any]], columns: list[str]) -> None:
        self._records = [{column: record.get(column) for column in columns} for record in records]
        self.columns = list(columns)

    @property
    def empty(self) -> bool:
        return len(self._records) == 0

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        if orient != "records":
            raise ValueError("RecordFrame only supports orient='records'.")
        return [dict(record) for record in self._records]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lower().strip(): value for key, value in record.items()}


def _read_parquet_records(path: str, columns: list[str]) -> RecordFrame:
    if not os.path.exists(path):
        return RecordFrame([], columns)
    records = [_normalize_record(record) for record in pq.read_table(path).to_pylist()]
    actual_columns = [str(name).lower().strip() for name in pq.read_schema(path).names]
    return RecordFrame(records, actual_columns)


def _require_columns(frame: RecordFrame, required: list[str], path: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in parquet {path}")


def _stable_id(*parts: Any) -> str:
    joined = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _ensure_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata must be valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
    if value is None:
        return {}
    raise ValueError("metadata must be a JSON object")


def _to_date(value: dt.date | dt.datetime | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _to_datetime(value: dt.datetime | dt.date | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    text = str(value)
    return dt.datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)


def _to_aware_utc(value: Any, field_name: str, path: str) -> dt.datetime:
    if value is None:
        raise ValueError(f"{field_name} is required in canonical parquet {path}")
    parsed = _to_datetime(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware in canonical parquet {path}")
    return parsed.astimezone(dt.timezone.utc)


def load_price_parquet(path: str) -> RecordFrame:
    frame = _read_parquet_records(path, PRICE_COLUMNS)
    if frame.empty:
        return RecordFrame([], PRICE_COLUMNS)
    _require_columns(frame, PRICE_COLUMNS, path)
    records = []
    for row in frame:
        records.append(
            {
                "symbol": str(row["symbol"]),
                "date": _to_date(row["date"]),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": float(row["close"]),
                "adj_close": row.get("adj_close"),
                "volume": row.get("volume"),
                "source": str(row.get("source") or "parquet"),
                "provider_record_id": row.get("provider_record_id"),
                "metadata": _ensure_metadata(row.get("metadata")),
            }
        )
    return RecordFrame(records, PRICE_COLUMNS)


def load_news_parquet(path: str) -> RecordFrame:
    """Strictly load canonical Stage 4 news without inferring availability."""
    frame = _read_parquet_records(path, NEWS_COLUMNS)
    if frame.empty:
        return RecordFrame([], NEWS_COLUMNS)
    _require_columns(frame, NEWS_COLUMNS, path)
    records = []
    for row in frame:
        if not row.get("event_id"):
            raise ValueError(f"event_id is required in canonical parquet {path}")
        records.append(
            {
                "event_id": str(row["event_id"]),
                "symbol": str(row["symbol"]),
                "published_at": _to_aware_utc(row.get("published_at"), "published_at", path),
                "available_at": _to_aware_utc(row.get("available_at"), "available_at", path),
                "title": row.get("title"),
                "text": str(row["text"]),
                "source": str(row.get("source") or "parquet"),
                "url": row.get("url"),
                "provider_record_id": row.get("provider_record_id"),
                "metadata": _ensure_metadata(row.get("metadata")),
            }
        )
    return RecordFrame(records, NEWS_COLUMNS)


def load_filings_parquet(path: str) -> RecordFrame:
    """Strictly load canonical Stage 4 filings without timestamp fallbacks."""
    frame = _read_parquet_records(path, FILING_COLUMNS)
    if frame.empty:
        return RecordFrame([], FILING_COLUMNS)
    _require_columns(frame, FILING_COLUMNS, path)
    records = []
    for row in frame:
        if not row.get("filing_id") or not row.get("accession_number"):
            raise ValueError(f"filing_id and accession_number are required in canonical parquet {path}")
        records.append(
            {
                "filing_id": str(row["filing_id"]),
                "symbol": str(row["symbol"]),
                "filing_type": str(row["filing_type"]),
                "filing_date": _to_date(row["filing_date"]),
                "accepted_at": _to_aware_utc(row.get("accepted_at"), "accepted_at", path),
                "available_at": _to_aware_utc(row.get("available_at"), "available_at", path),
                "text": str(row["text"]),
                "source": str(row.get("source") or "parquet"),
                "url": row.get("url"),
                "accession_number": str(row["accession_number"]),
                "provider_record_id": row.get("provider_record_id"),
                "metadata": _ensure_metadata(row.get("metadata")),
            }
        )
    return RecordFrame(records, FILING_COLUMNS)


def load_news_parquet_legacy(path: str) -> RecordFrame:
    warnings.warn(
        "load_news_parquet_legacy infers missing availability and is deprecated",
        DeprecationWarning,
        stacklevel=2,
    )
    frame = _read_parquet_records(path, NEWS_COLUMNS)
    if frame.empty:
        return RecordFrame([], NEWS_COLUMNS)
    _require_columns(frame, ["symbol", "text"], path)
    if "published_at" not in frame.columns and "available_at" not in frame.columns:
        raise ValueError(f"News parquet {path} must include published_at or available_at.")
    records = []
    for row in frame:
        published_at = _to_datetime(row.get("published_at") or row["available_at"])
        available_at = _to_datetime(row.get("available_at") or published_at)
        text = str(row["text"])
        event_id = row.get("event_id") or _stable_id(row["symbol"], published_at.isoformat(), available_at.isoformat(), text)
        records.append({
            "event_id": str(event_id), "symbol": str(row["symbol"]),
            "published_at": published_at, "available_at": available_at,
            "title": row.get("title"), "text": text,
            "source": str(row.get("source") or "parquet"), "url": row.get("url"),
            "provider_record_id": row.get("provider_record_id"),
            "metadata": _ensure_metadata(row.get("metadata")),
        })
    return RecordFrame(records, NEWS_COLUMNS)


def load_filings_parquet_legacy(path: str) -> RecordFrame:
    warnings.warn(
        "load_filings_parquet_legacy infers missing availability and is deprecated",
        DeprecationWarning,
        stacklevel=2,
    )
    frame = _read_parquet_records(path, FILING_COLUMNS)
    if frame.empty:
        return RecordFrame([], FILING_COLUMNS)
    _require_columns(frame, ["symbol", "text"], path)
    if "filing_date" not in frame.columns and "available_at" not in frame.columns:
        raise ValueError(f"Filing parquet {path} must include filing_date or available_at.")
    records = []
    for row in frame:
        filing_date = _to_date(row.get("filing_date") or row["available_at"])
        available_at = _to_datetime(row.get("available_at") or dt.datetime.combine(filing_date, dt.time.min))
        filing_type = str(row.get("filing_type") or "10-Q")
        text = str(row["text"])
        filing_id = row.get("filing_id") or _stable_id(row["symbol"], filing_type, filing_date, available_at, text)
        records.append({
            "filing_id": str(filing_id), "symbol": str(row["symbol"]),
            "filing_type": filing_type, "filing_date": filing_date,
            "accepted_at": _to_datetime(row["accepted_at"]) if row.get("accepted_at") else None,
            "available_at": available_at, "text": text,
            "source": str(row.get("source") or "parquet"), "url": row.get("url"),
            "accession_number": row.get("accession_number"),
            "provider_record_id": row.get("provider_record_id"),
            "metadata": _ensure_metadata(row.get("metadata")),
        })
    return RecordFrame(records, FILING_COLUMNS)


def load_market_parquets(base_dir: str) -> dict[str, RecordFrame]:
    return {
        "prices": load_price_parquet(os.path.join(base_dir, "prices.parquet")),
        "news": load_news_parquet(os.path.join(base_dir, "news.parquet")),
        "filings": load_filings_parquet(os.path.join(base_dir, "filings.parquet")),
    }
