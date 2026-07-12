from __future__ import annotations

import datetime as dt
import json
import os
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from zoneinfo import ZoneInfo

import duckdb

from puppy.common.schemas import FilingEvent, MarketStep, NewsEvent, PriceBar
from puppy.data_engine.normalization.config import cutoff_utc


UTC = dt.timezone.utc


def _records_from_rows(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        try:
            return [dict(item) for item in rows.to_dict(orient="records")]
        except TypeError:
            return [dict(item) for item in rows.to_dict("records")]
    if isinstance(rows, Mapping):
        return [dict(rows)]
    return [dict(item) for item in rows]


def _to_date(value: dt.date | dt.datetime | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _to_datetime(value: dt.datetime | dt.date | str | None) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime.combine(value, dt.time.min, UTC)
    else:
        text = str(value)
        parsed = dt.datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"event timestamp must be timezone-aware: {value!r}")
    return parsed.astimezone(UTC)


def _metadata_json(value: Any) -> str:
    if isinstance(value, str):
        json.loads(value)
        return value
    return json.dumps(value if isinstance(value, dict) else {}, sort_keys=True, separators=(",", ":"))


def _duplicates(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[tuple[Any, ...]]:
    counts = Counter(tuple(row.get(field) for field in key_fields) for row in records)
    return [key for key, count in counts.items() if count > 1]


class DuckDBMarketStore:
    def __init__(
        self,
        db_path: str = "data/duckdb/finswarm.db",
        *,
        market_timezone: str = "America/New_York",
        decision_cutoff: dt.time = dt.time(16, 10),
    ) -> None:
        self.db_path = db_path
        self.market_timezone = ZoneInfo(market_timezone)
        self.decision_cutoff = decision_cutoff
        self.conn: duckdb.DuckDBPyConnection | None = None
        if db_path != ":memory:":
            directory = os.path.dirname(db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def connect(self) -> None:
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.initialize_schema()

    def initialize_schema(self) -> None:
        if self.conn is None:
            raise RuntimeError("DuckDB connection is not open")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                source VARCHAR,
                provider_record_id VARCHAR,
                metadata_json VARCHAR,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news (
                event_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                published_at TIMESTAMPTZ,
                available_at TIMESTAMPTZ,
                title VARCHAR,
                text VARCHAR,
                source VARCHAR,
                url VARCHAR,
                provider_record_id VARCHAR,
                metadata_json VARCHAR
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filings (
                filing_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                filing_type VARCHAR,
                filing_date DATE,
                accepted_at TIMESTAMPTZ,
                available_at TIMESTAMPTZ,
                text VARCHAR,
                source VARCHAR,
                url VARCHAR,
                accession_number VARCHAR,
                provider_record_id VARCHAR,
                metadata_json VARCHAR
            )
            """
        )

    def _ensure_unique(self, table: str, records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> None:
        duplicate_keys = _duplicates(records, key_fields)
        if duplicate_keys:
            raise ValueError(f"duplicate primary key(s) for {table}: {duplicate_keys[:5]}")
        if not records:
            return
        self.connect()
        if len(key_fields) == 1:
            existing = self.conn.execute(
                f"SELECT {key_fields[0]} FROM {table} WHERE {key_fields[0]} IN (SELECT * FROM UNNEST(?))",
                [[row[key_fields[0]] for row in records]],
            ).fetchall()
        else:
            existing = []
            for row in records:
                where = " AND ".join(f"{field} = ?" for field in key_fields)
                if self.conn.execute(
                    f"SELECT 1 FROM {table} WHERE {where} LIMIT 1",
                    [row[field] for field in key_fields],
                ).fetchone():
                    existing.append(tuple(row[field] for field in key_fields))
        if existing:
            raise ValueError(f"duplicate primary key(s) already present in {table}: {existing[:5]}")

    def ingest_prices(self, rows: Any) -> None:
        records = _records_from_rows(rows)
        if not records:
            return
        self._ensure_unique("prices", records, ("symbol", "date"))
        normalized = [
            (
                str(row["symbol"]), _to_date(row["date"]), row.get("open"), row.get("high"),
                row.get("low"), float(row["close"]), row.get("adj_close"), row.get("volume"),
                row.get("source", "unknown"), row.get("provider_record_id"),
                _metadata_json(row.get("metadata", row.get("metadata_json", {}))),
            )
            for row in records
        ]
        self.conn.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", normalized)

    def ingest_news(self, rows: Any) -> None:
        records = _records_from_rows(rows)
        if not records:
            return
        self._ensure_unique("news", records, ("event_id",))
        normalized = [
            (
                str(row["event_id"]), str(row["symbol"]), _to_datetime(row["published_at"]),
                _to_datetime(row["available_at"]), row.get("title"), str(row["text"]),
                row.get("source", "unknown"), row.get("url"), row.get("provider_record_id"),
                _metadata_json(row.get("metadata", row.get("metadata_json", {}))),
            )
            for row in records
        ]
        self.conn.executemany("INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", normalized)

    def ingest_filings(self, rows: Any) -> None:
        records = _records_from_rows(rows)
        if not records:
            return
        self._ensure_unique("filings", records, ("filing_id",))
        normalized = [
            (
                str(row["filing_id"]), str(row["symbol"]), str(row["filing_type"]),
                _to_date(row["filing_date"]), _to_datetime(row["accepted_at"]),
                _to_datetime(row["available_at"]), str(row["text"]), row.get("source", "unknown"),
                row.get("url"), row.get("accession_number"), row.get("provider_record_id"),
                _metadata_json(row.get("metadata", row.get("metadata_json", {}))),
            )
            for row in records
        ]
        self.conn.executemany("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", normalized)

    def _register_symbols(self, symbols: list[str]) -> None:
        self.conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS temp_symbols(symbol VARCHAR)")
        self.conn.execute("DELETE FROM temp_symbols")
        self.conn.executemany("INSERT INTO temp_symbols VALUES (?)", [(symbol,) for symbol in symbols])

    def list_dates(self, symbols: Iterable[str]) -> list[dt.date]:
        self.connect()
        symbols = list(symbols)
        if not symbols:
            return []
        self._register_symbols(symbols)
        rows = self.conn.execute(
            "SELECT DISTINCT date FROM prices WHERE symbol IN (SELECT symbol FROM temp_symbols) ORDER BY date"
        ).fetchall()
        return [_to_date(row[0]) for row in rows]

    def list_common_dates(self, symbols: Iterable[str]) -> list[dt.date]:
        self.connect()
        symbols = list(symbols)
        if not symbols:
            return []
        self._register_symbols(symbols)
        rows = self.conn.execute(
            """
            SELECT date FROM prices
            WHERE symbol IN (SELECT symbol FROM temp_symbols)
            GROUP BY date HAVING COUNT(DISTINCT symbol) = ? ORDER BY date
            """,
            [len(set(symbols))],
        ).fetchall()
        return [_to_date(row[0]) for row in rows]

    def _price_objects(self, rows: list[tuple[Any, ...]]) -> dict[str, PriceBar]:
        return {
            row[0]: PriceBar(
                symbol=row[0], date=_to_date(row[1]), open=row[2], high=row[3], low=row[4],
                close=row[5], adj_close=row[6], volume=row[7], source=row[8],
                provider_record_id=row[9], metadata=json.loads(row[10]) if row[10] else {},
            )
            for row in rows
        }

    def _news_objects(self, rows: list[tuple[Any, ...]], symbols: list[str]) -> dict[str, list[NewsEvent]]:
        result = {symbol: [] for symbol in symbols}
        for row in rows:
            result[row[1]].append(NewsEvent(
                event_id=row[0], symbol=row[1], published_at=_to_datetime(row[2]),
                available_at=_to_datetime(row[3]), title=row[4], text=row[5], source=row[6],
                url=row[7], provider_record_id=row[8], metadata=json.loads(row[9]) if row[9] else {},
            ))
        return result

    def _filing_objects(self, rows: list[tuple[Any, ...]], symbols: list[str]) -> dict[str, list[FilingEvent]]:
        result = {symbol: [] for symbol in symbols}
        for row in rows:
            result[row[1]].append(FilingEvent(
                filing_id=row[0], symbol=row[1], filing_type=row[2], filing_date=_to_date(row[3]),
                accepted_at=_to_datetime(row[4]), available_at=_to_datetime(row[5]), text=row[6],
                source=row[7], url=row[8], accession_number=row[9], provider_record_id=row[10],
                metadata=json.loads(row[11]) if row[11] else {},
            ))
        return result

    def get_point_in_time_market_step(
        self,
        date: dt.date,
        symbols: list[str],
        common_dates: Iterable[dt.date],
    ) -> MarketStep:
        self.connect()
        dates = sorted({_to_date(item) for item in common_dates})
        if date not in dates:
            raise ValueError(f"Requested date {date} is not in the common trading calendar")
        current_index = dates.index(date)
        current_cutoff = cutoff_utc(date, self.decision_cutoff, self.market_timezone)
        left_censored = current_index == 0
        previous_cutoff = (
            None if left_censored
            else cutoff_utc(dates[current_index - 1], self.decision_cutoff, self.market_timezone)
        )
        lower_boundary = previous_cutoff or dt.datetime.combine(dates[0], dt.time.min, self.market_timezone).astimezone(UTC)
        if not symbols:
            return MarketStep(
                date=date, symbols=[], prices={}, news={}, filings={}, as_of=current_cutoff,
                metadata={
                    "previous_cutoff": previous_cutoff.isoformat() if previous_cutoff else None,
                    "current_cutoff": current_cutoff.isoformat(), "missing_symbols": [],
                    "stale_symbols": [], "backend": "duckdb",
                    "event_window": "previous_cutoff_open_current_cutoff_closed",
                    "left_censored": left_censored,
                },
            )
        self._register_symbols(symbols)
        price_rows = self.conn.execute(
            """
            SELECT symbol, date, open, high, low, close, adj_close, volume, source,
                   provider_record_id, metadata_json
            FROM prices WHERE date = ? AND symbol IN (SELECT symbol FROM temp_symbols)
            ORDER BY symbol
            """,
            [date],
        ).fetchall()
        present = {row[0] for row in price_rows}
        missing = sorted(set(symbols) - present)
        stale = []
        if missing:
            prior = self.conn.execute(
                """
                SELECT DISTINCT symbol FROM prices
                WHERE date < ? AND symbol IN (SELECT symbol FROM temp_symbols)
                """,
                [date],
            ).fetchall()
            stale = sorted(row[0] for row in prior if row[0] in missing)
        operator = ">=" if left_censored else ">"
        news_rows = self.conn.execute(
            f"""
            SELECT event_id, symbol, published_at, available_at, title, text, source, url,
                   provider_record_id, metadata_json
            FROM news WHERE symbol IN (SELECT symbol FROM temp_symbols)
              AND available_at {operator} ? AND available_at <= ?
            ORDER BY available_at, symbol, event_id
            """,
            [lower_boundary, current_cutoff],
        ).fetchall()
        filing_rows = self.conn.execute(
            f"""
            SELECT filing_id, symbol, filing_type, filing_date, accepted_at, available_at,
                   text, source, url, accession_number, provider_record_id, metadata_json
            FROM filings WHERE symbol IN (SELECT symbol FROM temp_symbols)
              AND available_at {operator} ? AND available_at <= ?
            ORDER BY available_at, symbol, filing_id
            """,
            [lower_boundary, current_cutoff],
        ).fetchall()
        return MarketStep(
            date=date,
            symbols=list(symbols),
            prices=self._price_objects(price_rows),
            news=self._news_objects(news_rows, symbols),
            filings=self._filing_objects(filing_rows, symbols),
            as_of=current_cutoff,
            metadata={
                "previous_cutoff": previous_cutoff.isoformat() if previous_cutoff else None,
                "current_cutoff": current_cutoff.isoformat(),
                "missing_symbols": missing,
                "stale_symbols": stale,
                "backend": "duckdb",
                "event_window": "previous_cutoff_open_current_cutoff_closed",
                "left_censored": left_censored,
            },
        )

    def get_market_step(
        self, date: dt.date, symbols: list[str], as_of: dt.datetime | None = None
    ) -> MarketStep:
        """Production point-in-time query; ``as_of`` is retained but must match the cutoff."""
        dates = self.list_common_dates(symbols) or self.list_dates(symbols)
        step = self.get_point_in_time_market_step(date, symbols, dates or [date])
        if as_of is not None and _to_datetime(as_of) != step.as_of:
            raise ValueError("Production MarketStep.as_of is fixed by the configured decision cutoff")
        return step

    def get_legacy_market_step(
        self, date: dt.date, symbols: list[str], as_of: dt.datetime | None = None
    ) -> MarketStep:
        warnings.warn(
            "get_legacy_market_step uses latest-prior prices and all-history events",
            DeprecationWarning,
            stacklevel=2,
        )
        self.connect()
        as_of = _to_datetime(as_of) if as_of is not None else dt.datetime.combine(date, dt.time.max, UTC)
        if not symbols:
            return MarketStep(date=date, symbols=[], prices={}, news={}, filings={}, as_of=as_of, metadata={"backend": "duckdb_legacy"})
        self._register_symbols(symbols)
        price_rows = self.conn.execute(
            """
            SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume,
                   p.source, p.provider_record_id, p.metadata_json
            FROM prices p JOIN (
                SELECT symbol, MAX(date) max_date FROM prices
                WHERE date <= ? AND symbol IN (SELECT symbol FROM temp_symbols) GROUP BY symbol
            ) latest ON p.symbol = latest.symbol AND p.date = latest.max_date
            """,
            [date],
        ).fetchall()
        news_rows = self.conn.execute(
            """
            SELECT event_id, symbol, published_at, available_at, title, text, source, url,
                   provider_record_id, metadata_json FROM news
            WHERE symbol IN (SELECT symbol FROM temp_symbols) AND available_at <= ?
            ORDER BY available_at, event_id
            """,
            [as_of],
        ).fetchall()
        filing_rows = self.conn.execute(
            """
            SELECT filing_id, symbol, filing_type, filing_date, accepted_at, available_at,
                   text, source, url, accession_number, provider_record_id, metadata_json FROM filings
            WHERE symbol IN (SELECT symbol FROM temp_symbols) AND available_at <= ?
            ORDER BY available_at, filing_id
            """,
            [as_of],
        ).fetchall()
        return MarketStep(
            date=date, symbols=list(symbols), prices=self._price_objects(price_rows),
            news=self._news_objects(news_rows, symbols), filings=self._filing_objects(filing_rows, symbols),
            as_of=as_of, metadata={"backend": "duckdb_legacy"},
        )

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
