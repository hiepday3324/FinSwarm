from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

import duckdb

from puppy.common.schemas import FilingEvent, MarketStep, NewsEvent, PriceBar


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
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    return dt.datetime.fromisoformat(str(value))


def _metadata_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if isinstance(value, dict) else {})


class DuckDBMarketStore:
    def __init__(self, db_path: str = "data/duckdb/finswarm.db") -> None:
        self.db_path = db_path
        self.conn: duckdb.DuckDBPyConnection | None = None

        if db_path != ":memory:":
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

    def connect(self) -> None:
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.initialize_schema()

    def initialize_schema(self) -> None:
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
                PRIMARY KEY (symbol, date)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news (
                event_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                published_at TIMESTAMP,
                available_at TIMESTAMP,
                title VARCHAR,
                text VARCHAR,
                source VARCHAR,
                url VARCHAR,
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
                accepted_at TIMESTAMP,
                available_at TIMESTAMP,
                text VARCHAR,
                source VARCHAR,
                url VARCHAR,
                metadata_json VARCHAR
            )
            """
        )

    def ingest_prices(self, rows: Any) -> None:
        records = _records_from_rows(rows)
        if not records:
            return
        self.connect()
        normalized = [
            (
                str(row["symbol"]),
                _to_date(row["date"]),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                float(row["close"]),
                row.get("adj_close"),
                row.get("volume"),
                row.get("source", "unknown"),
            )
            for row in records
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO prices
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized,
        )

    def ingest_news(self, rows: Any) -> None:
        records = _records_from_rows(rows)
        if not records:
            return
        self.connect()
        normalized = [
            (
                str(row["event_id"]),
                str(row["symbol"]),
                _to_datetime(row["published_at"]),
                _to_datetime(row["available_at"]),
                row.get("title"),
                str(row["text"]),
                row.get("source", "unknown"),
                row.get("url"),
                _metadata_json(row.get("metadata", row.get("metadata_json", {}))),
            )
            for row in records
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO news
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized,
        )

    def ingest_filings(self, rows: Any) -> None:
        records = _records_from_rows(rows)
        if not records:
            return
        self.connect()
        normalized = [
            (
                str(row["filing_id"]),
                str(row["symbol"]),
                str(row.get("filing_type", "10-Q")),
                _to_date(row["filing_date"]),
                _to_datetime(row.get("accepted_at")),
                _to_datetime(row["available_at"]),
                str(row["text"]),
                row.get("source", "unknown"),
                row.get("url"),
                _metadata_json(row.get("metadata", row.get("metadata_json", {}))),
            )
            for row in records
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO filings
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized,
        )

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
            """
            SELECT DISTINCT date
            FROM prices
            WHERE symbol IN (SELECT symbol FROM temp_symbols)
            ORDER BY date
            """
        ).fetchall()
        return [row[0] if isinstance(row[0], dt.date) else row[0].date() for row in rows]

    def get_market_step(
        self, date: dt.date, symbols: list[str], as_of: dt.datetime | None = None
    ) -> MarketStep:
        self.connect()
        as_of = as_of or dt.datetime.combine(date, dt.time.max)
        if not symbols:
            return MarketStep(
                date=date,
                symbols=[],
                prices={},
                news={},
                filings={},
                as_of=as_of,
                metadata={},
            )

        self._register_symbols(symbols)
        price_rows = self.conn.execute(
            """
            SELECT p1.symbol, p1.date, p1.open, p1.high, p1.low, p1.close,
                   p1.adj_close, p1.volume, p1.source
            FROM prices p1
            INNER JOIN (
                SELECT symbol, MAX(date) AS max_date
                FROM prices
                WHERE date <= ? AND symbol IN (SELECT symbol FROM temp_symbols)
                GROUP BY symbol
            ) p2 ON p1.symbol = p2.symbol AND p1.date = p2.max_date
            """,
            [date],
        ).fetchall()
        news_rows = self.conn.execute(
            """
            SELECT event_id, symbol, published_at, available_at, title, text, source, url, metadata_json
            FROM news
            WHERE symbol IN (SELECT symbol FROM temp_symbols) AND available_at <= ?
            ORDER BY available_at, event_id
            """,
            [as_of],
        ).fetchall()
        filing_rows = self.conn.execute(
            """
            SELECT filing_id, symbol, filing_type, filing_date, accepted_at, available_at,
                   text, source, url, metadata_json
            FROM filings
            WHERE symbol IN (SELECT symbol FROM temp_symbols) AND available_at <= ?
            ORDER BY available_at, filing_id
            """,
            [as_of],
        ).fetchall()

        prices = {
            row[0]: PriceBar(
                symbol=row[0],
                date=_to_date(row[1]),
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                adj_close=row[6],
                volume=row[7],
                source=row[8],
            )
            for row in price_rows
        }
        news = {symbol: [] for symbol in symbols}
        for row in news_rows:
            if row[1] in news:
                news[row[1]].append(
                    NewsEvent(
                        event_id=row[0],
                        symbol=row[1],
                        published_at=_to_datetime(row[2]),
                        available_at=_to_datetime(row[3]),
                        title=row[4],
                        text=row[5],
                        source=row[6],
                        url=row[7],
                        metadata=json.loads(row[8]) if row[8] else {},
                    )
                )

        filings = {symbol: [] for symbol in symbols}
        for row in filing_rows:
            if row[1] in filings:
                filings[row[1]].append(
                    FilingEvent(
                        filing_id=row[0],
                        symbol=row[1],
                        filing_type=row[2],
                        filing_date=_to_date(row[3]),
                        accepted_at=_to_datetime(row[4]),
                        available_at=_to_datetime(row[5]),
                        text=row[6],
                        source=row[7],
                        url=row[8],
                        metadata=json.loads(row[9]) if row[9] else {},
                    )
                )

        return MarketStep(
            date=date,
            symbols=symbols,
            prices=prices,
            news=news,
            filings=filings,
            as_of=as_of,
            metadata={"backend": "duckdb"},
        )

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
