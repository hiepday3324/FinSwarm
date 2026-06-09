import os
import json
import datetime as dt
from typing import Any
import pandas as pd
import duckdb

from puppy.common.schemas import PriceBar, NewsEvent, FilingEvent, MarketStep

class DuckDBMarketStore:
    def __init__(self, db_path: str = "data/duckdb/finswarm.db") -> None:
        self.db_path = db_path
        self.conn = None
        
        # Create folder if it does not exist and is not in-memory
        if db_path != ":memory:":
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

    def connect(self) -> None:
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.initialize_schema()

    def initialize_schema(self) -> None:
        # Prices table
        self.conn.execute("""
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
        """)
        
        # News table
        self.conn.execute("""
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
        """)
        
        # Filings table
        self.conn.execute("""
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
        """)

    def ingest_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        
        self.connect()
        df_copy = df.copy()
        
        # Fill missing optional columns
        for col in ["open", "high", "low", "close", "adj_close", "volume", "source"]:
            if col not in df_copy.columns:
                df_copy[col] = None
                
        # Reorder columns to match schema explicitly
        df_copy = df_copy[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]]
        
        self.conn.register("df_temp_prices", df_copy)
        self.conn.execute("""
            INSERT OR REPLACE INTO prices 
            SELECT symbol, date, open, high, low, close, adj_close, volume, source
            FROM df_temp_prices
        """)
        self.conn.unregister("df_temp_prices")

    def ingest_news(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        
        self.connect()
        df_copy = df.copy()
        
        if "metadata" not in df_copy.columns:
            df_copy["metadata"] = [{}] * len(df_copy)
        df_copy["metadata_json"] = df_copy["metadata"].apply(json.dumps)
        
        # Fill missing optional columns
        for col in ["event_id", "symbol", "published_at", "available_at", "title", "text", "source", "url", "metadata_json"]:
            if col not in df_copy.columns:
                df_copy[col] = None
                
        # Reorder columns to match schema explicitly
        df_copy = df_copy[["event_id", "symbol", "published_at", "available_at", "title", "text", "source", "url", "metadata_json"]]
        
        self.conn.register("df_temp_news", df_copy)
        self.conn.execute("""
            INSERT OR REPLACE INTO news
            SELECT event_id, symbol, published_at, available_at, title, text, source, url, metadata_json
            FROM df_temp_news
        """)
        self.conn.unregister("df_temp_news")

    def ingest_filings(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        
        self.connect()
        df_copy = df.copy()
        
        if "metadata" not in df_copy.columns:
            df_copy["metadata"] = [{}] * len(df_copy)
        df_copy["metadata_json"] = df_copy["metadata"].apply(json.dumps)
        
        # Fill missing optional columns
        for col in ["filing_id", "symbol", "filing_type", "filing_date", "accepted_at", "available_at", "text", "source", "url", "metadata_json"]:
            if col not in df_copy.columns:
                df_copy[col] = None
                
        # Reorder columns to match schema explicitly
        df_copy = df_copy[["filing_id", "symbol", "filing_type", "filing_date", "accepted_at", "available_at", "text", "source", "url", "metadata_json"]]
        
        self.conn.register("df_temp_filings", df_copy)
        self.conn.execute("""
            INSERT OR REPLACE INTO filings
            SELECT filing_id, symbol, filing_type, filing_date, accepted_at, available_at, text, source, url, metadata_json
            FROM df_temp_filings
        """)
        self.conn.unregister("df_temp_filings")

    def get_market_step(self, date: dt.date, symbols: list[str], as_of: dt.datetime | None = None) -> MarketStep:
        self.connect()
        if as_of is None:
            # default as_of is end of the day t
            as_of = dt.datetime.combine(date, dt.time.max)
            
        # 1. Query price with date <= t (using inner join subquery to find max date <= t for each symbol)
        symbols_str = ", ".join(f"'{s}'" for s in symbols)
        
        price_query = f"""
            SELECT p1.symbol, p1.date, p1.open, p1.high, p1.low, p1.close, p1.adj_close, p1.volume, p1.source
            FROM prices p1
            INNER JOIN (
                SELECT symbol, MAX(date) as max_date 
                FROM prices 
                WHERE date <= ? AND symbol IN ({symbols_str})
                GROUP BY symbol
            ) p2 ON p1.symbol = p2.symbol AND p1.date = p2.max_date
        """
        price_rows = self.conn.execute(price_query, [date]).fetchall()
        
        prices_dict = {}
        for row in price_rows:
            prices_dict[row[0]] = PriceBar(
                symbol=row[0],
                date=row[1] if isinstance(row[1], dt.date) else row[1].date(),
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                adj_close=row[6],
                volume=row[7],
                source=row[8]
            )
            
        # 2. Query news available_at <= as_of
        news_query = f"""
            SELECT event_id, symbol, published_at, available_at, title, text, source, url, metadata_json
            FROM news
            WHERE symbol IN ({symbols_str}) AND available_at <= ?
        """
        news_rows = self.conn.execute(news_query, [as_of]).fetchall()
        
        news_dict = {s: [] for s in symbols}
        for row in news_rows:
            meta = json.loads(row[8]) if row[8] else {}
            event = NewsEvent(
                event_id=row[0],
                symbol=row[1],
                published_at=row[2],
                available_at=row[3],
                title=row[4],
                text=row[5],
                source=row[6],
                url=row[7],
                metadata=meta
            )
            if row[1] in news_dict:
                news_dict[row[1]].append(event)
                
        # 3. Query filings available_at <= as_of
        filings_query = f"""
            SELECT filing_id, symbol, filing_type, filing_date, accepted_at, available_at, text, source, url, metadata_json
            FROM filings
            WHERE symbol IN ({symbols_str}) AND available_at <= ?
        """
        filings_rows = self.conn.execute(filings_query, [as_of]).fetchall()
        
        filings_dict = {s: [] for s in symbols}
        for row in filings_rows:
            meta = json.loads(row[9]) if row[9] else {}
            filing = FilingEvent(
                filing_id=row[0],
                symbol=row[1],
                filing_type=row[2],
                filing_date=row[3] if isinstance(row[3], dt.date) else row[3].date(),
                accepted_at=row[4],
                available_at=row[5],
                text=row[6],
                source=row[7],
                url=row[8],
                metadata=meta
            )
            if row[1] in filings_dict:
                filings_dict[row[1]].append(filing)
                
        return MarketStep(
            date=date,
            symbols=symbols,
            prices=prices_dict,
            news=news_dict,
            filings=filings_dict,
            as_of=as_of,
            metadata={}
        )

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
