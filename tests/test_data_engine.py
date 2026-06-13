import datetime as dt
import os
import tempfile
import unittest

from puppy.common.schemas import MarketStep
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.multi_asset_env import MultiAssetEnvironment
from puppy.data_engine.parquet_loader import _stable_id


class DataEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = DuckDBMarketStore(db_path=":memory:")
        self.store.connect()

    def tearDown(self) -> None:
        self.store.close()

    def test_schema_init_and_ingestion(self) -> None:
        tables = self.store.conn.execute("SHOW TABLES").fetchall()
        table_names = {table[0] for table in tables}
        self.assertIn("prices", table_names)
        self.assertIn("news", table_names)
        self.assertIn("filings", table_names)

        self.store.ingest_prices(
            [
                {"symbol": "TSLA", "date": dt.date(2026, 1, 2), "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "adj_close": 102.0, "volume": 10000, "source": "test"},
                {"symbol": "NVDA", "date": dt.date(2026, 1, 2), "open": 500.0, "high": 510.0, "low": 495.0, "close": 505.0, "adj_close": 505.0, "volume": 20000, "source": "test"},
            ]
        )
        self.store.ingest_news(
            [
                {
                    "event_id": "news_1",
                    "symbol": "TSLA",
                    "published_at": dt.datetime(2026, 1, 2, 10, 0, 0),
                    "available_at": dt.datetime(2026, 1, 2, 10, 5, 0),
                    "title": "TSLA sales up",
                    "text": "Tesla delivery numbers exceeded forecasts.",
                    "source": "test",
                    "url": None,
                    "metadata": {"sentiment": "positive"},
                }
            ]
        )
        self.store.ingest_filings(
            [
                {
                    "filing_id": "filing_1",
                    "symbol": "NVDA",
                    "filing_type": "10-Q",
                    "filing_date": dt.date(2026, 1, 2),
                    "accepted_at": dt.datetime(2026, 1, 2, 16, 0, 0),
                    "available_at": dt.datetime(2026, 1, 2, 16, 30, 0),
                    "text": "NVIDIA financial statements...",
                    "source": "test",
                    "url": None,
                    "metadata": {},
                }
            ]
        )

        step = self.store.get_market_step(
            date=dt.date(2026, 1, 2),
            symbols=["TSLA", "NVDA"],
            as_of=dt.datetime(2026, 1, 2, 23, 59, 59),
        )
        self.assertIsInstance(step, MarketStep)
        self.assertEqual(step.prices["TSLA"].close, 102.0)
        self.assertEqual(len(step.news["TSLA"]), 1)
        self.assertEqual(len(step.filings["NVDA"]), 1)

    def test_temporal_leakage_prevention_for_price_news_and_filing(self) -> None:
        t = dt.date(2026, 1, 2)
        as_of = dt.datetime(2026, 1, 2, 12, 0, 0)
        self.store.ingest_prices(
            [
                {"symbol": "TSLA", "date": dt.date(2026, 1, 1), "close": 99.0, "open": 99.0, "high": 99.0, "low": 99.0, "adj_close": 99.0, "volume": 100, "source": "test"},
                {"symbol": "TSLA", "date": t, "close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0, "adj_close": 100.0, "volume": 100, "source": "test"},
                {"symbol": "TSLA", "date": dt.date(2026, 1, 3), "close": 110.0, "open": 110.0, "high": 110.0, "low": 110.0, "adj_close": 110.0, "volume": 100, "source": "test"},
            ]
        )
        self.store.ingest_news(
            [
                {"event_id": "news_past", "symbol": "TSLA", "published_at": dt.datetime(2026, 1, 2, 9, 0, 0), "available_at": dt.datetime(2026, 1, 2, 9, 5, 0), "text": "Safe news", "source": "test", "metadata": {}},
                {"event_id": "news_future", "symbol": "TSLA", "published_at": dt.datetime(2026, 1, 2, 13, 0, 0), "available_at": dt.datetime(2026, 1, 2, 13, 5, 0), "text": "Future news", "source": "test", "metadata": {}},
            ]
        )
        self.store.ingest_filings(
            [
                {"filing_id": "filing_past", "symbol": "TSLA", "filing_type": "10-Q", "filing_date": t, "accepted_at": None, "available_at": dt.datetime(2026, 1, 2, 11, 0, 0), "text": "Safe filing", "source": "test", "metadata": {}},
                {"filing_id": "filing_future", "symbol": "TSLA", "filing_type": "10-Q", "filing_date": t, "accepted_at": None, "available_at": dt.datetime(2026, 1, 2, 13, 0, 0), "text": "Future filing", "source": "test", "metadata": {}},
            ]
        )

        step = self.store.get_market_step(date=t, symbols=["TSLA"], as_of=as_of)
        self.assertEqual(step.prices["TSLA"].close, 100.0)
        self.assertEqual([item.event_id for item in step.news["TSLA"]], ["news_past"])
        self.assertEqual([item.filing_id for item in step.filings["TSLA"]], ["filing_past"])

    def test_symbol_with_quote_does_not_break_query(self) -> None:
        symbol = "BRK'B;DROP"
        self.store.ingest_prices(
            [{"symbol": symbol, "date": dt.date(2026, 1, 2), "close": 300.0}]
        )
        step = self.store.get_market_step(dt.date(2026, 1, 2), [symbol])
        self.assertEqual(step.prices[symbol].close, 300.0)

    def test_empty_symbols_returns_empty_market_step(self) -> None:
        step = self.store.get_market_step(dt.date(2026, 1, 2), [])
        self.assertEqual(step.symbols, [])
        self.assertEqual(step.prices, {})

    def test_multi_asset_environment_stepping(self) -> None:
        dates = [dt.date(2026, 1, 2), dt.date(2026, 1, 3)]
        self.store.ingest_prices(
            [
                {"symbol": "TSLA", "date": dates[0], "close": 100.0},
                {"symbol": "TSLA", "date": dates[1], "close": 101.0},
            ]
        )
        env = MultiAssetEnvironment(self.store, ["TSLA"], dates=dates)
        self.assertEqual(env.simulation_length, 2)
        self.assertEqual(env.step().date, dates[0])
        self.assertEqual(env.step().date, dates[1])
        self.assertTrue(env.is_done)
        with self.assertRaises(IndexError):
            env.step()
        with self.assertRaises(ValueError):
            env.step(dt.date(2026, 1, 4))
        env.reset()
        self.assertFalse(env.is_done)

    def test_generated_id_is_deterministic_sha256(self) -> None:
        first = _stable_id("TSLA", "2026-01-02T10:00:00", "text")
        second = _stable_id("TSLA", "2026-01-02T10:00:00", "text")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_loader_missing_timestamp_raises_value_error(self) -> None:
        from puppy.data_engine.parquet_loader import load_news_parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "news.parquet")
            table = pa.Table.from_pylist(
                [{"symbol": "TSLA", "text": "missing timestamps"}]
            )
            pq.write_table(table, path)
            with self.assertRaises(ValueError):
                load_news_parquet(path)

    def test_news_loader_generates_deterministic_sha256_id(self) -> None:
        from puppy.data_engine.parquet_loader import load_news_parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "news.parquet")
            table = pa.Table.from_pylist(
                [
                    {
                        "symbol": "TSLA",
                        "text": "deterministic news",
                        "published_at": dt.datetime(2026, 1, 2, 10, 0, 0),
                    }
                ]
            )
            pq.write_table(table, path)
            first = load_news_parquet(path)
            second = load_news_parquet(path)
            first_id = first.to_dict("records")[0]["event_id"]
            second_id = second.to_dict("records")[0]["event_id"]
            self.assertEqual(first_id, second_id)
            self.assertEqual(len(first_id), 64)


if __name__ == "__main__":
    unittest.main()
