import unittest
import datetime as dt
import pandas as pd

from puppy.common.schemas import MarketStep
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.multi_asset_env import MultiAssetEnvironment


class DataEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        # Use an in-memory DuckDB for isolated unit tests
        self.store = DuckDBMarketStore(db_path=":memory:")
        self.store.connect()

    def tearDown(self) -> None:
        self.store.close()

    def test_schema_init_and_ingestion(self) -> None:
        # 1. Verify table creations
        tables = self.store.conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("prices", table_names)
        self.assertIn("news", table_names)
        self.assertIn("filings", table_names)

        # 2. Ingest sample prices
        df_prices = pd.DataFrame([
            {"symbol": "TSLA", "date": dt.date(2026, 1, 2), "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "adj_close": 102.0, "volume": 10000, "source": "test"},
            {"symbol": "NVDA", "date": dt.date(2026, 1, 2), "open": 500.0, "high": 510.0, "low": 495.0, "close": 505.0, "adj_close": 505.0, "volume": 20000, "source": "test"},
        ])
        self.store.ingest_prices(df_prices)

        # 3. Ingest sample news
        df_news = pd.DataFrame([
            {
                "event_id": "news_1",
                "symbol": "TSLA",
                "published_at": dt.datetime(2026, 1, 2, 10, 0, 0),
                "available_at": dt.datetime(2026, 1, 2, 10, 5, 0),
                "title": "TSLA sales up",
                "text": "Tesla delivery numbers exceeded forecasts.",
                "source": "test",
                "url": None,
                "metadata": {"sentiment": "positive"}
            }
        ])
        self.store.ingest_news(df_news)

        # 4. Ingest sample filings
        df_filings = pd.DataFrame([
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
                "metadata": {}
            }
        ])
        self.store.ingest_filings(df_filings)

        # Verify querying MarketStep at t
        step = self.store.get_market_step(
            date=dt.date(2026, 1, 2),
            symbols=["TSLA", "NVDA"],
            as_of=dt.datetime(2026, 1, 2, 23, 59, 59)
        )
        self.assertIsInstance(step, MarketStep)
        self.assertEqual(step.date, dt.date(2026, 1, 2))
        self.assertIn("TSLA", step.prices)
        self.assertEqual(step.prices["TSLA"].close, 102.0)
        self.assertEqual(len(step.news["TSLA"]), 1)
        self.assertEqual(step.news["TSLA"][0].title, "TSLA sales up")
        self.assertEqual(len(step.filings["NVDA"]), 1)

    def test_temporal_leakage_prevention(self) -> None:
        t = dt.date(2026, 1, 2)
        as_of = dt.datetime(2026, 1, 2, 12, 0, 0) # cutoff at noon

        # Ingest past price
        df_prices = pd.DataFrame([
            {"symbol": "TSLA", "date": dt.date(2026, 1, 1), "close": 99.0, "open": 99.0, "high": 99.0, "low": 99.0, "adj_close": 99.0, "volume": 100, "source": "test"},
            {"symbol": "TSLA", "date": t, "close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0, "adj_close": 100.0, "volume": 100, "source": "test"},
            # FUTURE PRICE (Leakage threat)
            {"symbol": "TSLA", "date": dt.date(2026, 1, 3), "close": 110.0, "open": 110.0, "high": 110.0, "low": 110.0, "adj_close": 110.0, "volume": 100, "source": "test"}
        ])
        self.store.ingest_prices(df_prices)

        # Ingest past and future news
        df_news = pd.DataFrame([
            # Available BEFORE cutoff (safe)
            {
                "event_id": "news_past",
                "symbol": "TSLA",
                "published_at": dt.datetime(2026, 1, 2, 9, 0, 0),
                "available_at": dt.datetime(2026, 1, 2, 9, 5, 0),
                "text": "Safe news",
                "source": "test",
                "metadata": {}
            },
            # Available AFTER cutoff (Leakage threat: published at 13:00)
            {
                "event_id": "news_future",
                "symbol": "TSLA",
                "published_at": dt.datetime(2026, 1, 2, 13, 0, 0),
                "available_at": dt.datetime(2026, 1, 2, 13, 5, 0),
                "text": "Future news!",
                "source": "test",
                "metadata": {}
            }
        ])
        self.store.ingest_news(df_news)

        step = self.store.get_market_step(date=t, symbols=["TSLA"], as_of=as_of)

        # Price assert: must be price of day t (100.0), NOT day t+1 (110.0)
        self.assertEqual(step.prices["TSLA"].close, 100.0)
        self.assertEqual(step.prices["TSLA"].date, t)

        # News assert: news_future must be excluded
        tsla_news = step.news["TSLA"]
        self.assertEqual(len(tsla_news), 1)
        self.assertEqual(tsla_news[0].event_id, "news_past")

    def test_multi_asset_environment_stepping(self) -> None:
        # Setup store and ingest sorted dates
        df_prices = pd.DataFrame([
            {"symbol": "TSLA", "date": dt.date(2026, 1, 2), "close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0, "adj_close": 100.0, "volume": 100, "source": "test"},
            {"symbol": "TSLA", "date": dt.date(2026, 1, 3), "close": 101.0, "open": 101.0, "high": 101.0, "low": 101.0, "adj_close": 101.0, "volume": 100, "source": "test"},
        ])
        self.store.ingest_prices(df_prices)

        env = MultiAssetEnvironment(
            market_store=self.store,
            symbols=["TSLA"],
            dates=[dt.date(2026, 1, 2), dt.date(2026, 1, 3)]
        )

        self.assertEqual(env.simulation_length, 2)
        self.assertFalse(env.is_done)

        # Step 1
        step1 = env.step()
        self.assertEqual(step1.date, dt.date(2026, 1, 2))
        self.assertEqual(step1.prices["TSLA"].close, 100.0)
        self.assertFalse(env.is_done)

        # Step 2
        step2 = env.step()
        self.assertEqual(step2.date, dt.date(2026, 1, 3))
        self.assertEqual(step2.prices["TSLA"].close, 101.0)
        self.assertTrue(env.is_done)

        # Reset
        env.reset()
        self.assertFalse(env.is_done)
        self.assertEqual(env.current_index, 0)


if __name__ == "__main__":
    unittest.main()
