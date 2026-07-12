from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.normalization.config import sha256_file
from puppy.data_engine.normalization.filings import normalize_filing_record
from puppy.data_engine.normalization.news import normalize_news_rows
from puppy.data_engine.normalization.prices import normalize_price_rows
from puppy.data_engine.normalization.text_extraction import (
    extract_filing_documents,
    html_to_text,
)
from puppy.data_engine.parquet_loader import (
    load_filings_parquet,
    load_news_parquet,
)
from puppy.quant.return_utils import (
    compute_legacy_close_to_close_return,
    compute_next_open_to_close_return,
)


UTC = dt.timezone.utc


class Stage4NormalizationTest(unittest.TestCase):
    def test_price_mapping_provenance_and_validation(self) -> None:
        raw = {
            "ticker": "TSLA",
            "market_date": "2026-01-02",
            "o": 100.0,
            "h": 105.0,
            "l": 98.0,
            "c": 102.0,
            "v": 10,
            "vw": 101.0,
            "n": 7,
            "t": 1767330000000,
            "otc": None,
            "timestamp_utc": "2026-01-02T05:00:00Z",
            "timestamp_market": "2026-01-02T00:00:00-05:00",
        }
        rows = normalize_price_rows([raw], "prices.jsonl", "abc", adjusted=True)
        row = rows[0]
        self.assertEqual(row["provider_record_id"], "massive:TSLA:1767330000000")
        self.assertIsNone(row["adj_close"])
        metadata = json.loads(row["metadata"])
        self.assertTrue(metadata["adjusted"])
        self.assertEqual(metadata["adjustment_type"], "split_adjusted")
        self.assertEqual(metadata["raw_line"], 1)

        with self.assertRaisesRegex(ValueError, "duplicate canonical price key"):
            normalize_price_rows([raw, raw], "prices.jsonl", "abc", adjusted=True)
        with self.assertRaisesRegex(ValueError, "invalid OHLC"):
            normalize_price_rows([{**raw, "o": 106}], "prices.jsonl", "abc", adjusted=True)
        with self.assertRaisesRegex(ValueError, "negative volume"):
            normalize_price_rows([{**raw, "v": -1}], "prices.jsonl", "abc", adjusted=True)

    def test_news_article_symbol_identity_dedup_and_metadata(self) -> None:
        base = {
            "id": "article-1",
            "title": "Title",
            "description": "Body",
            "published_utc": "2026-01-02T15:00:00Z",
            "publisher": {"name": "Wire"},
            "article_url": "https://example.test/article",
            "insights": [{"ticker": "TSLA", "sentiment": "positive"}],
        }
        rows, audit = normalize_news_rows(
            [({**base, "queried_symbol": "TSLA"}, "tsla.jsonl", 1, "a"),
             ({**base, "queried_symbol": "TSLA"}, "tsla.jsonl", 2, "a"),
             ({**base, "queried_symbol": "NVDA"}, "nvda.jsonl", 1, "b")]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(audit), 1)
        self.assertEqual({row["event_id"] for row in rows}, {
            "massive:article-1:TSLA", "massive:article-1:NVDA"
        })
        tsla = next(row for row in rows if row["symbol"] == "TSLA")
        self.assertEqual(tsla["published_at"].tzinfo, UTC)
        self.assertEqual(json.loads(tsla["metadata"])["insights"], base["insights"])

    def test_filing_normalization_requires_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = root / "primary.htm"
            primary.write_text("<html><body><h1>Results</h1><p>Visible prose.</p></body></html>", encoding="utf-8")
            raw = {
                "symbol": "AAPL",
                "accession_number": "0001-26-000001",
                "form": "10-Q",
                "filing_date": "2026-01-02",
                "acceptance_datetime": "2026-01-02T21:05:00Z",
                "primary_document": primary.name,
                "primary_document_local_path": str(primary),
                "primary_document_exists": True,
                "index_url": "https://www.sec.gov/test",
                "files": [{"name": primary.name, "local_path": str(primary), "type": "10-Q"}],
            }
            row, audit = normalize_filing_record(raw, "metadata.jsonl", 1, "hash")
            self.assertEqual(row["filing_id"], "sec:0001-26-000001")
            self.assertEqual(row["accepted_at"], row["available_at"])
            self.assertIn("Visible prose.", row["text"])
            self.assertTrue(audit[0]["included"])
            with self.assertRaisesRegex(ValueError, "acceptance_datetime"):
                normalize_filing_record({**raw, "acceptance_datetime": None}, "metadata.jsonl", 1, "hash")

    def test_filing_extraction_selects_primary_and_material_exhibit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {
                "primary.htm": "<h1>8-K</h1><p>Primary prose.</p><ix:hidden>duplicate fact</ix:hidden>",
                "earnings.htm": "<h2>Exhibit</h2><p>Material exhibit prose.</p>",
                "schema.xsd": "support",
                "instance.xml": "<xbrl>support</xbrl>",
                "style.css": "body{}",
                "script.js": "alert(1)",
                "archive.zip": "binary",
                "logo.png": "binary",
            }
            for name, content in files.items():
                (root / name).write_text(content, encoding="utf-8")
            entries = [
                {"name": name, "local_path": str(root / name), "type": "EX-99.1" if name == "earnings.htm" else "8-K" if name == "primary.htm" else ""}
                for name in files
            ]
            text, audit, included, excluded = extract_filing_documents(
                "AAPL", "0001", "8-K", "primary.htm", entries
            )
            self.assertIn("Primary prose.", text)
            self.assertIn("Material exhibit prose.", text)
            self.assertNotIn("duplicate fact", text)
            self.assertEqual(len(included), 2)
            self.assertEqual(len(excluded), 6)
            self.assertEqual(len(audit), 8)

    def test_html_to_text_is_deterministic_and_removes_hidden_content(self) -> None:
        source = "<html><style>x</style><script>y</script><ix:hidden><div style='display:none'>hidden</div></ix:hidden><h1>Heading</h1><p>Visible   text</p></html>"
        first = html_to_text(source)
        self.assertEqual(first, html_to_text(source))
        self.assertIn("Heading", first)
        self.assertIn("Visible text", first)
        self.assertNotIn("hidden", first)


class Stage4StorageTest(unittest.TestCase):
    def test_strict_parquet_round_trip_and_naive_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            news_path = os.path.join(temp_dir, "news.parquet")
            filing_path = os.path.join(temp_dir, "filings.parquet")
            pq.write_table(pa.Table.from_pylist([{
                "event_id": "n1", "symbol": "TSLA",
                "published_at": dt.datetime(2026, 1, 2, 10, tzinfo=UTC),
                "available_at": dt.datetime(2026, 1, 2, 10, tzinfo=UTC),
                "title": None, "text": "news", "source": "massive", "url": None,
                "provider_record_id": "raw1", "metadata": json.dumps({"x": 1}),
            }]), news_path)
            pq.write_table(pa.Table.from_pylist([{
                "filing_id": "f1", "symbol": "TSLA", "filing_type": "10-Q",
                "filing_date": dt.date(2026, 1, 2),
                "accepted_at": dt.datetime(2026, 1, 2, 21, tzinfo=UTC),
                "available_at": dt.datetime(2026, 1, 2, 21, tzinfo=UTC),
                "text": "filing", "source": "sec_edgar", "url": None,
                "accession_number": "0001", "provider_record_id": "0001",
                "metadata": json.dumps({"x": 1}),
            }]), filing_path)
            news = load_news_parquet(news_path)[0]
            filing = load_filings_parquet(filing_path)[0]
            self.assertEqual(news["provider_record_id"], "raw1")
            self.assertEqual(news["metadata"], {"x": 1})
            self.assertEqual(filing["accession_number"], "0001")
            self.assertEqual(filing["accepted_at"].tzinfo, UTC)

            pq.write_table(pa.Table.from_pylist([{
                "event_id": "bad", "symbol": "TSLA",
                "published_at": dt.datetime(2026, 1, 2, 10),
                "available_at": dt.datetime(2026, 1, 2, 10), "text": "bad",
                "title": None, "source": "massive", "url": None,
                "provider_record_id": "bad", "metadata": "{}",
            }]), news_path)
            with self.assertRaisesRegex(ValueError, "timezone-aware"):
                load_news_parquet(news_path)

    def test_point_in_time_windows_exact_prices_dst_and_legacy_warning(self) -> None:
        store = DuckDBMarketStore(":memory:")
        store.connect()
        dates = [dt.date(2026, 3, 6), dt.date(2026, 3, 9)]
        store.ingest_prices([{"symbol": "TSLA", "date": dates[0], "close": 100.0}])
        ny = ZoneInfo("America/New_York")
        previous = dt.datetime.combine(dates[0], dt.time(16, 10), ny).astimezone(UTC)
        current = dt.datetime.combine(dates[1], dt.time(16, 10), ny).astimezone(UTC)
        store.ingest_news([
            {"event_id": "at_previous", "symbol": "TSLA", "published_at": previous, "available_at": previous, "text": "old"},
            {"event_id": "after_hours", "symbol": "TSLA", "published_at": previous + dt.timedelta(hours=1), "available_at": previous + dt.timedelta(hours=1), "text": "after"},
            {"event_id": "weekend", "symbol": "TSLA", "published_at": previous + dt.timedelta(days=1), "available_at": previous + dt.timedelta(days=1), "text": "weekend"},
            {"event_id": "at_current", "symbol": "TSLA", "published_at": current, "available_at": current, "text": "edge"},
            {"event_id": "future", "symbol": "TSLA", "published_at": current + dt.timedelta(seconds=1), "available_at": current + dt.timedelta(seconds=1), "text": "future"},
        ])
        step = store.get_point_in_time_market_step(dates[1], ["TSLA"], dates)
        ids = [event.event_id for event in step.news["TSLA"]]
        self.assertEqual(ids, ["after_hours", "weekend", "at_current"])
        self.assertNotIn("TSLA", step.prices)
        self.assertEqual(step.metadata["missing_symbols"], ["TSLA"])
        self.assertEqual(step.metadata["stale_symbols"], ["TSLA"])
        self.assertEqual(step.as_of, current)
        self.assertEqual(previous.hour, 21)
        self.assertEqual(current.hour, 20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = store.get_legacy_market_step(dates[1], ["TSLA"], as_of=current)
        self.assertEqual(legacy.prices["TSLA"].close, 100.0)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        store.close()

    def test_first_step_left_censored_and_duplicate_ingestion_fails(self) -> None:
        store = DuckDBMarketStore(":memory:")
        store.connect()
        first = dt.date(2026, 1, 2)
        store.ingest_prices([{"symbol": "TSLA", "date": first, "close": 100.0}])
        step = store.get_point_in_time_market_step(first, ["TSLA"], [first])
        self.assertTrue(step.metadata["left_censored"])
        with self.assertRaisesRegex(ValueError, "duplicate primary key"):
            store.ingest_prices([
                {"symbol": "NVDA", "date": first, "close": 1.0},
                {"symbol": "NVDA", "date": first, "close": 2.0},
            ])
        types = {row[0]: row[1] for row in store.conn.execute("DESCRIBE news").fetchall()}
        self.assertEqual(types["published_at"], "TIMESTAMP WITH TIME ZONE")
        self.assertEqual(types["available_at"], "TIMESTAMP WITH TIME ZONE")
        store.close()

    def test_target_protocol_and_legacy_function(self) -> None:
        self.assertAlmostEqual(compute_next_open_to_close_return(100.0, 105.0), 0.05)
        self.assertAlmostEqual(compute_legacy_close_to_close_return(100.0, 110.0), 0.10)
        self.assertIsNone(compute_next_open_to_close_return(0.0, 105.0))

    def test_sha256_file_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "x"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
