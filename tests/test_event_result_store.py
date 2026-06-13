import datetime as dt
import os
import tempfile
import unittest

from puppy.common.schemas import PortfolioSnapshot, QuantFeatureTable
from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore


class EventResultStoreTest(unittest.TestCase):
    def test_event_store_append_and_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            store = EventStore(path)
            date = dt.date(2026, 1, 2)
            table = QuantFeatureTable(
                date=date,
                symbols=["TSLA"],
                action_signals={"TSLA": 1},
                returns={"TSLA": 0.01},
            )

            event = store.append_event("QuantFeatureTable", date, table, {"run": "test"})
            self.assertIn("event_id", event)
            self.assertEqual(len(store.read_events()), 1)
            self.assertEqual(len(store.read_events("QuantFeatureTable", date)), 1)
            self.assertEqual(len(store.read_events("PortfolioSnapshot", date)), 0)

    def test_result_store_typed_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            event_store = EventStore(path)
            date = dt.date(2026, 1, 2)
            event_store.append_event(
                "QuantFeatureTable",
                date,
                QuantFeatureTable(
                    date=date,
                    symbols=["TSLA"],
                    action_signals={"TSLA": 1},
                    returns={"TSLA": 0.01},
                ),
            )
            event_store.append_event(
                "PortfolioSnapshot",
                date,
                PortfolioSnapshot(
                    date=date,
                    cash=60000.0,
                    positions={"TSLA": 400.0},
                    weights={"TSLA": 0.4},
                    equity=100000.0,
                ),
            )

            result_store = ResultStore(event_store)
            self.assertEqual(result_store.read_quant_feature_tables()[0].symbols, ["TSLA"])
            self.assertEqual(result_store.read_portfolio_snapshots()[0].equity, 100000.0)


if __name__ == "__main__":
    unittest.main()
