import os
import tempfile
import unittest

from puppy.common.schemas import MarketStep, PortfolioSnapshot, QuantFeatureTable
from puppy.data_engine.event_store import EventStore
from scripts.run_milestone_1_mock import run_milestone_1_mock


class Milestone1VerticalSliceTest(unittest.TestCase):
    def test_vertical_slice_runs_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")
            summary = run_milestone_1_mock(event_store_path=event_path)
            store = EventStore(event_path)

            self.assertEqual(summary["dates_processed"], 2)
            self.assertEqual(summary["symbols"], ["TSLA", "NVDA"])
            self.assertIsInstance(summary["last_market_step"], MarketStep)
            self.assertIsInstance(summary["last_quant_table"], QuantFeatureTable)
            self.assertIsInstance(summary["last_portfolio_snapshot"], PortfolioSnapshot)
            self.assertGreater(summary["events_written"], 0)
            self.assertGreater(len(store.read_events("MarketStep")), 0)
            self.assertGreater(len(store.read_events("QuantFeatureTable")), 0)
            self.assertGreater(len(store.read_events("PortfolioSnapshot")), 0)


if __name__ == "__main__":
    unittest.main()
