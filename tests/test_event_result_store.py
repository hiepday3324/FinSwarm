import datetime as dt
import os
import tempfile
import unittest

from puppy.common.schemas import (
    AgentOutput,
    AgentSignal,
    PortfolioSnapshot,
    QuantFeatureTable,
    TextFeature,
)
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

    def test_agent_output_and_text_feature_typed_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            event_store = EventStore(path)
            date = dt.date(2026, 1, 2)
            signal = AgentSignal(
                agent_id="fake_tsla",
                symbol="TSLA",
                date=date,
                decision="buy",
                action_signal=1,
                reason="synthetic",
            )
            output = AgentOutput(signal=signal, model_name="fake")
            feature = TextFeature(
                agent_id="fake_tsla",
                symbol="TSLA",
                date=date,
                h_text=[0.1, 0.2, 0.3],
                raw_context="synthetic context",
                embedding_model="fake",
                dim=3,
            )

            event_store.append_agent_output(date, output)
            event_store.append_text_feature(date, feature)

            result_store = ResultStore(event_store)
            outputs = result_store.read_agent_outputs(date)
            features = result_store.read_text_features(date)
            self.assertEqual(outputs[0].signal.symbol, "TSLA")
            self.assertEqual(outputs[0].signal.action_signal, 1)
            self.assertEqual(features[0].h_text, [0.1, 0.2, 0.3])
            self.assertEqual(features[0].dim, 3)


if __name__ == "__main__":
    unittest.main()
