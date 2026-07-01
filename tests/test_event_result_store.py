import datetime as dt
import os
import tempfile
import unittest

from puppy.common.schemas import (
    AgentOutput,
    AgentSignal,
    GraphOutput,
    MemoryShareRoute,
    PortfolioSnapshot,
    QuantFeatureTable,
    ShadowPortfolioState,
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

    def test_shadow_portfolio_state_typed_helpers_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            event_store = EventStore(path)
            date = dt.date(2026, 1, 2)
            next_date = dt.date(2026, 1, 3)
            event_store.append_shadow_portfolio_state(
                date,
                ShadowPortfolioState(
                    date=date,
                    agent_id="sector_tsla",
                    symbol="TSLA",
                    value=1.04,
                    roi=0.04,
                ),
            )
            event_store.append_shadow_portfolio_state(
                next_date,
                ShadowPortfolioState(
                    date=next_date,
                    agent_id="sector_nvda",
                    symbol="NVDA",
                    value=0.98,
                    roi=-0.02,
                ),
            )

            result_store = ResultStore(event_store)
            self.assertEqual(len(result_store.read_shadow_portfolio_states()), 2)
            self.assertEqual(
                len(result_store.read_shadow_portfolio_states(date=date)),
                1,
            )
            self.assertEqual(
                result_store.read_shadow_portfolio_states(
                    agent_id="sector_tsla"
                )[0].symbol,
                "TSLA",
            )
            self.assertEqual(
                result_store.read_shadow_portfolio_states(symbol="NVDA")[0].agent_id,
                "sector_nvda",
            )

    def test_graph_output_and_memory_share_route_typed_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            event_store = EventStore(path)
            date = dt.date(2026, 1, 2)
            route = MemoryShareRoute(
                date=date,
                source_agent_id="sector_tsla",
                target_agent_id="sector_nvda",
                source_symbol="TSLA",
                target_symbol="NVDA",
                alpha=0.75,
                reason="graph attention",
            )
            event_store.append_graph_output(
                date,
                GraphOutput(
                    date=date,
                    symbols=["TSLA", "NVDA"],
                    graph_scores={"TSLA": -0.5, "NVDA": 0.5},
                ),
            )
            event_store.append_memory_share_route(date, route)

            result_store = ResultStore(event_store)
            self.assertEqual(result_store.read_graph_outputs(date)[0].symbols, ["TSLA", "NVDA"])
            self.assertEqual(
                result_store.read_memory_share_routes(
                    source_agent_id="sector_tsla"
                )[0].target_symbol,
                "NVDA",
            )
            self.assertEqual(
                result_store.read_memory_share_routes(
                    target_agent_id="sector_nvda"
                )[0].source_symbol,
                "TSLA",
            )
            self.assertEqual(
                result_store.read_memory_share_routes(source_symbol="TSLA")[0].alpha,
                0.75,
            )
            self.assertEqual(
                result_store.read_memory_share_routes(target_symbol="NVDA")[0].reason,
                "graph attention",
            )


if __name__ == "__main__":
    unittest.main()
