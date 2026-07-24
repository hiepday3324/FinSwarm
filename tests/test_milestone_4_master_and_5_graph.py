import os
import tempfile
import unittest
from unittest.mock import patch

from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore
from scripts.run_milestone_4_master_portfolio import run_milestone_4_master_portfolio
from scripts.run_milestone_5_graph import run_milestone_5_graph


class Milestone4MasterAnd5GraphIntegrationTest(unittest.TestCase):
    def test_master_accounting_and_graph_route_are_separate_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            portfolio_path = os.path.join(temp_dir, "m4.jsonl")
            graph_path = os.path.join(temp_dir, "m5.jsonl")
            portfolio_summary = run_milestone_4_master_portfolio(portfolio_path)
            with patch(
                "puppy.memory_ext.memory_sharing.MemorySharingService.apply_memory_share_route",
                side_effect=AssertionError("route application is outside this scope"),
            ):
                graph_summary = run_milestone_5_graph(graph_path)

            portfolio_store = EventStore(portfolio_path)
            graph_store = EventStore(graph_path)
            portfolio_results = ResultStore(portfolio_store)
            graph_results = ResultStore(graph_store)
            self.assertEqual(portfolio_summary["num_portfolio_snapshots"], 3)
            self.assertEqual(len(portfolio_results.read_portfolio_snapshots()), 3)
            self.assertEqual(len(graph_results.read_graph_outputs()), 1)
            self.assertGreater(len(graph_results.read_memory_share_routes()), 0)
            self.assertEqual(portfolio_store.read_events("GraphOutput"), [])
            self.assertEqual(graph_store.read_events("FusionOutput"), [])

            route = graph_summary["routes"][0]
            self.assertIsNotNone(route.query_text)
            self.assertEqual(route.ttl_days, 1)
            self.assertNotEqual(route.source_agent_id, route.target_agent_id)


if __name__ == "__main__":
    unittest.main()
