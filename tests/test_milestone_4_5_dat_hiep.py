import os
import tempfile
import unittest
from unittest.mock import patch

from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore
from scripts.run_milestone_4_shadow import run_milestone_4_shadow
from scripts.run_milestone_5_graph import run_milestone_5_graph


class Milestone45DatHiepIntegrationTest(unittest.TestCase):
    def test_shadow_and_graph_route_flow_stays_in_dat_hiep_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            shadow_path = os.path.join(temp_dir, "m4.jsonl")
            graph_path = os.path.join(temp_dir, "m5.jsonl")

            shadow_summary = run_milestone_4_shadow(event_store_path=shadow_path)
            with patch(
                "puppy.memory_ext.memory_sharing.MemorySharingService.apply_memory_share_route",
                side_effect=AssertionError("route application is outside this scope"),
            ):
                graph_summary = run_milestone_5_graph(event_store_path=graph_path)

            shadow_store = EventStore(shadow_path)
            graph_store = EventStore(graph_path)
            shadow_results = ResultStore(shadow_store)
            graph_results = ResultStore(graph_store)

            self.assertEqual(shadow_summary["num_shadow_states"], 9)
            self.assertEqual(
                len(shadow_results.read_shadow_portfolio_states(symbol="TSLA")),
                3,
            )
            self.assertEqual(len(graph_results.read_graph_outputs()), 1)
            self.assertGreater(len(graph_results.read_memory_share_routes()), 0)

            self.assertEqual(shadow_store.read_events("GraphOutput"), [])
            self.assertEqual(shadow_store.read_events("PsychologyState"), [])
            self.assertEqual(graph_store.read_events("FusionOutput"), [])
            self.assertEqual(graph_store.read_events("PsychologyState"), [])

            route = graph_summary["routes"][0]
            self.assertIsNotNone(route.query_text)
            self.assertEqual(route.ttl_days, 1)
            self.assertEqual(route.top_k, 3)
            self.assertNotEqual(route.source_agent_id, route.target_agent_id)
            self.assertNotEqual(route.source_symbol, route.target_symbol)


if __name__ == "__main__":
    unittest.main()
