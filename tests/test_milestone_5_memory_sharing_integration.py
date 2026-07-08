import os
import tempfile
import unittest

from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore
from scripts.run_milestone_5_memory_sharing import run_milestone_5_memory_sharing


class Milestone5MemorySharingIntegrationTest(unittest.TestCase):
    def test_graph_routes_inject_shared_context_and_write_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")

            summary = run_milestone_5_memory_sharing(event_store_path=event_path)
            event_store = EventStore(event_path)
            result_store = ResultStore(event_store)

            self.assertGreater(summary["num_routes"], 0)
            self.assertEqual(
                summary["num_memory_share_events"],
                summary["num_routes"],
            )
            self.assertGreater(summary["num_injected_agents"], 0)
            self.assertEqual(
                len(result_store.read_memory_share_events()),
                summary["num_memory_share_events"],
            )

            shared_context_counts = [
                len(context.shared_context) for context in summary["contexts"]
            ]
            self.assertGreater(sum(shared_context_counts), 0)
            raw_context = "\n".join(
                feature.raw_context for feature in summary["text_features"]
            )
            self.assertIn("Shared context", raw_context)
            self.assertIn("Graph influence alpha", raw_context)
            self.assertGreater(len(event_store.read_events("MemoryShareEvent")), 0)
            self.assertGreater(len(event_store.read_events("AgentContext")), 0)
            self.assertGreater(len(event_store.read_events("TextFeature")), 0)


if __name__ == "__main__":
    unittest.main()
