import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore
from scripts.run_milestone_6_7_dat_hiep import run_milestone_6_7_dat_hiep


class Milestone67DatHiepIntegrationTest(unittest.TestCase):
    def test_text_graph_fusion_allocation_portfolio_flow_stays_in_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")
            with patch(
                "puppy.memory_ext.memory_sharing.MemorySharingService.apply_memory_share_route",
                side_effect=AssertionError("memory sharing is outside M6/M7 scope"),
            ):
                summary = run_milestone_6_7_dat_hiep(event_store_path=event_path)

            event_store = EventStore(event_path)
            result_store = ResultStore(event_store)

            self.assertEqual(summary["num_fusion_outputs"], 1)
            self.assertEqual(summary["num_allocation_weights"], 1)
            self.assertEqual(summary["num_portfolio_snapshots"], 1)
            self.assertEqual(len(result_store.read_fusion_outputs()), 1)
            self.assertEqual(len(result_store.read_allocation_weights()), 1)
            self.assertEqual(len(result_store.read_portfolio_snapshots()), 1)

            self.assertEqual(event_store.read_events("DebateVerdict"), [])
            self.assertEqual(event_store.read_events("MemoryShareEvent"), [])
            self.assertEqual(event_store.read_events("APIResult"), [])
            self.assertEqual(event_store.read_events("EvaluationReport"), [])

            script_text = Path("scripts/run_milestone_6_7_dat_hiep.py").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("openai", script_text.lower())


if __name__ == "__main__":
    unittest.main()
