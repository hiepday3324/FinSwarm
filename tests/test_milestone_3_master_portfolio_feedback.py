import os
import tempfile
import unittest

from puppy.data_engine.event_store import EventStore
from scripts.run_milestone_3_master_portfolio_feedback import (
    run_milestone_3_master_portfolio_feedback,
)


class Milestone3MasterPortfolioFeedbackTest(unittest.TestCase):
    def test_master_snapshot_reaches_next_agent_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")
            summary = run_milestone_3_master_portfolio_feedback(event_path)
            store = EventStore(event_path)

            self.assertEqual(len(summary["day_t"]["agent_outputs"]), 3)
            self.assertEqual(len(summary["day_t2"]["contexts"]), 3)
            self.assertEqual(len(summary["day_t2"]["text_features"]), 3)
            self.assertGreater(summary["feedback_snapshot"].equity, 0.0)
            for context in summary["day_t2"]["contexts"]:
                self.assertEqual(
                    context.master_portfolio_state["date"],
                    summary["feedback_date"].isoformat(),
                )
                self.assertEqual(
                    context.psychology_state["source"],
                    "master_portfolio",
                )

            context_text = "\n".join(summary["feedback_context_raw"])
            self.assertIn("Master portfolio state", context_text)
            self.assertIn("Psychology state", context_text)
            self.assertEqual(len(store.read_events("AgentOutput")), 6)
            self.assertEqual(len(store.read_events("PortfolioSnapshot")), 2)
            self.assertEqual(len(store.read_events("PsychologyState")), 3)


if __name__ == "__main__":
    unittest.main()
