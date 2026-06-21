import os
import tempfile
import unittest

from puppy.data_engine.event_store import EventStore
from scripts.run_milestone_3_shadow import run_milestone_3_shadow


class Milestone4ShadowPsychologyFeedbackTest(unittest.TestCase):
    def test_agent_output_to_shadow_psychology_context_feedback_runs_end_to_end(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")

            summary = run_milestone_3_shadow(event_store_path=event_path)
            store = EventStore(event_path)

            self.assertEqual(len(summary["day_t"]["agent_outputs"]), 3)
            self.assertEqual(len(summary["day_t"]["text_features"]), 3)
            self.assertEqual(len(summary["shadow_states"]), 3)
            self.assertEqual(len(summary["psychology_states"]), 3)
            self.assertEqual(len(summary["day_t2"]["contexts"]), 3)
            self.assertEqual(len(summary["day_t2"]["text_features"]), 3)

            tsla_psychology = summary["psychology_states"]["sector_tsla"]
            self.assertEqual(tsla_psychology["state"], "greed")
            self.assertEqual(tsla_psychology["source"], "shadow_portfolio")

            context_text = "\n".join(summary["feedback_context_raw"])
            self.assertIn("Shadow state", context_text)
            self.assertIn("Psychology state", context_text)
            self.assertIn("greed", context_text)

            self.assertEqual(len(store.read_events("AgentOutput")), 6)
            self.assertEqual(len(store.read_events("TextFeature")), 6)
            self.assertEqual(len(store.read_events("AgentContext")), 6)
            self.assertEqual(len(store.read_events("ShadowPortfolioState")), 3)
            self.assertEqual(len(store.read_events("PsychologyState")), 3)


if __name__ == "__main__":
    unittest.main()
