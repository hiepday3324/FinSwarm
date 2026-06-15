import os
import tempfile
import unittest

from puppy.data_engine.event_store import EventStore
from scripts.run_milestone_3_text_score import run_milestone_3_text_score


class Milestone3TextScoreTest(unittest.TestCase):
    def test_text_score_integration_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")
            summary = run_milestone_3_text_score(event_store_path=event_path)
            store = EventStore(event_path)

            self.assertEqual(summary["num_text_features"], 9)
            self.assertEqual(summary["num_quant_tables"], 3)
            self.assertEqual(summary["num_samples"], 9)
            self.assertEqual(summary["input_dim"], 3)
            self.assertGreaterEqual(summary["accuracy"], 0.75)
            self.assertEqual(len(summary["text_scores"]), 9)
            self.assertGreater(len(store.read_events("TextFeature")), 0)
            self.assertGreater(len(store.read_events("QuantFeatureTable")), 0)
            self.assertEqual(store.read_events("FusionOutput"), [])
            self.assertEqual(store.read_events("GraphOutput"), [])


if __name__ == "__main__":
    unittest.main()
