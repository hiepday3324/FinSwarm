import datetime as dt
import tempfile
import unittest

from puppy.common.schemas import FusionOutput, GraphOutput
from puppy.fusion.gated_fusion import GatedFusion, build_fusion_output
from puppy.fusion.loss import binary_cross_entropy, calibration_metrics, mean_squared_error


class GatedFusionTest(unittest.TestCase):
    def test_build_fusion_output_schema_ranges_and_conflicts(self) -> None:
        date = dt.date(2026, 1, 8)
        output = build_fusion_output(
            date=date,
            text_scores={"TSLA": 0.9, "NVDA": 0.55},
            graph_output=GraphOutput(
                date=date,
                symbols=["TSLA", "NVDA"],
                graph_scores={"TSLA": -0.2, "NVDA": 0.2},
            ),
        )

        self.assertIsInstance(output, FusionOutput)
        self.assertEqual(output.symbols, ["TSLA", "NVDA"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in output.text_scores.values()))
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in output.graph_scores.values()))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in output.fusion_scores.values()))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in output.gates.values()))
        self.assertTrue(output.conflict_flags["TSLA"])
        self.assertFalse(output.conflict_flags["NVDA"])
        self.assertEqual(output.metadata["method"], "gated_fusion_mvp")

    def test_graph_score_conversion_and_save_load(self) -> None:
        date = dt.date(2026, 1, 8)
        graph_output = GraphOutput(
            date=date,
            symbols=["TSLA"],
            graph_scores={"TSLA": 1.0},
        )
        model = GatedFusion(base_gate=0.5, conflict_threshold=0.2)
        output = model.predict(date=date, text_scores={"TSLA": 0.5}, graph_output=graph_output)
        self.assertLess(output.gates["TSLA"], 0.5)
        self.assertAlmostEqual(output.fusion_scores["TSLA"], 0.875)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/fusion.json"
            model.save(path)
            loaded = GatedFusion.load(path)
        loaded_output = loaded.predict(date=date, text_scores={"TSLA": 0.5}, graph_output=graph_output)
        self.assertEqual(output.fusion_scores, loaded_output.fusion_scores)

    def test_invalid_text_score_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_fusion_output(
                date=dt.date(2026, 1, 8),
                text_scores={"TSLA": 1.2},
                graph_output=GraphOutput(
                    date=dt.date(2026, 1, 8),
                    symbols=["TSLA"],
                    graph_scores={"TSLA": 0.0},
                ),
            )

    def test_loss_helpers(self) -> None:
        self.assertGreater(binary_cross_entropy([1, 0], [0.9, 0.2]), 0.0)
        self.assertAlmostEqual(mean_squared_error([1, 0], [1, 0]), 0.0)
        metrics = calibration_metrics([1, 0], [0.9, 0.2])
        self.assertEqual(metrics["num_samples"], 2)
        self.assertIn("bce", metrics)


if __name__ == "__main__":
    unittest.main()
