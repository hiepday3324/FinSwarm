import datetime as dt
import tempfile
import unittest

import numpy as np

from puppy.common.schemas import GraphOutput, QuantFeatureTable, ShadowPortfolioState, TextFeature
from puppy.graph.gat_model import NumpyGATScorer
from puppy.graph.graph_builder import build_gat_graph_output
from puppy.graph.graph_dataset import build_graph_sample
from puppy.graph.graph_train_loop import train_gat_mvp


class GATMvpTest(unittest.TestCase):
    def test_build_graph_sample_shape(self) -> None:
        sample = build_graph_sample(
            quant_table=_quant_table(),
            shadow_states=_shadow_states(),
            text_features=_text_features(),
            return_history=_return_history(),
        )

        self.assertEqual(sample["symbols"], ["TSLA", "NVDA", "AAPL"])
        self.assertEqual(sample["feature_matrix"].shape, (3, 9))
        self.assertEqual(sample["adjacency_matrix"].shape, (3, 3))
        self.assertTrue(sample["metadata"]["has_labels"])

    def test_gat_forward_is_deterministic_and_bounded(self) -> None:
        sample = build_graph_sample(
            quant_table=_quant_table(),
            shadow_states=_shadow_states(),
            text_features=_text_features(),
            return_history=_return_history(),
        )
        model_a = NumpyGATScorer(input_dim=sample["feature_matrix"].shape[1], seed=9)
        model_b = NumpyGATScorer(input_dim=sample["feature_matrix"].shape[1], seed=9)

        scores_a = model_a.forward(sample["feature_matrix"], sample["adjacency_matrix"])
        scores_b = model_b.forward(sample["feature_matrix"], sample["adjacency_matrix"])

        np.testing.assert_allclose(scores_a, scores_b)
        self.assertTrue(np.all(scores_a >= -1.0))
        self.assertTrue(np.all(scores_a <= 1.0))
        self.assertTrue(np.all(np.diag(model_a.last_attention) == 0.0))

    def test_gat_save_and_load_preserve_scores(self) -> None:
        sample = build_graph_sample(_quant_table(), _shadow_states(), _text_features(), _return_history())
        model = NumpyGATScorer(input_dim=sample["feature_matrix"].shape[1], seed=5)
        expected = model.forward(sample["feature_matrix"], sample["adjacency_matrix"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/gat.json"
            model.save(path)
            loaded = NumpyGATScorer.load(path)
            actual = loaded.forward(sample["feature_matrix"], sample["adjacency_matrix"])
        np.testing.assert_allclose(expected, actual)

    def test_build_gat_graph_output_schema_and_metadata(self) -> None:
        output = build_gat_graph_output(
            date=dt.date(2026, 1, 8),
            quant_table=_quant_table(),
            shadow_states=_shadow_states(),
            text_features=_text_features(),
            return_history=_return_history(),
        )

        self.assertIsInstance(output, GraphOutput)
        self.assertEqual(set(output.graph_scores), {"TSLA", "NVDA", "AAPL"})
        self.assertTrue(output.metadata["uses_gat"])
        self.assertEqual(output.metadata["implementation"], "numpy_gat_mvp")
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in output.graph_scores.values()))

    def test_train_gat_mvp_forward_metrics(self) -> None:
        sample = build_graph_sample(_quant_table(), _shadow_states(), _text_features(), _return_history())
        model, metrics = train_gat_mvp([sample])

        self.assertIsInstance(model, NumpyGATScorer)
        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(metrics["num_symbols"], 3)
        self.assertEqual(metrics["input_dim"], 9)
        self.assertTrue(metrics["has_labels"])
        self.assertGreaterEqual(metrics["score_min"], -1.0)
        self.assertLessEqual(metrics["score_max"], 1.0)


def _quant_table() -> QuantFeatureTable:
    return QuantFeatureTable(
        date=dt.date(2026, 1, 8),
        symbols=["TSLA", "NVDA", "AAPL"],
        action_signals={"TSLA": 1, "NVDA": 1, "AAPL": -1},
        returns={"TSLA": 0.02, "NVDA": 0.01, "AAPL": -0.01},
        labels={"TSLA": 1, "NVDA": 1, "AAPL": 0},
    )


def _shadow_states() -> list[ShadowPortfolioState]:
    return [
        ShadowPortfolioState(date=dt.date(2026, 1, 8), agent_id="sector_tsla", symbol="TSLA", value=1.02, roi=0.02),
        ShadowPortfolioState(date=dt.date(2026, 1, 8), agent_id="sector_nvda", symbol="NVDA", value=1.01, roi=0.01),
        ShadowPortfolioState(date=dt.date(2026, 1, 8), agent_id="sector_aapl", symbol="AAPL", value=0.99, roi=-0.01),
    ]


def _text_features() -> list[TextFeature]:
    return [
        TextFeature(agent_id="sector_tsla", symbol="TSLA", date=dt.date(2026, 1, 8), h_text=[0.4, 0.2], raw_context="tsla", embedding_model="test-gat", dim=2),
        TextFeature(agent_id="sector_nvda", symbol="NVDA", date=dt.date(2026, 1, 8), h_text=[0.2, 0.5], raw_context="nvda", embedding_model="test-gat", dim=2),
        TextFeature(agent_id="sector_aapl", symbol="AAPL", date=dt.date(2026, 1, 8), h_text=[-0.1, 0.2], raw_context="aapl", embedding_model="test-gat", dim=2),
    ]


def _return_history() -> dict[str, list[float]]:
    return {
        "TSLA": [0.01, 0.03, -0.01, 0.02],
        "NVDA": [0.02, 0.04, -0.02, 0.03],
        "AAPL": [-0.01, -0.02, 0.01, -0.01],
    }


if __name__ == "__main__":
    unittest.main()
