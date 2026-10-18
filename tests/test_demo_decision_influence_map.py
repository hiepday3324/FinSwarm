from __future__ import annotations

import unittest

import numpy as np

from demo.graph_visualization import (
    EDGE_DECREASE,
    EDGE_INCREASE,
    EDGE_STABLE,
    MODE_CURRENT,
    MODE_DELTA,
    MODE_DOMINANT,
    dominant_attention_path,
    edge_color,
    edge_records,
    edge_visual_attributes,
    make_attention_heatmaps,
    make_network_figure,
    top_influence_rows,
)


SYMBOLS = ("TSLA", "NVDA", "AAPL")


class DecisionInfluenceMapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.current = np.array(
            [
                [0.90, 0.72, 0.20],
                [0.10, 0.95, 0.80],
                [0.42, 0.25, 0.97],
            ],
            dtype=np.float32,
        )
        self.previous = np.array(
            [
                [0.86, 0.60, 0.22],
                [0.14, 0.94, 0.55],
                [0.30, 0.30, 0.97],
            ],
            dtype=np.float32,
        )
        self.records = edge_records(
            self.current,
            self.previous,
            SYMBOLS,
            head_label="Head 2",
            replay_date="2024-11-19",
            previous_date="2024-11-18",
        )

    def test_source_target_direction_and_all_edges_are_preserved(self) -> None:
        self.assertEqual(len(self.records), 9)
        self.assertEqual(sum(item["is_self_loop"] for item in self.records), 3)
        edge = next(
            item
            for item in self.records
            if item["source"] == "AAPL" and item["target"] == "NVDA"
        )
        self.assertAlmostEqual(edge["current_alpha"], self.current[1, 2])
        self.assertAlmostEqual(edge["previous_alpha"], self.previous[1, 2])

    def test_edge_color_uses_signed_delta_threshold(self) -> None:
        self.assertEqual(edge_color(0.021), EDGE_INCREASE)
        self.assertEqual(edge_color(-0.021), EDGE_DECREASE)
        self.assertEqual(edge_color(0.019), EDGE_STABLE)
        self.assertEqual(edge_color(-0.019), EDGE_STABLE)

    def test_edge_width_uses_current_alpha_or_delta_by_mode(self) -> None:
        low = next(
            item
            for item in self.records
            if item["source"] == "TSLA" and item["target"] == "NVDA"
        )
        high = next(
            item
            for item in self.records
            if item["source"] == "AAPL" and item["target"] == "NVDA"
        )
        self.assertGreater(
            edge_visual_attributes(high, mode=MODE_CURRENT)["width"],
            edge_visual_attributes(low, mode=MODE_CURRENT)["width"],
        )
        self.assertGreater(
            edge_visual_attributes(high, mode=MODE_DELTA)["width"],
            edge_visual_attributes(low, mode=MODE_DELTA)["width"],
        )

    def test_self_loop_is_visible_but_reduced(self) -> None:
        loop = next(
            item
            for item in self.records
            if item["source"] == "NVDA" and item["target"] == "NVDA"
        )
        cross = dict(loop, source="AAPL", target="NVDA", is_self_loop=False)
        self.assertLess(
            edge_visual_attributes(loop, mode=MODE_CURRENT)["width"],
            edge_visual_attributes(cross, mode=MODE_CURRENT)["width"],
        )

    def test_sorting_matches_display_modes(self) -> None:
        current_rows = top_influence_rows(self.records, mode=MODE_CURRENT, limit=2)
        delta_rows = top_influence_rows(self.records, mode=MODE_DELTA, limit=2)
        self.assertGreaterEqual(
            current_rows[0]["current_alpha"], current_rows[1]["current_alpha"]
        )
        self.assertGreaterEqual(
            delta_rows[0]["absolute_delta"], delta_rows[1]["absolute_delta"]
        )

    def test_dominant_path_excludes_self_loops_and_uses_current_attention(self) -> None:
        path = dominant_attention_path(self.current, SYMBOLS)
        self.assertEqual(path["symbols"], ["AAPL", "NVDA"])
        self.assertEqual(path["edges"], [("AAPL", "NVDA")])
        self.assertAlmostEqual(path["score"], 0.80, places=6)
        self.assertTrue(all(source != target for source, target in path["edges"]))

    def test_dominant_mode_lists_path_edges_first(self) -> None:
        path = dominant_attention_path(self.current, SYMBOLS)
        rows = top_influence_rows(
            self.records,
            mode=MODE_DOMINANT,
            path_edges=set(path["edges"]),
            limit=2,
        )
        self.assertEqual(
            [row["edge"] for row in rows[: len(path["edges"])]],
            path["edges"],
        )

    def test_figure_exposes_all_edges_and_selected_highlighting(self) -> None:
        figure = make_network_figure(
            self.current,
            self.previous,
            SYMBOLS,
            text_probabilities={"TSLA": 0.41, "NVDA": 0.42, "AAPL": 0.43},
            graph_probabilities={"TSLA": 0.51, "NVDA": 0.52, "AAPL": 0.53},
            fusion_probabilities={"TSLA": 0.49, "NVDA": 0.50, "AAPL": 0.51},
            fusion_deltas={"TSLA": 0.01, "NVDA": -0.02, "AAPL": 0.03},
            actions={"TSLA": "hold", "NVDA": "buy", "AAPL": "sell"},
            selected_symbols={"AAPL"},
            head_label="Head 2",
            replay_date="2024-11-19",
            previous_date="2024-11-18",
            mode=MODE_CURRENT,
        )
        edge_traces = [
            trace for trace in figure.data if getattr(trace, "meta", {}).get("kind") == "edge"
        ]
        self.assertEqual(len(edge_traces), 9)
        self.assertEqual(sum(trace.meta["is_self_loop"] for trace in edge_traces), 3)
        unrelated = next(
            trace
            for trace in edge_traces
            if trace.meta["source"] == "NVDA" and trace.meta["target"] == "TSLA"
        )
        related = next(
            trace
            for trace in edge_traces
            if trace.meta["source"] == "AAPL" and trace.meta["target"] == "NVDA"
        )
        self.assertFalse(unrelated.meta["highlighted"])
        self.assertTrue(related.meta["highlighted"])

    def test_head_label_synchronizes_heatmaps(self) -> None:
        figure = make_attention_heatmaps(
            self.previous,
            self.current,
            SYMBOLS,
            head_label="Head 2",
        )
        self.assertIn("Head 2", figure.layout.title.text)


if __name__ == "__main__":
    unittest.main()
