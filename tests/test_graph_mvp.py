import datetime as dt
import unittest

from puppy.common.schemas import GraphOutput, PortfolioSnapshot, QuantFeatureTable, TextFeature
from puppy.graph.correlation_graph import (
    build_attention_from_adjacency,
    build_correlation_adjacency,
)
from puppy.graph.graph_builder import build_graph_output
from puppy.graph.graph_features import build_node_features
from puppy.graph.risk_router import build_memory_share_routes


class GraphMvpTest(unittest.TestCase):
    def test_build_node_features_has_required_fields(self) -> None:
        date = dt.date(2026, 1, 2)
        features = build_node_features(
            quant_table=_quant_table(date),
            master_portfolio_state=PortfolioSnapshot(
                date=date - dt.timedelta(days=1),
                cash=50000.0,
                cash_weight=0.5,
                positions={"TSLA": 500.0},
                weights={"TSLA": 0.5},
                equity=100000.0,
                daily_return=0.01,
                cumulative_return=0.02,
                drawdown=-0.01,
            ),
            text_features=[
                TextFeature(
                    agent_id="sector_tsla",
                    symbol="TSLA",
                    date=date,
                    h_text=[3.0, 4.0],
                    raw_context="ctx",
                    embedding_model="fake",
                    dim=2,
                )
            ],
        )

        self.assertEqual(features["TSLA"]["action_signal"], 1)
        self.assertEqual(features["TSLA"]["return"], 0.02)
        self.assertEqual(features["TSLA"]["label"], 1)
        self.assertEqual(features["TSLA"]["master_symbol_weight"], 0.5)
        self.assertEqual(features["TSLA"]["master_cash_weight"], 0.5)
        self.assertEqual(features["TSLA"]["master_daily_return"], 0.01)
        self.assertEqual(features["TSLA"]["master_drawdown"], -0.01)
        self.assertEqual(features["TSLA"]["text_norm"], 5.0)
        self.assertEqual(features["TSLA"]["num_text_features"], 1)

    def test_correlation_adjacency_and_attention(self) -> None:
        symbols = ["TSLA", "NVDA", "AAPL"]
        adjacency = build_correlation_adjacency(
            return_history={
                "TSLA": [0.01, 0.02, -0.01, 0.03],
                "NVDA": [0.02, 0.04, -0.02, 0.06],
                "AAPL": [-0.01, -0.02, 0.01, -0.03],
            },
            symbols=symbols,
        )
        attention = build_attention_from_adjacency(adjacency)

        for symbol in symbols:
            self.assertEqual(adjacency[symbol][symbol], 1.0)
        for target in symbols:
            for source in symbols:
                self.assertGreaterEqual(adjacency[target][source], -1.0)
                self.assertLessEqual(adjacency[target][source], 1.0)
            self.assertEqual(attention[target][target], 0.0)
            self.assertAlmostEqual(sum(attention[target].values()), 1.0)

    def test_graph_output_and_routes(self) -> None:
        date = dt.date(2026, 1, 2)
        graph_output = build_graph_output(
            date=date,
            quant_table=_quant_table(date),
            return_history={
                "TSLA": [0.01, 0.02, -0.01, 0.03],
                "NVDA": [0.02, 0.04, -0.02, 0.06],
                "AAPL": [-0.01, -0.02, 0.01, -0.03],
            },
        )
        self.assertIsInstance(graph_output, GraphOutput)
        self.assertEqual(set(graph_output.graph_scores), {"TSLA", "NVDA", "AAPL"})

        routes = build_memory_share_routes(
            graph_output=graph_output,
            agent_by_symbol={
                "TSLA": "sector_tsla",
                "NVDA": "sector_nvda",
                "AAPL": "sector_aapl",
            },
            threshold=0.49,
        )

        self.assertGreater(len(routes), 0)
        self.assertTrue(
            all(route.source_symbol != route.target_symbol for route in routes)
        )
        self.assertEqual(
            [route.alpha for route in routes],
            sorted([route.alpha for route in routes], reverse=True),
        )


def _quant_table(date: dt.date) -> QuantFeatureTable:
    return QuantFeatureTable(
        date=date,
        symbols=["TSLA", "NVDA", "AAPL"],
        action_signals={"TSLA": 1, "NVDA": -1, "AAPL": 0},
        returns={"TSLA": 0.02, "NVDA": -0.01, "AAPL": 0.0},
        labels={"TSLA": 1, "NVDA": -1, "AAPL": 0},
    )


if __name__ == "__main__":
    unittest.main()
