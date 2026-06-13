import datetime as dt
import unittest

from puppy.common.schemas import AgentOutput, AgentSignal, MarketStep, PriceBar
from puppy.quant.return_utils import (
    build_direction_label,
    build_quant_feature_table,
    compute_log_return,
    compute_simple_return,
)
from puppy.quant.risk_rules import apply_basic_risk_rules
from puppy.quant.shadow_metrics import (
    compute_drawdown,
    compute_roi,
    compute_rolling_return,
    compute_sharpe,
    compute_win_rate,
)
from puppy.quant.signal_utils import decision_to_action_signal


class QuantLayerTest(unittest.TestCase):
    def test_decision_to_action_signal(self) -> None:
        self.assertEqual(decision_to_action_signal("buy"), 1)
        self.assertEqual(decision_to_action_signal("hold"), 0)
        self.assertEqual(decision_to_action_signal("sell"), -1)
        self.assertEqual(decision_to_action_signal("unknown"), 0)

    def test_returns_labels_and_feature_table(self) -> None:
        date = dt.date(2026, 1, 2)
        next_date = dt.date(2026, 1, 3)
        current = MarketStep(
            date=date,
            symbols=["TSLA"],
            prices={"TSLA": PriceBar(symbol="TSLA", date=date, close=100.0)},
            as_of=dt.datetime(2026, 1, 2, 23, 59, 59),
        )
        nxt = MarketStep(
            date=next_date,
            symbols=["TSLA"],
            prices={"TSLA": PriceBar(symbol="TSLA", date=next_date, close=105.0)},
            as_of=dt.datetime(2026, 1, 3, 23, 59, 59),
        )
        output = AgentOutput(
            signal=AgentSignal(
                agent_id="fake",
                symbol="TSLA",
                date=date,
                decision="buy",
                action_signal=1,
                reason="test",
            )
        )

        self.assertAlmostEqual(compute_simple_return(100, 105), 0.05)
        self.assertAlmostEqual(compute_log_return(100, 105), 0.04879016416943205)
        self.assertEqual(build_direction_label(0.05), 1)
        table = build_quant_feature_table(date, ["TSLA"], [output], current, nxt)
        self.assertEqual(table.action_signals["TSLA"], 1)
        self.assertAlmostEqual(table.returns["TSLA"], 0.05)
        self.assertEqual(table.labels["TSLA"], 1)

    def test_shadow_metrics_and_risk_rules(self) -> None:
        self.assertAlmostEqual(compute_roi(110.0, 100.0), 0.1)
        self.assertLess(compute_drawdown([100.0, 90.0, 95.0]), 0.0)
        self.assertAlmostEqual(compute_rolling_return([0.1, -0.1]), -0.01)
        self.assertGreaterEqual(compute_sharpe([0.01, 0.02, -0.01]), 0.0)
        self.assertAlmostEqual(compute_win_rate([0.01, -0.02, 0.0]), 1 / 3)

        weights, cash = apply_basic_risk_rules(
            {"TSLA": 2.0, "NVDA": -1.0, "BAD": float("nan")},
            max_weight=0.4,
        )
        self.assertLessEqual(sum(weights.values()), 1.0)
        self.assertEqual(weights["TSLA"], 0.4)
        self.assertEqual(weights["NVDA"], 0.0)
        self.assertAlmostEqual(cash, 0.6)


if __name__ == "__main__":
    unittest.main()
