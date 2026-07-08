import datetime as dt
import unittest

from puppy.common.schemas import FusionOutput, PortfolioSnapshot
from puppy.portfolio_ext.allocation_engine import fusion_to_target_weights
from puppy.portfolio_ext.execution_policy import target_weights_to_trades
from puppy.portfolio_ext.master_portfolio import MasterPortfolio


class AllocationM7Test(unittest.TestCase):
    def test_fusion_to_target_weights_applies_thresholds_and_risk_rules(self) -> None:
        allocation = fusion_to_target_weights(
            _fusion_output(),
            max_weight=0.35,
            buy_threshold=0.55,
            conflict_weight_multiplier=0.5,
        )

        self.assertGreater(allocation.weights["TSLA"], 0.0)
        self.assertGreater(allocation.weights["NVDA"], 0.0)
        self.assertEqual(allocation.weights["AAPL"], 0.0)
        self.assertLessEqual(max(allocation.weights.values()), 0.35)
        self.assertAlmostEqual(
            allocation.cash_weight,
            1.0 - sum(allocation.weights.values()),
        )
        self.assertLess(allocation.weights["NVDA"], allocation.weights["TSLA"])
        self.assertEqual(allocation.metadata["method"], "fusion_to_target_weights_mvp")

    def test_master_portfolio_snapshot_from_fusion_allocation(self) -> None:
        allocation = fusion_to_target_weights(_fusion_output(), max_weight=0.35)
        trades = target_weights_to_trades(
            current_equity=100000.0,
            current_positions={},
            prices={"TSLA": 250.0, "NVDA": 500.0, "AAPL": 200.0},
            target_weights=allocation,
        )
        portfolio = MasterPortfolio(initial_cash=100000.0)
        snapshot = portfolio.apply_allocation_weights(
            allocation=allocation,
            prices={"TSLA": 250.0, "NVDA": 500.0, "AAPL": 200.0},
            date=allocation.date,
        )

        self.assertIsInstance(snapshot, PortfolioSnapshot)
        self.assertGreaterEqual(snapshot.cash, 0.0)
        self.assertAlmostEqual(snapshot.equity, 100000.0)
        self.assertEqual(set(trades), {"TSLA", "NVDA", "AAPL"})


def _fusion_output() -> FusionOutput:
    return FusionOutput(
        date=dt.date(2026, 1, 9),
        symbols=["TSLA", "NVDA", "AAPL"],
        fusion_scores={"TSLA": 0.82, "NVDA": 0.60, "AAPL": 0.40},
        conflict_flags={"TSLA": False, "NVDA": True, "AAPL": False},
    )


if __name__ == "__main__":
    unittest.main()
