import datetime as dt
import unittest

from puppy.common.schemas import AllocationWeights, PortfolioSnapshot
from puppy.portfolio_ext.allocation_engine import signals_to_target_weights
from puppy.portfolio_ext.execution_policy import target_weights_to_trades
from puppy.portfolio_ext.master_portfolio import MasterPortfolio
from puppy.portfolio_ext.shadow_portfolio import ShadowPortfolio


class PortfolioExtTest(unittest.TestCase):
    def test_shadow_portfolio_update(self) -> None:
        portfolio = ShadowPortfolio(initial_value=100.0)
        state = portfolio.update(
            agent_id="fake_tsla",
            symbol="TSLA",
            action_signal=1,
            realized_return=0.05,
            date=dt.date(2026, 1, 2),
        )
        self.assertAlmostEqual(state.value, 105.0)
        self.assertAlmostEqual(state.roi, 0.05)

    def test_allocation_weights_sum_with_risk_rules(self) -> None:
        weights = signals_to_target_weights(
            date=dt.date(2026, 1, 2),
            symbols=["TSLA", "NVDA"],
            action_signals={"TSLA": 1, "NVDA": 0},
            max_weight=0.4,
        )
        self.assertLessEqual(sum(weights.weights.values()), 1.0)
        self.assertAlmostEqual(weights.weights["TSLA"], 0.4)
        self.assertAlmostEqual(weights.cash_weight, 0.6)

    def test_execution_policy_and_master_portfolio_snapshot(self) -> None:
        allocation = AllocationWeights(
            date=dt.date(2026, 1, 2),
            weights={"TSLA": 0.4},
            cash_weight=0.6,
        )
        trades = target_weights_to_trades(
            current_equity=100000.0,
            current_positions={},
            prices={"TSLA": 100.0},
            target_weights=allocation,
        )
        self.assertAlmostEqual(trades["TSLA"], 400.0)

        portfolio = MasterPortfolio(initial_cash=100000.0)
        snapshot = portfolio.apply_target_weights(
            date=dt.date(2026, 1, 2),
            prices={"TSLA": 100.0},
            target_weights=allocation,
        )
        self.assertIsInstance(snapshot, PortfolioSnapshot)
        self.assertGreaterEqual(snapshot.cash, 0.0)
        self.assertAlmostEqual(snapshot.positions["TSLA"], 400.0)
        self.assertAlmostEqual(snapshot.equity, 100000.0)


if __name__ == "__main__":
    unittest.main()
