import datetime as dt
import unittest

from puppy.common.schemas import AllocationWeights
from puppy.portfolio_ext.master_portfolio import MasterPortfolio


class MasterPortfolioAccountingTest(unittest.TestCase):
    def test_master_tracks_equity_return_drawdown_and_win_rate(self) -> None:
        portfolio = MasterPortfolio(initial_cash=100.0)
        allocation = AllocationWeights(
            date=dt.date(2026, 1, 2),
            weights={"TSLA": 1.0},
            cash_weight=0.0,
        )
        first = portfolio.apply_target_weights(
            date=dt.date(2026, 1, 2),
            prices={"TSLA": 100.0},
            target_weights=allocation,
        )
        second = portfolio.apply_target_weights(
            date=dt.date(2026, 1, 3),
            prices={"TSLA": 80.0},
            target_weights=allocation,
        )
        self.assertAlmostEqual(first.equity, 100.0)
        self.assertAlmostEqual(second.equity, 80.0)
        self.assertAlmostEqual(second.daily_return, -0.20)
        self.assertAlmostEqual(second.cumulative_return, -0.20)
        self.assertAlmostEqual(second.drawdown, -0.20)
        self.assertAlmostEqual(second.win_rate, 0.0)

    def test_master_is_one_account_for_all_symbols(self) -> None:
        portfolio = MasterPortfolio(initial_cash=100000.0)
        allocation = AllocationWeights(
            date=dt.date(2026, 1, 2),
            weights={"TSLA": 0.3, "NVDA": 0.2},
            cash_weight=0.5,
        )
        snapshot = portfolio.apply_target_weights(
            date=allocation.date,
            prices={"TSLA": 100.0, "NVDA": 200.0},
            target_weights=allocation,
        )
        self.assertEqual(set(snapshot.positions), {"TSLA", "NVDA"})
        self.assertAlmostEqual(snapshot.weights["TSLA"], 0.3)
        self.assertAlmostEqual(snapshot.weights["NVDA"], 0.2)
        self.assertAlmostEqual(snapshot.cash_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
