import unittest

from demo.app import _replay_master_portfolio
from demo.data_loader import ReplayStore
from demo.inference import FrozenReplayEngine


class DemoMasterPortfolioRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = ReplayStore()
        cls.engine = FrozenReplayEngine()

    def test_replay_produces_one_valid_master_account_history(self) -> None:
        snapshots, allocations = _replay_master_portfolio(
            store=self.store,
            engine=self.engine,
            end_index=10,
            capital=100000.0,
        )

        self.assertEqual(len(snapshots), 11)
        self.assertEqual(len(allocations), 11)
        self.assertTrue(all(snapshot.equity > 0.0 for snapshot in snapshots))
        self.assertTrue(
            all(
                abs(
                    sum(snapshot.weights.values())
                    + float(snapshot.cash_weight or 0.0)
                    - 1.0
                )
                < 1e-9
                for snapshot in snapshots
            )
        )
        self.assertTrue(
            all(
                snapshots[index].date < snapshots[index + 1].date
                for index in range(len(snapshots) - 1)
            )
        )
        self.assertTrue(
            all(
                snapshot.metadata["allocation_source"]
                == "demo.allocator.allocate"
                for snapshot in snapshots
            )
        )

    def test_replay_context_uses_prior_master_snapshot(self) -> None:
        for index, date in enumerate(self.store.dates[:10]):
            for record in self.store.records_for_date(date):
                state = record["master_portfolio_state"]
                if index == 0:
                    self.assertIsNone(state)
                else:
                    self.assertLess(str(state["date"]), date.isoformat())


if __name__ == "__main__":
    unittest.main()
