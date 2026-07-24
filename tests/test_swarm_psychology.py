import datetime as dt
import unittest
from types import SimpleNamespace

from puppy.common.schemas import PortfolioSnapshot
from puppy.swarm.context_builder import build_agent_context, build_raw_context
from puppy.swarm.psychology import build_psychology_state, update_psychology_state


class SwarmPsychologyTest(unittest.TestCase):
    def test_master_performance_detects_greed(self) -> None:
        psychology = build_psychology_state(
            _snapshot(cumulative_return=0.08, drawdown=-0.01),
            symbol="NVDA",
        )
        self.assertEqual(psychology["state"], "greed")
        self.assertEqual(psychology["risk_attitude"], "risk_seeking")
        self.assertEqual(psychology["metrics"]["cumulative_return"], 0.08)
        self.assertEqual(psychology["source"], "master_portfolio")

    def test_master_drawdown_detects_fear(self) -> None:
        psychology = update_psychology_state(
            _snapshot(cumulative_return=-0.10, daily_return=-0.04, drawdown=-0.12),
            symbol="TSLA",
        )
        self.assertEqual(psychology["state"], "fear")
        self.assertEqual(psychology["risk_attitude"], "risk_averse")
        self.assertIn("drawdown", psychology["reason"])

    def test_missing_master_snapshot_is_neutral(self) -> None:
        psychology = build_psychology_state(None, symbol="AAPL")
        self.assertEqual(psychology["state"], "neutral")
        self.assertEqual(psychology["metrics"], {})

    def test_flat_master_snapshot_is_neutral(self) -> None:
        psychology = build_psychology_state(
            _snapshot(cumulative_return=0.0, daily_return=0.0, drawdown=0.0),
            symbol="NVDA",
        )
        self.assertEqual(psychology["state"], "neutral")

    def test_context_includes_master_portfolio_and_psychology(self) -> None:
        cur_date = dt.date(2026, 1, 4)
        snapshot = _snapshot(cumulative_return=0.06, drawdown=-0.01)
        psychology = build_psychology_state(snapshot, symbol="AAPL")
        sector_agent = SimpleNamespace(
            agent_id="sector_aapl",
            symbol="AAPL",
            external_context_buffer=[],
            llm_agent=SimpleNamespace(top_k=1, character_string="AAPL analyst", brain=None),
        )
        context = build_agent_context(
            sector_agent=sector_agent,
            cur_date=cur_date,
            latest_reason="Recent service revenue is resilient.",
            master_portfolio_state=snapshot.model_dump(mode="json"),
            psychology_state=psychology,
        )
        raw_context = build_raw_context(context)
        self.assertEqual(context.psychology_state["state"], "greed")
        self.assertIn("Master portfolio state", raw_context)
        self.assertIn("Psychology state", raw_context)
        self.assertIn('"source": "master_portfolio"', raw_context)


def _snapshot(
    *,
    cumulative_return: float,
    drawdown: float,
    daily_return: float = 0.01,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        date=dt.date(2026, 1, 3),
        as_of=dt.datetime(2026, 1, 3, 16, 10),
        cash=42000.0,
        cash_weight=0.42,
        positions={"TSLA": 120.0},
        weights={"TSLA": 0.58},
        equity=100000.0,
        daily_return=daily_return,
        cumulative_return=cumulative_return,
        rolling_return=cumulative_return,
        sharpe=0.8,
        drawdown=drawdown,
        win_rate=0.6,
    )


if __name__ == "__main__":
    unittest.main()
