import datetime as dt
import unittest
from types import SimpleNamespace

from puppy.common.schemas import ShadowPortfolioState
from puppy.swarm.context_builder import build_agent_context, build_raw_context
from puppy.swarm.psychology import build_psychology_state, update_psychology_state


class SwarmPsychologyTest(unittest.TestCase):
    def test_build_psychology_state_detects_greed(self) -> None:
        shadow_state = ShadowPortfolioState(
            date=dt.date(2026, 1, 3),
            agent_id="sector_nvda",
            symbol="NVDA",
            value=1.08,
            roi=0.08,
            rolling_return=0.08,
            sharpe=1.2,
            drawdown=-0.01,
            win_rate=0.7,
        )

        psychology = build_psychology_state(shadow_state)

        self.assertEqual(psychology["state"], "greed")
        self.assertEqual(psychology["risk_attitude"], "risk_seeking")
        self.assertEqual(psychology["metrics"]["roi"], 0.08)
        self.assertIn("downside", psychology["prompt_hint"])

    def test_build_psychology_state_detects_fear(self) -> None:
        psychology = update_psychology_state(
            {
                "date": dt.date(2026, 1, 3),
                "agent_id": "sector_tsla",
                "symbol": "TSLA",
                "value": 0.9,
                "roi": -0.1,
                "drawdown": -0.12,
                "win_rate": 0.25,
            }
        )

        self.assertEqual(psychology["state"], "fear")
        self.assertEqual(psychology["risk_attitude"], "risk_averse")
        self.assertIn("drawdown", psychology["reason"])

    def test_missing_shadow_state_is_neutral(self) -> None:
        psychology = build_psychology_state(None)

        self.assertEqual(psychology["state"], "neutral")
        self.assertEqual(psychology["metrics"], {})

    def test_flat_shadow_state_is_neutral(self) -> None:
        psychology = build_psychology_state(
            {
                "date": dt.date(2026, 1, 3),
                "agent_id": "sector_nvda",
                "symbol": "NVDA",
                "value": 1.0,
                "roi": 0.0,
                "drawdown": 0.0,
                "win_rate": 0.0,
            }
        )

        self.assertEqual(psychology["state"], "neutral")

    def test_context_includes_shadow_and_psychology_state(self) -> None:
        cur_date = dt.date(2026, 1, 4)
        shadow_state = ShadowPortfolioState(
            date=dt.date(2026, 1, 3),
            agent_id="sector_aapl",
            symbol="AAPL",
            value=1.06,
            roi=0.06,
            rolling_return=0.06,
            sharpe=0.9,
            drawdown=-0.01,
            win_rate=0.6,
        )
        psychology = build_psychology_state(shadow_state)
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
            shadow_state=shadow_state.model_dump(),
            psychology_state=psychology,
        )
        raw_context = build_raw_context(context)

        self.assertEqual(context.psychology_state["state"], "greed")
        self.assertIn("Shadow state", raw_context)
        self.assertIn("Psychology state", raw_context)
        self.assertIn("greed", raw_context)


if __name__ == "__main__":
    unittest.main()
