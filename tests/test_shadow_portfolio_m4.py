import datetime as dt
import unittest

from puppy.common.schemas import AgentOutput, AgentSignal, QuantFeatureTable
from puppy.portfolio_ext.shadow_portfolio import ShadowPortfolio


class ShadowPortfolioM4Test(unittest.TestCase):
    def test_shadow_portfolio_tracks_history_metrics_and_metadata(self) -> None:
        portfolio = ShadowPortfolio(initial_value=100.0)
        first = portfolio.update(
            agent_id="sector_tsla",
            symbol="TSLA",
            action_signal=1,
            realized_return=0.10,
            date=dt.date(2026, 1, 2),
        )
        second = portfolio.update(
            agent_id="sector_tsla",
            symbol="TSLA",
            action_signal=1,
            realized_return=-0.20,
            date=dt.date(2026, 1, 3),
        )

        self.assertAlmostEqual(first.value, 110.0)
        self.assertAlmostEqual(second.metadata["shadow_return"], -0.20)
        self.assertAlmostEqual(second.metadata["previous_value"], 110.0)
        self.assertAlmostEqual(second.metadata["value_next"], 88.0)
        self.assertAlmostEqual(second.metadata["initial_value"], 100.0)
        self.assertAlmostEqual(second.value, 88.0)
        self.assertAlmostEqual(second.roi, -0.12)
        self.assertAlmostEqual(second.drawdown, -0.20)
        self.assertAlmostEqual(second.win_rate, 0.5)

    def test_update_from_quant_table_handles_missing_returns_and_agent_keys(
        self,
    ) -> None:
        date = dt.date(2026, 1, 3)
        agent_outputs = [
            _agent_output("agent_a", "TSLA", 1, date),
            _agent_output("agent_b", "TSLA", -1, date),
            _agent_output("agent_c", "NVDA", 1, date),
        ]
        table = QuantFeatureTable(
            date=date,
            symbols=["TSLA", "NVDA"],
            action_signals={"TSLA": -1},
            returns={"TSLA": None},
        )
        portfolio = ShadowPortfolio(initial_value=10.0)

        states = portfolio.update_from_quant_table(table, agent_outputs)

        self.assertEqual(len(states), 3)
        self.assertEqual({state.agent_id for state in states}, {"agent_a", "agent_b", "agent_c"})
        self.assertEqual([state.symbol for state in states].count("TSLA"), 2)
        self.assertTrue(all(state.value == 10.0 for state in states))
        tsla_states = [state for state in states if state.symbol == "TSLA"]
        self.assertTrue(
            all(state.metadata["action_signal"] == -1 for state in tsla_states)
        )
        nvda_state = next(state for state in states if state.symbol == "NVDA")
        self.assertEqual(nvda_state.metadata["action_signal"], 1)
        self.assertEqual(nvda_state.metadata["realized_return"], 0.0)


def _agent_output(
    agent_id: str,
    symbol: str,
    action_signal: int,
    date: dt.date,
) -> AgentOutput:
    return AgentOutput(
        signal=AgentSignal(
            agent_id=agent_id,
            symbol=symbol,
            date=date,
            decision="buy" if action_signal > 0 else "sell",
            action_signal=action_signal,
            reason="test",
        )
    )


if __name__ == "__main__":
    unittest.main()
