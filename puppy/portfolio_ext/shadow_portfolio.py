from __future__ import annotations

import datetime as dt

from puppy.common.schemas import AgentOutput, QuantFeatureTable, ShadowPortfolioState
from puppy.quant.shadow_metrics import (
    compute_drawdown,
    compute_roi,
    compute_rolling_return,
    compute_sharpe,
    compute_win_rate,
)


class ShadowPortfolio:
    def __init__(self, initial_value: float = 1.0) -> None:
        self.initial_value = float(initial_value)
        self._values: dict[tuple[str, str], float] = {}
        self._equity_history: dict[tuple[str, str], list[float]] = {}
        self._return_history: dict[tuple[str, str], list[float]] = {}

    def update(
        self,
        agent_id: str,
        symbol: str,
        action_signal: int,
        realized_return: float | None,
        date: dt.date,
    ) -> ShadowPortfolioState:
        key = (agent_id, symbol)
        value_prev = self._values.get(key, self.initial_value)
        safe_return = 0.0 if realized_return is None else float(realized_return)
        shadow_return = int(action_signal) * safe_return
        value_next = value_prev * (1.0 + shadow_return)

        self._values[key] = value_next
        self._equity_history.setdefault(key, []).append(value_next)
        self._return_history.setdefault(key, []).append(shadow_return)

        returns = self._return_history[key]
        curve = self._equity_history[key]
        return ShadowPortfolioState(
            date=date,
            agent_id=agent_id,
            symbol=symbol,
            value=value_next,
            roi=compute_roi(value_next, self.initial_value),
            rolling_return=compute_rolling_return(returns),
            sharpe=compute_sharpe(returns),
            drawdown=compute_drawdown(curve),
            win_rate=compute_win_rate(returns),
            metadata={
                "action_signal": int(action_signal),
                "realized_return": safe_return,
                "shadow_return": shadow_return,
                "previous_value": value_prev,
                "value_next": value_next,
                "initial_value": self.initial_value,
            },
        )

    def update_from_quant_table(
        self,
        quant_table: QuantFeatureTable,
        agent_outputs: list[AgentOutput],
        state_date: dt.date | None = None,
    ) -> list[ShadowPortfolioState]:
        """Update shadow states from a realized quant table.

        ``QuantFeatureTable.returns`` may contain realized t+1 returns. Those
        returns are only valid after they have occurred, when updating
        ``ShadowPortfolioState``. Do not use ``ShadowPortfolioState_{t+1}`` to
        make decisions at day t.
        """

        effective_date = state_date or quant_table.date
        states: list[ShadowPortfolioState] = []
        for output in agent_outputs:
            signal = output.signal
            action_signal = quant_table.action_signals.get(
                signal.symbol, signal.action_signal
            )
            realized_return = quant_table.returns.get(signal.symbol)
            states.append(
                self.update(
                    agent_id=signal.agent_id,
                    symbol=signal.symbol,
                    action_signal=action_signal,
                    realized_return=realized_return,
                    date=effective_date,
                )
            )
        return states


def update_shadow_portfolio(
    agent_id: str,
    symbol: str,
    action_signal: int,
    realized_return: float | None,
    date: dt.date,
    initial_value: float = 1.0,
) -> ShadowPortfolioState:
    return ShadowPortfolio(initial_value=initial_value).update(
        agent_id=agent_id,
        symbol=symbol,
        action_signal=action_signal,
        realized_return=realized_return,
        date=date,
    )
