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
        self._states: dict[tuple[str, str], ShadowPortfolioState] = {}
        self._resolved_counts: dict[tuple[str, str], int] = {}

    def update(
        self,
        agent_id: str,
        symbol: str,
        action_signal: int,
        realized_return: float | None,
        date: dt.date,
    ) -> ShadowPortfolioState:
        return self.resolve_outcome(
            agent_id=agent_id,
            symbol=symbol,
            action_signal=action_signal,
            realized_return=realized_return,
            decision_date=date,
            outcome_date=date,
            as_of=dt.datetime.combine(date, dt.time.max, dt.timezone.utc),
        )

    def resolve_outcome(
        self,
        *,
        agent_id: str,
        symbol: str,
        action_signal: int,
        realized_return: float | None,
        decision_date: dt.date,
        outcome_date: dt.date,
        as_of: dt.datetime,
    ) -> ShadowPortfolioState:
        if outcome_date < decision_date:
            raise ValueError("Outcome date cannot precede decision date")
        key = (agent_id, symbol)
        value_prev = self._values.get(key, self.initial_value)
        safe_return = 0.0 if realized_return is None else float(realized_return)
        shadow_return = int(action_signal) * safe_return
        value_next = value_prev * (1.0 + shadow_return)

        self._values[key] = value_next
        self._equity_history.setdefault(key, []).append(value_next)
        self._return_history.setdefault(key, []).append(shadow_return)
        self._resolved_counts[key] = self._resolved_counts.get(key, 0) + 1

        returns = self._return_history[key]
        curve = self._equity_history[key]
        state = ShadowPortfolioState(
            date=outcome_date,
            agent_id=agent_id,
            symbol=symbol,
            value=value_next,
            roi=compute_roi(value_next, self.initial_value),
            rolling_return=compute_rolling_return(returns),
            sharpe=compute_sharpe(returns),
            drawdown=compute_drawdown(curve),
            win_rate=compute_win_rate(returns),
            as_of=as_of,
            last_resolved_decision_date=decision_date,
            last_outcome_date=outcome_date,
            number_of_resolved_decisions=self._resolved_counts[key],
            metadata={
                "action_signal": int(action_signal),
                "realized_return": safe_return,
                "shadow_return": shadow_return,
                "previous_value": value_prev,
                "value_next": value_next,
                "initial_value": self.initial_value,
                "last_resolved_decision_date": decision_date.isoformat(),
                "last_outcome_date": outcome_date.isoformat(),
                "as_of": as_of.isoformat(),
                "number_of_resolved_decisions": self._resolved_counts[key],
            },
        )
        self._states[key] = state
        return state

    def get_state(self, agent_id: str, symbol: str) -> ShadowPortfolioState | None:
        return self._states.get((agent_id, symbol))

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
