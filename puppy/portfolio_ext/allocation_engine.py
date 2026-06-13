from __future__ import annotations

import datetime as dt
from typing import Any

from puppy.common.schemas import AgentOutput, AllocationWeights
from puppy.quant.risk_rules import apply_basic_risk_rules
from puppy.quant.signal_utils import decision_to_action_signal


def _signals_from_agent_outputs(agent_outputs: list[AgentOutput]) -> dict[str, int]:
    return {
        output.signal.symbol: decision_to_action_signal(output.signal.decision)
        for output in agent_outputs
    }


def signals_to_target_weights(
    date: dt.date,
    symbols: list[str],
    action_signals: dict[str, int] | None = None,
    agent_outputs: list[AgentOutput] | None = None,
    max_weight: float = 0.4,
) -> AllocationWeights:
    if action_signals is None:
        action_signals = _signals_from_agent_outputs(agent_outputs or [])

    positive_symbols = [
        symbol for symbol in symbols if int(action_signals.get(symbol, 0)) > 0
    ]
    raw_weight = 1.0 / len(positive_symbols) if positive_symbols else 0.0
    raw_weights = {
        symbol: raw_weight if symbol in positive_symbols else 0.0 for symbol in symbols
    }
    weights, cash_weight = apply_basic_risk_rules(raw_weights, max_weight=max_weight)
    reason = (
        f"MVP allocation from positive action signals: "
        f"{len(positive_symbols)} active, max_weight={max_weight}."
    )
    return AllocationWeights(
        date=date,
        weights=weights,
        cash_weight=cash_weight,
        reason=reason,
        metadata={"action_signals": dict(action_signals)},
    )
