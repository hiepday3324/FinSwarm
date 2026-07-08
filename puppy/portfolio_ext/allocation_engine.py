from __future__ import annotations

import datetime as dt
from typing import Any

from puppy.common.schemas import AgentOutput, AllocationWeights, FusionOutput
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


def fusion_to_target_weights(
    fusion_output: FusionOutput,
    max_weight: float = 0.35,
    min_cash_weight: float = 0.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    conflict_weight_multiplier: float = 0.5,
) -> AllocationWeights:
    available_weight = max(0.0, min(1.0, 1.0 - float(min_cash_weight)))
    raw_weights: dict[str, float] = {}
    for symbol in fusion_output.symbols:
        score = float(fusion_output.fusion_scores.get(symbol, 0.0))
        if score >= buy_threshold:
            raw_weight = max(0.0, score - buy_threshold)
            if raw_weight == 0.0:
                raw_weight = score
            if fusion_output.conflict_flags.get(symbol, False):
                raw_weight *= max(0.0, float(conflict_weight_multiplier))
            raw_weights[symbol] = raw_weight
        else:
            raw_weights[symbol] = 0.0

    total_raw = sum(raw_weights.values())
    if total_raw > 0.0:
        scaled = {
            symbol: weight / total_raw * available_weight
            for symbol, weight in raw_weights.items()
        }
    else:
        scaled = dict(raw_weights)

    weights, _cash = apply_basic_risk_rules(scaled, max_weight=max_weight)
    total_weight = sum(weights.values())
    if total_weight > available_weight and total_weight > 0.0:
        scale = available_weight / total_weight
        weights = {symbol: weight * scale for symbol, weight in weights.items()}
        total_weight = sum(weights.values())
    cash_weight = max(0.0, 1.0 - total_weight)
    return AllocationWeights(
        date=fusion_output.date,
        weights=weights,
        cash_weight=cash_weight,
        reason=(
            "MVP allocation from FusionOutput scores with long-only risk rules."
        ),
        metadata={
            "method": "fusion_to_target_weights_mvp",
            "max_weight": float(max_weight),
            "min_cash_weight": float(min_cash_weight),
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
            "conflict_weight_multiplier": float(conflict_weight_multiplier),
            "source": "FusionOutput",
        },
    )
