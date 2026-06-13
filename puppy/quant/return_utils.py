from __future__ import annotations

import datetime as dt
import math
from typing import Any

from puppy.common.schemas import AgentOutput, MarketStep, QuantFeatureTable

from .signal_utils import decision_to_action_signal


def compute_simple_return(prev_price: float | int | None, next_price: float | int | None) -> float | None:
    if prev_price is None or next_price is None:
        return None
    prev = float(prev_price)
    nxt = float(next_price)
    if prev <= 0:
        return None
    return nxt / prev - 1.0


def compute_log_return(prev_price: float | int | None, next_price: float | int | None) -> float | None:
    if prev_price is None or next_price is None:
        return None
    prev = float(prev_price)
    nxt = float(next_price)
    if prev <= 0 or nxt <= 0:
        return None
    return math.log(nxt / prev)


def build_direction_label(return_value: float | None, threshold: float = 0.0) -> int | None:
    if return_value is None:
        return None
    if return_value > threshold:
        return 1
    if return_value < -threshold:
        return -1
    return 0


def _agent_output_symbol(output: AgentOutput | Any) -> str:
    return output.signal.symbol if hasattr(output, "signal") else output["signal"]["symbol"]


def _agent_output_decision(output: AgentOutput | Any) -> Any:
    return output.signal.decision if hasattr(output, "signal") else output["signal"]["decision"]


def build_quant_feature_table(
    date: dt.date,
    symbols: list[str],
    agent_outputs: list[AgentOutput],
    current_market_step: MarketStep,
    next_market_step: MarketStep | None = None,
) -> QuantFeatureTable:
    """Build quant features for day t.

    The optional t+1 return is only for labels/evaluation after the next price is
    realized. It must not be used to decide allocations for day t.
    """
    signal_by_symbol = {
        _agent_output_symbol(output): decision_to_action_signal(_agent_output_decision(output))
        for output in agent_outputs
    }
    action_signals = {symbol: int(signal_by_symbol.get(symbol, 0)) for symbol in symbols}
    returns: dict[str, float | None] = {}
    labels: dict[str, int | None] = {}
    features: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        current_bar = current_market_step.prices.get(symbol)
        next_bar = next_market_step.prices.get(symbol) if next_market_step else None
        realized_return = compute_simple_return(
            current_bar.close if current_bar else None,
            next_bar.close if next_bar else None,
        )
        returns[symbol] = realized_return
        labels[symbol] = build_direction_label(realized_return)
        features[symbol] = {
            "action_signal": action_signals[symbol],
            "current_close": current_bar.close if current_bar else None,
            "has_next_price": next_bar is not None,
            "news_count": len(current_market_step.news.get(symbol, [])),
            "filing_count": len(current_market_step.filings.get(symbol, [])),
        }

    return QuantFeatureTable(
        date=date,
        symbols=symbols,
        action_signals=action_signals,
        returns=returns,
        labels=labels,
        features=features,
        metadata={"label_return": "t+1_simple_return"},
    )
