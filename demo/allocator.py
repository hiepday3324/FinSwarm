from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTION_MULTIPLIERS = {"buy": 1.0, "hold": 0.3, "sell": 0.0}
MAX_STOCK_EXPOSURE = 0.90
MAX_SYMBOL_WEIGHT = 0.45


@dataclass(frozen=True)
class AllocationRow:
    symbol: str
    action: str
    fusion_probability: float
    action_multiplier: float
    raw_score: float
    final_weight: float
    dollar_allocation: float


def _capped_proportional_weights(raw_scores: Mapping[str, float]) -> dict[str, float]:
    positive = {key: float(value) for key, value in raw_scores.items() if value > 0.0}
    weights = {key: 0.0 for key in raw_scores}
    if not positive:
        return weights

    active = set(positive)
    remaining = MAX_STOCK_EXPOSURE
    while active and remaining > 1e-12:
        denominator = sum(positive[key] for key in active)
        if denominator <= 0.0:
            break
        proposals = {
            key: remaining * positive[key] / denominator for key in active
        }
        capped = [
            key
            for key, proposed in proposals.items()
            if proposed >= MAX_SYMBOL_WEIGHT - weights[key] - 1e-12
        ]
        if not capped:
            for key, proposed in proposals.items():
                weights[key] += proposed
            remaining = 0.0
            break
        for key in capped:
            addition = max(0.0, MAX_SYMBOL_WEIGHT - weights[key])
            weights[key] += addition
            remaining -= addition
            active.remove(key)
    return weights


def allocate(
    actions: Mapping[str, str],
    fusion_probabilities: Mapping[str, float],
    *,
    capital: float,
) -> dict[str, object]:
    if float(capital) < 0.0:
        raise ValueError("Portfolio capital must be non-negative")
    if set(actions) != set(fusion_probabilities):
        raise ValueError("Actions and probabilities must contain the same symbols")

    multipliers: dict[str, float] = {}
    raw_scores: dict[str, float] = {}
    normalized_actions: dict[str, str] = {}
    for symbol in actions:
        action = str(actions[symbol]).lower()
        if action not in ACTION_MULTIPLIERS:
            raise ValueError(f"Unsupported Agent action for {symbol}: {action}")
        probability = float(fusion_probabilities[symbol])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Fusion probability for {symbol} is outside [0, 1]")
        multiplier = ACTION_MULTIPLIERS[action]
        normalized_actions[symbol] = action
        multipliers[symbol] = multiplier
        raw_scores[symbol] = multiplier * max(0.0, 2.0 * probability - 1.0)

    weights = _capped_proportional_weights(raw_scores)
    stock_exposure = sum(weights.values())
    if stock_exposure > MAX_STOCK_EXPOSURE + 1e-10:
        raise AssertionError("Rule-based allocation exceeded stock exposure cap")
    if any(value > MAX_SYMBOL_WEIGHT + 1e-10 for value in weights.values()):
        raise AssertionError("Rule-based allocation exceeded per-symbol cap")
    cash_weight = 1.0 - stock_exposure
    total = stock_exposure + cash_weight
    if abs(total - 1.0) > 1e-12:
        cash_weight += 1.0 - total

    rows = [
        AllocationRow(
            symbol=symbol,
            action=normalized_actions[symbol],
            fusion_probability=float(fusion_probabilities[symbol]),
            action_multiplier=multipliers[symbol],
            raw_score=raw_scores[symbol],
            final_weight=weights[symbol],
            dollar_allocation=weights[symbol] * float(capital),
        )
        for symbol in actions
    ]
    return {
        "rows": rows,
        "weights": weights,
        "cash_weight": cash_weight,
        "cash_dollars": cash_weight * float(capital),
        "stock_exposure": stock_exposure,
        "total_weight": sum(weights.values()) + cash_weight,
        "capital": float(capital),
        "label": "Rule-based demo allocation",
        "disclaimer": "Not a trained Master Allocator",
    }
