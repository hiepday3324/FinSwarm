from __future__ import annotations

import math


def _safe_weight(value: float | int | None) -> float:
    if value is None:
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, number)


def clamp_weights(weights: dict[str, float], max_weight: float) -> dict[str, float]:
    cap = max(0.0, float(max_weight))
    return {symbol: min(_safe_weight(weight), cap) for symbol, weight in weights.items()}


def normalize_weights(weights: dict[str, float]) -> tuple[dict[str, float], float]:
    cleaned = {symbol: _safe_weight(weight) for symbol, weight in weights.items()}
    total = sum(cleaned.values())
    if total > 1.0:
        cleaned = {symbol: weight / total for symbol, weight in cleaned.items()}
        total = 1.0
    return cleaned, max(0.0, 1.0 - total)


def apply_basic_risk_rules(
    weights: dict[str, float], max_weight: float = 0.4
) -> tuple[dict[str, float], float]:
    return normalize_weights(clamp_weights(weights, max_weight=max_weight))
