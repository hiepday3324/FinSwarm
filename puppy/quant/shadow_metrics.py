from __future__ import annotations

import math
from collections.abc import Iterable


def _clean(values: Iterable[float | int | None]) -> list[float]:
    cleaned = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            cleaned.append(number)
    return cleaned


def compute_roi(value: float, initial_value: float) -> float:
    if initial_value == 0:
        return 0.0
    return float(value) / float(initial_value) - 1.0


def compute_drawdown(equity_curve: Iterable[float | int | None]) -> float:
    values = _clean(equity_curve)
    if not values:
        return 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = min(max_drawdown, value / peak - 1.0)
    return max_drawdown


def compute_rolling_return(returns: Iterable[float | int | None]) -> float:
    values = _clean(returns)
    if not values:
        return 0.0
    compounded = 1.0
    for value in values:
        compounded *= 1.0 + value
    return compounded - 1.0


def compute_sharpe(returns: Iterable[float | int | None], eps: float = 1e-12) -> float:
    values = _clean(returns)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(max(variance, 0.0))
    if std <= eps:
        return 0.0
    return mean / std


def compute_win_rate(returns: Iterable[float | int | None]) -> float:
    values = _clean(returns)
    if not values:
        return 0.0
    wins = sum(1 for value in values if value > 0)
    return wins / len(values)
