from __future__ import annotations

from typing import Any

from puppy.common.schemas import AllocationWeights


def _price_value(price: Any) -> float:
    return float(price.close if hasattr(price, "close") else price)


def target_weights_to_trades(
    current_equity: float,
    current_positions: dict[str, float],
    prices: dict[str, Any],
    target_weights: AllocationWeights | dict[str, float],
) -> dict[str, float]:
    weights = target_weights.weights if hasattr(target_weights, "weights") else target_weights
    trades: dict[str, float] = {}
    for symbol, weight in weights.items():
        price = _price_value(prices[symbol])
        if price <= 0:
            raise ValueError(f"Price must be positive for {symbol}.")
        target_shares = float(current_equity) * float(weight) / price
        trades[symbol] = target_shares - float(current_positions.get(symbol, 0.0))
    return trades
