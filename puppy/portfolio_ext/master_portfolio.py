from __future__ import annotations

import datetime as dt
from typing import Any

from puppy.common.schemas import AllocationWeights, PortfolioSnapshot
from puppy.quant.shadow_metrics import compute_drawdown


def _price_value(price: Any) -> float:
    return float(price.close if hasattr(price, "close") else price)


class MasterPortfolio:
    def __init__(self, initial_cash: float = 100000.0) -> None:
        self.cash = float(initial_cash)
        self.positions: dict[str, float] = {}
        self._last_equity: float | None = None
        self._equity_curve: list[float] = []

    def equity(self, prices: dict[str, Any]) -> float:
        position_value = sum(
            shares * _price_value(prices[symbol])
            for symbol, shares in self.positions.items()
            if symbol in prices
        )
        return self.cash + position_value

    def apply_target_weights(
        self,
        date: dt.date,
        prices: dict[str, Any],
        target_weights: AllocationWeights,
        transaction_cost: float = 0.0,
    ) -> PortfolioSnapshot:
        current_equity = self.equity(prices)
        new_positions: dict[str, float] = {}
        total_position_value = 0.0

        for symbol, weight in target_weights.weights.items():
            price = _price_value(prices[symbol])
            if price <= 0:
                raise ValueError(f"Price must be positive for {symbol}.")
            target_value = current_equity * max(0.0, float(weight))
            shares = target_value / price
            new_positions[symbol] = shares
            total_position_value += shares * price

        cost = max(0.0, float(transaction_cost))
        self.positions = new_positions
        self.cash = max(0.0, current_equity - total_position_value - cost)
        new_equity = self.equity(prices)
        daily_return = (
            None if self._last_equity in (None, 0.0) else new_equity / self._last_equity - 1.0
        )
        self._last_equity = new_equity
        self._equity_curve.append(new_equity)

        weights = {}
        if new_equity > 0:
            weights = {
                symbol: shares * _price_value(prices[symbol]) / new_equity
                for symbol, shares in self.positions.items()
            }

        return PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions=dict(self.positions),
            weights=weights,
            equity=new_equity,
            daily_return=daily_return,
            drawdown=compute_drawdown(self._equity_curve),
            metadata={
                "target_cash_weight": target_weights.cash_weight,
                "transaction_cost": cost,
            },
        )
