from __future__ import annotations

import datetime as dt
from typing import Any

from puppy.common.schemas import AllocationWeights, PortfolioSnapshot
from puppy.quant.portfolio_metrics import (
    compute_drawdown,
    compute_return,
    compute_rolling_return,
    compute_sharpe,
    compute_win_rate,
)


def _price_value(price: Any) -> float:
    return float(price.close if hasattr(price, "close") else price)


class MasterPortfolio:
    def __init__(self, initial_cash: float = 100000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.positions: dict[str, float] = {}
        self._last_equity: float = self.initial_cash
        self._equity_curve: list[float] = [self.initial_cash]
        self._return_history: list[float] = []

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
        as_of: dt.datetime | None = None,
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
            0.0 if self._last_equity == 0.0 else new_equity / self._last_equity - 1.0
        )
        self._last_equity = new_equity
        self._equity_curve.append(new_equity)
        self._return_history.append(daily_return)

        weights = {}
        if new_equity > 0:
            weights = {
                symbol: shares * _price_value(prices[symbol]) / new_equity
                for symbol, shares in self.positions.items()
            }
        cash_weight = self.cash / new_equity if new_equity > 0 else 0.0

        return PortfolioSnapshot(
            date=date,
            as_of=as_of,
            cash=self.cash,
            positions=dict(self.positions),
            weights=weights,
            equity=new_equity,
            cash_weight=cash_weight,
            daily_return=daily_return,
            cumulative_return=compute_return(new_equity, self.initial_cash),
            rolling_return=compute_rolling_return(self._return_history),
            sharpe=compute_sharpe(self._return_history),
            drawdown=compute_drawdown(self._equity_curve),
            win_rate=compute_win_rate(self._return_history),
            metadata={
                "initial_cash": self.initial_cash,
                "target_cash_weight": target_weights.cash_weight,
                "transaction_cost": cost,
                "observations": len(self._return_history),
            },
        )

    def apply_allocation_weights(
        self,
        allocation: AllocationWeights,
        prices: dict[str, Any],
        date: dt.date,
        transaction_cost: float = 0.0,
        as_of: dt.datetime | None = None,
    ) -> PortfolioSnapshot:
        return self.apply_target_weights(
            date=date,
            prices=prices,
            target_weights=allocation,
            transaction_cost=transaction_cost,
            as_of=as_of,
        )


def portfolio_snapshot_to_context(
    snapshot: PortfolioSnapshot | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    if hasattr(snapshot, "model_dump"):
        payload = snapshot.model_dump(mode="json")
    else:
        payload = dict(snapshot)
    return {
        key: payload.get(key)
        for key in (
            "date",
            "as_of",
            "cash",
            "positions",
            "weights",
            "equity",
            "cash_weight",
            "daily_return",
            "cumulative_return",
            "rolling_return",
            "sharpe",
            "drawdown",
            "win_rate",
            "metadata",
        )
        if key in payload
    }
