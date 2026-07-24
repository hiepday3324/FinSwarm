"""Portfolio-aware psychology state derived from the master account only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_THRESHOLDS: dict[str, float] = {
    "greed_cumulative_return": 0.05,
    "greed_drawdown_floor": -0.03,
    "fear_cumulative_return": -0.03,
    "fear_daily_return": -0.03,
    "fear_drawdown": -0.08,
}


def build_psychology_state(
    master_portfolio_state: Any | None,
    thresholds: Mapping[str, float] | None = None,
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Convert a prior MasterPortfolio snapshot into a prompt-ready regime.

    The psychology label is descriptive feedback, not another portfolio.  It
    summarizes the actual account's cumulative return, latest daily return and
    drawdown.  A symbol's prior weight is included only as context for that
    SectorAgent; performance metrics remain global because there is one account.
    """

    merged = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        merged.update({key: float(value) for key, value in thresholds.items()})

    metrics = _coerce_master_state(master_portfolio_state)
    if not metrics:
        return _build_state(
            state="neutral",
            risk_attitude="balanced",
            reason="No prior MasterPortfolio snapshot is available yet.",
            prompt_hint="Use normal risk discipline without portfolio-performance bias.",
            metrics={},
            thresholds=merged,
            symbol=symbol,
        )

    cumulative_return = _as_float(metrics.get("cumulative_return"))
    daily_return = _as_float(metrics.get("daily_return"))
    rolling_return = _as_float(metrics.get("rolling_return"))
    drawdown = _as_float(metrics.get("drawdown"))
    sharpe = _as_float(metrics.get("sharpe"))
    win_rate = _as_float(metrics.get("win_rate"))
    cash_weight = _cash_weight(metrics)
    symbol_weight = _symbol_weight(metrics, symbol)

    fear_reasons: list[str] = []
    if cumulative_return is not None and cumulative_return <= merged["fear_cumulative_return"]:
        fear_reasons.append(
            f"cumulative return {cumulative_return:.4f} is below the fear threshold"
        )
    if daily_return is not None and daily_return <= merged["fear_daily_return"]:
        fear_reasons.append(f"daily return {daily_return:.4f} is sharply negative")
    if drawdown is not None and drawdown <= merged["fear_drawdown"]:
        fear_reasons.append(f"drawdown {drawdown:.4f} exceeds the risk limit")

    greed = (
        cumulative_return is not None
        and cumulative_return >= merged["greed_cumulative_return"]
        and (drawdown is None or drawdown >= merged["greed_drawdown_floor"])
    )
    if fear_reasons:
        state = "fear"
        risk_attitude = "risk_averse"
        reason = "; ".join(fear_reasons)
        prompt_hint = (
            "The master account is weak or unstable. Require stronger evidence "
            "and prefer HOLD when the new context is ambiguous."
        )
    elif greed:
        state = "greed"
        risk_attitude = "risk_seeking"
        reason = (
            f"cumulative return {cumulative_return:.4f} is strong while "
            f"drawdown {(drawdown or 0.0):.4f} remains controlled"
        )
        prompt_hint = (
            "The master account is performing strongly. Check downside evidence "
            "explicitly and do not let recent gains alone justify BUY."
        )
    else:
        state = "neutral"
        risk_attitude = "balanced"
        reason = "MasterPortfolio metrics do not indicate a strong greed or fear regime."
        prompt_hint = (
            "Use current market context and memories without a performance-driven "
            "confidence adjustment."
        )

    return _build_state(
        state=state,
        risk_attitude=risk_attitude,
        reason=reason,
        prompt_hint=prompt_hint,
        metrics={
            "equity": _as_float(metrics.get("equity")),
            "cash_weight": cash_weight,
            "symbol_weight": symbol_weight,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "rolling_return": rolling_return,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "win_rate": win_rate,
        },
        thresholds=merged,
        symbol=symbol,
        date=metrics.get("date"),
    )


def update_psychology_state(
    master_portfolio_state: Any | None,
    thresholds: Mapping[str, float] | None = None,
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    return build_psychology_state(
        master_portfolio_state,
        thresholds,
        symbol=symbol,
    )


def psychology_update(
    master_portfolio_state: Any | None,
    thresholds: Mapping[str, float] | None = None,
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    return build_psychology_state(
        master_portfolio_state,
        thresholds,
        symbol=symbol,
    )


def _build_state(
    *,
    state: str,
    risk_attitude: str,
    reason: str,
    prompt_hint: str,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    symbol: str | None,
    date: Any | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "psychology": state,
        "risk_attitude": risk_attitude,
        "reason": reason,
        "prompt_hint": prompt_hint,
        "metrics": metrics,
        "thresholds": thresholds,
        "source": "master_portfolio",
        "symbol": symbol,
        "date": date,
    }


def _coerce_master_state(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in (
            "date",
            "cash",
            "cash_weight",
            "positions",
            "weights",
            "equity",
            "daily_return",
            "cumulative_return",
            "rolling_return",
            "sharpe",
            "drawdown",
            "win_rate",
        )
        if hasattr(value, key)
    }


def _cash_weight(metrics: Mapping[str, Any]) -> float | None:
    explicit = _as_float(metrics.get("cash_weight"))
    if explicit is not None:
        return explicit
    cash = _as_float(metrics.get("cash"))
    equity = _as_float(metrics.get("equity"))
    if cash is None or equity in (None, 0.0):
        return None
    return cash / equity


def _symbol_weight(metrics: Mapping[str, Any], symbol: str | None) -> float | None:
    if symbol is None:
        return None
    weights = metrics.get("weights")
    if not isinstance(weights, Mapping):
        return None
    return _as_float(weights.get(symbol))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
