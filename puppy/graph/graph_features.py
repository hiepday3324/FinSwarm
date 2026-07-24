"""Canonical graph node feature construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from puppy.common.schemas import (
    ObservedFeatureTable,
    PortfolioSnapshot,
    QuantFeatureTable,
    TextFeature,
)


MARKET_FEATURE_NAMES = (
    "observed_close_return_1d",
    "observed_intraday_return_1d",
    "momentum_5d",
    "momentum_20d",
    "rolling_volatility_5d",
    "rolling_volatility_20d",
    "volume_ratio_20d",
    "range_pct",
    "feature_missing_count",
)
ACTION_FEATURE_NAMES = ("action_signal",)
TEXT_SCORE_FEATURE_NAMES = ("text_score",)
MASTER_PORTFOLIO_FEATURE_NAMES = (
    "master_symbol_weight_prior",
    "master_cash_weight_prior",
    "master_daily_return_prior",
    "master_cumulative_return_prior",
    "master_drawdown_prior",
)
PSYCHOLOGY_FEATURE_NAMES = (
    "psychology_greed",
    "psychology_fear",
    "psychology_neutral",
)


def encode_psychology_state(value: Mapping[str, Any] | None) -> dict[str, float]:
    state = "neutral"
    if value:
        state = str(value.get("state", value.get("psychology", "neutral"))).lower()
    if state not in {"greed", "fear", "neutral"}:
        state = "neutral"
    return {
        "psychology_greed": float(state == "greed"),
        "psychology_fear": float(state == "fear"),
        "psychology_neutral": float(state == "neutral"),
    }


def build_causal_node_features(
    observed_table: ObservedFeatureTable,
    action_signals: Mapping[str, int],
    text_scores: Mapping[str, float | None],
    master_portfolio_state: PortfolioSnapshot | Mapping[str, Any] | None,
    psychology_states: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    """Build authoritative H0 blocks from decision-time and prior-only state."""

    result: dict[str, dict[str, Any]] = {}
    portfolio = _portfolio_payload(master_portfolio_state)
    weights = portfolio.get("weights")
    weights = weights if isinstance(weights, Mapping) else {}
    cash_weight = _portfolio_cash_weight(portfolio)
    for symbol in observed_table.symbols:
        text_score = text_scores.get(symbol)
        if text_score is not None and not 0.0 <= float(text_score) <= 1.0:
            raise ValueError(f"text_score for {symbol} must be in [0, 1]")
        market = {
            name: observed_table.features[symbol].get(name)
            for name in MARKET_FEATURE_NAMES
        }
        result[symbol] = {
            **market,
            "action_signal": int(action_signals.get(symbol, 0)),
            "text_score": 0.5 if text_score is None else float(text_score),
            "master_symbol_weight_prior": _safe_metric(weights.get(symbol)),
            "master_cash_weight_prior": cash_weight,
            "master_daily_return_prior": _safe_metric(portfolio.get("daily_return")),
            "master_cumulative_return_prior": _safe_metric(
                portfolio.get("cumulative_return")
            ),
            "master_drawdown_prior": _safe_metric(portfolio.get("drawdown")),
            **encode_psychology_state(psychology_states.get(symbol)),
            "_availability": {
                "text_score": text_score is not None,
                "master_portfolio_state": bool(portfolio),
                "psychology_state": psychology_states.get(symbol) is not None,
            },
        }
    return result


def build_node_features(
    quant_table: QuantFeatureTable,
    master_portfolio_state: PortfolioSnapshot | Mapping[str, Any] | None = None,
    text_features: list[TextFeature] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the explicitly legacy milestone graph feature contract."""

    portfolio = _portfolio_payload(master_portfolio_state)
    weights = portfolio.get("weights")
    weights = weights if isinstance(weights, Mapping) else {}
    cash_weight = _portfolio_cash_weight(portfolio)

    text_norms_by_symbol: dict[str, list[float]] = {}
    for feature in text_features or []:
        text_norms_by_symbol.setdefault(feature.symbol, []).append(
            _l2_norm(feature.h_text)
        )

    node_features: dict[str, dict[str, Any]] = {}
    for symbol in quant_table.symbols:
        norms = text_norms_by_symbol.get(symbol, [])
        text_norm = sum(norms) / len(norms) if norms else 0.0
        node_features[symbol] = {
            "action_signal": int(quant_table.action_signals.get(symbol, 0)),
            "return": quant_table.returns.get(symbol),
            "label": quant_table.labels.get(symbol),
            "master_symbol_weight": _safe_metric(weights.get(symbol)),
            "master_cash_weight": cash_weight,
            "master_daily_return": _safe_metric(portfolio.get("daily_return")),
            "master_drawdown": _safe_metric(portfolio.get("drawdown")),
            "text_norm": text_norm,
            "num_text_features": len(norms),
        }
    return node_features


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _portfolio_payload(
    value: PortfolioSnapshot | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return dict(value)


def _portfolio_cash_weight(value: Mapping[str, Any]) -> float:
    explicit = value.get("cash_weight")
    if explicit is not None:
        return _safe_metric(explicit)
    equity = _safe_metric(value.get("equity"))
    if equity <= 0.0:
        return 0.0
    return _safe_metric(value.get("cash")) / equity


def _safe_metric(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
