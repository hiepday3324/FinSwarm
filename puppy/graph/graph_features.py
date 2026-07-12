"""Canonical graph node feature construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from puppy.common.schemas import (
    ObservedFeatureTable,
    QuantFeatureTable,
    ShadowPortfolioState,
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
SHADOW_FEATURE_NAMES = (
    "shadow_roi_prior",
    "shadow_rolling_return_prior",
    "shadow_sharpe_prior",
    "shadow_drawdown_prior",
    "shadow_win_rate_prior",
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
    shadow_states: Mapping[str, ShadowPortfolioState | None],
    psychology_states: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    """Build authoritative H0 blocks from decision-time and prior-only state."""

    result: dict[str, dict[str, Any]] = {}
    for symbol in observed_table.symbols:
        state = shadow_states.get(symbol)
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
            "shadow_roi_prior": state.roi if state else 0.0,
            "shadow_rolling_return_prior": state.rolling_return if state else 0.0,
            "shadow_sharpe_prior": state.sharpe if state else 0.0,
            "shadow_drawdown_prior": state.drawdown if state else 0.0,
            "shadow_win_rate_prior": state.win_rate if state else 0.0,
            **encode_psychology_state(psychology_states.get(symbol)),
            "_availability": {
                "text_score": text_score is not None,
                "shadow_state": state is not None,
                "psychology_state": psychology_states.get(symbol) is not None,
            },
        }
    return result


def build_node_features(
    quant_table: QuantFeatureTable,
    shadow_states: list[ShadowPortfolioState] | None = None,
    text_features: list[TextFeature] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the explicitly legacy milestone graph feature contract."""

    latest_shadow_by_symbol: dict[str, ShadowPortfolioState] = {}
    for state in shadow_states or []:
        previous = latest_shadow_by_symbol.get(state.symbol)
        if previous is None or state.date >= previous.date:
            latest_shadow_by_symbol[state.symbol] = state

    text_norms_by_symbol: dict[str, list[float]] = {}
    for feature in text_features or []:
        text_norms_by_symbol.setdefault(feature.symbol, []).append(
            _l2_norm(feature.h_text)
        )

    node_features: dict[str, dict[str, Any]] = {}
    for symbol in quant_table.symbols:
        shadow_state = latest_shadow_by_symbol.get(symbol)
        norms = text_norms_by_symbol.get(symbol, [])
        text_norm = sum(norms) / len(norms) if norms else 0.0
        node_features[symbol] = {
            "action_signal": int(quant_table.action_signals.get(symbol, 0)),
            "return": quant_table.returns.get(symbol),
            "label": quant_table.labels.get(symbol),
            "shadow_roi": shadow_state.roi if shadow_state is not None else 0.0,
            "shadow_drawdown": shadow_state.drawdown if shadow_state is not None else 0.0,
            "shadow_sharpe": shadow_state.sharpe if shadow_state is not None else 0.0,
            "shadow_win_rate": shadow_state.win_rate if shadow_state is not None else 0.0,
            "text_norm": text_norm,
            "num_text_features": len(norms),
        }
    return node_features


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))
