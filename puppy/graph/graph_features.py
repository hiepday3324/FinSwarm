"""Graph node and edge feature construction."""

from __future__ import annotations

import math
from typing import Any

from puppy.common.schemas import QuantFeatureTable, ShadowPortfolioState, TextFeature


def build_node_features(
    quant_table: QuantFeatureTable,
    shadow_states: list[ShadowPortfolioState] | None = None,
    text_features: list[TextFeature] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-symbol graph features from the data explicitly passed in.

    Shadow states are joined by symbol and the latest state by date is used
    when several states exist for one symbol. Text features are joined by
    symbol and ``text_norm`` is the average L2 norm across all text vectors for
    that symbol.
    """

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
            "shadow_drawdown": (
                shadow_state.drawdown if shadow_state is not None else 0.0
            ),
            "shadow_sharpe": shadow_state.sharpe if shadow_state is not None else 0.0,
            "shadow_win_rate": (
                shadow_state.win_rate if shadow_state is not None else 0.0
            ),
            "text_norm": text_norm,
            "num_text_features": len(norms),
        }
    return node_features


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))
