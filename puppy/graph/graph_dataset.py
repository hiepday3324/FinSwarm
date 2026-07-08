"""Graph training dataset builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from puppy.common.schemas import QuantFeatureTable, ShadowPortfolioState, TextFeature
from puppy.fusion.text_feature_contract import validate_text_feature_batch
from puppy.graph.correlation_graph import build_correlation_adjacency
from puppy.graph.graph_features import build_node_features


FEATURE_NAMES = [
    "action_signal",
    "return",
    "label",
    "shadow_roi",
    "shadow_drawdown",
    "shadow_sharpe",
    "shadow_win_rate",
    "text_norm",
    "num_text_features",
]


def build_graph_sample(
    quant_table: QuantFeatureTable,
    shadow_states: list[ShadowPortfolioState] | None = None,
    text_features: list[TextFeature] | None = None,
    return_history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic graph sample for GAT MVP inference/training."""

    if text_features:
        validate_text_feature_batch(text_features)

    symbols = list(quant_table.symbols)
    node_features = build_node_features(
        quant_table=quant_table,
        shadow_states=shadow_states,
        text_features=text_features,
    )
    feature_matrix = np.asarray(
        [
            [_safe_float(node_features[symbol].get(name)) for name in FEATURE_NAMES]
            for symbol in symbols
        ],
        dtype=np.float32,
    )
    adjacency = build_correlation_adjacency(
        return_history=return_history,
        symbols=symbols,
    )
    adjacency_matrix = np.asarray(
        [[float(adjacency[target][source]) for source in symbols] for target in symbols],
        dtype=np.float32,
    )
    labels = {
        symbol: quant_table.labels.get(symbol)
        for symbol in symbols
        if quant_table.labels.get(symbol) is not None
    }
    return {
        "symbols": symbols,
        "node_features": node_features,
        "feature_matrix": feature_matrix,
        "adjacency": adjacency,
        "adjacency_matrix": adjacency_matrix,
        "labels": labels,
        "metadata": {
            "feature_names": list(FEATURE_NAMES),
            "has_labels": bool(labels),
            "uses_future_data": False,
        },
    }


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return number
