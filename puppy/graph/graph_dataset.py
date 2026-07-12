"""Graph training dataset builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from puppy.common.schemas import ObservedFeatureTable, QuantFeatureTable, ShadowPortfolioState, TextFeature
from puppy.fusion.text_feature_contract import validate_text_feature_batch
from puppy.graph.correlation_graph import build_correlation_adjacency
from puppy.graph.graph_features import build_causal_node_features, build_node_features


FEATURE_NAMES = [
    "action_signal",
    "observed_close_return_1d",
    "observed_intraday_return_1d",
    "momentum_5d",
    "momentum_20d",
    "rolling_volatility_5d",
    "rolling_volatility_20d",
    "volume_ratio_20d",
    "range_pct",
    "shadow_roi_prior",
    "shadow_rolling_return_prior",
    "shadow_drawdown_prior",
    "shadow_sharpe_prior",
    "shadow_win_rate_prior",
    "text_norm",
    "num_text_features",
    "feature_missing_count",
]

LEGACY_FEATURE_NAMES = [
    "action_signal", "return", "label", "shadow_roi", "shadow_drawdown",
    "shadow_sharpe", "shadow_win_rate", "text_norm", "num_text_features",
]


def build_causal_graph_sample(
    observed_table: ObservedFeatureTable,
    action_signals: dict[str, int],
    shadow_states: dict[str, ShadowPortfolioState | None],
    text_features: dict[str, TextFeature],
    return_history: dict[str, list[float]] | None = None,
    return_source_dates: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    symbols = list(observed_table.symbols)
    node_features = build_causal_node_features(
        observed_table,
        action_signals,
        shadow_states,
        text_features,
    )
    values = [
        [node_features[symbol].get(name) for name in FEATURE_NAMES]
        for symbol in symbols
    ]
    feature_matrix = np.asarray(
        [[_safe_float(value) for value in row] for row in values],
        dtype=np.float32,
    )
    feature_mask = np.asarray(
        [[value is not None for value in row] for row in values],
        dtype=np.bool_,
    )
    source_dates = return_source_dates or {}
    future_sources = [
        date for dates in source_dates.values() for date in dates
        if date > observed_table.date
    ]
    if future_sources:
        raise ValueError(f"Correlation history contains future source date: {min(future_sources)}")
    adjacency = build_correlation_adjacency(return_history, symbols)
    adjacency_matrix = np.asarray(
        [[float(adjacency[target][source]) for source in symbols] for target in symbols],
        dtype=np.float32,
    )
    max_source_date = max(
        (date for dates in source_dates.values() for date in dates),
        default=observed_table.date,
    )
    return {
        "symbols": symbols,
        "node_features": node_features,
        "feature_matrix": feature_matrix,
        "feature_mask": feature_mask,
        "adjacency": adjacency,
        "adjacency_matrix": adjacency_matrix,
        "metadata": {
            "feature_names": list(FEATURE_NAMES),
            "targets_in_feature_matrix": False,
            "max_source_date": max_source_date.isoformat(),
            "decision_date": observed_table.date.isoformat(),
        },
    }


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
            [_safe_float(node_features[symbol].get(name)) for name in LEGACY_FEATURE_NAMES]
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
            "feature_names": list(LEGACY_FEATURE_NAMES),
            "has_labels": bool(labels),
            "temporal_contract": "legacy_unverified",
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
