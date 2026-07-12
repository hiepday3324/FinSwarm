"""Graph training dataset builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from puppy.common.schemas import ObservedFeatureTable, QuantFeatureTable, ShadowPortfolioState, TextFeature
from puppy.fusion.text_feature_contract import validate_text_feature_batch
from puppy.graph.correlation_graph import build_correlation_adjacency
from puppy.graph.graph_features import (
    ACTION_FEATURE_NAMES,
    MARKET_FEATURE_NAMES,
    PSYCHOLOGY_FEATURE_NAMES,
    SHADOW_FEATURE_NAMES,
    TEXT_SCORE_FEATURE_NAMES,
    build_causal_node_features,
    build_node_features,
)


CANONICAL_FEATURE_NAMES = (
    MARKET_FEATURE_NAMES
    + ACTION_FEATURE_NAMES
    + TEXT_SCORE_FEATURE_NAMES
    + SHADOW_FEATURE_NAMES
    + PSYCHOLOGY_FEATURE_NAMES
)
FEATURE_NAMES = list(CANONICAL_FEATURE_NAMES)

LEGACY_FEATURE_NAMES = [
    "action_signal", "return", "label", "shadow_roi", "shadow_drawdown",
    "shadow_sharpe", "shadow_win_rate", "text_norm", "num_text_features",
]


def build_causal_graph_sample(
    observed_table: ObservedFeatureTable,
    action_signals: dict[str, int],
    shadow_states: dict[str, ShadowPortfolioState | None] | None = None,
    text_features: dict[str, TextFeature] | None = None,
    return_history: dict[str, list[float]] | None = None,
    return_source_dates: dict[str, list[Any]] | None = None,
    *,
    text_scores: dict[str, float | None] | None = None,
    psychology_states: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    symbols = list(observed_table.symbols)
    shadow_states = shadow_states or {}
    text_features = text_features or {}
    if text_scores is None:
        text_scores = {
            symbol: (
                getattr(text_features[symbol], "metadata", {}).get("text_score")
                if symbol in text_features
                else None
            )
            for symbol in symbols
        }
    psychology_states = psychology_states or {symbol: None for symbol in symbols}
    node_features = build_causal_node_features(
        observed_table,
        action_signals,
        text_scores,
        shadow_states,
        psychology_states,
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
        [
            [
                value is not None
                and not (
                    name == "text_score"
                    and not node_features[symbol]["_availability"]["text_score"]
                )
                for name, value in zip(FEATURE_NAMES, row)
            ]
            for symbol, row in zip(symbols, values)
        ],
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
            "feature_blocks": {
                "market_features": list(MARKET_FEATURE_NAMES),
                "action_signal": list(ACTION_FEATURE_NAMES),
                "text_score": list(TEXT_SCORE_FEATURE_NAMES),
                "prior_shadow_features": list(SHADOW_FEATURE_NAMES),
                "prior_psychology_features": list(PSYCHOLOGY_FEATURE_NAMES),
            },
            "feature_dimension": len(FEATURE_NAMES),
            "text_score_source": (
                "provided_causal_or_oof_score"
                if all(text_scores.get(symbol) is not None for symbol in symbols)
                else "pending_oof_replacement"
            ),
            "psychology_encoding": {"greed": [1, 0, 0], "fear": [0, 1, 0], "neutral": [0, 0, 1]},
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
