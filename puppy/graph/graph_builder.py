"""Graph state builder for SectorAgent relationships."""

from __future__ import annotations

import datetime as dt

from puppy.common.schemas import GraphOutput, QuantFeatureTable, ShadowPortfolioState, TextFeature
from puppy.graph.correlation_graph import (
    build_attention_from_adjacency,
    build_correlation_adjacency,
)
from puppy.graph.graph_features import build_node_features


def build_graph_output(
    date: dt.date,
    quant_table: QuantFeatureTable,
    shadow_states: list[ShadowPortfolioState] | None = None,
    text_features: list[TextFeature] | None = None,
    return_history: dict[str, list[float]] | None = None,
    min_abs_corr: float = 0.2,
) -> GraphOutput:
    """Build the correlation graph MVP without GAT or trained GNN logic."""

    symbols = list(quant_table.symbols)
    node_features = build_node_features(
        quant_table=quant_table,
        shadow_states=shadow_states,
        text_features=text_features,
    )
    adjacency = build_correlation_adjacency(
        return_history=return_history,
        symbols=symbols,
        min_abs_corr=min_abs_corr,
    )
    attention = build_attention_from_adjacency(adjacency)
    graph_scores: dict[str, float] = {}
    for target in symbols:
        score = 0.0
        for source, alpha in attention.get(target, {}).items():
            if source == target:
                continue
            source_signal = node_features.get(source, {}).get("action_signal", 0)
            score += float(alpha) * float(source_signal)
        graph_scores[target] = max(-1.0, min(1.0, score))

    return GraphOutput(
        date=date,
        symbols=symbols,
        node_features=node_features,
        adjacency=adjacency,
        attention=attention,
        graph_scores=graph_scores,
        metadata={
            "method": "correlation_graph_mvp",
            "score_range": "[-1, 1]",
            "uses_gat": False,
            "uses_future_data": False,
        },
    )
