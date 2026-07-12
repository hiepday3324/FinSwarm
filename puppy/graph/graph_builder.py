"""Graph state builder for SectorAgent relationships."""

from __future__ import annotations

import datetime as dt

import numpy as np
import torch

from puppy.common.schemas import GraphOutput, QuantFeatureTable, ShadowPortfolioState, TextFeature
from puppy.graph.correlation_graph import (
    build_attention_from_adjacency,
    build_correlation_adjacency,
)
from puppy.graph.gat_model import DenseSignedGAT, NumpyGATScorer
from puppy.graph.graph_dataset import build_graph_sample
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
            "temporal_contract": "legacy_unverified",
        },
    )


def build_gat_graph_output(
    date: dt.date,
    quant_table: QuantFeatureTable,
    shadow_states: list[ShadowPortfolioState] | None = None,
    text_features: list[TextFeature] | None = None,
    return_history: dict[str, list[float]] | None = None,
    model: NumpyGATScorer | None = None,
) -> GraphOutput:
    """Build a GraphOutput with the CPU-only numpy GAT MVP scorer."""

    sample = build_graph_sample(
        quant_table=quant_table,
        shadow_states=shadow_states,
        text_features=text_features,
        return_history=return_history,
    )
    feature_matrix = sample["feature_matrix"]
    scorer = model or NumpyGATScorer(input_dim=feature_matrix.shape[1])
    graph_scores = scorer.predict_scores(
        symbols=sample["symbols"],
        node_feature_matrix=feature_matrix,
        adjacency=sample["adjacency_matrix"],
    )
    attention = _attention_dict(
        symbols=sample["symbols"],
        attention_matrix=scorer.last_attention,
    )
    return GraphOutput(
        date=date,
        symbols=sample["symbols"],
        node_features=sample["node_features"],
        adjacency=sample["adjacency"],
        attention=attention,
        graph_scores=graph_scores,
        metadata={
            "method": "gat_mvp",
            "uses_gat": True,
            "implementation": "numpy_gat_mvp",
            "temporal_contract": "legacy_unverified",
            "score_range": "[-1, 1]",
            "feature_names": sample["metadata"]["feature_names"],
        },
    )


def build_canonical_gat_output(
    date: dt.date,
    graph_sample: dict[str, object],
    model: DenseSignedGAT,
    *,
    device: str = "cpu",
) -> GraphOutput:
    """Run the production GAT and adapt its structured tensors to GraphOutput."""

    symbols = [str(value) for value in graph_sample["symbols"]]
    feature_names = tuple(graph_sample["metadata"]["feature_names"])
    features = torch.as_tensor(
        np.asarray(graph_sample["feature_matrix"], dtype=np.float32)[None],
        dtype=torch.float32,
        device=device,
    )
    adjacency = torch.as_tensor(
        np.asarray(graph_sample["adjacency_matrix"], dtype=np.float32)[None],
        dtype=torch.float32,
        device=device,
    )
    feature_mask = torch.as_tensor(
        np.asarray(graph_sample["feature_mask"], dtype=bool)[None],
        dtype=torch.bool,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        output = model(
            features,
            adjacency,
            feature_mask,
            symbols=symbols,
            feature_names=feature_names,
            feature_provenance=dict(graph_sample["metadata"]),
        )
    logits = output.logits[0].cpu().numpy()
    probabilities = output.probabilities[0].cpu().numpy()
    embeddings = output.h_graph[0].cpu().numpy()
    alpha = output.alpha[0].cpu().numpy()
    return GraphOutput(
        date=date,
        symbols=symbols,
        node_features=dict(graph_sample["node_features"]),
        adjacency=dict(graph_sample["adjacency"]),
        graph_scores={
            symbol: float(probability * 2.0 - 1.0)
            for symbol, probability in zip(symbols, probabilities)
        },
        graph_logits={symbol: float(value) for symbol, value in zip(symbols, logits)},
        graph_probabilities={
            symbol: float(value) for symbol, value in zip(symbols, probabilities)
        },
        graph_embeddings={
            symbol: value.astype(np.float32).tolist()
            for symbol, value in zip(symbols, embeddings)
        },
        attention=_attention_dict(symbols, alpha),
        feature_names=list(feature_names),
        feature_provenance=dict(graph_sample["metadata"]),
        metadata={
            "method": "canonical_dense_signed_gat",
            "uses_gat": True,
            "production": True,
            "h_graph_dim": int(embeddings.shape[-1]),
        },
    )


def _attention_dict(
    symbols: list[str],
    attention_matrix: object,
) -> dict[str, dict[str, float]]:
    if attention_matrix is None:
        return {target: {source: 0.0 for source in symbols} for target in symbols}
    return {
        target: {
            source: float(attention_matrix[target_index, source_index])
            for source_index, source in enumerate(symbols)
        }
        for target_index, target in enumerate(symbols)
    }
