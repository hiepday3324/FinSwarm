"""Graph Attention Network MVP module."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


class NumpyGATScorer:
    """Small deterministic CPU-only graph attention scorer.

    The scorer is an inference-only MVP. It masks self attention, aggregates
    neighbor node features, and maps the aggregate to scores in ``[-1, 1]``.
    """

    def __init__(
        self,
        input_dim: int,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.target_weights = rng.normal(0.0, 0.1, size=self.input_dim)
        self.source_weights = rng.normal(0.0, 0.1, size=self.input_dim)
        self.output_weights = rng.normal(0.0, 0.1, size=self.input_dim)
        self.bias = 0.0
        self.last_attention: np.ndarray | None = None

    def forward(self, node_feature_matrix: Any, adjacency: Any) -> np.ndarray:
        features = self._prepare_features(node_feature_matrix)
        adjacency_matrix = self._prepare_adjacency(adjacency, features.shape[0])
        mask = np.abs(adjacency_matrix) > 0.0
        np.fill_diagonal(mask, False)

        attention = np.zeros_like(adjacency_matrix, dtype=np.float64)
        target_logits = features @ self.target_weights
        source_logits = features @ self.source_weights
        for target_idx in range(features.shape[0]):
            source_indices = np.where(mask[target_idx])[0]
            if source_indices.size == 0:
                continue
            logits = np.tanh(
                target_logits[target_idx]
                + source_logits[source_indices]
                + 0.25 * adjacency_matrix[target_idx, source_indices]
            )
            weights = _softmax(logits)
            attention[target_idx, source_indices] = weights

        aggregate = attention @ features
        raw_scores = aggregate @ self.output_weights + self.bias
        scores = np.tanh(raw_scores)
        self.last_attention = attention
        return np.asarray(scores, dtype=np.float64)

    def predict_scores(
        self,
        symbols: list[str],
        node_feature_matrix: Any,
        adjacency: Any,
    ) -> dict[str, float]:
        scores = self.forward(node_feature_matrix, adjacency)
        if len(symbols) != scores.shape[0]:
            raise ValueError("symbols length must match number of graph nodes.")
        return {
            symbol: float(max(-1.0, min(1.0, score)))
            for symbol, score in zip(symbols, scores)
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "input_dim": self.input_dim,
            "seed": self.seed,
            "target_weights": self.target_weights.tolist(),
            "source_weights": self.source_weights.tolist(),
            "output_weights": self.output_weights.tolist(),
            "bias": self.bias,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "NumpyGATScorer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = cls(input_dim=int(payload["input_dim"]), seed=int(payload["seed"]))
        model.target_weights = _load_vector(payload["target_weights"], model.input_dim)
        model.source_weights = _load_vector(payload["source_weights"], model.input_dim)
        model.output_weights = _load_vector(payload["output_weights"], model.input_dim)
        model.bias = float(payload.get("bias", 0.0))
        return model

    def _prepare_features(self, node_feature_matrix: Any) -> np.ndarray:
        features = np.asarray(node_feature_matrix, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("node_feature_matrix must be 2D.")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {features.shape[1]}."
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("node_feature_matrix contains NaN or inf.")
        return features

    @staticmethod
    def _prepare_adjacency(adjacency: Any, num_nodes: int) -> np.ndarray:
        matrix = np.asarray(adjacency, dtype=np.float64)
        if matrix.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"adjacency must have shape {(num_nodes, num_nodes)}, got {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("adjacency contains NaN or inf.")
        return matrix


SimpleGATScorer = NumpyGATScorer


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if total <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return exp_values / total


def _load_vector(values: list[float], expected_dim: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (expected_dim,):
        raise ValueError("Saved NumpyGATScorer vector does not match input_dim.")
    return vector
