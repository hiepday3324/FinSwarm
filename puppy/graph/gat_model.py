"""Graph Attention Network MVP module."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GATModelOutput:
    """Structured canonical GAT output for runtime and training."""

    h_graph: torch.Tensor
    alpha: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> torch.Size:
        return self.logits.shape


class DenseSignedGAT(nn.Module):
    """Canonical dense signed multi-head GAT."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.epsilon = float(epsilon)
        self.output_dim = self.hidden_dim * self.num_heads
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.attention_target = nn.Parameter(torch.empty(self.num_heads, self.hidden_dim))
        self.attention_source = nn.Parameter(torch.empty(self.num_heads, self.hidden_dim))
        self.attention_dropout = nn.Dropout(self.dropout_rate)
        self.output = nn.Linear(self.output_dim, 1)
        self.activation = nn.GELU()
        self.last_attention: torch.Tensor | None = None
        self.last_attention_heads: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.attention_source)
        self.output.reset_parameters()

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        feature_mask: torch.Tensor,
        *,
        symbols: tuple[str, ...] | list[str] | None = None,
        feature_names: tuple[str, ...] | list[str] | None = None,
        feature_provenance: dict[str, Any] | None = None,
        return_attention: bool = False,
    ) -> GATModelOutput | tuple[torch.Tensor, torch.Tensor]:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, features]")
        batch, nodes, features = node_features.shape
        if features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} graph features, got {features}")
        if adjacency.shape != (batch, nodes, nodes):
            raise ValueError("adjacency must have shape [batch, nodes, nodes]")
        if feature_mask.shape != node_features.shape:
            raise ValueError("feature_mask must match node_features shape")
        if not torch.isfinite(node_features).all() or not torch.isfinite(adjacency).all():
            raise ValueError("GAT inputs contain NaN or infinite values")

        values = torch.where(feature_mask.bool(), node_features, torch.zeros_like(node_features))
        values = self.input_norm(values)
        projected = self.projection(values).view(
            batch, nodes, self.num_heads, self.hidden_dim
        )
        projected = projected.permute(0, 2, 1, 3)

        target_score = (projected * self.attention_target[None, :, None, :]).sum(-1)
        source_score = (projected * self.attention_source[None, :, None, :]).sum(-1)
        attention_logits = torch.nn.functional.leaky_relu(
            target_score.unsqueeze(-1) + source_score.unsqueeze(-2),
            negative_slope=0.2,
        )
        weighted = adjacency.clone()
        diagonal = torch.arange(nodes, device=adjacency.device)
        weighted[:, diagonal, diagonal] = 1.0
        connectivity = weighted.abs() > 0
        weight_bias = torch.log(weighted.abs().clamp_min(self.epsilon))
        attention_logits = attention_logits + weight_bias.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(
            ~connectivity.unsqueeze(1), -torch.inf
        )
        attention_heads = torch.softmax(attention_logits, dim=-1)
        dropped_attention = self.attention_dropout(attention_heads)
        signed_attention = dropped_attention * torch.sign(weighted).unsqueeze(1)
        aggregated = torch.einsum("bhts,bhsd->bhtd", signed_attention, projected)
        h_graph = self.activation(aggregated).permute(0, 2, 1, 3).reshape(
            batch, nodes, self.output_dim
        )
        logits = self.output(h_graph).squeeze(-1)
        alpha = attention_heads.mean(dim=1)
        probabilities = torch.sigmoid(logits)
        if not all(
            torch.isfinite(value).all()
            for value in (h_graph, alpha, logits, probabilities)
        ):
            raise ValueError("GAT produced NaN or infinite output")
        self.last_attention = alpha.detach()
        self.last_attention_heads = attention_heads.detach()
        symbol_order = tuple(symbols or (f"node_{index}" for index in range(nodes)))
        if len(symbol_order) != nodes:
            raise ValueError("symbols length must match graph node count")
        names = tuple(feature_names or (f"feature_{index}" for index in range(features)))
        if len(names) != features:
            raise ValueError("feature_names length must match graph input dimension")
        result = GATModelOutput(
            h_graph=h_graph,
            alpha=alpha,
            logits=logits,
            probabilities=probabilities,
            symbols=symbol_order,
            feature_names=names,
            feature_provenance=dict(feature_provenance or {}),
        )
        if return_attention:
            return logits, attention_heads
        return result

    def architecture_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "epsilon": self.epsilon,
            "output_dim": self.output_dim,
            "architecture": "canonical-dense-signed-weighted-multihead-gat-v1",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DenseSignedGAT":
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
            epsilon=float(config.get("epsilon", 1e-8)),
        )


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
