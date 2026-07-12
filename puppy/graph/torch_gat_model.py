from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class DenseSignedGAT(nn.Module):
    """Dense multi-head GAT with weighted, signed target-source adjacency."""

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
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(
            self.input_dim,
            self.hidden_dim * self.num_heads,
            bias=False,
        )
        self.attention_target = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_dim)
        )
        self.attention_source = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_dim)
        )
        self.attention_dropout = nn.Dropout(self.dropout_rate)
        self.output = nn.Linear(self.hidden_dim * self.num_heads, 1)
        self.activation = nn.GELU()
        self.last_attention: torch.Tensor | None = None
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
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        logits = torch.nn.functional.leaky_relu(
            target_score.unsqueeze(-1) + source_score.unsqueeze(-2),
            negative_slope=0.2,
        )

        weighted = adjacency.clone()
        diagonal = torch.arange(nodes, device=adjacency.device)
        weighted[:, diagonal, diagonal] = 1.0
        connectivity = weighted.abs() > 0
        weight_bias = torch.log(weighted.abs().clamp_min(self.epsilon))
        logits = logits + weight_bias.unsqueeze(1)
        logits = logits.masked_fill(~connectivity.unsqueeze(1), -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        attention = self.attention_dropout(attention)
        signed_attention = attention * torch.sign(weighted).unsqueeze(1)
        aggregated = torch.einsum("bhts,bhsd->bhtd", signed_attention, projected)
        combined = self.activation(aggregated).permute(0, 2, 1, 3).reshape(
            batch, nodes, self.num_heads * self.hidden_dim
        )
        logits_out = self.output(combined).squeeze(-1)
        if not torch.isfinite(logits_out).all() or not torch.isfinite(attention).all():
            raise ValueError("GAT produced NaN or infinite output")
        self.last_attention = attention.detach()
        if return_attention:
            return logits_out, attention
        return logits_out

    def architecture_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "epsilon": self.epsilon,
            "architecture": "dense-signed-weighted-multihead-gat-v1",
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
