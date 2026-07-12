from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TrainableGatedFusion(nn.Module):
    def __init__(self, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.gate_network = nn.Sequential(
            nn.Linear(6, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )
        self.learned_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def gate_features(
        text_logit: torch.Tensor,
        graph_logit: torch.Tensor,
        text_available: torch.Tensor,
        graph_available: torch.Tensor,
    ) -> torch.Tensor:
        text_value = torch.where(text_available, text_logit, torch.zeros_like(text_logit))
        graph_value = torch.where(graph_available, graph_logit, torch.zeros_like(graph_logit))
        conflict = ((text_value >= 0) != (graph_value >= 0)).to(text_value.dtype)
        both = text_available & graph_available
        conflict = conflict * both.to(conflict.dtype)
        return torch.stack(
            (
                text_value,
                graph_value,
                torch.abs(text_value - graph_value),
                conflict,
                (~text_available).to(text_value.dtype),
                (~graph_available).to(text_value.dtype),
            ),
            dim=-1,
        )

    def forward(
        self,
        text_logit: torch.Tensor,
        graph_logit: torch.Tensor,
        text_available: torch.Tensor | None = None,
        graph_available: torch.Tensor | None = None,
        *,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if text_logit.shape != graph_logit.shape:
            raise ValueError("Text and graph logits must have the same shape")
        text_available = (
            torch.isfinite(text_logit) if text_available is None else text_available.bool()
        )
        graph_available = (
            torch.isfinite(graph_logit) if graph_available is None else graph_available.bool()
        )
        features = self.gate_features(
            text_logit,
            graph_logit,
            text_available,
            graph_available,
        )
        gate = torch.sigmoid(self.gate_network(features).squeeze(-1))
        both = text_available & graph_available
        text_only = text_available & ~graph_available
        graph_only = graph_available & ~text_available
        fused = torch.ones_like(text_logit) * self.learned_bias
        fused = torch.where(
            both,
            gate * text_logit + (1.0 - gate) * graph_logit + self.learned_bias,
            fused,
        )
        fused = torch.where(text_only, text_logit, fused)
        fused = torch.where(graph_only, graph_logit, fused)
        gate = torch.where(text_only, torch.ones_like(gate), gate)
        gate = torch.where(graph_only, torch.zeros_like(gate), gate)
        if not torch.isfinite(fused).all() or not torch.isfinite(gate).all():
            raise ValueError("Gated fusion produced NaN or infinite output")
        if return_gate:
            return fused, gate
        return fused

    def architecture_config(self) -> dict[str, Any]:
        return {
            "input_dim": 6,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "architecture": "gated-logit-fusion-v1",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainableGatedFusion":
        return cls(hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]))
