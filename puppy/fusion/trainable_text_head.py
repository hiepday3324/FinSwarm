from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TrainableTextHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected text embeddings [batch, {self.input_dim}], got {tuple(embeddings.shape)}"
            )
        logits = self.network(embeddings).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Text head produced NaN or infinite logits")
        return logits

    def architecture_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "architecture": "LayerNorm-Linear-GELU-Dropout-Linear",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainableTextHead":
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
        )
