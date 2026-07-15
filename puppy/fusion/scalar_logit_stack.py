"""Strict scalar-logit fusion heads for Stage 6 ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ScalarLogitStack(nn.Module):
    """Regularized affine stack over canonical Text and Graph logits."""

    def __init__(self) -> None:
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.weights = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(
        self,
        text_logits: torch.Tensor,
        graph_logits: torch.Tensor,
    ) -> torch.Tensor:
        if text_logits.shape != graph_logits.shape:
            raise ValueError("Text and Graph logits must have identical shapes")
        if text_logits.ndim != 1:
            raise ValueError("Scalar logit stack expects one-dimensional logit arrays")
        logits = (
            self.intercept
            + self.weights[0] * text_logits
            + self.weights[1] * graph_logits
        )
        if not torch.isfinite(logits).all():
            raise ValueError("Scalar logit stack produced non-finite logits")
        return logits

    def architecture_config(self) -> dict[str, Any]:
        return {"architecture": "scalar-logit-stack-v1", "input_dim": 2}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ScalarLogitStack":
        if config != {"architecture": "scalar-logit-stack-v1", "input_dim": 2}:
            raise ValueError(f"Unsupported scalar stack architecture: {config}")
        return cls()


class GraphOnlyFusion(nn.Module):
    """Parameter-free Stage 6 head that selects the Graph logit exactly."""

    def forward(
        self,
        text_logits: torch.Tensor,
        graph_logits: torch.Tensor,
    ) -> torch.Tensor:
        if text_logits.shape != graph_logits.shape:
            raise ValueError("Text and Graph logits must have identical shapes")
        if graph_logits.ndim != 1:
            raise ValueError("Graph-only fusion expects one-dimensional logit arrays")
        return graph_logits

    def architecture_config(self) -> dict[str, Any]:
        return {"architecture": "graph-only-fusion-v1", "input_dim": 2}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GraphOnlyFusion":
        if config != {"architecture": "graph-only-fusion-v1", "input_dim": 2}:
            raise ValueError(f"Unsupported graph-only architecture: {config}")
        return cls()
