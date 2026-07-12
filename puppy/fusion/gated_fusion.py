"""Canonical latent fusion plus an explicit legacy scalar baseline."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from puppy.common.schemas import FusionOutput, GraphOutput


@dataclass(frozen=True)
class LatentFusionResult:
    h_text_projected: torch.Tensor
    h_graph_projected: torch.Tensor
    gate: torch.Tensor
    h_final: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    conflict_flags: torch.Tensor | None = None


class LatentGatedFusion(nn.Module):
    """Authoritative vector-gated fusion of H_text and H_graph."""

    def __init__(
        self,
        text_dim: int = 384,
        graph_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(int(text_dim), int(graph_dim), int(latent_dim)) <= 0:
            raise ValueError("text_dim, graph_dim and latent_dim must be positive")
        self.text_dim = int(text_dim)
        self.graph_dim = int(graph_dim)
        self.latent_dim = int(latent_dim)
        self.dropout_rate = float(dropout)
        self.text_projection = nn.Linear(self.text_dim, self.latent_dim)
        self.graph_projection = nn.Linear(self.graph_dim, self.latent_dim)
        self.gate_projection = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.output_projection = nn.Linear(self.latent_dim, 1)

    def forward(
        self,
        h_text: torch.Tensor,
        h_graph: torch.Tensor,
        *,
        text_scores: torch.Tensor | None = None,
        graph_scores: torch.Tensor | None = None,
    ) -> LatentFusionResult:
        if h_text.ndim < 2 or h_text.shape[-1] != self.text_dim:
            raise ValueError(
                f"Expected H_text [..., {self.text_dim}], got {tuple(h_text.shape)}"
            )
        if h_graph.ndim < 2 or h_graph.shape[-1] != self.graph_dim:
            raise ValueError(
                f"Expected H_graph [..., {self.graph_dim}], got {tuple(h_graph.shape)}"
            )
        if h_text.shape[:-1] != h_graph.shape[:-1]:
            raise ValueError("H_text and H_graph must share leading dimensions")
        h_text_projected = self.text_projection(h_text)
        h_graph_projected = self.graph_projection(h_graph)
        gate = torch.sigmoid(
            self.gate_projection(
                torch.cat((h_text_projected, h_graph_projected), dim=-1)
            )
        )
        h_final = gate * h_text_projected + (1.0 - gate) * h_graph_projected
        logits = self.output_projection(h_final).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        conflict = None
        if text_scores is not None or graph_scores is not None:
            if text_scores is None or graph_scores is None:
                raise ValueError("Both text_scores and graph_scores are required for conflict")
            if text_scores.shape != logits.shape or graph_scores.shape != logits.shape:
                raise ValueError("Conflict scores must match fusion output shape")
            conflict = self.compute_conflict(text_scores, graph_scores)
        if not all(
            torch.isfinite(value).all()
            for value in (
                h_text_projected,
                h_graph_projected,
                gate,
                h_final,
                logits,
                probabilities,
            )
        ):
            raise ValueError("Latent fusion produced NaN or infinite output")
        return LatentFusionResult(
            h_text_projected=h_text_projected,
            h_graph_projected=h_graph_projected,
            gate=gate,
            h_final=h_final,
            logits=logits,
            probabilities=probabilities,
            conflict_flags=conflict,
        )

    @staticmethod
    def compute_conflict(
        text_scores: torch.Tensor,
        graph_scores: torch.Tensor,
    ) -> torch.Tensor:
        if text_scores.shape != graph_scores.shape:
            raise ValueError("Text and graph scores must have the same shape")
        return torch.sign(text_scores - 0.5) != torch.sign(graph_scores - 0.5)

    def predict_proba(self, h_text: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(h_text, h_graph).probabilities

    def architecture_config(self) -> dict[str, Any]:
        return {
            "text_dim": self.text_dim,
            "graph_dim": self.graph_dim,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout_rate,
            "architecture": "canonical-latent-gated-fusion-v1",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LatentGatedFusion":
        return cls(
            text_dim=int(config["text_dim"]),
            graph_dim=int(config["graph_dim"]),
            latent_dim=int(config["latent_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "model_type": "canonical_latent_gated_fusion",
                "architecture": self.architecture_config(),
                "state_dict": self.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "LatentGatedFusion":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls.from_config(checkpoint["architecture"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class LegacyScalarGatedFusion:
    """Deterministic milestone baseline over two scalar probabilities."""

    def __init__(self, base_gate: float = 0.5, conflict_threshold: float = 0.35) -> None:
        self.base_gate = _clamp_probability(base_gate)
        self.conflict_threshold = float(conflict_threshold)

    def fit(self, *_args: Any, **_kwargs: Any) -> "LegacyScalarGatedFusion":
        return self

    def predict(
        self,
        date: dt.date,
        text_scores: dict[str, float],
        graph_output: GraphOutput,
        metadata: dict[str, Any] | None = None,
    ) -> FusionOutput:
        return build_fusion_output(
            date=date,
            text_scores=text_scores,
            graph_output=graph_output,
            metadata=metadata,
            model=self,
            conflict_threshold=self.conflict_threshold,
        )

    def gate(self, text_score: float, graph_prob: float) -> float:
        text_strength = abs(text_score - 0.5)
        graph_strength = abs(graph_prob - 0.5)
        return _clamp_probability(
            self.base_gate + 0.5 * (text_strength - graph_strength)
        )

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_gate": self.base_gate,
                    "conflict_threshold": self.conflict_threshold,
                },
                handle,
                sort_keys=True,
            )

    @classmethod
    def load(cls, path: str) -> "LegacyScalarGatedFusion":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            base_gate=float(payload["base_gate"]),
            conflict_threshold=float(payload["conflict_threshold"]),
        )


# Milestone compatibility. Production code imports LatentGatedFusion explicitly.
GatedFusion = LegacyScalarGatedFusion


def build_latent_fusion_output(
    date: dt.date,
    symbols: list[str] | tuple[str, ...],
    h_text: Any,
    h_graph: Any,
    text_scores: dict[str, float],
    graph_probabilities: dict[str, float],
    model: LatentGatedFusion,
    *,
    device: str = "cpu",
    metadata: dict[str, Any] | None = None,
) -> FusionOutput:
    """Adapt canonical latent tensors to the shared runtime FusionOutput."""

    symbol_order = list(symbols)
    text_tensor = torch.as_tensor(h_text, dtype=torch.float32, device=device)
    graph_tensor = torch.as_tensor(h_graph, dtype=torch.float32, device=device)
    text_probability = torch.as_tensor(
        [text_scores[symbol] for symbol in symbol_order],
        dtype=torch.float32,
        device=device,
    )
    graph_probability = torch.as_tensor(
        [graph_probabilities[symbol] for symbol in symbol_order],
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        result = model(
            text_tensor,
            graph_tensor,
            text_scores=text_probability,
            graph_scores=graph_probability,
        )
    text_projected = result.h_text_projected.cpu().numpy()
    graph_projected = result.h_graph_projected.cpu().numpy()
    gate = result.gate.cpu().numpy()
    h_final = result.h_final.cpu().numpy()
    logits = result.logits.cpu().numpy()
    probabilities = result.probabilities.cpu().numpy()
    conflicts = result.conflict_flags.cpu().numpy()
    output_metadata = {
        "method": "canonical_latent_gated_fusion",
        "production": True,
        "latent_dim": model.latent_dim,
        "equation": "Z*H_text_projected+(1-Z)*H_graph_projected",
    }
    output_metadata.update(metadata or {})
    return FusionOutput(
        date=date,
        symbols=symbol_order,
        text_scores={symbol: float(value) for symbol, value in zip(symbol_order, text_probability.cpu())},
        graph_scores={
            symbol: float(value * 2.0 - 1.0)
            for symbol, value in zip(symbol_order, graph_probability.cpu())
        },
        fusion_scores={
            symbol: float(value) for symbol, value in zip(symbol_order, probabilities)
        },
        gates={symbol: float(value.mean()) for symbol, value in zip(symbol_order, gate)},
        vector_gates={symbol: value.tolist() for symbol, value in zip(symbol_order, gate)},
        text_projected={
            symbol: value.tolist() for symbol, value in zip(symbol_order, text_projected)
        },
        graph_projected={
            symbol: value.tolist() for symbol, value in zip(symbol_order, graph_projected)
        },
        final_embeddings={
            symbol: value.tolist() for symbol, value in zip(symbol_order, h_final)
        },
        final_logits={symbol: float(value) for symbol, value in zip(symbol_order, logits)},
        conflict_flags={
            symbol: bool(value) for symbol, value in zip(symbol_order, conflicts)
        },
        metadata=output_metadata,
    )


def build_fusion_output(
    date: dt.date,
    text_scores: dict[str, float],
    graph_output: GraphOutput,
    metadata: dict[str, Any] | None = None,
    model: LegacyScalarGatedFusion | None = None,
    conflict_threshold: float = 0.35,
) -> FusionOutput:
    fusion_model = model or LegacyScalarGatedFusion(
        conflict_threshold=conflict_threshold
    )
    symbols = list(graph_output.symbols)
    normalized_text_scores: dict[str, float] = {}
    graph_scores: dict[str, float] = {}
    fusion_scores: dict[str, float] = {}
    gates: dict[str, float] = {}
    conflict_flags: dict[str, bool] = {}
    for symbol in symbols:
        if symbol not in text_scores:
            raise KeyError(f"Missing text score for {symbol}")
        text_score = _validate_range(text_scores[symbol], 0.0, 1.0, "text_score")
        graph_score = _validate_range(
            graph_output.graph_scores.get(symbol, 0.0), -1.0, 1.0, "graph_score"
        )
        graph_prob = (graph_score + 1.0) / 2.0
        gate = fusion_model.gate(text_score, graph_prob)
        normalized_text_scores[symbol] = text_score
        graph_scores[symbol] = graph_score
        gates[symbol] = gate
        fusion_scores[symbol] = _clamp_probability(
            gate * text_score + (1.0 - gate) * graph_prob
        )
        conflict_flags[symbol] = bool(
            (text_score - 0.5 > 0) != (graph_prob - 0.5 > 0)
        )
    output_metadata = {
        "method": "gated_fusion_mvp",
        "production": False,
        "implementation": "legacy_scalar_gated_fusion",
        "conflict_threshold": fusion_model.conflict_threshold,
    }
    output_metadata.update(metadata or {})
    return FusionOutput(
        date=date,
        symbols=symbols,
        text_scores=normalized_text_scores,
        graph_scores=graph_scores,
        fusion_scores=fusion_scores,
        gates=gates,
        conflict_flags=conflict_flags,
        metadata=output_metadata,
    )


def _validate_range(value: float, lower: float, upper: float, name: str) -> float:
    number = float(value)
    if number < lower or number > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {number}")
    return number


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
