"""Gated fusion over text and graph streams."""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

from puppy.common.schemas import FusionOutput, GraphOutput


class GatedFusion:
    """Deterministic MVP gate over text and graph probabilities."""

    def __init__(
        self,
        base_gate: float = 0.5,
        conflict_threshold: float = 0.35,
    ) -> None:
        self.base_gate = _clamp_probability(base_gate)
        self.conflict_threshold = float(conflict_threshold)

    def fit(self, *_args: Any, **_kwargs: Any) -> "GatedFusion":
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
        gate = self.base_gate + 0.5 * (text_strength - graph_strength)
        return _clamp_probability(gate)

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "base_gate": self.base_gate,
            "conflict_threshold": self.conflict_threshold,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "GatedFusion":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            base_gate=float(payload["base_gate"]),
            conflict_threshold=float(payload["conflict_threshold"]),
        )


def build_fusion_output(
    date: dt.date,
    text_scores: dict[str, float],
    graph_output: GraphOutput,
    metadata: dict[str, Any] | None = None,
    model: GatedFusion | None = None,
    conflict_threshold: float = 0.35,
) -> FusionOutput:
    fusion_model = model or GatedFusion(conflict_threshold=conflict_threshold)
    threshold = fusion_model.conflict_threshold
    symbols = list(graph_output.symbols)
    normalized_text_scores: dict[str, float] = {}
    graph_scores: dict[str, float] = {}
    fusion_scores: dict[str, float] = {}
    gates: dict[str, float] = {}
    conflict_flags: dict[str, bool] = {}

    for symbol in symbols:
        if symbol not in text_scores:
            raise KeyError(f"Missing text score for {symbol}.")
        text_score = _validate_range(text_scores[symbol], 0.0, 1.0, "text_score")
        graph_score = _validate_range(
            graph_output.graph_scores.get(symbol, 0.0),
            -1.0,
            1.0,
            "graph_score",
        )
        graph_prob = (graph_score + 1.0) / 2.0
        gate = fusion_model.gate(text_score=text_score, graph_prob=graph_prob)
        fusion_score = gate * text_score + (1.0 - gate) * graph_prob

        normalized_text_scores[symbol] = text_score
        graph_scores[symbol] = graph_score
        gates[symbol] = gate
        fusion_scores[symbol] = _clamp_probability(fusion_score)
        conflict_flags[symbol] = abs(text_score - graph_prob) >= threshold

    output_metadata = {
        "method": "gated_fusion_mvp",
        "text_score_range": "[0, 1]",
        "graph_score_range": "[-1, 1]",
        "fusion_score_range": "[0, 1]",
        "conflict_threshold": threshold,
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
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {number}.")
    return number


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
