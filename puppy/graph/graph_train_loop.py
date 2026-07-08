"""Offline graph model training loop."""

from __future__ import annotations

from typing import Any

import numpy as np

from puppy.graph.gat_model import NumpyGATScorer


def train_gat_mvp(
    samples: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[NumpyGATScorer, dict[str, Any]]:
    """Create a deterministic GAT MVP scorer and run forward metrics.

    This is intentionally forward-only. Labels are reported when present but
    the function does not require labels and does not perform deep training.
    """

    if not samples:
        raise ValueError("At least one graph sample is required.")

    first_matrix = np.asarray(samples[0]["feature_matrix"], dtype=np.float64)
    if first_matrix.ndim != 2 or first_matrix.shape[1] <= 0:
        raise ValueError("sample feature_matrix must be 2D with positive input_dim.")
    input_dim = int(first_matrix.shape[1])
    model = NumpyGATScorer(input_dim=input_dim, seed=seed)

    all_scores: list[float] = []
    num_symbols = 0
    has_labels = False
    for sample in samples:
        feature_matrix = np.asarray(sample["feature_matrix"], dtype=np.float64)
        adjacency_matrix = np.asarray(sample["adjacency_matrix"], dtype=np.float64)
        if feature_matrix.ndim != 2 or feature_matrix.shape[1] != input_dim:
            raise ValueError("All graph samples must share the same input_dim.")
        scores = model.forward(feature_matrix, adjacency_matrix)
        all_scores.extend(float(score) for score in scores)
        num_symbols += int(feature_matrix.shape[0])
        has_labels = has_labels or bool(sample.get("labels"))

    score_min = min(all_scores) if all_scores else 0.0
    score_max = max(all_scores) if all_scores else 0.0
    metrics = {
        "num_samples": len(samples),
        "num_symbols": num_symbols,
        "input_dim": input_dim,
        "has_labels": has_labels,
        "score_min": float(score_min),
        "score_max": float(score_max),
    }
    return model, metrics
