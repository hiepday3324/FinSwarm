from __future__ import annotations

import numpy as np

from puppy.common.schemas import QuantFeatureTable, TextFeature

from .dataset import build_text_score_dataset
from .text_score_head import TextScoreHead


def train_text_score_head(
    text_features: list[TextFeature],
    quant_tables: list[QuantFeatureTable],
    input_dim: int | None = None,
    seed: int = 42,
) -> tuple[TextScoreHead, dict]:
    x, y, _metadata = build_text_score_dataset(text_features, quant_tables)
    if x.shape[0] < 2:
        raise ValueError("At least two labeled text samples are required to train TextScoreHead.")
    inferred_dim = int(x.shape[1])
    if input_dim is not None and int(input_dim) != inferred_dim:
        raise ValueError(f"input_dim={input_dim} does not match dataset dim={inferred_dim}.")

    model = TextScoreHead(input_dim=inferred_dim, seed=seed)
    model.fit(x, y)
    predictions = model.predict(x)
    accuracy = float(np.mean(predictions == y))
    metrics = {
        "num_samples": int(x.shape[0]),
        "input_dim": inferred_dim,
        "accuracy": accuracy,
        "positive_rate": float(np.mean(y)),
    }
    return model, metrics
