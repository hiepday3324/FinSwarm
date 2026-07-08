from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from puppy.common.schemas import TextFeature


class TextFeatureContractError(ValueError):
    """Raised when TextFeature artifacts are not stable enough for fusion."""


def validate_text_feature(
    feature: TextFeature,
    expected_dim: int | None = None,
    expected_dtype: str = "float32",
) -> TextFeature:
    vector = np.asarray(feature.h_text, dtype=np.float32)
    if vector.ndim != 1:
        raise TextFeatureContractError(
            f"h_text for {feature.symbol} on {feature.date} must be a 1D vector."
        )
    if vector.size == 0:
        raise TextFeatureContractError(
            f"h_text for {feature.symbol} on {feature.date} must not be empty."
        )
    if feature.dim != int(vector.size):
        raise TextFeatureContractError(
            f"TextFeature dim mismatch for {feature.symbol} on {feature.date}: "
            f"dim={feature.dim}, len(h_text)={vector.size}."
        )
    if expected_dim is not None and feature.dim != int(expected_dim):
        raise TextFeatureContractError(
            f"Expected h_text dim={expected_dim}, got {feature.dim} for "
            f"{feature.symbol} on {feature.date}."
        )
    if feature.dtype != expected_dtype:
        raise TextFeatureContractError(
            f"Expected dtype={expected_dtype}, got {feature.dtype} for "
            f"{feature.symbol} on {feature.date}."
        )
    if not np.all(np.isfinite(vector)):
        raise TextFeatureContractError(
            f"h_text for {feature.symbol} on {feature.date} contains NaN or inf."
        )
    if not feature.raw_context:
        raise TextFeatureContractError(
            f"raw_context for {feature.symbol} on {feature.date} must not be empty."
        )
    if not feature.embedding_model:
        raise TextFeatureContractError(
            f"embedding_model for {feature.symbol} on {feature.date} is required."
        )
    return feature


def validate_text_feature_batch(
    features: list[TextFeature],
    expected_dim: int | None = None,
    require_same_model: bool = True,
) -> dict[str, Any]:
    if not features:
        raise TextFeatureContractError("At least one TextFeature is required.")

    dim = expected_dim if expected_dim is not None else features[0].dim
    model = features[0].embedding_model
    dtype = features[0].dtype
    for feature in features:
        validate_text_feature(feature, expected_dim=dim, expected_dtype=dtype)
        if require_same_model and feature.embedding_model != model:
            raise TextFeatureContractError(
                "All TextFeature artifacts in a fusion batch must use the same "
                f"embedding_model. Expected {model}, got {feature.embedding_model}."
            )

    return {
        "num_features": len(features),
        "dim": int(dim),
        "dtype": dtype,
        "embedding_model": model if require_same_model else None,
        "symbols": sorted({feature.symbol for feature in features}),
        "dates": sorted(_date_key(feature.date) for feature in features),
    }


def build_text_feature_matrix(
    features: list[TextFeature],
    sort: bool = True,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    validate_text_feature_batch(features)
    ordered = sorted(features, key=lambda item: (_date_key(item.date), item.symbol)) if sort else features
    matrix = np.asarray([feature.h_text for feature in ordered], dtype=np.float32)
    metadata = [
        {
            "date": _date_key(feature.date),
            "symbol": feature.symbol,
            "agent_id": feature.agent_id,
            "embedding_model": feature.embedding_model,
        }
        for feature in ordered
    ]
    return matrix, metadata


def _date_key(value: dt.date | str) -> str:
    return value.isoformat() if isinstance(value, dt.date) else str(value)
