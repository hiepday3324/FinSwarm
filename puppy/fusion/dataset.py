from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from puppy.common.schemas import QuantFeatureTable, TextFeature


def _date_key(value: dt.date | str) -> str:
    return value.isoformat() if isinstance(value, dt.date) else str(value)


def _binary_label(label: int | float | None) -> int | None:
    if label is None:
        return None
    return 1 if float(label) == 1.0 else 0


def build_text_score_dataset(
    text_features: list[TextFeature],
    quant_tables: list[QuantFeatureTable],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Join TextFeature rows to QuantFeatureTable labels by (date, symbol)."""
    expected_dim: int | None = None
    for feature in text_features:
        dim = len(feature.h_text)
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            raise ValueError("All TextFeature.h_text vectors must have the same dimension.")
        if feature.dim != dim:
            raise ValueError(f"TextFeature dim mismatch for {feature.symbol} on {feature.date}.")

    label_by_key: dict[tuple[str, str], int] = {}
    for table in quant_tables:
        date_key = _date_key(table.date)
        for symbol, raw_label in table.labels.items():
            label = _binary_label(raw_label)
            if label is not None:
                label_by_key[(date_key, symbol)] = label

    rows: list[list[float]] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for feature in text_features:
        key = (_date_key(feature.date), feature.symbol)
        if key not in label_by_key:
            continue
        label = label_by_key[key]
        rows.append([float(value) for value in feature.h_text])
        labels.append(label)
        metadata.append(
            {
                "date": key[0],
                "symbol": feature.symbol,
                "agent_id": feature.agent_id,
                "label": label,
            }
        )

    input_dim = expected_dim or 0
    if rows:
        x = np.asarray(rows, dtype=np.float32)
    else:
        x = np.empty((0, input_dim), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, metadata
