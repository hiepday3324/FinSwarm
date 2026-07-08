"""Fusion and text-head loss functions."""

from __future__ import annotations

from typing import Any

import numpy as np


def binary_cross_entropy(
    y_true: Any,
    y_pred: Any,
    eps: float = 1e-12,
) -> float:
    labels = _prepare_1d(y_true, "y_true")
    predictions = np.clip(_prepare_1d(y_pred, "y_pred"), eps, 1.0 - eps)
    if labels.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("y_true must contain binary labels 0/1.")
    loss = -np.mean(
        labels * np.log(predictions) + (1.0 - labels) * np.log(1.0 - predictions)
    )
    return float(loss)


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    labels = _prepare_1d(y_true, "y_true")
    predictions = _prepare_1d(y_pred, "y_pred")
    if labels.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean((labels - predictions) ** 2))


def calibration_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    labels = _prepare_1d(y_true, "y_true")
    predictions = _prepare_1d(y_pred, "y_pred")
    if labels.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return {
        "num_samples": int(labels.shape[0]),
        "bce": binary_cross_entropy(labels, predictions),
        "mse": mean_squared_error(labels, predictions),
        "avg_label": float(np.mean(labels)) if labels.size else 0.0,
        "avg_prediction": float(np.mean(predictions)) if predictions.size else 0.0,
    }


def _prepare_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf.")
    return array
