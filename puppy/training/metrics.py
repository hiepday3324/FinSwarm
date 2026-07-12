from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    symbols: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_all = np.asarray(targets, dtype=np.int64)
    raw_probability = np.asarray(probabilities, dtype=np.float64)
    finite = np.isfinite(raw_probability)
    missing_count = int(np.sum(~finite))
    y = y_all[finite]
    p = np.clip(raw_probability[finite], 1e-7, 1 - 1e-7)
    if not len(y):
        raise ValueError("Metrics require at least one finite prediction")
    if symbols is not None:
        symbols = np.asarray(symbols)[finite]
    predicted = (p >= threshold).astype(np.int64)
    classes, counts = np.unique(y, return_counts=True)
    auc = None
    warning = None
    if len(classes) > 1:
        auc = float(roc_auc_score(y, p))
    else:
        warning = "ROC-AUC undefined because only one target class is present."
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    result: dict[str, Any] = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": expected_calibration_error(y, p),
        "positive_rate": float(np.mean(y)),
        "sample_count": int(len(y)),
        "class_counts": {str(int(key)): int(value) for key, value in zip(classes, counts)},
        "masked_neutral_count": 0,
        "missing_prediction_count": missing_count,
        "warning": warning,
    }
    if symbols is not None:
        result["per_symbol"] = {
            str(symbol): binary_metrics(y[symbols == symbol], p[symbols == symbol])
            for symbol in sorted(set(symbols.tolist()))
        }
    return result


def expected_calibration_error(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    value = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if np.any(mask):
            value += float(np.sum(mask) / total) * abs(float(np.mean(probabilities[mask])) - float(np.mean(targets[mask])))
    return value
