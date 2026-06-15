from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


class TextScoreHead:
    """Small CPU-only logistic head over h_text vectors."""

    def __init__(
        self,
        input_dim: int,
        seed: int = 42,
        learning_rate: float = 0.1,
        epochs: int = 200,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=self.input_dim).astype(np.float64)
        self.bias = 0.0

    def _prepare_x(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("x must be a vector or a 2D batch of vectors.")
        if arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {arr.shape[1]}.")
        return arr

    @staticmethod
    def _prepare_y(y: Any, n_samples: int) -> np.ndarray:
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_samples:
            raise ValueError(f"Expected {n_samples} labels, got {arr.shape[0]}.")
        if not np.all((arr == 0.0) | (arr == 1.0)):
            raise ValueError("y must contain binary labels 0/1.")
        return arr

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def fit(self, x: Any, y: Any) -> "TextScoreHead":
        x_arr = self._prepare_x(x)
        y_arr = self._prepare_y(y, x_arr.shape[0])
        if x_arr.shape[0] == 0:
            raise ValueError("Cannot fit TextScoreHead with zero samples.")

        for _epoch in range(max(0, self.epochs)):
            proba = self._sigmoid(x_arr @ self.weights + self.bias)
            error = proba - y_arr
            grad_w = x_arr.T @ error / x_arr.shape[0]
            grad_b = float(np.mean(error))
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        x_arr = self._prepare_x(x)
        return self._sigmoid(x_arr @ self.weights + self.bias)

    def predict(self, x: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= float(threshold)).astype(int)

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "input_dim": self.input_dim,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "TextScoreHead":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = cls(
            input_dim=int(payload["input_dim"]),
            seed=int(payload["seed"]),
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
        )
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.bias = float(payload["bias"])
        if model.weights.shape != (model.input_dim,):
            raise ValueError("Saved TextScoreHead weights do not match input_dim.")
        return model
