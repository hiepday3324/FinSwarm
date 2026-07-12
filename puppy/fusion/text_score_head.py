from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class TextScoreHead(nn.Module):
    """Canonical trainable score head over frozen ``H_text`` vectors."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        *,
        seed: int | None = None,
        learning_rate: float = 0.1,
        epochs: int = 200,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.seed = None if seed is None else int(seed)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        def build_network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, 1),
            )

        if self.seed is None:
            self.network = build_network()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.network = build_network()

    def forward(self, h_text: torch.Tensor) -> torch.Tensor:
        if h_text.ndim != 2 or h_text.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected H_text [batch, {self.input_dim}], got {tuple(h_text.shape)}"
            )
        logits = self.network(h_text).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Text score head produced NaN or infinite logits")
        return logits

    def predict_proba(self, h_text: Any) -> np.ndarray | torch.Tensor:
        tensor_input = isinstance(h_text, torch.Tensor)
        device = next(self.parameters()).device
        values = (
            h_text.to(device=device, dtype=torch.float32)
            if tensor_input
            else torch.as_tensor(h_text, dtype=torch.float32, device=device)
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self(values))
        return probabilities if tensor_input else probabilities.cpu().numpy()

    def predict(self, h_text: Any, threshold: float = 0.5) -> np.ndarray:
        probability = self.predict_proba(h_text)
        if isinstance(probability, torch.Tensor):
            probability = probability.detach().cpu().numpy()
        return (probability >= float(threshold)).astype(int)

    def fit(self, h_text: Any, targets: Any) -> "TextScoreHead":
        features = torch.as_tensor(h_text, dtype=torch.float32)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected H_text [batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        labels = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
        if len(features) == 0:
            raise ValueError("Cannot fit TextScoreHead with zero samples")
        if labels.shape[0] != features.shape[0]:
            raise ValueError(f"Expected {features.shape[0]} labels, got {labels.shape[0]}")
        if not bool(torch.all((labels == 0) | (labels == 1))):
            raise ValueError("targets must contain binary labels 0/1")
        device = next(self.parameters()).device
        features = features.to(device)
        labels = labels.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.BCEWithLogitsLoss()
        with torch.random.fork_rng(devices=[]):
            if self.seed is not None:
                torch.manual_seed(self.seed)
            for _ in range(max(self.epochs, 0)):
                self.train()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self(features), labels)
                loss.backward()
                optimizer.step()
        self.eval()
        return self

    def architecture_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "architecture": "canonical-linear-gelu-text-score-head-v2",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TextScoreHead":
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            seed=(
                None
                if config.get("seed") is None
                else int(config["seed"])
            ),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "model_type": "canonical_text_score_head",
                "architecture": self.architecture_config(),
                "state_dict": self.state_dict(),
                "seed": self.seed,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
            },
            temporary,
        )
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "TextScoreHead":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls.from_config(
                {**checkpoint["architecture"], "seed": checkpoint.get("seed")}
        ).to(device)
        model.learning_rate = float(checkpoint.get("learning_rate", model.learning_rate))
        model.epochs = int(checkpoint.get("epochs", model.epochs))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class NumpyLogisticTextScoreBaseline:
    """Explicit milestone-compatible NumPy logistic baseline."""

    def __init__(
        self,
        input_dim: int,
        seed: int = 42,
        learning_rate: float = 0.1,
        epochs: int = 200,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=self.input_dim).astype(np.float64)
        self.bias = 0.0

    def _prepare_x(self, values: Any) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2 or result.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {result.shape}")
        return result

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    def fit(self, values: Any, targets: Any) -> "NumpyLogisticTextScoreBaseline":
        features = self._prepare_x(values)
        labels = np.asarray(targets, dtype=np.float64).reshape(-1)
        if len(labels) != len(features):
            raise ValueError("Feature/target row count mismatch")
        for _ in range(max(self.epochs, 0)):
            error = self._sigmoid(features @ self.weights + self.bias) - labels
            self.weights -= self.learning_rate * (features.T @ error / len(features))
            self.bias -= self.learning_rate * float(np.mean(error))
        return self

    def predict_proba(self, values: Any) -> np.ndarray:
        features = self._prepare_x(values)
        return self._sigmoid(features @ self.weights + self.bias)

    def predict(self, values: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(values) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "input_dim": self.input_dim,
                    "seed": self.seed,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "weights": self.weights.tolist(),
                    "bias": self.bias,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyLogisticTextScoreBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            int(payload["input_dim"]),
            int(payload["seed"]),
            float(payload["learning_rate"]),
            int(payload["epochs"]),
        )
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.bias = float(payload["bias"])
        return model


# Compatibility name for old imports while all production imports move here.
TrainableTextHead = TextScoreHead
