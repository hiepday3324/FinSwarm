from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from puppy.fusion.trainable_text_head import TrainableTextHead

from .checkpoint import (
    TrainOnlyScaler,
    atomic_json,
    atomic_torch_save,
    seed_everything,
    state_dict_sha256,
)
from .config import TrainingConfig
from .metrics import binary_metrics
from .split import EffectiveSplits
from .text_dataset import TextDataset, load_supervised_text_dataset
from .text_embedding import sha256_path
from .training_guard import SplitAccessGuard


@dataclass
class LogisticBaseline:
    coefficient: np.ndarray
    intercept: float
    seed: int

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, *, seed: int) -> "LogisticBaseline":
        if len(np.unique(targets)) != 2:
            raise ValueError("Logistic baseline requires both target classes")
        estimator = LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=2000,
            solver="lbfgs",
        )
        estimator.fit(features, targets.astype(np.int64))
        return cls(
            coefficient=np.asarray(estimator.coef_[0], dtype=np.float64),
            intercept=float(estimator.intercept_[0]),
            seed=seed,
        )

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.coefficient + self.intercept

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.clip(self.predict_logits(features), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str | Path) -> None:
        atomic_json(
            path,
            {
                "model_type": "logistic_regression",
                "coefficient": self.coefficient.tolist(),
                "intercept": self.intercept,
                "seed": self.seed,
                "class_weight": "balanced",
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticBaseline":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            coefficient=np.asarray(raw["coefficient"], dtype=np.float64),
            intercept=float(raw["intercept"]),
            seed=int(raw["seed"]),
        )


def _positive_weight(targets: np.ndarray) -> float:
    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training requires both positive and negative classes")
    return negatives / positives


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def predict_mlp(
    model: TrainableTextHead,
    features: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            tensor = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            values.append(model(tensor).cpu().numpy())
    logits = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    return logits, _sigmoid(logits)


def fit_mlp(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    model_config: dict[str, Any],
    *,
    seed: int,
    device: str = "cpu",
    fixed_epochs: int | None = None,
) -> tuple[TrainableTextHead, dict[str, Any]]:
    seed_everything(seed)
    model = TrainableTextHead(
        input_dim=train_features.shape[1],
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(_positive_weight(train_targets), device=device)
    )
    max_epochs = fixed_epochs or int(model_config["max_epochs"])
    patience = int(model_config["patience"])
    batch_size = int(model_config["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    curves: list[dict[str, float | int]] = []
    x_train = torch.as_tensor(train_features, dtype=torch.float32)
    y_train = torch.as_tensor(train_targets, dtype=torch.float32)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = x_train[indices].to(device)
            batch_y = y_train[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        _, validation_probability = predict_mlp(model, validation_features, device=device)
        validation_loss = float(
            binary_metrics(validation_targets, validation_probability)["log_loss"]
        )
        curves.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_targets),
                "validation_log_loss": validation_loss,
            }
        )
        if fixed_epochs is not None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_loss = validation_loss
            continue
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training produced no checkpoint")
    model.load_state_dict(best_state)
    restored_hash = state_dict_sha256(model.state_dict())
    return model, {
        "learning_curve": curves,
        "best_epoch": best_epoch,
        "best_validation_log_loss": best_loss,
        "parameter_changed": initial_hash != restored_hash,
        "initial_state_sha256": initial_hash,
        "best_state_sha256": restored_hash,
        "best_checkpoint_restored": restored_hash == state_dict_sha256(best_state),
        "fixed_epochs": fixed_epochs,
    }


def train_text_stream(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    guard: SplitAccessGuard | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    guard = guard or SplitAccessGuard()
    seed = config.seed if seed is None else int(seed)
    guard.assert_allowed("train", "scaler_fit")
    guard.assert_allowed("validation_selection", "early_stopping")
    dataset = load_supervised_text_dataset(config, splits, guard=guard)
    train = dataset.subset(splits.train_effective)
    selection = dataset.subset(splits.validation_selection)
    if not len(train) or not len(selection):
        raise ValueError("Text train or validation-selection dataset is empty")

    scaler = TrainOnlyScaler([f"embedding_{index:03d}" for index in range(384)])
    scaler.fit(
        train.embeddings,
        dates=train.dates.tolist(),
        input_manifest_hash=sha256_path(config.artifact_dir / "data_manifest.json"),
    )
    train_features = scaler.transform(train.embeddings)
    selection_features = scaler.transform(selection.embeddings)
    model_dir = config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(model_dir / "text_scaler.json")

    logistic = LogisticBaseline.fit(train_features, train.targets, seed=seed)
    logistic.save(model_dir / "text_logistic_baseline.json")
    logistic_train = binary_metrics(
        train.targets,
        logistic.predict_proba(train_features),
        symbols=train.symbols,
    )
    logistic_selection = binary_metrics(
        selection.targets,
        logistic.predict_proba(selection_features),
        symbols=selection.symbols,
    )

    mlp, mlp_history = fit_mlp(
        train_features,
        train.targets,
        selection_features,
        selection.targets,
        config.text_model,
        seed=seed,
        device=device,
    )
    mlp_train_logits, mlp_train_probability = predict_mlp(
        mlp, train_features, device=device
    )
    mlp_selection_logits, mlp_selection_probability = predict_mlp(
        mlp, selection_features, device=device
    )
    mlp_train = binary_metrics(train.targets, mlp_train_probability, symbols=train.symbols)
    mlp_selection = binary_metrics(
        selection.targets, mlp_selection_probability, symbols=selection.symbols
    )
    selected_model = (
        "mlp"
        if mlp_selection["log_loss"] < logistic_selection["log_loss"]
        else "logistic_regression"
    )
    checkpoint = {
        "model_type": "mlp",
        "architecture": mlp.architecture_config(),
        "state_dict": mlp.state_dict(),
        "seed": seed,
        "selected": selected_model == "mlp",
        "best_epoch": mlp_history["best_epoch"],
        "validation_prediction_probe": {
            "logit": float(mlp_selection_logits[0]),
            "probability": float(mlp_selection_probability[0]),
        },
    }
    atomic_torch_save(model_dir / "text_model.pt", checkpoint)
    loaded = load_text_mlp(model_dir / "text_model.pt", device=device)
    loaded_logits, _ = predict_mlp(loaded, selection_features[:4], device=device)
    save_load_equivalent = bool(
        np.allclose(loaded_logits, mlp_selection_logits[:4], atol=1e-7, rtol=1e-6)
    )
    if not mlp_history["parameter_changed"]:
        raise RuntimeError("MLP parameters did not change during training")
    if not mlp_history["best_checkpoint_restored"]:
        raise RuntimeError("MLP best validation checkpoint was not restored")
    if not save_load_equivalent:
        raise RuntimeError("MLP save/load prediction mismatch")

    history = {
        "selected_text_model": selected_model,
        "train_range": [min(train.dates).isoformat(), max(train.dates).isoformat()],
        "validation_selection_range": [
            min(selection.dates).isoformat(),
            max(selection.dates).isoformat(),
        ],
        "train_rows": len(train),
        "validation_selection_rows": len(selection),
        "logistic_regression": {
            "train_metrics": logistic_train,
            "validation_selection_metrics": logistic_selection,
        },
        "mlp": {
            **mlp_history,
            "train_metrics": mlp_train,
            "validation_selection_metrics": mlp_selection,
            "save_load_prediction_equivalent": save_load_equivalent,
        },
        "dataset_filters": dataset.metadata,
    }
    atomic_json(model_dir / "text_training_history.json", history)
    return history


def load_text_mlp(path: str | Path, *, device: str = "cpu") -> TrainableTextHead:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = TrainableTextHead.from_config(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def selected_text_predictions(
    model_dir: str | Path,
    features: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, str]:
    directory = Path(model_dir)
    history = json.loads((directory / "text_training_history.json").read_text(encoding="utf-8"))
    selected = history["selected_text_model"]
    if selected == "mlp":
        logits, probability = predict_mlp(
            load_text_mlp(directory / "text_model.pt", device=device),
            features,
            device=device,
        )
    else:
        model = LogisticBaseline.load(directory / "text_logistic_baseline.json")
        logits = model.predict_logits(features)
        probability = model.predict_proba(features)
    return logits, probability, selected
