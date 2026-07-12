from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puppy.graph.torch_gat_model import DenseSignedGAT

from .checkpoint import (
    TrainOnlyScaler,
    atomic_json,
    atomic_torch_save,
    seed_everything,
    state_dict_sha256,
)
from .config import TrainingConfig
from .graph_dataset_loader import (
    GRAPH_FEATURE_NAMES,
    GraphDataset,
    graph_feature_schema,
    load_graph_dataset,
)
from .metrics import binary_metrics
from .split import EffectiveSplits
from .text_embedding import sha256_path
from .training_guard import SplitAccessGuard


def _positive_weight(targets: np.ndarray) -> float:
    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Graph training requires both target classes")
    return negatives / positives


def fit_graph_scaler(
    dataset: GraphDataset,
    *,
    input_manifest_hash: str,
) -> TrainOnlyScaler:
    values = np.where(dataset.feature_mask, dataset.features, np.nan).reshape(
        -1, dataset.features.shape[-1]
    )
    dates = np.repeat(dataset.dates, len(dataset.symbols)).tolist()
    return TrainOnlyScaler(list(dataset.feature_names)).fit(
        values,
        dates=dates,
        input_manifest_hash=input_manifest_hash,
    )


def scale_graph_dataset(dataset: GraphDataset, scaler: TrainOnlyScaler) -> GraphDataset:
    shape = dataset.features.shape
    values = np.where(dataset.feature_mask, dataset.features, np.nan).reshape(-1, shape[-1])
    scaled = scaler.transform(values).reshape(shape)
    scaled = np.where(dataset.feature_mask, scaled, 0.0).astype(np.float32)
    return GraphDataset(
        sample_ids=dataset.sample_ids,
        dates=dataset.dates,
        symbols=dataset.symbols,
        features=scaled,
        adjacency=dataset.adjacency,
        feature_mask=dataset.feature_mask,
        targets=dataset.targets,
        label_mask=dataset.label_mask,
        decision_ids=dataset.decision_ids,
        feature_names=dataset.feature_names,
    )


def predict_gat(
    model: DenseSignedGAT,
    dataset: GraphDataset,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    include_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    logits_batches: list[np.ndarray] = []
    attention_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            features = torch.as_tensor(
                dataset.features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            adjacency = torch.as_tensor(
                dataset.adjacency[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            mask = torch.as_tensor(
                dataset.feature_mask[start : start + batch_size],
                dtype=torch.bool,
                device=device,
            )
            if include_attention:
                logits, attention = model(
                    features, adjacency, mask, return_attention=True
                )
                attention_batches.append(attention.cpu().numpy())
            else:
                logits = model(features, adjacency, mask)
            logits_batches.append(logits.cpu().numpy())
    logits_array = np.concatenate(logits_batches, axis=0)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -60.0, 60.0)))
    attention_array = (
        np.concatenate(attention_batches, axis=0) if attention_batches else None
    )
    return logits_array, probability, attention_array


def _masked_metrics(dataset: GraphDataset, probabilities: np.ndarray) -> dict[str, Any]:
    symbols = np.broadcast_to(np.asarray(dataset.symbols), dataset.targets.shape)
    result = binary_metrics(
        dataset.targets[dataset.label_mask],
        probabilities[dataset.label_mask],
        symbols=symbols[dataset.label_mask],
    )
    result["masked_neutral_count"] = int(np.size(dataset.label_mask) - dataset.label_mask.sum())
    return result


def fit_gat(
    train: GraphDataset,
    validation: GraphDataset,
    model_config: dict[str, Any],
    *,
    seed: int,
    device: str = "cpu",
    fixed_epochs: int | None = None,
) -> tuple[DenseSignedGAT, dict[str, Any]]:
    seed_everything(seed)
    model = DenseSignedGAT(
        input_dim=train.features.shape[-1],
        hidden_dim=int(model_config["hidden_dim"]),
        num_heads=int(model_config["num_heads"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    pos_weight = torch.tensor(
        _positive_weight(train.targets[train.label_mask]), device=device
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none",
    )
    max_epochs = fixed_epochs or int(model_config["max_epochs"])
    patience = int(model_config["patience"])
    batch_size = int(model_config["batch_size"])
    clip_norm = float(model_config.get("gradient_clip_norm", 1.0))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    curves: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        total_loss = 0.0
        total_targets = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].numpy()
            features = torch.as_tensor(train.features[indices], dtype=torch.float32, device=device)
            adjacency = torch.as_tensor(train.adjacency[indices], dtype=torch.float32, device=device)
            feature_mask = torch.as_tensor(train.feature_mask[indices], dtype=torch.bool, device=device)
            targets = torch.as_tensor(train.targets[indices], dtype=torch.float32, device=device)
            label_mask = torch.as_tensor(train.label_mask[indices], dtype=torch.bool, device=device)
            if not bool(label_mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, adjacency, feature_mask)
            losses = loss_function(logits, targets)
            loss = losses[label_mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            count = int(label_mask.sum())
            total_loss += float(loss.detach()) * count
            total_targets += count
        _, validation_probability, _ = predict_gat(model, validation, device=device)
        validation_loss = float(
            _masked_metrics(validation, validation_probability)["log_loss"]
        )
        curves.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_targets, 1),
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
        raise RuntimeError("GAT training produced no checkpoint")
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


def train_graph_stream(
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
    dataset = load_graph_dataset(config, splits, guard=guard)
    train_raw = dataset.subset(splits.train_effective)
    selection_raw = dataset.subset(splits.validation_selection)
    if not len(train_raw) or not len(selection_raw):
        raise ValueError("Graph train or validation-selection dataset is empty")
    scaler = fit_graph_scaler(
        train_raw,
        input_manifest_hash=sha256_path(config.artifact_dir / "data_manifest.json"),
    )
    train = scale_graph_dataset(train_raw, scaler)
    selection = scale_graph_dataset(selection_raw, scaler)
    model_dir = config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(model_dir / "graph_scaler.json")
    atomic_json(model_dir / "feature_schema.json", graph_feature_schema())

    model, training = fit_gat(
        train,
        selection,
        config.graph_model,
        seed=seed,
        device=device,
    )
    train_logits, train_probability, _ = predict_gat(model, train, device=device)
    selection_logits, selection_probability, attention = predict_gat(
        model,
        selection,
        device=device,
        include_attention=True,
    )
    train_metrics = _masked_metrics(train, train_probability)
    selection_metrics = _masked_metrics(selection, selection_probability)
    checkpoint = {
        "model_type": "dense_signed_gat",
        "architecture": model.architecture_config(),
        "state_dict": model.state_dict(),
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "symbols": list(dataset.symbols),
        "seed": seed,
        "best_epoch": training["best_epoch"],
        "validation_prediction_probe": {
            "logits": selection_logits[0].tolist(),
            "probabilities": selection_probability[0].tolist(),
        },
    }
    atomic_torch_save(model_dir / "graph_model.pt", checkpoint)
    attention_path = model_dir / "graph_attention_validation.npz"
    temporary = attention_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        attention=attention,
        dates=np.asarray([value.isoformat() for value in selection.dates]),
        symbols=np.asarray(selection.symbols),
    )
    os.replace(temporary, attention_path)

    loaded = load_gat(model_dir / "graph_model.pt", device=device)
    loaded_logits, _, _ = predict_gat(
        loaded,
        GraphDataset(
            sample_ids=selection.sample_ids[:2],
            dates=selection.dates[:2],
            symbols=selection.symbols,
            features=selection.features[:2],
            adjacency=selection.adjacency[:2],
            feature_mask=selection.feature_mask[:2],
            targets=selection.targets[:2],
            label_mask=selection.label_mask[:2],
            decision_ids=selection.decision_ids[:2],
            feature_names=selection.feature_names,
        ),
        device=device,
    )
    save_load_equivalent = bool(
        np.allclose(loaded_logits, selection_logits[:2], atol=1e-7, rtol=1e-6)
    )
    if not training["parameter_changed"]:
        raise RuntimeError("GAT parameters did not change during training")
    if not training["best_checkpoint_restored"]:
        raise RuntimeError("GAT best validation checkpoint was not restored")
    if not save_load_equivalent:
        raise RuntimeError("GAT save/load prediction mismatch")
    history = {
        **training,
        "architecture": model.architecture_config(),
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "train_range": [min(train.dates).isoformat(), max(train.dates).isoformat()],
        "validation_selection_range": [
            min(selection.dates).isoformat(),
            max(selection.dates).isoformat(),
        ],
        "train_days": len(train),
        "validation_selection_days": len(selection),
        "train_supervised_rows": train.supervised_count,
        "validation_selection_supervised_rows": selection.supervised_count,
        "train_metrics": train_metrics,
        "validation_selection_metrics": selection_metrics,
        "attention_artifact": attention_path.name,
        "save_load_prediction_equivalent": save_load_equivalent,
    }
    atomic_json(model_dir / "graph_training_history.json", history)
    return history


def load_gat(path: str | Path, *, device: str = "cpu") -> DenseSignedGAT:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = DenseSignedGAT.from_config(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
