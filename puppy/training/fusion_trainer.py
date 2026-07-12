from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from puppy.fusion.gated_fusion import LatentGatedFusion

from .checkpoint import atomic_json, atomic_torch_save, seed_everything, state_dict_sha256
from .config import TrainingConfig
from .metrics import binary_metrics
from .oof import generate_validation_base_predictions
from .split import EffectiveSplits
from .training_guard import SplitAccessGuard


CanonicalFusionModel = LatentGatedFusion
CANONICAL_FUSION_SELECTION = "canonical_latent_fusion"


def _arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Fusion dataset is empty")
    required = ("h_text", "h_graph", "text_logit", "graph_logit", "target", "symbol")
    missing = [name for name in required if any(row.get(name) is None for row in rows)]
    if missing:
        raise ValueError(f"Latent fusion rows are missing required fields: {missing}")
    return {
        "h_text": np.asarray([row["h_text"] for row in rows], dtype=np.float32),
        "h_graph": np.asarray([row["h_graph"] for row in rows], dtype=np.float32),
        "text_logit": np.asarray([row["text_logit"] for row in rows], dtype=np.float32),
        "graph_logit": np.asarray([row["graph_logit"] for row in rows], dtype=np.float32),
        "text_probability": np.asarray(
            [row.get("text_probability", _sigmoid_scalar(row["text_logit"])) for row in rows],
            dtype=np.float32,
        ),
        "graph_probability": np.asarray(
            [row.get("graph_probability", _sigmoid_scalar(row["graph_logit"])) for row in rows],
            dtype=np.float32,
        ),
        "target": np.asarray([row["target"] for row in rows], dtype=np.float32),
        "symbol": np.asarray([row["symbol"] for row in rows]),
    }


def _positive_weight(targets: np.ndarray) -> float:
    positive = int(np.sum(targets == 1))
    negative = int(np.sum(targets == 0))
    if not positive or not negative:
        raise ValueError("Fusion training requires both target classes")
    return negative / positive


def _sigmoid_scalar(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))


def _probability(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def predict_fusion(
    model: LatentGatedFusion,
    values: dict[str, np.ndarray],
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = model(
            torch.as_tensor(values["h_text"], dtype=torch.float32, device=device),
            torch.as_tensor(values["h_graph"], dtype=torch.float32, device=device),
            text_scores=torch.as_tensor(
                values["text_probability"], dtype=torch.float32, device=device
            ),
            graph_scores=torch.as_tensor(
                values["graph_probability"], dtype=torch.float32, device=device
            ),
        )
    return (
        output.logits.cpu().numpy(),
        output.probabilities.cpu().numpy(),
        output.gate.cpu().numpy(),
    )


def fixed_average_logits(values: dict[str, np.ndarray]) -> np.ndarray:
    return 0.5 * (values["text_logit"] + values["graph_logit"])


def fit_fusion(
    train_values: dict[str, np.ndarray],
    validation_values: dict[str, np.ndarray],
    model_config: dict[str, Any],
    *,
    seed: int,
    device: str = "cpu",
) -> tuple[LatentGatedFusion, dict[str, Any]]:
    seed_everything(seed)
    latent_dim = int(model_config.get("latent_dim", model_config.get("hidden_dim", 32)))
    model = LatentGatedFusion(
        text_dim=int(train_values["h_text"].shape[-1]),
        graph_dim=int(train_values["h_graph"].shape[-1]),
        latent_dim=latent_dim,
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(_positive_weight(train_values["target"]), device=device)
    )
    batch_size = int(model_config["batch_size"])
    max_epochs = int(model_config["max_epochs"])
    patience = int(model_config["patience"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    curves: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_values["target"]), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].numpy()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.as_tensor(train_values["h_text"][indices], dtype=torch.float32, device=device),
                torch.as_tensor(train_values["h_graph"][indices], dtype=torch.float32, device=device),
            )
            targets = torch.as_tensor(
                train_values["target"][indices], dtype=torch.float32, device=device
            )
            loss = loss_function(output.logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        _, validation_probability, _ = predict_fusion(model, validation_values, device=device)
        validation_loss = float(
            binary_metrics(validation_values["target"], validation_probability)["log_loss"]
        )
        curves.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_values["target"]),
                "validation_log_loss": validation_loss,
            }
        )
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
        raise RuntimeError("Latent fusion training produced no checkpoint")
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
    }


def _candidate_metrics(
    values: dict[str, np.ndarray],
    fusion_probability: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        "text_only": binary_metrics(
            values["target"], _probability(values["text_logit"]), symbols=values["symbol"]
        ),
        "graph_only": binary_metrics(
            values["target"], _probability(values["graph_logit"]), symbols=values["symbol"]
        ),
        "fixed_0.5_average": binary_metrics(
            values["target"], _probability(fixed_average_logits(values)), symbols=values["symbol"]
        ),
        CANONICAL_FUSION_SELECTION: binary_metrics(
            values["target"], fusion_probability, symbols=values["symbol"]
        ),
    }


def train_fusion(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    guard: SplitAccessGuard | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    guard = guard or SplitAccessGuard()
    seed = config.seed if seed is None else int(seed)
    guard.assert_allowed("train", "fit")
    guard.assert_allowed("validation_selection", "model_selection")
    validation_summary = generate_validation_base_predictions(
        config, splits, guard=guard, device=device
    )
    oof_rows = pq.read_table(config.model_dir / "oof_predictions.parquet").to_pylist()
    validate_oof_rows(
        oof_rows,
        int(config.split["embargo_trading_days"]),
        trading_dates=splits.train_effective,
    )
    validation_rows = pq.read_table(
        config.model_dir / "validation_base_predictions.parquet",
        filters=[("partition", "=", "validation_selection")],
    ).to_pylist()
    train_values = _arrays(oof_rows)
    validation_values = _arrays(validation_rows)
    model, training = fit_fusion(
        train_values,
        validation_values,
        config.fusion_model,
        seed=seed,
        device=device,
    )
    _, train_probability, train_gate = predict_fusion(model, train_values, device=device)
    validation_logits, validation_probability, validation_gate = predict_fusion(
        model, validation_values, device=device
    )
    candidates = _candidate_metrics(validation_values, validation_probability)
    best_ablation = min(
        (name for name in candidates if name != CANONICAL_FUSION_SELECTION),
        key=lambda name: candidates[name]["log_loss"],
    )
    checkpoint = {
        "model_type": CANONICAL_FUSION_SELECTION,
        "architecture": model.architecture_config(),
        "state_dict": model.state_dict(),
        "selected_fusion_model": CANONICAL_FUSION_SELECTION,
        "seed": seed,
        "best_epoch": training["best_epoch"],
        "validation_prediction_probe": {
            "logit": float(validation_logits[0]),
            "probability": float(validation_probability[0]),
            "gate": validation_gate[0].tolist(),
        },
    }
    atomic_torch_save(config.model_dir / "fusion_model.pt", checkpoint)
    loaded, selected = load_fusion_model(config.model_dir / "fusion_model.pt", device=device)
    loaded_logits, _, _ = predict_fusion(loaded, validation_values, device=device)
    save_load_equivalent = bool(
        selected == CANONICAL_FUSION_SELECTION
        and np.allclose(loaded_logits, validation_logits, atol=1e-7, rtol=1e-6)
    )
    if not training["parameter_changed"] or not training["best_checkpoint_restored"]:
        raise RuntimeError("Latent fusion did not produce a valid trained checkpoint")
    if not save_load_equivalent:
        raise RuntimeError("Latent fusion save/load prediction mismatch")
    history = {
        **training,
        "selected_fusion_model": CANONICAL_FUSION_SELECTION,
        "production_selection_rule": "canonical_architecture_fixed",
        "best_validation_ablation": best_ablation,
        "fit_source": "expanding_window_oof_latents_only",
        "oof_only_verified": True,
        "oof_train_rows": len(oof_rows),
        "validation_selection_rows": len(validation_rows),
        "candidate_validation_selection_metrics": candidates,
        "canonical_fusion_oof_metrics": binary_metrics(
            train_values["target"], train_probability, symbols=train_values["symbol"]
        ),
        "gate_values": {
            "oof_mean": float(np.mean(train_gate)),
            "oof_min": float(np.min(train_gate)),
            "oof_max": float(np.max(train_gate)),
            "validation_mean": float(np.mean(validation_gate)),
        },
        "latent_dimensions": model.architecture_config(),
        "fixed_average_baseline_reported": True,
        "save_load_prediction_equivalent": save_load_equivalent,
        "validation_base_predictions": validation_summary,
    }
    atomic_json(config.model_dir / "fusion_training_history.json", history)
    return history


def validate_oof_rows(
    rows: list[dict[str, Any]],
    embargo_days: int,
    *,
    trading_dates: tuple[Any, ...] | None = None,
) -> None:
    if not rows or not all(bool(row.get("is_oof")) for row in rows):
        raise ValueError("Fusion requires OOF-only training predictions")
    for row in rows:
        if not row["model_train_end"] < row["decision_date"]:
            raise ValueError("Fusion received an in-sample base prediction")
        if int(row["embargo_trading_days"]) != embargo_days:
            raise ValueError("Fusion received an OOF embargo violation")
        if not bool(row.get("text_prediction_is_oof")):
            raise ValueError("Fusion requires out-of-fold Text predictions")
        if not bool(row.get("graph_representation_is_oof")):
            raise ValueError("Fusion requires out-of-fold H_graph representations")
        if row.get("h_text") is None or row.get("h_graph") is None:
            raise ValueError("Fusion requires H_text and OOF H_graph latent vectors")
    if trading_dates is not None:
        positions = {date: index for index, date in enumerate(trading_dates)}
        for row in rows:
            try:
                distance = positions[row["decision_date"]] - positions[row["model_train_end"]] - 1
            except KeyError as error:
                raise ValueError("OOF date is absent from the train trading calendar") from error
            if distance < embargo_days:
                raise ValueError("Fusion received an OOF trading-calendar embargo violation")


def load_fusion_model(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[LatentGatedFusion, str]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    selected = str(checkpoint.get("selected_fusion_model", ""))
    if selected != CANONICAL_FUSION_SELECTION:
        raise ValueError("Production bundle must select canonical latent fusion")
    model = LatentGatedFusion.from_config(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, selected


def selected_fusion_logits(
    model: LatentGatedFusion,
    selected: str,
    values: dict[str, np.ndarray],
    *,
    device: str = "cpu",
) -> np.ndarray:
    if selected != CANONICAL_FUSION_SELECTION:
        raise ValueError("Production inference cannot select a Text-only or Graph-only ablation")
    return predict_fusion(model, values, device=device)[0]
