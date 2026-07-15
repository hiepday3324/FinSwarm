from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import yaml

from puppy.fusion.scalar_logit_stack import ScalarLogitStack
from puppy.fusion.text_score_head import TextScoreHead
from puppy.graph.gat_model import DenseSignedGAT
from puppy.training.checkpoint import (
    TrainOnlyScaler,
    atomic_json,
    atomic_torch_save,
    seed_everything,
    state_dict_sha256,
)
from puppy.training.config import TrainingConfig, load_training_config
from puppy.training.fusion_trainer import (
    SCALAR_FUSION_SELECTION,
    load_fusion_model,
    selected_fusion_logits,
)
from puppy.training.graph_dataset_loader import (
    GraphDataset,
    graph_feature_schema,
    load_graph_dataset,
)
from puppy.training.graph_trainer import (
    fit_graph_scaler,
    load_gat,
    predict_gat,
    scale_graph_dataset,
)
from puppy.training.metrics import binary_metrics
from puppy.training.split import EffectiveSplits
from puppy.training.text_dataset import TextDataset, load_supervised_text_dataset
from puppy.training.text_embedding import sha256_path
from puppy.training.text_trainer import predict_mlp
from puppy.training.training_guard import SplitAccessGuard

from demo.constants import (
    ATTENTION_CONVENTION,
    DEFAULT_CONFIG_PATH,
    DISPLAY_ORDER,
    EXPECTED_ALIGNED_ROWS,
    EXPECTED_SUPERVISED_ROWS,
    EXPECTED_TRAIN_DATES,
    FUSION_PREDICTION_SOURCE,
    GRAPH_PREDICTION_SOURCE,
    NODE_ORDER,
    REPO_ROOT,
    TEXT_SCORE_SOURCE,
    TRAIN_END,
    TRAIN_START,
)


DEMO_CONFIG_SECTIONS = {
    "run": {"run_id", "seed"},
    "inputs": {"experiment_config", "artifact_dir", "split_manifest"},
    "output": {"model_dir"},
    "text_embedding": {
        "model_name",
        "dimension",
        "pooling",
        "normalize",
        "max_length",
        "batch_size",
        "text_view_version",
        "local_files_only",
    },
    "split": {
        "embargo_trading_days",
        "validation_selection_fraction",
        "oof_folds",
        "oof_min_train_days",
    },
    "labels": {
        "positive_label",
        "negative_label",
        "neutral_label",
        "mask_neutral",
        "threshold_bps",
    },
    "graph": {"topology", "include_text_score"},
    "fusion": {"head"},
    "oof": {"training_policy"},
    "text_model": {
        "baseline",
        "primary",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
    },
    "graph_model": {
        "hidden_dim",
        "num_heads",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "gradient_clip_norm",
    },
    "fusion_model": {
        "latent_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
    },
    "calibration": {"method"},
    "demo_fit": {
        "demo_only",
        "intentionally_in_sample",
        "evaluation_mode",
        "model_frozen_during_ui",
        "train_start",
        "train_end",
        "checkpoint_selection",
        "early_stopping",
        "stop_roc_auc",
        "stop_balanced_accuracy",
        "expected_supervised_rows",
        "validation_access_count",
        "test_access_count",
    },
}

PROTECTED_PATHS = (
    "config/swarm/training_v1.yaml",
    "data/training/finswarm_v1",
    "data/model_artifacts/finswarm_v1",
    "data/model_artifacts/ablations/e1_identity_no_text_graph_only",
    "data/model_artifacts/ablations/e2_full_no_text_graph_only",
    "data/model_artifacts/ablations/e3_full_text_graph_only",
    "data/model_artifacts/ablations/e4_scalar_logit_stack",
)


@dataclass(frozen=True)
class DemoConfig:
    training: TrainingConfig
    demo_fit: dict[str, Any]
    raw: dict[str, Any]


def _strict_mapping(name: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Demo config section {name} must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ValueError(f"Missing demo config field(s): {name}.{sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown demo config field(s): {name}.{sorted(unknown)}")
    return dict(value)


def load_demo_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DemoConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Demo training config must be a mapping")
    if set(raw) != set(DEMO_CONFIG_SECTIONS):
        missing = set(DEMO_CONFIG_SECTIONS) - set(raw)
        unknown = set(raw) - set(DEMO_CONFIG_SECTIONS)
        raise ValueError(
            f"Demo config sections mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    for section, expected in DEMO_CONFIG_SECTIONS.items():
        _strict_mapping(section, raw[section], expected)

    training = load_training_config(config_path)
    demo = dict(raw["demo_fit"])
    expected_output = (REPO_ROOT / "data/model_artifacts/finswarm_e4_demo_fit").resolve()
    required = {
        "run_id": training.run_id == "finswarm_e4_demo_fit",
        "output": training.model_dir.resolve() == expected_output,
        "topology": training.graph_topology == "full",
        "text_score": training.graph_includes_text_score,
        "fusion": training.fusion_head == "scalar_logit_stack",
        "oof_contract": training.oof_training_policy == "fixed_epochs",
        "demo_only": demo["demo_only"] is True,
        "in_sample": demo["intentionally_in_sample"] is True,
        "evaluation": demo["evaluation_mode"] == "train_replay",
        "frozen": demo["model_frozen_during_ui"] is True,
        "selection": demo["checkpoint_selection"] == "minimum_train_loss",
        "early_stopping": demo["early_stopping"] == "disabled",
        "train_start": str(demo["train_start"]) == TRAIN_START,
        "train_end": str(demo["train_end"]) == TRAIN_END,
        "expected_rows": int(demo["expected_supervised_rows"])
        == EXPECTED_SUPERVISED_ROWS,
        "validation_access": int(demo["validation_access_count"]) == 0,
        "test_access": int(demo["test_access_count"]) == 0,
        "text_dropout": float(training.text_model["dropout"]) == 0.0,
        "text_decay": float(training.text_model["weight_decay"]) == 0.0,
        "graph_dropout": float(training.graph_model["dropout"]) == 0.0,
        "graph_decay": float(training.graph_model["weight_decay"]) == 0.0,
        "fusion_decay": float(training.fusion_model["weight_decay"]) == 0.0,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise ValueError(f"Demo-fit contract violation(s): {failed}")
    return DemoConfig(training=training, demo_fit=demo, raw=raw)


def protected_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, int]]:
    """Capture stat-only evidence without opening protected artifact contents."""

    snapshot: dict[str, dict[str, int]] = {}
    for relative in PROTECTED_PATHS:
        target = repo_root / relative
        if target.is_file():
            stat = target.stat()
            snapshot[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            continue
        if not target.is_dir():
            snapshot[relative] = {"missing": 1}
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            stat = path.stat()
            key = path.relative_to(repo_root).as_posix()
            snapshot[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return snapshot


def _train_only_splits(config: TrainingConfig) -> EffectiveSplits:
    manifest = json.loads(config.split_manifest.read_text(encoding="utf-8"))
    train = tuple(
        dt.date.fromisoformat(value) for value in manifest["splits"]["train"]["dates"]
    )
    embargo = int(config.split["embargo_trading_days"])
    if embargo <= 0 or len(train) <= embargo:
        raise ValueError("Invalid demo-fit train embargo")
    effective = train[:-embargo]
    if (
        len(effective) != EXPECTED_TRAIN_DATES
        or effective[0].isoformat() != TRAIN_START
        or effective[-1].isoformat() != TRAIN_END
    ):
        raise ValueError("Effective train partition does not match the audited contract")
    return EffectiveSplits(
        warmup=(),
        train_manifest=train,
        train_effective=effective,
        embargoed_train=train[-embargo:],
        validation_selection=(),
        validation_calibration=(),
        test_metadata_dates=(),
    )


def _positive_weight(targets: np.ndarray) -> float:
    values = np.asarray(targets, dtype=np.float32)
    positives = int(np.sum(values == 1))
    negatives = int(np.sum(values == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Demo fitting requires both target classes")
    return negatives / positives


def _probability(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _should_stop(metrics: dict[str, Any], demo_fit: dict[str, Any]) -> str | None:
    auc = metrics.get("roc_auc")
    if auc is not None and float(auc) >= float(demo_fit["stop_roc_auc"]):
        return "train_roc_auc_threshold"
    if float(metrics["balanced_accuracy"]) >= float(
        demo_fit["stop_balanced_accuracy"]
    ):
        return "train_balanced_accuracy_threshold"
    return None


def _fit_text_demo(
    features: np.ndarray,
    targets: np.ndarray,
    symbols: np.ndarray,
    config: DemoConfig,
    *,
    device: str,
) -> tuple[TextScoreHead, dict[str, Any]]:
    model_config = config.training.text_model
    seed_everything(config.training.seed)
    model = TextScoreHead(
        input_dim=features.shape[1],
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        seed=config.training.seed,
    ).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    pos_weight = torch.tensor(_positive_weight(targets), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(config.training.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs"
    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), int(model_config["batch_size"])):
            indices = order[start : start + int(model_config["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices].to(device))
            loss = criterion(logits, y[indices].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full_logits = model(x.to(device))
            train_loss = float(criterion(full_logits, y.to(device)).cpu())
        probability = _probability(full_logits.detach().cpu().numpy())
        metrics = binary_metrics(targets, probability, symbols=symbols)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_roc_auc": metrics["roc_auc"],
                "train_balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[Text] epoch={epoch} train_loss={train_loss:.6f} "
                f"roc_auc={metrics['roc_auc']:.4f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
        if train_loss < best_loss - 1e-10:
            best_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        threshold_stop = _should_stop(metrics, config.demo_fit)
        if threshold_stop is not None:
            stop_reason = threshold_stop
            break
    if best_state is None:
        raise RuntimeError("Text demo fitting produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_logits, final_probability = predict_mlp(model, features, device=device)
    final_metrics = binary_metrics(targets, final_probability, symbols=symbols)
    restored_hash = state_dict_sha256(model.state_dict())
    return model, {
        "model": "canonical_text_score_head",
        "training_source": "all_eligible_effective_train_rows",
        "prediction_source": TEXT_SCORE_SOURCE,
        "checkpoint_selection": "minimum_train_loss",
        "validation_early_stopping": False,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "stop_reason": stop_reason,
        "learning_curve": history,
        "train_metrics": final_metrics,
        "initial_state_sha256": initial_hash,
        "best_state_sha256": restored_hash,
        "parameter_changed": initial_hash != restored_hash,
        "best_checkpoint_restored": restored_hash == state_dict_sha256(best_state),
        "prediction_probe": {
            "logit": float(final_logits[0]),
            "probability": float(final_probability[0]),
        },
    }


def _graph_train_loss(
    model: DenseSignedGAT,
    dataset: GraphDataset,
    *,
    pos_weight: torch.Tensor,
    device: str,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(
            torch.as_tensor(dataset.features, dtype=torch.float32, device=device),
            torch.as_tensor(dataset.adjacency, dtype=torch.float32, device=device),
            torch.as_tensor(dataset.feature_mask, dtype=torch.bool, device=device),
            symbols=dataset.symbols,
            feature_names=dataset.feature_names,
            feature_provenance={"source": "effective_train_in_sample_demo"},
        )
        targets = torch.as_tensor(dataset.targets, dtype=torch.float32, device=device)
        mask = torch.as_tensor(dataset.label_mask, dtype=torch.bool, device=device)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        return float(losses[mask].mean().cpu())


def _fit_graph_demo(
    dataset: GraphDataset,
    config: DemoConfig,
    *,
    device: str,
) -> tuple[DenseSignedGAT, dict[str, Any]]:
    model_config = config.training.graph_model
    seed_everything(config.training.seed)
    model = DenseSignedGAT(
        input_dim=dataset.features.shape[-1],
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
        _positive_weight(dataset.targets[dataset.label_mask]), device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    generator = torch.Generator(device="cpu").manual_seed(config.training.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs"
    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(dataset), generator=generator)
        for start in range(0, len(order), int(model_config["batch_size"])):
            indices = order[start : start + int(model_config["batch_size"])].numpy()
            features = torch.as_tensor(
                dataset.features[indices], dtype=torch.float32, device=device
            )
            adjacency = torch.as_tensor(
                dataset.adjacency[indices], dtype=torch.float32, device=device
            )
            feature_mask = torch.as_tensor(
                dataset.feature_mask[indices], dtype=torch.bool, device=device
            )
            targets = torch.as_tensor(
                dataset.targets[indices], dtype=torch.float32, device=device
            )
            label_mask = torch.as_tensor(
                dataset.label_mask[indices], dtype=torch.bool, device=device
            )
            if not bool(label_mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            output = model(
                features,
                adjacency,
                feature_mask,
                symbols=dataset.symbols,
                feature_names=dataset.feature_names,
                feature_provenance={"source": "effective_train_in_sample_demo"},
            )
            loss = criterion(output.logits, targets)[label_mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(model_config["gradient_clip_norm"])
            )
            optimizer.step()
        train_loss = _graph_train_loss(
            model, dataset, pos_weight=pos_weight, device=device
        )
        _, probability, _ = predict_gat(model, dataset, device=device)
        expanded_symbols = np.broadcast_to(
            np.asarray(dataset.symbols), dataset.targets.shape
        )
        metrics = binary_metrics(
            dataset.targets[dataset.label_mask],
            probability[dataset.label_mask],
            symbols=expanded_symbols[dataset.label_mask],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_roc_auc": metrics["roc_auc"],
                "train_balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[Graph] epoch={epoch} train_loss={train_loss:.6f} "
                f"roc_auc={metrics['roc_auc']:.4f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
        if train_loss < best_loss - 1e-10:
            best_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        threshold_stop = _should_stop(metrics, config.demo_fit)
        if threshold_stop is not None:
            stop_reason = threshold_stop
            break
    if best_state is None:
        raise RuntimeError("Graph demo fitting produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_logits, final_probability, _ = predict_gat(model, dataset, device=device)
    expanded_symbols = np.broadcast_to(np.asarray(dataset.symbols), dataset.targets.shape)
    final_metrics = binary_metrics(
        dataset.targets[dataset.label_mask],
        final_probability[dataset.label_mask],
        symbols=expanded_symbols[dataset.label_mask],
    )
    restored_hash = state_dict_sha256(model.state_dict())
    return model, {
        "model": "canonical_dense_signed_gat",
        "training_source": "all_eligible_effective_train_rows_with_in_sample_text",
        "prediction_source": GRAPH_PREDICTION_SOURCE,
        "text_score_source": TEXT_SCORE_SOURCE,
        "checkpoint_selection": "minimum_train_loss",
        "validation_early_stopping": False,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "stop_reason": stop_reason,
        "learning_curve": history,
        "train_metrics": final_metrics,
        "initial_state_sha256": initial_hash,
        "best_state_sha256": restored_hash,
        "parameter_changed": initial_hash != restored_hash,
        "best_checkpoint_restored": restored_hash == state_dict_sha256(best_state),
        "architecture": model.architecture_config(),
        "attention_convention": ATTENTION_CONVENTION,
        "prediction_probe": {
            "logits": final_logits[0].tolist(),
            "probabilities": final_probability[0].tolist(),
        },
    }


def _fit_scalar_demo(
    text_logits: np.ndarray,
    graph_logits: np.ndarray,
    targets: np.ndarray,
    symbols: np.ndarray,
    config: DemoConfig,
    *,
    device: str,
) -> tuple[ScalarLogitStack, dict[str, Any]]:
    model_config = config.training.fusion_model
    seed_everything(config.training.seed)
    model = ScalarLogitStack().to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_config["learning_rate"])
    )
    regularization = float(model_config["weight_decay"])
    pos_weight = torch.tensor(_positive_weight(targets), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    text_tensor = torch.as_tensor(text_logits, dtype=torch.float32)
    graph_tensor = torch.as_tensor(graph_logits, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(config.training.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs"
    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(target_tensor), generator=generator)
        for start in range(0, len(order), int(model_config["batch_size"])):
            indices = order[start : start + int(model_config["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                text_tensor[indices].to(device), graph_tensor[indices].to(device)
            )
            loss = criterion(logits, target_tensor[indices].to(device))
            if regularization:
                loss = loss + regularization * torch.sum(model.weights.square())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full_logits = model(text_tensor.to(device), graph_tensor.to(device))
            train_loss = float(
                criterion(full_logits, target_tensor.to(device)).detach().cpu()
            )
            if regularization:
                train_loss += regularization * float(
                    torch.sum(model.weights.square()).detach().cpu()
                )
        probability = _probability(full_logits.detach().cpu().numpy())
        metrics = binary_metrics(targets, probability, symbols=symbols)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_roc_auc": metrics["roc_auc"],
                "train_balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"[Scalar Fusion] epoch={epoch} train_loss={train_loss:.6f} "
                f"roc_auc={metrics['roc_auc']:.4f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
        if train_loss < best_loss - 1e-10:
            best_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        threshold_stop = _should_stop(metrics, config.demo_fit)
        if threshold_stop is not None:
            stop_reason = threshold_stop
            break
    if best_state is None:
        raise RuntimeError("Scalar demo fitting produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(
            text_tensor.to(device), graph_tensor.to(device)
        ).detach().cpu().numpy()
    final_probability = _probability(final_logits)
    final_metrics = binary_metrics(targets, final_probability, symbols=symbols)
    restored_hash = state_dict_sha256(model.state_dict())
    coefficients = {
        "intercept": float(model.intercept.detach().cpu()),
        "weight_text": float(model.weights[0].detach().cpu()),
        "weight_graph": float(model.weights[1].detach().cpu()),
    }
    return model, {
        "model": "scalar_logit_stack",
        "training_source": "aligned_in_sample_text_and_graph_logits",
        "prediction_source": FUSION_PREDICTION_SOURCE,
        "checkpoint_selection": "minimum_train_loss",
        "validation_early_stopping": False,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "stop_reason": stop_reason,
        "learning_curve": history,
        "train_metrics": final_metrics,
        "initial_state_sha256": initial_hash,
        "best_state_sha256": restored_hash,
        "parameter_changed": initial_hash != restored_hash,
        "best_checkpoint_restored": restored_hash == state_dict_sha256(best_state),
        "coefficients": coefficients,
        "regularization": {
            "type": "l2",
            "strength": regularization,
            "applies_to": ["weight_text", "weight_graph"],
            "intercept_regularized": False,
        },
        "prediction_equation": (
            "fusion_logit = intercept + weight_text * text_logit + "
            "weight_graph * graph_logit"
        ),
        "prediction_probe": {
            "logit": float(final_logits[0]),
            "probability": float(final_probability[0]),
        },
    }


def _filtered_rows(
    path: Path,
    dates: Iterable[dt.date],
    *,
    split: str = "train",
) -> list[dict[str, Any]]:
    dataset = ds.dataset(path, format="parquet")
    filter_expression = (ds.field("split") == split) & ds.field(
        "decision_date"
    ).isin(tuple(dates))
    return dataset.to_table(filter=filter_expression).to_pylist()


def _all_train_rows(
    config: DemoConfig,
    dates: tuple[dt.date, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    artifact_dir = config.training.artifact_dir
    embeddings = _filtered_rows(
        artifact_dir / "text_embeddings_384.parquet", dates
    )
    pending = _filtered_rows(artifact_dir / "pending_decisions.parquet", dates)
    if len(embeddings) != EXPECTED_ALIGNED_ROWS or len(pending) != EXPECTED_ALIGNED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ALIGNED_ROWS} aligned train rows, got "
            f"embeddings={len(embeddings)}, pending={len(pending)}"
        )
    embedding_by_id = {str(row["decision_id"]): row for row in embeddings}
    pending_by_id = {str(row["decision_id"]): row for row in pending}
    if len(embedding_by_id) != len(embeddings) or len(pending_by_id) != len(pending):
        raise ValueError("Replay source artifacts contain duplicate decision IDs")
    if set(embedding_by_id) != set(pending_by_id):
        raise ValueError("Replay Text embeddings and pending decisions are misaligned")
    access = {
        "validation_access_count": 0,
        "test_access_count": 0,
        "files": [
            {
                "path": "data/training/finswarm_v1/text_embeddings_384.parquet",
                "filter": "split=train and decision_date in effective_train",
            },
            {
                "path": "data/training/finswarm_v1/pending_decisions.parquet",
                "filter": "split=train and decision_date in effective_train",
            },
            {
                "path": "data/training/finswarm_v1/outcomes.parquet",
                "filter": "split=train and decision_date in effective_train",
                "use": "supervision_only",
            },
            {
                "path": "data/training/finswarm_v1/state/graph_shards/*.npz",
                "filter": f"{TRAIN_START} through {TRAIN_END}",
            },
        ],
    }
    return embeddings, pending, access


def _ordered_text_arrays(
    graph: GraphDataset,
    embeddings: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    by_id = {str(row["decision_id"]): row for row in embeddings}
    vectors: list[np.ndarray] = []
    inputs: list[str] = []
    ids: list[str] = []
    for date_index, _ in enumerate(graph.dates):
        for node, _symbol in enumerate(graph.symbols):
            decision_id = str(graph.decision_ids[date_index, node])
            row = by_id.get(decision_id)
            if row is None:
                raise ValueError(f"Missing replay Text embedding for {decision_id}")
            vector = np.asarray(row["embedding"], dtype=np.float32)
            if vector.shape != (384,):
                raise ValueError(f"Invalid replay embedding shape for {decision_id}")
            vectors.append(vector)
            inputs.append(str(row["text_input"]))
            ids.append(decision_id)
    return (
        np.stack(vectors).reshape(len(graph), len(graph.symbols), 384),
        np.asarray(inputs).reshape(len(graph), len(graph.symbols)),
        np.asarray(ids).reshape(len(graph), len(graph.symbols)),
        by_id,
    )


def _attention_outputs(
    model: DenseSignedGAT,
    dataset: GraphDataset,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_batches: list[np.ndarray] = []
    heads_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 256):
            output = model(
                torch.as_tensor(
                    dataset.features[start : start + 256],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    dataset.adjacency[start : start + 256],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    dataset.feature_mask[start : start + 256],
                    dtype=torch.bool,
                    device=device,
                ),
                symbols=dataset.symbols,
                feature_names=dataset.feature_names,
                feature_provenance={"source": "effective_train_in_sample_demo"},
            )
            if model.last_attention_heads is None:
                raise RuntimeError("GAT did not retain per-head attention")
            logits_batches.append(output.logits.detach().cpu().numpy())
            heads_batches.append(model.last_attention_heads.detach().cpu().numpy())
    logits = np.concatenate(logits_batches).astype(np.float32)
    probability = _probability(logits).astype(np.float32)
    heads = np.concatenate(heads_batches).astype(np.float32)
    mean = heads.mean(axis=1).astype(np.float32)
    return logits, probability, heads, mean


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def _load_journal_record(artifact_dir: Path, date: dt.date, symbol: str) -> dict[str, Any]:
    path = artifact_dir / "state/causal_decision_journal" / (
        f"{date.isoformat()}__{symbol}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing train decision journal record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _materialize_replay_records(
    output_dir: Path,
    graph: GraphDataset,
    pending_rows: list[dict[str, Any]],
    text_inputs: np.ndarray,
    config: DemoConfig,
) -> tuple[int, int]:
    pending_by_id = {str(row["decision_id"]): row for row in pending_rows}
    records: list[dict[str, Any]] = []
    news_routes: dict[str, dict[str, Any]] = {}
    all_event_ids: set[str] = set()
    for date_index, date in enumerate(graph.dates):
        for node, symbol in enumerate(graph.symbols):
            decision_id = str(graph.decision_ids[date_index, node])
            pending = pending_by_id[decision_id]
            context_snapshot = json.loads(str(pending["context_snapshot"]))
            agent_context = dict(context_snapshot["agent_context"])
            news_ids = [str(value) for value in context_snapshot.get("news_event_ids", [])]
            all_event_ids.update(news_ids)
            journal = _load_journal_record(config.training.artifact_dir, date, symbol)
            if str(journal["decision_id"]) != decision_id:
                raise ValueError(f"Journal decision mismatch for {symbol} on {date}")
            for event_id in news_ids:
                route = news_routes.setdefault(
                    event_id,
                    {
                        "decision_date": date,
                        "decision_cutoff": pending["decision_cutoff"],
                        "agents": set(),
                    },
                )
                route["agents"].add(symbol)
            records.append(
                {
                    "decision_date": date,
                    "symbol": symbol,
                    "node_index": node,
                    "decision_id": decision_id,
                    "agent_id": str(pending["agent_id"]),
                    "decision_cutoff": pending["decision_cutoff"],
                    "decision": str(pending["decision"]),
                    "action_signal": int(pending["action_signal"]),
                    "reason": str(pending["reason"]),
                    "context_hash": str(pending["context_hash"]),
                    "context_snapshot_json": json.dumps(
                        context_snapshot, sort_keys=True, default=str
                    ),
                    "memory_refs_json": str(pending["memory_refs"]),
                    "news_event_ids_json": json.dumps(news_ids),
                    "psychology_state_json": json.dumps(
                        agent_context.get("psychology_state"),
                        sort_keys=True,
                        default=str,
                    ),
                    "shadow_state_json": json.dumps(
                        agent_context.get("shadow_state"),
                        sort_keys=True,
                        default=str,
                    ),
                    "text_input": str(text_inputs[date_index, node]),
                    "raw_agent_output_json": json.dumps(
                        journal["agent_output"], sort_keys=True, default=str
                    ),
                    "causal_provenance_json": json.dumps(
                        journal.get("provenance", {}), sort_keys=True, default=str
                    ),
                    "cited_memory_ids_json": json.dumps(
                        journal.get("cited_memory_ids", {}), sort_keys=True
                    ),
                    "prompt_version": str(pending["prompt_version"]),
                    "prompt_artifact_status": "unavailable",
                    "confidence_artifact_status": "unavailable",
                    "routing_provenance": "context_snapshot.news_event_ids",
                    "split": "train",
                }
            )
    _atomic_parquet(output_dir / "replay_records.parquet", records)

    if not all_event_ids:
        _atomic_parquet(output_dir / "replay_news.parquet", [])
        return len(records), 0
    news_dataset = ds.dataset(
        config.training.repo_root / "data/parquet/news.parquet", format="parquet"
    )
    news_rows = news_dataset.to_table(
        filter=ds.field("event_id").isin(sorted(all_event_ids))
    ).to_pylist()
    news_by_id = {str(row["event_id"]): row for row in news_rows}
    missing = all_event_ids - set(news_by_id)
    if missing:
        raise ValueError(f"Replay news events missing from source artifact: {sorted(missing)[:3]}")

    grouped: dict[tuple[dt.date, str], dict[str, Any]] = {}
    for event_id, route in news_routes.items():
        row = news_by_id[event_id]
        provider_id = str(row.get("provider_record_id") or event_id)
        key = (route["decision_date"], provider_id)
        group = grouped.setdefault(
            key,
            {
                "decision_date": route["decision_date"],
                "news_group_id": provider_id,
                "headline": str(row.get("title") or "Unavailable in artifact"),
                "source": str(row.get("source") or "Unavailable in artifact"),
                "published_at": row.get("published_at"),
                "available_at": row.get("available_at"),
                "url": str(row.get("url") or ""),
                "text": str(row.get("text") or ""),
                "event_ids": set(),
                "related_symbols": set(),
                "agents_received": set(),
                "decision_cutoffs": set(),
                "causally_available": True,
            },
        )
        group["event_ids"].add(event_id)
        group["related_symbols"].add(str(row["symbol"]))
        group["agents_received"].update(route["agents"])
        cutoff = route["decision_cutoff"]
        group["decision_cutoffs"].add(str(cutoff))
        group["causally_available"] = bool(
            group["causally_available"]
            and row.get("available_at") is not None
            and row["available_at"] <= cutoff
        )
    materialized_news: list[dict[str, Any]] = []
    for group in grouped.values():
        cutoffs = sorted(group.pop("decision_cutoffs"))
        event_ids = sorted(group.pop("event_ids"))
        related_symbols = sorted(group.pop("related_symbols"))
        received = group.pop("agents_received")
        materialized_news.append(
            {
                **group,
                "event_ids_json": json.dumps(event_ids),
                "related_symbols_json": json.dumps(related_symbols),
                "agents_received_json": json.dumps(
                    [symbol for symbol in DISPLAY_ORDER if symbol in received]
                ),
                "decision_cutoff": cutoffs[0] if len(cutoffs) == 1 else None,
                "routing_provenance": "direct_context_event_id_membership",
            }
        )
    materialized_news.sort(
        key=lambda row: (row["decision_date"], row["published_at"], row["news_group_id"])
    )
    _atomic_parquet(output_dir / "replay_news.parquet", materialized_news)
    return len(records), len(materialized_news)


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _model_hashes(
    text_model: TextScoreHead,
    graph_model: DenseSignedGAT,
    fusion_model: ScalarLogitStack,
) -> dict[str, str]:
    return {
        "text_state_sha256": state_dict_sha256(text_model.state_dict()),
        "gat_state_sha256": state_dict_sha256(graph_model.state_dict()),
        "scalar_fusion_state_sha256": state_dict_sha256(fusion_model.state_dict()),
    }


def _verify_save_load(
    output_dir: Path,
    text_embeddings: np.ndarray,
    graph_raw: GraphDataset,
    expected_text_logits: np.ndarray,
    expected_graph_logits: np.ndarray,
    expected_heads: np.ndarray,
    expected_fusion_logits: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    text_model = TextScoreHead.load(output_dir / "text_model.pt", device=device)
    graph_model = load_gat(output_dir / "graph_model.pt", device=device)
    fusion_model, selected = load_fusion_model(
        output_dir / "fusion_model.pt", device=device
    )
    if selected != SCALAR_FUSION_SELECTION or not isinstance(
        fusion_model, ScalarLogitStack
    ):
        raise ValueError("Demo bundle did not reload ScalarLogitStack")
    text_scaler = TrainOnlyScaler.load(output_dir / "text_scaler.json")
    graph_scaler = TrainOnlyScaler.load(output_dir / "graph_scaler.json")
    probe_dates = min(3, len(graph_raw))
    probe_text = text_embeddings[:probe_dates].reshape(-1, 384)
    text_logits, text_probability = predict_mlp(
        text_model, text_scaler.transform(probe_text), device=device
    )
    raw = graph_raw.features[:probe_dates].copy()
    text_index = graph_raw.feature_names.index("text_score")
    raw[:, :, text_index] = text_probability.reshape(probe_dates, len(NODE_ORDER))
    probe_graph = GraphDataset(
        sample_ids=graph_raw.sample_ids[:probe_dates],
        dates=graph_raw.dates[:probe_dates],
        symbols=graph_raw.symbols,
        features=raw,
        adjacency=graph_raw.adjacency[:probe_dates],
        feature_mask=graph_raw.feature_mask[:probe_dates],
        targets=np.zeros((probe_dates, len(NODE_ORDER)), dtype=np.float32),
        label_mask=np.zeros((probe_dates, len(NODE_ORDER)), dtype=bool),
        decision_ids=graph_raw.decision_ids[:probe_dates],
        feature_names=graph_raw.feature_names,
        metadata=dict(graph_raw.metadata),
    )
    scaled = scale_graph_dataset(probe_graph, graph_scaler)
    graph_logits, _, heads, _ = _attention_outputs(
        graph_model, scaled, device=device
    )
    values = {
        "text_logit": text_logits,
        "graph_logit": graph_logits.reshape(-1),
    }
    fusion_logits = selected_fusion_logits(
        fusion_model, selected, values, device=device
    )
    expected_text = expected_text_logits[:probe_dates].reshape(-1)
    expected_graph = expected_graph_logits[:probe_dates]
    expected_fusion = expected_fusion_logits[:probe_dates].reshape(-1)
    checks = {
        "text": bool(np.allclose(text_logits, expected_text, atol=1e-6, rtol=1e-6)),
        "graph": bool(
            np.allclose(graph_logits, expected_graph, atol=1e-6, rtol=1e-6)
        ),
        "attention": bool(
            np.allclose(heads, expected_heads[:probe_dates], atol=1e-6, rtol=1e-6)
        ),
        "fusion": bool(
            np.allclose(fusion_logits, expected_fusion, atol=1e-6, rtol=1e-6)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Demo bundle save/load equivalence failed: {checks}")
    _freeze(text_model)
    _freeze(graph_model)
    _freeze(fusion_model)
    before = _model_hashes(text_model, graph_model, fusion_model)
    _ = predict_mlp(text_model, text_scaler.transform(probe_text), device=device)
    after = _model_hashes(text_model, graph_model, fusion_model)
    frozen = all(
        not parameter.requires_grad
        for module in (text_model, graph_model, fusion_model)
        for parameter in module.parameters()
    )
    if before != after or not frozen:
        raise RuntimeError("Frozen model state changed during replay equivalence probe")
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "probe_dates": probe_dates,
        "state_hash_before_replay": before,
        "state_hash_after_replay": after,
        "all_parameters_frozen": frozen,
    }


def _largest_attention_shifts(
    dates: np.ndarray,
    symbols: tuple[str, ...],
    attention: np.ndarray,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for date_index in range(1, len(dates)):
        delta = attention[date_index] - attention[date_index - 1]
        for target_index, target in enumerate(symbols):
            for source_index, source in enumerate(symbols):
                records.append(
                    {
                        "date": dates[date_index].isoformat(),
                        "previous_date": dates[date_index - 1].isoformat(),
                        "source": source,
                        "target": target,
                        "previous_alpha": float(
                            attention[date_index - 1, target_index, source_index]
                        ),
                        "current_alpha": float(
                            attention[date_index, target_index, source_index]
                        ),
                        "delta": float(delta[target_index, source_index]),
                        "absolute_delta": float(abs(delta[target_index, source_index])),
                        "head": "Mean across heads",
                    }
                )
    records.sort(key=lambda row: row["absolute_delta"], reverse=True)
    return records[:limit]


def _bundle_manifest(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    files = {
        path.relative_to(output_dir).as_posix(): sha256_path(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    manifest = {
        "bundle_version": "finswarm-e4-demo-fit-v1",
        "run_id": "finswarm_e4_demo_fit",
        "demo_only": True,
        "intentionally_in_sample": True,
        "evaluation_mode": "train_replay",
        "model_frozen_during_ui": True,
        "selected_text_model": "canonical_text_score_head",
        "selected_graph_model": "canonical_dense_signed_gat",
        "selected_fusion_model": "scalar_logit_stack",
        "train_range": [TRAIN_START, TRAIN_END],
        "train_dates": EXPECTED_TRAIN_DATES,
        "aligned_replay_rows": EXPECTED_ALIGNED_ROWS,
        "supervised_train_rows": metadata["supervised_train_rows"],
        "node_order": list(NODE_ORDER),
        "attention_convention": ATTENTION_CONVENTION,
        "validation_access_count": 0,
        "test_access_count": 0,
        "file_sha256": files,
        "model_hashes": metadata["model_hashes"],
    }
    atomic_json(output_dir / "bundle_manifest.json", manifest)
    return manifest


def train_demo_fit(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    config = load_demo_config(config_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = config.training.model_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Demo-fit output already exists; refusing to overwrite: {output_dir}"
        )
    build_dir = output_dir.parent / f".{output_dir.name}.build-{uuid.uuid4().hex}"
    build_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_snapshot()
    try:
        print(f"[Demo Fit] device={device} output={output_dir}", flush=True)
        splits = _train_only_splits(config.training)
        guard = SplitAccessGuard()
        supervised = load_supervised_text_dataset(
            config.training,
            splits,
            guard=guard,
            allowed_splits=("train",),
            allowed_dates=splits.train_effective,
            purpose="fit",
        )
        if len(supervised) != EXPECTED_SUPERVISED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_SUPERVISED_ROWS} supervised rows, got {len(supervised)}"
            )
        print(
            f"[Demo Fit] effective_train_dates={len(splits.train_effective)} "
            f"supervised_rows={len(supervised)}",
            flush=True,
        )
        embedding_rows, pending_rows, source_access = _all_train_rows(
            config, splits.train_effective
        )

        # Load graph identity first; Text overrides are applied after in-sample fitting.
        placeholder_scores = {
            str(row["decision_id"]): 0.5 for row in embedding_rows
        }
        raw_graph = load_graph_dataset(
            config.training,
            splits,
            dates=splits.train_effective,
            purpose="fit",
            guard=guard,
            text_scores_by_decision=placeholder_scores,
            require_oof_text_scores=True,
        )
        if raw_graph.symbols != NODE_ORDER:
            raise ValueError(f"Unexpected graph node order: {raw_graph.symbols}")
        if len(raw_graph) != EXPECTED_TRAIN_DATES:
            raise ValueError(f"Expected {EXPECTED_TRAIN_DATES} graph dates")
        if raw_graph.supervised_count != EXPECTED_SUPERVISED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_SUPERVISED_ROWS} graph labels, got "
                f"{raw_graph.supervised_count}"
            )
        text_embeddings, text_inputs, ordered_ids, _ = _ordered_text_arrays(
            raw_graph, embedding_rows
        )
        if not np.array_equal(ordered_ids.astype(str), raw_graph.decision_ids.astype(str)):
            raise ValueError("Text/Graph replay decision IDs are not aligned")

        text_scaler = TrainOnlyScaler(
            [f"embedding_{index:03d}" for index in range(384)]
        ).fit(
            supervised.embeddings,
            dates=supervised.dates.tolist(),
            input_manifest_hash=sha256_path(
                config.training.artifact_dir / "data_manifest.json"
            ),
        )
        text_scaler.save(build_dir / "text_scaler.json")
        text_model, text_history = _fit_text_demo(
            text_scaler.transform(supervised.embeddings),
            supervised.targets,
            supervised.symbols,
            config,
            device=device,
        )
        print(
            f"[Text] selected_epoch={text_history['best_epoch']} "
            f"train_auc={text_history['train_metrics']['roc_auc']:.4f}",
            flush=True,
        )
        all_text_scaled = text_scaler.transform(
            text_embeddings.reshape(-1, text_embeddings.shape[-1])
        )
        all_text_logits_flat, all_text_probability_flat = predict_mlp(
            text_model, all_text_scaled, device=device
        )
        all_text_logits = all_text_logits_flat.reshape(
            len(raw_graph), len(raw_graph.symbols)
        ).astype(np.float32)
        all_text_probability = all_text_probability_flat.reshape(
            len(raw_graph), len(raw_graph.symbols)
        ).astype(np.float32)
        text_scores = {
            str(ordered_ids[date_index, node]): float(
                all_text_probability[date_index, node]
            )
            for date_index in range(len(raw_graph))
            for node in range(len(raw_graph.symbols))
        }
        raw_graph = load_graph_dataset(
            config.training,
            splits,
            dates=splits.train_effective,
            purpose="fit",
            guard=guard,
            text_scores_by_decision=text_scores,
            require_oof_text_scores=True,
        )
        raw_graph.metadata["text_score_source"] = TEXT_SCORE_SOURCE
        raw_graph.metadata["strict_text_scores_required_for_all_replay_nodes"] = True
        if not bool(raw_graph.feature_mask[:, :, raw_graph.feature_names.index("text_score")].all()):
            raise ValueError("Demo Graph contains a missing in-sample text_score")

        graph_scaler = fit_graph_scaler(
            raw_graph,
            input_manifest_hash=sha256_path(
                config.training.artifact_dir / "data_manifest.json"
            ),
        )
        graph_scaler.save(build_dir / "graph_scaler.json")
        scaled_graph = scale_graph_dataset(raw_graph, graph_scaler)
        graph_model, graph_history = _fit_graph_demo(
            scaled_graph, config, device=device
        )
        print(
            f"[Graph] selected_epoch={graph_history['best_epoch']} "
            f"train_auc={graph_history['train_metrics']['roc_auc']:.4f}",
            flush=True,
        )
        graph_logits, graph_probability, attention_heads, attention_mean = (
            _attention_outputs(graph_model, scaled_graph, device=device)
        )
        if attention_heads.shape != (
            EXPECTED_TRAIN_DATES,
            int(config.training.graph_model["num_heads"]),
            len(NODE_ORDER),
            len(NODE_ORDER),
        ):
            raise ValueError(f"Unexpected attention shape: {attention_heads.shape}")
        if not np.allclose(attention_heads.sum(axis=-1), 1.0, atol=1e-6):
            raise ValueError("GAT attention is not normalized across sources")

        supervised_mask = raw_graph.label_mask
        target_values = raw_graph.targets[supervised_mask]
        symbol_grid = np.broadcast_to(np.asarray(NODE_ORDER), raw_graph.targets.shape)
        supervised_symbols = symbol_grid[supervised_mask]
        fusion_model, fusion_history = _fit_scalar_demo(
            all_text_logits[supervised_mask],
            graph_logits[supervised_mask],
            target_values,
            supervised_symbols,
            config,
            device=device,
        )
        print(
            f"[Scalar Fusion] selected_epoch={fusion_history['best_epoch']} "
            f"train_auc={fusion_history['train_metrics']['roc_auc']:.4f}",
            flush=True,
        )
        with torch.no_grad():
            fusion_logits_flat = fusion_model(
                torch.as_tensor(
                    all_text_logits.reshape(-1), dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    graph_logits.reshape(-1), dtype=torch.float32, device=device
                ),
            ).detach().cpu().numpy()
        fusion_logits = fusion_logits_flat.reshape(
            EXPECTED_TRAIN_DATES, len(NODE_ORDER)
        ).astype(np.float32)
        fusion_probability = _probability(fusion_logits).astype(np.float32)

        _freeze(text_model)
        _freeze(graph_model)
        _freeze(fusion_model)
        model_hashes = _model_hashes(text_model, graph_model, fusion_model)
        text_checkpoint = {
            "model_type": "canonical_text_score_head",
            "architecture": text_model.architecture_config(),
            "state_dict": text_model.state_dict(),
            "seed": config.training.seed,
            "best_epoch": text_history["best_epoch"],
            "checkpoint_selection": "minimum_train_loss",
            "training_source": "effective_train_in_sample_demo",
            "prediction_source": TEXT_SCORE_SOURCE,
            "frozen_for_ui": True,
        }
        graph_checkpoint = {
            "model_type": "canonical_dense_signed_gat",
            "architecture": graph_model.architecture_config(),
            "state_dict": graph_model.state_dict(),
            "feature_names": list(raw_graph.feature_names),
            "topology": "full",
            "include_text_score": True,
            "symbols": list(raw_graph.symbols),
            "seed": config.training.seed,
            "best_epoch": graph_history["best_epoch"],
            "checkpoint_selection": "minimum_train_loss",
            "training_source": "effective_train_in_sample_demo",
            "text_score_source": TEXT_SCORE_SOURCE,
            "prediction_source": GRAPH_PREDICTION_SOURCE,
            "attention_convention": ATTENTION_CONVENTION,
            "frozen_for_ui": True,
        }
        fusion_checkpoint = {
            "model_type": SCALAR_FUSION_SELECTION,
            "architecture": fusion_model.architecture_config(),
            "state_dict": fusion_model.state_dict(),
            "selected_fusion_model": SCALAR_FUSION_SELECTION,
            "seed": config.training.seed,
            "best_epoch": fusion_history["best_epoch"],
            "checkpoint_selection": "minimum_train_loss",
            "training_source": "aligned_effective_train_in_sample_demo",
            "prediction_source": FUSION_PREDICTION_SOURCE,
            "coefficients": fusion_history["coefficients"],
            "regularization": fusion_history["regularization"],
            "frozen_for_ui": True,
        }
        atomic_torch_save(build_dir / "text_model.pt", text_checkpoint)
        atomic_torch_save(build_dir / "graph_model.pt", graph_checkpoint)
        atomic_torch_save(build_dir / "fusion_model.pt", fusion_checkpoint)
        atomic_json(build_dir / "text_training_history.json", text_history)
        atomic_json(build_dir / "graph_training_history.json", graph_history)
        atomic_json(build_dir / "fusion_training_history.json", fusion_history)
        atomic_json(build_dir / "feature_schema.json", graph_feature_schema(config.training))
        atomic_json(build_dir / "node_order.json", {"symbols": list(NODE_ORDER)})
        atomic_json(build_dir / "attention_contract.json", ATTENTION_CONVENTION)
        atomic_json(build_dir / "model_state_hashes.json", model_hashes)
        shutil.copy2(Path(config_path).resolve(), build_dir / "training_config.yaml")

        date_strings = np.asarray(
            [value.isoformat() for value in raw_graph.dates], dtype="U10"
        )
        _atomic_npz(
            build_dir / "replay_inputs.npz",
            dates=date_strings,
            symbols=np.asarray(NODE_ORDER, dtype="U4"),
            decision_ids=raw_graph.decision_ids.astype("U64"),
            text_embeddings=text_embeddings.astype(np.float32),
            raw_graph_features=raw_graph.features.astype(np.float32),
            graph_feature_mask=raw_graph.feature_mask.astype(bool),
            adjacency=raw_graph.adjacency.astype(np.float32),
            graph_feature_names=np.asarray(raw_graph.feature_names, dtype="U48"),
        )
        _atomic_npz(
            build_dir / "replay_expected_outputs.npz",
            text_logits=all_text_logits,
            text_probability=all_text_probability,
            graph_logits=graph_logits,
            graph_probability=graph_probability,
            attention_heads=attention_heads,
            attention_mean=attention_mean,
            fusion_logits=fusion_logits,
            fusion_probability=fusion_probability,
        )
        replay_rows, replay_news = _materialize_replay_records(
            build_dir, raw_graph, pending_rows, text_inputs, config
        )
        replay_index = {
            "dates": date_strings.tolist(),
            "symbols": list(NODE_ORDER),
            "display_order": list(DISPLAY_ORDER),
            "decision_ids_by_date": {
                date_strings[index]: raw_graph.decision_ids[index].astype(str).tolist()
                for index in range(len(raw_graph))
            },
            "train_range": [TRAIN_START, TRAIN_END],
            "date_count": len(raw_graph),
            "aligned_row_count": replay_rows,
            "supervised_train_row_count": int(raw_graph.supervised_count),
            "news_row_count": replay_news,
            "text_score_source": TEXT_SCORE_SOURCE,
            "graph_prediction_source": GRAPH_PREDICTION_SOURCE,
            "fusion_prediction_source": FUSION_PREDICTION_SOURCE,
        }
        atomic_json(build_dir / "replay_index.json", replay_index)
        shifts = _largest_attention_shifts(
            raw_graph.dates, raw_graph.symbols, attention_mean
        )
        atomic_json(build_dir / "attention_shift_summary.json", {"largest": shifts})

        equivalence = _verify_save_load(
            build_dir,
            text_embeddings,
            raw_graph,
            all_text_logits,
            graph_logits,
            attention_heads,
            fusion_logits,
            device=device,
        )
        protected_after = protected_snapshot()
        protected_unchanged = protected_before == protected_after
        if not protected_unchanged:
            raise RuntimeError("A protected canonical/research artifact changed during demo fit")
        source_access["guard_access_log"] = guard.access_log
        source_access["test_feature_access_count"] = guard.test_feature_access_count
        if guard.test_feature_access_count != 0:
            raise RuntimeError("Demo fitting accessed the test split")
        metadata = {
            "run_id": config.training.run_id,
            "demo_only": True,
            "intentionally_in_sample": True,
            "evaluation_mode": "train_replay",
            "model_frozen_during_ui": True,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "train_dates": len(raw_graph),
            "aligned_replay_rows": replay_rows,
            "supervised_train_rows": int(raw_graph.supervised_count),
            "supervised_rows_by_symbol": {
                symbol: int(np.sum(supervised.symbols == symbol))
                for symbol in NODE_ORDER
            },
            "validation_access_count": 0,
            "test_access_count": 0,
            "validation_labels_accessed": False,
            "test_labels_accessed": False,
            "test_features_accessed": False,
            "checkpoint_selection": "minimum_train_loss",
            "validation_early_stopping": False,
            "calibration_fitted": False,
            "node_order": list(NODE_ORDER),
            "attention_shape": list(attention_heads.shape),
            "attention_convention": ATTENTION_CONVENTION,
            "text_score_source": TEXT_SCORE_SOURCE,
            "graph_prediction_source": GRAPH_PREDICTION_SOURCE,
            "fusion_prediction_source": FUSION_PREDICTION_SOURCE,
            "text_metrics": text_history["train_metrics"],
            "graph_metrics": graph_history["train_metrics"],
            "fusion_metrics": fusion_history["train_metrics"],
            "fusion_coefficients": fusion_history["coefficients"],
            "model_hashes": model_hashes,
            "save_load_prediction_equivalence": equivalence,
            "source_access": source_access,
            "protected_artifact_snapshot_before": protected_before,
            "protected_artifact_snapshot_after": protected_after,
            "protected_artifacts_unchanged": protected_unchanged,
            "device": device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        atomic_json(build_dir / "training_metadata.json", metadata)
        _bundle_manifest(build_dir, metadata)
        os.replace(build_dir, output_dir)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    from demo.scenario_builder import build_scenarios

    scenarios = build_scenarios(output_dir)
    result = {
        "model_dir": output_dir.as_posix(),
        "device": device,
        "train_range": [TRAIN_START, TRAIN_END],
        "train_dates": EXPECTED_TRAIN_DATES,
        "aligned_replay_rows": EXPECTED_ALIGNED_ROWS,
        "supervised_train_rows": EXPECTED_SUPERVISED_ROWS,
        "text_metrics": text_history["train_metrics"],
        "graph_metrics": graph_history["train_metrics"],
        "fusion_metrics": fusion_history["train_metrics"],
        "model_hashes": model_hashes,
        "scenario_count": len(scenarios),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the isolated E4 in-sample historical replay bundle."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    result = train_demo_fit(args.config, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
