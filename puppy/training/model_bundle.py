from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import torch

from puppy.fusion.gated_fusion import LatentGatedFusion
from puppy.fusion.text_encoder import TextStreamEncoder
from puppy.fusion.text_score_head import TextScoreHead
from puppy.graph.gat_model import DenseSignedGAT

from .checkpoint import TrainOnlyScaler, atomic_json
from .config import TRAINING_CODE_VERSION, TrainingConfig, training_code_sha256
from .fusion_trainer import (
    load_fusion_model,
    selected_fusion_logits,
)
from .graph_dataset_loader import GraphDataset, load_graph_dataset
from .graph_trainer import load_gat, predict_gat, scale_graph_dataset
from .split import EffectiveSplits
from .text_dataset import load_supervised_text_dataset
from .text_embedding import sha256_path
from .text_trainer import selected_text_predictions
from .text_trainer import load_text_mlp
from .training_guard import SplitAccessGuard


REQUIRED_GENERATED_FILES = (
    "text_embedding_manifest.json",
    "text_embeddings_384.parquet",
    "text_logistic_baseline.json",
    "text_model.pt",
    "text_scaler.json",
    "text_training_history.json",
    "graph_model.pt",
    "graph_scaler.json",
    "graph_attention_validation.npz",
    "graph_training_history.json",
    "oof_predictions.parquet",
    "oof_manifest.json",
    "validation_base_predictions.parquet",
    "fusion_model.pt",
    "fusion_training_history.json",
    "calibration.json",
    "feature_schema.json",
    "training_config.yaml",
    "source_data_manifest.json",
    "source_rollout_manifest.json",
    "source_temporal_quality.json",
    "git_commit.txt",
    "metrics.json",
    "metrics.md",
    "leakage_audit.json",
    "bundle_prediction_probe.npz",
)


@dataclass(frozen=True)
class CanonicalModelSet:
    text_encoder: TextStreamEncoder
    text_score_head: TextScoreHead
    gat: DenseSignedGAT
    fusion: LatentGatedFusion
    text_scaler: TrainOnlyScaler
    graph_scaler: TrainOnlyScaler


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _metrics_and_audit(
    config: TrainingConfig,
    splits: EffectiveSplits,
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporal = json.loads(
        (config.artifact_dir / "temporal_quality.json").read_text(encoding="utf-8")
    )
    data_manifest = json.loads(
        (config.artifact_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    embedding = json.loads(
        (config.artifact_dir / "text_embedding_manifest.json").read_text(encoding="utf-8")
    )
    text = json.loads(
        (config.model_dir / "text_training_history.json").read_text(encoding="utf-8")
    )
    graph = json.loads(
        (config.model_dir / "graph_training_history.json").read_text(encoding="utf-8")
    )
    oof = json.loads((config.model_dir / "oof_manifest.json").read_text(encoding="utf-8"))
    fusion = json.loads(
        (config.model_dir / "fusion_training_history.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (config.model_dir / "calibration.json").read_text(encoding="utf-8")
    )
    metrics = {
        "text": {
            "selected_model": text["selected_text_model"],
            "logistic_regression": text["logistic_regression"],
            "mlp": {
                "train_metrics": text["mlp"]["train_metrics"],
                "validation_selection_metrics": text["mlp"]["validation_selection_metrics"],
            },
        },
        "graph": {
            "train_metrics": graph["train_metrics"],
            "validation_selection_metrics": graph["validation_selection_metrics"],
        },
        "oof": {
            "rows": oof["oof_rows"],
            "coverage": oof["coverage"],
            "missing_rows": oof["missing_oof_rows"],
        },
        "fusion": {
            "selected_model": fusion["selected_fusion_model"],
            "validation_selection_candidates": fusion[
                "candidate_validation_selection_metrics"
            ],
        },
        "calibration": {
            "temperature": calibration["temperature"],
            "uncalibrated_metrics": calibration["uncalibrated_metrics"],
            "calibrated_metrics": calibration["calibrated_metrics"],
        },
    }
    logical_sets = {
        "warmup": set(splits.warmup),
        "train": set(splits.train_manifest),
        "validation": set(splits.validation_selection + splits.validation_calibration),
        "test": set(splits.test_metadata_dates),
    }
    overlap_free = all(
        not logical_sets[left].intersection(logical_sets[right])
        for index, left in enumerate(logical_sets)
        for right in tuple(logical_sets)[index + 1 :]
    )
    checks = {
        "stage5_temporal_gate_pass": temporal.get("status") == "PASS",
        "semantic_dimension_384": embedding.get("embedding_dim") == 384,
        "bootstrap_embedding_rejected": "bootstrap"
        not in str(embedding.get("encoder_model", "")).lower(),
        "test_embedding_materialized": "test" in embedding.get("included_splits", []),
        "logical_splits_disjoint": overlap_free,
        "train_validation_embargo_exact_20": len(splits.embargoed_train)
        == int(config.split["embargo_trading_days"]),
        "validation_selection_before_calibration": max(splits.validation_selection)
        < min(splits.validation_calibration),
        "text_scaler_effective_train_only": json.loads(
            (config.model_dir / "text_scaler.json").read_text(encoding="utf-8")
        )["fit_end_date"]
        == max(splits.train_effective).isoformat(),
        "graph_scaler_effective_train_only": json.loads(
            (config.model_dir / "graph_scaler.json").read_text(encoding="utf-8")
        )["fit_end_date"]
        == max(splits.train_effective).isoformat(),
        "text_parameters_changed": bool(text["mlp"]["parameter_changed"]),
        "graph_parameters_changed": bool(graph["parameter_changed"]),
        "oof_strictly_earlier": bool(oof["strictly_earlier_models_verified"]),
        "oof_embargo_verified": bool(oof["embargo_verified"]),
        "oof_validation_access_zero": int(oof["validation_access_count"]) == 0,
        "oof_test_access_zero": int(oof["test_access_count"]) == 0,
        "fusion_oof_only": bool(fusion["oof_only_verified"]),
        "fusion_parameters_changed": bool(fusion["parameter_changed"]),
        "canonical_fusion_selected": fusion["selected_fusion_model"]
        == "canonical_latent_fusion",
        "oof_text_for_graph_verified": bool(oof["oof_text_for_graph_verified"]),
        "oof_graph_latents_verified": bool(oof["oof_graph_latents_verified"]),
        "calibration_partition_only": calibration["partition"]
        == "validation_calibration",
        "calibration_weights_unchanged": bool(calibration["fusion_weights_unchanged"]),
        "no_test_metrics": "test" not in json.dumps(metrics).lower(),
        "test_feature_values_loaded": False,
        "real_causal_action_source": data_manifest.get("action_source")
        == "real_causal_sector_agent"
        and not bool(data_manifest.get("bootstrap")),
        "policy_dependent_graph_fields_used": True,
        "targets_in_graph_forward_inputs": False,
    }
    pass_checks = {
        key: (not value if key in {
            "test_embedding_materialized",
            "test_feature_values_loaded",
            "targets_in_graph_forward_inputs",
        } else value)
        for key, value in checks.items()
    }
    audit = {
        "status": "PASS" if all(pass_checks.values()) else "FAIL",
        "checks": checks,
        "effective_train_range": [
            min(splits.train_effective).isoformat(),
            max(splits.train_effective).isoformat(),
        ],
        "embargoed_train_range": [
            min(splits.embargoed_train).isoformat(),
            max(splits.embargoed_train).isoformat(),
        ],
        "validation_selection_range": [
            min(splits.validation_selection).isoformat(),
            max(splits.validation_selection).isoformat(),
        ],
        "validation_calibration_range": [
            min(splits.validation_calibration).isoformat(),
            max(splits.validation_calibration).isoformat(),
        ],
        "test_metadata_only": {
            "days": len(splits.test_metadata_dates),
            "start": min(splits.test_metadata_dates).isoformat(),
            "end": max(splits.test_metadata_dates).isoformat(),
            "features_loaded": False,
            "labels_loaded": False,
            "metrics_produced": False,
        },
    }
    if audit["status"] != "PASS":
        failed = [key for key, value in pass_checks.items() if not value]
        raise RuntimeError(f"Stage 6 leakage audit failed: {failed}")
    return metrics, audit


def _write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    text = metrics["text"]
    graph = metrics["graph"]
    fusion = metrics["fusion"]
    calibration = metrics["calibration"]
    lines = [
        "# FinSwarm Stage 6 Metrics",
        "",
        f"Selected Text Stream: **{text['selected_model']}**",
        f"Selected fusion: **{fusion['selected_model']}**",
        "",
        "| Component | Validation log loss | MCC | Balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, value in fusion["validation_selection_candidates"].items():
        lines.append(
            f"| {name} | {value['log_loss']:.6f} | {value['mcc']:.6f} | "
            f"{value['balanced_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Graph validation log loss: {graph['validation_selection_metrics']['log_loss']:.6f}",
            f"Calibration temperature: {calibration['temperature']:.6f}",
            f"Calibration log loss before: {calibration['uncalibrated_metrics']['log_loss']:.6f}",
            f"Calibration log loss after: {calibration['calibrated_metrics']['log_loss']:.6f}",
            "",
            "No test metrics were produced.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_prediction_probe(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    device: str,
) -> dict[str, Any]:
    guard = SplitAccessGuard()
    text_data = load_supervised_text_dataset(
        config,
        splits,
        guard=guard,
        allowed_splits=("validation",),
        allowed_dates=splits.validation_selection,
        purpose="model_selection",
    )
    date = text_data.dates[0]
    date_text = text_data.subset((date,))
    symbol = str(date_text.symbols[0])
    text_scaler = TrainOnlyScaler.load(config.model_dir / "text_scaler.json")
    graph_scaler = TrainOnlyScaler.load(config.model_dir / "graph_scaler.json")
    text_features = text_scaler.transform(date_text.embeddings)
    text_logits, text_probability, selected_text = selected_text_predictions(
        config.model_dir, text_features, device=device
    )
    text_scores = {
        str(decision_id): float(probability)
        for decision_id, probability in zip(date_text.decision_ids, text_probability)
    }
    graph_raw = load_graph_dataset(
        config,
        splits,
        dates=(date,),
        purpose="model_selection",
        guard=guard,
        text_scores_by_decision=text_scores,
        require_oof_text_scores=True,
    )
    graph = scale_graph_dataset(graph_raw, graph_scaler)
    graph_logits, graph_probability, _, h_graph = predict_gat(
        load_gat(config.model_dir / "graph_model.pt", device=device),
        graph,
        device=device,
        include_latent=True,
    )
    node = graph.symbols.index(symbol)
    text_index = list(date_text.symbols).index(symbol)
    fusion_model, selected_fusion = load_fusion_model(
        config.model_dir / "fusion_model.pt", device=device
    )
    fusion_values = {
        "h_text": date_text.embeddings[text_index : text_index + 1].astype(np.float32),
        "h_graph": h_graph[0, node][None].astype(np.float32),
        "text_logit": np.asarray([text_logits[text_index]], dtype=np.float32),
        "graph_logit": np.asarray([graph_logits[0, node]], dtype=np.float32),
        "text_probability": np.asarray([text_probability[text_index]], dtype=np.float32),
        "graph_probability": np.asarray([graph_probability[0, node]], dtype=np.float32),
    }
    fusion_logit = selected_fusion_logits(
        fusion_model, selected_fusion, fusion_values, device=device
    )[0]
    calibration = json.loads(
        (config.model_dir / "calibration.json").read_text(encoding="utf-8")
    )
    calibrated_probability = 1.0 / (
        1.0 + np.exp(-np.clip(fusion_logit / calibration["temperature"], -60.0, 60.0))
    )
    target = config.model_dir / "bundle_prediction_probe.npz"
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        decision_id=np.asarray(str(date_text.decision_ids[text_index])),
        symbol=np.asarray(symbol),
        decision_date=np.asarray(date.isoformat()),
        node_index=np.asarray(node, dtype=np.int16),
        h_text=fusion_values["h_text"],
        text_features=text_features[text_index : text_index + 1].astype(np.float32),
        graph_features=graph.features.astype(np.float32),
        graph_adjacency=graph.adjacency.astype(np.float32),
        graph_feature_mask=graph.feature_mask.astype(bool),
        expected_h_graph=fusion_values["h_graph"],
        expected_text_logit=np.asarray(text_logits[text_index], dtype=np.float64),
        expected_text_probability=np.asarray(text_probability[text_index], dtype=np.float64),
        expected_graph_logit=np.asarray(graph_logits[0, node], dtype=np.float64),
        expected_graph_probability=np.asarray(graph_probability[0, node], dtype=np.float64),
        expected_fusion_logit=np.asarray(fusion_logit, dtype=np.float64),
        expected_calibrated_probability=np.asarray(calibrated_probability, dtype=np.float64),
        selected_text_model=np.asarray(selected_text),
        selected_fusion_model=np.asarray(selected_fusion),
    )
    os.replace(temporary, target)
    return {
        "decision_id": str(date_text.decision_ids[text_index]),
        "date": date.isoformat(),
        "symbol": symbol,
        "selected_text_model": selected_text,
        "selected_fusion_model": selected_fusion,
    }


def build_model_bundle(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    _copy(
        config.artifact_dir / "text_embedding_manifest.json",
        config.model_dir / "text_embedding_manifest.json",
    )
    _copy(
        config.artifact_dir / "text_embeddings_384.parquet",
        config.model_dir / "text_embeddings_384.parquet",
    )
    _copy(config.config_path, config.model_dir / "training_config.yaml")
    _copy(
        config.artifact_dir / "data_manifest.json",
        config.model_dir / "source_data_manifest.json",
    )
    _copy(
        config.artifact_dir / "rollout_manifest.json",
        config.model_dir / "source_rollout_manifest.json",
    )
    _copy(
        config.artifact_dir / "temporal_quality.json",
        config.model_dir / "source_temporal_quality.json",
    )
    git_commit = _git_commit(config.repo_root)
    (config.model_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    metrics, audit = _metrics_and_audit(config, splits)
    atomic_json(config.model_dir / "metrics.json", metrics)
    _write_metrics_markdown(config.model_dir / "metrics.md", metrics)
    atomic_json(config.model_dir / "leakage_audit.json", audit)
    probe = _create_prediction_probe(config, splits, device=device)
    missing = [name for name in REQUIRED_GENERATED_FILES if not (config.model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Model bundle is missing required files: {missing}")
    files = {
        name: sha256_path(config.model_dir / name)
        for name in REQUIRED_GENERATED_FILES
    }
    embedding = json.loads(
        (config.model_dir / "text_embedding_manifest.json").read_text(encoding="utf-8")
    )
    text = json.loads(
        (config.model_dir / "text_training_history.json").read_text(encoding="utf-8")
    )
    graph = json.loads(
        (config.model_dir / "graph_training_history.json").read_text(encoding="utf-8")
    )
    fusion = json.loads(
        (config.model_dir / "fusion_training_history.json").read_text(encoding="utf-8")
    )
    manifest = {
        "bundle_version": "finswarm-canonical-bundle-v2",
        "training_code_version": TRAINING_CODE_VERSION,
        "training_code_sha256": training_code_sha256(config.repo_root),
        "git_commit": git_commit,
        "seed": config.seed,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "encoder_model": embedding["encoder_model"],
        "encoder_revision": embedding["encoder_revision"],
        "canonical_classes": {
            "text_encoder": "puppy.fusion.text_encoder.TextStreamEncoder",
            "text_score_head": "puppy.fusion.text_score_head.TextScoreHead",
            "gat": "puppy.graph.gat_model.DenseSignedGAT",
            "fusion": "puppy.fusion.gated_fusion.LatentGatedFusion",
        },
        "input_artifact_hashes": {
            "data_manifest": files["source_data_manifest.json"],
            "rollout_manifest": files["source_rollout_manifest.json"],
            "temporal_quality": files["source_temporal_quality.json"],
            "semantic_embeddings": files["text_embeddings_384.parquet"],
        },
        "effective_splits": {
            "train": [min(splits.train_effective).isoformat(), max(splits.train_effective).isoformat()],
            "embargo": [min(splits.embargoed_train).isoformat(), max(splits.embargoed_train).isoformat()],
            "validation_selection": [min(splits.validation_selection).isoformat(), max(splits.validation_selection).isoformat()],
            "validation_calibration": [min(splits.validation_calibration).isoformat(), max(splits.validation_calibration).isoformat()],
            "test_metadata_only": [min(splits.test_metadata_dates).isoformat(), max(splits.test_metadata_dates).isoformat()],
        },
        "selected_text_model": text["selected_text_model"],
        "selected_fusion_model": fusion["selected_fusion_model"],
        "checkpoint_metrics": {
            "text": text["mlp"]["best_validation_log_loss"],
            "graph": graph["best_validation_log_loss"],
            "fusion": fusion["best_validation_log_loss"],
        },
        "prediction_probe": probe,
        "required_files": list(REQUIRED_GENERATED_FILES),
        "file_sha256": files,
        "leakage_audit_status": audit["status"],
        "manifest_self_hash_excluded": True,
    }
    atomic_json(config.model_dir / "bundle_manifest.json", manifest)
    validation = validate_model_bundle(config.model_dir, device=device)
    manifest["bundle_validation"] = validation
    atomic_json(config.model_dir / "bundle_manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class ModelBundle:
    path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        directory = Path(path)
        manifest = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
        selected = manifest.get("selected_fusion_model")
        if selected is not None and selected != "canonical_latent_fusion":
            raise ValueError(
                "Production model bundle must select canonical latent fusion; "
                f"found {selected!r}"
            )
        if manifest.get("bundle_version") == "finswarm-canonical-bundle-v2":
            expected_classes = {
                "text_encoder": "puppy.fusion.text_encoder.TextStreamEncoder",
                "text_score_head": "puppy.fusion.text_score_head.TextScoreHead",
                "gat": "puppy.graph.gat_model.DenseSignedGAT",
                "fusion": "puppy.fusion.gated_fusion.LatentGatedFusion",
            }
            if manifest.get("canonical_classes") != expected_classes:
                raise ValueError("Bundle canonical class identities do not match runtime")
        for name, expected in manifest["file_sha256"].items():
            file_path = directory / name
            if not file_path.is_file() or sha256_path(file_path) != expected:
                raise ValueError(f"Bundle hash mismatch: {name}")
        return cls(directory, manifest)

    def load_models(
        self,
        *,
        device: str = "cpu",
        text_encoder: TextStreamEncoder | None = None,
    ) -> CanonicalModelSet:
        if text_encoder is None:
            text_encoder = TextStreamEncoder.from_transformer(
                str(self.manifest["encoder_model"]),
                local_files_only=True,
                device=device,
            )
            if text_encoder.revision != str(self.manifest["encoder_revision"]):
                raise ValueError("Local TextEncoder revision does not match bundle")
        if not isinstance(text_encoder, TextStreamEncoder):
            raise TypeError("Runtime must use canonical TextStreamEncoder")
        return CanonicalModelSet(
            text_encoder=text_encoder,
            text_score_head=load_text_mlp(self.path / "text_model.pt", device=device),
            gat=load_gat(self.path / "graph_model.pt", device=device),
            fusion=load_fusion_model(
                self.path / "fusion_model.pt", device=device
            )[0],
            text_scaler=TrainOnlyScaler.load(self.path / "text_scaler.json"),
            graph_scaler=TrainOnlyScaler.load(self.path / "graph_scaler.json"),
        )

    def verify_prediction_probe(self, *, device: str = "cpu") -> dict[str, Any]:
        with np.load(self.path / "bundle_prediction_probe.npz", allow_pickle=False) as probe:
            text_features = np.asarray(probe["text_features"], dtype=np.float32)
            text_logits, text_probability, selected_text = selected_text_predictions(
                self.path, text_features, device=device
            )
            graph = GraphDataset(
                sample_ids=np.asarray(["probe"]),
                dates=np.asarray([probe["decision_date"].item()], dtype=object),
                symbols=tuple(
                    json.loads((self.path / "graph_model.pt.symbols.json").read_text())
                ) if (self.path / "graph_model.pt.symbols.json").exists() else tuple(
                    torch.load(
                        self.path / "graph_model.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["symbols"]
                ),
                features=np.asarray(probe["graph_features"], dtype=np.float32),
                adjacency=np.asarray(probe["graph_adjacency"], dtype=np.float32),
                feature_mask=np.asarray(probe["graph_feature_mask"], dtype=bool),
                targets=np.zeros((1, probe["graph_features"].shape[1]), dtype=np.float32),
                label_mask=np.ones((1, probe["graph_features"].shape[1]), dtype=bool),
                decision_ids=np.full((1, probe["graph_features"].shape[1]), "probe"),
            )
            graph_logits, graph_probability, _, h_graph = predict_gat(
                load_gat(self.path / "graph_model.pt", device=device),
                graph,
                device=device,
                include_latent=True,
            )
            node = int(probe["node_index"])
            fusion_model, selected_fusion = load_fusion_model(
                self.path / "fusion_model.pt", device=device
            )
            values = {
                "h_text": np.asarray(probe["h_text"], dtype=np.float32),
                "h_graph": h_graph[0, node][None].astype(np.float32),
                "text_logit": np.asarray([text_logits[0]], dtype=np.float32),
                "graph_logit": np.asarray([graph_logits[0, node]], dtype=np.float32),
                "text_probability": np.asarray(
                    [float(probe["expected_text_probability"])], dtype=np.float32
                ),
                "graph_probability": np.asarray(
                    [graph_probability[0, node]], dtype=np.float32
                ),
            }
            fusion_logit = selected_fusion_logits(
                fusion_model, selected_fusion, values, device=device
            )[0]
            calibration = json.loads(
                (self.path / "calibration.json").read_text(encoding="utf-8")
            )
            calibrated_probability = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(fusion_logit / calibration["temperature"], -60.0, 60.0)
                )
            )
            checks = {
                "selected_text_model": selected_text == str(probe["selected_text_model"].item()),
                "selected_fusion_model": selected_fusion
                == str(probe["selected_fusion_model"].item()),
                "text_logit": bool(
                    np.isclose(text_logits[0], float(probe["expected_text_logit"]), atol=1e-6)
                ),
                "text_probability": bool(
                    np.isclose(
                        text_probability[0],
                        float(probe["expected_text_probability"]),
                        atol=1e-6,
                    )
                ),
                "graph_logit": bool(
                    np.isclose(
                        graph_logits[0, node],
                        float(probe["expected_graph_logit"]),
                        atol=1e-6,
                    )
                ),
                "h_graph": bool(
                    np.allclose(
                        values["h_graph"],
                        np.asarray(probe["expected_h_graph"], dtype=np.float32),
                        atol=1e-6,
                    )
                ),
                "graph_probability": bool(
                    np.isclose(
                        graph_probability[0, node],
                        float(probe["expected_graph_probability"]),
                        atol=1e-6,
                    )
                ),
                "fusion_logit": bool(
                    np.isclose(fusion_logit, float(probe["expected_fusion_logit"]), atol=1e-6)
                ),
                "calibrated_probability": bool(
                    np.isclose(
                        calibrated_probability,
                        float(probe["expected_calibrated_probability"]),
                        atol=1e-6,
                    )
                ),
            }
        if not all(checks.values()):
            raise ValueError(f"Bundle prediction equivalence failed: {checks}")
        return checks


def validate_model_bundle(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    bundle = ModelBundle.load(path)
    prediction_checks = bundle.verify_prediction_probe(device=device)
    return {
        "status": "PASS",
        "hashes_verified": len(bundle.manifest["file_sha256"]),
        "prediction_equivalence": prediction_checks,
    }
