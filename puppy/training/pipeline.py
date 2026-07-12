from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .calibration import calibrate_fusion
from .checkpoint import atomic_json
from .config import TRAINING_CODE_VERSION, TrainingConfig, training_code_sha256
from .fusion_trainer import train_fusion
from .graph_trainer import train_graph_stream
from .model_bundle import build_model_bundle
from .oof import generate_oof_predictions
from .split import EffectiveSplits, build_effective_splits
from .text_embedding import materialize_text_embeddings, sha256_path
from .text_trainer import train_text_stream
from .training_guard import SplitAccessGuard


STAGES = ("verify", "embed", "split", "text", "oof", "graph", "fusion", "calibrate", "bundle")


def effective_splits(config: TrainingConfig) -> EffectiveSplits:
    return build_effective_splits(
        config.split_manifest,
        embargo_trading_days=int(config.split["embargo_trading_days"]),
        validation_selection_fraction=float(
            config.split["validation_selection_fraction"]
        ),
    )


def verify_causal_inputs(config: TrainingConfig) -> dict[str, Any]:
    required = (
        "text_samples.parquet",
        "agent_outputs.parquet",
        "observed_features.parquet",
        "outcomes.parquet",
        "pending_decisions.parquet",
        "reflection_updates.parquet",
        "graph_samples.npz",
        "feature_schema.json",
        "data_manifest.json",
        "rollout_manifest.json",
        "temporal_quality.json",
    )
    missing = [name for name in required if not (config.artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 5 causal artifacts: {missing}")
    temporal = json.loads(
        (config.artifact_dir / "temporal_quality.json").read_text(encoding="utf-8")
    )
    if temporal.get("status") != "PASS":
        raise ValueError("Stage 5 temporal gate must be PASS before Stage 6 training")
    rollout = json.loads(
        (config.artifact_dir / "rollout_manifest.json").read_text(encoding="utf-8")
    )
    if rollout.get("status") != "complete" or rollout.get("temporal_gate") != "PASS":
        raise ValueError("Stage 5 rollout is incomplete or its temporal gate failed")
    data_manifest = json.loads(
        (config.artifact_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    bootstrap = bool(data_manifest.get("bootstrap") or rollout.get("bootstrap"))
    real_actions = (
        not bootstrap
        and data_manifest.get("action_source") == "real_causal_sector_agent"
        and not data_manifest.get("bootstrap_policy")
    )
    if not real_actions:
        policy = data_manifest.get("bootstrap_policy", "unknown")
        raise ValueError(
            "Final canonical training rejects bootstrap action artifacts; "
            f"action_source={policy}, external_llm_used="
            f"{bool(data_manifest.get('external_llm_used'))}. Regenerate Stage 5 "
            "with real causal SectorAgent decisions."
        )
    return {
        "status": "PASS",
        "temporal_gate": "PASS",
        "run_id": rollout.get("run_id"),
        "artifact_counts": rollout.get("artifact_counts", {}),
        "action_source": "real_causal_sector_agent",
    }


def split_summary(config: TrainingConfig, splits: EffectiveSplits) -> dict[str, Any]:
    summary = {
        "train_manifest_days": len(splits.train_manifest),
        "train_effective_days": len(splits.train_effective),
        "embargoed_train_days": len(splits.embargoed_train),
        "validation_days": len(splits.validation_selection)
        + len(splits.validation_calibration),
        "validation_selection_days": len(splits.validation_selection),
        "validation_calibration_days": len(splits.validation_calibration),
        "test_days_not_loaded": len(splits.test_metadata_dates),
        "ranges": {
            "train_effective": [
                min(splits.train_effective).isoformat(),
                max(splits.train_effective).isoformat(),
            ],
            "embargoed_train": [
                min(splits.embargoed_train).isoformat(),
                max(splits.embargoed_train).isoformat(),
            ],
            "validation_selection": [
                min(splits.validation_selection).isoformat(),
                max(splits.validation_selection).isoformat(),
            ],
            "validation_calibration": [
                min(splits.validation_calibration).isoformat(),
                max(splits.validation_calibration).isoformat(),
            ],
            "test_metadata_only": [
                min(splits.test_metadata_dates).isoformat(),
                max(splits.test_metadata_dates).isoformat(),
            ],
        },
        "test_features_loaded": False,
        "test_labels_loaded": False,
    }
    atomic_json(config.model_dir / "split_summary.json", summary)
    return summary


def _checkpoint_path(config: TrainingConfig, stage: str) -> Path:
    return config.model_dir / ".stage_checkpoints" / f"{stage}.json"


def _fingerprint(config: TrainingConfig, stage: str, seed: int, device: str) -> dict[str, Any]:
    inputs = {
        "training_config": sha256_path(config.config_path),
        "data_manifest": sha256_path(config.artifact_dir / "data_manifest.json"),
        "temporal_quality": sha256_path(config.artifact_dir / "temporal_quality.json"),
        "split_manifest": sha256_path(config.split_manifest),
    }
    embedding_manifest = config.artifact_dir / "text_embedding_manifest.json"
    if stage not in {"verify", "embed"} and embedding_manifest.exists():
        inputs["text_embedding_manifest"] = sha256_path(embedding_manifest)
    return {
        "stage": stage,
        "training_code_version": TRAINING_CODE_VERSION,
        "training_code_sha256": training_code_sha256(config.repo_root),
        "config_hash": config.config_hash,
        "seed": seed,
        "device": device,
        "input_hashes": inputs,
    }


def _can_resume(
    config: TrainingConfig,
    stage: str,
    fingerprint: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    path = _checkpoint_path(config, stage)
    if not path.is_file():
        return False, None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("fingerprint") != fingerprint:
        raise ValueError(f"Incompatible checkpoint for stage {stage}; use --force")
    for output, expected in checkpoint.get("output_hashes", {}).items():
        output_path = config.repo_root / output
        if not output_path.is_file() or sha256_path(output_path) != expected:
            raise ValueError(f"Checkpoint output mismatch for stage {stage}: {output}")
    return True, checkpoint.get("result")


def _mark_complete(
    config: TrainingConfig,
    stage: str,
    fingerprint: dict[str, Any],
    result: dict[str, Any],
    outputs: tuple[Path, ...],
) -> None:
    output_hashes = {
        path.resolve().relative_to(config.repo_root).as_posix(): sha256_path(path)
        for path in outputs
        if path.is_file()
    }
    atomic_json(
        _checkpoint_path(config, stage),
        {
            "fingerprint": fingerprint,
            "output_hashes": output_hashes,
            "result": result,
        },
    )


def run_training_pipeline(
    config: TrainingConfig,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    resume: bool = False,
    force: bool = False,
    seed: int | None = None,
    device: str = "cpu",
    local_files_only: bool | None = None,
) -> dict[str, Any]:
    start = STAGES.index(from_stage or STAGES[0])
    end = STAGES.index(to_stage or STAGES[-1])
    if start > end:
        raise ValueError("--from-stage must not occur after --to-stage")
    seed = config.seed if seed is None else int(seed)
    splits = effective_splits(config)
    results: dict[str, Any] = {}
    labels = {
        "verify": "verify causal artifacts",
        "embed": "materialize semantic embeddings",
        "split": "build chronological splits",
        "text": "train Text Stream",
        "graph": "train Graph Stream",
        "oof": "generate expanding-window OOF predictions",
        "fusion": "train Gated Fusion",
        "calibrate": "calibrate on validation-calibration",
        "bundle": "save and validate model bundle",
    }
    for index, stage in enumerate(STAGES[start : end + 1], start=start + 1):
        print(f"[{index}/9] {labels[stage]}", flush=True)
        fingerprint = _fingerprint(config, stage, seed, device)
        if resume and not force:
            reused, result = _can_resume(config, stage, fingerprint)
            if reused:
                results[stage] = result or {"status": "reused"}
                continue
        guard = SplitAccessGuard()
        if stage == "verify":
            result = verify_causal_inputs(config)
            outputs: tuple[Path, ...] = ()
        elif stage == "embed":
            result = materialize_text_embeddings(
                config,
                local_files_only=local_files_only,
                force=force,
                device=device,
            )
            outputs = (
                config.artifact_dir / "text_embeddings_384.parquet",
                config.artifact_dir / "text_embedding_manifest.json",
            )
        elif stage == "split":
            result = split_summary(config, splits)
            outputs = (config.model_dir / "split_summary.json",)
        elif stage == "text":
            result = train_text_stream(
                config, splits, guard=guard, seed=seed, device=device
            )
            outputs = (
                config.model_dir / "text_logistic_baseline.json",
                config.model_dir / "text_model.pt",
                config.model_dir / "text_scaler.json",
                config.model_dir / "text_training_history.json",
            )
        elif stage == "graph":
            result = train_graph_stream(
                config, splits, guard=guard, seed=seed, device=device
            )
            outputs = (
                config.model_dir / "graph_model.pt",
                config.model_dir / "graph_scaler.json",
                config.model_dir / "graph_attention_validation.npz",
                config.model_dir / "graph_training_history.json",
                config.model_dir / "feature_schema.json",
            )
        elif stage == "oof":
            result = generate_oof_predictions(
                config, splits, guard=guard, seed=seed, device=device
            )
            outputs = (
                config.model_dir / "oof_predictions.parquet",
                config.model_dir / "oof_manifest.json",
            )
        elif stage == "fusion":
            result = train_fusion(
                config, splits, guard=guard, seed=seed, device=device
            )
            outputs = (
                config.model_dir / "validation_base_predictions.parquet",
                config.model_dir / "fusion_model.pt",
                config.model_dir / "fusion_training_history.json",
            )
        elif stage == "calibrate":
            result = calibrate_fusion(
                config, splits, guard=guard, seed=seed, device=device
            )
            outputs = (config.model_dir / "calibration.json",)
        else:
            result = build_model_bundle(config, splits, device=device)
            outputs = (config.model_dir / "bundle_manifest.json",)
        results[stage] = result
        _mark_complete(config, stage, fingerprint, result, outputs)
    return results
