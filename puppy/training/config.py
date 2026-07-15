from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


TRAINING_CODE_VERSION = "stage6-offline-training-v2"

GRAPH_TOPOLOGIES = frozenset({"full", "identity"})
FUSION_HEADS = frozenset({"latent_gate", "scalar_logit_stack", "graph_only"})
OOF_TRAINING_POLICIES = frozenset(
    {"fixed_epochs", "inner_chronological_early_stop"}
)


def training_code_sha256(repo_root: str | Path) -> str:
    root = Path(repo_root)
    paths = sorted(
        list((root / "puppy" / "training").glob("*.py"))
        + [
            root / "puppy" / "fusion" / "text_encoder.py",
            root / "puppy" / "fusion" / "text_score_head.py",
            root / "puppy" / "fusion" / "gated_fusion.py",
            root / "puppy" / "fusion" / "scalar_logit_stack.py",
            root / "puppy" / "graph" / "gat_model.py",
            root / "puppy" / "graph" / "graph_features.py",
            root / "puppy" / "graph" / "graph_dataset.py",
        ]
        + list((root / "scripts").glob("*finswarm*.py"))
        + list((root / "scripts").glob("*oof*.py"))
        + list((root / "scripts").glob("train_text_head.py"))
        + list((root / "scripts").glob("train_gat.py"))
        + list((root / "scripts").glob("train_fusion.py"))
        + list((root / "scripts").glob("materialize_text_embeddings.py"))
        + list((root / "scripts").glob("validate_model_bundle.py"))
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    config_path: Path
    repo_root: Path
    run_id: str
    seed: int
    experiment_config: Path
    artifact_dir: Path
    split_manifest: Path
    model_dir: Path
    text_embedding: dict[str, Any]
    split: dict[str, Any]
    labels: dict[str, Any]
    text_model: dict[str, Any]
    graph_model: dict[str, Any]
    fusion_model: dict[str, Any]
    calibration: dict[str, Any]
    raw: dict[str, Any]
    graph: dict[str, Any] = field(
        default_factory=lambda: {"topology": "full", "include_text_score": True}
    )
    fusion: dict[str, Any] = field(default_factory=lambda: {"head": "latent_gate"})
    oof: dict[str, Any] = field(default_factory=lambda: {"training_policy": "fixed_epochs"})

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def graph_topology(self) -> str:
        return str(self.graph.get("topology", "full"))

    @property
    def graph_includes_text_score(self) -> bool:
        return self.graph.get("include_text_score", True) is True

    @property
    def fusion_head(self) -> str:
        return str(self.fusion.get("head", "latent_gate"))

    @property
    def oof_training_policy(self) -> str:
        return str(self.oof.get("training_policy", "fixed_epochs"))

    @property
    def ablation_contract(self) -> dict[str, Any]:
        return {
            "graph": {
                "topology": self.graph_topology,
                "include_text_score": self.graph_includes_text_score,
            },
            "fusion": {"head": self.fusion_head},
            "oof": {"training_policy": self.oof_training_policy},
        }

    @property
    def ablation_fingerprint(self) -> str:
        payload = json.dumps(
            self.ablation_contract, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def is_canonical_ablation_contract(self) -> bool:
        return self.ablation_contract == {
            "graph": {"topology": "full", "include_text_score": True},
            "fusion": {"head": "latent_gate"},
            "oof": {"training_policy": "fixed_epochs"},
        }

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root).as_posix()


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a mapping")
    repo_root = config_path.parents[2]

    def get(*keys: str) -> Any:
        value: Any = raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Missing training config field: {'.'.join(keys)}")
            value = value[key]
        return value

    def resolve(value: Any) -> Path:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else repo_root / candidate

    config = TrainingConfig(
        config_path=config_path,
        repo_root=repo_root,
        run_id=str(get("run", "run_id")),
        seed=int(get("run", "seed")),
        experiment_config=resolve(get("inputs", "experiment_config")),
        artifact_dir=resolve(get("inputs", "artifact_dir")),
        split_manifest=resolve(get("inputs", "split_manifest")),
        model_dir=resolve(get("output", "model_dir")),
        text_embedding=dict(get("text_embedding")),
        split=dict(get("split")),
        labels=dict(get("labels")),
        text_model=dict(get("text_model")),
        graph_model=dict(get("graph_model")),
        fusion_model=dict(get("fusion_model")),
        calibration=dict(get("calibration")),
        raw=raw,
        graph=dict(get("graph")),
        fusion=dict(get("fusion")),
        oof=dict(get("oof")),
    )
    _validate(config)
    return config


def _validate(config: TrainingConfig) -> None:
    _validate_contract_section(
        "graph",
        config.graph,
        required={"topology", "include_text_score"},
    )
    _validate_contract_section("fusion", config.fusion, required={"head"})
    _validate_contract_section("oof", config.oof, required={"training_policy"})
    if config.graph_topology not in GRAPH_TOPOLOGIES:
        raise ValueError(
            f"Unsupported graph.topology: {config.graph_topology!r}; "
            f"expected one of {sorted(GRAPH_TOPOLOGIES)}"
        )
    if type(config.graph.get("include_text_score")) is not bool:
        raise ValueError("graph.include_text_score must be a boolean")
    if config.fusion_head not in FUSION_HEADS:
        raise ValueError(
            f"Unsupported fusion.head: {config.fusion_head!r}; "
            f"expected one of {sorted(FUSION_HEADS)}"
        )
    if config.oof_training_policy not in OOF_TRAINING_POLICIES:
        raise ValueError(
            f"Unsupported oof.training_policy: {config.oof_training_policy!r}; "
            f"expected one of {sorted(OOF_TRAINING_POLICIES)}"
        )
    if config.text_embedding.get("dimension") != 384:
        raise ValueError("Real semantic embedding dimension must be 384")
    if config.text_embedding.get("pooling") != "mean":
        raise ValueError("Only attention-mask-aware mean pooling is supported")
    fraction = float(config.split.get("validation_selection_fraction", 0))
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_selection_fraction must be between 0 and 1")
    for section, keys in (
        (config.text_embedding, ("batch_size", "max_length")),
        (config.split, ("embargo_trading_days", "oof_folds", "oof_min_train_days")),
        (config.text_model, ("hidden_dim", "batch_size", "max_epochs", "patience")),
        (config.graph_model, ("hidden_dim", "num_heads", "batch_size", "max_epochs", "patience")),
        (config.fusion_model, ("latent_dim", "batch_size", "max_epochs", "patience")),
    ):
        for key in keys:
            if int(section[key]) <= 0:
                raise ValueError(f"{key} must be positive")
    if config.calibration.get("method") != "temperature":
        raise ValueError("Only temperature calibration is supported")
    canonical_model_dir = (
        config.repo_root / "data" / "model_artifacts" / "finswarm_v1"
    ).resolve()
    if (
        not config.is_canonical_ablation_contract
        and config.model_dir.resolve() == canonical_model_dir
    ):
        raise ValueError(
            "Stage 6 ablations must use a separate output.model_dir and may not "
            "overwrite data/model_artifacts/finswarm_v1"
        )
    required = (
        config.experiment_config,
        config.artifact_dir / "data_manifest.json",
        config.artifact_dir / "temporal_quality.json",
        config.split_manifest,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 6 input(s): {missing}")


def _validate_contract_section(
    name: str,
    section: dict[str, Any],
    *,
    required: set[str],
) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"Training config section {name} must be a mapping")
    missing = required - set(section)
    unknown = set(section) - required
    if missing:
        raise ValueError(f"Missing training config field(s): {name}.{sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown training config field(s): {name}.{sorted(unknown)}")
