from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .checkpoint import atomic_json
from .config import TrainingConfig


ALLOWED_EMBEDDING_SPLITS = ("warmup", "train", "validation")
BOOTSTRAP_ENCODER_VERSION = "sha256-frozen-encoder-v1"
BOOTSTRAP_MODEL_MARKERS = ("bootstrap", "sha256-frozen")
TEXT_VIEW_VERSION = "causal-memory-only-v1"
EXCLUDED_BOOTSTRAP_FIELDS = (
    "latest_reason",
    "decision",
    "action_signal",
    "reflection_memory",
    "shadow_state",
    "psychology_state",
    "shared_context",
    "outcome",
    "direction_label",
    "future_price",
)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_memory_text(pending: dict[str, Any]) -> tuple[str, str]:
    snapshot = pending.get("context_snapshot", {})
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    context = snapshot.get("agent_context", snapshot)
    source_hash = str(pending["context_hash"])
    inner_hash = context.get("context_hash")
    if inner_hash is not None and str(inner_hash) != source_hash:
        raise ValueError(f"Context hash mismatch for decision {pending['decision_id']}")

    symbol = str(pending["symbol"])
    decision_date = pending["decision_date"].isoformat()
    if context.get("symbol") not in (None, symbol):
        raise ValueError(f"Context symbol mismatch for decision {pending['decision_id']}")
    context_date = context.get("date")
    if context_date not in (None, decision_date):
        raise ValueError(f"Context date mismatch for decision {pending['decision_id']}")

    lines = [f"Symbol: {symbol}", f"Decision date: {decision_date}"]
    for title, key in (
        ("Short memory", "short_memory"),
        ("Mid memory", "mid_memory"),
        ("Long memory", "long_memory"),
    ):
        lines.extend(("", f"{title}:"))
        memories = context.get(key) or []
        if not isinstance(memories, list):
            raise ValueError(f"{key} must be a list for decision {pending['decision_id']}")
        if memories:
            lines.extend(f"- {str(value).strip()}" for value in memories)
        else:
            lines.append("None.")
    text = "\n".join(lines).strip()
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class FrozenSemanticEncoder:
    model_name: str
    max_length: int = 512
    normalize: bool = True
    local_files_only: bool = False
    device: str = "cpu"
    tokenizer: Any | None = None
    model: Any | None = None
    last_effective_batch_size: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        import torch

        if self.tokenizer is None or self.model is None:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        self.model.to(torch.device(self.device))
        self.model.eval()

    @property
    def revision(self) -> str:
        config = getattr(self.model, "config", None)
        return str(getattr(config, "_commit_hash", None) or "unknown")

    @property
    def dimension(self) -> int:
        config = getattr(self.model, "config", None)
        value = getattr(config, "hidden_size", None)
        if value is None:
            raise ValueError("Encoder config does not expose hidden_size")
        return int(value)

    def encode(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        effective_batch_size = min(batch_size, 8) if self.device == "cpu" else batch_size
        self.last_effective_batch_size = effective_batch_size
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                encoded = self.tokenizer(
                    values[start : start + effective_batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                if self.normalize:
                    pooled = functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32))
        result = np.concatenate(batches, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Semantic encoder produced NaN or infinite values")
        return result


def _model_files_hash(model_name: str, revision: str) -> tuple[str, list[dict[str, str]]]:
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=model_name,
                revision=None if revision == "unknown" else revision,
                local_files_only=True,
            )
        )
    except Exception:
        return "unavailable", []
    records: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        file_hash = sha256_path(path)
        records.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest(), records


def _in_memory_model_hash(model: Any) -> str:
    digest = hashlib.sha256()
    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "to_dict"):
            value = config.to_dict()
        else:
            value = vars(config)
        digest.update(
            json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_pending_without_test(path: Path) -> list[dict[str, Any]]:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=ds.field("split").isin(ALLOWED_EMBEDDING_SPLITS))
    return table.to_pylist()


def validate_semantic_embedding_artifact(path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if int(manifest.get("embedding_dim", -1)) != 384:
        raise ValueError("Real Text Stream requires 384-dimensional semantic embeddings")
    encoder_version = str(manifest.get("encoder_version", ""))
    model_name = str(manifest.get("encoder_model", ""))
    if encoder_version == BOOTSTRAP_ENCODER_VERSION or any(
        marker in model_name.lower() for marker in BOOTSTRAP_MODEL_MARKERS
    ):
        raise ValueError("Bootstrap hash embeddings are forbidden for real Text Stream training")
    artifact = Path(path)
    if sha256_path(artifact) != manifest.get("artifact_sha256"):
        raise ValueError("Semantic embedding artifact hash mismatch")
    schema = pq.read_schema(artifact)
    if "embedding" not in schema.names:
        raise ValueError("Semantic embedding artifact is missing embedding column")
    return manifest


def materialize_text_embeddings(
    config: TrainingConfig,
    *,
    local_files_only: bool | None = None,
    force: bool = False,
    device: str = "cpu",
    encoder: FrozenSemanticEncoder | None = None,
) -> dict[str, Any]:
    output_path = config.artifact_dir / "text_embeddings_384.parquet"
    manifest_path = config.artifact_dir / "text_embedding_manifest.json"
    if output_path.exists() and manifest_path.exists() and not force:
        manifest = validate_semantic_embedding_artifact(output_path, manifest_path)
        if (
            manifest.get("encoder_model") == config.text_embedding["model_name"]
            and manifest.get("text_view_version") == config.text_embedding["text_view_version"]
            and manifest.get("source_pending_sha256")
            == sha256_path(config.artifact_dir / "pending_decisions.parquet")
        ):
            return manifest
        raise ValueError("Existing semantic embedding artifact is incompatible; use --force")

    pending_path = config.artifact_dir / "pending_decisions.parquet"
    pending_rows = _read_pending_without_test(pending_path)
    pending_rows.sort(key=lambda row: (row["decision_date"], row["symbol"], row["decision_id"]))
    if len({row["decision_id"] for row in pending_rows}) != len(pending_rows):
        raise ValueError("Duplicate decision_id in pending decisions")
    views = [canonical_memory_text(row) for row in pending_rows]

    use_local = (
        bool(config.text_embedding.get("local_files_only"))
        if local_files_only is None
        else local_files_only
    )
    semantic_encoder = encoder or FrozenSemanticEncoder(
        model_name=str(config.text_embedding["model_name"]),
        max_length=int(config.text_embedding["max_length"]),
        normalize=bool(config.text_embedding["normalize"]),
        local_files_only=use_local,
        device=device,
    )
    if semantic_encoder.dimension != int(config.text_embedding["dimension"]):
        raise ValueError(
            f"Encoder dimension {semantic_encoder.dimension} does not match configured dimension 384"
        )
    embeddings = semantic_encoder.encode(
        (item[0] for item in views),
        batch_size=int(config.text_embedding["batch_size"]),
    )
    if embeddings.shape != (len(pending_rows), 384):
        raise ValueError(f"Unexpected semantic embedding shape: {embeddings.shape}")

    revision = semantic_encoder.revision
    model_hash, model_files = _model_files_hash(semantic_encoder.model_name, revision)
    if model_hash == "unavailable":
        model_hash = _in_memory_model_hash(semantic_encoder.model)
        model_files = [{"path": "in-memory-model-config-and-state", "sha256": model_hash}]
    metadata = json.dumps(
        {
            "excluded_bootstrap_fields": list(EXCLUDED_BOOTSTRAP_FIELDS),
            "test_split_materialized": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    records = []
    for row, (text_input, text_hash), embedding in zip(pending_rows, views, embeddings):
        records.append(
            {
                "decision_id": row["decision_id"],
                "agent_id": row["agent_id"],
                "symbol": row["symbol"],
                "decision_date": row["decision_date"],
                "decision_cutoff": row["decision_cutoff"],
                "split": row["split"],
                "source_context_hash": row["context_hash"],
                "text_view_hash": text_hash,
                "text_view_version": str(config.text_embedding["text_view_version"]),
                "text_input": text_input,
                "embedding": embedding.tolist(),
                "embedding_dim": 384,
                "encoder_model": semantic_encoder.model_name,
                "encoder_revision": revision,
                "pooling": "mean",
                "normalized": bool(config.text_embedding["normalize"]),
                "metadata": metadata,
            }
        )
    schema = pa.schema(
        [
            ("decision_id", pa.string()),
            ("agent_id", pa.string()),
            ("symbol", pa.string()),
            ("decision_date", pa.date32()),
            ("decision_cutoff", pa.timestamp("us", tz="UTC")),
            ("split", pa.string()),
            ("source_context_hash", pa.string()),
            ("text_view_hash", pa.string()),
            ("text_view_version", pa.string()),
            ("text_input", pa.string()),
            ("embedding", pa.list_(pa.float32(), 384)),
            ("embedding_dim", pa.int32()),
            ("encoder_model", pa.string()),
            ("encoder_revision", pa.string()),
            ("pooling", pa.string()),
            ("normalized", pa.bool_()),
            ("metadata", pa.string()),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression="zstd")
    os.replace(temporary, output_path)
    artifact_hash = sha256_path(output_path)
    manifest = {
        "artifact": output_path.name,
        "artifact_sha256": artifact_hash,
        "source_pending_sha256": sha256_path(pending_path),
        "encoder_model": semantic_encoder.model_name,
        "encoder_revision": revision,
        "encoder_version": f"transformers-frozen:{revision}",
        "model_files_config_sha256": model_hash,
        "model_files": model_files,
        "embedding_dim": 384,
        "pooling": "mean",
        "normalized": bool(config.text_embedding["normalize"]),
        "configured_batch_size": int(config.text_embedding["batch_size"]),
        "effective_batch_size": semantic_encoder.last_effective_batch_size,
        "sample_count": len(records),
        "text_view_version": str(config.text_embedding["text_view_version"]),
        "included_splits": list(ALLOWED_EMBEDDING_SPLITS),
        "sealed_splits": ["test"],
        "excluded_bootstrap_fields": list(EXCLUDED_BOOTSTRAP_FIELDS),
    }
    atomic_json(manifest_path, manifest)
    return manifest
