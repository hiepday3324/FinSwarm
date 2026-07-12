from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from puppy.common.schemas import AgentContext, SharedContext
from puppy.fusion.text_encoder import (
    CANONICAL_TEXT_VIEW_VERSION,
    FrozenTransformerBackend,
    TextStreamEncoder,
)

from .checkpoint import atomic_json
from .config import TrainingConfig


ALLOWED_EMBEDDING_SPLITS = ("warmup", "train", "validation")
BOOTSTRAP_ENCODER_VERSION = "sha256-frozen-encoder-v1"
BOOTSTRAP_MODEL_MARKERS = ("bootstrap", "sha256-frozen")
TEXT_VIEW_VERSION = CANONICAL_TEXT_VIEW_VERSION
EXCLUDED_BOOTSTRAP_FIELDS = (
    "decision",
    "action_signal",
    "reason",
    "outcome",
    "direction_label",
    "future_price",
    "future_return",
)

# Backward-compatible import name. The implementation is canonical and lives in
# puppy.fusion.text_encoder.
FrozenSemanticEncoder = FrozenTransformerBackend


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_context_from_pending(pending: dict[str, Any]) -> AgentContext:
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

    normalized = dict(context)
    normalized.setdefault("agent_id", pending.get("agent_id") or f"sector_{symbol.lower()}")
    normalized.setdefault("symbol", symbol)
    normalized.setdefault("date", decision_date)
    shared_context = []
    for item in normalized.get("shared_context") or []:
        if isinstance(item, str):
            item = SharedContext(
                source_agent_id="legacy-shared",
                target_agent_id=str(normalized["agent_id"]),
                source_symbol=None,
                target_symbol=symbol,
                text=item,
                reason="Legacy shared-context snapshot.",
            ).model_dump(mode="json")
        shared_context.append(item)
    normalized["shared_context"] = shared_context
    return AgentContext.model_validate(normalized)


def canonical_memory_text(pending: dict[str, Any]) -> tuple[str, str]:
    agent_context = _agent_context_from_pending(pending)
    serializer = TextStreamEncoder(
        embedding_func=lambda _text: np.empty(0, dtype=np.float32),
        embedding_model="serialization-only",
    )
    text = serializer.serialize_context(agent_context)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    encoder: FrozenSemanticEncoder | TextStreamEncoder | None = None,
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
    contexts = [_agent_context_from_pending(row) for row in pending_rows]

    use_local = (
        bool(config.text_embedding.get("local_files_only"))
        if local_files_only is None
        else local_files_only
    )
    if isinstance(encoder, TextStreamEncoder):
        text_encoder = encoder
    else:
        semantic_backend = encoder or FrozenSemanticEncoder(
            model_name=str(config.text_embedding["model_name"]),
            max_length=int(config.text_embedding["max_length"]),
            normalize=bool(config.text_embedding["normalize"]),
            local_files_only=use_local,
            device=device,
        )
        text_encoder = TextStreamEncoder(
            transformer_backend=semantic_backend,
            embedding_model=semantic_backend.model_name,
        )
    if text_encoder.dimension != int(config.text_embedding["dimension"]):
        raise ValueError(
            f"Encoder dimension {text_encoder.dimension} does not match configured dimension 384"
        )
    embeddings, text_inputs = text_encoder.encode_batch_arrays(
        contexts,
        batch_size=int(config.text_embedding["batch_size"]),
    )
    views = [
        (text, hashlib.sha256(text.encode("utf-8")).hexdigest())
        for text in text_inputs
    ]
    if embeddings.shape != (len(pending_rows), 384):
        raise ValueError(f"Unexpected semantic embedding shape: {embeddings.shape}")

    revision = text_encoder.revision
    model_name = text_encoder.embedding_model
    model_hash, model_files = _model_files_hash(model_name, revision)
    if model_hash == "unavailable":
        backend = text_encoder.transformer_backend
        if backend is None:
            raise ValueError("Semantic materialization requires the canonical Transformer backend")
        model_hash = _in_memory_model_hash(backend.model)
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
                "encoder_model": model_name,
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
        "encoder_model": model_name,
        "encoder_revision": revision,
        "encoder_version": f"transformers-frozen:{revision}",
        "model_files_config_sha256": model_hash,
        "model_files": model_files,
        "embedding_dim": 384,
        "pooling": "mean",
        "normalized": bool(config.text_embedding["normalize"]),
        "configured_batch_size": int(config.text_embedding["batch_size"]),
        "effective_batch_size": text_encoder.last_effective_batch_size,
        "sample_count": len(records),
        "text_view_version": str(config.text_embedding["text_view_version"]),
        "included_splits": list(ALLOWED_EMBEDDING_SPLITS),
        "sealed_splits": ["test"],
        "excluded_bootstrap_fields": list(EXCLUDED_BOOTSTRAP_FIELDS),
    }
    atomic_json(manifest_path, manifest)
    return manifest
