from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .rollout_state import canonical_json


TEXT_COLUMNS = [
    "text_sample_id", "decision_id", "run_id", "agent_id", "symbol", "date", "as_of",
    "context_hash", "raw_context", "h_text", "embedding_model", "encoder_version",
    "prompt_version", "split", "metadata",
]
OUTPUT_COLUMNS = [
    "decision_id", "run_id", "agent_id", "symbol", "decision_date", "decision_cutoff",
    "decision", "action_signal", "reason", "parse_ok", "model_name", "context_hash", "prompt_version",
    "adapter_version", "text_score", "split", "metadata",
]
OBSERVED_COLUMNS = [
    "sample_id", "run_id", "symbol", "date", "as_of", "context_hash", "split",
    "source_dates", "metadata",
]
OUTCOME_COLUMNS = [
    "outcome_id", "decision_id", "run_id", "agent_id", "symbol", "decision_date",
    "outcome_date", "available_at", "open_price", "close_price", "asset_return",
    "action_signal", "strategy_return", "direction_label", "train_mask", "threshold_bps",
    "formula_version", "split", "cross_split_outcome", "supervised_eligible", "metadata",
]
PENDING_COLUMNS = [
    "decision_id", "sample_id", "run_id", "agent_id", "symbol", "decision_date",
    "decision_cutoff", "expected_outcome_date", "decision", "action_signal", "reason",
    "context_hash", "context_snapshot", "memory_refs", "text_sample_id", "graph_sample_id",
    "prompt_version", "encoder_version", "adapter_version", "status", "split", "metadata",
]
REFLECTION_COLUMNS = [
    "reflection_id", "decision_id", "outcome_id", "run_id", "agent_id", "symbol",
    "source_decision_date", "outcome_date", "available_at", "text", "memory_layer",
    "prompt_version", "context_hash", "metadata",
]


def _write_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    json_fields: Iterable[str] = (),
) -> None:
    json_fields = set(json_fields)
    normalized = []
    for row in rows:
        value = {}
        for column in columns:
            item = row.get(column)
            value[column] = canonical_json(item) if column in json_fields else item
        normalized.append(value)
    table = pa.Table.from_pylist(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False, version="2.6")
    os.replace(temporary, path)


def export_artifact_tables(output_dir: Path, artifacts: dict[str, list[dict[str, Any]]]) -> None:
    _write_parquet(output_dir / "text_samples.parquet", artifacts["text_samples"], TEXT_COLUMNS, json_fields={"metadata"})
    _write_parquet(output_dir / "agent_outputs.parquet", artifacts["agent_outputs"], OUTPUT_COLUMNS, json_fields={"metadata"})
    observed_columns = OBSERVED_COLUMNS[:-1] + list(artifacts["observed_feature_names"]) + ["metadata"]
    _write_parquet(output_dir / "observed_features.parquet", artifacts["observed_features"], observed_columns, json_fields={"source_dates", "metadata"})
    _write_parquet(output_dir / "outcomes.parquet", artifacts["outcomes"], OUTCOME_COLUMNS, json_fields={"metadata"})
    _write_parquet(output_dir / "pending_decisions.parquet", artifacts["pending_decisions"], PENDING_COLUMNS, json_fields={"context_snapshot", "memory_refs", "metadata"})
    _write_parquet(output_dir / "reflection_updates.parquet", artifacts["reflection_updates"], REFLECTION_COLUMNS, json_fields={"metadata"})


def export_graph_npz(output_dir: Path, samples: list[dict[str, Any]]) -> None:
    target = output_dir / "graph_samples.npz"
    temporary = output_dir / "graph_samples.tmp.npz"
    np.savez_compressed(
        temporary,
        sample_ids=np.asarray([sample["sample_id"] for sample in samples]),
        decision_dates=np.asarray([sample["decision_date"] for sample in samples]),
        symbols=np.asarray([sample["symbols"] for sample in samples]),
        feature_matrix=np.stack([sample["feature_matrix"] for sample in samples]),
        adjacency_matrix=np.stack([sample["adjacency_matrix"] for sample in samples]),
        feature_mask=np.stack([sample["feature_mask"] for sample in samples]),
        max_source_dates=np.asarray([sample["max_source_date"] for sample in samples]),
    )
    os.replace(temporary, target)


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows
