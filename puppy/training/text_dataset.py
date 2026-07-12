from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.dataset as ds

from .config import TrainingConfig
from .split import EffectiveSplits
from .text_embedding import validate_semantic_embedding_artifact
from .training_guard import SplitAccessGuard


TRAINING_SPLITS = ("warmup", "train", "validation")


@dataclass(frozen=True)
class TextDataset:
    decision_ids: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray
    embeddings: np.ndarray
    targets: np.ndarray
    source_context_hashes: np.ndarray
    metadata: dict[str, int]

    def subset(self, dates: Iterable[dt.date]) -> "TextDataset":
        allowed = set(dates)
        mask = np.asarray([value in allowed for value in self.dates], dtype=bool)
        return TextDataset(
            decision_ids=self.decision_ids[mask],
            dates=self.dates[mask],
            symbols=self.symbols[mask],
            embeddings=self.embeddings[mask],
            targets=self.targets[mask],
            source_context_hashes=self.source_context_hashes[mask],
            metadata=dict(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.targets)


def _read_partitions(path: Path, partitions: tuple[str, ...]) -> list[dict[str, Any]]:
    dataset = ds.dataset(path, format="parquet")
    if "test" in partitions:
        raise ValueError("Test partition must not be requested from the text dataset loader")
    return dataset.to_table(filter=ds.field("split").isin(partitions)).to_pylist()


def _unique_by(rows: list[dict[str, Any]], key: str, name: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key} in {name}: {value}")
        result[value] = row
    return result


def load_supervised_text_dataset(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    guard: SplitAccessGuard | None = None,
    allowed_splits: tuple[str, ...] = TRAINING_SPLITS,
    allowed_dates: Iterable[dt.date] | None = None,
    purpose: str = "fit",
) -> TextDataset:
    guard = guard or SplitAccessGuard()
    if "test" in allowed_splits:
        guard.assert_allowed("test", purpose)
        raise AssertionError("Split access guard did not reject test text partition")
    for split in allowed_splits:
        guard.assert_allowed(split, purpose)
    embedding_path = config.artifact_dir / "text_embeddings_384.parquet"
    validate_semantic_embedding_artifact(
        embedding_path,
        config.artifact_dir / "text_embedding_manifest.json",
    )

    pending_rows = _read_partitions(
        config.artifact_dir / "pending_decisions.parquet", allowed_splits
    )
    outcome_rows = _read_partitions(config.artifact_dir / "outcomes.parquet", allowed_splits)
    embedding_rows = _read_partitions(embedding_path, allowed_splits)
    observed_rows = _read_partitions(
        config.artifact_dir / "observed_features.parquet", allowed_splits
    )
    pending = _unique_by(pending_rows, "decision_id", "pending decisions")
    outcomes = _unique_by(outcome_rows, "decision_id", "outcomes")
    embeddings = _unique_by(embedding_rows, "decision_id", "semantic embeddings")
    observed = _unique_by(
        [dict(row, observed_key=(row["symbol"], row["date"])) for row in observed_rows],
        "observed_key",
        "observed features",
    )

    supervised_dates = set(
        allowed_dates
        if allowed_dates is not None
        else splits.train_effective
        + splits.validation_selection
        + splits.validation_calibration
    )
    records: list[tuple[Any, ...]] = []
    counts = {
        "masked_neutral_count": 0,
        "cross_split_excluded_count": 0,
        "warmup_excluded_count": 0,
        "ineligible_count": 0,
        "missing_outcome_count": 0,
    }
    for decision_id, pending_row in pending.items():
        if pending_row["split"] == "warmup":
            counts["warmup_excluded_count"] += 1
            continue
        if pending_row["decision_date"] not in supervised_dates:
            continue
        outcome = outcomes.get(decision_id)
        if outcome is None:
            counts["missing_outcome_count"] += 1
            if pending_row.get("status") != "unresolved_end_of_data":
                raise ValueError(f"Missing non-boundary outcome for decision {decision_id}")
            continue
        if pending_row.get("status") != "resolved":
            raise ValueError(f"Resolved outcome has non-resolved pending status: {decision_id}")
        embedding = embeddings.get(decision_id)
        if embedding is None:
            raise ValueError(f"Missing semantic embedding for decision {decision_id}")
        if pending_row["symbol"] != outcome["symbol"] or embedding["symbol"] != outcome["symbol"]:
            raise ValueError(f"Text/outcome symbol mismatch for decision {decision_id}")
        if (
            pending_row["decision_date"] != outcome["decision_date"]
            or embedding["decision_date"] != outcome["decision_date"]
        ):
            raise ValueError(f"Text/outcome date mismatch for decision {decision_id}")
        if pending_row["split"] != outcome["split"] or embedding["split"] != outcome["split"]:
            raise ValueError(f"Text/outcome split mismatch for decision {decision_id}")
        observed_row = observed.get((outcome["symbol"], outcome["decision_date"]))
        if observed_row is None:
            raise ValueError(f"Missing observed feature row for decision {decision_id}")
        expected_hash = str(pending_row["context_hash"])
        actual_hashes = {
            str(embedding["source_context_hash"]),
            str(observed_row["context_hash"]),
        }
        if actual_hashes != {expected_hash}:
            raise ValueError(f"Context hash mismatch for decision {decision_id}")
        label = int(outcome["direction_label"])
        if label not in (-1, 0, 1):
            raise ValueError(f"Unexpected target value {label} for decision {decision_id}")
        if bool(outcome["cross_split_outcome"]):
            counts["cross_split_excluded_count"] += 1
            continue
        if not bool(outcome["supervised_eligible"]):
            counts["ineligible_count"] += 1
            continue
        if label == 0 or not bool(outcome["train_mask"]):
            counts["masked_neutral_count"] += 1
            continue
        vector = np.asarray(embedding["embedding"], dtype=np.float32)
        if int(embedding["embedding_dim"]) != 384 or vector.shape != (384,):
            raise ValueError(f"Invalid semantic embedding for decision {decision_id}")
        records.append(
            (
                decision_id,
                outcome["decision_date"],
                outcome["symbol"],
                vector,
                1 if label == 1 else 0,
                expected_hash,
            )
        )
    records.sort(key=lambda item: (item[1], item[2], item[0]))
    if not records:
        raise ValueError("No supervised text rows remain after leakage filters")
    return TextDataset(
        decision_ids=np.asarray([item[0] for item in records]),
        dates=np.asarray([item[1] for item in records], dtype=object),
        symbols=np.asarray([item[2] for item in records]),
        embeddings=np.stack([item[3] for item in records]).astype(np.float32),
        targets=np.asarray([item[4] for item in records], dtype=np.float32),
        source_context_hashes=np.asarray([item[5] for item in records]),
        metadata=counts,
    )


def write_dataset_summary(path: str | Path, dataset: TextDataset) -> None:
    value = {
        "rows": len(dataset),
        "start_date": min(dataset.dates).isoformat(),
        "end_date": max(dataset.dates).isoformat(),
        "symbols": sorted(set(dataset.symbols.tolist())),
        **dataset.metadata,
    }
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
