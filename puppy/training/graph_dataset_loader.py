from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.dataset as ds
import yaml

from puppy.graph.graph_dataset import CANONICAL_FEATURE_NAMES
from puppy.graph.graph_features import (
    ACTION_FEATURE_NAMES,
    MARKET_FEATURE_NAMES,
    PSYCHOLOGY_FEATURE_NAMES,
    SHADOW_FEATURE_NAMES,
    TEXT_SCORE_FEATURE_NAMES,
)

from .config import TrainingConfig
from .split import EffectiveSplits
from .training_guard import SplitAccessGuard


GRAPH_FEATURE_NAMES = CANONICAL_FEATURE_NAMES
POLICY_DEPENDENT_GRAPH_FIELDS = {
    "action_signal",
    "shadow_roi_prior",
    "shadow_rolling_return_prior",
    "shadow_drawdown_prior",
    "shadow_sharpe_prior",
    "shadow_win_rate_prior",
    "text_score",
    "psychology_greed",
    "psychology_fear",
    "psychology_neutral",
}


def configured_graph_feature_names(config: TrainingConfig) -> tuple[str, ...]:
    if config.graph_includes_text_score:
        return GRAPH_FEATURE_NAMES
    return tuple(name for name in GRAPH_FEATURE_NAMES if name != "text_score")


@dataclass(frozen=True)
class GraphDataset:
    sample_ids: np.ndarray
    dates: np.ndarray
    symbols: tuple[str, ...]
    features: np.ndarray
    adjacency: np.ndarray
    feature_mask: np.ndarray
    targets: np.ndarray
    label_mask: np.ndarray
    decision_ids: np.ndarray
    feature_names: tuple[str, ...] = GRAPH_FEATURE_NAMES
    metadata: dict[str, Any] = field(default_factory=dict)

    def subset(self, dates: Iterable[dt.date]) -> "GraphDataset":
        allowed = set(dates)
        mask = np.asarray([value in allowed for value in self.dates], dtype=bool)
        return GraphDataset(
            sample_ids=self.sample_ids[mask],
            dates=self.dates[mask],
            symbols=self.symbols,
            features=self.features[mask],
            adjacency=self.adjacency[mask],
            feature_mask=self.feature_mask[mask],
            targets=self.targets[mask],
            label_mask=self.label_mask[mask],
            decision_ids=self.decision_ids[mask],
            feature_names=self.feature_names,
            metadata=dict(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def supervised_count(self) -> int:
        return int(self.label_mask.sum())


def _outcomes_for_dates(path: Path, dates: tuple[dt.date, ...]) -> list[dict[str, Any]]:
    dataset = ds.dataset(path, format="parquet")
    return dataset.to_table(filter=ds.field("decision_date").isin(dates)).to_pylist()


def _expected_symbols(config: TrainingConfig) -> tuple[str, ...]:
    with config.experiment_config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    symbols = tuple(str(value) for value in raw["assets"]["symbols"])
    if not symbols:
        raise ValueError("Experiment config contains no symbols")
    return symbols


def load_graph_dataset(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    dates: Iterable[dt.date] | None = None,
    purpose: str = "fit",
    guard: SplitAccessGuard | None = None,
    text_scores_by_decision: Mapping[str, float] | None = None,
    require_oof_text_scores: bool = False,
) -> GraphDataset:
    """Load canonical graph shards and align optional text-score overrides.

    ``require_oof_text_scores`` is retained as a legacy API name. Overrides are
    fold OOF predictions for training data and final canonical-model predictions
    for validation data. Strict presence checks apply only to supervised nodes.
    """
    guard = guard or SplitAccessGuard()
    include_text_score = config.graph_includes_text_score
    if not include_text_score and text_scores_by_decision is not None:
        raise ValueError(
            "graph.include_text_score=false forbids text-score overrides"
        )
    if not include_text_score and require_oof_text_scores:
        raise ValueError(
            "graph.include_text_score=false cannot require text-score overrides"
        )
    requested_dates = tuple(
        sorted(
            set(dates)
            if dates is not None
            else set(
                splits.train_effective
                + splits.validation_selection
                + splits.validation_calibration
            )
        )
    )
    test_dates = set(splits.test_metadata_dates)
    if test_dates.intersection(requested_dates):
        guard.assert_allowed("test", purpose)
        raise AssertionError("Split access guard did not reject test graph dates")
    known_dates = set(
        splits.warmup
        + splits.train_manifest
        + splits.validation_selection
        + splits.validation_calibration
        + splits.test_metadata_dates
    )
    unknown = set(requested_dates) - known_dates
    if unknown:
        raise ValueError(f"Graph dates are absent from split manifest: {sorted(unknown)}")
    for split_name, split_dates in (
        ("warmup", splits.warmup),
        ("train", splits.train_manifest),
        ("validation_selection", splits.validation_selection),
        ("validation_calibration", splits.validation_calibration),
    ):
        if set(split_dates).intersection(requested_dates):
            guard.assert_allowed(split_name, purpose)

    feature_schema = json.loads(
        (config.artifact_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    source_names = list(feature_schema["graph_feature_names"])
    selected_names = configured_graph_feature_names(config)
    missing = [name for name in selected_names if name not in source_names]
    if missing:
        raise ValueError(f"Stage 5 graph artifact is missing required features: {missing}")
    if tuple(source_names) != GRAPH_FEATURE_NAMES:
        raise ValueError(
            "Stage 5 graph artifact does not use the canonical H0 feature order"
        )
    indices = [source_names.index(name) for name in selected_names]
    text_score_index = (
        selected_names.index("text_score") if include_text_score else None
    )

    outcome_rows = _outcomes_for_dates(
        config.artifact_dir / "outcomes.parquet", requested_dates
    )
    outcomes: dict[tuple[dt.date, str], dict[str, Any]] = {}
    for row in outcome_rows:
        key = (row["decision_date"], str(row["symbol"]))
        if key in outcomes:
            raise ValueError(f"Duplicate graph outcome key: {key}")
        outcomes[key] = row

    expected_symbols = _expected_symbols(config)
    sample_ids: list[str] = []
    matrices: list[np.ndarray] = []
    adjacencies: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    label_masks: list[np.ndarray] = []
    decision_ids: list[np.ndarray] = []
    shard_dir = config.artifact_dir / "state" / "graph_shards"
    for date in requested_dates:
        path = shard_dir / f"{date.isoformat()}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 5 graph shard: {path}")
        with np.load(path, allow_pickle=False) as shard:
            shard_date = dt.date.fromisoformat(str(shard["decision_date"].item()))
            if shard_date != date:
                raise ValueError(f"Graph shard date mismatch in {path}")
            symbols = tuple(str(value) for value in shard["symbols"].tolist())
            if symbols != expected_symbols:
                raise ValueError(
                    f"Graph symbol-order mismatch on {date}: {symbols} != {expected_symbols}"
                )
            full_features = np.asarray(shard["feature_matrix"], dtype=np.float32)
            full_mask = np.asarray(shard["feature_mask"], dtype=bool)
            selected = full_features[:, indices]
            selected_mask = full_mask[:, indices]
            adjacency = np.asarray(shard["adjacency_matrix"], dtype=np.float32)
            if adjacency.shape != (len(symbols), len(symbols)):
                raise ValueError(f"Invalid adjacency shape on {date}: {adjacency.shape}")
            if config.graph_topology == "identity":
                adjacency = np.eye(len(symbols), dtype=np.float32)
            day_targets = np.zeros(len(symbols), dtype=np.float32)
            day_mask = np.zeros(len(symbols), dtype=bool)
            day_ids = np.full(len(symbols), "", dtype=object)
            for node, symbol in enumerate(symbols):
                resolved_outcome = outcomes.get((date, symbol))
                if resolved_outcome is None:
                    continue
                day_ids[node] = resolved_outcome["decision_id"]
                decision_id = str(resolved_outcome["decision_id"])
                label = int(resolved_outcome["direction_label"])
                if label not in (-1, 0, 1):
                    raise ValueError(f"Unexpected graph target {label} on {date} {symbol}")
                eligible = (
                    bool(resolved_outcome["train_mask"])
                    and bool(resolved_outcome["supervised_eligible"])
                    and not bool(resolved_outcome["cross_split_outcome"])
                    and label in (-1, 1)
                )
                if (
                    include_text_score
                    and text_scores_by_decision is not None
                    and decision_id in text_scores_by_decision
                ):
                    score = float(text_scores_by_decision[decision_id])
                    if not 0.0 <= score <= 1.0:
                        raise ValueError(
                            f"Text score override for {decision_id} is outside [0, 1]"
                        )
                    if text_score_index is None:
                        raise AssertionError("Text-score index is unavailable")
                    selected[node, text_score_index] = score
                    selected_mask[node, text_score_index] = True
                elif include_text_score and require_oof_text_scores and eligible:
                    raise ValueError(
                        "Missing required text score override for supervised "
                        f"decision {decision_id}"
                    )
                if eligible:
                    day_targets[node] = float(label == 1)
                    day_mask[node] = True
            sample_ids.append(str(shard["sample_id"].item()))
            matrices.append(selected)
            masks.append(selected_mask)
            adjacencies.append(adjacency)
            labels.append(day_targets)
            label_masks.append(day_mask)
            decision_ids.append(day_ids)
    if not matrices:
        raise ValueError("No graph dates requested")
    result = GraphDataset(
        sample_ids=np.asarray(sample_ids),
        dates=np.asarray(requested_dates, dtype=object),
        symbols=expected_symbols,
        features=np.stack(matrices).astype(np.float32),
        adjacency=np.stack(adjacencies).astype(np.float32),
        feature_mask=np.stack(masks).astype(bool),
        targets=np.stack(labels).astype(np.float32),
        label_mask=np.stack(label_masks).astype(bool),
        decision_ids=np.stack(decision_ids),
        feature_names=selected_names,
        metadata={
            "adapter_only": True,
            "canonical_feature_order_verified": True,
            "topology": config.graph_topology,
            "include_text_score": include_text_score,
            "text_score_source": (
                "oof_by_decision_id"
                if text_scores_by_decision is not None
                else "disabled"
                if not include_text_score
                else "stage5_causal_or_pending_oof"
            ),
            "oof_text_scores_required": bool(require_oof_text_scores),
        },
    )
    if result.features.shape[-1] != len(selected_names):
        raise AssertionError("Graph feature selection shape does not match selected schema")
    return result


def graph_feature_schema(config: TrainingConfig | None = None) -> dict[str, Any]:
    feature_names = (
        configured_graph_feature_names(config)
        if config is not None
        else GRAPH_FEATURE_NAMES
    )
    include_text_score = "text_score" in feature_names
    return {
        "version": "canonical-graph-features-v2",
        "feature_names": list(feature_names),
        "input_dim": len(feature_names),
        "topology": config.graph_topology if config is not None else "full",
        "include_text_score": include_text_score,
        "feature_blocks": {
            "market_features": list(MARKET_FEATURE_NAMES),
            "action_signal": list(ACTION_FEATURE_NAMES),
            "text_score": list(TEXT_SCORE_FEATURE_NAMES) if include_text_score else [],
            "prior_shadow_features": list(SHADOW_FEATURE_NAMES),
            "prior_psychology_features": list(PSYCHOLOGY_FEATURE_NAMES),
        },
        "required_policy_dependent_fields": sorted(
            POLICY_DEPENDENT_GRAPH_FIELDS
            if include_text_score
            else POLICY_DEPENDENT_GRAPH_FIELDS - {"text_score"}
        ),
        "psychology_encoding": {
            "greed": [1, 0, 0],
            "fear": [0, 1, 0],
            "neutral": [0, 0, 1],
        },
        "targets_in_forward_inputs": False,
    }
