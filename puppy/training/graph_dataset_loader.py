from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.dataset as ds
import yaml

from .config import TrainingConfig
from .split import EffectiveSplits
from .training_guard import SplitAccessGuard


GRAPH_FEATURE_NAMES = (
    "observed_close_return_1d",
    "observed_intraday_return_1d",
    "momentum_5d",
    "momentum_20d",
    "rolling_volatility_5d",
    "rolling_volatility_20d",
    "volume_ratio_20d",
    "range_pct",
    "feature_missing_count",
)
POLICY_DEPENDENT_GRAPH_FIELDS = {
    "action_signal",
    "shadow_roi_prior",
    "shadow_rolling_return_prior",
    "shadow_drawdown_prior",
    "shadow_sharpe_prior",
    "shadow_win_rate_prior",
    "text_norm",
    "num_text_features",
}


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
) -> GraphDataset:
    guard = guard or SplitAccessGuard()
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
    missing = [name for name in GRAPH_FEATURE_NAMES if name not in source_names]
    if missing:
        raise ValueError(f"Stage 5 graph artifact is missing required features: {missing}")
    if POLICY_DEPENDENT_GRAPH_FIELDS.intersection(GRAPH_FEATURE_NAMES):
        raise AssertionError("Policy-dependent field entered Graph Stream v1 schema")
    indices = [source_names.index(name) for name in GRAPH_FEATURE_NAMES]

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
            day_targets = np.zeros(len(symbols), dtype=np.float32)
            day_mask = np.zeros(len(symbols), dtype=bool)
            day_ids = np.full(len(symbols), "", dtype=object)
            for node, symbol in enumerate(symbols):
                outcome = outcomes.get((date, symbol))
                if outcome is None:
                    continue
                day_ids[node] = outcome["decision_id"]
                label = int(outcome["direction_label"])
                eligible = (
                    bool(outcome["train_mask"])
                    and bool(outcome["supervised_eligible"])
                    and not bool(outcome["cross_split_outcome"])
                    and label in (-1, 1)
                )
                if eligible:
                    day_targets[node] = float(label == 1)
                    day_mask[node] = True
                elif label not in (-1, 0, 1):
                    raise ValueError(f"Unexpected graph target {label} on {date} {symbol}")
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
    )
    if result.features.shape[-1] != len(GRAPH_FEATURE_NAMES):
        raise AssertionError("Graph feature selection shape does not match selected schema")
    return result


def graph_feature_schema() -> dict[str, Any]:
    return {
        "version": "graph-training-features-v1",
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "excluded_policy_dependent_fields": sorted(POLICY_DEPENDENT_GRAPH_FIELDS),
        "targets_in_forward_inputs": False,
    }
