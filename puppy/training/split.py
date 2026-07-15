from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .training_guard import SplitAccessGuard


@dataclass(frozen=True)
class OOFFold:
    fold_id: int
    train_dates: tuple[dt.date, ...]
    embargo_dates: tuple[dt.date, ...]
    prediction_dates: tuple[dt.date, ...]


@dataclass(frozen=True)
class InnerChronologicalSplit:
    train_dates: tuple[dt.date, ...]
    embargo_dates: tuple[dt.date, ...]
    validation_dates: tuple[dt.date, ...]


@dataclass(frozen=True)
class EffectiveSplits:
    warmup: tuple[dt.date, ...]
    train_manifest: tuple[dt.date, ...]
    train_effective: tuple[dt.date, ...]
    embargoed_train: tuple[dt.date, ...]
    validation_selection: tuple[dt.date, ...]
    validation_calibration: tuple[dt.date, ...]
    test_metadata_dates: tuple[dt.date, ...]

    def dates(self, name: str, purpose: str, guard: SplitAccessGuard) -> tuple[dt.date, ...]:
        guard.assert_allowed(name, purpose)
        mapping = {
            "warmup": self.warmup,
            "train": self.train_effective,
            "validation_selection": self.validation_selection,
            "validation_calibration": self.validation_calibration,
            "test": self.test_metadata_dates,
        }
        return mapping[name]


def build_effective_splits(
    manifest_path: str | Path,
    *,
    embargo_trading_days: int,
    validation_selection_fraction: float,
) -> EffectiveSplits:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)["splits"]
    parse = lambda name: tuple(dt.date.fromisoformat(value) for value in raw[name]["dates"])
    train = parse("train")
    validation = parse("validation")
    if len(train) <= embargo_trading_days:
        raise ValueError("Training split is shorter than embargo")
    selection_count = int(math.floor(len(validation) * validation_selection_fraction + 1e-9))
    if selection_count <= 0 or selection_count >= len(validation):
        raise ValueError("Validation subdivision produced an empty segment")
    return EffectiveSplits(
        warmup=parse("warmup"),
        train_manifest=train,
        train_effective=train[:-embargo_trading_days],
        embargoed_train=train[-embargo_trading_days:],
        validation_selection=validation[:selection_count],
        validation_calibration=validation[selection_count:],
        test_metadata_dates=parse("test"),
    )


def build_oof_folds(
    train_dates: tuple[dt.date, ...],
    *,
    folds: int,
    min_train_days: int,
    embargo_days: int,
) -> list[OOFFold]:
    first_prediction_index = min_train_days + embargo_days
    prediction_pool = train_dates[first_prediction_index:]
    if len(prediction_pool) < folds:
        raise ValueError("Insufficient dates for requested OOF folds")
    block_size = math.ceil(len(prediction_pool) / folds)
    result: list[OOFFold] = []
    for fold_id in range(folds):
        start = first_prediction_index + fold_id * block_size
        end = min(first_prediction_index + (fold_id + 1) * block_size, len(train_dates))
        if start >= end:
            continue
        train_end = start - embargo_days
        fold_train = train_dates[:train_end]
        embargo = train_dates[train_end:start]
        predictions = train_dates[start:end]
        if len(fold_train) < min_train_days or len(embargo) != embargo_days:
            raise AssertionError("Invalid OOF train/embargo construction")
        result.append(OOFFold(fold_id, fold_train, embargo, predictions))
    return result


def build_inner_chronological_split(
    history_dates: tuple[dt.date, ...],
    *,
    embargo_days: int,
    validation_fraction: float = 0.2,
) -> InnerChronologicalSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Inner validation fraction must be between 0 and 1")
    if tuple(sorted(set(history_dates))) != history_dates:
        raise ValueError("Inner chronological history dates must be unique and sorted")
    validation_count = max(1, int(math.floor(len(history_dates) * validation_fraction)))
    validation_start = len(history_dates) - validation_count
    train_end = validation_start - embargo_days
    if train_end <= 0:
        raise ValueError(
            "Insufficient historical dates for inner chronological train, embargo, "
            "and validation segments"
        )
    train = history_dates[:train_end]
    embargo = history_dates[train_end:validation_start]
    validation = history_dates[validation_start:]
    if not train or len(embargo) != embargo_days or not validation:
        raise AssertionError("Invalid inner chronological split")
    if not max(train) < min(validation):
        raise AssertionError("Inner chronological train must precede validation")
    return InnerChronologicalSplit(train, embargo, validation)
