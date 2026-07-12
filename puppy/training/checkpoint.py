from __future__ import annotations

import json
import os
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainOnlyScaler:
    feature_names: list[str]
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def fit(
        self,
        values: np.ndarray,
        *,
        dates: list[Any],
        input_manifest_hash: str,
    ) -> "TrainOnlyScaler":
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("Scaler requires a non-empty 2D train matrix")
        if values.shape[1] != len(self.feature_names):
            raise ValueError("Scaler feature-name count does not match matrix width")
        if len(dates) != len(values):
            raise ValueError("Scaler dates must align one-to-one with fit rows")
        self.mean = np.nanmean(values, axis=0)
        self.scale = np.nanstd(values, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 0), self.scale, 1.0)
        self.metadata = {
            "fit_start_date": min(dates).isoformat(),
            "fit_end_date": max(dates).isoformat(),
            "fit_row_count": len(values),
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "input_manifest_hash": input_manifest_hash,
        }
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler has not been fit")
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("Scaler transform matrix has an unexpected shape")
        finite = np.where(np.isfinite(values), values, self.mean)
        return ((finite - self.mean) / self.scale).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.metadata is None:
            raise RuntimeError("Scaler has not been fit")
        atomic_json(path, self.metadata)

    @classmethod
    def load(cls, path: str | Path) -> "TrainOnlyScaler":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=list(raw["feature_names"]),
            mean=np.asarray(raw["mean"], dtype=np.float64),
            scale=np.asarray(raw["scale"], dtype=np.float64),
            metadata=raw,
        )


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
