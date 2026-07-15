from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from .checkpoint import atomic_json, seed_everything, state_dict_sha256
from .config import TrainingConfig
from .fusion_trainer import (
    CANONICAL_FUSION_SELECTION,
    _arrays,
    load_fusion_model,
    selected_fusion_logits,
)
from .metrics import binary_metrics
from .split import EffectiveSplits
from .training_guard import SplitAccessGuard


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    if len(logits) == 0 or len(np.unique(targets)) < 2:
        raise ValueError("Temperature scaling requires non-empty samples with both classes")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    targets_tensor = torch.as_tensor(targets, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tensor / temperature,
            targets_tensor,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))
    original = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tensor, targets_tensor
        )
    )
    calibrated = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tensor / temperature, targets_tensor
        )
    )
    return temperature if calibrated <= original + 1e-10 else 1.0


def calibrate_fusion(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    guard: SplitAccessGuard | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    guard = guard or SplitAccessGuard()
    seed_everything(config.seed if seed is None else int(seed))
    guard.assert_allowed("validation_calibration", "calibration")
    rows = pq.read_table(
        config.model_dir / "validation_base_predictions.parquet",
        filters=[("partition", "=", "validation_calibration")],
    ).to_pylist()
    expected_dates = set(splits.validation_calibration)
    if not rows or any(row["decision_date"] not in expected_dates for row in rows):
        raise ValueError("Calibration received rows outside validation-calibration")
    model, selected = load_fusion_model(
        config.model_dir / "fusion_model.pt", device=device
    )
    values = _arrays(
        rows,
        require_latents=selected == CANONICAL_FUSION_SELECTION,
    )
    before_hash = state_dict_sha256(model.state_dict())
    logits = selected_fusion_logits(model, selected, values, device=device)
    temperature = fit_temperature(logits, values["target"])
    after_hash = state_dict_sha256(model.state_dict())
    if before_hash != after_hash:
        raise RuntimeError("Calibration modified fusion model weights")
    uncalibrated_probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    calibrated_probability = 1.0 / (
        1.0 + np.exp(-np.clip(logits / temperature, -60.0, 60.0))
    )
    result = {
        "method": "temperature",
        "temperature": temperature,
        "fit_start_date": min(row["decision_date"] for row in rows).isoformat(),
        "fit_end_date": max(row["decision_date"] for row in rows).isoformat(),
        "sample_count": len(rows),
        "partition": "validation_calibration",
        "selected_fusion_model": selected,
        "uncalibrated_metrics": binary_metrics(
            values["target"],
            uncalibrated_probability,
            symbols=values["symbol"],
        ),
        "calibrated_metrics": binary_metrics(
            values["target"],
            calibrated_probability,
            symbols=values["symbol"],
        ),
        "fusion_weights_unchanged": True,
        "fusion_state_sha256": before_hash,
        "test_access_count": guard.test_feature_access_count,
    }
    atomic_json(config.model_dir / "calibration.json", result)
    return result
