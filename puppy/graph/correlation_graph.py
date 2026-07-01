"""Correlation graph MVP for asset relationships."""

from __future__ import annotations

import math


def build_correlation_adjacency(
    return_history: dict[str, list[float]] | None,
    symbols: list[str],
    min_abs_corr: float = 0.2,
    default_edge_weight: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Build ``adjacency[target][source]`` from supplied return histories."""

    adjacency: dict[str, dict[str, float]] = {}
    histories = return_history or {}
    for target in symbols:
        adjacency[target] = {}
        for source in symbols:
            if source == target:
                adjacency[target][source] = 1.0
                continue

            corr = _pearson_or_none(histories.get(target), histories.get(source))
            if corr is None:
                weight = default_edge_weight
            else:
                weight = corr if abs(corr) >= min_abs_corr else 0.0
            adjacency[target][source] = _clamp(weight)
    return adjacency


def build_attention_from_adjacency(
    adjacency: dict[str, dict[str, float]],
    use_abs: bool = True,
    normalize: bool = True,
) -> dict[str, dict[str, float]]:
    """Convert adjacency weights to ``attention[target][source]`` weights."""

    attention: dict[str, dict[str, float]] = {}
    for target, sources in adjacency.items():
        target_attention: dict[str, float] = {}
        for source, raw_weight in sources.items():
            if source == target:
                target_attention[source] = 0.0
                continue
            weight = abs(raw_weight) if use_abs else max(raw_weight, 0.0)
            target_attention[source] = float(weight)

        if normalize:
            total = sum(target_attention.values())
            if total > 0.0:
                target_attention = {
                    source: weight / total
                    for source, weight in target_attention.items()
                }
        attention[target] = target_attention
    return attention


def _pearson_or_none(
    left: list[float] | None,
    right: list[float] | None,
) -> float | None:
    if not left or not right:
        return None

    paired = [
        (float(left_value), float(right_value))
        for left_value, right_value in zip(left, right)
        if _is_finite(left_value) and _is_finite(right_value)
    ]
    if len(paired) < 2:
        return None

    xs = [value[0] for value in paired]
    ys = [value[1] for value in paired]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = denom_x * denom_y
    if denominator == 0.0:
        return None
    return _clamp(numerator / denominator)


def _is_finite(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))
