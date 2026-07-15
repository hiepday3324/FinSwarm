from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from demo.allocator import allocate
from demo.constants import DEFAULT_BUNDLE_DIR, DISPLAY_ORDER, NODE_ORDER, SCENARIO_DIR
from demo.data_loader import ReplayStore


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _allocation_dispersion(actions: dict[str, str], probabilities: dict[str, float]) -> float:
    result = allocate(actions, probabilities, capital=100_000.0)
    weights = np.asarray([result["weights"][symbol] for symbol in DISPLAY_ORDER])
    return float(np.max(weights) - np.min(weights))


def build_scenarios(
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    *,
    scenario_dir: str | Path = SCENARIO_DIR,
) -> list[dict[str, Any]]:
    """Select train-only presentation dates without reading outcomes or labels."""

    bundle = Path(bundle_dir).resolve()
    target_dir = Path(scenario_dir).resolve()
    store = ReplayStore(bundle)
    with np.load(bundle / "replay_expected_outputs.npz", allow_pickle=False) as data:
        attention = np.asarray(data["attention_mean"], dtype=np.float32)
        fusion_probability = np.asarray(data["fusion_probability"], dtype=np.float32)
    symbol_index = {symbol: index for index, symbol in enumerate(NODE_ORDER)}
    candidates: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in DISPLAY_ORDER}
    for date_index in range(1, len(store.dates)):
        date = store.dates[date_index]
        records = {row["symbol"]: row for row in store.records_for_date(date)}
        news = store.news_for_date(date)
        actions = {symbol: str(records[symbol]["decision"]) for symbol in DISPLAY_ORDER}
        probabilities = {
            symbol: float(fusion_probability[date_index, symbol_index[symbol]])
            for symbol in DISPLAY_ORDER
        }
        separation = float(np.max(list(probabilities.values())) - np.min(list(probabilities.values())))
        allocation_dispersion = _allocation_dispersion(actions, probabilities)
        delta = attention[date_index] - attention[date_index - 1]
        for central in DISPLAY_ORDER:
            central_news = [row for row in news if central in row["agents_received"]]
            if not central_news:
                continue
            node = symbol_index[central]
            connected = np.concatenate((np.abs(delta[node, :]), np.abs(delta[:, node])))
            attention_shift = float(np.max(connected))
            context = records[central]["context_snapshot"].get("agent_context", {})
            completeness = sum(
                bool(context.get(field))
                for field in ("memory_refs", "psychology_state", "shadow_state")
            ) / 3.0
            score = (
                5.0 * attention_shift
                + 2.0 * separation
                + allocation_dispersion
                + 0.20 * min(len(central_news), 3)
                + 0.15 * completeness
            )
            candidates[central].append(
                {
                    "date": date.isoformat(),
                    "date_index": date_index,
                    "central_symbol": central,
                    "main_headline": central_news[-1]["headline"],
                    "agent_actions": actions,
                    "attention_shift": attention_shift,
                    "fusion_probability_separation": separation,
                    "allocation_weight_dispersion": allocation_dispersion,
                    "score": score,
                    "selection_basis": [
                        "direct causal news provenance",
                        "stored Agent context and output",
                        "sample-specific mean attention shift",
                        "fusion probability separation",
                        "rule-based allocation dispersion",
                    ],
                    "selection_excluded_fields": [
                        "outcomes",
                        "targets",
                        "prediction correctness",
                        "validation metrics",
                        "test metrics",
                    ],
                    "why_visually_informative": (
                        f"{central} has {len(central_news)} directly routed news item(s); "
                        f"the largest connected mean-attention shift is "
                        f"{attention_shift:.3f}, while Fusion probabilities span "
                        f"{separation:.3f}."
                    ),
                }
            )
    selected: list[dict[str, Any]] = []
    used_dates: set[str] = set()
    for central in DISPLAY_ORDER:
        ranked = sorted(candidates[central], key=lambda row: row["score"], reverse=True)
        choice = next((row for row in ranked if row["date"] not in used_dates), None)
        if choice is None and ranked:
            choice = ranked[0]
        if choice is None:
            raise RuntimeError(f"No train-only replay scenario found for {central}")
        used_dates.add(choice["date"])
        selected.append(choice)
        _atomic_json(target_dir / f"{central.lower()}_centred.json", choice)
    index = {
        "version": "finswarm-e4-demo-scenarios-v1",
        "selection_partition": "effective_train_only",
        "uses_outcomes": False,
        "uses_validation": False,
        "uses_test": False,
        "scenarios": selected,
    }
    _atomic_json(target_dir / "index.json", index)
    return selected


def main() -> None:
    scenarios = build_scenarios()
    print(json.dumps({"scenarios": scenarios}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
