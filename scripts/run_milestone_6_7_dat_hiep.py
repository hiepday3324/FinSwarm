from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.data_engine.event_store import EventStore  # noqa: E402
from puppy.data_engine.result_store import ResultStore  # noqa: E402
from puppy.portfolio_ext.allocation_engine import fusion_to_target_weights  # noqa: E402
from puppy.portfolio_ext.master_portfolio import MasterPortfolio  # noqa: E402
from scripts.run_milestone_6_fusion import run_milestone_6_fusion  # noqa: E402


def run_milestone_6_7_dat_hiep(
    event_store_path: str | None = None,
) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_6_7_dat_hiep.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    fusion_summary = run_milestone_6_fusion(event_store_path=path)
    fusion_output = fusion_summary["fusion_output"]
    allocation = fusion_to_target_weights(fusion_output=fusion_output)
    portfolio = MasterPortfolio(initial_cash=100000.0)
    snapshot = portfolio.apply_allocation_weights(
        allocation=allocation,
        prices={"TSLA": 250.0, "NVDA": 500.0, "AAPL": 200.0},
        date=fusion_output.date,
    )

    event_store = EventStore(path)
    event_store.append_allocation_weights(fusion_output.date, allocation)
    event_store.append_portfolio_snapshot(fusion_output.date, snapshot)
    result_store = ResultStore(event_store)
    return {
        "num_fusion_outputs": len(result_store.read_fusion_outputs()),
        "num_allocation_weights": len(result_store.read_allocation_weights()),
        "num_portfolio_snapshots": len(result_store.read_portfolio_snapshots()),
        "symbols": fusion_output.symbols,
        "fusion_scores": fusion_output.fusion_scores,
        "weights": allocation.weights,
        "cash_weight": allocation.cash_weight,
        "equity": snapshot.equity,
        "event_store_path": path,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_6_7_dat_hiep()
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
