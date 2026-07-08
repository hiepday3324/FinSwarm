from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.common.schemas import FusionOutput  # noqa: E402
from puppy.data_engine.event_store import EventStore  # noqa: E402
from puppy.data_engine.result_store import ResultStore  # noqa: E402
from puppy.portfolio_ext.allocation_engine import fusion_to_target_weights  # noqa: E402
from puppy.portfolio_ext.master_portfolio import MasterPortfolio  # noqa: E402


def _fusion_output(date: dt.date) -> FusionOutput:
    return FusionOutput(
        date=date,
        symbols=["TSLA", "NVDA", "AAPL"],
        text_scores={"TSLA": 0.78, "NVDA": 0.62, "AAPL": 0.35},
        graph_scores={"TSLA": 0.6, "NVDA": 0.2, "AAPL": -0.4},
        fusion_scores={"TSLA": 0.76, "NVDA": 0.58, "AAPL": 0.33},
        gates={"TSLA": 0.52, "NVDA": 0.48, "AAPL": 0.50},
        conflict_flags={"TSLA": False, "NVDA": True, "AAPL": False},
        metadata={"source": "milestone_7_allocation_mock"},
    )


def run_milestone_7_allocation(event_store_path: str | None = None) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_7_allocation.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    date = dt.date(2026, 1, 9)
    fusion_output = _fusion_output(date)
    allocation = fusion_to_target_weights(fusion_output, max_weight=0.35)
    portfolio = MasterPortfolio(initial_cash=100000.0)
    snapshot = portfolio.apply_allocation_weights(
        allocation=allocation,
        prices={"TSLA": 250.0, "NVDA": 500.0, "AAPL": 200.0},
        date=date,
    )

    event_store = EventStore(path)
    event_store.append_fusion_output(date, fusion_output)
    event_store.append_allocation_weights(date, allocation)
    event_store.append_portfolio_snapshot(date, snapshot)
    result_store = ResultStore(event_store)
    return {
        "num_fusion_outputs": len(result_store.read_fusion_outputs()),
        "num_allocation_weights": len(result_store.read_allocation_weights()),
        "num_portfolio_snapshots": len(result_store.read_portfolio_snapshots()),
        "weights": allocation.weights,
        "cash_weight": allocation.cash_weight,
        "equity": snapshot.equity,
        "event_store_path": path,
        "fusion_output": fusion_output,
        "allocation": allocation,
        "portfolio_snapshot": snapshot,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_7_allocation()
    printable = {
        "num_fusion_outputs": summary["num_fusion_outputs"],
        "num_allocation_weights": summary["num_allocation_weights"],
        "num_portfolio_snapshots": summary["num_portfolio_snapshots"],
        "weights": summary["weights"],
        "cash_weight": summary["cash_weight"],
        "equity": summary["equity"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
