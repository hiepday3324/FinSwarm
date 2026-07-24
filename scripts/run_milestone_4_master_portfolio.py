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
from puppy.portfolio_ext.allocation_engine import signals_to_target_weights  # noqa: E402
from puppy.portfolio_ext.master_portfolio import MasterPortfolio  # noqa: E402


SYMBOLS = ["TSLA", "NVDA", "AAPL"]


def run_milestone_4_master_portfolio(
    event_store_path: str | None = None,
) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_4_master_portfolio.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    dates = [dt.date(2026, 1, 3) + dt.timedelta(days=index) for index in range(3)]
    signals = [
        {"TSLA": 1, "NVDA": -1, "AAPL": 0},
        {"TSLA": -1, "NVDA": 1, "AAPL": 1},
        {"TSLA": 1, "NVDA": 0, "AAPL": -1},
    ]
    returns = [
        {"TSLA": 0.04, "NVDA": -0.02, "AAPL": 0.0},
        {"TSLA": -0.03, "NVDA": 0.01, "AAPL": 0.015},
        {"TSLA": 0.02, "NVDA": 0.0, "AAPL": -0.01},
    ]
    prices = {"TSLA": 250.0, "NVDA": 500.0, "AAPL": 200.0}
    portfolio = MasterPortfolio(initial_cash=100000.0)
    event_store = EventStore(path)

    for date, action_signals, daily_returns in zip(dates, signals, returns):
        prices = {
            symbol: prices[symbol] * (1.0 + daily_returns[symbol])
            for symbol in SYMBOLS
        }
        allocation = signals_to_target_weights(
            date=date,
            symbols=SYMBOLS,
            action_signals=action_signals,
            max_weight=0.35,
        )
        snapshot = portfolio.apply_target_weights(
            date=date,
            prices=prices,
            target_weights=allocation,
            as_of=dt.datetime.combine(date, dt.time(16, 10), dt.timezone.utc),
        )
        event_store.append_allocation_weights(date, allocation)
        event_store.append_portfolio_snapshot(date, snapshot)

    result_store = ResultStore(event_store)
    snapshots = result_store.read_portfolio_snapshots()
    return {
        "num_portfolio_snapshots": len(snapshots),
        "symbols": SYMBOLS,
        "final_snapshot": snapshots[-1],
        "event_store_path": path,
        "snapshots": snapshots,
    }


def main() -> None:
    summary = run_milestone_4_master_portfolio()
    printable = {
        "num_portfolio_snapshots": summary["num_portfolio_snapshots"],
        "symbols": summary["symbols"],
        "final_snapshot": summary["final_snapshot"].model_dump(mode="json"),
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
