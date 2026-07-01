from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.common.schemas import AgentOutput, AgentSignal, QuantFeatureTable  # noqa: E402
from puppy.data_engine.event_store import EventStore  # noqa: E402
from puppy.data_engine.result_store import ResultStore  # noqa: E402
from puppy.portfolio_ext.shadow_portfolio import ShadowPortfolio  # noqa: E402


SYMBOLS = ["TSLA", "NVDA", "AAPL"]


def _agent_outputs(date: dt.date) -> list[AgentOutput]:
    decisions = {"TSLA": "buy", "NVDA": "sell", "AAPL": "hold"}
    signals = {"TSLA": 1, "NVDA": -1, "AAPL": 0}
    return [
        AgentOutput(
            signal=AgentSignal(
                agent_id=f"sector_{symbol.lower()}",
                symbol=symbol,
                date=date,
                decision=decisions[symbol],
                action_signal=signals[symbol],
                reason=f"Mock {symbol} signal for shadow update.",
            ),
            model_name="milestone_4_shadow_mock",
        )
        for symbol in SYMBOLS
    ]


def _quant_tables() -> list[QuantFeatureTable]:
    start = dt.date(2026, 1, 3)
    rows = [
        (
            {"TSLA": 1, "NVDA": -1, "AAPL": 0},
            {"TSLA": 0.04, "NVDA": -0.02, "AAPL": None},
        ),
        (
            {"TSLA": -1, "NVDA": 1, "AAPL": 1},
            {"TSLA": -0.03, "NVDA": 0.01, "AAPL": 0.015},
        ),
        (
            {"TSLA": 1, "NVDA": 0, "AAPL": -1},
            {"TSLA": 0.02, "NVDA": None, "AAPL": -0.01},
        ),
    ]
    return [
        QuantFeatureTable(
            date=start + dt.timedelta(days=index),
            symbols=SYMBOLS,
            action_signals=action_signals,
            returns=returns,
            metadata={"source": "milestone_4_shadow_mock"},
        )
        for index, (action_signals, returns) in enumerate(rows)
    ]


def run_milestone_4_shadow(event_store_path: str | None = None) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_4_shadow.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    event_store = EventStore(path)
    result_store = ResultStore(event_store)
    shadow_portfolio = ShadowPortfolio(initial_value=1.0)
    all_states = []

    for quant_table in _quant_tables():
        trade_date = quant_table.date - dt.timedelta(days=1)
        states = shadow_portfolio.update_from_quant_table(
            quant_table=quant_table,
            agent_outputs=_agent_outputs(trade_date),
        )
        all_states.extend(states)
        for state in states:
            event_store.append_shadow_portfolio_state(state.date, state)

    stored_states = result_store.read_shadow_portfolio_states()
    last_state_by_agent_symbol = {
        f"{state.agent_id}|{state.symbol}": state.model_dump(mode="json")
        for state in stored_states
    }
    return {
        "num_shadow_states": len(stored_states),
        "agents": sorted({state.agent_id for state in stored_states}),
        "symbols": sorted({state.symbol for state in stored_states}),
        "last_state_by_agent_symbol": last_state_by_agent_symbol,
        "event_store_path": path,
        "states": all_states,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_4_shadow()
    printable = {
        "num_shadow_states": summary["num_shadow_states"],
        "agents": summary["agents"],
        "symbols": summary["symbols"],
        "last_state_by_agent_symbol": summary["last_state_by_agent_symbol"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
