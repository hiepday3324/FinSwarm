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
from puppy.fusion.text_encoder import TextStreamEncoder  # noqa: E402
from puppy.portfolio_ext.allocation_engine import signals_to_target_weights  # noqa: E402
from puppy.portfolio_ext.master_portfolio import MasterPortfolio  # noqa: E402
from puppy.run_type import RunMode  # noqa: E402
from puppy.swarm.orchestrator import SwarmOrchestrator  # noqa: E402
from puppy.swarm.run_swarm import build_mock_agents, deterministic_embedding  # noqa: E402


DEFAULT_REALIZED_RETURNS = {"TSLA": 0.06, "NVDA": 0.00, "AAPL": -0.04}


def run_milestone_3_master_portfolio_feedback(
    event_store_path: str | None = None,
    realized_returns_by_symbol: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run AgentOutput -> allocation -> MasterPortfolio -> next context."""

    path = event_store_path or "data/event_store/milestone_3_master_portfolio.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    agents = build_mock_agents()
    symbols = [agent.symbol for agent in agents]
    orchestrator = SwarmOrchestrator(
        agents=agents,
        text_encoder=TextStreamEncoder(
            embedding_func=deterministic_embedding,
            embedding_model="deterministic-m3-master",
        ),
    )
    portfolio = MasterPortfolio(initial_cash=100000.0)
    trade_date = dt.date(2026, 1, 2)
    feedback_date = trade_date + dt.timedelta(days=1)
    next_context_date = trade_date + dt.timedelta(days=2)

    day_t = orchestrator.run_day(
        cur_date=trade_date,
        market_info_by_symbol=_market_info_by_symbol(agents, trade_date),
        run_mode=RunMode.Test,
    )
    allocation = signals_to_target_weights(
        date=trade_date,
        symbols=symbols,
        action_signals={
            output.signal.symbol: output.signal.action_signal
            for output in day_t["agent_outputs"]
        },
        max_weight=0.35,
    )
    initial_prices = {symbol: 100.0 for symbol in symbols}
    trade_snapshot = portfolio.apply_target_weights(
        date=trade_date,
        prices=initial_prices,
        target_weights=allocation,
        as_of=_cutoff(trade_date),
    )
    returns = realized_returns_by_symbol or DEFAULT_REALIZED_RETURNS
    feedback_prices = {
        symbol: initial_prices[symbol] * (1.0 + float(returns.get(symbol, 0.0)))
        for symbol in symbols
    }
    feedback_snapshot = portfolio.apply_target_weights(
        date=feedback_date,
        prices=feedback_prices,
        target_weights=allocation,
        as_of=_cutoff(feedback_date),
    )
    day_t2 = orchestrator.run_day(
        cur_date=next_context_date,
        market_info_by_symbol=_market_info_by_symbol(agents, next_context_date),
        run_mode=RunMode.Test,
        master_portfolio_state=feedback_snapshot,
    )

    event_store = EventStore(path)
    _store_day(event_store, day_t)
    event_store.append_allocation_weights(trade_date, allocation)
    event_store.append_portfolio_snapshot(trade_date, trade_snapshot)
    event_store.append_portfolio_snapshot(feedback_date, feedback_snapshot)
    for context in day_t2["contexts"]:
        event_store.append_event("PsychologyState", next_context_date, context.psychology_state)
    _store_day(event_store, day_t2)
    return {
        "trade_date": trade_date,
        "feedback_date": feedback_date,
        "next_context_date": next_context_date,
        "day_t": day_t,
        "allocation": allocation,
        "trade_snapshot": trade_snapshot,
        "feedback_snapshot": feedback_snapshot,
        "day_t2": day_t2,
        "feedback_context_raw": [item.raw_context for item in day_t2["text_features"]],
        "event_store_path": path,
    }


def _market_info_by_symbol(agents: list[Any], cur_date: dt.date) -> dict[str, tuple]:
    return {
        agent.symbol: (cur_date, 100.0, None, None, [], 0.0, False)
        for agent in agents
    }


def _cutoff(value: dt.date) -> dt.datetime:
    return dt.datetime.combine(value, dt.time(16, 10), dt.timezone.utc)


def _store_day(event_store: EventStore, result: dict[str, Any]) -> None:
    date = result["date"]
    for output in result["agent_outputs"]:
        event_store.append_agent_output(date, output)
    for context in result["contexts"]:
        event_store.append_event("AgentContext", date, context)
    for feature in result["text_features"]:
        event_store.append_text_feature(date, feature)
    event_store.append_event("AllocationDecision", date, result["allocation"])


def main() -> None:
    summary = run_milestone_3_master_portfolio_feedback()
    printable = {
        "trade_date": summary["trade_date"],
        "feedback_date": summary["feedback_date"],
        "next_context_date": summary["next_context_date"],
        "allocation": summary["allocation"].model_dump(mode="json"),
        "feedback_snapshot": summary["feedback_snapshot"].model_dump(mode="json"),
        "event_store_path": summary["event_store_path"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
