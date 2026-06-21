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
from puppy.portfolio_ext.shadow_portfolio import ShadowPortfolio  # noqa: E402
from puppy.run_type import RunMode  # noqa: E402
from puppy.swarm.orchestrator import SwarmOrchestrator  # noqa: E402
from puppy.swarm.psychology import build_psychology_state  # noqa: E402
from puppy.swarm.run_swarm import (  # noqa: E402
    build_mock_agents,
    deterministic_embedding,
)


DEFAULT_REALIZED_RETURNS = {
    "TSLA": 0.06,
    "NVDA": 0.00,
    "AAPL": -0.04,
}


def run_milestone_3_shadow(
    event_store_path: str | None = None,
    realized_returns_by_symbol: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the Tien-side Milestone 3/4 feedback slice.

    The demo follows the roadmap flow:

    AgentOutput_t -> h_text_t -> ShadowPortfolioState_t+1
    -> PsychologyState_t+1 -> Context_t+2.
    """

    path = event_store_path or "data/event_store/milestone_3_shadow.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    agents = build_mock_agents()
    text_encoder = TextStreamEncoder(
        embedding_func=deterministic_embedding,
        embedding_model="deterministic-m3-m4",
    )
    orchestrator = SwarmOrchestrator(agents=agents, text_encoder=text_encoder)

    trade_date = dt.date(2026, 1, 2)
    feedback_date = trade_date + dt.timedelta(days=1)
    next_context_date = trade_date + dt.timedelta(days=2)

    day_t = orchestrator.run_day(
        cur_date=trade_date,
        market_info_by_symbol=_market_info_by_symbol(agents, trade_date),
        run_mode=RunMode.Test,
    )

    returns = realized_returns_by_symbol or DEFAULT_REALIZED_RETURNS
    shadow_portfolio = ShadowPortfolio(initial_value=1.0)
    shadow_states = []
    for output in day_t["agent_outputs"]:
        signal = output.signal
        shadow_states.append(
            shadow_portfolio.update(
                agent_id=signal.agent_id,
                symbol=signal.symbol,
                action_signal=signal.action_signal,
                realized_return=returns.get(signal.symbol, 0.0),
                date=feedback_date,
            )
        )

    shadow_state_by_agent = {state.agent_id: state for state in shadow_states}
    psychology_state_by_agent = {
        state.agent_id: build_psychology_state(state) for state in shadow_states
    }

    day_t2 = orchestrator.run_day(
        cur_date=next_context_date,
        market_info_by_symbol=_market_info_by_symbol(agents, next_context_date),
        run_mode=RunMode.Test,
        shadow_state_by_agent=shadow_state_by_agent,
        psychology_state_by_agent=psychology_state_by_agent,
    )

    event_store = EventStore(path)
    _store_day(event_store, day_t)
    for state in shadow_states:
        event_store.append_event("ShadowPortfolioState", state.date, state)
    for state in psychology_state_by_agent.values():
        event_store.append_event("PsychologyState", feedback_date, state)
    _store_day(event_store, day_t2)

    return {
        "trade_date": trade_date,
        "feedback_date": feedback_date,
        "next_context_date": next_context_date,
        "day_t": day_t,
        "shadow_states": shadow_states,
        "psychology_states": psychology_state_by_agent,
        "day_t2": day_t2,
        "feedback_context_raw": [
            feature.raw_context for feature in day_t2["text_features"]
        ],
        "event_store_path": path,
    }


def _market_info_by_symbol(agents: list[Any], cur_date: dt.date) -> dict[str, tuple]:
    return {
        agent.symbol: (cur_date, 100.0, None, None, [], 0.0, False)
        for agent in agents
    }


def _store_day(event_store: EventStore, result: dict[str, Any]) -> None:
    date = result["date"]
    for output in result["agent_outputs"]:
        event_store.append_agent_output(date, output)
    for context in result["contexts"]:
        event_store.append_event("AgentContext", date, context)
    for feature in result["text_features"]:
        event_store.append_text_feature(date, feature)
    event_store.append_event("AllocationDecision", date, result["allocation"])


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_3_shadow()
    printable = {
        "trade_date": summary["trade_date"],
        "feedback_date": summary["feedback_date"],
        "next_context_date": summary["next_context_date"],
        "agent_outputs_t": [
            output.model_dump(mode="json") for output in summary["day_t"]["agent_outputs"]
        ],
        "shadow_states": [
            state.model_dump(mode="json") for state in summary["shadow_states"]
        ],
        "psychology_states": summary["psychology_states"],
        "text_features_t2": [
            feature.model_dump(mode="json")
            for feature in summary["day_t2"]["text_features"]
        ],
        "event_store_path": summary["event_store_path"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
