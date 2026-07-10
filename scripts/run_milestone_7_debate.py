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
from puppy.debate.coordinator import DebateCoordinator  # noqa: E402
from puppy.fusion.text_encoder import TextStreamEncoder  # noqa: E402
from puppy.run_type import RunMode  # noqa: E402
from puppy.swarm.orchestrator import SwarmOrchestrator  # noqa: E402
from puppy.swarm.run_swarm import (  # noqa: E402
    build_mock_agents,
    deterministic_embedding,
)


def _fusion_output(date: dt.date) -> FusionOutput:
    return FusionOutput(
        date=date,
        symbols=["TSLA", "NVDA", "AAPL"],
        text_scores={"TSLA": 0.80, "NVDA": 0.55, "AAPL": 0.32},
        graph_scores={"TSLA": -0.65, "NVDA": 0.25, "AAPL": -0.40},
        fusion_scores={"TSLA": 0.42, "NVDA": 0.58, "AAPL": 0.18},
        gates={"TSLA": 0.50, "NVDA": 0.48, "AAPL": 0.52},
        conflict_flags={"TSLA": True, "NVDA": True, "AAPL": False},
        metadata={
            "source": "milestone_7_debate_mock",
            "scope": "tien_debate_layer",
        },
    )


def run_milestone_7_debate(event_store_path: str | None = None) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_7_debate.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    date = dt.date(2026, 1, 9)
    agents = build_mock_agents()
    text_encoder = TextStreamEncoder(
        embedding_func=deterministic_embedding,
        embedding_model="deterministic-m7-debate",
    )
    orchestrator = SwarmOrchestrator(agents=agents, text_encoder=text_encoder)
    market_info_by_symbol = {
        agent.symbol: (date, 100.0, None, None, [], 0.0, False)
        for agent in agents
    }
    swarm_result = orchestrator.run_day(
        cur_date=date,
        market_info_by_symbol=market_info_by_symbol,
        run_mode=RunMode.Test,
    )
    fusion_output = _fusion_output(date)

    event_store = EventStore(path)
    for output in swarm_result["agent_outputs"]:
        event_store.append_agent_output(date, output)
    for feature in swarm_result["text_features"]:
        event_store.append_text_feature(date, feature)
    event_store.append_fusion_output(date, fusion_output)

    coordinator = DebateCoordinator()
    verdicts = coordinator.run_for_fusion(
        fusion_output=fusion_output,
        agent_signals=[output.signal for output in swarm_result["agent_outputs"]],
        sector_agents={agent.agent_id: agent for agent in agents},
        event_store=event_store,
    )

    result_store = ResultStore(event_store)
    stored_requests = result_store.read_debate_requests()
    stored_transcripts = result_store.read_debate_transcripts()
    stored_verdicts = result_store.read_debate_verdicts()
    return {
        "num_agent_outputs": len(result_store.read_agent_outputs()),
        "num_text_features": len(result_store.read_text_features()),
        "num_fusion_outputs": len(result_store.read_fusion_outputs()),
        "num_debate_requests": len(stored_requests),
        "num_debate_transcripts": len(stored_transcripts),
        "num_debate_verdicts": len(stored_verdicts),
        "debated_symbols": [verdict.symbol for verdict in verdicts],
        "verdicts": verdicts,
        "event_store_path": path,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_7_debate()
    printable = {
        "num_agent_outputs": summary["num_agent_outputs"],
        "num_text_features": summary["num_text_features"],
        "num_fusion_outputs": summary["num_fusion_outputs"],
        "num_debate_requests": summary["num_debate_requests"],
        "num_debate_transcripts": summary["num_debate_transcripts"],
        "num_debate_verdicts": summary["num_debate_verdicts"],
        "debated_symbols": summary["debated_symbols"],
        "verdicts": [
            verdict.model_dump(mode="json") for verdict in summary["verdicts"]
        ],
        "event_store_path": summary["event_store_path"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
