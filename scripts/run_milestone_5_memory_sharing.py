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
from puppy.memory_ext.memory_sharing import MemorySharingService  # noqa: E402
from puppy.swarm.context_builder import build_agent_context  # noqa: E402
from puppy.swarm.run_swarm import (  # noqa: E402
    build_mock_agents,
    deterministic_embedding,
)
from scripts.run_milestone_5_graph import run_milestone_5_graph  # noqa: E402


def run_milestone_5_memory_sharing(
    event_store_path: str | None = None,
) -> dict[str, Any]:
    """Apply graph-generated MemoryShareRoute artifacts to SectorAgents."""

    path = event_store_path or "data/event_store/milestone_5_memory_sharing.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    graph_summary = run_milestone_5_graph(event_store_path=path)
    routes = graph_summary["routes"]
    agents = build_mock_agents()
    source_memory_stores = _source_memory_stores()
    event_store = EventStore(path)
    service = MemorySharingService()
    events = service.apply_memory_share_routes(
        routes=routes,
        source_memory_stores=source_memory_stores,
        target_agents={agent.agent_id: agent for agent in agents},
        event_store=event_store,
    )

    context_date = dt.date(2026, 1, 7)
    encoder = TextStreamEncoder(
        embedding_func=deterministic_embedding,
        embedding_model="deterministic-m5-memory-sharing",
    )
    contexts = [
        build_agent_context(sector_agent=agent, cur_date=context_date)
        for agent in agents
    ]
    text_features = [encoder.encode(context) for context in contexts]
    for context in contexts:
        event_store.append_event("AgentContext", context.date, context)
    for feature in text_features:
        event_store.append_text_feature(feature.date, feature)

    return {
        "num_routes": len(routes),
        "num_memory_share_events": len(events),
        "num_injected_agents": sum(1 for agent in agents if agent.external_context_buffer),
        "events": events,
        "contexts": contexts,
        "text_features": text_features,
        "event_store_path": path,
    }


def _source_memory_stores() -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        "TSLA": {
            "short": [
                {
                    "memory_id": "TSLA-m5-short-1",
                    "text": "TSLA price action is influencing peer risk appetite.",
                    "layer": "short",
                    "symbol": "TSLA",
                    "score": 0.91,
                }
            ],
        },
        "NVDA": {
            "short": [
                {
                    "memory_id": "NVDA-m5-short-1",
                    "text": "NVDA semiconductor demand can spill over to adjacent names.",
                    "layer": "short",
                    "symbol": "NVDA",
                    "score": 0.88,
                }
            ],
        },
        "AAPL": {
            "short": [
                {
                    "memory_id": "AAPL-m5-short-1",
                    "text": "AAPL hardware demand weakness can affect supplier sentiment.",
                    "layer": "short",
                    "symbol": "AAPL",
                    "score": 0.84,
                }
            ],
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_5_memory_sharing()
    printable = {
        "num_routes": summary["num_routes"],
        "num_memory_share_events": summary["num_memory_share_events"],
        "num_injected_agents": summary["num_injected_agents"],
        "event_store_path": summary["event_store_path"],
        "events": [event.model_dump(mode="json") for event in summary["events"]],
        "text_features": [
            feature.model_dump(mode="json") for feature in summary["text_features"]
        ],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
