from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.common.schemas import QuantFeatureTable, ShadowPortfolioState, TextFeature  # noqa: E402
from puppy.data_engine.event_store import EventStore  # noqa: E402
from puppy.data_engine.result_store import ResultStore  # noqa: E402
from puppy.graph.graph_builder import build_graph_output  # noqa: E402
from puppy.graph.risk_router import build_memory_share_routes  # noqa: E402


SYMBOLS = ["TSLA", "NVDA", "AAPL"]


def _quant_table(date: dt.date) -> QuantFeatureTable:
    return QuantFeatureTable(
        date=date,
        symbols=SYMBOLS,
        action_signals={"TSLA": 1, "NVDA": -1, "AAPL": 0},
        returns={"TSLA": 0.02, "NVDA": -0.01, "AAPL": 0.0},
        labels={"TSLA": 1, "NVDA": -1, "AAPL": 0},
        metadata={"source": "milestone_5_graph_mock"},
    )


def _shadow_states(date: dt.date) -> list[ShadowPortfolioState]:
    return [
        ShadowPortfolioState(
            date=date,
            agent_id="sector_tsla",
            symbol="TSLA",
            value=1.04,
            roi=0.04,
            rolling_return=0.04,
            sharpe=0.8,
            drawdown=0.0,
            win_rate=1.0,
        ),
        ShadowPortfolioState(
            date=date,
            agent_id="sector_nvda",
            symbol="NVDA",
            value=1.02,
            roi=0.02,
            rolling_return=0.02,
            sharpe=0.5,
            drawdown=-0.01,
            win_rate=0.5,
        ),
        ShadowPortfolioState(
            date=date,
            agent_id="sector_aapl",
            symbol="AAPL",
            value=1.0,
            roi=0.0,
            rolling_return=0.0,
            sharpe=0.0,
            drawdown=-0.02,
            win_rate=0.0,
        ),
    ]


def _text_features(date: dt.date) -> list[TextFeature]:
    vectors = {
        "TSLA": [0.4, 0.3, 0.1],
        "NVDA": [-0.2, 0.5, 0.3],
        "AAPL": [0.1, -0.1, 0.2],
    }
    return [
        TextFeature(
            agent_id=f"sector_{symbol.lower()}",
            symbol=symbol,
            date=date,
            h_text=vectors[symbol],
            raw_context=f"Mock text feature for {symbol}.",
            embedding_model="milestone_5_graph_mock",
            dim=3,
        )
        for symbol in SYMBOLS
    ]


def run_milestone_5_graph(event_store_path: str | None = None) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_5_graph.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    date = dt.date(2026, 1, 6)
    return_history = {
        "TSLA": [0.01, 0.02, -0.01, 0.03],
        "NVDA": [0.02, 0.04, -0.02, 0.06],
        "AAPL": [-0.01, -0.02, 0.01, -0.03],
    }
    agent_by_symbol = {
        "TSLA": "sector_tsla",
        "NVDA": "sector_nvda",
        "AAPL": "sector_aapl",
    }

    graph_output = build_graph_output(
        date=date,
        quant_table=_quant_table(date),
        shadow_states=_shadow_states(date),
        text_features=_text_features(date),
        return_history=return_history,
    )
    routes = build_memory_share_routes(
        graph_output=graph_output,
        agent_by_symbol=agent_by_symbol,
        threshold=0.49,
    )

    event_store = EventStore(path)
    event_store.append_graph_output(date, graph_output)
    for route in routes:
        event_store.append_memory_share_route(date, route)

    result_store = ResultStore(event_store)
    stored_graph_outputs = result_store.read_graph_outputs()
    stored_routes = result_store.read_memory_share_routes()
    return {
        "num_graph_outputs": len(stored_graph_outputs),
        "num_memory_share_routes": len(stored_routes),
        "symbols": graph_output.symbols,
        "top_routes": [
            {
                "source_symbol": route.source_symbol,
                "target_symbol": route.target_symbol,
                "alpha": route.alpha,
            }
            for route in stored_routes[:3]
        ],
        "event_store_path": path,
        "graph_output": graph_output,
        "routes": stored_routes,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_5_graph()
    printable = {
        "num_graph_outputs": summary["num_graph_outputs"],
        "num_memory_share_routes": summary["num_memory_share_routes"],
        "symbols": summary["symbols"],
        "top_routes": summary["top_routes"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
