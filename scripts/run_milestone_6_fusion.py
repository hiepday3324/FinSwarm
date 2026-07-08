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
from puppy.fusion.dataset import build_text_score_dataset  # noqa: E402
from puppy.fusion.gated_fusion import GatedFusion  # noqa: E402
from puppy.fusion.text_feature_contract import build_text_feature_matrix  # noqa: E402
from puppy.fusion.text_score_head import TextScoreHead  # noqa: E402
from puppy.graph.graph_builder import build_gat_graph_output  # noqa: E402


SYMBOLS = ["TSLA", "NVDA", "AAPL"]


def _text_features(date: dt.date) -> list[TextFeature]:
    vectors = {
        "TSLA": [0.8, 0.3, 0.1],
        "NVDA": [0.2, 0.7, 0.4],
        "AAPL": [-0.4, -0.2, 0.1],
    }
    return [
        TextFeature(
            agent_id=f"sector_{symbol.lower()}",
            symbol=symbol,
            date=date,
            h_text=vectors[symbol],
            raw_context=f"Contract-valid fusion context for {symbol}.",
            embedding_model="deterministic-m6",
            dim=3,
        )
        for symbol in SYMBOLS
    ]


def _quant_table(date: dt.date) -> QuantFeatureTable:
    return QuantFeatureTable(
        date=date,
        symbols=SYMBOLS,
        action_signals={"TSLA": 1, "NVDA": 1, "AAPL": -1},
        returns={"TSLA": 0.02, "NVDA": 0.01, "AAPL": -0.01},
        labels={"TSLA": 1, "NVDA": 1, "AAPL": 0},
        metadata={"source": "milestone_6_fusion_mock"},
    )


def _shadow_states(date: dt.date) -> list[ShadowPortfolioState]:
    return [
        ShadowPortfolioState(
            date=date,
            agent_id=f"sector_{symbol.lower()}",
            symbol=symbol,
            value=value,
            roi=value - 1.0,
            rolling_return=value - 1.0,
            sharpe=0.5,
            drawdown=drawdown,
            win_rate=win_rate,
        )
        for symbol, value, drawdown, win_rate in [
            ("TSLA", 1.04, 0.0, 1.0),
            ("NVDA", 1.02, -0.01, 0.5),
            ("AAPL", 0.99, -0.03, 0.0),
        ]
    ]


def run_milestone_6_fusion(event_store_path: str | None = None) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_6_fusion.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    date = dt.date(2026, 1, 8)
    text_features = _text_features(date)
    quant_table = _quant_table(date)
    build_text_feature_matrix(text_features)
    x, _y, metadata = build_text_score_dataset(text_features, [quant_table])
    text_head = TextScoreHead(input_dim=x.shape[1], seed=11, epochs=0)
    probabilities = text_head.predict_proba(x)
    text_scores = {
        item["symbol"]: float(score) for item, score in zip(metadata, probabilities)
    }
    graph_output = build_gat_graph_output(
        date=date,
        quant_table=quant_table,
        shadow_states=_shadow_states(date),
        text_features=text_features,
        return_history={
            "TSLA": [0.01, 0.03, -0.01, 0.02],
            "NVDA": [0.02, 0.04, -0.02, 0.03],
            "AAPL": [-0.01, -0.02, 0.01, -0.01],
        },
    )
    fusion_output = GatedFusion().predict(
        date=date,
        text_scores=text_scores,
        graph_output=graph_output,
    )

    event_store = EventStore(path)
    event_store.append_fusion_output(date, fusion_output)
    result_store = ResultStore(event_store)
    stored_outputs = result_store.read_fusion_outputs()
    return {
        "num_fusion_outputs": len(stored_outputs),
        "symbols": fusion_output.symbols,
        "text_scores": fusion_output.text_scores,
        "graph_scores": fusion_output.graph_scores,
        "fusion_scores": fusion_output.fusion_scores,
        "gates": fusion_output.gates,
        "conflict_flags": fusion_output.conflict_flags,
        "event_store_path": path,
        "fusion_output": fusion_output,
        "graph_output": graph_output,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def main() -> None:
    summary = run_milestone_6_fusion()
    printable = {
        "num_fusion_outputs": summary["num_fusion_outputs"],
        "symbols": summary["symbols"],
        "text_scores": summary["text_scores"],
        "graph_scores": summary["graph_scores"],
        "fusion_scores": summary["fusion_scores"],
        "gates": summary["gates"],
        "conflict_flags": summary["conflict_flags"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
