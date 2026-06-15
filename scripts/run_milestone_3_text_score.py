from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.common.schemas import QuantFeatureTable, TextFeature
from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore
from puppy.fusion.dataset import build_text_score_dataset
from puppy.fusion.train_loop import train_text_score_head


SYMBOLS = ["TSLA", "NVDA", "AAPL"]


def _feature_vector(symbol: str, day_index: int) -> list[float]:
    if symbol == "TSLA":
        return [1.0, 0.4 + day_index * 0.05, 0.2]
    if symbol == "NVDA":
        return [-1.0, -0.3 - day_index * 0.05, 0.1]
    return [-0.8, 0.0, -0.2 - day_index * 0.05]


def build_mock_text_score_artifacts() -> tuple[list[TextFeature], list[QuantFeatureTable]]:
    start = dt.date(2026, 1, 2)
    text_features: list[TextFeature] = []
    quant_tables: list[QuantFeatureTable] = []
    for day_index in range(3):
        date = start + dt.timedelta(days=day_index)
        labels = {"TSLA": 1, "NVDA": -1, "AAPL": 0}
        quant_tables.append(
            QuantFeatureTable(
                date=date,
                symbols=SYMBOLS,
                action_signals={"TSLA": 1, "NVDA": -1, "AAPL": 0},
                returns={"TSLA": 0.02, "NVDA": -0.01, "AAPL": 0.0},
                labels=labels,
                metadata={"source": "milestone_3_text_score_mock"},
            )
        )
        for symbol in SYMBOLS:
            text_features.append(
                TextFeature(
                    agent_id=f"fake_{symbol.lower()}",
                    symbol=symbol,
                    date=date,
                    h_text=_feature_vector(symbol, day_index),
                    raw_context=f"Deterministic text feature for {symbol} on {date}.",
                    embedding_model="deterministic-m3",
                    dim=3,
                )
            )
    return text_features, quant_tables


def run_milestone_3_text_score(event_store_path: str | None = None) -> dict[str, Any]:
    path = event_store_path or "data/event_store/milestone_3_text_score.jsonl"
    if event_store_path is None and os.path.exists(path):
        os.remove(path)

    event_store = EventStore(path)
    text_features, quant_tables = build_mock_text_score_artifacts()
    for feature in text_features:
        event_store.append_text_feature(feature.date, feature)
    for table in quant_tables:
        event_store.append_event("QuantFeatureTable", table.date, table)

    result_store = ResultStore(event_store)
    stored_text_features = result_store.read_text_features()
    stored_quant_tables = result_store.read_quant_feature_tables()
    x, y, metadata = build_text_score_dataset(stored_text_features, stored_quant_tables)
    model, metrics = train_text_score_head(stored_text_features, stored_quant_tables, seed=7)
    scores = model.predict_proba(x)

    text_scores = [
        {
            "date": item["date"],
            "symbol": item["symbol"],
            "agent_id": item["agent_id"],
            "label": item["label"],
            "score": float(score),
        }
        for item, score in zip(metadata, scores)
    ]
    return {
        "num_text_features": len(stored_text_features),
        "num_quant_tables": len(stored_quant_tables),
        "num_samples": metrics["num_samples"],
        "input_dim": metrics["input_dim"],
        "accuracy": metrics["accuracy"],
        "positive_rate": metrics["positive_rate"],
        "text_scores": text_scores,
        "event_store_path": path,
    }


def main() -> None:
    summary = run_milestone_3_text_score()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
