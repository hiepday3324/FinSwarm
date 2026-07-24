from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.constants import (  # noqa: E402
    ATTENTION_CONVENTION,
    DEFAULT_BUNDLE_DIR,
    DISPLAY_ORDER,
    EXPECTED_ALIGNED_ROWS,
    EXPECTED_SUPERVISED_ROWS,
    EXPECTED_TRAIN_DATES,
    NODE_ORDER,
    TEXT_SCORE_SOURCE,
    GRAPH_PREDICTION_SOURCE,
    FUSION_PREDICTION_SOURCE,
    TRAIN_END,
    TRAIN_START,
)
from demo.scenario_builder import build_scenarios  # noqa: E402
from demo.training import protected_snapshot  # noqa: E402
from puppy.fusion.scalar_logit_stack import ScalarLogitStack  # noqa: E402
from puppy.fusion.text_score_head import TextScoreHead  # noqa: E402
from puppy.graph.gat_model import DenseSignedGAT  # noqa: E402
from puppy.training.checkpoint import TrainOnlyScaler, state_dict_sha256  # noqa: E402
from puppy.training.fusion_trainer import SCALAR_FUSION_SELECTION  # noqa: E402
from puppy.training.text_trainer import predict_mlp  # noqa: E402


TEXT_DIM = 384
FEATURE_NAMES = (
    "return_1d",
    "return_5d",
    "volume_z",
    "volatility_10d",
    "action_signal",
    "text_score",
    "master_symbol_weight_prior",
    "master_cash_weight_prior",
    "risk_appetite",
    "uncertainty",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_dates() -> list[dt.date]:
    manifest_path = ROOT / "data" / "manifests" / "price_split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = [
            dt.date.fromisoformat(value)
            for value in manifest["splits"]["train"]["dates"][:EXPECTED_TRAIN_DATES]
        ]
    else:
        dates = []
        current = dt.date.fromisoformat(TRAIN_START)
        while len(dates) < EXPECTED_TRAIN_DATES:
            if current.weekday() < 5:
                dates.append(current)
            current += dt.timedelta(days=1)
    if len(dates) != EXPECTED_TRAIN_DATES:
        raise ValueError(f"Expected {EXPECTED_TRAIN_DATES} train dates, got {len(dates)}")
    if dates[0].isoformat() != TRAIN_START or dates[-1].isoformat() != TRAIN_END:
        raise ValueError("Mock replay dates do not match demo constants")
    return dates


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _build_inputs(dates: list[dt.date]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    day = np.arange(len(dates), dtype=np.float32)[:, None]
    node = np.arange(len(NODE_ORDER), dtype=np.float32)[None, :]
    dim = np.arange(TEXT_DIM, dtype=np.float32)[None, None, :]
    text = (
        0.50 * np.sin((day[:, :, None] + 1.0) / 17.0 + dim / 23.0)
        + 0.35 * np.cos((node[:, :, None] + 1.0) * (dim + 3.0) / 71.0)
        + 0.08 * (node[:, :, None] - 1.0)
    ).astype(np.float32)

    features = np.zeros((len(dates), len(NODE_ORDER), len(FEATURE_NAMES)), dtype=np.float32)
    for t in range(len(dates)):
        for n, symbol in enumerate(NODE_ORDER):
            phase = (t + 1) * (n + 1)
            action = [-1.0, 0.0, 1.0][(t + n) % 3]
            features[t, n] = np.asarray(
                [
                    0.025 * np.sin(phase / 9.0),
                    0.040 * np.cos(phase / 21.0),
                    0.700 * np.sin((t + 3 * n) / 13.0),
                    0.180 + 0.030 * np.cos(phase / 17.0),
                    action,
                    0.5,
                    0.020 * np.cos((t + n) / 11.0),
                    0.250 + 0.050 * n,
                    0.550 + 0.180 * np.sin((t + n) / 19.0),
                    0.350 + 0.150 * np.cos((t + 2 * n) / 23.0),
                ],
                dtype=np.float32,
            )
            if symbol == "NVDA":
                features[t, n, 1] += 0.015
            if symbol == "TSLA":
                features[t, n, 3] += 0.020

    mask = np.ones_like(features, dtype=bool)
    adjacency = np.zeros((len(dates), len(NODE_ORDER), len(NODE_ORDER)), dtype=np.float32)
    for t in range(len(dates)):
        base = np.asarray(
            [
                [1.0, 0.64 + 0.05 * np.sin(t / 17.0), 0.52 + 0.04 * np.cos(t / 19.0)],
                [0.57 + 0.03 * np.cos(t / 13.0), 1.0, 0.60 + 0.05 * np.sin(t / 23.0)],
                [0.50 + 0.04 * np.sin(t / 11.0), 0.55 + 0.04 * np.cos(t / 29.0), 1.0],
            ],
            dtype=np.float32,
        )
        adjacency[t] = base

    decision_ids = np.asarray(
        [
            [f"mock-{date.isoformat()}-{symbol}" for symbol in NODE_ORDER]
            for date in dates
        ],
        dtype="U64",
    )
    return text, features, mask, adjacency, decision_ids


def _feature_schema() -> dict[str, Any]:
    return {
        "schema_version": "mock-demo-graph-feature-schema-v2-master-portfolio",
        "feature_names": list(FEATURE_NAMES),
        "feature_blocks": {
            "market_features": ["return_1d", "return_5d", "volume_z", "volatility_10d"],
            "action_signal": ["action_signal"],
            "text_score": ["text_score"],
            "prior_master_portfolio_features": [
                "master_symbol_weight_prior",
                "master_cash_weight_prior",
            ],
            "prior_psychology_features": ["risk_appetite", "uncertainty"],
        },
        "mock_bundle": True,
    }


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _model_hashes(
    text_model: TextScoreHead,
    graph_model: DenseSignedGAT,
    fusion_model: ScalarLogitStack,
) -> dict[str, str]:
    return {
        "text_state_sha256": state_dict_sha256(text_model.state_dict()),
        "gat_state_sha256": state_dict_sha256(graph_model.state_dict()),
        "scalar_fusion_state_sha256": state_dict_sha256(fusion_model.state_dict()),
    }


def _write_models(
    output_dir: Path,
    text_model: TextScoreHead,
    graph_model: DenseSignedGAT,
    fusion_model: ScalarLogitStack,
) -> dict[str, str]:
    _freeze(text_model)
    _freeze(graph_model)
    _freeze(fusion_model)
    hashes = _model_hashes(text_model, graph_model, fusion_model)
    torch.save(
        {
            "model_type": "canonical_text_score_head",
            "architecture": text_model.architecture_config(),
            "state_dict": text_model.state_dict(),
            "seed": 42,
            "best_epoch": 0,
            "checkpoint_selection": "mock_deterministic",
            "training_source": "mock_demo_bundle",
            "prediction_source": TEXT_SCORE_SOURCE,
            "frozen_for_ui": True,
        },
        output_dir / "text_model.pt",
    )
    torch.save(
        {
            "model_type": "canonical_dense_signed_gat",
            "architecture": graph_model.architecture_config(),
            "state_dict": graph_model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "topology": "full",
            "include_text_score": True,
            "symbols": list(NODE_ORDER),
            "seed": 42,
            "best_epoch": 0,
            "checkpoint_selection": "mock_deterministic",
            "training_source": "mock_demo_bundle",
            "text_score_source": TEXT_SCORE_SOURCE,
            "prediction_source": GRAPH_PREDICTION_SOURCE,
            "attention_convention": ATTENTION_CONVENTION,
            "frozen_for_ui": True,
        },
        output_dir / "graph_model.pt",
    )
    coefficients = {
        "intercept": float(fusion_model.intercept.detach().cpu()),
        "weight_text": float(fusion_model.weights[0].detach().cpu()),
        "weight_graph": float(fusion_model.weights[1].detach().cpu()),
    }
    torch.save(
        {
            "model_type": SCALAR_FUSION_SELECTION,
            "architecture": fusion_model.architecture_config(),
            "state_dict": fusion_model.state_dict(),
            "selected_fusion_model": SCALAR_FUSION_SELECTION,
            "seed": 42,
            "best_epoch": 0,
            "checkpoint_selection": "mock_deterministic",
            "training_source": "mock_demo_bundle",
            "prediction_source": FUSION_PREDICTION_SOURCE,
            "coefficients": coefficients,
            "regularization": {"weight_decay": 0.0},
            "frozen_for_ui": True,
        },
        output_dir / "fusion_model.pt",
    )
    return hashes


def _expected_outputs(
    text_model: TextScoreHead,
    graph_model: DenseSignedGAT,
    fusion_model: ScalarLogitStack,
    text_scaler: TrainOnlyScaler,
    graph_scaler: TrainOnlyScaler,
    text_embeddings: np.ndarray,
    raw_graph_features: np.ndarray,
    feature_mask: np.ndarray,
    adjacency: np.ndarray,
) -> dict[str, np.ndarray]:
    text_logits = []
    text_probability = []
    graph_logits = []
    graph_probability = []
    attention_heads = []
    attention_mean = []
    fusion_logits = []
    fusion_probability = []
    text_score_index = FEATURE_NAMES.index("text_score")
    for index in range(len(text_embeddings)):
        text_features = text_scaler.transform(text_embeddings[index])
        cur_text_logits, cur_text_probability = predict_mlp(
            text_model, text_features, device="cpu"
        )
        raw_graph = raw_graph_features[index].copy()
        raw_graph[:, text_score_index] = cur_text_probability
        scaled_graph = graph_scaler.transform(raw_graph)
        scaled_graph = np.where(feature_mask[index], scaled_graph, 0.0).astype(np.float32)
        with torch.no_grad():
            graph_output = graph_model(
                torch.as_tensor(scaled_graph[None], dtype=torch.float32),
                torch.as_tensor(adjacency[index][None], dtype=torch.float32),
                torch.as_tensor(feature_mask[index][None], dtype=torch.bool),
                symbols=NODE_ORDER,
                feature_names=FEATURE_NAMES,
                feature_provenance={"source": "mock_demo_bundle"},
            )
            cur_graph_logits = graph_output.logits[0].detach().cpu().numpy()
            cur_graph_probability = graph_output.probabilities[0].detach().cpu().numpy()
            cur_attention_heads = graph_model.last_attention_heads[0].detach().cpu().numpy()
            cur_fusion_logits = fusion_model(
                torch.as_tensor(cur_text_logits, dtype=torch.float32),
                torch.as_tensor(cur_graph_logits, dtype=torch.float32),
            ).detach().cpu().numpy()
        text_logits.append(cur_text_logits.astype(np.float32))
        text_probability.append(cur_text_probability.astype(np.float32))
        graph_logits.append(cur_graph_logits.astype(np.float32))
        graph_probability.append(cur_graph_probability.astype(np.float32))
        attention_heads.append(cur_attention_heads.astype(np.float32))
        attention_mean.append(cur_attention_heads.mean(axis=0).astype(np.float32))
        fusion_logits.append(cur_fusion_logits.astype(np.float32))
        fusion_probability.append(_sigmoid(cur_fusion_logits).astype(np.float32))
    return {
        "text_logits": np.asarray(text_logits, dtype=np.float32),
        "text_probability": np.asarray(text_probability, dtype=np.float32),
        "graph_logits": np.asarray(graph_logits, dtype=np.float32),
        "graph_probability": np.asarray(graph_probability, dtype=np.float32),
        "attention_heads": np.asarray(attention_heads, dtype=np.float32),
        "attention_mean": np.asarray(attention_mean, dtype=np.float32),
        "fusion_logits": np.asarray(fusion_logits, dtype=np.float32),
        "fusion_probability": np.asarray(fusion_probability, dtype=np.float32),
    }


def _decision(probability: float) -> str:
    if probability >= 0.56:
        return "buy"
    if probability <= 0.44:
        return "sell"
    return "hold"


def _records(
    dates: list[dt.date],
    decision_ids: np.ndarray,
    expected: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    replay_records: list[dict[str, Any]] = []
    news_records: list[dict[str, Any]] = []
    index = {
        "version": "mock-finswarm-e4-demo-index-v1",
        "dates": [value.isoformat() for value in dates],
        "symbols": list(NODE_ORDER),
        "decision_ids_by_date": {},
        "mock_bundle": True,
    }
    for date_index, date in enumerate(dates):
        date_key = date.isoformat()
        index["decision_ids_by_date"][date_key] = decision_ids[date_index].astype(str).tolist()
        master_portfolio_state = None
        if date_index > 0:
            prior_date = dates[date_index - 1]
            equity = 100000.0 * (1.0 + 0.0008 * date_index)
            weights = {
                symbol: round(0.20 + 0.04 * symbol_index, 4)
                for symbol_index, symbol in enumerate(NODE_ORDER)
            }
            stock_weight = sum(weights.values())
            master_portfolio_state = {
                "date": prior_date.isoformat(),
                "as_of": f"{prior_date.isoformat()}T16:10:00-04:00",
                "cash": round(equity * (1.0 - stock_weight), 2),
                "cash_weight": round(1.0 - stock_weight, 4),
                "positions": {},
                "weights": weights,
                "equity": round(equity, 2),
                "daily_return": 0.0008,
                "cumulative_return": round(0.0008 * date_index, 6),
                "rolling_return": round(0.0008 * date_index, 6),
                "sharpe": 0.5,
                "drawdown": -0.005,
                "win_rate": 0.6,
                "metadata": {"source": "mock_master_portfolio"},
            }
        for symbol_index, symbol in enumerate(NODE_ORDER):
            probability = float(expected["fusion_probability"][date_index, symbol_index])
            text_probability = float(expected["text_probability"][date_index, symbol_index])
            graph_probability = float(expected["graph_probability"][date_index, symbol_index])
            decision = _decision(probability)
            action_signal = {"buy": 1, "hold": 0, "sell": -1}[decision]
            news_group_id = f"news-{date_key}-{symbol.lower()}"
            event_id = f"event-{date_key}-{symbol.lower()}"
            memory_refs = [
                {
                    "memory_id": f"mem-{symbol.lower()}-{date_index:03d}-short",
                    "layer": "short",
                    "symbol": symbol,
                    "score": round(0.72 + 0.03 * symbol_index, 4),
                    "text_preview": f"{symbol} mock short-term context for {date_key}.",
                },
                {
                    "memory_id": f"mem-{symbol.lower()}-{date_index:03d}-reflection",
                    "layer": "reflection",
                    "symbol": symbol,
                    "score": round(0.64 + 0.02 * date_index / len(dates), 4),
                    "text_preview": f"{symbol} mock reflection cites prior decision behavior.",
                },
            ]
            psychology = {
                "state": ["cautious", "balanced", "confident"][(date_index + symbol_index) % 3],
                "risk_appetite": round(0.45 + 0.2 * np.sin((date_index + symbol_index) / 17.0), 4),
                "uncertainty": round(0.35 + 0.15 * np.cos((date_index + symbol_index) / 19.0), 4),
            }
            reason = (
                f"Mock replay: {symbol} {decision.upper()} because text probability "
                f"{text_probability:.3f}, graph probability {graph_probability:.3f}, "
                f"and fused probability {probability:.3f}."
            )
            context = {
                "agent_context": {
                    "symbol": symbol,
                    "date": date_key,
                    "latest_reason": reason,
                    "memory_refs": memory_refs,
                    "psychology_state": psychology,
                    "master_portfolio_state": master_portfolio_state,
                }
            }
            replay_records.append(
                {
                    "decision_id": str(decision_ids[date_index, symbol_index]),
                    "agent_id": f"sector_{symbol.lower()}",
                    "symbol": symbol,
                    "decision_date": date,
                    "decision_cutoff": f"{date_key}T16:10:00-04:00",
                    "decision": decision,
                    "action_signal": action_signal,
                    "reason": reason,
                    "text_input": (
                        f"{symbol} mock causal context on {date_key}. "
                        f"News {event_id}; memory refs cite short and reflection layers."
                    ),
                    "context_snapshot_json": json.dumps(context, sort_keys=True),
                    "memory_refs_json": json.dumps(memory_refs, sort_keys=True),
                    "news_event_ids_json": json.dumps([event_id], sort_keys=True),
                    "psychology_state_json": json.dumps(psychology, sort_keys=True),
                    "master_portfolio_state_json": json.dumps(
                        master_portfolio_state,
                        sort_keys=True,
                    ),
                    "raw_agent_output_json": json.dumps(
                        {
                            "signal": {
                                "decision": decision,
                                "action_signal": action_signal,
                                "probability": probability,
                            },
                            "reason": reason,
                            "mock_bundle": True,
                        },
                        sort_keys=True,
                    ),
                    "causal_provenance_json": json.dumps(
                        {
                            "mode": "mock_demo_bundle",
                            "provider": "none",
                            "model": "deterministic_mock",
                            "synthetic": True,
                        },
                        sort_keys=True,
                    ),
                    "cited_memory_ids_json": json.dumps(
                        {
                            "short": [memory_refs[0]["memory_id"]],
                            "reflection": [memory_refs[1]["memory_id"]],
                        },
                        sort_keys=True,
                    ),
                }
            )
            news_records.append(
                {
                    "news_group_id": news_group_id,
                    "decision_date": date,
                    "published_at": dt.datetime.combine(
                        date, dt.time(14, 15), tzinfo=dt.timezone.utc
                    ),
                    "decision_cutoff": f"{date_key}T16:10:00-04:00",
                    "source": "Mock financial news",
                    "headline": f"{symbol} mock market update for {date_key}",
                    "summary": f"Deterministic mock news routed to {symbol}.",
                    "causally_available": True,
                    "event_ids_json": json.dumps([event_id], sort_keys=True),
                    "related_symbols_json": json.dumps([symbol], sort_keys=True),
                    "agents_received_json": json.dumps([symbol], sort_keys=True),
                }
            )
    return replay_records, news_records, index


def _histories() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "roc_auc": 1.0,
        "note": "Deterministic mock bundle; not a research metric.",
    }
    text = {"best_epoch": 0, "train_metrics": metrics, "mock_bundle": True}
    graph = {"best_epoch": 0, "train_metrics": metrics, "mock_bundle": True}
    fusion = {
        "best_epoch": 0,
        "train_metrics": metrics,
        "coefficients": {"intercept": 0.0, "weight_text": 0.55, "weight_graph": 0.45},
        "regularization": {"weight_decay": 0.0},
        "mock_bundle": True,
    }
    return text, graph, fusion


def _bundle_manifest(output_dir: Path, model_hashes: dict[str, str]) -> dict[str, Any]:
    files = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    manifest = {
        "bundle_version": "finswarm-e4-demo-fit-v1",
        "run_id": "finswarm_e4_demo_fit",
        "demo_only": True,
        "mock_bundle": True,
        "intentionally_in_sample": True,
        "evaluation_mode": "train_replay",
        "model_frozen_during_ui": True,
        "selected_text_model": "canonical_text_score_head",
        "selected_graph_model": "canonical_dense_signed_gat",
        "selected_fusion_model": "scalar_logit_stack",
        "train_range": [TRAIN_START, TRAIN_END],
        "train_dates": EXPECTED_TRAIN_DATES,
        "aligned_replay_rows": EXPECTED_ALIGNED_ROWS,
        "supervised_train_rows": EXPECTED_SUPERVISED_ROWS,
        "node_order": list(NODE_ORDER),
        "attention_convention": ATTENTION_CONVENTION,
        "validation_access_count": 0,
        "test_access_count": 0,
        "file_sha256": files,
        "model_hashes": model_hashes,
    }
    _atomic_json(output_dir / "bundle_manifest.json", manifest)
    return manifest


def create_mock_bundle(
    output_dir: str | Path = DEFAULT_BUNDLE_DIR,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Bundle already exists; pass --overwrite: {target}")
    if target.exists() and overwrite:
        shutil.rmtree(target)
    build_dir = target.parent / f".{target.name}.mock-build-{uuid.uuid4().hex}"
    build_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.manual_seed(42)
        dates = _train_dates()
        text_embeddings, raw_graph_features, feature_mask, adjacency, decision_ids = _build_inputs(dates)

        text_scaler = TrainOnlyScaler([f"h_text_{index:03d}" for index in range(TEXT_DIM)])
        repeated_dates = [date for date in dates for _ in NODE_ORDER]
        text_scaler.fit(
            text_embeddings.reshape(-1, TEXT_DIM),
            dates=repeated_dates,
            input_manifest_hash="mock-demo-bundle",
        )
        graph_scaler = TrainOnlyScaler(list(FEATURE_NAMES))
        graph_scaler.fit(
            raw_graph_features.reshape(-1, len(FEATURE_NAMES)),
            dates=repeated_dates,
            input_manifest_hash="mock-demo-bundle",
        )

        text_model = TextScoreHead(input_dim=TEXT_DIM, hidden_dim=128, dropout=0.0, seed=42)
        graph_model = DenseSignedGAT(
            input_dim=len(FEATURE_NAMES), hidden_dim=32, num_heads=4, dropout=0.0
        )
        fusion_model = ScalarLogitStack()
        with torch.no_grad():
            fusion_model.intercept.fill_(0.0)
            fusion_model.weights[:] = torch.tensor([0.55, 0.45], dtype=torch.float32)

        expected = _expected_outputs(
            text_model,
            graph_model,
            fusion_model,
            text_scaler,
            graph_scaler,
            text_embeddings,
            raw_graph_features,
            feature_mask,
            adjacency,
        )
        model_hashes = _write_models(build_dir, text_model, graph_model, fusion_model)
        text_scaler.save(build_dir / "text_scaler.json")
        graph_scaler.save(build_dir / "graph_scaler.json")
        _atomic_json(build_dir / "feature_schema.json", _feature_schema())
        _atomic_json(build_dir / "node_order.json", {"symbols": list(NODE_ORDER)})
        _atomic_json(build_dir / "attention_contract.json", ATTENTION_CONVENTION)
        _atomic_json(build_dir / "model_state_hashes.json", model_hashes)
        _atomic_json(
            build_dir / "training_config.yaml",
            {
                "source": "scripts/create_demo_mock_bundle.py",
                "mock_bundle": True,
                "note": "This unblocks the Streamlit demo when real E4 artifacts are absent.",
            },
        )

        _atomic_npz(
            build_dir / "replay_inputs.npz",
            dates=np.asarray([value.isoformat() for value in dates], dtype="U10"),
            symbols=np.asarray(NODE_ORDER, dtype="U4"),
            decision_ids=decision_ids.astype("U64"),
            text_embeddings=text_embeddings.astype(np.float32),
            raw_graph_features=raw_graph_features.astype(np.float32),
            graph_feature_mask=feature_mask.astype(bool),
            adjacency=adjacency.astype(np.float32),
            graph_feature_names=np.asarray(FEATURE_NAMES, dtype="U48"),
        )
        _atomic_npz(build_dir / "replay_expected_outputs.npz", **expected)

        replay_records, news_records, replay_index = _records(dates, decision_ids, expected)
        _atomic_parquet(build_dir / "replay_records.parquet", replay_records)
        _atomic_parquet(build_dir / "replay_news.parquet", news_records)
        _atomic_json(build_dir / "replay_index.json", replay_index)

        text_history, graph_history, fusion_history = _histories()
        _atomic_json(build_dir / "text_training_history.json", text_history)
        _atomic_json(build_dir / "graph_training_history.json", graph_history)
        _atomic_json(build_dir / "fusion_training_history.json", fusion_history)
        protected = protected_snapshot()
        _atomic_json(
            build_dir / "training_metadata.json",
            {
                "run_id": "finswarm_e4_demo_fit",
                "demo_only": True,
                "mock_bundle": True,
                "intentionally_in_sample": True,
                "evaluation_mode": "train_replay",
                "model_frozen_during_ui": True,
                "train_start": TRAIN_START,
                "train_end": TRAIN_END,
                "train_dates": EXPECTED_TRAIN_DATES,
                "aligned_replay_rows": EXPECTED_ALIGNED_ROWS,
                "supervised_train_rows": EXPECTED_SUPERVISED_ROWS,
                "supervised_rows_by_symbol": {
                    symbol: EXPECTED_SUPERVISED_ROWS // len(NODE_ORDER)
                    for symbol in NODE_ORDER
                },
                "validation_access_count": 0,
                "test_access_count": 0,
                "validation_labels_accessed": False,
                "test_labels_accessed": False,
                "test_features_accessed": False,
                "checkpoint_selection": "minimum_train_loss",
                "validation_early_stopping": False,
                "calibration_fitted": False,
                "node_order": list(NODE_ORDER),
                "attention_shape": [EXPECTED_TRAIN_DATES, 4, len(NODE_ORDER), len(NODE_ORDER)],
                "attention_convention": ATTENTION_CONVENTION,
                "text_score_source": TEXT_SCORE_SOURCE,
                "graph_prediction_source": GRAPH_PREDICTION_SOURCE,
                "fusion_prediction_source": FUSION_PREDICTION_SOURCE,
                "text_metrics": text_history["train_metrics"],
                "graph_metrics": graph_history["train_metrics"],
                "fusion_metrics": fusion_history["train_metrics"],
                "fusion_coefficients": fusion_history["coefficients"],
                "model_hashes": model_hashes,
                "save_load_prediction_equivalence": {"all_passed": True},
                "source_access": {
                    "guard_access_log": [{"split": "train", "purpose": "mock_demo"}],
                    "test_feature_access_count": 0,
                    "files": [{"split": "train", "filter": "mock train only"}],
                },
                "protected_artifact_snapshot_before": protected,
                "protected_artifact_snapshot_after": protected,
                "protected_artifacts_unchanged": True,
                "device": "cpu",
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
        )
        _bundle_manifest(build_dir, model_hashes)
        os.replace(build_dir, target)
        scenarios = build_scenarios(target)
        return {
            "status": "created",
            "bundle_dir": target.as_posix(),
            "mock_bundle": True,
            "train_dates": len(dates),
            "aligned_replay_rows": EXPECTED_ALIGNED_ROWS,
            "scenario_count": len(scenarios),
        }
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic mock Streamlit demo bundle.")
    parser.add_argument("--output-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = create_mock_bundle(args.output_dir, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
