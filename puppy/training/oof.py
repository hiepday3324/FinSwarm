from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint import TrainOnlyScaler, atomic_json
from .config import TrainingConfig
from .graph_dataset_loader import GraphDataset, load_graph_dataset
from .graph_trainer import (
    fit_gat,
    fit_graph_scaler,
    load_gat,
    predict_gat,
    scale_graph_dataset,
)
from .split import EffectiveSplits, build_oof_folds
from .text_dataset import TextDataset, load_supervised_text_dataset
from .text_embedding import sha256_path
from .text_trainer import (
    LogisticBaseline,
    fit_mlp,
    predict_mlp,
    selected_text_predictions,
)
from .training_guard import SplitAccessGuard


def _write_parquet(path: Path, records: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression="zstd")
    os.replace(temporary, path)


def _scale_text(
    train: TextDataset,
    prediction: TextDataset,
    manifest_hash: str,
) -> tuple[TrainOnlyScaler, np.ndarray, np.ndarray]:
    scaler = TrainOnlyScaler([f"embedding_{index:03d}" for index in range(384)])
    scaler.fit(train.embeddings, dates=train.dates.tolist(), input_manifest_hash=manifest_hash)
    return scaler, scaler.transform(train.embeddings), scaler.transform(prediction.embeddings)


def generate_oof_predictions(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    guard: SplitAccessGuard | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    guard = guard or SplitAccessGuard()
    access_start = len(guard.access_log)
    seed = config.seed if seed is None else int(seed)
    guard.assert_allowed("train", "oof")
    text_data = load_supervised_text_dataset(
        config,
        splits,
        guard=guard,
        allowed_splits=("train",),
        allowed_dates=splits.train_effective,
        purpose="oof",
    )
    graph_data = load_graph_dataset(
        config,
        splits,
        dates=splits.train_effective,
        purpose="oof",
        guard=guard,
    )
    folds = build_oof_folds(
        splits.train_effective,
        folds=int(config.split["oof_folds"]),
        min_train_days=int(config.split["oof_min_train_days"]),
        embargo_days=int(config.split["embargo_trading_days"]),
    )
    manifest_hash = sha256_path(config.artifact_dir / "data_manifest.json")
    selected_text_model = json.loads(
        (config.model_dir / "text_training_history.json").read_text(encoding="utf-8")
    )["selected_text_model"]
    fixed_epochs = min(30, int(config.text_model["max_epochs"]), int(config.graph_model["max_epochs"]))
    date_positions = {date: index for index, date in enumerate(splits.train_effective)}
    records: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        text_train = text_data.subset(fold.train_dates)
        text_prediction = text_data.subset(fold.prediction_dates)
        graph_train_raw = graph_data.subset(fold.train_dates)
        graph_prediction_raw = graph_data.subset(fold.prediction_dates)
        if not len(text_train) or not len(text_prediction):
            raise ValueError(f"OOF fold {fold.fold_id} has empty text data")
        text_scaler, text_train_x, text_prediction_x = _scale_text(
            text_train, text_prediction, manifest_hash
        )
        if selected_text_model == "mlp":
            text_model, text_history = fit_mlp(
                text_train_x,
                text_train.targets,
                text_train_x,
                text_train.targets,
                config.text_model,
                seed=seed + fold.fold_id * 2,
                device=device,
                fixed_epochs=fixed_epochs,
            )
            text_logits, text_probability = predict_mlp(
                text_model, text_prediction_x, device=device
            )
        elif selected_text_model == "logistic_regression":
            text_model = LogisticBaseline.fit(
                text_train_x, text_train.targets, seed=seed + fold.fold_id * 2
            )
            text_logits = text_model.predict_logits(text_prediction_x)
            text_probability = text_model.predict_proba(text_prediction_x)
            text_history = {"parameter_changed": True, "model_fitted": True}
        else:
            raise ValueError(f"Unsupported selected Text Stream model: {selected_text_model}")
        if not text_history["parameter_changed"]:
            raise RuntimeError(f"OOF text model did not train in fold {fold.fold_id}")

        graph_scaler = fit_graph_scaler(
            graph_train_raw,
            input_manifest_hash=manifest_hash,
        )
        graph_train = scale_graph_dataset(graph_train_raw, graph_scaler)
        graph_prediction = scale_graph_dataset(graph_prediction_raw, graph_scaler)
        graph_model, graph_history = fit_gat(
            graph_train,
            graph_train,
            config.graph_model,
            seed=seed + fold.fold_id * 2 + 1,
            device=device,
            fixed_epochs=fixed_epochs,
        )
        graph_logits, graph_probability, _ = predict_gat(
            graph_model, graph_prediction, device=device
        )
        if not graph_history["parameter_changed"]:
            raise RuntimeError(f"OOF GAT did not train in fold {fold.fold_id}")
        graph_lookup: dict[tuple[Any, str], tuple[str, float, float]] = {}
        for day_index, date in enumerate(graph_prediction.dates):
            for node, symbol in enumerate(graph_prediction.symbols):
                graph_lookup[(date, symbol)] = (
                    str(graph_prediction.decision_ids[day_index, node]),
                    float(graph_logits[day_index, node]),
                    float(graph_probability[day_index, node]),
                )

        train_end = fold.train_dates[-1]
        prediction_start = fold.prediction_dates[0]
        embargo_distance = date_positions[prediction_start] - date_positions[train_end] - 1
        if embargo_distance != int(config.split["embargo_trading_days"]):
            raise ValueError(f"OOF embargo violation in fold {fold.fold_id}")
        for index in range(len(text_prediction)):
            date = text_prediction.dates[index]
            symbol = str(text_prediction.symbols[index])
            graph_value = graph_lookup.get((date, symbol))
            if graph_value is None:
                raise ValueError(f"Missing OOF graph prediction for {date} {symbol}")
            graph_decision_id, graph_logit, graph_prob = graph_value
            decision_id = str(text_prediction.decision_ids[index])
            if graph_decision_id != decision_id:
                raise ValueError(
                    f"OOF graph/text decision mismatch for {date} {symbol}"
                )
            if not train_end < date:
                raise ValueError(f"OOF in-sample prediction detected for {decision_id}")
            records.append(
                {
                    "decision_id": decision_id,
                    "decision_date": date,
                    "symbol": symbol,
                    "fold_id": fold.fold_id,
                    "model_train_start": fold.train_dates[0],
                    "model_train_end": train_end,
                    "prediction_start": prediction_start,
                    "prediction_end": fold.prediction_dates[-1],
                    "embargo_trading_days": int(config.split["embargo_trading_days"]),
                    "text_logit": float(text_logits[index]),
                    "text_probability": float(text_probability[index]),
                    "graph_logit": graph_logit,
                    "graph_probability": graph_prob,
                    "text_available": True,
                    "graph_available": True,
                    "target": int(text_prediction.targets[index]),
                    "is_oof": True,
                }
            )
        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "model_train_start": fold.train_dates[0].isoformat(),
                "model_train_end": train_end.isoformat(),
                "embargo_start": fold.embargo_dates[0].isoformat(),
                "embargo_end": fold.embargo_dates[-1].isoformat(),
                "embargo_trading_days": len(fold.embargo_dates),
                "prediction_start": prediction_start.isoformat(),
                "prediction_end": fold.prediction_dates[-1].isoformat(),
                "prediction_rows": len(text_prediction),
                "text_scaler_fit_start": text_scaler.metadata["fit_start_date"],
                "text_scaler_fit_end": text_scaler.metadata["fit_end_date"],
                "graph_scaler_fit_start": graph_scaler.metadata["fit_start_date"],
                "graph_scaler_fit_end": graph_scaler.metadata["fit_end_date"],
                "text_parameters_changed": text_history["parameter_changed"],
                "text_model": selected_text_model,
                "graph_parameters_changed": graph_history["parameter_changed"],
            }
        )
        del text_model, graph_model, text_scaler, graph_scaler
    records.sort(key=lambda row: (row["decision_date"], row["symbol"], row["decision_id"]))
    if len(records) < 30 or len({row["target"] for row in records}) < 2:
        raise ValueError("OOF coverage is too small or contains only one target class")
    schema = pa.schema(
        [
            ("decision_id", pa.string()),
            ("decision_date", pa.date32()),
            ("symbol", pa.string()),
            ("fold_id", pa.int16()),
            ("model_train_start", pa.date32()),
            ("model_train_end", pa.date32()),
            ("prediction_start", pa.date32()),
            ("prediction_end", pa.date32()),
            ("embargo_trading_days", pa.int16()),
            ("text_logit", pa.float32()),
            ("text_probability", pa.float32()),
            ("graph_logit", pa.float32()),
            ("graph_probability", pa.float32()),
            ("text_available", pa.bool_()),
            ("graph_available", pa.bool_()),
            ("target", pa.int8()),
            ("is_oof", pa.bool_()),
        ]
    )
    output_path = config.model_dir / "oof_predictions.parquet"
    _write_parquet(output_path, records, schema)
    stage_access = guard.access_log[access_start:]
    assert_oof_only_access(stage_access)
    manifest = {
        "artifact": output_path.name,
        "artifact_sha256": sha256_path(output_path),
        "folds": fold_reports,
        "fold_count": len(fold_reports),
        "oof_rows": len(records),
        "eligible_effective_train_rows": len(text_data),
        "coverage": len(records) / len(text_data),
        "missing_oof_rows": len(text_data) - len(records),
        "first_oof_date": records[0]["decision_date"].isoformat(),
        "last_oof_date": records[-1]["decision_date"].isoformat(),
        "target_classes": sorted({row["target"] for row in records}),
        "embargo_verified": True,
        "strictly_earlier_models_verified": True,
        "validation_access_count": sum(
            item["split"].startswith("validation") for item in stage_access
        ),
        "test_access_count": sum(item["split"] == "test" for item in stage_access),
        "base_models": {
            "text": f"fresh_{selected_text_model}",
            "graph": "fresh_dense_signed_gat",
        },
        "fixed_fold_epochs": fixed_epochs,
    }
    if manifest["validation_access_count"] or manifest["test_access_count"]:
        raise ValueError("OOF generation accessed validation or test")
    atomic_json(config.model_dir / "oof_manifest.json", manifest)
    return manifest


def assert_oof_only_access(access_log: list[dict[str, str]]) -> None:
    forbidden = [
        item for item in access_log
        if item["split"] == "test" or item["split"].startswith("validation")
    ]
    if forbidden:
        raise ValueError(f"OOF generation accessed validation or test: {forbidden}")


def generate_validation_base_predictions(
    config: TrainingConfig,
    splits: EffectiveSplits,
    *,
    guard: SplitAccessGuard | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    guard = guard or SplitAccessGuard()
    validation_dates = splits.validation_selection + splits.validation_calibration
    text_data = load_supervised_text_dataset(
        config,
        splits,
        guard=guard,
        allowed_splits=("validation",),
        allowed_dates=validation_dates,
        purpose="model_selection",
    )
    graph_raw = load_graph_dataset(
        config,
        splits,
        dates=validation_dates,
        purpose="model_selection",
        guard=guard,
    )
    text_scaler = TrainOnlyScaler.load(config.model_dir / "text_scaler.json")
    graph_scaler = TrainOnlyScaler.load(config.model_dir / "graph_scaler.json")
    text_features = text_scaler.transform(text_data.embeddings)
    text_logits, text_probability, selected_text = selected_text_predictions(
        config.model_dir, text_features, device=device
    )
    graph = scale_graph_dataset(graph_raw, graph_scaler)
    graph_logits, graph_probability, _ = predict_gat(
        load_gat(config.model_dir / "graph_model.pt", device=device),
        graph,
        device=device,
    )
    graph_lookup: dict[tuple[Any, str], tuple[str, float, float]] = {}
    for day_index, date in enumerate(graph.dates):
        for node, symbol in enumerate(graph.symbols):
            graph_lookup[(date, symbol)] = (
                str(graph.decision_ids[day_index, node]),
                float(graph_logits[day_index, node]),
                float(graph_probability[day_index, node]),
            )
    selection_dates = set(splits.validation_selection)
    records = []
    for index in range(len(text_data)):
        date = text_data.dates[index]
        symbol = str(text_data.symbols[index])
        graph_id, graph_logit, graph_prob = graph_lookup[(date, symbol)]
        decision_id = str(text_data.decision_ids[index])
        if graph_id != decision_id:
            raise ValueError(f"Validation graph/text decision mismatch for {decision_id}")
        records.append(
            {
                "decision_id": decision_id,
                "decision_date": date,
                "symbol": symbol,
                "partition": (
                    "validation_selection"
                    if date in selection_dates
                    else "validation_calibration"
                ),
                "text_logit": float(text_logits[index]),
                "text_probability": float(text_probability[index]),
                "graph_logit": graph_logit,
                "graph_probability": graph_prob,
                "text_available": True,
                "graph_available": True,
                "target": int(text_data.targets[index]),
            }
        )
    schema = pa.schema(
        [
            ("decision_id", pa.string()),
            ("decision_date", pa.date32()),
            ("symbol", pa.string()),
            ("partition", pa.string()),
            ("text_logit", pa.float32()),
            ("text_probability", pa.float32()),
            ("graph_logit", pa.float32()),
            ("graph_probability", pa.float32()),
            ("text_available", pa.bool_()),
            ("graph_available", pa.bool_()),
            ("target", pa.int8()),
        ]
    )
    output_path = config.model_dir / "validation_base_predictions.parquet"
    _write_parquet(output_path, records, schema)
    return {
        "artifact": output_path.name,
        "artifact_sha256": sha256_path(output_path),
        "selected_text_model": selected_text,
        "validation_selection_rows": sum(
            row["partition"] == "validation_selection" for row in records
        ),
        "validation_calibration_rows": sum(
            row["partition"] == "validation_calibration" for row in records
        ),
    }
