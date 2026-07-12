from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint import TrainOnlyScaler, atomic_json
from .config import TrainingConfig
from .graph_dataset_loader import GRAPH_FEATURE_NAMES, GraphDataset, load_graph_dataset
from .graph_trainer import fit_gat, fit_graph_scaler, load_gat, predict_gat, scale_graph_dataset
from .split import EffectiveSplits, OOFFold, build_oof_folds
from .text_dataset import TextDataset, load_supervised_text_dataset
from .text_embedding import sha256_path
from .text_trainer import fit_mlp, predict_mlp, selected_text_predictions
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


def _generate_text_oof(
    config: TrainingConfig,
    folds: tuple[OOFFold, ...],
    text_data: TextDataset,
    *,
    seed: int,
    device: str,
    fixed_epochs: int,
    manifest_hash: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    reports = []
    for fold in folds:
        train = text_data.subset(fold.train_dates)
        prediction = text_data.subset(fold.prediction_dates)
        if not len(train) or not len(prediction):
            raise ValueError(f"OOF fold {fold.fold_id} has empty Text data")
        scaler, train_x, prediction_x = _scale_text(train, prediction, manifest_hash)
        model, history = fit_mlp(
            train_x,
            train.targets,
            train_x,
            train.targets,
            config.text_model,
            seed=seed + fold.fold_id * 2,
            device=device,
            fixed_epochs=fixed_epochs,
        )
        logits, probabilities = predict_mlp(model, prediction_x, device=device)
        if not history["parameter_changed"]:
            raise RuntimeError(f"OOF TextScoreHead did not train in fold {fold.fold_id}")
        for index in range(len(prediction)):
            decision_id = str(prediction.decision_ids[index])
            if decision_id in records:
                raise ValueError(f"Duplicate Text OOF prediction for {decision_id}")
            records[decision_id] = {
                "decision_id": decision_id,
                "decision_date": prediction.dates[index],
                "symbol": str(prediction.symbols[index]),
                "fold_id": fold.fold_id,
                "text_model_train_start": fold.train_dates[0],
                "text_model_train_end": fold.train_dates[-1],
                "prediction_start": fold.prediction_dates[0],
                "prediction_end": fold.prediction_dates[-1],
                "embargo_trading_days": len(fold.embargo_dates),
                "h_text": prediction.embeddings[index].astype(np.float32).tolist(),
                "text_logit": float(logits[index]),
                "text_probability": float(probabilities[index]),
                "target": int(prediction.targets[index]),
                "text_prediction_is_oof": True,
            }
        reports.append(
            {
                "fold_id": fold.fold_id,
                "model_train_start": fold.train_dates[0].isoformat(),
                "model_train_end": fold.train_dates[-1].isoformat(),
                "embargo_start": fold.embargo_dates[0].isoformat(),
                "embargo_end": fold.embargo_dates[-1].isoformat(),
                "embargo_trading_days": len(fold.embargo_dates),
                "prediction_start": fold.prediction_dates[0].isoformat(),
                "prediction_end": fold.prediction_dates[-1].isoformat(),
                "prediction_rows": len(prediction),
                "scaler_fit_start": scaler.metadata["fit_start_date"],
                "scaler_fit_end": scaler.metadata["fit_end_date"],
                "parameters_changed": history["parameter_changed"],
                "model": "canonical_text_score_head",
            }
        )
    return records, reports


def _complete_oof_graph_train_dates(
    dataset: GraphDataset,
    dates: tuple[Any, ...],
) -> tuple[Any, ...]:
    text_index = dataset.feature_names.index("text_score")
    result = []
    for date in dates:
        matches = np.flatnonzero(dataset.dates == date)
        if len(matches) != 1:
            continue
        day = int(matches[0])
        supervised = dataset.label_mask[day]
        if bool(supervised.any()) and bool(
            dataset.feature_mask[day, :, text_index][supervised].all()
        ):
            result.append(date)
    return tuple(result)


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
    folds = build_oof_folds(
        splits.train_effective,
        folds=int(config.split["oof_folds"]),
        min_train_days=int(config.split["oof_min_train_days"]),
        embargo_days=int(config.split["embargo_trading_days"]),
    )
    manifest_hash = sha256_path(config.artifact_dir / "data_manifest.json")
    fixed_epochs = min(
        30,
        int(config.text_model["max_epochs"]),
        int(config.graph_model["max_epochs"]),
    )
    text_oof, text_reports = _generate_text_oof(
        config,
        folds,
        text_data,
        seed=seed,
        device=device,
        fixed_epochs=fixed_epochs,
        manifest_hash=manifest_hash,
    )
    text_scores = {
        decision_id: float(record["text_probability"])
        for decision_id, record in text_oof.items()
    }
    graph_data = load_graph_dataset(
        config,
        splits,
        dates=splits.train_effective,
        purpose="oof",
        guard=guard,
        text_scores_by_decision=text_scores,
    )
    date_positions = {date: index for index, date in enumerate(splits.train_effective)}
    records: list[dict[str, Any]] = []
    graph_reports: list[dict[str, Any]] = []
    for fold in folds:
        graph_train_dates = _complete_oof_graph_train_dates(graph_data, fold.train_dates)
        if not graph_train_dates:
            graph_reports.append(
                {"fold_id": fold.fold_id, "status": "skipped_no_prior_text_oof"}
            )
            continue
        train_raw = graph_data.subset(graph_train_dates)
        prediction_raw = graph_data.subset(fold.prediction_dates)
        if not len(prediction_raw):
            continue
        targets = train_raw.targets[train_raw.label_mask]
        if len(np.unique(targets)) < 2:
            graph_reports.append(
                {"fold_id": fold.fold_id, "status": "skipped_single_class_graph_train"}
            )
            continue
        scaler = fit_graph_scaler(train_raw, input_manifest_hash=manifest_hash)
        train = scale_graph_dataset(train_raw, scaler)
        prediction = scale_graph_dataset(prediction_raw, scaler)
        model, history = fit_gat(
            train,
            train,
            config.graph_model,
            seed=seed + fold.fold_id * 2 + 1,
            device=device,
            fixed_epochs=fixed_epochs,
        )
        graph_logits, graph_probabilities, alpha, h_graph = predict_gat(
            model,
            prediction,
            device=device,
            include_attention=True,
            include_latent=True,
        )
        if not history["parameter_changed"]:
            raise RuntimeError(f"OOF GAT did not train in fold {fold.fold_id}")
        model_train_end = graph_train_dates[-1]
        prediction_start = fold.prediction_dates[0]
        if date_positions[prediction_start] - date_positions[model_train_end] - 1 < int(
            config.split["embargo_trading_days"]
        ):
            raise ValueError(f"Nested OOF graph embargo violation in fold {fold.fold_id}")
        for day, date in enumerate(prediction.dates):
            for node, symbol in enumerate(prediction.symbols):
                decision_id = str(prediction.decision_ids[day, node])
                text_record = text_oof.get(decision_id)
                if text_record is None or not prediction.label_mask[day, node]:
                    continue
                records.append(
                    {
                        **text_record,
                        "model_train_start": graph_train_dates[0],
                        "model_train_end": model_train_end,
                        "graph_model_train_start": graph_train_dates[0],
                        "graph_model_train_end": model_train_end,
                        "graph_logit": float(graph_logits[day, node]),
                        "graph_probability": float(graph_probabilities[day, node]),
                        "h_graph": h_graph[day, node].astype(np.float32).tolist(),
                        "alpha": alpha[day].astype(np.float32).tolist(),
                        "graph_representation_is_oof": True,
                        "graph_text_score_is_oof": True,
                        "is_oof": True,
                    }
                )
        graph_reports.append(
            {
                "fold_id": fold.fold_id,
                "status": "complete",
                "model_train_start": graph_train_dates[0].isoformat(),
                "model_train_end": model_train_end.isoformat(),
                "train_days_with_text_oof": len(graph_train_dates),
                "prediction_start": prediction_start.isoformat(),
                "prediction_end": fold.prediction_dates[-1].isoformat(),
                "parameters_changed": history["parameter_changed"],
                "text_score_source": "oof_by_decision_id",
            }
        )
    records.sort(key=lambda row: (row["decision_date"], row["symbol"], row["decision_id"]))
    if len(records) < 30 or len({row["target"] for row in records}) < 2:
        raise ValueError("Canonical graph OOF coverage is too small or single-class")
    graph_dim = len(records[0]["h_graph"])
    nodes = len(records[0]["alpha"])
    schema = pa.schema(
        [
            ("decision_id", pa.string()),
            ("decision_date", pa.date32()),
            ("symbol", pa.string()),
            ("fold_id", pa.int16()),
            ("text_model_train_start", pa.date32()),
            ("text_model_train_end", pa.date32()),
            ("graph_model_train_start", pa.date32()),
            ("graph_model_train_end", pa.date32()),
            ("model_train_start", pa.date32()),
            ("model_train_end", pa.date32()),
            ("prediction_start", pa.date32()),
            ("prediction_end", pa.date32()),
            ("embargo_trading_days", pa.int16()),
            ("h_text", pa.list_(pa.float32(), 384)),
            ("text_logit", pa.float32()),
            ("text_probability", pa.float32()),
            ("h_graph", pa.list_(pa.float32(), graph_dim)),
            ("alpha", pa.list_(pa.list_(pa.float32(), nodes), nodes)),
            ("graph_logit", pa.float32()),
            ("graph_probability", pa.float32()),
            ("target", pa.int8()),
            ("text_prediction_is_oof", pa.bool_()),
            ("graph_representation_is_oof", pa.bool_()),
            ("graph_text_score_is_oof", pa.bool_()),
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
        "text_folds": text_reports,
        "graph_folds": graph_reports,
        "fold_count": len(folds),
        "oof_rows": len(records),
        "eligible_effective_train_rows": len(text_data),
        "coverage": len(records) / len(text_data),
        "missing_oof_rows": len(text_data) - len(records),
        "first_oof_date": records[0]["decision_date"].isoformat(),
        "last_oof_date": records[-1]["decision_date"].isoformat(),
        "target_classes": sorted({row["target"] for row in records}),
        "embargo_verified": True,
        "strictly_earlier_models_verified": True,
        "oof_text_for_graph_verified": True,
        "oof_graph_latents_verified": True,
        "validation_access_count": sum(
            item["split"].startswith("validation") for item in stage_access
        ),
        "test_access_count": sum(item["split"] == "test" for item in stage_access),
        "base_models": {
            "text": "fresh_canonical_text_score_head",
            "graph": "fresh_canonical_dense_signed_gat",
        },
        "h_text_dim": 384,
        "h_graph_dim": graph_dim,
        "fixed_fold_epochs": fixed_epochs,
    }
    if manifest["validation_access_count"] or manifest["test_access_count"]:
        raise ValueError("OOF generation accessed validation or test")
    atomic_json(config.model_dir / "oof_manifest.json", manifest)
    return manifest


def assert_oof_only_access(access_log: list[dict[str, str]]) -> None:
    forbidden = [
        item
        for item in access_log
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
    text_scaler = TrainOnlyScaler.load(config.model_dir / "text_scaler.json")
    text_features = text_scaler.transform(text_data.embeddings)
    text_logits, text_probability, selected_text = selected_text_predictions(
        config.model_dir, text_features, device=device
    )
    text_scores = {
        str(decision_id): float(probability)
        for decision_id, probability in zip(text_data.decision_ids, text_probability)
    }
    graph_raw = load_graph_dataset(
        config,
        splits,
        dates=validation_dates,
        purpose="model_selection",
        guard=guard,
        text_scores_by_decision=text_scores,
        require_oof_text_scores=True,
    )
    graph_scaler = TrainOnlyScaler.load(config.model_dir / "graph_scaler.json")
    graph = scale_graph_dataset(graph_raw, graph_scaler)
    graph_logits, graph_probability, alpha, h_graph = predict_gat(
        load_gat(config.model_dir / "graph_model.pt", device=device),
        graph,
        device=device,
        include_attention=True,
        include_latent=True,
    )
    graph_lookup: dict[tuple[Any, str], tuple[str, float, float, list[float], list[list[float]]]] = {}
    for day, date in enumerate(graph.dates):
        for node, symbol in enumerate(graph.symbols):
            graph_lookup[(date, symbol)] = (
                str(graph.decision_ids[day, node]),
                float(graph_logits[day, node]),
                float(graph_probability[day, node]),
                h_graph[day, node].astype(np.float32).tolist(),
                alpha[day].astype(np.float32).tolist(),
            )
    selection_dates = set(splits.validation_selection)
    records = []
    for index in range(len(text_data)):
        date = text_data.dates[index]
        symbol = str(text_data.symbols[index])
        graph_id, graph_logit, graph_prob, graph_latent, attention = graph_lookup[(date, symbol)]
        decision_id = str(text_data.decision_ids[index])
        if graph_id != decision_id:
            raise ValueError(f"Validation graph/text decision mismatch for {decision_id}")
        records.append(
            {
                "decision_id": decision_id,
                "decision_date": date,
                "symbol": symbol,
                "partition": (
                    "validation_selection" if date in selection_dates else "validation_calibration"
                ),
                "h_text": text_data.embeddings[index].astype(np.float32).tolist(),
                "text_logit": float(text_logits[index]),
                "text_probability": float(text_probability[index]),
                "h_graph": graph_latent,
                "alpha": attention,
                "graph_logit": graph_logit,
                "graph_probability": graph_prob,
                "target": int(text_data.targets[index]),
            }
        )
    graph_dim = len(records[0]["h_graph"])
    nodes = len(graph.symbols)
    schema = pa.schema(
        [
            ("decision_id", pa.string()),
            ("decision_date", pa.date32()),
            ("symbol", pa.string()),
            ("partition", pa.string()),
            ("h_text", pa.list_(pa.float32(), 384)),
            ("text_logit", pa.float32()),
            ("text_probability", pa.float32()),
            ("h_graph", pa.list_(pa.float32(), graph_dim)),
            ("alpha", pa.list_(pa.list_(pa.float32(), nodes), nodes)),
            ("graph_logit", pa.float32()),
            ("graph_probability", pa.float32()),
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
        "h_text_dim": 384,
        "h_graph_dim": graph_dim,
    }
