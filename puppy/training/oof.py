from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint import TrainOnlyScaler, atomic_json
from .config import TrainingConfig
from .graph_dataset_loader import GraphDataset, load_graph_dataset
from .graph_trainer import fit_gat, fit_graph_scaler, load_gat, predict_gat, scale_graph_dataset
from .split import (
    EffectiveSplits,
    OOFFold,
    build_inner_chronological_split,
    build_oof_folds,
)
from .text_dataset import TextDataset, load_supervised_text_dataset
from .text_embedding import sha256_path
from .text_trainer import fit_mlp, predict_mlp, selected_text_predictions
from .training_guard import SplitAccessGuard


MINIMUM_HISTORY_CANDIDATES = (
    (0, 0),
    (20, 60),
    (40, 120),
    (60, 180),
    (80, 240),
    (100, 300),
)


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
    training_policy: str,
    embargo_days: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    reports = []
    for fold in folds:
        train = text_data.subset(fold.train_dates)
        prediction = text_data.subset(fold.prediction_dates)
        if not len(train) or not len(prediction):
            raise ValueError(f"OOF fold {fold.fold_id} has empty Text data")
        inner_report: dict[str, Any] | None = None
        selected_epochs = fixed_epochs
        if training_policy == "inner_chronological_early_stop":
            inner = build_inner_chronological_split(
                fold.train_dates,
                embargo_days=embargo_days,
            )
            inner_train = text_data.subset(inner.train_dates)
            inner_validation = text_data.subset(inner.validation_dates)
            if not len(inner_train) or not len(inner_validation):
                raise ValueError(
                    f"OOF fold {fold.fold_id} has an empty inner Text segment"
                )
            _, inner_train_x, inner_validation_x = _scale_text(
                inner_train, inner_validation, manifest_hash
            )
            _, tuning = fit_mlp(
                inner_train_x,
                inner_train.targets,
                inner_validation_x,
                inner_validation.targets,
                config.text_model,
                seed=seed + fold.fold_id * 2,
                device=device,
            )
            selected_epochs = int(tuning["best_epoch"])
            inner_report = {
                "train_start": inner.train_dates[0].isoformat(),
                "train_end": inner.train_dates[-1].isoformat(),
                "train_rows": len(inner_train),
                "embargo_start": inner.embargo_dates[0].isoformat(),
                "embargo_end": inner.embargo_dates[-1].isoformat(),
                "embargo_trading_days": len(inner.embargo_dates),
                "validation_start": inner.validation_dates[0].isoformat(),
                "validation_end": inner.validation_dates[-1].isoformat(),
                "validation_rows": len(inner_validation),
                "selected_epochs": selected_epochs,
            }
        scaler, train_x, prediction_x = _scale_text(train, prediction, manifest_hash)
        model, history = fit_mlp(
            train_x,
            train.targets,
            train_x,
            train.targets,
            config.text_model,
            seed=seed + fold.fold_id * 2,
            device=device,
            fixed_epochs=selected_epochs,
        )
        if inner_report is not None and not inner.validation_dates[-1] < fold.prediction_dates[0]:
            raise ValueError(
                f"Inner Text validation overlaps outer prediction fold {fold.fold_id}"
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
                "train_rows": len(train),
                "training_policy": training_policy,
                "epochs": selected_epochs,
                "inner": inner_report,
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


def minimum_history_coverage_table(
    records: list[dict[str, Any]],
    graph_reports: list[dict[str, Any]],
    symbols: tuple[str, ...],
) -> list[dict[str, Any]]:
    complete = {
        int(report["fold_id"]): report
        for report in graph_reports
        if report.get("status") == "complete"
    }
    baseline_by_symbol = {
        symbol: sum(str(row["symbol"]) == symbol for row in records)
        for symbol in symbols
    }
    table = []
    for minimum_days, minimum_rows in MINIMUM_HISTORY_CANDIDATES:
        retained = sorted(
            fold_id
            for fold_id, report in complete.items()
            if int(report["graph_train_days"]) >= minimum_days
            and int(report["graph_train_rows"]) >= minimum_rows
        )
        selected = [row for row in records if int(row["fold_id"]) in retained]
        symbol_coverage = {}
        for symbol in symbols:
            count = sum(str(row["symbol"]) == symbol for row in selected)
            denominator = baseline_by_symbol[symbol]
            symbol_coverage[symbol] = {
                "rows": count,
                "fraction_of_available_oof": (
                    count / denominator if denominator else 0.0
                ),
            }
        table.append(
            {
                "minimum_graph_train_days": minimum_days,
                "minimum_graph_train_rows": minimum_rows,
                "folds_retained": retained,
                "oof_days_retained": len(
                    {row["decision_date"] for row in selected}
                ),
                "oof_supervised_rows_retained": len(selected),
                "per_symbol_coverage": symbol_coverage,
                "fusion_rows_remaining": len(selected),
            }
        )
    return table


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
    training_policy = config.oof_training_policy
    embargo_days = int(config.split["embargo_trading_days"])
    text_oof, text_reports = _generate_text_oof(
        config,
        folds,
        text_data,
        seed=seed,
        device=device,
        fixed_epochs=fixed_epochs,
        manifest_hash=manifest_hash,
        training_policy=training_policy,
        embargo_days=embargo_days,
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
        text_scores_by_decision=(
            text_scores if config.graph_includes_text_score else None
        ),
    )
    date_positions = {date: index for index, date in enumerate(splits.train_effective)}
    records: list[dict[str, Any]] = []
    graph_reports: list[dict[str, Any]] = []
    for fold in folds:
        graph_train_dates = (
            _complete_oof_graph_train_dates(graph_data, fold.train_dates)
            if config.graph_includes_text_score
            else fold.train_dates
        )
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
        selected_epochs = fixed_epochs
        inner_report: dict[str, Any] | None = None
        if training_policy == "inner_chronological_early_stop":
            try:
                inner = build_inner_chronological_split(
                    graph_train_dates,
                    embargo_days=embargo_days,
                )
            except ValueError as error:
                graph_reports.append(
                    {
                        "fold_id": fold.fold_id,
                        "status": "skipped_insufficient_inner_history",
                        "graph_train_days": len(graph_train_dates),
                        "reason": str(error),
                    }
                )
                continue
            inner_train_raw = graph_data.subset(inner.train_dates)
            inner_validation_raw = graph_data.subset(inner.validation_dates)
            inner_targets = inner_train_raw.targets[inner_train_raw.label_mask]
            if len(np.unique(inner_targets)) < 2:
                graph_reports.append(
                    {
                        "fold_id": fold.fold_id,
                        "status": "skipped_single_class_inner_graph_train",
                        "graph_train_days": len(graph_train_dates),
                    }
                )
                continue
            inner_scaler = fit_graph_scaler(
                inner_train_raw, input_manifest_hash=manifest_hash
            )
            inner_train = scale_graph_dataset(inner_train_raw, inner_scaler)
            inner_validation = scale_graph_dataset(
                inner_validation_raw, inner_scaler
            )
            _, tuning = fit_gat(
                inner_train,
                inner_validation,
                config.graph_model,
                seed=seed + fold.fold_id * 2 + 1,
                device=device,
            )
            selected_epochs = int(tuning["best_epoch"])
            inner_report = {
                "train_start": inner.train_dates[0].isoformat(),
                "train_end": inner.train_dates[-1].isoformat(),
                "train_days": len(inner.train_dates),
                "train_rows": inner_train.supervised_count,
                "embargo_start": inner.embargo_dates[0].isoformat(),
                "embargo_end": inner.embargo_dates[-1].isoformat(),
                "embargo_trading_days": len(inner.embargo_dates),
                "validation_start": inner.validation_dates[0].isoformat(),
                "validation_end": inner.validation_dates[-1].isoformat(),
                "validation_days": len(inner.validation_dates),
                "validation_rows": inner_validation.supervised_count,
                "selected_epochs": selected_epochs,
            }
        scaler = fit_graph_scaler(train_raw, input_manifest_hash=manifest_hash)
        train = scale_graph_dataset(train_raw, scaler)
        prediction = scale_graph_dataset(prediction_raw, scaler)
        model, history = fit_gat(
            train,
            train,
            config.graph_model,
            seed=seed + fold.fold_id * 2 + 1,
            device=device,
            fixed_epochs=selected_epochs,
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
        if inner_report is not None and not inner.validation_dates[-1] < prediction_start:
            raise ValueError(
                f"Inner Graph validation overlaps outer prediction fold {fold.fold_id}"
            )
        if date_positions[prediction_start] - date_positions[model_train_end] - 1 < int(
            config.split["embargo_trading_days"]
        ):
            raise ValueError(f"Nested OOF graph embargo violation in fold {fold.fold_id}")
        for day, date in enumerate(prediction.dates):
            for node, symbol in enumerate(prediction.symbols):
                decision_id = str(prediction.decision_ids[day, node])
                if not prediction.label_mask[day, node]:
                    continue
                text_record = text_oof.get(decision_id)
                if text_record is None:
                    raise ValueError(
                        "Supervised Graph OOF node is missing its Text OOF record: "
                        f"{decision_id}"
                    )
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
                        "graph_text_score_is_oof": config.graph_includes_text_score,
                        "graph_text_score_used": config.graph_includes_text_score,
                        "is_oof": True,
                    }
                )
        graph_reports.append(
            {
                "fold_id": fold.fold_id,
                "status": "complete",
                "model_train_start": graph_train_dates[0].isoformat(),
                "model_train_end": model_train_end.isoformat(),
                "graph_train_days": len(graph_train_dates),
                "graph_train_rows": train.supervised_count,
                "train_days_with_text_oof": (
                    len(graph_train_dates) if config.graph_includes_text_score else 0
                ),
                "prediction_start": prediction_start.isoformat(),
                "prediction_end": fold.prediction_dates[-1].isoformat(),
                "prediction_days": len(prediction.dates),
                "prediction_rows": int(prediction.label_mask.sum()),
                "parameters_changed": history["parameter_changed"],
                "training_policy": training_policy,
                "epochs": selected_epochs,
                "inner": inner_report,
                "text_score_source": (
                    "oof_by_decision_id"
                    if config.graph_includes_text_score
                    else "disabled"
                ),
            }
        )
    records.sort(key=lambda row: (row["decision_date"], row["symbol"], row["decision_id"]))
    record_ids = [str(row["decision_id"]) for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("OOF predictions contain duplicate decision IDs")
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
            ("graph_text_score_used", pa.bool_()),
            ("is_oof", pa.bool_()),
        ]
    )
    output_path = config.model_dir / "oof_predictions.parquet"
    _write_parquet(output_path, records, schema)
    stage_access = guard.access_log[access_start:]
    assert_oof_only_access(stage_access)
    history_threshold_table = minimum_history_coverage_table(
        records, graph_reports, graph_data.symbols
    )
    manifest = {
        "artifact": output_path.name,
        "artifact_sha256": sha256_path(output_path),
        "text_folds": text_reports,
        "graph_folds": graph_reports,
        "training_policy": training_policy,
        "graph_topology": config.graph_topology,
        "graph_include_text_score": config.graph_includes_text_score,
        "ablation_fingerprint": config.ablation_fingerprint,
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
        "graph_text_score_used": config.graph_includes_text_score,
        "minimum_history_threshold_applied": False,
        "minimum_history_coverage_table": history_threshold_table,
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
        "fixed_fold_epochs": (
            fixed_epochs if training_policy == "fixed_epochs" else None
        ),
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
        text_scores_by_decision=(
            text_scores if config.graph_includes_text_score else None
        ),
        require_oof_text_scores=config.graph_includes_text_score,
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
        "graph_topology": config.graph_topology,
        "graph_include_text_score": config.graph_includes_text_score,
        "graph_feature_names": list(graph.feature_names),
        "ablation_fingerprint": config.ablation_fingerprint,
    }
