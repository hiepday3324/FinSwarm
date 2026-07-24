from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from demo.allocator import MAX_STOCK_EXPOSURE, MAX_SYMBOL_WEIGHT, allocate
from demo.constants import (
    ATTENTION_CONVENTION,
    DEFAULT_BUNDLE_DIR,
    DEFAULT_CONFIG_PATH,
    DISPLAY_ORDER,
    EXPECTED_ALIGNED_ROWS,
    EXPECTED_SUPERVISED_ROWS,
    EXPECTED_TRAIN_DATES,
    NODE_ORDER,
    REPO_ROOT,
    TRAIN_END,
    TRAIN_START,
)
from demo.data_loader import ReplayStore
from demo.graph_visualization import attention_views, edge_records
from demo.inference import FrozenReplayEngine
from demo.training import load_demo_config, protected_snapshot


UI_EXECUTION_FILES = (
    "demo/app.py",
    "demo/data_loader.py",
    "demo/inference.py",
    "demo/graph_visualization.py",
    "demo/allocator.py",
    "demo/components.py",
    "demo/scenario_builder.py",
)


def _assert_ui_inference_only() -> None:
    forbidden = {
        "optimizer": "optimizer construction/use",
        ".backward(": "backward pass",
        "torch.optim": "optimizer namespace",
        ".optimizer.step(": "optimizer update",
    }
    for relative in UI_EXECUTION_FILES:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8").lower()
        for token, description in forbidden.items():
            if token in source:
                raise AssertionError(f"UI file {relative} contains {description}: {token}")


def _assert_allocation(
    store: ReplayStore,
    engine: FrozenReplayEngine,
    date_index: int,
) -> None:
    output = engine.infer_date(date_index)
    records = {row["symbol"]: row for row in store.records_for_date(output["date"])}
    actions = {symbol: str(records[symbol]["decision"]) for symbol in DISPLAY_ORDER}
    probabilities = {
        symbol: float(output["fusion_probability"][NODE_ORDER.index(symbol)])
        for symbol in DISPLAY_ORDER
    }
    result = allocate(actions, probabilities, capital=1_000_000.0)
    assert result["stock_exposure"] <= MAX_STOCK_EXPOSURE + 1e-10
    assert all(
        value <= MAX_SYMBOL_WEIGHT + 1e-10 for value in result["weights"].values()
    )
    assert abs(float(result["total_weight"]) - 1.0) < 1e-12


def run_smoke_test(
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    store = ReplayStore(bundle_dir)
    config = None if store.manifest.get("mock_bundle") else load_demo_config(config_path)
    engine = FrozenReplayEngine(bundle_dir, device="cpu")
    metadata = store.training_metadata
    manifest = store.manifest

    if config is not None:
        assert config.training.model_dir.resolve() == Path(bundle_dir).resolve()
    assert metadata["demo_only"] is True
    assert metadata["intentionally_in_sample"] is True
    assert metadata["evaluation_mode"] == "train_replay"
    assert metadata["model_frozen_during_ui"] is True
    assert metadata["train_start"] == TRAIN_START
    assert metadata["train_end"] == TRAIN_END
    assert metadata["train_dates"] == EXPECTED_TRAIN_DATES
    assert metadata["aligned_replay_rows"] == EXPECTED_ALIGNED_ROWS
    assert metadata["supervised_train_rows"] == EXPECTED_SUPERVISED_ROWS
    assert metadata["validation_access_count"] == 0
    assert metadata["test_access_count"] == 0
    assert metadata["validation_labels_accessed"] is False
    assert metadata["test_labels_accessed"] is False
    assert metadata["test_features_accessed"] is False
    assert metadata["checkpoint_selection"] == "minimum_train_loss"
    assert metadata["validation_early_stopping"] is False
    assert metadata["calibration_fitted"] is False
    assert metadata["protected_artifacts_unchanged"] is True
    assert metadata["source_access"]["test_feature_access_count"] == 0
    assert all(
        entry["split"] == "train"
        for entry in metadata["source_access"]["guard_access_log"]
    )
    assert not any(
        "validation" in str(entry.get("filter", "")).lower()
        or "test" in str(entry.get("filter", "")).lower()
        for entry in metadata["source_access"]["files"]
    )
    assert metadata["save_load_prediction_equivalence"]["all_passed"] is True

    assert tuple(manifest["node_order"]) == NODE_ORDER
    assert "prior_master_portfolio_features" in store.feature_schema["feature_blocks"]
    assert tuple(store.symbols) == NODE_ORDER
    assert tuple(metadata["node_order"]) == NODE_ORDER
    assert metadata["attention_convention"] == ATTENTION_CONVENTION
    assert metadata["attention_shape"] == [EXPECTED_TRAIN_DATES, 4, 3, 3]
    assert engine.graph_model.num_heads == 4
    assert engine.all_parameters_frozen
    assert engine.state_hashes() == metadata["model_hashes"]

    assert len(store.dates) == EXPECTED_TRAIN_DATES
    assert store.dates[0].isoformat() == TRAIN_START
    assert store.dates[-1].isoformat() == TRAIN_END
    assert tuple(sorted(store.dates)) == store.dates
    flattened_ids = engine.decision_ids.reshape(-1).astype(str).tolist()
    assert len(flattened_ids) == EXPECTED_ALIGNED_ROWS
    assert len(flattened_ids) == len(set(flattened_ids))
    for date_index, date in enumerate(store.dates):
        records = store.records_for_date(date)
        assert tuple(row["symbol"] for row in records) == DISPLAY_ORDER
        for row in records:
            state = row["master_portfolio_state"]
            if state is not None:
                assert str(state["date"]) < date.isoformat()
        expected = store.index["decision_ids_by_date"][date.isoformat()]
        assert engine.decision_ids[date_index].astype(str).tolist() == expected

    attention_heads = engine.expected["attention_heads"]
    attention_mean = engine.expected["attention_mean"]
    assert attention_heads.shape == (EXPECTED_TRAIN_DATES, 4, 3, 3)
    assert attention_mean.shape == (EXPECTED_TRAIN_DATES, 3, 3)
    assert np.allclose(attention_heads.sum(axis=-1), 1.0, atol=1e-6)
    assert np.allclose(attention_mean, attention_heads.mean(axis=1), atol=1e-7)

    scenario_indices = [store.date_index(item["date"]) for item in store.scenarios()]
    probe_indices = sorted(set([0, len(store.dates) // 2, len(store.dates) - 1, *scenario_indices]))
    prediction_deltas = {
        str(index): engine.assert_expected(index) for index in probe_indices
    }
    before_hashes = engine.state_hashes()
    output = engine.infer_date(probe_indices[-1])
    after_hashes = engine.state_hashes()
    assert before_hashes == after_hashes == engine.startup_state_hashes

    previous_index = max(0, probe_indices[-1] - 1)
    previous = engine.infer_date(previous_index)
    matrix = output["attention_mean"]
    previous_matrix = previous["attention_mean"]
    records = edge_records(
        matrix,
        previous_matrix,
        NODE_ORDER,
        head_label="Mean across heads",
        replay_date=output["date"],
        previous_date=previous["date"],
    )
    assert len(records) == 9
    for item in records:
        target = NODE_ORDER.index(item["target"])
        source = NODE_ORDER.index(item["source"])
        assert np.isclose(item["current_alpha"], matrix[target, source])
        assert np.isclose(item["previous_alpha"], previous_matrix[target, source])
    heatmap_previous, heatmap_current, heatmap_delta = attention_views(
        previous_matrix, matrix
    )
    assert np.array_equal(heatmap_previous, previous_matrix)
    assert np.array_equal(heatmap_current, matrix)
    assert np.allclose(heatmap_delta, matrix - previous_matrix)

    decomposition = engine.fusion_decomposition(output)
    for row in decomposition:
        assert np.isclose(
            row["fusion_logit"], row["reconstructed_logit"], atol=1e-6
        )
    for index in probe_indices:
        _assert_allocation(store, engine, index)

    _assert_ui_inference_only()
    current_protected = protected_snapshot()
    assert current_protected == metadata["protected_artifact_snapshot_after"]
    scenarios = store.scenarios()
    assert len(scenarios) == 3
    assert {item["central_symbol"] for item in scenarios} == set(DISPLAY_ORDER)
    assert len({item["date"] for item in scenarios}) == 3
    for item in scenarios:
        assert TRAIN_START <= item["date"] <= TRAIN_END
        assert "outcomes" in item["selection_excluded_fields"]
        assert "validation metrics" in item["selection_excluded_fields"]
        assert "test metrics" in item["selection_excluded_fields"]

    checkpoint_files = ("text_model.pt", "graph_model.pt", "fusion_model.pt")
    assert all((Path(bundle_dir) / name).is_file() for name in checkpoint_files)
    return {
        "status": "PASS",
        "bundle": str(Path(bundle_dir).resolve()),
        "train_range": [TRAIN_START, TRAIN_END],
        "train_dates": len(store.dates),
        "aligned_replay_rows": len(flattened_ids),
        "supervised_train_rows": metadata["supervised_train_rows"],
        "validation_access_count": metadata["validation_access_count"],
        "test_access_count": metadata["test_access_count"],
        "node_order": list(NODE_ORDER),
        "attention_shape": list(attention_heads.shape),
        "attention_normalization_axis": -1,
        "source_target_convention": "alpha[target, source] = source -> target",
        "model_hashes_unchanged": before_hashes == after_hashes,
        "save_load_max_abs_delta": {
            name: max(values[name] for values in prediction_deltas.values())
            for name in next(iter(prediction_deltas.values()))
        },
        "scenario_count": len(scenarios),
        "protected_artifacts_unchanged": True,
        "ui_inference_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the frozen E4 replay demo.")
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke_test(args.bundle_dir, args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
