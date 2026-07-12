from __future__ import annotations

import datetime as dt
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from puppy.fusion.gated_fusion import LatentGatedFusion
from puppy.fusion.text_score_head import TextScoreHead
from puppy.graph.gat_model import DenseSignedGAT, NumpyGATScorer
from puppy.training.calibration import fit_temperature
from puppy.training.checkpoint import (
    TrainOnlyScaler,
    atomic_torch_save,
    seed_everything,
    state_dict_sha256,
)
from puppy.training.config import TrainingConfig
from puppy.training.fusion_trainer import (
    _candidate_metrics,
    fit_fusion,
    load_fusion_model,
    predict_fusion,
    validate_oof_rows,
)
from puppy.training.graph_dataset_loader import (
    GRAPH_FEATURE_NAMES,
    POLICY_DEPENDENT_GRAPH_FIELDS,
    GraphDataset,
)
from puppy.training.graph_trainer import (
    fit_gat,
    fit_graph_scaler,
    load_gat,
    predict_gat,
    scale_graph_dataset,
)
from puppy.training.metrics import binary_metrics
from puppy.training.model_bundle import ModelBundle
from puppy.training.oof import assert_oof_only_access
from puppy.training.pipeline import verify_causal_inputs
from puppy.training.split import EffectiveSplits, build_effective_splits, build_oof_folds
from puppy.training.text_dataset import load_supervised_text_dataset
from puppy.training.text_embedding import (
    FrozenSemanticEncoder,
    canonical_memory_text,
    sha256_path,
    validate_semantic_embedding_artifact,
)
from puppy.training.text_trainer import (
    LogisticBaseline,
    fit_mlp,
    load_text_mlp,
    predict_mlp,
)
from puppy.training.training_guard import SplitAccessGuard, TestSplitAccessError


def _config(root: Path) -> TrainingConfig:
    artifact = root / "data" / "training" / "finswarm_v1"
    model = root / "data" / "model_artifacts" / "finswarm_v1"
    artifact.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    config_path = root / "config" / "swarm" / "training_v1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("run: {run_id: finswarm_v1, seed: 42}\n", encoding="utf-8")
    experiment = root / "config" / "swarm" / "experiment_v1.yaml"
    experiment.write_text("assets: {symbols: [TSLA, NVDA, AAPL]}\n", encoding="utf-8")
    split_manifest = root / "split.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    raw = {
        "run": {"run_id": "finswarm_v1", "seed": 42},
        "text_embedding": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "pooling": "mean",
            "normalize": True,
            "max_length": 512,
            "batch_size": 4,
            "text_view_version": "full-causal-agent-context-v1",
        },
        "split": {
            "embargo_trading_days": 20,
            "validation_selection_fraction": 0.6666667,
            "oof_folds": 5,
            "oof_min_train_days": 60,
        },
        "labels": {},
        "text_model": {},
        "graph_model": {},
        "fusion_model": {},
        "calibration": {"method": "temperature"},
    }
    return TrainingConfig(
        config_path=config_path,
        repo_root=root,
        run_id="finswarm_v1",
        seed=42,
        experiment_config=experiment,
        artifact_dir=artifact,
        split_manifest=split_manifest,
        model_dir=model,
        text_embedding=raw["text_embedding"],
        split=raw["split"],
        labels=raw["labels"],
        text_model=raw["text_model"],
        graph_model=raw["graph_model"],
        fusion_model=raw["fusion_model"],
        calibration=raw["calibration"],
        raw=raw,
    )


class _FakeTokenizer:
    def __call__(self, texts, **_: object):
        encoded = [[(ord(char) % 19) + 1 for char in text[:12]] or [1] for text in texts]
        width = max(len(row) for row in encoded)
        ids = [row + [0] * (width - len(row)) for row in encoded]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384, _commit_hash="fake-revision")

    def forward(self, input_ids, attention_mask):
        offsets = torch.arange(384, device=input_ids.device, dtype=torch.float32) / 1000
        hidden = torch.sin(input_ids.float().unsqueeze(-1) / 10 + offsets)
        return SimpleNamespace(last_hidden_state=hidden)


def _pending() -> dict[str, object]:
    context = {
        "context_hash": "context-hash",
        "symbol": "TSLA",
        "date": "2025-01-02",
        "short_memory": ["short alpha"],
        "mid_memory": ["mid beta"],
        "long_memory": ["long gamma"],
        "latest_reason": "FORBIDDEN_REASON",
        "reflection_memory": ["FORBIDDEN_REFLECTION"],
        "shadow_state": {"secret": "FORBIDDEN_SHADOW"},
        "shared_context": ["FORBIDDEN_SHARED"],
        "psychology_state": {"state": "fear"},
    }
    return {
        "decision_id": "decision",
        "symbol": "TSLA",
        "decision_date": dt.date(2025, 1, 2),
        "context_hash": "context-hash",
        "context_snapshot": {"agent_context": context},
        "reason": "FORBIDDEN_CURRENT_DECISION_REASON",
    }


def _semantic_encoder() -> FrozenSemanticEncoder:
    return FrozenSemanticEncoder(
        "fake",
        tokenizer=_FakeTokenizer(),
        model=_FakeTransformer(),
        normalize=True,
    )


class Stage6TextEmbeddingTest(unittest.TestCase):
    def test_bootstrap_16_dimensional_hash_embedding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "embedding_dim": 384,
                        "encoder_model": "bootstrap-sha256-frozen",
                        "encoder_version": "sha256-frozen-encoder-v1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_semantic_embedding_artifact(Path(directory) / "missing.parquet", manifest)

    def test_semantic_view_excludes_current_decision_reason(self) -> None:
        text, _ = canonical_memory_text(_pending())
        self.assertNotIn("FORBIDDEN_CURRENT_DECISION_REASON", text)
        self.assertIn("FORBIDDEN_REASON", text)

    def test_semantic_view_includes_eligible_reflection_memory(self) -> None:
        text, _ = canonical_memory_text(_pending())
        self.assertIn("FORBIDDEN_REFLECTION", text)

    def test_semantic_view_includes_prior_shadow_and_psychology(self) -> None:
        text, _ = canonical_memory_text(_pending())
        self.assertIn("FORBIDDEN_SHADOW", text)
        self.assertIn('"state": "fear"', text)

    def test_semantic_embedding_dimension_is_384(self) -> None:
        self.assertEqual(_semantic_encoder().encode(["alpha"]).shape, (1, 384))

    def test_semantic_embedding_is_deterministic_in_eval_mode(self) -> None:
        encoder = _semantic_encoder()
        self.assertTrue(np.array_equal(encoder.encode(["alpha"]), encoder.encode(["alpha"])))

    def test_encoder_batch_and_single_results_match(self) -> None:
        encoder = _semantic_encoder()
        batch = encoder.encode(["alpha", "longer beta"], batch_size=2)
        single = np.concatenate(
            [encoder.encode(["alpha"], batch_size=1), encoder.encode(["longer beta"], batch_size=1)]
        )
        self.assertTrue(np.allclose(batch, single, atol=1e-6))

    def test_text_view_hash_changes_when_allowed_memory_changes(self) -> None:
        first = _pending()
        second = _pending()
        second["context_snapshot"]["agent_context"]["short_memory"] = ["different"]
        self.assertNotEqual(canonical_memory_text(first)[1], canonical_memory_text(second)[1])


class Stage6DatasetJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = _config(Path(self.temporary.name))
        dates = [dt.date(2025, 1, day) for day in range(1, 8)]
        self.splits = EffectiveSplits(
            warmup=(dates[0],),
            train_manifest=(dates[1], dates[2], dates[3]),
            train_effective=(dates[1], dates[2], dates[3]),
            embargoed_train=(),
            validation_selection=(dates[4],),
            validation_calibration=(dates[5],),
            test_metadata_dates=(dates[6],),
        )
        specs = [
            ("warmup", dates[0], "warmup", -1, True, False, True),
            ("train-positive", dates[1], "train", 1, True, False, True),
            ("train-neutral", dates[2], "train", 0, False, False, True),
            ("train-cross", dates[3], "train", -1, True, True, False),
            ("val-negative", dates[4], "validation", -1, True, False, True),
            ("cal-positive", dates[5], "validation", 1, True, False, True),
        ]
        pending = []
        outcomes = []
        embeddings = []
        observed = []
        for decision_id, date, split, label, mask, cross, eligible in specs:
            context_hash = f"hash-{decision_id}"
            pending.append(
                {
                    "decision_id": decision_id,
                    "symbol": "TSLA",
                    "decision_date": date,
                    "context_hash": context_hash,
                    "status": "resolved",
                    "split": split,
                }
            )
            outcomes.append(
                {
                    "decision_id": decision_id,
                    "symbol": "TSLA",
                    "decision_date": date,
                    "direction_label": label,
                    "train_mask": mask,
                    "cross_split_outcome": cross,
                    "supervised_eligible": eligible,
                    "split": split,
                }
            )
            embeddings.append(
                {
                    "decision_id": decision_id,
                    "symbol": "TSLA",
                    "decision_date": date,
                    "source_context_hash": context_hash,
                    "embedding": np.full(384, len(decision_id), dtype=np.float32).tolist(),
                    "embedding_dim": 384,
                    "split": split,
                }
            )
            observed.append(
                {
                    "sample_id": f"sample-{decision_id}",
                    "symbol": "TSLA",
                    "date": date,
                    "context_hash": context_hash,
                    "split": split,
                }
            )
        pq.write_table(pa.Table.from_pylist(list(reversed(pending))), self.config.artifact_dir / "pending_decisions.parquet")
        pq.write_table(pa.Table.from_pylist(outcomes[::2] + outcomes[1::2]), self.config.artifact_dir / "outcomes.parquet")
        embedding_path = self.config.artifact_dir / "text_embeddings_384.parquet"
        pq.write_table(pa.Table.from_pylist(embeddings[1:] + embeddings[:1]), embedding_path)
        pq.write_table(pa.Table.from_pylist(observed), self.config.artifact_dir / "observed_features.parquet")
        (self.config.artifact_dir / "text_embedding_manifest.json").write_text(
            json.dumps(
                {
                    "embedding_dim": 384,
                    "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "encoder_version": "transformers-frozen:fake",
                    "artifact_sha256": sha256_path(embedding_path),
                }
            ),
            encoding="utf-8",
        )
        self.dataset = load_supervised_text_dataset(self.config, self.splits)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dataset_joins_by_decision_id_not_row_order(self) -> None:
        self.assertEqual(
            self.dataset.decision_ids.tolist(),
            ["train-positive", "val-negative", "cal-positive"],
        )

    def test_neutral_targets_are_masked(self) -> None:
        self.assertEqual(self.dataset.metadata["masked_neutral_count"], 1)
        self.assertEqual(set(self.dataset.targets.tolist()), {0.0, 1.0})

    def test_cross_split_outcomes_are_excluded(self) -> None:
        self.assertEqual(self.dataset.metadata["cross_split_excluded_count"], 1)
        self.assertNotIn("train-cross", self.dataset.decision_ids)

    def test_warmup_is_excluded_from_supervised_fitting(self) -> None:
        self.assertEqual(self.dataset.metadata["warmup_excluded_count"], 1)
        self.assertNotIn("warmup", self.dataset.decision_ids)


class Stage6SplitAndScalerTest(unittest.TestCase):
    def _manifest(self, path: Path) -> None:
        start = dt.date(2024, 1, 1)
        groups = {
            "warmup": [start + dt.timedelta(days=index) for index in range(2)],
            "train": [start + dt.timedelta(days=10 + index) for index in range(30)],
            "validation": [start + dt.timedelta(days=50 + index) for index in range(6)],
            "test": [start + dt.timedelta(days=70 + index) for index in range(3)],
        }
        path.write_text(
            json.dumps(
                {
                    "splits": {
                        name: {"dates": [date.isoformat() for date in dates]}
                        for name, dates in groups.items()
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_stage5_temporal_gate_must_be_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            names = (
                "text_samples.parquet", "agent_outputs.parquet", "observed_features.parquet",
                "outcomes.parquet", "pending_decisions.parquet", "reflection_updates.parquet",
                "graph_samples.npz", "feature_schema.json", "data_manifest.json",
            )
            for name in names:
                (config.artifact_dir / name).write_bytes(b"fixture")
            (config.artifact_dir / "temporal_quality.json").write_text('{"status":"FAIL"}', encoding="utf-8")
            (config.artifact_dir / "rollout_manifest.json").write_text(
                '{"status":"complete","temporal_gate":"PASS"}', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                verify_causal_inputs(config)

    def test_train_to_validation_embargo_is_exactly_20_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            self._manifest(path)
            splits = build_effective_splits(path, embargo_trading_days=20, validation_selection_fraction=0.6666667)
            self.assertEqual(len(splits.embargoed_train), 20)
            self.assertEqual(len(splits.train_effective), 10)

    def test_validation_selection_precedes_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            self._manifest(path)
            splits = build_effective_splits(path, embargo_trading_days=20, validation_selection_fraction=0.6666667)
            self.assertLess(max(splits.validation_selection), min(splits.validation_calibration))

    def test_test_split_access_raises(self) -> None:
        with self.assertRaises(TestSplitAccessError):
            SplitAccessGuard().assert_allowed("test", "fit")

    def test_text_scaler_records_train_only_dates(self) -> None:
        dates = [dt.date(2025, 1, 2), dt.date(2025, 1, 3)]
        scaler = TrainOnlyScaler(["a", "b"]).fit(
            np.asarray([[1.0, 4.0], [3.0, 4.0]]), dates=dates, input_manifest_hash="hash"
        )
        self.assertEqual(scaler.metadata["fit_end_date"], "2025-01-03")
        self.assertEqual(scaler.scale[1], 1.0)

    def test_graph_scaler_records_effective_train_only(self) -> None:
        graph = _graph_dataset()
        scaler = fit_graph_scaler(graph.subset(graph.dates[:2]), input_manifest_hash="hash")
        self.assertEqual(scaler.metadata["fit_end_date"], graph.dates[1].isoformat())
        self.assertEqual(scaler.metadata["fit_row_count"], 6)


def _text_config() -> dict[str, object]:
    return {
        "hidden_dim": 8,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 4,
        "max_epochs": 5,
        "patience": 2,
    }


class Stage6TextModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.asarray(
            [[-2, 0, 1, 0], [-1, 0, 0, 1], [-0.5, 1, 0, 0], [-0.2, 0, 1, 1],
             [0.2, 1, 0, 1], [0.5, 0, 1, 0], [1, 1, 0, 0], [2, 0, 0, 1]],
            dtype=np.float32,
        )
        self.y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

    def test_logistic_baseline_save_load_equivalence(self) -> None:
        model = LogisticBaseline.fit(self.x, self.y, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logistic.json"
            model.save(path)
            self.assertTrue(np.allclose(model.predict_proba(self.x), LogisticBaseline.load(path).predict_proba(self.x)))

    def test_mlp_parameters_change_after_backward(self) -> None:
        _, history = fit_mlp(self.x, self.y, self.x, self.y, _text_config(), seed=42, fixed_epochs=2)
        self.assertTrue(history["parameter_changed"])

    def test_mlp_best_validation_checkpoint_is_restored(self) -> None:
        _, history = fit_mlp(self.x, self.y, self.x, self.y, _text_config(), seed=42)
        self.assertTrue(history["best_checkpoint_restored"])

    def test_same_seed_produces_equivalent_text_parameters(self) -> None:
        first, _ = fit_mlp(self.x, self.y, self.x, self.y, _text_config(), seed=7, fixed_epochs=2)
        second, _ = fit_mlp(self.x, self.y, self.x, self.y, _text_config(), seed=7, fixed_epochs=2)
        self.assertEqual(state_dict_sha256(first.state_dict()), state_dict_sha256(second.state_dict()))

    def test_different_seed_changes_initial_text_parameters(self) -> None:
        seed_everything(1)
        first = TextScoreHead(input_dim=4, hidden_dim=8, dropout=0)
        seed_everything(2)
        second = TextScoreHead(input_dim=4, hidden_dim=8, dropout=0)
        self.assertNotEqual(state_dict_sha256(first.state_dict()), state_dict_sha256(second.state_dict()))


def _graph_dataset() -> GraphDataset:
    rng = np.random.default_rng(42)
    dates = np.asarray([dt.date(2025, 1, day) for day in range(2, 6)], dtype=object)
    features = rng.normal(size=(4, 3, 3)).astype(np.float32)
    adjacency = np.asarray(
        [[[1.0, 0.7, -0.3], [0.7, 1.0, 0.2], [-0.3, 0.2, 1.0]]] * 4,
        dtype=np.float32,
    )
    targets = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=np.float32)
    return GraphDataset(
        sample_ids=np.asarray([f"g{index}" for index in range(4)]),
        dates=dates,
        symbols=("TSLA", "NVDA", "AAPL"),
        features=features,
        adjacency=adjacency,
        feature_mask=np.ones_like(features, dtype=bool),
        targets=targets,
        label_mask=np.ones_like(targets, dtype=bool),
        decision_ids=np.asarray([[f"d{day}-{node}" for node in range(3)] for day in range(4)]),
        feature_names=("a", "b", "c"),
    )


def _graph_config() -> dict[str, object]:
    return {
        "hidden_dim": 4,
        "num_heads": 2,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 2,
        "max_epochs": 4,
        "patience": 2,
        "gradient_clip_norm": 1.0,
    }


class Stage6GraphModelTest(unittest.TestCase):
    def test_graph_features_include_action_signal(self) -> None:
        self.assertIn("action_signal", GRAPH_FEATURE_NAMES)

    def test_graph_features_exclude_text_features(self) -> None:
        self.assertIn("text_score", GRAPH_FEATURE_NAMES)
        self.assertTrue({"text_norm", "num_text_features"}.isdisjoint(GRAPH_FEATURE_NAMES))

    def test_graph_features_include_prior_state_contract(self) -> None:
        self.assertTrue(POLICY_DEPENDENT_GRAPH_FIELDS.issubset(GRAPH_FEATURE_NAMES))

    def test_gat_parameters_change_after_backward(self) -> None:
        graph = _graph_dataset()
        model, history = fit_gat(graph, graph, _graph_config(), seed=42, fixed_epochs=2)
        self.assertTrue(history["parameter_changed"])
        self.assertTrue(np.isfinite(predict_gat(model, graph)[0]).all())

    def test_gat_output_shape_is_batch_by_nodes(self) -> None:
        model = DenseSignedGAT(3, hidden_dim=4, num_heads=2, dropout=0)
        graph = _graph_dataset()
        output = model(torch.tensor(graph.features), torch.tensor(graph.adjacency), torch.tensor(graph.feature_mask))
        self.assertEqual(tuple(output.logits.shape), (4, 3))
        self.assertEqual(tuple(output.h_graph.shape), (4, 3, 8))

    def test_gat_attention_respects_adjacency_mask(self) -> None:
        model = DenseSignedGAT(2, hidden_dim=2, num_heads=2, dropout=0).eval()
        features = torch.ones((1, 3, 2))
        adjacency = torch.eye(3).unsqueeze(0)
        _, attention = model(features, adjacency, torch.ones_like(features, dtype=torch.bool), return_attention=True)
        off_diagonal = attention * (1 - torch.eye(3)[None, None])
        self.assertEqual(float(off_diagonal.abs().max()), 0.0)

    def test_gat_supports_signed_adjacency(self) -> None:
        seed_everything(42)
        model = DenseSignedGAT(2, hidden_dim=3, num_heads=2, dropout=0).eval()
        features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.ones_like(features, dtype=torch.bool)
        positive = torch.tensor([[[1.0, 0.8], [0.8, 1.0]]])
        negative = torch.tensor([[[1.0, -0.8], [-0.8, 1.0]]])
        self.assertFalse(
            torch.allclose(
                model(features, positive, mask).logits,
                model(features, negative, mask).logits,
            )
        )

    def test_gat_save_load_prediction_equivalence(self) -> None:
        graph = _graph_dataset()
        model = DenseSignedGAT(3, hidden_dim=4, num_heads=2, dropout=0).eval()
        expected = predict_gat(model, graph)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.pt"
            atomic_torch_save(path, {"architecture": model.architecture_config(), "state_dict": model.state_dict()})
            actual = predict_gat(load_gat(path), graph)[0]
        self.assertTrue(np.allclose(expected, actual))

    def test_no_labels_occur_in_gat_forward_inputs(self) -> None:
        self.assertNotIn("target", inspect.signature(DenseSignedGAT.forward).parameters)
        self.assertNotIn("label", inspect.signature(DenseSignedGAT.forward).parameters)


class Stage6OOFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=index) for index in range(120))
        self.folds = build_oof_folds(self.dates, folds=5, min_train_days=60, embargo_days=20)

    def test_every_oof_model_is_trained_on_strictly_earlier_dates(self) -> None:
        self.assertTrue(all(fold.train_dates[-1] < fold.prediction_dates[0] for fold in self.folds))

    def test_every_oof_fold_has_exact_20_day_embargo(self) -> None:
        self.assertTrue(all(len(fold.embargo_dates) == 20 for fold in self.folds))

    def test_oof_access_audit_rejects_validation(self) -> None:
        with self.assertRaises(ValueError):
            assert_oof_only_access([{"split": "validation_selection", "purpose": "oof"}])

    def test_oof_access_audit_rejects_test(self) -> None:
        with self.assertRaises(ValueError):
            assert_oof_only_access([{"split": "test", "purpose": "oof"}])

    def test_fusion_rejects_in_sample_predictions(self) -> None:
        row = {
            "is_oof": True,
            "model_train_end": dt.date(2025, 1, 2),
            "decision_date": dt.date(2025, 1, 2),
            "embargo_trading_days": 20,
        }
        with self.assertRaises(ValueError):
            validate_oof_rows([row], 20)


def _fusion_values() -> dict[str, np.ndarray]:
    text_logit = np.asarray([-2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2], dtype=np.float32)
    graph_logit = np.asarray([-1, -2, 0.2, -0.3, 0.4, -0.1, 2, 1], dtype=np.float32)
    return {
        "h_text": np.stack([text_logit, text_logit**2, np.ones(8)], axis=1),
        "h_graph": np.stack(
            [graph_logit, graph_logit**2, np.ones(8), text_logit - graph_logit],
            axis=1,
        ),
        "text_logit": text_logit,
        "graph_logit": graph_logit,
        "text_probability": 1.0 / (1.0 + np.exp(-text_logit)),
        "graph_probability": 1.0 / (1.0 + np.exp(-graph_logit)),
        "target": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32),
        "symbol": np.asarray(["TSLA", "NVDA", "AAPL", "TSLA", "NVDA", "AAPL", "TSLA", "NVDA"]),
    }


def _fusion_config() -> dict[str, object]:
    return {
        "latent_dim": 4,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 4,
        "max_epochs": 5,
        "patience": 2,
    }


class Stage6FusionCalibrationTest(unittest.TestCase):
    def test_fusion_accepts_valid_oof_rows(self) -> None:
        validate_oof_rows(
            [{
                "is_oof": True,
                "text_prediction_is_oof": True,
                "graph_representation_is_oof": True,
                "h_text": [0.1, 0.2],
                "h_graph": [0.3, 0.4],
                "model_train_end": dt.date(2025, 1, 1),
                "decision_date": dt.date(2025, 1, 22),
                "embargo_trading_days": 20,
            }],
            20,
        )

    def test_fusion_parameters_change_after_backward(self) -> None:
        _, history = fit_fusion(_fusion_values(), _fusion_values(), _fusion_config(), seed=42)
        self.assertTrue(history["parameter_changed"])

    def test_fusion_save_load_prediction_equivalence(self) -> None:
        model, _ = fit_fusion(_fusion_values(), _fusion_values(), _fusion_config(), seed=42)
        expected = predict_fusion(model, _fusion_values())[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.pt"
            atomic_torch_save(
                path,
                {"architecture": model.architecture_config(), "state_dict": model.state_dict(), "selected_fusion_model": "canonical_latent_fusion"},
            )
            loaded, selected = load_fusion_model(path)
            actual = predict_fusion(loaded, _fusion_values())[0]
        self.assertEqual(selected, "canonical_latent_fusion")
        self.assertTrue(np.allclose(expected, actual))

    def test_scalar_only_inputs_are_rejected(self) -> None:
        model = LatentGatedFusion(text_dim=3, graph_dim=4, latent_dim=4, dropout=0).eval()
        with self.assertRaises(ValueError):
            model(torch.tensor([1.25, 9.0]), torch.tensor([8.0, -0.75]))

    def test_fixed_average_baseline_is_reported(self) -> None:
        values = _fusion_values()
        model = LatentGatedFusion(text_dim=3, graph_dim=4, latent_dim=4, dropout=0)
        probability = predict_fusion(model, values)[1]
        self.assertIn("fixed_0.5_average", _candidate_metrics(values, probability))

    def test_calibration_guard_rejects_test(self) -> None:
        with self.assertRaises(TestSplitAccessError):
            SplitAccessGuard().assert_allowed("test", "calibration")

    def test_temperature_is_positive(self) -> None:
        values = _fusion_values()
        self.assertGreater(fit_temperature(values["text_logit"], values["target"]), 0)

    def test_temperature_fitting_does_not_change_fusion_weights(self) -> None:
        model = LatentGatedFusion(text_dim=3, graph_dim=4, latent_dim=4, dropout=0)
        before = state_dict_sha256(model.state_dict())
        values = _fusion_values()
        fit_temperature(values["text_logit"], values["target"])
        self.assertEqual(before, state_dict_sha256(model.state_dict()))


class Stage6BundleAndMetricsTest(unittest.TestCase):
    def test_bundle_loader_verifies_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.bin").write_bytes(b"verified")
            (root / "bundle_manifest.json").write_text(
                json.dumps({"file_sha256": {"artifact.bin": sha256_path(root / "artifact.bin")}}),
                encoding="utf-8",
            )
            self.assertEqual(ModelBundle.load(root).path, root)

    def test_bundle_loader_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.bin").write_bytes(b"tampered")
            (root / "bundle_manifest.json").write_text(
                json.dumps({"file_sha256": {"artifact.bin": "0" * 64}}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                ModelBundle.load(root)

    def test_metrics_handle_single_class_roc_auc_safely(self) -> None:
        with self.assertWarns(RuntimeWarning):
            result = binary_metrics(np.zeros(3), np.asarray([0.1, 0.2, 0.3]))
        self.assertIsNone(result["roc_auc"])
        self.assertIn("undefined", result["warning"])

    def test_small_synthetic_pipeline_completes_end_to_end(self) -> None:
        text_x = np.asarray([[-2, 0], [-1, 1], [-0.5, 0], [0.5, 1], [1, 0], [2, 1]], dtype=np.float32)
        targets = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)
        text_model, _ = fit_mlp(text_x, targets, text_x, targets, {**_text_config(), "hidden_dim": 4}, seed=3, fixed_epochs=2)
        text_logits = predict_mlp(text_model, text_x)[0]
        graph = _graph_dataset()
        graph_model, _ = fit_gat(graph, graph, _graph_config(), seed=4, fixed_epochs=2)
        graph_logits, graph_probability, _, graph_latent = predict_gat(
            graph_model, graph, include_latent=True
        )
        graph_logits = graph_logits.reshape(-1)[:6]
        values = {
            "h_text": text_x,
            "h_graph": graph_latent.reshape(-1, graph_latent.shape[-1])[:6],
            "text_logit": text_logits,
            "graph_logit": graph_logits,
            "text_probability": 1.0 / (1.0 + np.exp(-text_logits)),
            "graph_probability": graph_probability.reshape(-1)[:6],
            "target": targets,
            "symbol": np.asarray(["TSLA", "NVDA", "AAPL", "TSLA", "NVDA", "AAPL"]),
        }
        fusion, _ = fit_fusion(values, values, _fusion_config(), seed=5)
        fused = predict_fusion(fusion, values)[0]
        temperature = fit_temperature(fused, targets)
        self.assertTrue(np.isfinite(fused / temperature).all())

    def test_stage6_metrics_do_not_contain_test_metrics(self) -> None:
        result = binary_metrics(np.asarray([0, 1]), np.asarray([0.2, 0.8]))
        self.assertNotIn("test", json.dumps(result).lower())

    def test_numpy_mvp_gat_remains_a_separate_baseline(self) -> None:
        self.assertIsNot(NumpyGATScorer, DenseSignedGAT)
        self.assertEqual(NumpyGATScorer.__module__, DenseSignedGAT.__module__)


if __name__ == "__main__":
    unittest.main()
