from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from puppy.fusion.gated_fusion import LatentGatedFusion
from puppy.fusion.scalar_logit_stack import GraphOnlyFusion, ScalarLogitStack
from puppy.fusion.text_score_head import TextScoreHead
from puppy.graph.gat_model import DenseSignedGAT, NumpyGATScorer
from puppy.training.calibration import fit_temperature
from puppy.training.checkpoint import (
    TrainOnlyScaler,
    atomic_torch_save,
    seed_everything,
    state_dict_sha256,
)
from puppy.training.config import TrainingConfig, _validate
from puppy.training.fusion_trainer import (
    GRAPH_ONLY_SELECTION,
    SCALAR_FUSION_SELECTION,
    _candidate_metrics,
    fit_fusion,
    fit_scalar_logit_stack,
    load_fusion_model,
    predict_fusion,
    selected_fusion_logits,
    train_fusion,
    validate_oof_rows,
)
from puppy.training.graph_dataset_loader import (
    GRAPH_FEATURE_NAMES,
    POLICY_DEPENDENT_GRAPH_FIELDS,
    GraphDataset,
    configured_graph_feature_names,
    graph_feature_schema,
    load_graph_dataset,
)
from puppy.training.graph_trainer import (
    fit_gat,
    fit_graph_scaler,
    load_gat,
    predict_gat,
    scale_graph_dataset,
    train_graph_stream,
)
from puppy.training.metrics import binary_metrics
from puppy.training.model_bundle import ModelBundle
from puppy.training.oof import (
    _generate_text_oof,
    assert_oof_only_access,
    generate_validation_base_predictions,
    minimum_history_coverage_table,
)
from puppy.training.pipeline import _can_resume, verify_causal_inputs
from puppy.training.split import (
    EffectiveSplits,
    build_effective_splits,
    build_inner_chronological_split,
    build_oof_folds,
)
from puppy.training.text_dataset import TextDataset, load_supervised_text_dataset
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
        "graph": {"topology": "full", "include_text_score": True},
        "fusion": {"head": "latent_gate"},
        "oof": {"training_policy": "fixed_epochs"},
        "text_model": {
            "hidden_dim": 8, "batch_size": 4, "max_epochs": 5, "patience": 2,
        },
        "graph_model": {
            "hidden_dim": 4, "num_heads": 2, "batch_size": 2,
            "max_epochs": 4, "patience": 2,
        },
        "fusion_model": {
            "latent_dim": 4, "batch_size": 4, "max_epochs": 5, "patience": 2,
        },
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
        graph=raw["graph"],
        fusion=raw["fusion"],
        oof=raw["oof"],
    )


class Stage6AblationConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = _config(Path(self.temporary.name))
        (self.config.artifact_dir / "data_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (self.config.artifact_dir / "temporal_quality.json").write_text(
            '{"status":"PASS"}\n', encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unknown_and_unsupported_ablation_values_fail_closed(self) -> None:
        invalid = (
            replace(
                self.config,
                graph={
                    "topology": "full",
                    "include_text_score": True,
                    "ignored": True,
                },
            ),
            replace(
                self.config,
                graph={"topology": "dense", "include_text_score": True},
            ),
            replace(self.config, fusion={"head": "average"}),
            replace(self.config, oof={"training_policy": "random_kfold"}),
            replace(
                self.config,
                graph={"topology": "full", "include_text_score": 1},
            ),
        )
        for config in invalid:
            with self.subTest(contract=config.ablation_contract), self.assertRaises(
                ValueError
            ):
                _validate(config)

    def test_every_ablation_has_a_distinct_fingerprint_and_output(self) -> None:
        root = self.config.repo_root / "data" / "model_artifacts" / "ablations"
        contracts = (
            ({"topology": "identity", "include_text_score": False}, {"head": "graph_only"}),
            ({"topology": "full", "include_text_score": False}, {"head": "graph_only"}),
            ({"topology": "full", "include_text_score": True}, {"head": "graph_only"}),
            ({"topology": "full", "include_text_score": True}, {"head": "scalar_logit_stack"}),
        )
        configs = [
            replace(
                self.config,
                graph=graph,
                fusion=fusion,
                model_dir=root / f"e{index}",
            )
            for index, (graph, fusion) in enumerate(contracts, start=1)
        ]
        self.assertEqual(len({item.ablation_fingerprint for item in configs}), 4)
        self.assertEqual(len({item.model_dir for item in configs}), 4)
        self.assertTrue(
            all(item.model_dir != self.config.model_dir for item in configs)
        )

    def test_noncanonical_contract_cannot_target_canonical_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate output.model_dir"):
            _validate(
                replace(
                    self.config,
                    graph={"topology": "identity", "include_text_score": False},
                )
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
        "master_portfolio_state": {"secret": "MASTER_PORTFOLIO_PRIOR"},
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

    def test_semantic_view_includes_prior_master_portfolio_and_psychology(self) -> None:
        text, _ = canonical_memory_text(_pending())
        self.assertIn("MASTER_PORTFOLIO_PRIOR", text)
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


class Stage6GraphTextScoreAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = _config(Path(self.temporary.name))
        self.date = dt.date(2025, 1, 5)
        self.splits = EffectiveSplits(
            warmup=(),
            train_manifest=(),
            train_effective=(),
            embargoed_train=(),
            validation_selection=(self.date,),
            validation_calibration=(),
            test_metadata_dates=(),
        )
        (self.config.artifact_dir / "feature_schema.json").write_text(
            json.dumps({"graph_feature_names": list(GRAPH_FEATURE_NAMES)}),
            encoding="utf-8",
        )
        outcomes = [
            {
                "decision_id": "supervised",
                "decision_date": self.date,
                "symbol": "TSLA",
                "direction_label": 1,
                "train_mask": True,
                "supervised_eligible": True,
                "cross_split_outcome": False,
            },
            {
                "decision_id": "neutral",
                "decision_date": self.date,
                "symbol": "NVDA",
                "direction_label": 0,
                "train_mask": False,
                "supervised_eligible": True,
                "cross_split_outcome": False,
            },
            {
                "decision_id": "cross-split",
                "decision_date": self.date,
                "symbol": "AAPL",
                "direction_label": -1,
                "train_mask": True,
                "supervised_eligible": False,
                "cross_split_outcome": True,
            },
        ]
        pq.write_table(
            pa.Table.from_pylist(outcomes),
            self.config.artifact_dir / "outcomes.parquet",
        )
        text_index = GRAPH_FEATURE_NAMES.index("text_score")
        features = np.zeros((3, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
        features[:, text_index] = 0.5
        feature_mask = np.ones_like(features, dtype=bool)
        feature_mask[:, text_index] = False
        self.adjacency = np.asarray(
            [[1.0, 0.4, -0.2], [0.4, 1.0, 0.3], [-0.2, 0.3, 1.0]],
            dtype=np.float32,
        )
        shard_dir = self.config.artifact_dir / "state" / "graph_shards"
        shard_dir.mkdir(parents=True)
        np.savez_compressed(
            shard_dir / f"{self.date.isoformat()}.npz",
            sample_id=np.asarray("graph-date"),
            decision_date=np.asarray(self.date.isoformat()),
            symbols=np.asarray(["TSLA", "NVDA", "AAPL"]),
            feature_matrix=features,
            adjacency_matrix=self.adjacency,
            feature_mask=feature_mask,
            max_source_date=np.asarray(self.date.isoformat()),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load(self, scores: dict[str, float] | None) -> GraphDataset:
        return load_graph_dataset(
            self.config,
            self.splits,
            dates=(self.date,),
            purpose="model_selection",
            text_scores_by_decision=scores,
            require_oof_text_scores=True,
        )

    def _load_configured(
        self,
        config: TrainingConfig,
        scores: dict[str, float] | None = None,
    ) -> GraphDataset:
        return load_graph_dataset(
            config,
            self.splits,
            dates=(self.date,),
            purpose="model_selection",
            text_scores_by_decision=scores,
            require_oof_text_scores=config.graph_includes_text_score,
        )

    def test_neutral_node_remains_masked_without_text_score(self) -> None:
        dataset = self._load({"supervised": 0.8})
        text_index = GRAPH_FEATURE_NAMES.index("text_score")
        self.assertEqual(dataset.decision_ids[0, 1], "neutral")
        self.assertFalse(dataset.label_mask[0, 1])
        self.assertFalse(dataset.feature_mask[0, 1, text_index])
        self.assertTrue(np.array_equal(dataset.adjacency[0], self.adjacency))

    def test_cross_split_node_remains_masked_without_text_score(self) -> None:
        dataset = self._load({"supervised": 0.8})
        text_index = GRAPH_FEATURE_NAMES.index("text_score")
        self.assertEqual(dataset.decision_ids[0, 2], "cross-split")
        self.assertFalse(dataset.label_mask[0, 2])
        self.assertFalse(dataset.feature_mask[0, 2, text_index])
        self.assertEqual(dataset.symbols, ("TSLA", "NVDA", "AAPL"))

    def test_supervised_node_missing_text_score_still_raises(self) -> None:
        for scores in ({}, None):
            with self.subTest(scores=scores), self.assertRaisesRegex(
                ValueError,
                "Missing required text score override for supervised decision supervised",
            ):
                self._load(scores)

    def test_identity_topology_preserves_node_target_and_feature_alignment(self) -> None:
        scores = {"supervised": 0.8}
        full = self._load_configured(self.config, scores)
        identity = self._load_configured(
            replace(
                self.config,
                graph={"topology": "identity", "include_text_score": True},
                model_dir=self.config.model_dir.parent / "identity",
            ),
            scores,
        )
        self.assertTrue(np.array_equal(identity.decision_ids, full.decision_ids))
        self.assertTrue(np.array_equal(identity.targets, full.targets))
        self.assertTrue(np.array_equal(identity.label_mask, full.label_mask))
        self.assertTrue(np.array_equal(identity.features, full.features))
        self.assertTrue(np.array_equal(identity.feature_mask, full.feature_mask))
        self.assertTrue(np.array_equal(identity.adjacency[0], np.eye(3)))

    def test_no_text_graph_removes_dependency_and_covers_every_eligible_row(self) -> None:
        config = replace(
            self.config,
            graph={"topology": "full", "include_text_score": False},
            model_dir=self.config.model_dir.parent / "no_text",
        )
        dataset = self._load_configured(config)
        self.assertNotIn("text_score", dataset.feature_names)
        self.assertEqual(
            dataset.feature_names, configured_graph_feature_names(config)
        )
        self.assertEqual(dataset.features.shape[-1], len(GRAPH_FEATURE_NAMES) - 1)
        schema = graph_feature_schema(config)
        self.assertEqual(schema["input_dim"], dataset.features.shape[-1])
        self.assertEqual(schema["feature_names"], list(dataset.feature_names))
        self.assertFalse(schema["include_text_score"])
        self.assertEqual(dataset.supervised_count, 1)
        self.assertEqual(dataset.decision_ids[dataset.label_mask].tolist(), ["supervised"])
        with self.assertRaisesRegex(ValueError, "forbids text-score overrides"):
            self._load_configured(config, {"supervised": 0.8})

    def test_missing_required_graph_feature_fails_closed(self) -> None:
        missing = [name for name in GRAPH_FEATURE_NAMES if name != "action_signal"]
        (self.config.artifact_dir / "feature_schema.json").write_text(
            json.dumps({"graph_feature_names": missing}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "missing required features"):
            self._load_configured(self.config, {"supervised": 0.8})


class Stage6ValidationBaseAlignmentTest(unittest.TestCase):
    def test_validation_base_emits_exact_supervised_id_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            first = dt.date(2025, 1, 1)
            dates = tuple(first + dt.timedelta(days=index) for index in range(63))
            splits = EffectiveSplits(
                warmup=(),
                train_manifest=(),
                train_effective=(),
                embargoed_train=(),
                validation_selection=dates[:42],
                validation_calibration=dates[42:],
                test_metadata_dates=(first + dt.timedelta(days=100),),
            )
            symbols = ("TSLA", "NVDA", "AAPL")
            graph_ids = np.asarray(
                [[f"{date.isoformat()}-{symbol}" for symbol in symbols] for date in dates]
            )
            flattened = graph_ids.reshape(-1).tolist()
            excluded = set(flattened[:9] + flattened[126:131])
            eligible_ids = [decision_id for decision_id in flattened if decision_id not in excluded]
            positions = {
                graph_ids[day, node]: (date, symbol)
                for day, date in enumerate(dates)
                for node, symbol in enumerate(symbols)
            }
            text_data = TextDataset(
                decision_ids=np.asarray(eligible_ids),
                dates=np.asarray([positions[value][0] for value in eligible_ids], dtype=object),
                symbols=np.asarray([positions[value][1] for value in eligible_ids]),
                embeddings=np.zeros((175, 384), dtype=np.float32),
                targets=np.asarray([index % 2 for index in range(175)], dtype=np.float32),
                source_context_hashes=np.asarray([f"hash-{index}" for index in range(175)]),
                metadata={
                    "masked_neutral_count": 11,
                    "cross_split_excluded_count": 3,
                    "warmup_excluded_count": 0,
                    "ineligible_count": 0,
                    "missing_outcome_count": 0,
                },
            )
            graph = GraphDataset(
                sample_ids=np.asarray([f"graph-{date.isoformat()}" for date in dates]),
                dates=np.asarray(dates, dtype=object),
                symbols=symbols,
                features=np.zeros((63, 3, len(GRAPH_FEATURE_NAMES)), dtype=np.float32),
                adjacency=np.broadcast_to(np.eye(3, dtype=np.float32), (63, 3, 3)).copy(),
                feature_mask=np.ones(
                    (63, 3, len(GRAPH_FEATURE_NAMES)), dtype=bool
                ),
                targets=np.zeros((63, 3), dtype=np.float32),
                label_mask=~np.isin(graph_ids, list(excluded)),
                decision_ids=graph_ids,
            )
            scaler = mock.Mock()
            scaler.transform.side_effect = lambda values: values
            probabilities = np.linspace(0.1, 0.9, 175, dtype=np.float32)
            logits = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
            graph_logits = np.zeros((63, 3), dtype=np.float32)
            graph_probabilities = np.full((63, 3), 0.5, dtype=np.float32)
            alpha = np.broadcast_to(np.eye(3, dtype=np.float32), (63, 3, 3)).copy()
            h_graph = np.zeros((63, 3, 4), dtype=np.float32)
            guard = SplitAccessGuard()

            def load_graph(*_args, **kwargs):
                self.assertTrue(kwargs["require_oof_text_scores"])
                self.assertEqual(
                    set(kwargs["text_scores_by_decision"]), set(eligible_ids)
                )
                kwargs["guard"].assert_allowed(
                    "validation_selection", kwargs["purpose"]
                )
                kwargs["guard"].assert_allowed(
                    "validation_calibration", kwargs["purpose"]
                )
                return graph

            with (
                mock.patch(
                    "puppy.training.oof.load_supervised_text_dataset",
                    return_value=text_data,
                ),
                mock.patch("puppy.training.oof.TrainOnlyScaler.load", return_value=scaler),
                mock.patch(
                    "puppy.training.oof.selected_text_predictions",
                    return_value=(logits, probabilities, "canonical_text_score_head"),
                ),
                mock.patch("puppy.training.oof.load_graph_dataset", side_effect=load_graph),
                mock.patch("puppy.training.oof.scale_graph_dataset", return_value=graph),
                mock.patch("puppy.training.oof.load_gat", return_value=mock.Mock()),
                mock.patch(
                    "puppy.training.oof.predict_gat",
                    return_value=(
                        graph_logits,
                        graph_probabilities,
                        alpha,
                        h_graph,
                    ),
                ),
            ):
                summary = generate_validation_base_predictions(
                    config, splits, guard=guard, device="cpu"
                )

            rows = pq.read_table(
                config.model_dir / "validation_base_predictions.parquet"
            ).to_pylist()
            output_ids = [row["decision_id"] for row in rows]
            self.assertEqual(len(rows), 175)
            self.assertEqual(len(output_ids), len(set(output_ids)))
            self.assertEqual(set(output_ids), set(eligible_ids))
            self.assertTrue(
                all(
                    graph_ids[dates.index(row["decision_date"]), symbols.index(row["symbol"])]
                    == row["decision_id"]
                    for row in rows
                )
            )
            self.assertEqual(summary["validation_selection_rows"], 117)
            self.assertEqual(summary["validation_calibration_rows"], 58)
            self.assertEqual(guard.test_feature_access_count, 0)


class Stage6FusionBoundaryAndCheckpointTest(unittest.TestCase):
    @staticmethod
    def _fusion_row(decision_id: str, partition: str, target: int) -> dict[str, object]:
        return {
            "decision_id": decision_id,
            "decision_date": dt.date(2025, 1, 2),
            "symbol": "TSLA",
            "partition": partition,
            "h_text": [0.1, 0.2],
            "h_graph": [0.3, 0.4],
            "text_logit": 0.1,
            "text_probability": 0.525,
            "graph_logit": -0.1,
            "graph_probability": 0.475,
            "target": target,
        }

    def test_fusion_model_selection_excludes_calibration_labels(self) -> None:
        class SelectionChecked(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            splits = EffectiveSplits(
                warmup=(),
                train_manifest=(),
                train_effective=(dt.date(2024, 12, 1),),
                embargoed_train=(),
                validation_selection=(dt.date(2025, 1, 2),),
                validation_calibration=(dt.date(2025, 1, 3),),
                test_metadata_dates=(dt.date(2025, 1, 4),),
            )
            oof_rows = [
                self._fusion_row("oof-negative", "train", 0),
                self._fusion_row("oof-positive", "train", 1),
            ]
            validation_rows = [
                self._fusion_row("selection-negative", "validation_selection", 0),
                self._fusion_row("selection-positive", "validation_selection", 1),
                self._fusion_row("calibration-sentinel", "validation_calibration", 99),
            ]
            pq.write_table(
                pa.Table.from_pylist(oof_rows),
                config.model_dir / "oof_predictions.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(validation_rows),
                config.model_dir / "validation_base_predictions.parquet",
            )
            guard = SplitAccessGuard()

            def check_selection(_train, validation, *_args, **_kwargs):
                self.assertEqual(validation["target"].tolist(), [0.0, 1.0])
                raise SelectionChecked

            with (
                mock.patch(
                    "puppy.training.fusion_trainer.generate_validation_base_predictions",
                    return_value={"validation_selection_rows": 2},
                ),
                mock.patch("puppy.training.fusion_trainer.validate_oof_rows"),
                mock.patch(
                    "puppy.training.fusion_trainer.fit_fusion",
                    side_effect=check_selection,
                ),
                self.assertRaises(SelectionChecked),
            ):
                train_fusion(config, splits, guard=guard, device="cpu")

            self.assertEqual(guard.test_feature_access_count, 0)

    def test_scalar_stack_selection_excludes_calibration_labels(self) -> None:
        class SelectionChecked(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            config = replace(_config(Path(directory)), fusion={"head": "scalar_logit_stack"})
            splits = EffectiveSplits(
                warmup=(),
                train_manifest=(),
                train_effective=(dt.date(2024, 12, 1),),
                embargoed_train=(),
                validation_selection=(dt.date(2025, 1, 2),),
                validation_calibration=(dt.date(2025, 1, 3),),
                test_metadata_dates=(dt.date(2025, 1, 4),),
            )
            oof_rows = [
                self._fusion_row("oof-negative", "train", 0),
                self._fusion_row("oof-positive", "train", 1),
            ]
            validation_rows = [
                self._fusion_row("selection-negative", "validation_selection", 0),
                self._fusion_row("selection-positive", "validation_selection", 1),
                self._fusion_row("calibration-sentinel", "validation_calibration", 99),
            ]
            pq.write_table(
                pa.Table.from_pylist(oof_rows),
                config.model_dir / "oof_predictions.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(validation_rows),
                config.model_dir / "validation_base_predictions.parquet",
            )

            def check_selection(train, validation, *_args, **_kwargs):
                self.assertNotIn("h_text", train)
                self.assertNotIn("h_graph", train)
                self.assertEqual(validation["target"].tolist(), [0.0, 1.0])
                raise SelectionChecked

            with (
                mock.patch(
                    "puppy.training.fusion_trainer.generate_validation_base_predictions",
                    return_value={"validation_selection_rows": 2},
                ),
                mock.patch("puppy.training.fusion_trainer.validate_oof_rows"),
                mock.patch(
                    "puppy.training.fusion_trainer.fit_scalar_logit_stack",
                    side_effect=check_selection,
                ),
                self.assertRaises(SelectionChecked),
            ):
                train_fusion(config, splits, guard=SplitAccessGuard(), device="cpu")

    def test_incompatible_fusion_checkpoint_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            stale_output = config.model_dir / "validation_base_predictions.parquet"
            stale_output.write_bytes(b"stale fusion output")
            checkpoint = config.model_dir / ".stage_checkpoints" / "fusion.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(
                json.dumps(
                    {
                        "fingerprint": {
                            "stage": "fusion",
                            "training_code_sha256": "old-code",
                        },
                        "output_hashes": {
                            config.relative(stale_output): sha256_path(stale_output)
                        },
                        "result": {"status": "stale"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "Incompatible checkpoint for stage fusion"
            ):
                _can_resume(
                    config,
                    "fusion",
                    {"stage": "fusion", "training_code_sha256": "new-code"},
                )


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

    def test_no_text_graph_trainer_uses_all_eligible_supervised_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            config = replace(
                config,
                graph={"topology": "full", "include_text_score": False},
                graph_model=_graph_config(),
                model_dir=config.model_dir.parent / "no_text_graph",
            )
            config.model_dir.mkdir(parents=True)
            (config.artifact_dir / "data_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            source = _graph_dataset()
            feature_names = configured_graph_feature_names(config)
            graph = replace(
                source,
                features=np.zeros(
                    (len(source), len(source.symbols), len(feature_names)),
                    dtype=np.float32,
                ),
                feature_mask=np.ones(
                    (len(source), len(source.symbols), len(feature_names)),
                    dtype=bool,
                ),
                feature_names=feature_names,
            )
            splits = EffectiveSplits(
                warmup=(),
                train_manifest=tuple(graph.dates[:3]),
                train_effective=tuple(graph.dates[:3]),
                embargoed_train=(),
                validation_selection=(graph.dates[3],),
                validation_calibration=(),
                test_metadata_dates=(),
            )
            with (
                mock.patch(
                    "puppy.training.graph_trainer.load_graph_dataset",
                    return_value=graph,
                ),
                mock.patch(
                    "puppy.training.graph_trainer.load_supervised_text_dataset",
                    side_effect=AssertionError("No-text Graph must not load Text data"),
                ),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                history = train_graph_stream(config, splits, device="cpu")
            self.assertEqual(history["eligible_train_supervised_rows"], 9)
            self.assertEqual(history["train_supervised_rows"], 9)
            self.assertEqual(history["excluded_train_supervised_rows"], 0)
            self.assertEqual(history["text_score_source"], "disabled")


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

    def test_inner_early_stop_split_is_chronological_and_embargoed(self) -> None:
        history = self.folds[-1].train_dates
        inner = build_inner_chronological_split(history, embargo_days=20)
        self.assertEqual(len(inner.embargo_dates), 20)
        self.assertLess(max(inner.train_dates), min(inner.validation_dates))
        self.assertLess(max(inner.validation_dates), self.folds[-1].prediction_dates[0])
        self.assertTrue(
            set(inner.train_dates + inner.embargo_dates + inner.validation_dates)
            == set(history)
        )

    def test_minimum_history_table_reports_coverage_without_applying_threshold(self) -> None:
        records = [
            {
                "fold_id": fold,
                "decision_date": dt.date(2025, 1, fold + 1),
                "symbol": symbol,
            }
            for fold in (0, 1)
            for symbol in ("TSLA", "NVDA", "AAPL")
        ]
        reports = [
            {"fold_id": 0, "status": "complete", "graph_train_days": 30, "graph_train_rows": 90},
            {"fold_id": 1, "status": "complete", "graph_train_days": 70, "graph_train_rows": 210},
        ]
        table = minimum_history_coverage_table(
            records, reports, ("TSLA", "NVDA", "AAPL")
        )
        baseline = table[0]
        self.assertEqual(baseline["folds_retained"], [0, 1])
        self.assertEqual(baseline["fusion_rows_remaining"], 6)
        threshold_40 = next(
            row for row in table if row["minimum_graph_train_days"] == 40
        )
        self.assertEqual(threshold_40["folds_retained"], [1])
        self.assertEqual(threshold_40["oof_supervised_rows_retained"], 3)

    def test_inner_early_stop_never_uses_outer_prediction_rows(self) -> None:
        fold = self.folds[0]
        dates = fold.train_dates + fold.prediction_dates
        text_data = TextDataset(
            decision_ids=np.asarray([f"decision-{index}" for index in range(len(dates))]),
            dates=np.asarray(dates, dtype=object),
            symbols=np.asarray(["TSLA"] * len(dates)),
            embeddings=np.zeros((len(dates), 384), dtype=np.float32),
            targets=np.asarray([index % 2 for index in range(len(dates))], dtype=np.float32),
            source_context_hashes=np.asarray([f"hash-{index}" for index in range(len(dates))]),
            metadata={},
        )
        calls = []

        def fake_fit(*args, **kwargs):
            calls.append(
                {
                    "train_rows": len(args[0]),
                    "validation_rows": len(args[2]),
                    "fixed_epochs": kwargs.get("fixed_epochs"),
                }
            )
            return object(), {"best_epoch": 3, "parameter_changed": True}

        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            with (
                mock.patch("puppy.training.oof.fit_mlp", side_effect=fake_fit),
                mock.patch(
                    "puppy.training.oof.predict_mlp",
                    side_effect=lambda _model, values, **_kwargs: (
                        np.zeros(len(values), dtype=np.float32),
                        np.full(len(values), 0.5, dtype=np.float32),
                    ),
                ),
            ):
                records, reports = _generate_text_oof(
                    config,
                    (fold,),
                    text_data,
                    seed=42,
                    device="cpu",
                    fixed_epochs=7,
                    manifest_hash="manifest",
                    training_policy="inner_chronological_early_stop",
                    embargo_days=20,
                )
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["fixed_epochs"])
        self.assertEqual(calls[1]["fixed_epochs"], 3)
        self.assertEqual(calls[1]["train_rows"], len(fold.train_dates))
        self.assertEqual(calls[1]["validation_rows"], len(fold.train_dates))
        self.assertEqual(len(records), len(fold.prediction_dates))
        self.assertEqual(reports[0]["epochs"], 3)
        self.assertLess(
            dt.date.fromisoformat(reports[0]["inner"]["validation_end"]),
            fold.prediction_dates[0],
        )


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
                "decision_id": "valid-oof",
                "is_oof": True,
                "text_prediction_is_oof": True,
                "graph_representation_is_oof": True,
                "h_text": [0.1, 0.2],
                "h_graph": [0.3, 0.4],
                "text_logit": 0.1,
                "graph_logit": -0.1,
                "model_train_end": dt.date(2025, 1, 1),
                "decision_date": dt.date(2025, 1, 22),
                "embargo_trading_days": 20,
            }],
            20,
        )

    def test_scalar_stack_accepts_oof_logits_without_latent_vectors(self) -> None:
        validate_oof_rows(
            [{
                "decision_id": "scalar-oof",
                "is_oof": True,
                "text_prediction_is_oof": True,
                "graph_representation_is_oof": True,
                "text_logit": 0.2,
                "graph_logit": -0.3,
                "model_train_end": dt.date(2025, 1, 1),
                "decision_date": dt.date(2025, 1, 22),
                "embargo_trading_days": 20,
            }],
            20,
            require_latents=False,
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

    def test_scalar_stack_parameters_change_and_save_load_equivalence(self) -> None:
        values = _fusion_values()
        model, history = fit_scalar_logit_stack(
            values, values, _fusion_config(), seed=42
        )
        self.assertTrue(history["parameter_changed"])
        expected = selected_fusion_logits(model, SCALAR_FUSION_SELECTION, values)
        manual = (
            float(model.intercept.detach())
            + float(model.weights[0].detach()) * values["text_logit"]
            + float(model.weights[1].detach()) * values["graph_logit"]
        )
        self.assertTrue(np.allclose(expected, manual))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar.pt"
            atomic_torch_save(
                path,
                {
                    "architecture": model.architecture_config(),
                    "state_dict": model.state_dict(),
                    "selected_fusion_model": SCALAR_FUSION_SELECTION,
                },
            )
            loaded, selected = load_fusion_model(path)
            actual = selected_fusion_logits(loaded, selected, values)
        self.assertEqual(selected, SCALAR_FUSION_SELECTION)
        self.assertTrue(np.allclose(expected, actual))

    def test_graph_only_head_is_exact_and_parameter_free(self) -> None:
        values = _fusion_values()
        model = GraphOnlyFusion()
        logits = selected_fusion_logits(model, GRAPH_ONLY_SELECTION, values)
        self.assertTrue(np.array_equal(logits, values["graph_logit"]))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)

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
    @staticmethod
    def _ablation_manifest() -> dict[str, object]:
        contract = {
            "graph": {"topology": "identity", "include_text_score": False},
            "fusion": {"head": "graph_only"},
            "oof": {"training_policy": "fixed_epochs"},
        }
        encoded = json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            "bundle_version": "finswarm-stage6-ablation-bundle-v1",
            "selected_fusion_model": "graph_only",
            "ablation_contract": contract,
            "ablation_fingerprint": hashlib.sha256(encoded).hexdigest(),
            "graph_input": {
                "topology": "identity",
                "include_text_score": False,
                "input_dim": 1,
                "feature_names": ["action_signal"],
            },
            "model_classes": {
                "text_encoder": "puppy.fusion.text_encoder.TextStreamEncoder",
                "text_score_head": "puppy.fusion.text_score_head.TextScoreHead",
                "gat": "puppy.graph.gat_model.DenseSignedGAT",
                "fusion": "puppy.fusion.scalar_logit_stack.GraphOnlyFusion",
            },
            "file_sha256": {},
        }

    def test_ablation_bundle_requires_matching_contract_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._ablation_manifest()
            (root / "bundle_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(ModelBundle.load(root).path, root)
            manifest["ablation_fingerprint"] = "0" * 64
            (root / "bundle_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                ModelBundle.load(root)

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
