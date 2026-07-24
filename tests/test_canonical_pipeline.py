from __future__ import annotations

import datetime as dt
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from puppy.common.schemas import (
    AgentContext,
    AllocationWeights,
    CausalRunContext,
    GraphOutput,
    MarketStep,
    PriceBar,
    PortfolioSnapshot,
    SharedContext,
)
from puppy.debate.coordinator import DebateCoordinator
from puppy.memory_ext.memory_sharing import MemorySharingService
from puppy.portfolio_ext.master_portfolio import MasterPortfolio
from puppy.fusion.gated_fusion import LatentGatedFusion
from puppy.fusion.text_encoder import TextStreamEncoder
from puppy.fusion.text_score_head import TextScoreHead
from puppy.graph.gat_model import DenseSignedGAT, GATModelOutput
from puppy.graph.graph_dataset import CANONICAL_FEATURE_NAMES, build_causal_graph_sample
from puppy.graph.graph_features import (
    ACTION_FEATURE_NAMES,
    MARKET_FEATURE_NAMES,
    MASTER_PORTFOLIO_FEATURE_NAMES,
    PSYCHOLOGY_FEATURE_NAMES,
    TEXT_SCORE_FEATURE_NAMES,
)
from puppy.graph.risk_router import build_memory_share_routes
from puppy.swarm.sector_agent import SectorAgent
from puppy.swarm.causal_orchestrator import CausalOrchestrator
from puppy.swarm.legacy_finmem_adapter import LegacyFinMemAdapter
from puppy.swarm.psychology import build_psychology_state
from puppy.training.checkpoint import TrainOnlyScaler, atomic_torch_save
from puppy.training.config import TrainingConfig
from puppy.training.fusion_trainer import CanonicalFusionModel
from puppy.training.graph_trainer import CanonicalGATModel
from puppy.training.model_bundle import ModelBundle
from puppy.training.pipeline import verify_causal_inputs
from puppy.training.text_trainer import CanonicalTextScoreHead
from puppy.training.text_embedding import sha256_path


class _Observed:
    date = dt.date(2025, 1, 3)
    as_of = dt.datetime(2025, 1, 3, 21, 10, tzinfo=dt.timezone.utc)
    symbols = ["TSLA", "NVDA"]
    features = {
        "TSLA": {
            "observed_close_return_1d": 0.01,
            "observed_intraday_return_1d": 0.02,
            "momentum_5d": 0.03,
            "momentum_20d": 0.04,
            "rolling_volatility_5d": 0.05,
            "rolling_volatility_20d": 0.06,
            "volume_ratio_20d": 1.1,
            "range_pct": 0.02,
            "feature_missing_count": 0,
        },
        "NVDA": {
            "observed_close_return_1d": -0.01,
            "observed_intraday_return_1d": -0.02,
            "momentum_5d": -0.03,
            "momentum_20d": -0.04,
            "rolling_volatility_5d": 0.07,
            "rolling_volatility_20d": 0.08,
            "volume_ratio_20d": 0.9,
            "range_pct": 0.03,
            "feature_missing_count": 0,
        },
    }


def _context() -> AgentContext:
    return AgentContext(
        agent_id="sector_tsla",
        symbol="TSLA",
        date=dt.date(2025, 1, 3),
        latest_reason="Prior decision reason, not the current decision.",
        short_memory=["shallow memory"],
        mid_memory=["intermediate memory"],
        long_memory=["deep memory"],
        reflection_memory=["eligible retrospective reflection"],
        shared_context=[
            SharedContext(
                source_agent_id="sector_nvda",
                target_agent_id="sector_tsla",
                source_symbol="NVDA",
                target_symbol="TSLA",
                text="eligible shared memory",
                reason="cross-sector provenance",
                alpha=0.8,
                created_at=dt.date(2025, 1, 2),
            )
        ],
        master_portfolio_state={
            "date": "2025-01-02",
            "cash": 50000.0,
            "cash_weight": 0.5,
            "positions": {"TSLA": 300.0},
            "weights": {"TSLA": 0.3, "NVDA": 0.2},
            "equity": 100000.0,
            "daily_return": 0.01,
            "cumulative_return": 0.1,
            "rolling_return": 0.08,
            "sharpe": 1.2,
            "drawdown": -0.03,
            "win_rate": 0.6,
        },
        psychology_state={"state": "greed", "source": "master_portfolio"},
        metadata={"built_before_decision": True},
    )


class CanonicalTextContractTest(unittest.TestCase):
    def test_full_causal_agent_context_is_serialized(self) -> None:
        encoder = TextStreamEncoder(
            embedding_func=lambda text: np.asarray([len(text)], dtype=np.float32),
            embedding_model="test",
        )
        text = encoder.serialize_context(_context())
        for value in (
            "shallow memory",
            "intermediate memory",
            "deep memory",
            "eligible retrospective reflection",
            "eligible shared memory",
            '"cumulative_return": 0.1',
            '"state": "greed"',
            "sector_nvda",
            "cross-sector provenance",
        ):
            self.assertIn(value, text)

    def test_serialized_snapshot_excludes_output_and_future_fields(self) -> None:
        payload = _context().model_dump(mode="json")
        payload.update(
            {
                "current_decision_reason": "FORBIDDEN_CURRENT_REASON",
                "outcome": "FORBIDDEN_OUTCOME",
                "direction_label": 1,
                "future_price": 999.0,
            }
        )
        encoder = TextStreamEncoder(
            embedding_func=lambda _text: np.ones(2, dtype=np.float32),
            embedding_model="test",
        )
        text = encoder.serialize_context(payload)
        self.assertNotIn("FORBIDDEN_CURRENT_REASON", text)
        self.assertNotIn("FORBIDDEN_OUTCOME", text)
        self.assertNotIn("future_price", text)
        self.assertNotIn("direction_label", text)

    def test_batch_and_single_encoding_match(self) -> None:
        encoder = TextStreamEncoder(
            embedding_func=lambda text: np.asarray([len(text), text.count("memory")], dtype=np.float32),
            embedding_model="test",
        )
        contexts = [_context(), _context().model_copy(update={"symbol": "NVDA"})]
        batch = encoder.encode_batch_arrays(contexts)
        singles = np.stack([encoder.encode_array(item)[0] for item in contexts])
        np.testing.assert_array_equal(batch[0], singles)

    def test_training_and_runtime_share_text_head_class(self) -> None:
        self.assertIs(CanonicalTextScoreHead, TextScoreHead)


class CanonicalGraphContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.master = PortfolioSnapshot(
            date=dt.date(2025, 1, 2),
            cash=50000.0,
            cash_weight=0.5,
            positions={"TSLA": 300.0, "NVDA": 200.0},
            weights={"TSLA": 0.3, "NVDA": 0.2},
            equity=100000.0,
            daily_return=0.01,
            cumulative_return=0.1,
            rolling_return=0.08,
            sharpe=1.2,
            drawdown=-0.03,
            win_rate=0.6,
        )

    def _sample(self):
        return build_causal_graph_sample(
            _Observed(),
            action_signals={"TSLA": 1, "NVDA": -1},
            text_scores={"TSLA": 0.7, "NVDA": 0.3},
            master_portfolio_state=self.master,
            psychology_states={
                "TSLA": {"state": "greed"},
                "NVDA": {"state": "fear"},
            },
            return_history={"TSLA": [0.1, 0.2], "NVDA": [-0.1, -0.2]},
        )

    def test_h0_has_named_ordered_blocks(self) -> None:
        expected = (
            MARKET_FEATURE_NAMES
            + ACTION_FEATURE_NAMES
            + TEXT_SCORE_FEATURE_NAMES
            + MASTER_PORTFOLIO_FEATURE_NAMES
            + PSYCHOLOGY_FEATURE_NAMES
        )
        self.assertEqual(CANONICAL_FEATURE_NAMES, expected)
        self.assertEqual(self._sample()["metadata"]["feature_names"], list(expected))

    def test_h0_contains_action_text_master_portfolio_and_psychology(self) -> None:
        sample = self._sample()
        tsla = sample["node_features"]["TSLA"]
        self.assertEqual(tsla["action_signal"], 1)
        self.assertEqual(tsla["text_score"], 0.7)
        self.assertEqual(tsla["master_symbol_weight_prior"], 0.3)
        self.assertEqual(tsla["master_cash_weight_prior"], 0.5)
        self.assertEqual(tsla["master_cumulative_return_prior"], 0.1)
        self.assertEqual(tsla["psychology_greed"], 1.0)
        self.assertEqual(tsla["psychology_fear"], 0.0)

    def test_h0_never_contains_target_or_future_return(self) -> None:
        names = set(CANONICAL_FEATURE_NAMES)
        self.assertTrue(
            names.isdisjoint(
                {"target", "label", "asset_return", "strategy_return", "future_return"}
            )
        )

    def test_gat_returns_latent_attention_logits_and_probabilities(self) -> None:
        sample = self._sample()
        model = DenseSignedGAT(
            len(CANONICAL_FEATURE_NAMES), hidden_dim=4, num_heads=2, dropout=0
        ).eval()
        result = model(
            torch.tensor(sample["feature_matrix"])[None],
            torch.tensor(sample["adjacency_matrix"])[None],
            torch.tensor(sample["feature_mask"])[None],
            symbols=tuple(sample["symbols"]),
            feature_names=CANONICAL_FEATURE_NAMES,
        )
        self.assertIsInstance(result, GATModelOutput)
        self.assertEqual(tuple(result.h_graph.shape), (1, 2, 8))
        self.assertEqual(tuple(result.alpha.shape), (1, 2, 2))
        self.assertEqual(tuple(result.logits.shape), (1, 2))
        self.assertEqual(tuple(result.probabilities.shape), (1, 2))
        self.assertEqual(result.symbols, tuple(sample["symbols"]))

    def test_training_and_runtime_share_gat_class(self) -> None:
        self.assertIs(CanonicalGATModel, DenseSignedGAT)


class CanonicalFusionContractTest(unittest.TestCase):
    def test_latent_fusion_matches_authoritative_equation(self) -> None:
        torch.manual_seed(7)
        model = LatentGatedFusion(text_dim=5, graph_dim=8, latent_dim=4, dropout=0).eval()
        result = model(torch.randn(3, 5), torch.randn(3, 8))
        self.assertEqual(tuple(result.h_text_projected.shape), (3, 4))
        self.assertEqual(tuple(result.h_graph_projected.shape), (3, 4))
        self.assertEqual(tuple(result.gate.shape), (3, 4))
        expected = (
            result.gate * result.h_text_projected
            + (1.0 - result.gate) * result.h_graph_projected
        )
        self.assertTrue(torch.allclose(result.h_final, expected))
        self.assertEqual(tuple(result.logits.shape), (3,))
        self.assertEqual(tuple(result.probabilities.shape), (3,))

    def test_conflict_uses_text_and_graph_probability_sign(self) -> None:
        text = torch.tensor([0.8, 0.2, 0.5])
        graph = torch.tensor([0.3, 0.1, 0.8])
        conflict = LatentGatedFusion.compute_conflict(text, graph)
        self.assertEqual(conflict.tolist(), [True, False, True])

    def test_training_and_runtime_share_fusion_class(self) -> None:
        self.assertIs(CanonicalFusionModel, LatentGatedFusion)


class CanonicalMemoryAndGateTest(unittest.TestCase):
    def test_memory_route_threshold_uses_alpha_times_importance(self) -> None:
        graph = GraphOutput(
            date=dt.date(2025, 1, 3),
            symbols=["TSLA", "NVDA"],
            attention={
                "TSLA": {"TSLA": 1.0, "NVDA": 0.8},
                "NVDA": {"TSLA": 0.2, "NVDA": 1.0},
            },
        )
        routes = build_memory_share_routes(
            graph,
            {"TSLA": "sector_tsla", "NVDA": "sector_nvda"},
            threshold=0.5,
            importance_by_source={"TSLA": 0.9, "NVDA": 0.7},
        )
        self.assertEqual(len(routes), 1)
        self.assertAlmostEqual(routes[0].routing_score, 0.56)
        self.assertAlmostEqual(routes[0].memory_importance, 0.7)

    def test_same_day_shared_context_is_not_eligible(self) -> None:
        agent = SectorAgent("sector_tsla", "TSLA", SimpleNamespace())
        agent.inject_external_context(
            SharedContext(
                source_agent_id="sector_nvda",
                target_agent_id="sector_tsla",
                text="later-cycle only",
                reason="attention route",
                created_at=dt.date(2025, 1, 3),
            )
        )
        self.assertEqual(agent.eligible_external_context(dt.date(2025, 1, 3)), [])
        self.assertEqual(len(agent.eligible_external_context(dt.date(2025, 1, 4))), 1)

    def test_final_training_rejects_bootstrap_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "data" / "training" / "finswarm_v1"
            artifact.mkdir(parents=True)
            config_path = root / "config" / "swarm" / "training_v1.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("run: {}\n", encoding="utf-8")
            experiment = config_path.parent / "experiment_v1.yaml"
            experiment.write_text("assets: {symbols: [TSLA]}\n", encoding="utf-8")
            split = root / "split.json"
            split.write_text("{}\n", encoding="utf-8")
            for name in (
                "text_samples.parquet",
                "agent_outputs.parquet",
                "observed_features.parquet",
                "outcomes.parquet",
                "pending_decisions.parquet",
                "reflection_updates.parquet",
                "graph_samples.npz",
                "feature_schema.json",
            ):
                (artifact / name).write_bytes(b"fixture")
            (artifact / "data_manifest.json").write_text(
                json.dumps(
                    {
                        "bootstrap": True,
                        "bootstrap_policy": "sha256-context-bucket-v1",
                        "external_llm_used": False,
                    }
                ),
                encoding="utf-8",
            )
            (artifact / "rollout_manifest.json").write_text(
                '{"status":"complete","temporal_gate":"PASS","bootstrap":true}',
                encoding="utf-8",
            )
            (artifact / "temporal_quality.json").write_text(
                '{"status":"PASS"}', encoding="utf-8"
            )
            config = TrainingConfig(
                config_path=config_path,
                repo_root=root,
                run_id="finswarm_v1",
                seed=42,
                experiment_config=experiment,
                artifact_dir=artifact,
                split_manifest=split,
                model_dir=root / "data" / "model_artifacts" / "finswarm_v1",
                text_embedding={},
                split={},
                labels={},
                text_model={},
                graph_model={},
                fusion_model={},
                calibration={},
                raw={},
            )
            with self.assertRaisesRegex(ValueError, "bootstrap action"):
                verify_causal_inputs(config)

    def test_bundle_rejects_graph_only_production_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "bundle_manifest.json").write_text(
                json.dumps(
                    {
                        "file_sha256": {},
                        "selected_fusion_model": "graph_only",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical latent fusion"):
                ModelBundle.load(path)

    def test_deleted_duplicate_modules_have_no_imports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = (
            ".".join(("puppy", "fusion", "trainable_text_head")),
            ".".join(("puppy", "graph", "torch_gat_model")),
            ".".join(("puppy", "fusion", "trainable_gated_fusion")),
        )
        source_files = list((root / "puppy").rglob("*.py")) + list(
            (root / "scripts").rglob("*.py")
        ) + list((root / "tests").rglob("*.py"))
        source_files = [path for path in source_files if path.resolve() != Path(__file__).resolve()]
        for module in forbidden:
            self.assertFalse(any(module in path.read_text(encoding="utf-8") for path in source_files))

    def test_milestone_scripts_still_import(self) -> None:
        for module in (
            "scripts.run_milestone_1_mock",
            "scripts.run_milestone_3_agents",
            "scripts.run_milestone_4_master_portfolio",
            "scripts.run_milestone_5_graph",
            "scripts.run_milestone_5_memory_sharing",
            "scripts.run_milestone_6_fusion",
            "scripts.run_milestone_7_debate",
            "scripts.run_milestone_7_allocation",
        ):
            with self.subTest(module=module):
                importlib.import_module(module)


class CanonicalRuntimeIntegrationTest(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[ModelBundle, TextStreamEncoder]:
        text_head = TextScoreHead(input_dim=4, hidden_dim=6, dropout=0, seed=7)
        gat = DenseSignedGAT(
            len(CANONICAL_FEATURE_NAMES), hidden_dim=2, num_heads=2, dropout=0
        )
        fusion = LatentGatedFusion(text_dim=4, graph_dim=4, latent_dim=3, dropout=0)
        atomic_torch_save(
            root / "text_model.pt",
            {
                "model_type": "canonical_text_score_head",
                "architecture": text_head.architecture_config(),
                "state_dict": text_head.state_dict(),
                "seed": 7,
                "selected": True,
            },
        )
        atomic_torch_save(
            root / "graph_model.pt",
            {
                "model_type": "canonical_dense_signed_gat",
                "architecture": gat.architecture_config(),
                "state_dict": gat.state_dict(),
                "feature_names": list(CANONICAL_FEATURE_NAMES),
                "symbols": list(_Observed.symbols),
            },
        )
        atomic_torch_save(
            root / "fusion_model.pt",
            {
                "model_type": "canonical_latent_fusion",
                "architecture": fusion.architecture_config(),
                "state_dict": fusion.state_dict(),
                "selected_fusion_model": "canonical_latent_fusion",
            },
        )
        text_scaler = TrainOnlyScaler([f"embedding_{index:03d}" for index in range(4)])
        text_scaler.fit(
            np.asarray([[0.0] * 4, [1.0] * 4]),
            dates=[dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
            input_manifest_hash="synthetic",
        ).save(root / "text_scaler.json")
        graph_scaler = TrainOnlyScaler(list(CANONICAL_FEATURE_NAMES))
        graph_scaler.fit(
            np.asarray([[0.0] * len(CANONICAL_FEATURE_NAMES), [1.0] * len(CANONICAL_FEATURE_NAMES)]),
            dates=[dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
            input_manifest_hash="synthetic",
        ).save(root / "graph_scaler.json")
        files = {
            name: sha256_path(root / name)
            for name in (
                "text_model.pt",
                "graph_model.pt",
                "fusion_model.pt",
                "text_scaler.json",
                "graph_scaler.json",
            )
        }
        (root / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "bundle_version": "finswarm-canonical-bundle-v2",
                    "selected_fusion_model": "canonical_latent_fusion",
                    "encoder_model": "synthetic-canonical",
                    "encoder_revision": "synthetic-v1",
                    "canonical_classes": {
                        "text_encoder": "puppy.fusion.text_encoder.TextStreamEncoder",
                        "text_score_head": "puppy.fusion.text_score_head.TextScoreHead",
                        "gat": "puppy.graph.gat_model.DenseSignedGAT",
                        "fusion": "puppy.fusion.gated_fusion.LatentGatedFusion",
                    },
                    "file_sha256": files,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        encoder = TextStreamEncoder(
            embedding_func=lambda text: np.asarray(
                [len(text) / 1000.0, text.count("memory"), text.count("Master"), text.count("Psychology")],
                dtype=np.float32,
            ),
            embedding_model="synthetic-canonical",
        )
        return ModelBundle.load(root), encoder

    def test_bundle_to_memory_debate_allocation_and_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, encoder = self._bundle(Path(directory))
            models = bundle.load_models(text_encoder=encoder)
            self.assertIsInstance(models.text_score_head, TextScoreHead)
            self.assertIsInstance(models.gat, DenseSignedGAT)
            self.assertIsInstance(models.fusion, LatentGatedFusion)

            date = dt.date(2025, 1, 3)
            cutoff = dt.datetime(2025, 1, 3, 21, 10, tzinfo=dt.timezone.utc)
            step = MarketStep(
                date=date,
                symbols=list(_Observed.symbols),
                prices={
                    symbol: PriceBar(symbol=symbol, date=date, open=100.0, close=101.0)
                    for symbol in _Observed.symbols
                },
                as_of=cutoff,
            )
            run_context = CausalRunContext(
                run_id="synthetic-runtime",
                config_path="synthetic",
                current_date=date,
                current_cutoff=cutoff,
                bootstrap=False,
                prompt_version="synthetic-prompt-v1",
                encoder_version="synthetic-v1",
                adapter_version="legacy-finmem-causal-v1",
                git_commit="synthetic",
            )
            agents = {}
            bundles = {}
            psychology_states = {}
            portfolio = MasterPortfolio(initial_cash=100000.0)
            prior_snapshot = portfolio.apply_target_weights(
                date=dt.date(2025, 1, 2),
                prices={symbol: 100.0 for symbol in _Observed.symbols},
                target_weights=AllocationWeights(
                    date=dt.date(2025, 1, 2),
                    weights={"TSLA": 0.3, "NVDA": 0.2},
                    cash_weight=0.5,
                ),
                as_of=dt.datetime(2025, 1, 2, 21, 10, tzinfo=dt.timezone.utc),
            )
            orchestrator = CausalOrchestrator(
                models.text_encoder,
                models.text_score_head,
                models.text_scaler,
            )
            for symbol, decision in zip(_Observed.symbols, ("buy", "sell")):
                agent_id = f"sector_{symbol.lower()}"
                llm = SimpleNamespace(
                    model_name="synthetic-causal-model",
                    causal_decide=lambda _step, _context, _run, value=decision: {
                        "investment_decision": value,
                        "summary_reason": f"synthetic {value}",
                    },
                )
                agent = SectorAgent(
                    agent_id,
                    symbol,
                    llm,
                    causal_adapter=LegacyFinMemAdapter(
                        llm, agent_id, symbol, bootstrap=False
                    ),
                )
                psychology = build_psychology_state(
                    prior_snapshot,
                    symbol=symbol,
                )
                agents[symbol] = agent
                psychology_states[symbol] = psychology
                bundles[symbol] = orchestrator.decide(
                    agent=agent,
                    market_step=step,
                    run_context=run_context,
                    expected_outcome_date=dt.date(2025, 1, 6),
                    split="validation",
                    master_portfolio_state=prior_snapshot,
                    psychology_state=psychology,
                )

            streams = orchestrator.run_canonical_streams(
                bundles_by_symbol=bundles,
                observed_table=_Observed(),
                master_portfolio_state=prior_snapshot,
                psychology_states=psychology_states,
                gat_model=models.gat,
                fusion_model=models.fusion,
                graph_scaler=models.graph_scaler,
                return_history={"TSLA": [0.1, 0.2], "NVDA": [-0.1, -0.2]},
                master_portfolio=portfolio,
                prices=step.prices,
            )
            self.assertTrue(streams.graph_output.graph_embeddings)
            self.assertTrue(streams.fusion_output.final_embeddings)
            self.assertIsNotNone(streams.allocation)
            self.assertIsNotNone(streams.portfolio_snapshot)

            routes = build_memory_share_routes(
                streams.graph_output,
                {symbol: agents[symbol].agent_id for symbol in _Observed.symbols},
                threshold=0.0,
                importance_by_source={symbol: 1.0 for symbol in _Observed.symbols},
                top_k=1,
            )
            events = MemorySharingService().apply_memory_share_routes(
                routes,
                {
                    symbol: {
                        "short": [
                            {
                                "memory_id": f"{symbol}-memory",
                                "text": f"{symbol} shared memory",
                                "importance": 1.0,
                            }
                        ]
                    }
                    for symbol in _Observed.symbols
                },
                agents,
            )
            self.assertTrue(events)
            self.assertTrue(
                any(
                    agent.eligible_external_context(dt.date(2025, 1, 4))
                    for agent in agents.values()
                )
            )

            signals = [bundles[symbol].output.signal for symbol in _Observed.symbols]
            verdicts = DebateCoordinator().run_for_fusion(
                streams.fusion_output,
                signals,
                agents,
            )
            self.assertIsInstance(verdicts, list)
            self.assertEqual(set(streams.allocation.weights), set(_Observed.symbols))
            self.assertGreater(streams.portfolio_snapshot.equity, 0.0)


if __name__ == "__main__":
    unittest.main()
