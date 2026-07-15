from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import yaml

from puppy.common.schemas import (
    AgentContext,
    CausalRunContext,
    MarketStep,
    PendingDecision,
    PriceBar,
)
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.normalization import FILING_SCHEMA, NEWS_SCHEMA, PRICE_SCHEMA, write_records
from puppy.graph.graph_dataset import FEATURE_NAMES, build_causal_graph_sample
from puppy.portfolio_ext.shadow_portfolio import ShadowPortfolio
from puppy.swarm.causal_orchestrator import CausalOrchestrator, build_context_hash
from puppy.swarm.legacy_finmem_adapter import ADAPTER_VERSION, LegacyFinMemAdapter
from puppy.swarm.sector_agent import SectorAgent
from puppy.training.artifact_generator import generate_training_artifacts
from puppy.training.artifact_validation import validate_training_artifacts
from puppy.training.observed_features import build_observed_feature_table
from puppy.training.outcome_resolver import resolve_outcome
from puppy.training.rollout_state import canonical_json, stable_id


UTC = dt.timezone.utc


class _ExplodingLegacyAgent:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        raise AssertionError("legacy LLMAgent.step must not be called")


class _CachedCausalAgent:
    model_name = "local-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def causal_decide(self, market_step, agent_context, run_context):
        self.calls += 1
        return {"investment_decision": "buy", "summary_reason": "cached causal response"}


class Stage5CausalContractTest(unittest.TestCase):
    def test_sector_decide_never_calls_legacy_step_and_context_precedes_output(self) -> None:
        date = dt.date(2026, 1, 2)
        legacy = _ExplodingLegacyAgent()
        adapter = LegacyFinMemAdapter(legacy, "sector_tsla", "TSLA", bootstrap=True)
        agent = SectorAgent("sector_tsla", "TSLA", legacy, causal_adapter=adapter)
        step = _market_step(date)
        run = _run_context(date, step.as_of)
        adapter.advance_memory_clock(date, run)
        adapter.ingest_market_events(step, run)
        bundle = CausalOrchestrator().decide(
            agent=agent,
            market_step=step,
            run_context=run,
            expected_outcome_date=dt.date(2026, 1, 5),
            split="train",
        )
        self.assertEqual(legacy.step_calls, 0)
        self.assertTrue(bundle.context.metadata["built_before_decision"])
        self.assertNotIn(bundle.output.signal.reason, bundle.text_feature.raw_context)
        self.assertEqual(bundle.context.context_hash, bundle.text_feature.context_hash)
        self.assertEqual(bundle.context.context_hash, bundle.pending.context_hash)

    def test_memory_routing_idempotence_feedback_and_decay_boundaries(self) -> None:
        date = dt.date(2026, 1, 2)
        adapter = LegacyFinMemAdapter(SimpleNamespace(), "sector_tsla", "TSLA", bootstrap=True)
        run = _run_context(date, _cutoff(date))
        step = _market_step(date, filing_types=("8-K", "10-Q", "10-K"))
        adapter.advance_memory_clock(date, run)
        with self.assertRaises(RuntimeError):
            adapter.advance_memory_clock(date, run)
        refs = adapter.ingest_market_events(step, run)
        self.assertEqual(adapter.decay_count, 1)
        self.assertEqual({str(ref.layer) for ref in refs}, {"short", "mid", "long"})
        self.assertEqual(adapter.ingest_market_events(step, run), [])
        self.assertTrue(all(ref.source_event_id and ref.available_at for ref in refs))
        self.assertEqual(adapter.feedback_count, 0)

    def test_learning_requires_outcome_and_uses_stored_snapshot_only(self) -> None:
        decision_date = dt.date(2026, 1, 2)
        outcome_date = dt.date(2026, 1, 5)
        adapter = LegacyFinMemAdapter(SimpleNamespace(), "sector_tsla", "TSLA", bootstrap=True)
        pending = _pending(decision_date, outcome_date)
        run = _run_context(outcome_date, _cutoff(outcome_date))
        with self.assertRaisesRegex(ValueError, "realized outcome"):
            adapter.learn_from_outcome(pending, None, run)
        outcome = resolve_outcome(
            pending,
            PriceBar(symbol="TSLA", date=outcome_date, open=100, close=110),
            outcome_date=outcome_date,
            available_at=run.current_cutoff,
        )
        reflection = adapter.learn_from_outcome(pending, outcome, run)
        self.assertEqual(reflection.source_decision_date, decision_date)
        self.assertEqual(reflection.available_at, run.current_cutoff)
        self.assertIn("decision_context_snapshot_sha256", reflection.text)
        self.assertNotIn("new t+1 headline", reflection.text)
        self.assertEqual(adapter.feedback_count, len(pending.memory_refs))

    def test_reflection_becomes_available_only_in_later_cycle(self) -> None:
        source_date = dt.date(2026, 1, 2)
        outcome_date = dt.date(2026, 1, 5)
        adapter = LegacyFinMemAdapter(SimpleNamespace(), "sector_tsla", "TSLA", bootstrap=True)
        pending = _pending(source_date, outcome_date)
        run = _run_context(outcome_date, _cutoff(outcome_date))
        outcome = resolve_outcome(
            pending, PriceBar(symbol="TSLA", date=outcome_date, open=100, close=102),
            outcome_date=outcome_date, available_at=run.current_cutoff,
        )
        reflection = adapter.learn_from_outcome(pending, outcome, run)
        before, _ = adapter.retrieve(_cutoff(source_date))
        self.assertEqual(before["reflection"], [])
        adapter.ingest_reflection(reflection, run)
        later, refs = adapter.retrieve(run.current_cutoff)
        self.assertEqual(len(later["reflection"]), 1)
        self.assertEqual(refs[-1].date, source_date)

    def test_non_bootstrap_decision_cache_and_force_llm(self) -> None:
        date = dt.date(2026, 1, 2)
        llm = _CachedCausalAgent()
        adapter = LegacyFinMemAdapter(llm, "sector_tsla", "TSLA", bootstrap=False)
        step = _market_step(date)
        context = AgentContext(agent_id="sector_tsla", symbol="TSLA", date=date, as_of=step.as_of, context_hash="a" * 64)
        run = _run_context(date, step.as_of).model_copy(update={"bootstrap": False})
        first = adapter.decide(step, context, run)
        second = adapter.decide(step, context, run)
        self.assertEqual(llm.calls, 1)
        self.assertFalse(first.metadata["cache_hit"])
        self.assertTrue(second.metadata["cache_hit"])
        forced = run.model_copy(update={"metadata": {"force_llm": True, "generation_parameters": {"temperature": 0.0}}})
        adapter.decide(step, context, forced)
        self.assertEqual(llm.calls, 2)

    def test_outcome_formula_next_common_date_threshold_and_shadow_timing(self) -> None:
        friday = dt.date(2026, 1, 2)
        monday = dt.date(2026, 1, 5)
        pending = _pending(friday, monday)
        neutral = resolve_outcome(
            pending,
            PriceBar(symbol="TSLA", date=monday, open=100, close=100.05),
            outcome_date=monday,
            available_at=_cutoff(monday),
        )
        self.assertEqual(neutral.direction_label, 0)
        self.assertFalse(neutral.train_mask)
        positive = resolve_outcome(
            pending,
            PriceBar(symbol="TSLA", date=monday, open=100, close=101),
            outcome_date=monday,
            available_at=_cutoff(monday),
        )
        self.assertAlmostEqual(positive.asset_return, 0.01)
        self.assertEqual(positive.direction_label, 1)
        with self.assertRaisesRegex(ValueError, "next common"):
            resolve_outcome(
                pending,
                PriceBar(symbol="TSLA", date=dt.date(2026, 1, 3), open=100, close=101),
                outcome_date=dt.date(2026, 1, 3),
                available_at=_cutoff(monday),
            )
        portfolio = ShadowPortfolio()
        self.assertIsNone(portfolio.get_state("sector_tsla", "TSLA"))
        state = portfolio.resolve_outcome(
            agent_id="sector_tsla", symbol="TSLA", action_signal=1,
            realized_return=positive.asset_return, decision_date=friday,
            outcome_date=monday, as_of=_cutoff(monday),
        )
        self.assertEqual(state.last_resolved_decision_date, friday)
        self.assertEqual(state.last_outcome_date, monday)
        self.assertEqual(state.number_of_resolved_decisions, 1)

    def test_future_price_perturbation_leaves_decision_time_artifacts_unchanged(self) -> None:
        t = dt.date(2026, 1, 2)
        nxt = dt.date(2026, 1, 5)
        step = _market_step(t)
        histories = {
            "TSLA": [step.prices["TSLA"], PriceBar(symbol="TSLA", date=nxt, open=1, close=1000)],
        }
        observed_a = build_observed_feature_table(step, histories)
        histories["TSLA"][-1] = PriceBar(symbol="TSLA", date=nxt, open=1000, close=1)
        observed_b = build_observed_feature_table(step, histories)
        self.assertEqual(canonical_json(observed_a), canonical_json(observed_b))

        adapter = LegacyFinMemAdapter(SimpleNamespace(), "sector_tsla", "TSLA", bootstrap=True)
        agent = SectorAgent("sector_tsla", "TSLA", SimpleNamespace(), causal_adapter=adapter)
        run = _run_context(t, step.as_of)
        adapter.advance_memory_clock(t, run)
        bundle = CausalOrchestrator().decide(
            agent=agent, market_step=step, run_context=run,
            expected_outcome_date=nxt, split="train",
        )
        graph_a = build_causal_graph_sample(
            observed_a, {"TSLA": bundle.output.signal.action_signal}, {"TSLA": None},
            {"TSLA": bundle.text_feature}, {"TSLA": []}, {"TSLA": []},
        )
        graph_b = build_causal_graph_sample(
            observed_b, {"TSLA": bundle.output.signal.action_signal}, {"TSLA": None},
            {"TSLA": bundle.text_feature}, {"TSLA": []}, {"TSLA": []},
        )
        np.testing.assert_array_equal(graph_a["feature_matrix"], graph_b["feature_matrix"])
        before = canonical_json({
            "context": bundle.context,
            "context_hash": bundle.context.context_hash,
            "text": bundle.text_feature,
            "observed": observed_a,
            "output_input": bundle.output.context_hash,
            "graph": graph_a["feature_matrix"],
            "pending": bundle.pending,
        })
        after = canonical_json({
            "context": bundle.context,
            "context_hash": bundle.context.context_hash,
            "text": bundle.text_feature,
            "observed": observed_b,
            "output_input": bundle.output.context_hash,
            "graph": graph_b["feature_matrix"],
            "pending": bundle.pending,
        })
        self.assertEqual(before, after)
        outcome_a = resolve_outcome(bundle.pending, PriceBar(symbol="TSLA", date=nxt, open=1, close=1000), outcome_date=nxt, available_at=_cutoff(nxt))
        outcome_b = resolve_outcome(bundle.pending, PriceBar(symbol="TSLA", date=nxt, open=1000, close=1), outcome_date=nxt, available_at=_cutoff(nxt))
        self.assertNotEqual(outcome_a.asset_return, outcome_b.asset_return)
        portfolio_a = ShadowPortfolio()
        portfolio_b = ShadowPortfolio()
        state_a = portfolio_a.resolve_outcome(agent_id="sector_tsla", symbol="TSLA", action_signal=1, realized_return=outcome_a.asset_return, decision_date=t, outcome_date=nxt, as_of=_cutoff(nxt))
        state_b = portfolio_b.resolve_outcome(agent_id="sector_tsla", symbol="TSLA", action_signal=1, realized_return=outcome_b.asset_return, decision_date=t, outcome_date=nxt, as_of=_cutoff(nxt))
        self.assertNotEqual(state_a.value, state_b.value)
        reflection_a = adapter.learn_from_outcome(bundle.pending, outcome_a, _run_context(nxt, _cutoff(nxt)))
        reflection_b = adapter.learn_from_outcome(bundle.pending, outcome_b, _run_context(nxt, _cutoff(nxt)))
        self.assertNotEqual(reflection_a.text, reflection_b.text)
        run_next = _run_context(nxt, _cutoff(nxt))
        next_step_a = MarketStep(
            date=nxt, symbols=["TSLA"],
            prices={"TSLA": PriceBar(symbol="TSLA", date=nxt, open=1, high=1000, low=1, close=1000)},
            news={"TSLA": []}, filings={"TSLA": []}, as_of=run_next.current_cutoff,
        )
        next_step_b = MarketStep(
            date=nxt, symbols=["TSLA"],
            prices={"TSLA": PriceBar(symbol="TSLA", date=nxt, open=1000, high=1000, low=1, close=1)},
            news={"TSLA": []}, filings={"TSLA": []}, as_of=run_next.current_cutoff,
        )
        adapter_a = LegacyFinMemAdapter(SimpleNamespace(), "sector_tsla", "TSLA", bootstrap=True)
        adapter_b = LegacyFinMemAdapter(SimpleNamespace(), "sector_tsla", "TSLA", bootstrap=True)
        adapter_a.advance_memory_clock(nxt, run_next)
        adapter_b.advance_memory_clock(nxt, run_next)
        adapter_a.ingest_reflection(reflection_a, run_next)
        adapter_b.ingest_reflection(reflection_b, run_next)
        later_a = CausalOrchestrator().decide(
            agent=SectorAgent("sector_tsla", "TSLA", SimpleNamespace(), causal_adapter=adapter_a),
            market_step=next_step_a, run_context=run_next,
            expected_outcome_date=dt.date(2026, 1, 6), split="train", shadow_state=state_a,
        )
        later_b = CausalOrchestrator().decide(
            agent=SectorAgent("sector_tsla", "TSLA", SimpleNamespace(), causal_adapter=adapter_b),
            market_step=next_step_b, run_context=run_next,
            expected_outcome_date=dt.date(2026, 1, 6), split="train", shadow_state=state_b,
        )
        self.assertNotEqual(later_a.context.context_hash, later_b.context.context_hash)

    def test_production_graph_contract_excludes_targets_and_checks_source_dates(self) -> None:
        self.assertFalse({"return", "label", "asset_return", "direction_label"}.intersection(FEATURE_NAMES))
        date = dt.date(2026, 1, 2)
        step = _market_step(date)
        observed = build_observed_feature_table(step, {"TSLA": [step.prices["TSLA"]]})
        text = SimpleNamespace(h_text=[0.1, 0.2])
        sample = build_causal_graph_sample(
            observed, {"TSLA": 1}, {"TSLA": None}, {"TSLA": text},
            {"TSLA": [0.01]}, {"TSLA": [date]},
        )
        self.assertEqual(sample["feature_matrix"].shape[1], len(FEATURE_NAMES))
        self.assertNotIn("labels", sample)
        self.assertFalse(sample["metadata"]["targets_in_feature_matrix"])
        with self.assertRaisesRegex(ValueError, "future source"):
            build_causal_graph_sample(
                observed, {"TSLA": 1}, {"TSLA": None}, {"TSLA": text},
                {"TSLA": [0.01]}, {"TSLA": [dt.date(2026, 1, 5)]},
            )

    def test_stable_ids_are_deterministic(self) -> None:
        self.assertEqual(stable_id("run", "agent", "date"), stable_id("run", "agent", "date"))
        self.assertNotEqual(stable_id("run", "agent", "date"), stable_id("run", "agent", "other"))


class Stage5ArtifactGeneratorTest(unittest.TestCase):
    def test_thirty_day_rollout_resume_mismatch_and_end_of_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, dates = _synthetic_repository(root, 31)
            output_root = root / "data/training"
            first = generate_training_artifacts(
                str(config_path), start=dates[0].isoformat(), end=dates[29].isoformat(),
                bootstrap=True, overwrite=True, output_root=output_root,
            )
            self.assertEqual(first["artifact_counts"]["agent_outputs"], 90)
            self.assertEqual(first["artifact_counts"]["outcomes"], 90)
            self.assertEqual(first["artifact_counts"]["graph_samples"], 30)
            artifact_dir = output_root / "finswarm_v1"
            validation = _validate_at_root(config_path, artifact_dir)
            self.assertEqual(validation["status"], "PASS")
            ids_before = pq.read_table(artifact_dir / "agent_outputs.parquet", columns=["decision_id"]).to_pylist()
            resumed = generate_training_artifacts(
                str(config_path), start=dates[0].isoformat(), end=dates[29].isoformat(),
                bootstrap=True, resume=True, output_root=output_root,
            )
            ids_after = pq.read_table(artifact_dir / "agent_outputs.parquet", columns=["decision_id"]).to_pylist()
            self.assertEqual(ids_before, ids_after)
            self.assertEqual(resumed["resume"]["duplicated_rows"], 0)
            outcomes = pq.read_table(artifact_dir / "outcomes.parquet").to_pylist()
            self.assertEqual(sum(bool(row["cross_split_outcome"]) for row in outcomes), 0)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                generate_training_artifacts(
                    str(config_path), start=dates[0].isoformat(), end=dates[28].isoformat(),
                    bootstrap=True, resume=True, output_root=output_root,
                )
            full = generate_training_artifacts(
                str(config_path), start=dates[0].isoformat(), end=dates[30].isoformat(),
                bootstrap=True, overwrite=True, output_root=output_root,
            )
            self.assertEqual(full["artifact_counts"]["agent_outputs"], 93)
            self.assertEqual(full["artifact_counts"]["outcomes"], 90)
            pending = pq.read_table(artifact_dir / "pending_decisions.parquet").to_pylist()
            self.assertEqual(sum(row["status"] == "unresolved_end_of_data" for row in pending), 3)


def _cutoff(date: dt.date) -> dt.datetime:
    return dt.datetime.combine(date, dt.time(21, 10), UTC)


def _run_context(date: dt.date, cutoff: dt.datetime) -> CausalRunContext:
    return CausalRunContext(
        run_id="test", config_path="test.yaml", current_date=date,
        current_cutoff=cutoff, bootstrap=True, prompt_version="prompt-v1",
        encoder_version="encoder-v1", adapter_version=ADAPTER_VERSION,
        git_commit="test", memory_sharing_enabled=False, debate_enabled=False,
        gat_enabled=False, fusion_enabled=False,
    )


def _market_step(date: dt.date, filing_types: tuple[str, ...] = ()) -> MarketStep:
    from puppy.common.schemas import FilingEvent, NewsEvent

    cutoff = _cutoff(date)
    filings = [
        FilingEvent(
            filing_id=f"filing-{form}", symbol="TSLA", filing_type=form,
            filing_date=date, accepted_at=cutoff - dt.timedelta(hours=1),
            available_at=cutoff - dt.timedelta(hours=1), text=f"{form} filing",
        )
        for form in filing_types
    ]
    return MarketStep(
        date=date, symbols=["TSLA"],
        prices={"TSLA": PriceBar(symbol="TSLA", date=date, open=100, high=105, low=99, close=102, volume=1000)},
        news={"TSLA": [NewsEvent(event_id="news-1", symbol="TSLA", published_at=cutoff - dt.timedelta(hours=2), available_at=cutoff - dt.timedelta(hours=2), text="news text")]},
        filings={"TSLA": filings}, as_of=cutoff,
    )


def _pending(decision_date: dt.date, outcome_date: dt.date) -> PendingDecision:
    return PendingDecision(
        decision_id="decision-1", sample_id="sample-1", run_id="test",
        agent_id="sector_tsla", symbol="TSLA", decision_date=decision_date,
        decision_cutoff=_cutoff(decision_date), expected_outcome_date=outcome_date,
        decision="buy", action_signal=1, reason="prior reason", context_hash="abc",
        context_snapshot={"frozen": "decision-time", "not_future": True}, memory_refs=[],
        text_sample_id="text-1", prompt_version="p", encoder_version="e",
        adapter_version="a", status="pending",
    )


def _trading_dates(count: int) -> list[dt.date]:
    values = []
    date = dt.date(2026, 1, 2)
    while len(values) < count:
        if date.weekday() < 5:
            values.append(date)
        date += dt.timedelta(days=1)
    return values


def _synthetic_repository(root: Path, count: int) -> tuple[Path, list[dt.date]]:
    dates = _trading_dates(count)
    config_path = root / "config/swarm/experiment_v1.yaml"
    config_path.parent.mkdir(parents=True)
    manifest_path = root / "data/manifests/price_split_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "splits": {
            "warmup": {"dates": [date.isoformat() for date in dates]},
            "train": {"dates": []}, "validation": {"dates": []}, "test": {"dates": []},
        }
    }), encoding="utf-8")
    config = {
        "run": {"run_id": "finswarm_v1"},
        "assets": {"symbols": ["TSLA", "NVDA", "AAPL"]},
        "time": {"timezone": "America/New_York", "decision_cutoff": "16:10:00"},
        "data": {
            "start": dates[0].isoformat(), "end": dates[-1].isoformat(),
            "split_manifest": "data/manifests/price_split_manifest.json",
            "raw": {"prices": "data/raw/prices", "news": "data/raw/news", "filings": "data/raw/filings"},
            "canonical": {"prices": "data/parquet/prices.parquet", "news": "data/parquet/news.parquet", "filings": "data/parquet/filings.parquet"},
            "duckdb": "data/duckdb/finswarm.db",
        },
        "normalization": {"filings": {"include_8k_exhibits": ["EX-99", "EX-99.1", "EX-99.2"]}},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    price_rows = []
    for symbol_index, symbol in enumerate(("TSLA", "NVDA", "AAPL")):
        for index, date in enumerate(dates):
            price = 100.0 + symbol_index * 10 + index
            price_rows.append({
                "symbol": symbol, "date": date, "open": price, "high": price + 2,
                "low": price - 1, "close": price + 1, "adj_close": None, "volume": 1000 + index,
                "source": "synthetic", "provider_record_id": f"{symbol}-{date}", "metadata": "{}",
            })
    parquet_dir = root / "data/parquet"
    write_records(parquet_dir / "prices.parquet", price_rows, PRICE_SCHEMA)
    write_records(parquet_dir / "news.parquet", [], NEWS_SCHEMA)
    write_records(parquet_dir / "filings.parquet", [], FILING_SCHEMA)
    db_path = root / "data/duckdb/finswarm.db"
    store = DuckDBMarketStore(str(db_path))
    store.connect()
    store.ingest_prices(price_rows)
    store.close()
    return config_path, dates


def _validate_at_root(config_path: Path, artifact_dir: Path) -> dict:
    expected = config_path.parents[2] / "data/training/finswarm_v1"
    if expected != artifact_dir:
        raise AssertionError("Synthetic artifact root must match production layout")
    return validate_training_artifacts(
        str(config_path), "finswarm_v1", require_final=False
    )


if __name__ == "__main__":
    unittest.main()
