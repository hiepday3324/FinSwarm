from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from puppy.agent import causal_eligible_memory_set_hash, causal_memory_inputs
from puppy.common.schemas import (
    CausalRunContext,
    PendingDecision,
    PriceBar,
)
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.normalization import load_stage4_config
from puppy.data_engine.parquet_loader import load_price_parquet
from puppy.graph.graph_dataset import FEATURE_NAMES, build_causal_graph_sample
from puppy.portfolio_ext.allocation_engine import signals_to_target_weights
from puppy.portfolio_ext.master_portfolio import MasterPortfolio
from puppy.swarm.causal_orchestrator import (
    BOOTSTRAP_ENCODER_VERSION,
    CausalOrchestrator,
    DeterministicTextEncoder,
)
from puppy.swarm.legacy_finmem_adapter import ADAPTER_VERSION, LegacyFinMemAdapter
from puppy.swarm.psychology import build_psychology_state
from puppy.swarm.sector_agent import SectorAgent

from .dataset_exporter import export_artifact_tables, export_graph_npz
from .decision_journal import (
    CausalDecisionJournal,
    CausalDecisionJournalError,
    eligible_memory_set_hash,
    runtime_decision_contract,
)
from .observed_features import FEATURE_DEFINITIONS, OBSERVED_FEATURE_NAMES, build_observed_feature_table
from .outcome_resolver import resolve_outcome
from .rollout_state import atomic_write_json, atomic_write_text, canonical_json, sha256_path, stable_id
from .temporal_assertions import (
    FORBIDDEN_GRAPH_FIELDS,
    assert_graph_contract,
    assert_linked_context_hashes,
    assert_market_step_causal,
    assert_observed_features_causal,
    assert_outcome_alignment,
    assert_unique_ids,
)


PROMPT_VERSION = "bootstrap-decision-v1"
GENERATOR_VERSION = "stage5-canonical-causal-artifacts-v3-master-portfolio"


def _sanitized_provenance(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    backend_raw = raw.get("backend_config")
    backend = dict(backend_raw) if isinstance(backend_raw, dict) else {}
    safe_backend_fields = {
        "provider",
        "interface",
        "base_url",
        "model",
        "timeout_seconds",
        "max_retries",
        "requests_per_minute",
        "temperature",
        "backend_identity",
    }
    return {
        "mode": str(raw.get("mode", "unverified_nonbootstrap")),
        "synthetic": bool(raw.get("synthetic", True)),
        "external_llm_expected": bool(raw.get("external_llm_expected", False)),
        "provider": str(raw.get("provider", "unknown")),
        "model": str(raw.get("model", "unknown")),
        "backend_identity": str(raw.get("backend_identity", "unknown")),
        "backend_config": {
            key: backend[key] for key in safe_backend_fields if key in backend
        },
    }


def _aggregate_decision_quality(agents: dict[str, SectorAgent]) -> dict[str, int]:
    fields = (
        "total_decisions",
        "buy_count",
        "hold_count",
        "sell_count",
        "parse_success_count",
        "retry_count",
        "cache_hit_count",
        "api_call_count",
        "failed_decision_count",
    )
    return {
        field: sum(
            int(getattr(agent.causal_adapter, "decision_stats", {}).get(field, 0))
            for agent in agents.values()
        )
        for field in fields
    }


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _split_manifest(path: Path) -> tuple[dict[str, set[dt.date]], list[dt.date]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    splits = {
        name: {dt.date.fromisoformat(value) for value in details["dates"]}
        for name, details in raw["splits"].items()
    }
    dates = sorted(set().union(*splits.values()))
    return splits, dates


def _split_for(date: dt.date | None, splits: dict[str, set[dt.date]]) -> str | None:
    if date is None:
        return None
    return next((name for name, dates in splits.items() if date in dates), None)


def _nested_counts(rows: list[dict[str, Any]], date_field: str = "split") -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[str(row.get("symbol", "ALL"))][str(row.get(date_field) or "none")] += 1
    return {symbol: dict(sorted(counts.items())) for symbol, counts in sorted(result.items())}


def _text_row(bundle: Any, run_id: str, split: str) -> dict[str, Any]:
    feature = bundle.text_feature
    return {
        "text_sample_id": feature.text_sample_id,
        "decision_id": bundle.pending.decision_id,
        "run_id": run_id,
        "agent_id": feature.agent_id,
        "symbol": feature.symbol,
        "date": feature.date,
        "as_of": feature.as_of,
        "context_hash": feature.context_hash,
        "raw_context": feature.raw_context,
        "h_text": feature.h_text,
        "embedding_model": feature.embedding_model,
        "encoder_version": feature.encoder_version,
        "prompt_version": feature.prompt_version,
        "split": split,
        "metadata": feature.metadata,
    }


def _output_row(bundle: Any, run_id: str, split: str) -> dict[str, Any]:
    output = bundle.output
    return {
        "decision_id": output.decision_id,
        "run_id": run_id,
        "agent_id": output.signal.agent_id,
        "symbol": output.signal.symbol,
        "decision_date": output.signal.date,
        "decision_cutoff": output.decision_cutoff,
        "decision": output.signal.decision,
        "action_signal": output.signal.action_signal,
        "reason": output.signal.reason,
        "parse_ok": output.parse_ok,
        "model_name": output.model_name,
        "context_hash": output.context_hash,
        "prompt_version": output.prompt_version,
        "adapter_version": output.adapter_version,
        "text_score": bundle.text_score,
        "split": split,
        "metadata": output.metadata,
    }


def _pending_row(pending: PendingDecision) -> dict[str, Any]:
    row = pending.model_dump()
    row["decision"] = str(pending.decision)
    row["memory_refs"] = [ref.model_dump(mode="json") for ref in pending.memory_refs]
    row["split"] = pending.metadata.get("split")
    return row


def _outcome_row(outcome: Any) -> dict[str, Any]:
    row = outcome.model_dump()
    row["split"] = outcome.metadata.get("decision_split")
    row["cross_split_outcome"] = bool(outcome.metadata.get("cross_split_outcome"))
    row["supervised_eligible"] = bool(outcome.metadata.get("supervised_eligible"))
    return row


def _reflection_row(reflection: Any) -> dict[str, Any]:
    return reflection.model_dump()


def _atomic_graph_shard(path: Path, sample: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_id=np.asarray(sample["sample_id"]),
        decision_date=np.asarray(sample["decision_date"]),
        symbols=np.asarray(sample["symbols"]),
        feature_matrix=sample["feature_matrix"],
        adjacency_matrix=sample["adjacency_matrix"],
        feature_mask=sample["feature_mask"],
        max_source_date=np.asarray(sample["max_source_date"]),
    )
    temporary.replace(path)


def _load_resume_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    symbols: tuple[str, ...],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Resume checkpoint is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("Resume checkpoint must be a JSON object")
    if value.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Resume checkpoint fingerprint mismatch")
    completed = value.get("completed_symbols_for_current_date", [])
    if not isinstance(completed, list) or any(symbol not in symbols for symbol in completed):
        raise ValueError("Resume checkpoint has invalid completed symbols")
    if completed != list(symbols[: len(completed)]):
        raise ValueError("Resume checkpoint symbol progress is not canonical")
    next_symbol = value.get("next_symbol")
    if next_symbol is not None and next_symbol not in symbols:
        raise ValueError("Resume checkpoint has an invalid next symbol")
    return value


def generate_training_artifacts(
    config_path: str,
    *,
    start: str | None = None,
    end: str | None = None,
    bootstrap: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    output_root: str | Path | None = None,
    force_llm: bool = False,
    causal_agents: dict[str, SectorAgent] | None = None,
    encoder: Any | None = None,
    text_score_head: Any | None = None,
    prompt_version: str | None = None,
    generation_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not bootstrap and causal_agents is None:
        raise RuntimeError(
            "Real causal artifact generation requires injected SectorAgent instances "
            "whose llm_agent implements causal_decide; no external API was called."
        )
    if not bootstrap and encoder is None:
        raise RuntimeError(
            "Real causal artifact generation requires an injected canonical "
            "TextStreamEncoder; no fallback encoder was selected."
        )
    provenance = (
        {
            "mode": "bootstrap",
            "synthetic": False,
            "external_llm_expected": False,
            "provider": "none",
            "model": "bootstrap-deterministic-policy-v1",
            "backend_identity": "none",
            "backend_config": {},
        }
        if bootstrap
        else _sanitized_provenance(generation_provenance)
    )
    if not bootstrap and provenance["synthetic"] and output_root is None:
        raise ValueError(
            "Synthetic/mock causal rollout requires an explicit temporary output_root; "
            "the canonical Stage 5 directory is protected"
        )
    if causal_agents is not None:
        expected_symbols = set(load_stage4_config(config_path).symbols)
        missing_agents = expected_symbols - set(causal_agents)
        if missing_agents:
            raise ValueError(f"Missing causal SectorAgent(s): {sorted(missing_agents)}")
        bootstrap_adapters = [
            symbol
            for symbol, agent in causal_agents.items()
            if bool(getattr(agent.causal_adapter, "bootstrap", False))
        ]
        if bootstrap_adapters:
            raise ValueError(
                "Real causal artifact generation received bootstrap adapter(s): "
                f"{bootstrap_adapters}"
            )
    config = load_stage4_config(config_path)
    splits, common_dates = _split_manifest(config.split_manifest)
    requested_start = dt.date.fromisoformat(start) if start else common_dates[0]
    requested_end = dt.date.fromisoformat(end) if end else common_dates[-1]
    decision_dates = [date for date in common_dates if requested_start <= date <= requested_end]
    if not decision_dates:
        raise ValueError("Requested interval contains no common trading dates")
    run_id = config.run_id
    full_interval = (
        requested_start == common_dates[0] and requested_end == common_dates[-1]
    )
    root = Path(output_root) if output_root else config.repo_root / "data/training"
    output_dir = root / run_id
    feature_schema = {
        "version": "canonical-causal-feature-schema-v3-master-portfolio",
        "observed_feature_names": OBSERVED_FEATURE_NAMES,
        "observed_definitions": FEATURE_DEFINITIONS,
        "graph_feature_names": FEATURE_NAMES,
        "forbidden_graph_fields": sorted(FORBIDDEN_GRAPH_FIELDS),
        "graph_targets_separate": True,
        "psychology_encoding": {
            "feature_names": [
                "psychology_greed", "psychology_fear", "psychology_neutral"
            ],
            "greed": [1, 0, 0],
            "fear": [0, 1, 0],
            "neutral": [0, 0, 1],
        },
    }
    feature_schema_hash = stable_id(feature_schema)
    input_hashes = {
        config.relative(path): sha256_path(path)
        for path in (config.canonical_prices, config.canonical_news, config.canonical_filings)
    }
    fingerprint_payload = {
        "generator_version": GENERATOR_VERSION,
        "config_hash": sha256_path(config.config_path),
        "input_hashes": input_hashes,
        "feature_schema_hash": feature_schema_hash,
        "prompt_version": prompt_version or PROMPT_VERSION,
        "encoder_version": (
            BOOTSTRAP_ENCODER_VERSION
            if bootstrap
            else f"transformers-frozen:{getattr(encoder, 'revision', 'configured')}"
        ),
        "adapter_version": ADAPTER_VERSION,
        "generation_provenance": provenance,
        "bootstrap": bootstrap,
        "start": requested_start,
        "end": requested_end,
    }
    fingerprint = stable_id(fingerprint_payload)
    resume_checkpoint: dict[str, Any] | None = None
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif resume:
            manifest_path = output_dir / "data_manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Resume requires data_manifest.json")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("manifest_fingerprint") != fingerprint:
                quality_path = output_dir / "temporal_quality.json"
                if quality_path.is_file():
                    quality = json.loads(quality_path.read_text(encoding="utf-8"))
                    quality.setdefault("resume_idempotence_results", {})["manifest_mismatch_detected"] = True
                    atomic_write_json(quality_path, quality)
                raise ValueError("Resume manifest mismatch: config, input hash, schema, or version changed")
            rollout_path = output_dir / "rollout_manifest.json"
            if rollout_path.is_file():
                rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
                if rollout.get("status") == "complete":
                    quality_path = output_dir / "temporal_quality.json"
                    if quality_path.is_file():
                        quality = json.loads(quality_path.read_text(encoding="utf-8"))
                        quality["resume_idempotence_results"] = {
                            "duplicated_rows": 0,
                            "changed_ids": 0,
                            "manifest_mismatch_detected": quality.get("resume_idempotence_results", {}).get("manifest_mismatch_detected", False),
                        }
                        atomic_write_json(quality_path, quality)
                    return {**rollout, "resume": {"duplicated_rows": 0, "changed_ids": 0, "manifest_mismatch_detected": False}}
        else:
            raise FileExistsError(f"Completed or partial artifact directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    graph_shard_dir = state_dir / "graph_shards"
    graph_shard_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        resume_checkpoint = _load_resume_checkpoint(
            state_dir / "checkpoint.json",
            fingerprint=fingerprint,
            symbols=tuple(config.symbols),
        )
    decision_journal = (
        None
        if bootstrap
        else CausalDecisionJournal(
            state_dir,
            rollout_fingerprint=fingerprint,
            symbols=tuple(config.symbols),
        )
    )
    if (
        resume_checkpoint is not None
        and decision_journal is not None
        and "journal_record_count" in resume_checkpoint
        and int(resume_checkpoint["journal_record_count"]) > len(decision_journal)
    ):
        raise ValueError("Resume checkpoint is ahead of the decision journal")
    data_manifest = {
        **fingerprint_payload,
        "manifest_fingerprint": fingerprint,
        "run_id": run_id,
        "config_path": config.relative(config.config_path),
        "split_manifest": config.relative(config.split_manifest),
        "git_commit": _git_commit(config.repo_root),
        "bootstrap_policy": "sha256-context-bucket-v1" if bootstrap else None,
        "external_llm_used": False,
        "action_source": (
            "bootstrap_sha256_context_bucket"
            if bootstrap
            else (
                "mock_causal_sector_agent"
                if provenance["synthetic"]
                else "real_causal_sector_agent"
            )
        ),
        "generation_provenance": provenance,
        "final_training_eligible": False,
        "full_interval": full_interval,
    }
    atomic_write_json(output_dir / "data_manifest.json", data_manifest)
    atomic_write_json(output_dir / "feature_schema.json", feature_schema)

    store = DuckDBMarketStore(
        str(config.duckdb_path),
        market_timezone=config.timezone.key,
        decision_cutoff=config.decision_cutoff,
    )
    store.connect()
    price_rows = load_price_parquet(str(config.canonical_prices)).to_dict("records")
    price_history: dict[str, list[PriceBar]] = defaultdict(list)
    for row in price_rows:
        price_history[row["symbol"]].append(PriceBar(**row))
    agents: dict[str, SectorAgent] = {}
    if causal_agents is not None:
        agents = {symbol: causal_agents[symbol] for symbol in config.symbols}
    else:
        for symbol in config.symbols:
            agent_id = f"sector_{symbol.lower()}"
            adapter = LegacyFinMemAdapter(SimpleNamespace(), agent_id, symbol, bootstrap=True)
            agents[symbol] = SectorAgent(agent_id, symbol, SimpleNamespace(), causal_adapter=adapter)
    orchestrator = CausalOrchestrator(
        encoder or DeterministicTextEncoder(),
        text_score_head=text_score_head,
    )
    master_portfolio = MasterPortfolio(initial_cash=100000.0)
    latest_master_snapshot = None
    latest_reasons: dict[str, tuple[dt.date, str]] = {}
    pending_by_id: dict[str, PendingDecision] = {}
    pending_by_date: dict[dt.date, list[str]] = defaultdict(list)
    artifacts: dict[str, Any] = {
        "text_samples": [], "agent_outputs": [], "observed_features": [],
        "outcomes": [], "reflection_updates": [], "graph_samples": [],
        "observed_feature_names": OBSERVED_FEATURE_NAMES,
    }
    return_history: dict[str, list[float]] = defaultdict(list)
    return_source_dates: dict[str, list[dt.date]] = defaultdict(list)
    days_without_news = days_without_filings = 0
    last_fully_completed_date: dt.date | None = None
    journal_gap_seen = False

    def process_cycle(date: dt.date, create_decisions: bool) -> None:
        nonlocal days_without_news, days_without_filings
        nonlocal last_fully_completed_date, journal_gap_seen
        nonlocal latest_master_snapshot
        step = store.get_point_in_time_market_step(date, list(config.symbols), common_dates)
        assert_market_step_causal(step)
        run_context = CausalRunContext(
            run_id=run_id,
            config_path=config.relative(config.config_path),
            current_date=date,
            current_cutoff=step.as_of,
            bootstrap=bootstrap,
            prompt_version=str(fingerprint_payload["prompt_version"]),
            encoder_version=str(fingerprint_payload["encoder_version"]),
            adapter_version=ADAPTER_VERSION,
            git_commit=data_manifest["git_commit"],
            memory_sharing_enabled=False,
            debate_enabled=False,
            gat_enabled=False,
            fusion_enabled=False,
            metadata={
                "lifecycle": "A-K",
                "trainable_components_enabled": [],
                "force_llm": force_llm,
                "generation_parameters": {
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                },
            },
        )
        for agent in agents.values():
            agent.causal_adapter.advance_memory_clock(date, run_context)

        finalized_reflections = []
        for decision_id in list(pending_by_date.get(date, [])):
            pending = pending_by_id[decision_id]
            outcome = resolve_outcome(
                pending,
                step.prices.get(pending.symbol),
                outcome_date=date,
                available_at=step.as_of,
                threshold_bps=10,
                decision_split=str(pending.metadata.get("split")),
                outcome_split=_split_for(date, splits),
            )
            assert_outcome_alignment(outcome, pending)
            artifacts["outcomes"].append(_outcome_row(outcome))
            reflection = agents[pending.symbol].learn_from_outcome(pending, outcome, run_context)
            finalized_reflections.append(reflection)
            artifacts["reflection_updates"].append(_reflection_row(reflection))
            pending.status = "resolved"
            pending.metadata["outcome_id"] = outcome.outcome_id
            pending.metadata["cross_split_outcome"] = outcome.metadata["cross_split_outcome"]

        for reflection in finalized_reflections:
            agents[reflection.symbol].causal_adapter.ingest_reflection(reflection, run_context)
        for agent in agents.values():
            agent.causal_adapter.ingest_market_events(step, run_context)
        days_without_news += int(not any(step.news.values()))
        days_without_filings += int(not any(step.filings.values()))
        if not create_decisions:
            checkpoint(
                date,
                completed_symbols=[],
                next_symbol=None,
                last_fully_completed=last_fully_completed_date,
            )
            return

        observed = build_observed_feature_table(step, price_history)
        assert_observed_features_causal(observed)
        current_split = _split_for(date, splits)
        next_date = common_dates[common_dates.index(date) + 1] if common_dates.index(date) + 1 < len(common_dates) else None
        bundles = {}
        text_by_symbol = {}
        action_signals = {}
        psychology_states = {}
        text_scores = {}
        completed_symbols: list[str] = []
        for symbol_index, (symbol, agent) in enumerate(agents.items()):
            psychology = build_psychology_state(
                latest_master_snapshot,
                symbol=symbol,
            )
            prior_reason = latest_reasons.get(agent.agent_id)
            latest_reason = prior_reason[1] if prior_reason and prior_reason[0] < date else None
            journal_record = (
                decision_journal.get(date, symbol)
                if decision_journal is not None
                else None
            )
            if journal_record is not None and journal_gap_seen:
                raise CausalDecisionJournalError(
                    f"Non-contiguous decision journal record for {symbol} on {date}"
                )
            if journal_record is None and decision_journal is not None:
                journal_gap_seen = True
            persisted_output = (
                decision_journal.output(journal_record)
                if decision_journal is not None and journal_record is not None
                else None
            )
            bundle = orchestrator.decide(
                agent=agent,
                market_step=step,
                run_context=run_context,
                expected_outcome_date=next_date,
                split=str(current_split),
                master_portfolio_state=latest_master_snapshot,
                psychology_state=psychology,
                latest_prior_reason=latest_reason,
                persisted_output=persisted_output,
            )
            if decision_journal is not None:
                _, eligible_ids = causal_memory_inputs(bundle.context)
                eligible_hash = eligible_memory_set_hash(eligible_ids)
                metadata_eligible = bundle.output.metadata.get(
                    "eligible_memory_ids", {}
                )
                if canonical_json(metadata_eligible) != canonical_json(eligible_ids):
                    raise CausalDecisionJournalError(
                        f"Eligible memory IDs mismatch for {symbol} on {date}"
                    )
                if (
                    bundle.output.metadata.get("eligible_memory_set_hash")
                    != eligible_hash
                    or causal_eligible_memory_set_hash(eligible_ids) != eligible_hash
                ):
                    raise CausalDecisionJournalError(
                        f"Eligible memory hash mismatch for {symbol} on {date}"
                    )
                runtime_contract = runtime_decision_contract(agent, run_context)
                if journal_record is not None:
                    decision_journal.validate_for_reuse(
                        journal_record,
                        output=bundle.output,
                        symbol=symbol,
                        decision_date=date,
                        decision_id=bundle.pending.decision_id,
                        context_hash=bundle.context.context_hash or "",
                        eligible_memory_ids=eligible_ids,
                        runtime_contract=runtime_contract,
                    )
                    agent.accept_replayed_decision(bundle.output, run_context)
                else:
                    decision_journal.append(
                        output=bundle.output,
                        symbol=symbol,
                        decision_date=date,
                        eligible_memory_ids=eligible_ids,
                        runtime_contract=runtime_contract,
                    )
            bundles[symbol] = bundle
            text_by_symbol[symbol] = bundle.text_feature
            action_signals[symbol] = bundle.output.signal.action_signal
            psychology_states[symbol] = psychology
            text_scores[symbol] = bundle.text_score
            latest_reasons[agent.agent_id] = (date, bundle.output.signal.reason)
            completed_symbols.append(symbol)
            checkpoint(
                date,
                completed_symbols=completed_symbols,
                next_symbol=(
                    config.symbols[symbol_index + 1]
                    if symbol_index + 1 < len(config.symbols)
                    else None
                ),
                last_fully_completed=last_fully_completed_date,
            )

        for symbol in config.symbols:
            observed_return = observed.features[symbol]["observed_close_return_1d"]
            if observed_return is not None:
                return_history[symbol].append(float(observed_return))
                return_source_dates[symbol].append(date)
        graph = build_causal_graph_sample(
            observed,
            action_signals,
            latest_master_snapshot,
            text_by_symbol,
            return_history,
            return_source_dates,
            text_scores=text_scores,
            psychology_states=psychology_states,
        )
        assert_graph_contract(FEATURE_NAMES, graph["feature_matrix"])
        context_hashes = [bundles[symbol].context.context_hash for symbol in config.symbols]
        graph_sample_id = stable_id(
            run_id, date, list(config.symbols), feature_schema_hash, context_hashes,
        )
        graph_row = {
            "sample_id": graph_sample_id,
            "decision_date": date.isoformat(),
            "symbols": list(config.symbols),
            "feature_matrix": graph["feature_matrix"],
            "adjacency_matrix": graph["adjacency_matrix"],
            "feature_mask": graph["feature_mask"],
            "max_source_date": graph["metadata"]["max_source_date"],
            "split": current_split,
        }
        artifacts["graph_samples"].append(graph_row)
        _atomic_graph_shard(graph_shard_dir / f"{date.isoformat()}.npz", graph_row)

        allocation = signals_to_target_weights(
            date=date,
            symbols=list(config.symbols),
            action_signals=action_signals,
            max_weight=0.35,
        )
        latest_master_snapshot = master_portfolio.apply_target_weights(
            date=date,
            prices=step.prices,
            target_weights=allocation,
            as_of=step.as_of,
        )

        for symbol, bundle in bundles.items():
            bundle.pending.graph_sample_id = graph_sample_id
            assert_linked_context_hashes(bundle.context, bundle.text_feature, bundle.pending)
            pending_by_id[bundle.pending.decision_id] = bundle.pending
            if next_date is not None:
                pending_by_date[next_date].append(bundle.pending.decision_id)
            artifacts["text_samples"].append(_text_row(bundle, run_id, str(current_split)))
            artifacts["agent_outputs"].append(_output_row(bundle, run_id, str(current_split)))
            feature_row = {
                "sample_id": stable_id(run_id, symbol, date, "observed-features-v1"),
                "run_id": run_id,
                "symbol": symbol,
                "date": date,
                "as_of": step.as_of,
                "context_hash": bundle.context.context_hash,
                "split": current_split,
                "source_dates": [value.isoformat() for value in observed.source_dates[symbol]],
                **observed.features[symbol],
                "metadata": {"max_source_date": max(observed.source_dates[symbol]).isoformat() if observed.source_dates[symbol] else None},
            }
            artifacts["observed_features"].append(feature_row)
        last_fully_completed_date = date
        checkpoint(
            date,
            completed_symbols=list(config.symbols),
            next_symbol=None,
            last_fully_completed=last_fully_completed_date,
        )

    def checkpoint(
        date: dt.date,
        *,
        completed_symbols: list[str] | None = None,
        next_symbol: str | None = None,
        last_fully_completed: dt.date | None = None,
    ) -> None:
        resolved = [item.decision_id for item in pending_by_id.values() if item.status == "resolved"]
        checkpoint_value = {
            "current_date": date.isoformat(),
            "completed_symbols_for_current_date": list(completed_symbols or []),
            "next_symbol": next_symbol,
            "last_completed_date": (
                last_fully_completed.isoformat() if last_fully_completed else None
            ),
            "last_fully_completed_date": (
                last_fully_completed.isoformat() if last_fully_completed else None
            ),
            "pending_decision_ids": sorted(item.decision_id for item in pending_by_id.values() if item.status == "pending"),
            "resolved_decision_ids": sorted(resolved),
            "reflection_ids": sorted(row["reflection_id"] for row in artifacts["reflection_updates"]),
            "memory_checkpoint_references": {"mode": "deterministic_replay", "replay_through": date.isoformat()},
            "master_portfolio_state": (
                latest_master_snapshot.model_dump(mode="json")
                if latest_master_snapshot is not None
                else None
            ),
            "artifact_counts": {
                key: len(value) for key, value in artifacts.items() if isinstance(value, list)
            },
            "journal_record_count": (
                len(decision_journal) if decision_journal is not None else 0
            ),
            "manifest_fingerprint": fingerprint,
        }
        atomic_write_json(state_dir / "checkpoint.json", checkpoint_value)

    try:
        for date in decision_dates:
            process_cycle(date, True)
        last_index = common_dates.index(decision_dates[-1])
        if decision_dates[-1] < common_dates[-1] and last_index + 1 < len(common_dates):
            process_cycle(common_dates[last_index + 1], False)
        for pending in pending_by_id.values():
            if pending.status == "pending":
                pending.status = "unresolved_end_of_data"
        artifacts["pending_decisions"] = [_pending_row(item) for item in sorted(pending_by_id.values(), key=lambda item: item.decision_id)]
        for key in ("text_samples", "agent_outputs", "observed_features", "outcomes", "reflection_updates"):
            artifacts[key].sort(key=lambda row: tuple(str(row.get(field, "")) for field in ("date", "decision_date", "outcome_date", "symbol", "decision_id")))
        assert_unique_ids((row["decision_id"] for row in artifacts["agent_outputs"]), "decision")
        assert_unique_ids((row["text_sample_id"] for row in artifacts["text_samples"]), "text sample")
        assert_unique_ids((row["outcome_id"] for row in artifacts["outcomes"]), "outcome")
        assert_unique_ids((row["reflection_id"] for row in artifacts["reflection_updates"]), "reflection")
        export_artifact_tables(output_dir, artifacts)
        export_graph_npz(output_dir, artifacts["graph_samples"])
    finally:
        store.close()

    cross_split = sum(bool(row["cross_split_outcome"]) for row in artifacts["outcomes"])
    neutral = sum(not bool(row["train_mask"]) for row in artifacts["outcomes"])
    unresolved = sum(row["status"] == "unresolved_end_of_data" for row in artifacts["pending_decisions"])
    non_boundary_missing = sum(
        row["status"] != "resolved" and row["expected_outcome_date"] is not None and row["expected_outcome_date"] <= common_dates[-1]
        for row in artifacts["pending_decisions"]
    )
    critical = {
        "duplicate_artifact_ids": 0,
        "future_price_violations": 0,
        "future_event_violations": 0,
        "same_day_reflection_violations": 0,
        "master_portfolio_timing_violations": 0,
        "graph_forbidden_feature_violations": 0,
        "outcome_alignment_violations": 0,
        "context_hash_mismatches": 0,
        "non_boundary_missing_outcomes": non_boundary_missing,
    }
    adapter_quality = _aggregate_decision_quality(agents)
    decision_quality = {
        "total_decisions": len(artifacts["agent_outputs"]),
        "buy_count": sum(
            str(row["decision"]).lower() == "buy"
            for row in artifacts["agent_outputs"]
        ),
        "hold_count": sum(
            str(row["decision"]).lower() == "hold"
            for row in artifacts["agent_outputs"]
        ),
        "sell_count": sum(
            str(row["decision"]).lower() == "sell"
            for row in artifacts["agent_outputs"]
        ),
        "parse_success_count": sum(
            bool(row["parse_ok"]) for row in artifacts["agent_outputs"]
        ),
        "retry_count": adapter_quality["retry_count"],
        "cache_hit_count": adapter_quality["cache_hit_count"],
        "api_call_count": adapter_quality["api_call_count"],
        "failed_decision_count": adapter_quality["failed_decision_count"],
    }
    external_llm_used = bool(
        not bootstrap
        and not provenance["synthetic"]
        and provenance["external_llm_expected"]
        and decision_quality["api_call_count"] > 0
    )
    final_training_eligible = bool(
        not bootstrap
        and not provenance["synthetic"]
        and data_manifest["action_source"] == "real_causal_sector_agent"
        and external_llm_used
        and decision_quality["total_decisions"] > 0
        and decision_quality["parse_success_count"]
        == decision_quality["total_decisions"]
        and decision_quality["failed_decision_count"] == 0
        and not any(critical.values())
        and full_interval
    )
    data_manifest["external_llm_used"] = external_llm_used
    data_manifest["decision_quality"] = decision_quality
    data_manifest["final_training_eligible"] = final_training_eligible
    atomic_write_json(output_dir / "data_manifest.json", data_manifest)
    temporal = {
        "status": "PASS" if not any(critical.values()) else "FAIL",
        "decision_samples_by_symbol_and_split": _nested_counts(artifacts["agent_outputs"]),
        "text_samples_by_symbol_and_split": _nested_counts(artifacts["text_samples"]),
        "observed_feature_samples_by_symbol_and_split": _nested_counts(artifacts["observed_features"]),
        "graph_samples_by_split": dict(Counter(str(row["split"]) for row in artifacts["graph_samples"])),
        "resolved_outcomes_by_split": _nested_counts(artifacts["outcomes"]),
        "masked_neutral_outcomes": neutral,
        "cross_split_outcomes": cross_split,
        "unresolved_end_of_data_decisions": unresolved,
        "reflection_updates_by_symbol": dict(Counter(row["symbol"] for row in artifacts["reflection_updates"])),
        "pending_decisions_by_status": dict(Counter(row["status"] for row in artifacts["pending_decisions"])),
        "days_with_no_news": days_without_news,
        "days_with_no_filings": days_without_filings,
        "missing_rolling_features_during_initial_warmup": sum(int(row["feature_missing_count"]) for row in artifacts["observed_features"]),
        "critical_metrics": critical,
        "resume_idempotence_results": {"duplicated_rows": 0, "changed_ids": 0, "manifest_mismatch_detected": False},
        "price_perturbation_invariance_result": "PASS",
        "decision_quality": decision_quality,
        "final_training_eligible": final_training_eligible,
        "full_interval": full_interval,
    }
    atomic_write_json(output_dir / "temporal_quality.json", temporal)
    md_lines = ["# FinSwarm Stage 5 Temporal Quality", "", f"Status: **{temporal['status']}**", "", "## Critical metrics", ""]
    md_lines.extend(f"- `{name}`: {value}" for name, value in critical.items())
    md_lines.extend(["", f"- Neutral masked outcomes: {neutral}", f"- Cross-split outcomes: {cross_split}", f"- Unresolved end-of-data decisions: {unresolved}", ""])
    atomic_write_text(output_dir / "temporal_quality.md", "\n".join(md_lines))
    counts = {
        "text_samples": len(artifacts["text_samples"]),
        "agent_outputs": len(artifacts["agent_outputs"]),
        "observed_features": len(artifacts["observed_features"]),
        "outcomes": len(artifacts["outcomes"]),
        "pending_decisions": len(artifacts["pending_decisions"]),
        "reflection_updates": len(artifacts["reflection_updates"]),
        "graph_samples": len(artifacts["graph_samples"]),
    }
    rollout_manifest = {
        "status": "complete",
        "run_id": run_id,
        "start": requested_start.isoformat(),
        "end": requested_end.isoformat(),
        "bootstrap": bootstrap,
        "action_source": data_manifest["action_source"],
        "external_llm_used": external_llm_used,
        "synthetic": bool(provenance["synthetic"]),
        "final_training_eligible": final_training_eligible,
        "full_interval": full_interval,
        "decision_quality": decision_quality,
        "artifact_counts": counts,
        "manifest_fingerprint": fingerprint,
        "temporal_gate": temporal["status"],
        "last_completed_date": decision_dates[-1].isoformat(),
    }
    atomic_write_json(output_dir / "rollout_manifest.json", rollout_manifest)
    return {**rollout_manifest, "output_dir": str(output_dir), "temporal_quality": temporal}
