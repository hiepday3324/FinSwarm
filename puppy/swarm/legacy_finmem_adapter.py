from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from puppy.common.enums import MemoryLayer
from puppy.common.schemas import (
    AgentContext,
    AgentOutput,
    AgentSignal,
    CausalRunContext,
    MarketStep,
    MemoryRef,
    OutcomeTable,
    PendingDecision,
    ReflectionUpdate,
)


ADAPTER_VERSION = "legacy-finmem-causal-v2"
REFLECTION_PROMPT_VERSION = "retrospective-v1"


class CausalDecisionFailure(RuntimeError):
    """Sanitized terminal failure for one frozen causal decision."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        response_hash: str,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.response_hash = response_hash


def _stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CausalMemoryRecord:
    memory_id: str
    text: str
    layer: str
    source_event_id: str
    symbol: str
    source_type: str
    source_available_at: dt.datetime
    ingested_at: dt.datetime
    rollout_date: dt.date
    metadata: dict[str, Any]


class LegacyFinMemAdapter:
    """Causal boundary around legacy FinMem memory and optional LLM behavior."""

    def __init__(
        self,
        llm_agent: Any,
        agent_id: str,
        symbol: str,
        *,
        bootstrap: bool = False,
        max_records_per_layer: int = 256,
    ) -> None:
        self.llm_agent = llm_agent
        self.agent_id = agent_id
        self.symbol = symbol
        self.bootstrap = bootstrap
        self.max_records_per_layer = max_records_per_layer
        self._records: dict[str, list[CausalMemoryRecord]] = defaultdict(list)
        self._seen_source_ids: set[str] = set()
        self._advanced_dates: set[dt.date] = set()
        self.decay_count = 0
        self.feedback_count = 0
        self._decision_cache: dict[str, dict[str, Any]] = {}
        self._decision_stats: Counter[str] = Counter()

    @property
    def decision_stats(self) -> dict[str, int]:
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
        return {field: int(self._decision_stats[field]) for field in fields}

    def record_replayed_decision(self, output: AgentOutput) -> None:
        """Restore decision accounting without invoking the chat backend."""

        if output.signal.symbol != self.symbol or not output.parse_ok:
            raise ValueError("Replayed AgentOutput does not belong to this adapter")
        decision = str(output.signal.decision).lower()
        if decision not in {"buy", "hold", "sell"}:
            raise ValueError("Replayed AgentOutput has an invalid decision")
        provenance = dict(output.metadata.get("causal_provenance", {}))
        self._decision_stats["total_decisions"] += 1
        self._decision_stats[f"{decision}_count"] += 1
        self._decision_stats["parse_success_count"] += 1
        self._decision_stats["retry_count"] += int(provenance.get("retry_count", 0))
        self._decision_stats["cache_hit_count"] += int(
            bool(output.metadata.get("cache_hit", False))
        )
        self._decision_stats["api_call_count"] += int(
            provenance.get("api_call_count", 0)
        )

    def advance_memory_clock(
        self,
        current_date: dt.date,
        run_context: CausalRunContext,
    ) -> None:
        if current_date != run_context.current_date:
            raise ValueError("Memory clock date must match run context")
        if current_date in self._advanced_dates:
            raise RuntimeError(f"Memory clock already advanced for {current_date}")
        self._advanced_dates.add(current_date)
        self.decay_count += 1
        brain = getattr(self.llm_agent, "brain", None)
        if brain is not None and hasattr(brain, "step"):
            brain.step()

    def ingest_market_events(
        self,
        market_step: MarketStep,
        run_context: CausalRunContext,
    ) -> list[MemoryRef]:
        if market_step.as_of != run_context.current_cutoff:
            raise ValueError("MarketStep cutoff does not match run context")
        refs: list[MemoryRef] = []
        for event in market_step.news.get(self.symbol, []):
            refs.extend(self._insert(
                source_event_id=event.event_id,
                text=event.text,
                layer=MemoryLayer.SHORT.value,
                source_type="massive_news",
                available_at=event.available_at,
                run_context=run_context,
                metadata={"title": event.title, "url": event.url},
            ))
        for filing in market_step.filings.get(self.symbol, []):
            form = filing.filing_type.upper()
            if form in {"8-K", "8-K/A"}:
                layer = MemoryLayer.SHORT.value
            elif form in {"10-Q", "10-Q/A"}:
                layer = MemoryLayer.MID.value
            elif form in {"10-K", "10-K/A"}:
                layer = MemoryLayer.LONG.value
            else:
                layer = MemoryLayer.MID.value
            refs.extend(self._insert(
                source_event_id=filing.filing_id,
                text=filing.text,
                layer=layer,
                source_type=f"sec_{form.lower()}",
                available_at=filing.available_at,
                run_context=run_context,
                metadata={"accession_number": filing.accession_number, "url": filing.url},
            ))
        return refs

    def ingest_reflection(
        self,
        reflection: ReflectionUpdate,
        run_context: CausalRunContext,
    ) -> list[MemoryRef]:
        if reflection.memory_layer != MemoryLayer.REFLECTION.value:
            raise ValueError("ReflectionUpdate must route to reflection memory")
        return self._insert(
            source_event_id=reflection.reflection_id,
            text=reflection.text,
            layer=MemoryLayer.REFLECTION.value,
            source_type="retrospective_reflection",
            available_at=reflection.available_at,
            run_context=run_context,
            metadata={
                "decision_id": reflection.decision_id,
                "outcome_id": reflection.outcome_id,
                "source_decision_date": reflection.source_decision_date.isoformat(),
            },
        )

    def _insert(
        self,
        *,
        source_event_id: str,
        text: str,
        layer: str,
        source_type: str,
        available_at: dt.datetime,
        run_context: CausalRunContext,
        metadata: dict[str, Any],
    ) -> list[MemoryRef]:
        if available_at > run_context.current_cutoff:
            raise ValueError(f"Cannot ingest future event {source_event_id}")
        idempotence_key = f"{self.symbol}:{source_event_id}:{layer}"
        if idempotence_key in self._seen_source_ids:
            return []
        self._seen_source_ids.add(idempotence_key)
        memory_id = _stable_hash(self.agent_id, idempotence_key)
        memory_text = text if len(text) <= 4000 else text[:4000] + "\n[deterministic memory excerpt truncated]"
        record = CausalMemoryRecord(
            memory_id=memory_id,
            text=memory_text,
            layer=layer,
            source_event_id=source_event_id,
            symbol=self.symbol,
            source_type=source_type,
            source_available_at=available_at,
            ingested_at=run_context.current_cutoff,
            rollout_date=run_context.current_date,
            metadata={**metadata, "original_character_count": len(text)},
        )
        bucket = self._records[layer]
        bucket.append(record)
        if len(bucket) > self.max_records_per_layer:
            del bucket[: len(bucket) - self.max_records_per_layer]
        brain = getattr(self.llm_agent, "brain", None)
        add_method = getattr(brain, f"add_memory_{'mid' if layer == 'mid' else layer}", None)
        if add_method is not None:
            add_method(self.symbol, run_context.current_date, memory_text)
        return [self._to_ref(record)]

    def retrieve(
        self,
        current_cutoff: dt.datetime,
        *,
        top_k: int = 3,
    ) -> tuple[dict[str, list[str]], list[MemoryRef]]:
        memories = {"short": [], "mid": [], "long": [], "reflection": []}
        refs: list[MemoryRef] = []
        for layer in memories:
            eligible = [
                record for record in self._records[layer]
                if record.source_available_at <= current_cutoff
            ][-top_k:]
            memories[layer] = [record.text for record in eligible]
            refs.extend(self._to_ref(record) for record in eligible)
        return memories, refs

    def decide(
        self,
        market_step: MarketStep,
        agent_context: AgentContext,
        run_context: CausalRunContext,
    ) -> AgentOutput:
        if market_step.date != run_context.current_date or agent_context.date != run_context.current_date:
            raise ValueError("Decision inputs must share the current causal date")
        if agent_context.context_hash is None:
            raise ValueError("Frozen AgentContext must have a context_hash")
        if self.bootstrap:
            bucket = int(agent_context.context_hash[:8], 16) % 3
            decision = ("sell", "hold", "buy")[bucket]
            action = (-1, 0, 1)[bucket]
            reason = f"Deterministic bootstrap policy bucket={bucket}; no external LLM used."
            raw = {"bootstrap": True, "policy": "sha256-context-bucket-v1"}
            model_name = "bootstrap-deterministic-policy-v1"
        else:
            if run_context.bootstrap:
                raise RuntimeError("Non-bootstrap adapter cannot run in bootstrap mode")
            causal_decide = getattr(self.llm_agent, "causal_decide", None)
            if causal_decide is None:
                raise RuntimeError("Non-bootstrap adapter requires llm_agent.causal_decide; LLMAgent.step is forbidden")
            generation_parameters = dict(run_context.metadata.get("generation_parameters", {}))
            generation_parameters.setdefault("temperature", 0.0)
            generation_parameters.setdefault("reasoning_effort", "low")
            model_identity = str(getattr(self.llm_agent, "model_name", "unknown"))
            backend_identity = str(
                getattr(self.llm_agent, "backend_identity", "unknown-backend")
            )
            backend = getattr(self.llm_agent, "chat_backend", None)
            backend_provider = str(
                getattr(backend, "service_provider", "unknown-provider")
            )
            backend_interface = str(getattr(backend, "provider", "unknown-interface"))
            base_url_identity = str(
                getattr(backend, "base_url_identity", "unknown-base-url")
            )
            base_url = str(getattr(backend, "base_url", "unknown-base-url"))
            schema_version = str(
                getattr(self.llm_agent, "causal_schema_version", "unknown-schema")
            )
            cache_key = _stable_hash(
                backend_provider,
                backend_interface,
                base_url_identity,
                model_identity,
                schema_version,
                run_context.prompt_version,
                self.symbol,
                agent_context.context_hash,
                generation_parameters,
            )
            force_llm = bool(run_context.metadata.get("force_llm", False))
            cache_hit = cache_key in self._decision_cache and not force_llm
            api_calls_before = int(getattr(backend, "api_call_count", 0))
            transport_retries_before = int(getattr(backend, "retry_count", 0))
            if cache_hit:
                raw = dict(self._decision_cache[cache_key])
            else:
                try:
                    raw = dict(causal_decide(market_step, agent_context, run_context))
                except Exception as exc:
                    self._decision_stats["failed_decision_count"] += 1
                    response_hash = str(
                        getattr(
                            exc,
                            "response_hash",
                            getattr(backend, "last_response_hash", None) or "unavailable",
                        )
                    )
                    category = str(
                        getattr(exc, "category", "structured_parse_failed")
                    )
                    returned_model = str(
                        getattr(exc, "returned_model", None) or "unknown"
                    )
                    raise CausalDecisionFailure(
                        f"Causal decision failed; symbol={self.symbol}; "
                        f"date={run_context.current_date.isoformat()}; "
                        f"category={category}; response_hash={response_hash}; "
                        f"model={model_identity}; returned_model={returned_model}; "
                        f"prompt_version={run_context.prompt_version}",
                        category=category,
                        response_hash=response_hash,
                    ) from exc
                self._decision_cache[cache_key] = dict(raw)
            original_parse_retries = int(raw.pop("_causal_parse_retries", 0))
            parse_retries = 0 if cache_hit else original_parse_retries
            response_hash = str(raw.pop("_causal_response_hash", "unavailable"))
            causal_provenance = dict(raw.pop("_causal_provenance", {}))
            eligible_memory_ids = dict(raw.pop("_causal_eligible_memory_ids", {}))
            eligible_memory_set_hash = str(
                raw.pop("_causal_eligible_memory_set_hash", "unavailable")
            )
            decision_raw = raw.get("investment_decision")
            decision = decision_raw if isinstance(decision_raw, str) else ""
            if decision not in {"buy", "hold", "sell"}:
                self._decision_stats["failed_decision_count"] += 1
                raise CausalDecisionFailure(
                    f"Causal decision failed; symbol={self.symbol}; "
                    f"date={run_context.current_date.isoformat()}; "
                    "error=invalid_explicit_action; "
                    f"response_hash={response_hash}; "
                    f"prompt_version={run_context.prompt_version}",
                    category="invalid_decision_enum",
                    response_hash=response_hash,
                )
            action = {"sell": -1, "hold": 0, "buy": 1}[decision]
            reason = str(raw.get("summary_reason") or "No causal decision reason supplied.")
            model_name = model_identity
            api_calls = int(getattr(backend, "api_call_count", 0)) - api_calls_before
            transport_retries = (
                int(getattr(backend, "retry_count", 0)) - transport_retries_before
            )
            self._decision_stats["total_decisions"] += 1
            self._decision_stats[f"{decision}_count"] += 1
            self._decision_stats["parse_success_count"] += 1
            self._decision_stats["retry_count"] += parse_retries + transport_retries
            self._decision_stats["cache_hit_count"] += int(cache_hit)
            self._decision_stats["api_call_count"] += max(0, api_calls)
            causal_provenance.update(
                {
                    "provider": causal_provenance.get("provider", backend_provider),
                    "interface": causal_provenance.get("interface", backend_interface),
                    "base_url": base_url,
                    "base_url_identity": base_url_identity,
                    "requested_model": causal_provenance.get(
                        "requested_model", model_identity
                    ),
                    "returned_model": causal_provenance.get(
                        "returned_model",
                        getattr(backend, "last_returned_model", None),
                    ),
                    "prompt_version": run_context.prompt_version,
                    "schema_version": schema_version,
                    "temperature": generation_parameters["temperature"],
                    "reasoning_effort": generation_parameters["reasoning_effort"],
                    "context_hash": agent_context.context_hash,
                    "response_hash": response_hash,
                    "retry_count": causal_provenance.get(
                        "retry_count", parse_retries + transport_retries
                    ),
                    "api_call_count": max(0, api_calls),
                    "cache_hit": cache_hit,
                    "eligible_memory_set_hash": eligible_memory_set_hash,
                }
            )
        return AgentOutput(
            signal=AgentSignal(
                agent_id=self.agent_id,
                symbol=self.symbol,
                date=run_context.current_date,
                decision=decision,
                action_signal=action,
                reason=reason,
                memory_refs=list(agent_context.memory_refs),
            ),
            raw_reflection=raw,
            model_name=model_name,
            parse_ok=True,
            context_hash=agent_context.context_hash,
            decision_cutoff=run_context.current_cutoff,
            prompt_version=run_context.prompt_version,
            adapter_version=run_context.adapter_version,
            metadata={
                "bootstrap": self.bootstrap,
                "cache_hit": False if self.bootstrap else cache_hit,
                "generation_parameters": {} if self.bootstrap else generation_parameters,
                "cache_key": None if self.bootstrap else cache_key,
                "backend_identity": None if self.bootstrap else backend_identity,
                "backend_provider": (
                    None if self.bootstrap else backend_provider
                ),
                "backend_interface": None if self.bootstrap else backend_interface,
                "base_url_identity": None if self.bootstrap else base_url_identity,
                "schema_version": None if self.bootstrap else schema_version,
                "eligible_memory_ids": (
                    {} if self.bootstrap else eligible_memory_ids
                ),
                "eligible_memory_set_hash": (
                    None if self.bootstrap else eligible_memory_set_hash
                ),
                "causal_provenance": {} if self.bootstrap else causal_provenance,
                "synthetic": (
                    False if self.bootstrap else bool(getattr(backend, "is_mock", False))
                ),
            },
        )

    def learn_from_outcome(
        self,
        pending_decision: PendingDecision,
        realized_outcome: OutcomeTable | None,
        run_context: CausalRunContext,
    ) -> ReflectionUpdate:
        if realized_outcome is None:
            raise ValueError("A realized outcome is required before learning")
        if realized_outcome.decision_id != pending_decision.decision_id:
            raise ValueError("Outcome does not belong to pending decision")
        if realized_outcome.available_at <= pending_decision.decision_cutoff:
            raise ValueError("Outcome must become available after decision cutoff")
        feedback = 1 if realized_outcome.strategy_return > 0 else -1 if realized_outcome.strategy_return < 0 else 0
        brain = getattr(self.llm_agent, "brain", None)
        numeric_ids = [ref.memory_id for ref in pending_decision.memory_refs if isinstance(ref.memory_id, int)]
        if brain is not None and numeric_ids and hasattr(brain, "update_access_count_with_feed_back"):
            brain.update_access_count_with_feed_back(self.symbol, numeric_ids, feedback)
        self.feedback_count += len(pending_decision.memory_refs)
        outcome_id = realized_outcome.outcome_id
        reflection_id = _stable_hash(pending_decision.decision_id, outcome_id, REFLECTION_PROMPT_VERSION)
        snapshot_hash = _stable_hash(pending_decision.context_snapshot)
        text = (
            f"Retrospective for {pending_decision.symbol} decision {pending_decision.decision_id}. "
            f"Decision={pending_decision.decision}; action={pending_decision.action_signal}; "
            f"asset_return={realized_outcome.asset_return:.10f}; "
            f"strategy_return={realized_outcome.strategy_return:.10f}; "
            f"decision_context_snapshot_sha256={snapshot_hash}."
        )
        return ReflectionUpdate(
            reflection_id=reflection_id,
            decision_id=pending_decision.decision_id,
            outcome_id=outcome_id,
            run_id=pending_decision.run_id,
            agent_id=pending_decision.agent_id,
            symbol=pending_decision.symbol,
            source_decision_date=pending_decision.decision_date,
            outcome_date=realized_outcome.outcome_date,
            available_at=run_context.current_cutoff,
            text=text,
            memory_layer=MemoryLayer.REFLECTION.value,
            prompt_version=REFLECTION_PROMPT_VERSION,
            context_hash=pending_decision.context_hash,
            metadata={
                "context_snapshot_sha256": snapshot_hash,
                "feedback": feedback,
                "memory_ref_count": len(pending_decision.memory_refs),
            },
        )

    @staticmethod
    def _to_ref(record: CausalMemoryRecord) -> MemoryRef:
        source_date = record.rollout_date
        if record.layer == MemoryLayer.REFLECTION.value and record.metadata.get("source_decision_date"):
            source_date = dt.date.fromisoformat(str(record.metadata["source_decision_date"]))
        return MemoryRef(
            memory_id=record.memory_id,
            layer=record.layer,
            symbol=record.symbol,
            date=source_date,
            text_preview=record.text[:160],
            source_event_id=record.source_event_id,
            source_type=record.source_type,
            available_at=record.source_available_at,
            metadata={
                "ingested_at": record.ingested_at.isoformat(),
                "rollout_date": record.rollout_date.isoformat(),
                "memory_layer": record.layer,
                **record.metadata,
            },
        )
