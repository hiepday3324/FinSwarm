from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import defaultdict
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


ADAPTER_VERSION = "legacy-finmem-causal-v1"
REFLECTION_PROMPT_VERSION = "retrospective-v1"


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
            causal_decide = getattr(self.llm_agent, "causal_decide", None)
            if causal_decide is None:
                raise RuntimeError("Non-bootstrap adapter requires llm_agent.causal_decide; LLMAgent.step is forbidden")
            generation_parameters = dict(run_context.metadata.get("generation_parameters", {}))
            cache_key = _stable_hash(
                getattr(self.llm_agent, "model_name", None),
                run_context.prompt_version,
                agent_context.context_hash,
                generation_parameters,
            )
            force_llm = bool(run_context.metadata.get("force_llm", False))
            cache_hit = cache_key in self._decision_cache and not force_llm
            if cache_hit:
                raw = dict(self._decision_cache[cache_key])
            else:
                raw = dict(causal_decide(market_step, agent_context, run_context))
                self._decision_cache[cache_key] = dict(raw)
            decision = str(raw.get("investment_decision", "hold")).lower()
            if decision not in {"buy", "hold", "sell"}:
                decision = "hold"
            action = {"sell": -1, "hold": 0, "buy": 1}[decision]
            reason = str(raw.get("summary_reason") or "No causal decision reason supplied.")
            model_name = getattr(self.llm_agent, "model_name", None)
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
