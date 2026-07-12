from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from puppy.common.schemas import (
    AgentContext,
    AgentOutput,
    CausalRunContext,
    MarketStep,
    PendingDecision,
    ShadowPortfolioState,
    TextFeature,
)
from puppy.swarm.context_builder import build_raw_context
from puppy.training.rollout_state import canonical_json, stable_id
from puppy.training.temporal_assertions import assert_context_causal, assert_shadow_prior


BOOTSTRAP_ENCODER_VERSION = "sha256-frozen-encoder-v1"


@dataclass(frozen=True)
class CausalDecisionBundle:
    context: AgentContext
    text_feature: TextFeature
    output: AgentOutput
    pending: PendingDecision


class DeterministicTextEncoder:
    def __init__(self, dimension: int = 16, version: str = BOOTSTRAP_ENCODER_VERSION) -> None:
        self.dimension = dimension
        self.version = version
        self.embedding_model = "bootstrap-sha256-frozen"

    def encode(
        self,
        context: AgentContext,
        run_context: CausalRunContext,
        text_sample_id: str,
    ) -> TextFeature:
        before = canonical_json(context)
        raw_context = build_raw_context(context)
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{raw_context}|{counter}".encode("utf-8")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        if canonical_json(context) != before:
            raise AssertionError("Text encoder mutated AgentContext")
        return TextFeature(
            agent_id=context.agent_id,
            symbol=context.symbol,
            date=context.date,
            h_text=np.asarray(values[: self.dimension], dtype=np.float32).tolist(),
            raw_context=raw_context,
            embedding_model=self.embedding_model,
            dim=self.dimension,
            dtype="float32",
            text_sample_id=text_sample_id,
            as_of=run_context.current_cutoff,
            context_hash=context.context_hash,
            encoder_version=self.version,
            prompt_version=run_context.prompt_version,
            metadata={"bootstrap": True, "deterministic": True},
        )


def build_context_hash(
    *,
    agent_id: str,
    symbol: str,
    market_step: MarketStep,
    context: AgentContext,
    shadow_state: ShadowPortfolioState | None,
    psychology_state: dict[str, Any] | None,
    prompt_version: str,
    adapter_version: str,
) -> str:
    price = market_step.prices.get(symbol)
    payload = {
        "agent_id": agent_id,
        "symbol": symbol,
        "decision_date": market_step.date,
        "decision_cutoff": market_step.as_of,
        "price_bar": price.model_dump(mode="json") if price else None,
        "news_event_ids": sorted(event.event_id for event in market_step.news.get(symbol, [])),
        "filing_event_ids": sorted(event.filing_id for event in market_step.filings.get(symbol, [])),
        "memories": sorted(
            (
                str(ref.memory_id), str(ref.layer), ref.text_preview or "",
                ref.available_at.isoformat() if ref.available_at else None,
            )
            for ref in context.memory_refs
        ),
        "memory_text": {
            "short": list(context.short_memory), "mid": list(context.mid_memory),
            "long": list(context.long_memory), "reflection": list(context.reflection_memory),
        },
        "shadow_state_prior": shadow_state.model_dump(mode="json") if shadow_state else None,
        "psychology_state_prior": psychology_state,
        "external_context_ids": sorted(
            f"{item.source_agent_id}:{item.target_agent_id}:{item.created_at}"
            for item in context.shared_context
        ),
        "latest_prior_reason": context.latest_reason,
        "prompt_version": prompt_version,
        "adapter_version": adapter_version,
    }
    return stable_id(payload)


class CausalOrchestrator:
    def __init__(self, encoder: DeterministicTextEncoder | None = None) -> None:
        self.encoder = encoder or DeterministicTextEncoder()

    def decide(
        self,
        *,
        agent: Any,
        market_step: MarketStep,
        run_context: CausalRunContext,
        expected_outcome_date: Any,
        split: str,
        shadow_state: ShadowPortfolioState | None = None,
        psychology_state: dict[str, Any] | None = None,
        latest_prior_reason: str | None = None,
    ) -> CausalDecisionBundle:
        assert_shadow_prior(shadow_state, market_step.date)
        adapter = agent.causal_adapter
        memories, refs = adapter.retrieve(run_context.current_cutoff)
        context = AgentContext(
            agent_id=agent.agent_id,
            symbol=agent.symbol,
            date=market_step.date,
            latest_reason=latest_prior_reason,
            short_memory=memories["short"],
            mid_memory=memories["mid"],
            long_memory=memories["long"],
            reflection_memory=memories["reflection"],
            memory_refs=refs,
            shared_context=list(agent.external_context_buffer) if run_context.memory_sharing_enabled else [],
            shadow_state=shadow_state.model_dump(mode="json") if shadow_state else None,
            psychology_state=psychology_state,
            as_of=run_context.current_cutoff,
            metadata={"built_before_decision": True, "split": split},
        )
        context.context_hash = build_context_hash(
            agent_id=agent.agent_id,
            symbol=agent.symbol,
            market_step=market_step,
            context=context,
            shadow_state=shadow_state,
            psychology_state=psychology_state,
            prompt_version=run_context.prompt_version,
            adapter_version=run_context.adapter_version,
        )
        assert_context_causal(context, run_context.current_cutoff)
        text_sample_id = stable_id(
            run_context.run_id, agent.agent_id, agent.symbol, market_step.date,
            run_context.current_cutoff, context.context_hash, run_context.encoder_version,
        )
        decision_id = stable_id(
            run_context.run_id, agent.agent_id, agent.symbol, market_step.date,
            run_context.current_cutoff, context.context_hash, run_context.prompt_version,
            run_context.adapter_version,
        )
        text_feature = self.encoder.encode(context, run_context, text_sample_id)
        text_feature.decision_id = decision_id
        output = agent.decide(market_step, context.model_copy(deep=True), run_context)
        output.decision_id = decision_id
        output.h_text = list(text_feature.h_text)
        output.raw_context = text_feature.raw_context
        context_snapshot = {
            "agent_context": context.model_dump(mode="json"),
            "price_bar": market_step.prices.get(agent.symbol).model_dump(mode="json") if agent.symbol in market_step.prices else None,
            "news_event_ids": sorted(event.event_id for event in market_step.news.get(agent.symbol, [])),
            "filing_event_ids": sorted(event.filing_id for event in market_step.filings.get(agent.symbol, [])),
        }
        pending = PendingDecision(
            decision_id=decision_id,
            sample_id=stable_id("pending", decision_id),
            run_id=run_context.run_id,
            agent_id=agent.agent_id,
            symbol=agent.symbol,
            decision_date=market_step.date,
            decision_cutoff=run_context.current_cutoff,
            expected_outcome_date=expected_outcome_date,
            decision=output.signal.decision,
            action_signal=output.signal.action_signal,
            reason=output.signal.reason,
            context_hash=context.context_hash,
            context_snapshot=context_snapshot,
            memory_refs=list(context.memory_refs),
            text_sample_id=text_sample_id,
            prompt_version=run_context.prompt_version,
            encoder_version=run_context.encoder_version,
            adapter_version=run_context.adapter_version,
            status="pending" if expected_outcome_date else "unresolved_end_of_data",
            metadata={"split": split, "context_built_before_output": True},
        )
        return CausalDecisionBundle(context, text_feature, output, pending)
