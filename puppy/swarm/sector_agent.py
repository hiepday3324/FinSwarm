from datetime import date, timedelta
from typing import Any

from puppy.common.enums import MemoryLayer
from puppy.common.schemas import (
    AgentOutput,
    AgentSignal,
    DebateRequest,
    MemoryRef,
    SharedContext,
)
from puppy.run_type import RunMode

from .protocols import decision_to_action, is_valid_decision, normalize_decision


MEMORY_INDEX_FIELDS = {
    "short_memory_index": MemoryLayer.SHORT,
    "middle_memory_index": MemoryLayer.MID,
    "long_memory_index": MemoryLayer.LONG,
    "reflection_memory_index": MemoryLayer.REFLECTION,
}


class SectorAgent:
    def __init__(self, agent_id: str, symbol: str, llm_agent: Any):
        self.agent_id = agent_id
        self.symbol = symbol
        self.llm_agent = llm_agent
        self.external_context_buffer: list[SharedContext] = []
        self.outputs: dict[date, AgentOutput] = {}

    def inject_external_context(self, shared_context: SharedContext | str) -> None:
        if isinstance(shared_context, str):
            shared_context = SharedContext(
                source_agent_id="external",
                target_agent_id=self.agent_id,
                target_symbol=self.symbol,
                text=shared_context,
                reason="Injected external context.",
            )
        self.external_context_buffer.append(shared_context)

    def clear_expired_external_context(self, current_date: date | None = None) -> None:
        if current_date is None:
            self.external_context_buffer = [
                item
                for item in self.external_context_buffer
                if item.ttl_days is None or item.ttl_days > 0
            ]
            return

        active_context = []
        for item in self.external_context_buffer:
            if item.ttl_days is None or item.created_at is None:
                active_context.append(item)
                continue
            expires_at = item.created_at + timedelta(days=item.ttl_days)
            if current_date <= expires_at:
                active_context.append(item)
        self.external_context_buffer = active_context

    def step(self, market_info: tuple, run_mode: RunMode) -> AgentOutput:
        cur_date = market_info[0]
        self.clear_expired_external_context(cur_date)
        self.llm_agent.step(market_info=market_info, run_mode=run_mode)
        raw_reflection = self.llm_agent.reflection_result_series_dict.get(cur_date, {})
        output = self._build_output(cur_date=cur_date, raw_reflection=raw_reflection)
        self.outputs[cur_date] = output
        return output

    def get_signal(self, cur_date: date) -> AgentSignal:
        if cur_date not in self.outputs:
            raise KeyError(f"No AgentOutput for {self.agent_id} on {cur_date}")
        return self.outputs[cur_date].signal

    def answer_debate_question(
        self, question: str, request: DebateRequest | dict[str, Any]
    ) -> str:
        request_date = request.date if isinstance(request, DebateRequest) else request["date"]
        signal = self.get_signal(request_date)
        cited_ids = ", ".join(
            f"{ref.layer}:{ref.memory_id}" for ref in signal.memory_refs
        )
        if not cited_ids:
            cited_ids = "no cited memory"
        return (
            f"Local decision is {signal.decision}. Reason: {signal.reason} "
            f"Cited memory: {cited_ids}. Question: {question}"
        )

    def _build_output(self, cur_date: date, raw_reflection: dict[str, Any]) -> AgentOutput:
        decision_raw = raw_reflection.get("investment_decision")
        parse_ok = bool(raw_reflection) and is_valid_decision(decision_raw)
        decision = normalize_decision(decision_raw)
        reason = raw_reflection.get("summary_reason") or (
            "No valid LLM reflection output; defaulting to hold."
        )
        memory_refs = self._extract_memory_refs(raw_reflection, cur_date)
        shared_refs = self._shared_context_refs(cur_date)

        signal = AgentSignal(
            agent_id=self.agent_id,
            symbol=self.symbol,
            date=cur_date,
            decision=decision,
            action_signal=decision_to_action(decision),
            reason=reason,
            memory_refs=memory_refs,
            shared_context_used=shared_refs,
        )

        return AgentOutput(
            signal=signal,
            raw_reflection=raw_reflection,
            model_name=getattr(self.llm_agent, "model_name", None),
            parse_ok=parse_ok,
        )

    def _extract_memory_refs(
        self, raw_reflection: dict[str, Any], cur_date: date
    ) -> list[MemoryRef]:
        refs: list[MemoryRef] = []
        for field_name, layer in MEMORY_INDEX_FIELDS.items():
            raw_items = raw_reflection.get(field_name) or []
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            for item in raw_items:
                memory_id = self._extract_memory_id(item)
                if memory_id is None or memory_id == -1:
                    continue
                refs.append(
                    MemoryRef(
                        memory_id=memory_id,
                        layer=layer,
                        symbol=self.symbol,
                        date=cur_date,
                        source_agent_id=self.agent_id,
                    )
                )
        return refs

    def _shared_context_refs(self, cur_date: date) -> list[MemoryRef]:
        refs: list[MemoryRef] = []
        for shared_context in self.external_context_buffer:
            for ref in shared_context.memory_refs:
                refs.append(
                    MemoryRef(
                        memory_id=ref.memory_id,
                        layer=MemoryLayer.SHARED,
                        symbol=ref.symbol or shared_context.source_symbol,
                        date=ref.date or shared_context.created_at or cur_date,
                        score=ref.score,
                        source_agent_id=shared_context.source_agent_id,
                        text_preview=ref.text_preview,
                    )
                )
        return refs

    @staticmethod
    def _extract_memory_id(item: Any) -> int | str | None:
        if isinstance(item, dict):
            return item.get("memory_index") or item.get("memory_id") or item.get("id")
        if isinstance(item, (int, str)):
            return item
        if hasattr(item, "memory_index"):
            return item.memory_index
        return None
