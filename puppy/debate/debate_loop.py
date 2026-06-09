from typing import Any

from puppy.common.schemas import DebateRequest, DebateTranscript, MemoryRef

from .prompts import DEBATE_QUESTION_TEMPLATE


class DebateLoop:
    def run_debate(
        self,
        request: DebateRequest,
        sector_agent: Any | None = None,
    ) -> DebateTranscript:
        question = self._build_question(request)
        answer = self._build_answer(request=request, question=question, sector_agent=sector_agent)
        cited_refs = (
            request.agent_signal.memory_refs
            if request.agent_signal is not None
            else []
        )
        return DebateTranscript(
            date=request.date,
            agent_id=request.agent_id,
            symbol=request.symbol,
            request=request,
            question=question,
            answer=answer,
            cited_memory_refs=list(cited_refs),
        )

    def run(
        self, request: DebateRequest, sector_agent: Any | None = None
    ) -> DebateTranscript:
        return self.run_debate(request=request, sector_agent=sector_agent)

    def _build_question(self, request: DebateRequest) -> str:
        return DEBATE_QUESTION_TEMPLATE.format(
            symbol=request.symbol,
            local_decision=request.local_decision,
            text_signal=request.text_signal,
            graph_signal=request.graph_signal,
            fusion_score=request.fusion_score,
        )

    def _build_answer(
        self,
        request: DebateRequest,
        question: str,
        sector_agent: Any | None,
    ) -> str:
        if sector_agent is not None and hasattr(sector_agent, "answer_debate_question"):
            return sector_agent.answer_debate_question(question=question, request=request)

        cited_refs: list[MemoryRef] = (
            request.agent_signal.memory_refs
            if request.agent_signal is not None
            else []
        )
        cited_text = ", ".join(f"{ref.layer}:{ref.memory_id}" for ref in cited_refs)
        if not cited_text:
            cited_text = "no cited memory"
        return (
            f"Local decision is {request.local_decision}. "
            f"Reason: {request.reason}. Cited memory: {cited_text}."
        )
