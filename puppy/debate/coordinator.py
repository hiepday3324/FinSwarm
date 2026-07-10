from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from puppy.common.enums import DecisionMode
from puppy.common.schemas import (
    AgentSignal,
    DebateRequest,
    DebateVerdict,
    FusionOutput,
)

from .conflict_detector import build_debate_request
from .debate_loop import DebateLoop
from .judge import DebateJudge


def graph_score_to_signal(
    graph_score: float | int | None,
    threshold: float = 0.0,
) -> int:
    """Convert a graph score in [-1, 1] into sell/hold/buy signal."""

    score = 0.0 if graph_score is None else float(graph_score)
    cutoff = abs(float(threshold))
    if score > cutoff:
        return 1
    if score < -cutoff:
        return -1
    return 0


def build_debate_requests_from_fusion(
    agent_signals: Iterable[AgentSignal],
    fusion_output: FusionOutput,
    graph_signal_threshold: float = 0.0,
    include_conflict_flags: bool = True,
    include_review: bool = False,
) -> list[DebateRequest]:
    """Build M7 debate requests from agent signals and fusion artifacts."""

    signal_by_symbol = {signal.symbol: signal for signal in agent_signals}
    requests: list[DebateRequest] = []
    for symbol in fusion_output.symbols:
        if symbol not in signal_by_symbol:
            continue
        if symbol not in fusion_output.fusion_scores:
            raise KeyError(f"Missing fusion score for {symbol}.")

        agent_signal = signal_by_symbol[symbol]
        graph_signal = graph_score_to_signal(
            fusion_output.graph_scores.get(symbol, 0.0),
            threshold=graph_signal_threshold,
        )
        request = build_debate_request(
            agent_signal=agent_signal,
            fusion_score=float(fusion_output.fusion_scores[symbol]),
            graph_signal=graph_signal,
            cur_date=fusion_output.date,
            reason=(
                f"M7 debate check for {symbol}: "
                f"fusion_score={fusion_output.fusion_scores[symbol]:.3f}, "
                f"graph_score={fusion_output.graph_scores.get(symbol, 0.0):.3f}."
            ),
        )

        is_fusion_conflict = bool(fusion_output.conflict_flags.get(symbol, False))
        should_debate = request.mode == DecisionMode.DEBATE
        if include_conflict_flags:
            should_debate = should_debate or is_fusion_conflict
        if should_debate or include_review:
            requests.append(request)

    return requests


class DebateCoordinator:
    """M7 coordinator for allocator questions, agent answers, and verdicts."""

    def __init__(
        self,
        debate_loop: DebateLoop | None = None,
        judge: DebateJudge | None = None,
    ) -> None:
        self.debate_loop = debate_loop or DebateLoop()
        self.judge = judge or DebateJudge()

    def run_for_fusion(
        self,
        fusion_output: FusionOutput,
        agent_signals: Iterable[AgentSignal],
        sector_agents: Mapping[str, Any] | Iterable[Any] | None = None,
        event_store: Any | None = None,
        graph_signal_threshold: float = 0.0,
    ) -> list[DebateVerdict]:
        requests = build_debate_requests_from_fusion(
            agent_signals=agent_signals,
            fusion_output=fusion_output,
            graph_signal_threshold=graph_signal_threshold,
        )
        verdicts: list[DebateVerdict] = []
        for request in requests:
            sector_agent = _resolve_sector_agent(sector_agents, request)
            transcript = self.debate_loop.run_debate(
                request=request,
                sector_agent=sector_agent,
            )
            verdict = self.judge.make_verdict(transcript)
            verdicts.append(verdict)
            _append_debate_artifacts(event_store, request, transcript, verdict)
        return verdicts


def _resolve_sector_agent(
    sector_agents: Mapping[str, Any] | Iterable[Any] | None,
    request: DebateRequest,
) -> Any | None:
    if sector_agents is None:
        return None
    if isinstance(sector_agents, Mapping):
        for key in (request.agent_id, request.symbol):
            if key in sector_agents:
                return sector_agents[key]
        return None

    for agent in sector_agents:
        if getattr(agent, "agent_id", None) == request.agent_id:
            return agent
        if getattr(agent, "symbol", None) == request.symbol:
            return agent
    return None


def _append_debate_artifacts(
    event_store: Any | None,
    request: DebateRequest,
    transcript: Any,
    verdict: DebateVerdict,
) -> None:
    if event_store is None:
        return
    if hasattr(event_store, "append_debate_request"):
        event_store.append_debate_request(request.date, request)
        event_store.append_debate_transcript(transcript.date, transcript)
        event_store.append_debate_verdict(verdict.date, verdict)
        return

    event_store.append_event("DebateRequest", request.date, request)
    event_store.append_event("DebateTranscript", transcript.date, transcript)
    event_store.append_event("DebateVerdict", verdict.date, verdict)
