from datetime import date

from puppy.common.enums import DecisionMode
from puppy.common.schemas import AgentSignal, DebateRequest
from puppy.swarm.protocols import decision_to_action


def detect_conflict(
    text_signal: int,
    graph_signal: int,
    fusion_score: float,
    grey_zone_low: float = 0.3,
    grey_zone_high: float = 0.7,
) -> bool:
    in_grey_zone = grey_zone_low <= fusion_score <= grey_zone_high
    signal_conflict = text_signal != graph_signal
    return in_grey_zone or signal_conflict


def detect_decision_mode(
    text_signal: int,
    graph_signal: int,
    fusion_score: float,
) -> DecisionMode:
    if detect_conflict(text_signal, graph_signal, fusion_score):
        return DecisionMode.DEBATE
    if fusion_score > 0.8 or fusion_score < 0.2:
        return DecisionMode.FAST_EXECUTION
    return DecisionMode.REVIEW


def build_debate_request(
    agent_signal: AgentSignal,
    fusion_score: float,
    graph_signal: int,
    cur_date: date | None = None,
    reason: str | None = None,
) -> DebateRequest:
    text_signal = decision_to_action(agent_signal.decision)
    return DebateRequest(
        date=cur_date or agent_signal.date,
        agent_id=agent_signal.agent_id,
        symbol=agent_signal.symbol,
        local_decision=agent_signal.decision,
        text_signal=text_signal,
        graph_signal=graph_signal,
        fusion_score=fusion_score,
        reason=reason or agent_signal.reason,
        mode=detect_decision_mode(text_signal, graph_signal, fusion_score),
        agent_signal=agent_signal,
    )
