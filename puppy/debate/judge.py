from puppy.common.enums import DebateVerdictAction, Decision
from puppy.common.schemas import DebateTranscript, DebateVerdict
from puppy.swarm.protocols import normalize_decision


class DebateJudge:
    def make_verdict(self, transcript: DebateTranscript) -> DebateVerdict:
        request = transcript.request
        local_decision = normalize_decision(request.local_decision)
        verdict = DebateVerdictAction.KEEP
        target_decision: Decision | None = None

        if request.fusion_score <= 0.25:
            verdict = DebateVerdictAction.OVERRIDE_SELL
            target_decision = Decision.SELL
        elif local_decision == Decision.BUY and request.graph_signal < 0:
            if request.fusion_score <= 0.45:
                verdict = DebateVerdictAction.REDUCE
            else:
                verdict = DebateVerdictAction.OVERRIDE_HOLD
                target_decision = Decision.HOLD
        elif local_decision == Decision.SELL and request.graph_signal > 0:
            verdict = DebateVerdictAction.REDUCE
        elif 0.3 <= request.fusion_score <= 0.7:
            verdict = DebateVerdictAction.REDUCE

        reason = (
            f"Rule-based judge: fusion_score={request.fusion_score:.3f}, "
            f"text_signal={request.text_signal}, graph_signal={request.graph_signal}, "
            f"local_decision={request.local_decision}, verdict={verdict.value}."
        )

        return DebateVerdict(
            date=transcript.date,
            agent_id=transcript.agent_id,
            symbol=transcript.symbol,
            verdict=verdict,
            reason=reason,
            transcript=transcript,
            target_decision=target_decision,
        )
