from datetime import date

from puppy.common.enums import AllocationAction, Decision
from puppy.common.schemas import AgentSignal, AllocationDecision
from puppy.swarm.protocols import normalize_decision


class AllocatorAgent:
    def aggregate(
        self, signals: list[AgentSignal], cur_date: date | None = None
    ) -> AllocationDecision:
        score = sum(signal.action_signal for signal in signals)
        if score > 0:
            action = AllocationAction.INCREASE_EXPOSURE
        elif score < 0:
            action = AllocationAction.DECREASE_EXPOSURE
        else:
            action = AllocationAction.HOLD_EXPOSURE

        decisions = [normalize_decision(signal.decision) for signal in signals]
        buy_count = sum(decision == Decision.BUY for decision in decisions)
        sell_count = sum(decision == Decision.SELL for decision in decisions)
        hold_count = sum(decision == Decision.HOLD for decision in decisions)
        reason = (
            f"{action.value}: buy={buy_count}, sell={sell_count}, "
            f"hold={hold_count}, net_signal={score}."
        )

        return AllocationDecision(
            date=cur_date,
            action=action,
            score=score,
            reason=reason,
            signals=signals,
        )

    def decide(
        self, signals: list[AgentSignal], cur_date: date | None = None
    ) -> AllocationDecision:
        return self.aggregate(signals=signals, cur_date=cur_date)
