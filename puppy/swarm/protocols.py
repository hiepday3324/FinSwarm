from typing import Any

from puppy.common.enums import Decision


DECISION_TO_ACTION = {
    Decision.BUY.value: 1,
    Decision.HOLD.value: 0,
    Decision.SELL.value: -1,
}


def normalize_decision(decision: Any) -> Decision:
    if isinstance(decision, Decision):
        return decision
    if decision is None:
        return Decision.HOLD
    normalized = str(decision).strip().lower()
    try:
        return Decision(normalized)
    except ValueError:
        return Decision.HOLD


def is_valid_decision(decision: Any) -> bool:
    if isinstance(decision, Decision):
        return True
    if decision is None:
        return False
    return str(decision).strip().lower() in DECISION_TO_ACTION


def decision_to_action(decision: Any) -> int:
    return DECISION_TO_ACTION[normalize_decision(decision).value]
