from __future__ import annotations

from typing import Any

from puppy.common.enums import Decision


def decision_to_action_signal(decision: Any) -> int:
    value = decision.value if hasattr(decision, "value") else decision
    normalized = str(value).strip().lower()
    if normalized == Decision.BUY.value:
        return 1
    if normalized == Decision.SELL.value:
        return -1
    return 0
