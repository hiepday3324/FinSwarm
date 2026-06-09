from enum import Enum


class Decision(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"


class MemoryLayer(str, Enum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"
    REFLECTION = "reflection"
    SHARED = "shared"


class DecisionMode(str, Enum):
    FAST_EXECUTION = "fast_execution"
    REVIEW = "review"
    DEBATE = "debate"


class DebateVerdictAction(str, Enum):
    KEEP = "keep"
    REDUCE = "reduce"
    OVERRIDE_HOLD = "override_hold"
    OVERRIDE_SELL = "override_sell"


class AllocationAction(str, Enum):
    INCREASE_EXPOSURE = "increase_exposure"
    DECREASE_EXPOSURE = "decrease_exposure"
    HOLD_EXPOSURE = "hold_exposure"
