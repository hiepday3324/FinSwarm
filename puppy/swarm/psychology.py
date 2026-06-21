"""Psychology state update rules from shadow portfolio metrics.

The state returned here is intentionally a plain dictionary because
``AgentContext`` already accepts ``psychology_state`` as structured metadata.
That keeps this module compatible with the current common schema while still
making the feedback loop explicit for prompts and text encoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_THRESHOLDS: dict[str, float] = {
    "greed_roi": 0.05,
    "greed_drawdown_floor": -0.03,
    "greed_win_rate": 0.55,
    "fear_roi": -0.03,
    "fear_drawdown": -0.08,
    "fear_win_rate": 0.45,
}


def build_psychology_state(
    shadow_state: Any | None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Convert shadow portfolio metrics into a prompt-ready psychology state.

    The labels mirror the roadmap formula:

    - ``greed``: ROI is strong and drawdown is small.
    - ``fear``: ROI is negative, drawdown is large, or win rate is weak.
    - ``neutral``: no strong performance feedback yet.
    """

    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        merged_thresholds.update({key: float(value) for key, value in thresholds.items()})

    metrics = _coerce_shadow_state(shadow_state)
    if not metrics:
        return _build_state(
            state="neutral",
            risk_attitude="balanced",
            reason="No shadow portfolio metrics are available yet.",
            prompt_hint=(
                "Use normal risk discipline; do not increase or decrease "
                "confidence from shadow performance."
            ),
            metrics={},
            thresholds=merged_thresholds,
        )

    roi = _as_float(metrics.get("roi"))
    drawdown = _as_float(metrics.get("drawdown"))
    sharpe = _as_float(metrics.get("sharpe"))
    win_rate = _as_float(metrics.get("win_rate"))

    fear_reasons: list[str] = []
    greed_reasons: list[str] = []

    if roi is not None and roi <= merged_thresholds["fear_roi"]:
        fear_reasons.append(f"ROI {roi:.4f} is below fear threshold")
    if drawdown is not None and drawdown <= merged_thresholds["fear_drawdown"]:
        fear_reasons.append(f"drawdown {drawdown:.4f} is beyond fear threshold")
    has_negative_performance = (
        (roi is not None and roi < 0.0)
        or (drawdown is not None and drawdown < 0.0)
    )
    if (
        win_rate is not None
        and win_rate <= merged_thresholds["fear_win_rate"]
        and has_negative_performance
    ):
        fear_reasons.append(f"win rate {win_rate:.4f} is weak")

    if roi is not None and roi >= merged_thresholds["greed_roi"]:
        greed_reasons.append(f"ROI {roi:.4f} is above greed threshold")
    if drawdown is not None and drawdown >= merged_thresholds["greed_drawdown_floor"]:
        greed_reasons.append(f"drawdown {drawdown:.4f} is controlled")
    if win_rate is not None and win_rate >= merged_thresholds["greed_win_rate"]:
        greed_reasons.append(f"win rate {win_rate:.4f} is strong")

    if fear_reasons:
        state = "fear"
        risk_attitude = "risk_averse"
        reason = "; ".join(fear_reasons)
        prompt_hint = (
            "Shadow performance is weak or unstable. Ask for stronger evidence, "
            "avoid overreacting to one bullish memory, and prefer HOLD when the "
            "new context is ambiguous."
        )
    elif _has_greed_signal(greed_reasons, roi, drawdown, win_rate, merged_thresholds):
        state = "greed"
        risk_attitude = "risk_seeking"
        reason = "; ".join(greed_reasons)
        prompt_hint = (
            "Shadow performance is strong. Keep reasoning disciplined, explicitly "
            "check downside evidence, and do not let recent wins alone justify BUY."
        )
    else:
        state = "neutral"
        risk_attitude = "balanced"
        reason = "Shadow metrics do not show a strong greed or fear regime."
        prompt_hint = (
            "Use the current market context and retrieved memories without a "
            "performance-driven bias adjustment."
        )

    return _build_state(
        state=state,
        risk_attitude=risk_attitude,
        reason=reason,
        prompt_hint=prompt_hint,
        metrics={
            "roi": roi,
            "drawdown": drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "rolling_return": _as_float(metrics.get("rolling_return")),
            "value": _as_float(metrics.get("value")),
        },
        thresholds=merged_thresholds,
        agent_id=metrics.get("agent_id"),
        symbol=metrics.get("symbol"),
        date=metrics.get("date"),
    )


def update_psychology_state(
    shadow_state: Any | None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Alias used by the milestone roadmap wording."""

    return build_psychology_state(shadow_state=shadow_state, thresholds=thresholds)


def psychology_update(
    shadow_state: Any | None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Backward-friendly alias for notebooks and scripts."""

    return build_psychology_state(shadow_state=shadow_state, thresholds=thresholds)


def _has_greed_signal(
    greed_reasons: list[str],
    roi: float | None,
    drawdown: float | None,
    win_rate: float | None,
    thresholds: Mapping[str, float],
) -> bool:
    if not greed_reasons:
        return False
    roi_ok = roi is not None and roi >= thresholds["greed_roi"]
    drawdown_ok = (
        drawdown is None or drawdown >= thresholds["greed_drawdown_floor"]
    )
    win_rate_ok = win_rate is None or win_rate >= thresholds["greed_win_rate"]
    return roi_ok and drawdown_ok and win_rate_ok


def _build_state(
    *,
    state: str,
    risk_attitude: str,
    reason: str,
    prompt_hint: str,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    agent_id: Any | None = None,
    symbol: Any | None = None,
    date: Any | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "psychology": state,
        "risk_attitude": risk_attitude,
        "reason": reason,
        "prompt_hint": prompt_hint,
        "metrics": metrics,
        "thresholds": thresholds,
        "source": "shadow_portfolio",
        "agent_id": agent_id,
        "symbol": symbol,
        "date": date,
    }


def _coerce_shadow_state(shadow_state: Any | None) -> dict[str, Any]:
    if shadow_state is None:
        return {}
    if hasattr(shadow_state, "model_dump"):
        return dict(shadow_state.model_dump())
    if isinstance(shadow_state, Mapping):
        return dict(shadow_state)
    return {
        key: getattr(shadow_state, key)
        for key in (
            "date",
            "agent_id",
            "symbol",
            "value",
            "roi",
            "rolling_return",
            "sharpe",
            "drawdown",
            "win_rate",
        )
        if hasattr(shadow_state, key)
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
