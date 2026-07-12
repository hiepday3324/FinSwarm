from __future__ import annotations

import datetime as dt

from puppy.common.schemas import OutcomeTable, PendingDecision, PriceBar

from .rollout_state import stable_id


FORMULA_VERSION = "next-open-to-close-v1"


def resolve_outcome(
    pending: PendingDecision,
    price_bar: PriceBar | None,
    *,
    outcome_date: dt.date,
    available_at: dt.datetime,
    threshold_bps: int = 10,
    decision_split: str | None = None,
    outcome_split: str | None = None,
) -> OutcomeTable:
    if pending.expected_outcome_date != outcome_date:
        raise ValueError("Outcome date must equal the next common trading date")
    if price_bar is None or price_bar.date != outcome_date:
        raise ValueError("Exact outcome-date price bar is required")
    if price_bar.open is None or price_bar.open <= 0:
        raise ValueError("Positive outcome-date open is required")
    if available_at <= pending.decision_cutoff:
        raise ValueError("Outcome availability must follow the decision cutoff")
    asset_return = float(price_bar.close) / float(price_bar.open) - 1.0
    threshold = threshold_bps / 10_000.0
    direction = 1 if asset_return > threshold else -1 if asset_return < -threshold else 0
    cross_split = bool(decision_split and outcome_split and decision_split != outcome_split)
    return OutcomeTable(
        outcome_id=stable_id(pending.decision_id, outcome_date, FORMULA_VERSION),
        decision_id=pending.decision_id,
        run_id=pending.run_id,
        agent_id=pending.agent_id,
        symbol=pending.symbol,
        decision_date=pending.decision_date,
        outcome_date=outcome_date,
        available_at=available_at,
        open_price=float(price_bar.open),
        close_price=float(price_bar.close),
        asset_return=asset_return,
        action_signal=pending.action_signal,
        strategy_return=pending.action_signal * asset_return,
        direction_label=direction,
        train_mask=direction != 0,
        threshold_bps=threshold_bps,
        formula_version=FORMULA_VERSION,
        metadata={
            "decision_split": decision_split,
            "outcome_split": outcome_split,
            "cross_split_outcome": cross_split,
            "supervised_eligible": not cross_split,
        },
    )
