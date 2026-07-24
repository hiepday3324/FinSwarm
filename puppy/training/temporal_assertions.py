from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any, Iterable

from puppy.common.schemas import (
    AgentContext,
    MarketStep,
    ObservedFeatureTable,
    OutcomeTable,
    PendingDecision,
    PortfolioSnapshot,
    TextFeature,
)
from puppy.graph.graph_dataset import CANONICAL_FEATURE_NAMES


FORBIDDEN_GRAPH_FIELDS = {"return", "label", "asset_return", "strategy_return", "direction_label", "outcome_id"}


def assert_observed_features_causal(table: ObservedFeatureTable) -> None:
    if table.as_of.tzinfo is None:
        raise AssertionError("ObservedFeatureTable.as_of must be timezone-aware")
    for symbol, dates in table.source_dates.items():
        if any(date > table.date for date in dates):
            raise AssertionError(f"Future observed-feature source date for {symbol}")


def assert_market_step_causal(step: MarketStep) -> None:
    for collection in (step.news, step.filings):
        for events in collection.values():
            if any(event.available_at > step.as_of for event in events):
                raise AssertionError("MarketStep contains a future event")


def assert_context_causal(context: AgentContext, cutoff: dt.datetime) -> None:
    if context.as_of != cutoff:
        raise AssertionError("AgentContext cutoff mismatch")
    for ref in context.memory_refs:
        if ref.available_at is not None and ref.available_at > cutoff:
            raise AssertionError("AgentContext contains future memory")
        if ref.layer == "reflection" and ref.date is not None and ref.date >= context.date:
            raise AssertionError("Decision sees same-day or future reflection source")
    if any(
        item.created_at is not None and item.created_at >= context.date
        for item in context.shared_context
    ):
        raise AssertionError("Decision sees same-day or future shared memory")
    assert_master_portfolio_prior(context.master_portfolio_state, cutoff)


def assert_master_portfolio_prior(
    state: PortfolioSnapshot | dict[str, Any] | None,
    cutoff: dt.datetime,
) -> None:
    if state is None:
        return
    payload = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    snapshot_as_of = _coerce_datetime(payload.get("as_of"))
    if snapshot_as_of is not None and not _is_before(snapshot_as_of, cutoff):
        raise AssertionError("MasterPortfolio snapshot must precede the decision cutoff")
    snapshot_date = _coerce_date(payload.get("date"))
    if snapshot_date is not None and snapshot_date >= cutoff.date():
        raise AssertionError("MasterPortfolio snapshot must come from a prior date")


def _coerce_datetime(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _coerce_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    return None


def _is_before(left: dt.datetime, right: dt.datetime) -> bool:
    if left.tzinfo is None and right.tzinfo is not None:
        left = left.replace(tzinfo=right.tzinfo)
    elif left.tzinfo is not None and right.tzinfo is None:
        right = right.replace(tzinfo=left.tzinfo)
    return left < right


def assert_linked_context_hashes(
    context: AgentContext,
    text_feature: TextFeature,
    pending: PendingDecision,
) -> None:
    if not context.context_hash or len({context.context_hash, text_feature.context_hash, pending.context_hash}) != 1:
        raise AssertionError("Context hash mismatch across decision artifacts")


def assert_outcome_alignment(outcome: OutcomeTable, pending: PendingDecision) -> None:
    if outcome.outcome_date != pending.expected_outcome_date:
        raise AssertionError("Outcome is not aligned to next common trading date")
    if outcome.available_at <= pending.decision_cutoff:
        raise AssertionError("Outcome was available before its decision")


def assert_graph_contract(feature_names: list[str], feature_matrix: Any) -> None:
    forbidden = FORBIDDEN_GRAPH_FIELDS.intersection(feature_names)
    if forbidden:
        raise AssertionError(f"Forbidden graph feature(s): {sorted(forbidden)}")
    if getattr(feature_matrix, "shape", (0, 0))[-1] != len(feature_names):
        raise AssertionError("Graph feature matrix width does not match feature names")
    if tuple(feature_names) != CANONICAL_FEATURE_NAMES:
        raise AssertionError(
            "Graph H0 must use the canonical market/action/text/master/psychology order"
        )


def assert_unique_ids(values: Iterable[str], artifact_name: str) -> None:
    counts = Counter(values)
    duplicate = [value for value, count in counts.items() if count > 1]
    if duplicate:
        raise AssertionError(f"Duplicate {artifact_name} IDs: {duplicate[:5]}")
