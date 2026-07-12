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
    ShadowPortfolioState,
    TextFeature,
)


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


def assert_shadow_prior(state: ShadowPortfolioState | None, decision_date: dt.date) -> None:
    if state and state.last_resolved_decision_date and state.last_resolved_decision_date >= decision_date:
        raise AssertionError("ShadowPortfolio includes current/future decision outcome")


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


def assert_unique_ids(values: Iterable[str], artifact_name: str) -> None:
    counts = Counter(values)
    duplicate = [value for value, count in counts.items() if count > 1]
    if duplicate:
        raise AssertionError(f"Duplicate {artifact_name} IDs: {duplicate[:5]}")
