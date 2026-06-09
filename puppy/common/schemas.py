from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    AllocationAction,
    DebateVerdictAction,
    Decision,
    DecisionMode,
    MemoryLayer,
)


class SwarmBaseModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)


class MemoryRef(SwarmBaseModel):
    memory_id: int | str
    layer: MemoryLayer | str
    symbol: str | None = None
    date: dt.date | None = None
    score: float | None = None
    source_agent_id: str | None = None
    text_preview: str | None = None


class SharedContext(SwarmBaseModel):
    source_agent_id: str
    target_agent_id: str
    source_symbol: str | None = None
    target_symbol: str | None = None
    text: str
    reason: str
    alpha: float | None = None
    created_at: dt.date | None = None
    ttl_days: int | None = 1
    memory_refs: list[MemoryRef] = Field(default_factory=list)


class AgentContext(SwarmBaseModel):
    agent_id: str
    symbol: str
    date: dt.date
    latest_reason: str | None = None
    short_memory: list[str] = Field(default_factory=list)
    mid_memory: list[str] = Field(default_factory=list)
    long_memory: list[str] = Field(default_factory=list)
    reflection_memory: list[str] = Field(default_factory=list)
    memory_refs: list[MemoryRef] = Field(default_factory=list)
    shared_context: list[SharedContext] = Field(default_factory=list)
    shadow_state: dict[str, Any] | None = None
    psychology_state: dict[str, Any] | None = None


class TextFeature(SwarmBaseModel):
    agent_id: str
    symbol: str
    date: dt.date
    h_text: list[float]
    raw_context: str
    embedding_model: str
    dim: int
    dtype: str = "float32"


class AgentSignal(SwarmBaseModel):
    agent_id: str
    symbol: str
    date: dt.date
    decision: Decision | str
    action_signal: int
    reason: str
    memory_refs: list[MemoryRef] = Field(default_factory=list)
    shared_context_used: list[MemoryRef] = Field(default_factory=list)


class AgentOutput(SwarmBaseModel):
    signal: AgentSignal
    raw_reflection: dict[str, Any] = Field(default_factory=dict)
    h_text: list[float] | None = None
    raw_context: str | None = None
    model_name: str | None = None
    parse_ok: bool = True


class AllocationDecision(SwarmBaseModel):
    date: dt.date | None = None
    action: AllocationAction | str
    score: int
    reason: str
    signals: list[AgentSignal] = Field(default_factory=list)


class MemoryShareRoute(SwarmBaseModel):
    date: dt.date
    source_agent_id: str
    target_agent_id: str
    source_symbol: str
    target_symbol: str
    alpha: float
    reason: str
    layer: MemoryLayer | str = MemoryLayer.SHORT
    top_k: int = 3
    query_text: str | None = None
    ttl_days: int | None = 1


class MemoryShareEvent(SwarmBaseModel):
    date: dt.date
    route: MemoryShareRoute
    shared_context: SharedContext
    shared_memory_ids: list[int | str] = Field(default_factory=list)
    deduped: bool = False


class DebateRequest(SwarmBaseModel):
    date: dt.date
    agent_id: str
    symbol: str
    local_decision: Decision | str
    text_signal: int
    graph_signal: int
    fusion_score: float
    reason: str
    mode: DecisionMode | str = DecisionMode.DEBATE
    agent_signal: AgentSignal | None = None


class DebateTranscript(SwarmBaseModel):
    date: dt.date
    agent_id: str
    symbol: str
    request: DebateRequest
    question: str
    answer: str
    cited_memory_refs: list[MemoryRef] = Field(default_factory=list)


class DebateVerdict(SwarmBaseModel):
    date: dt.date
    agent_id: str
    symbol: str
    verdict: DebateVerdictAction | str
    reason: str
    transcript: DebateTranscript
    target_decision: Decision | str | None = None


# --- Milestone 2 Additions ---

class PriceBar(SwarmBaseModel):
    symbol: str
    date: dt.date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    adj_close: float | None = None
    volume: float | int | None = None
    source: str | None = None


class NewsEvent(SwarmBaseModel):
    event_id: str
    symbol: str
    published_at: dt.datetime
    available_at: dt.datetime
    title: str | None = None
    text: str
    source: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilingEvent(SwarmBaseModel):
    filing_id: str
    symbol: str
    filing_type: str
    filing_date: dt.date
    accepted_at: dt.datetime | None = None
    available_at: dt.datetime
    text: str
    source: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketStep(SwarmBaseModel):
    date: dt.date
    symbols: list[str]
    prices: dict[str, PriceBar]
    news: dict[str, list[NewsEvent]] = Field(default_factory=dict)
    filings: dict[str, list[FilingEvent]] = Field(default_factory=dict)
    as_of: dt.datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantFeatureTable(SwarmBaseModel):
    date: dt.date
    symbols: list[str]
    action_signals: dict[str, int]
    returns: dict[str, float | None]
    labels: dict[str, int | float | None] = Field(default_factory=dict)
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShadowPortfolioState(SwarmBaseModel):
    date: dt.date
    agent_id: str
    symbol: str
    value: float
    roi: float
    rolling_return: float | None = None
    sharpe: float | None = None
    drawdown: float | None = None
    win_rate: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioSnapshot(SwarmBaseModel):
    date: dt.date
    cash: float
    positions: dict[str, float]
    weights: dict[str, float]
    equity: float
    daily_return: float | None = None
    drawdown: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
