from __future__ import annotations

import datetime as dt
from typing import Any

from puppy.common.schemas import (
    AgentOutput,
    AllocationDecision,
    AllocationWeights,
    DebateRequest,
    DebateTranscript,
    DebateVerdict,
    FusionOutput,
    GraphOutput,
    MemoryShareEvent,
    MemoryShareRoute,
    PortfolioSnapshot,
    QuantFeatureTable,
    TextFeature,
)

from .event_store import EventStore


class ResultStore:
    """Typed readers over EventStore JSONL artifacts."""

    def __init__(self, event_store: EventStore | str = "data/event_store/events.jsonl") -> None:
        self.event_store = event_store if isinstance(event_store, EventStore) else EventStore(event_store)

    def read_artifacts(
        self, artifact_type: str, date: dt.date | str | None = None
    ) -> list[dict[str, Any]]:
        return [
            event["payload"]
            for event in self.event_store.read_events(artifact_type=artifact_type, date=date)
        ]

    def read_quant_feature_tables(
        self, date: dt.date | str | None = None
    ) -> list[QuantFeatureTable]:
        return [
            QuantFeatureTable(**payload)
            for payload in self.read_artifacts("QuantFeatureTable", date=date)
        ]

    def read_agent_outputs(
        self, date: dt.date | str | None = None
    ) -> list[AgentOutput]:
        return [
            AgentOutput(**payload)
            for payload in self.read_artifacts("AgentOutput", date=date)
        ]

    def read_text_features(
        self, date: dt.date | str | None = None
    ) -> list[TextFeature]:
        return [
            TextFeature(**payload)
            for payload in self.read_artifacts("TextFeature", date=date)
        ]

    def read_portfolio_snapshots(
        self, date: dt.date | str | None = None
    ) -> list[PortfolioSnapshot]:
        return [
            PortfolioSnapshot(**payload)
            for payload in self.read_artifacts("PortfolioSnapshot", date=date)
        ]

    def read_graph_outputs(
        self,
        date: dt.date | str | None = None,
    ) -> list[GraphOutput]:
        return [
            GraphOutput(**payload)
            for payload in self.read_artifacts("GraphOutput", date=date)
        ]

    def read_memory_share_routes(
        self,
        date: dt.date | str | None = None,
        source_agent_id: str | None = None,
        target_agent_id: str | None = None,
        source_symbol: str | None = None,
        target_symbol: str | None = None,
    ) -> list[MemoryShareRoute]:
        routes = [
            MemoryShareRoute(**payload)
            for payload in self.read_artifacts("MemoryShareRoute", date=date)
        ]
        if source_agent_id is not None:
            routes = [
                route for route in routes if route.source_agent_id == source_agent_id
            ]
        if target_agent_id is not None:
            routes = [
                route for route in routes if route.target_agent_id == target_agent_id
            ]
        if source_symbol is not None:
            routes = [route for route in routes if route.source_symbol == source_symbol]
        if target_symbol is not None:
            routes = [route for route in routes if route.target_symbol == target_symbol]
        return routes

    def read_memory_share_events(
        self,
        date: dt.date | str | None = None,
        source_agent_id: str | None = None,
        target_agent_id: str | None = None,
    ) -> list[MemoryShareEvent]:
        events = [
            MemoryShareEvent(**payload)
            for payload in self.read_artifacts("MemoryShareEvent", date=date)
        ]
        if source_agent_id is not None:
            events = [
                event
                for event in events
                if event.route.source_agent_id == source_agent_id
            ]
        if target_agent_id is not None:
            events = [
                event
                for event in events
                if event.route.target_agent_id == target_agent_id
            ]
        return events

    def read_fusion_outputs(
        self,
        date: dt.date | str | None = None,
    ) -> list[FusionOutput]:
        return [
            FusionOutput(**payload)
            for payload in self.read_artifacts("FusionOutput", date=date)
        ]

    def read_allocation_weights(
        self,
        date: dt.date | str | None = None,
    ) -> list[AllocationWeights]:
        return [
            AllocationWeights(**payload)
            for payload in self.read_artifacts("AllocationWeights", date=date)
        ]

    def read_allocation_decisions(
        self,
        date: dt.date | str | None = None,
    ) -> list[AllocationDecision]:
        return [
            AllocationDecision(**payload)
            for payload in self.read_artifacts("AllocationDecision", date=date)
        ]

    def read_debate_requests(
        self,
        date: dt.date | str | None = None,
        agent_id: str | None = None,
        symbol: str | None = None,
    ) -> list[DebateRequest]:
        requests = [
            DebateRequest(**payload)
            for payload in self.read_artifacts("DebateRequest", date=date)
        ]
        if agent_id is not None:
            requests = [request for request in requests if request.agent_id == agent_id]
        if symbol is not None:
            requests = [request for request in requests if request.symbol == symbol]
        return requests

    def read_debate_transcripts(
        self,
        date: dt.date | str | None = None,
        agent_id: str | None = None,
        symbol: str | None = None,
    ) -> list[DebateTranscript]:
        transcripts = [
            DebateTranscript(**payload)
            for payload in self.read_artifacts("DebateTranscript", date=date)
        ]
        if agent_id is not None:
            transcripts = [
                transcript
                for transcript in transcripts
                if transcript.agent_id == agent_id
            ]
        if symbol is not None:
            transcripts = [
                transcript for transcript in transcripts if transcript.symbol == symbol
            ]
        return transcripts

    def read_debate_verdicts(
        self,
        date: dt.date | str | None = None,
        agent_id: str | None = None,
        symbol: str | None = None,
        verdict: str | None = None,
    ) -> list[DebateVerdict]:
        verdicts = [
            DebateVerdict(**payload)
            for payload in self.read_artifacts("DebateVerdict", date=date)
        ]
        if agent_id is not None:
            verdicts = [item for item in verdicts if item.agent_id == agent_id]
        if symbol is not None:
            verdicts = [item for item in verdicts if item.symbol == symbol]
        if verdict is not None:
            verdicts = [item for item in verdicts if item.verdict == verdict]
        return verdicts

    def to_dataframe(self, artifact_type: str, date: dt.date | str | None = None):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required to export EventStore artifacts to DataFrame."
            ) from exc
        return pd.DataFrame(self.read_artifacts(artifact_type, date=date))
