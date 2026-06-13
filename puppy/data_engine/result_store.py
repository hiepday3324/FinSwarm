from __future__ import annotations

import datetime as dt
from typing import Any

from puppy.common.schemas import PortfolioSnapshot, QuantFeatureTable

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

    def read_portfolio_snapshots(
        self, date: dt.date | str | None = None
    ) -> list[PortfolioSnapshot]:
        return [
            PortfolioSnapshot(**payload)
            for payload in self.read_artifacts("PortfolioSnapshot", date=date)
        ]

    def to_dataframe(self, artifact_type: str, date: dt.date | str | None = None):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required to export EventStore artifacts to DataFrame."
            ) from exc
        return pd.DataFrame(self.read_artifacts(artifact_type, date=date))
