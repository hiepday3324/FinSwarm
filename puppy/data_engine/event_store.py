from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

from puppy.common.schemas import AgentOutput, TextFeature


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return payload
    raise TypeError("Event payload must be a dict or Pydantic model.")


class EventStore:
    """Append-only JSONL store for typed pipeline artifacts."""

    def __init__(self, path: str = "data/event_store/events.jsonl") -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def append_event(
        self,
        artifact_type: str,
        date: dt.date | str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "artifact_type": artifact_type,
            "date": date.isoformat() if isinstance(date, dt.date) else str(date),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "payload": _payload_to_dict(payload),
            "metadata": metadata or {},
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_json_default, sort_keys=True) + "\n")
        return event

    def append_agent_output(
        self,
        date: dt.date | str,
        agent_output: AgentOutput,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.append_event("AgentOutput", date, agent_output, metadata=metadata)

    def append_text_feature(
        self,
        date: dt.date | str,
        text_feature: TextFeature,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.append_event("TextFeature", date, text_feature, metadata=metadata)

    def read_events(
        self,
        artifact_type: str | None = None,
        date: dt.date | str | None = None,
    ) -> list[dict[str, Any]]:
        return list(self.iter_events(artifact_type=artifact_type, date=date))

    def iter_events(
        self,
        artifact_type: str | None = None,
        date: dt.date | str | None = None,
    ) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return

        date_str = date.isoformat() if isinstance(date, dt.date) else str(date) if date is not None else None
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if artifact_type is not None and event.get("artifact_type") != artifact_type:
                    continue
                if date_str is not None and event.get("date") != date_str:
                    continue
                yield event
