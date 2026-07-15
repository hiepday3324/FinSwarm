from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from demo.constants import (
    DEFAULT_BUNDLE_DIR,
    DISPLAY_ORDER,
    NODE_ORDER,
    SCENARIO_DIR,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


class ReplayStore:
    def __init__(self, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        if not self.bundle_dir.is_dir():
            raise FileNotFoundError(f"Demo-fit bundle not found: {self.bundle_dir}")
        self.manifest = json.loads(
            (self.bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("bundle_version") != "finswarm-e4-demo-fit-v1":
            raise ValueError("Unsupported E4 demo-fit bundle version")
        if tuple(self.manifest.get("node_order", ())) != NODE_ORDER:
            raise ValueError("Demo bundle node order does not match the audited contract")
        self._validate_hashes()
        self.index = json.loads(
            (self.bundle_dir / "replay_index.json").read_text(encoding="utf-8")
        )
        self.dates = tuple(dt.date.fromisoformat(value) for value in self.index["dates"])
        self.symbols = tuple(self.index["symbols"])
        if self.symbols != NODE_ORDER:
            raise ValueError("Replay index changed the canonical node order")
        self.feature_schema = json.loads(
            (self.bundle_dir / "feature_schema.json").read_text(encoding="utf-8")
        )
        self.training_metadata = json.loads(
            (self.bundle_dir / "training_metadata.json").read_text(encoding="utf-8")
        )
        record_rows = pq.read_table(self.bundle_dir / "replay_records.parquet").to_pylist()
        self._records: dict[tuple[dt.date, str], dict[str, Any]] = {}
        for row in record_rows:
            key = (row["decision_date"], str(row["symbol"]))
            if key in self._records:
                raise ValueError(f"Duplicate replay Agent row: {key}")
            enriched = dict(row)
            enriched["context_snapshot"] = _json(row["context_snapshot_json"], {})
            enriched["memory_refs"] = _json(row["memory_refs_json"], [])
            enriched["news_event_ids"] = _json(row["news_event_ids_json"], [])
            enriched["psychology_state"] = _json(row["psychology_state_json"], None)
            enriched["shadow_state"] = _json(row["shadow_state_json"], None)
            enriched["raw_agent_output"] = _json(row["raw_agent_output_json"], {})
            enriched["causal_provenance"] = _json(
                row["causal_provenance_json"], {}
            )
            enriched["cited_memory_ids"] = _json(row["cited_memory_ids_json"], {})
            self._records[key] = enriched
        expected = len(self.dates) * len(self.symbols)
        if len(self._records) != expected:
            raise ValueError(f"Expected {expected} replay Agent rows, got {len(self._records)}")

        news_path = self.bundle_dir / "replay_news.parquet"
        news_rows = pq.read_table(news_path).to_pylist()
        self._news_by_date: dict[dt.date, list[dict[str, Any]]] = {}
        for row in news_rows:
            enriched = dict(row)
            enriched["event_ids"] = _json(row["event_ids_json"], [])
            enriched["related_symbols"] = _json(row["related_symbols_json"], [])
            enriched["agents_received"] = _json(row["agents_received_json"], [])
            self._news_by_date.setdefault(row["decision_date"], []).append(enriched)
        for rows in self._news_by_date.values():
            rows.sort(key=lambda row: (row["published_at"], row["news_group_id"]))

    def _validate_hashes(self) -> None:
        for name, expected in self.manifest.get("file_sha256", {}).items():
            path = self.bundle_dir / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"Demo bundle hash mismatch: {name}")

    def date_index(self, value: dt.date | str) -> int:
        date = dt.date.fromisoformat(value) if isinstance(value, str) else value
        try:
            return self.dates.index(date)
        except ValueError as error:
            raise ValueError(f"Date is outside the train replay index: {date}") from error

    def records_for_date(self, value: dt.date | str) -> list[dict[str, Any]]:
        date = dt.date.fromisoformat(value) if isinstance(value, str) else value
        rows = [self._records[(date, symbol)] for symbol in DISPLAY_ORDER]
        decision_ids = [str(row["decision_id"]) for row in rows]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError(f"Duplicate decision IDs in replay date {date}")
        return rows

    def record(self, value: dt.date | str, symbol: str) -> dict[str, Any]:
        date = dt.date.fromisoformat(value) if isinstance(value, str) else value
        return self._records[(date, str(symbol))]

    def news_for_date(self, value: dt.date | str) -> list[dict[str, Any]]:
        date = dt.date.fromisoformat(value) if isinstance(value, str) else value
        return list(self._news_by_date.get(date, ()))

    def scenarios(self) -> list[dict[str, Any]]:
        path = SCENARIO_DIR / "index.json"
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("scenarios", []))

    def previous_date(self, value: dt.date | str) -> dt.date | None:
        index = self.date_index(value)
        return None if index == 0 else self.dates[index - 1]
