from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from puppy.common.schemas import AgentOutput

from .rollout_state import canonical_json


JOURNAL_VERSION = "stage5-causal-decision-journal-v1"
JOURNAL_DIRECTORY = "causal_decision_journal"

_CITATION_FIELDS = {
    "short": "short_memory_index",
    "middle": "middle_memory_index",
    "long": "long_memory_index",
    "reflection": "reflection_memory_index",
}


class CausalDecisionJournalError(ValueError):
    pass


def eligible_memory_set_hash(eligible_memory_ids: Mapping[str, list[int]]) -> str:
    normalized = {
        str(layer): sorted({int(value) for value in values})
        for layer, values in sorted(eligible_memory_ids.items())
    }
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def cited_memory_ids(output: AgentOutput) -> dict[str, list[int]]:
    cited: dict[str, list[int]] = {}
    for layer, field in _CITATION_FIELDS.items():
        values: list[int] = []
        for item in output.raw_reflection.get(field, []):
            value = item.get("memory_index") if isinstance(item, dict) else None
            if isinstance(value, bool) or not isinstance(value, int):
                raise CausalDecisionJournalError(
                    f"Invalid cited memory ID in journal output field {field}"
                )
            values.append(value)
        cited[layer] = values
    return cited


def runtime_decision_contract(agent: Any, run_context: Any) -> dict[str, Any]:
    llm_agent = agent.llm_agent
    backend = getattr(llm_agent, "chat_backend", None)
    public_config = getattr(backend, "public_config", {})
    public_config = dict(public_config) if isinstance(public_config, Mapping) else {}
    return {
        "provider": str(
            public_config.get(
                "provider", getattr(backend, "service_provider", "unknown-provider")
            )
        ),
        "interface": str(
            public_config.get(
                "interface", getattr(backend, "provider", "unknown-interface")
            )
        ),
        "base_url": str(public_config.get("base_url", getattr(backend, "base_url", "unknown"))),
        "requested_model": str(getattr(llm_agent, "model_name", "unknown")),
        "prompt_version": str(run_context.prompt_version),
        "schema_version": str(
            getattr(llm_agent, "causal_schema_version", "unknown-schema")
        ),
        "generation_settings": dict(
            run_context.metadata.get("generation_parameters", {})
        ),
    }


class CausalDecisionJournal:
    """Append-only atomic storage for canonical Stage 5 AgentOutput records."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        rollout_fingerprint: str,
        symbols: list[str] | tuple[str, ...],
    ) -> None:
        self.directory = Path(state_dir) / JOURNAL_DIRECTORY
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rollout_fingerprint = str(rollout_fingerprint)
        self.symbols = tuple(str(symbol) for symbol in symbols)
        self._records: dict[tuple[dt.date, str], dict[str, Any]] = {}
        self._load()

    def __len__(self) -> int:
        return len(self._records)

    def get(self, date: dt.date, symbol: str) -> dict[str, Any] | None:
        return self._records.get((date, symbol))

    @staticmethod
    def output(record: Mapping[str, Any]) -> AgentOutput:
        return AgentOutput.model_validate(record["agent_output"]).model_copy(deep=True)

    def append(
        self,
        *,
        output: AgentOutput,
        symbol: str,
        decision_date: dt.date,
        eligible_memory_ids: Mapping[str, list[int]],
        runtime_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = self._build_record(
            output=output,
            symbol=symbol,
            decision_date=decision_date,
            eligible_memory_ids=eligible_memory_ids,
            runtime_contract=runtime_contract,
        )
        key = (decision_date, symbol)
        existing = self._records.get(key)
        if existing is not None:
            if existing["record_hash"] != record["record_hash"]:
                raise CausalDecisionJournalError(
                    f"Append-only journal conflict for {symbol} on {decision_date}"
                )
            return existing

        target = self._path(decision_date, symbol)
        if target.exists():
            on_disk = self._read_record(target)
            if on_disk["record_hash"] != record["record_hash"]:
                raise CausalDecisionJournalError(
                    f"Append-only journal conflict for {symbol} on {decision_date}"
                )
            self._records[key] = on_disk
            return on_disk

        temporary = self.directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
        serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                on_disk = self._read_record(target)
                if on_disk["record_hash"] != record["record_hash"]:
                    raise CausalDecisionJournalError(
                        f"Append-only journal conflict for {symbol} on {decision_date}"
                    )
                record = on_disk
        finally:
            temporary.unlink(missing_ok=True)

        self._records[key] = record
        return record

    def validate_for_reuse(
        self,
        record: Mapping[str, Any],
        *,
        output: AgentOutput,
        symbol: str,
        decision_date: dt.date,
        decision_id: str,
        context_hash: str,
        eligible_memory_ids: Mapping[str, list[int]],
        runtime_contract: Mapping[str, Any],
    ) -> None:
        normalized_eligible = self._normalize_eligible(eligible_memory_ids)
        expected = {
            "rollout_fingerprint": self.rollout_fingerprint,
            "symbol": symbol,
            "decision_date": decision_date.isoformat(),
            "decision_id": decision_id,
            "context_hash": context_hash,
            "eligible_memory_ids": normalized_eligible,
            "eligible_memory_set_hash": eligible_memory_set_hash(normalized_eligible),
            **dict(runtime_contract),
        }
        for field, value in expected.items():
            if canonical_json(record.get(field)) != canonical_json(value):
                raise CausalDecisionJournalError(
                    f"Decision journal {field} mismatch for {symbol} on {decision_date}"
                )
        if canonical_json(record["agent_output"]) != canonical_json(
            output.model_dump(mode="json")
        ):
            raise CausalDecisionJournalError(
                f"Decision journal AgentOutput mismatch for {symbol} on {decision_date}"
            )

    def _load(self) -> None:
        for path in sorted(self.directory.glob("*.json")):
            record = self._read_record(path)
            date = dt.date.fromisoformat(record["decision_date"])
            symbol = str(record["symbol"])
            key = (date, symbol)
            if key in self._records:
                raise CausalDecisionJournalError(
                    f"Duplicate decision journal record for {symbol} on {date}"
                )
            self._records[key] = record

    def _read_record(self, path: Path) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CausalDecisionJournalError(
                f"Invalid decision journal record: {path.name}"
            ) from exc
        if not isinstance(record, dict):
            raise CausalDecisionJournalError(
                f"Decision journal record must be an object: {path.name}"
            )
        return self._validate_record(record, path.name)

    def _validate_record(self, record: dict[str, Any], name: str) -> dict[str, Any]:
        stored_hash = record.get("record_hash")
        payload = {key: value for key, value in record.items() if key != "record_hash"}
        calculated_hash = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        if stored_hash != calculated_hash:
            raise CausalDecisionJournalError(
                f"Decision journal record hash mismatch: {name}"
            )
        if record.get("journal_version") != JOURNAL_VERSION:
            raise CausalDecisionJournalError(
                f"Decision journal version mismatch: {name}"
            )
        if record.get("rollout_fingerprint") != self.rollout_fingerprint:
            raise CausalDecisionJournalError(
                f"Decision journal rollout fingerprint mismatch: {name}"
            )
        symbol = str(record.get("symbol", ""))
        if symbol not in self.symbols:
            raise CausalDecisionJournalError(
                f"Decision journal symbol mismatch: {name}"
            )
        try:
            decision_date = dt.date.fromisoformat(str(record["decision_date"]))
            output = AgentOutput.model_validate(record["agent_output"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CausalDecisionJournalError(
                f"Decision journal canonical output is invalid: {name}"
            ) from exc
        if output.signal.symbol != symbol or output.signal.date != decision_date:
            raise CausalDecisionJournalError(
                f"Decision journal signal identity mismatch: {name}"
            )
        if not output.parse_ok or not output.decision_id:
            raise CausalDecisionJournalError(
                f"Decision journal contains an unvalidated output: {name}"
            )
        decision = str(output.signal.decision).lower()
        expected_action = {"sell": -1, "hold": 0, "buy": 1}.get(decision)
        if expected_action is None or output.signal.action_signal != expected_action:
            raise CausalDecisionJournalError(
                f"Decision journal action contract mismatch: {name}"
            )
        if record.get("decision_id") != output.decision_id:
            raise CausalDecisionJournalError(
                f"Decision journal decision ID mismatch: {name}"
            )
        if record.get("context_hash") != output.context_hash:
            raise CausalDecisionJournalError(
                f"Decision journal context hash mismatch: {name}"
            )
        eligible = self._normalize_eligible(record.get("eligible_memory_ids", {}))
        if record.get("eligible_memory_set_hash") != eligible_memory_set_hash(eligible):
            raise CausalDecisionJournalError(
                f"Decision journal eligible-memory hash mismatch: {name}"
            )
        cited = cited_memory_ids(output)
        if canonical_json(record.get("cited_memory_ids")) != canonical_json(cited):
            raise CausalDecisionJournalError(
                f"Decision journal cited-memory mismatch: {name}"
            )
        for layer, values in cited.items():
            if not set(values).issubset(set(eligible.get(layer, []))):
                raise CausalDecisionJournalError(
                    f"Decision journal cites ineligible {layer} memory: {name}"
                )
        return record

    def _build_record(
        self,
        *,
        output: AgentOutput,
        symbol: str,
        decision_date: dt.date,
        eligible_memory_ids: Mapping[str, list[int]],
        runtime_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        if output.signal.symbol != symbol or output.signal.date != decision_date:
            raise CausalDecisionJournalError("Cannot journal a mismatched AgentOutput")
        if not output.parse_ok or not output.decision_id or not output.context_hash:
            raise CausalDecisionJournalError(
                "Only complete validated AgentOutput records may be journaled"
            )
        eligible = self._normalize_eligible(eligible_memory_ids)
        provenance = dict(output.metadata.get("causal_provenance", {}))
        payload = {
            "journal_version": JOURNAL_VERSION,
            "rollout_fingerprint": self.rollout_fingerprint,
            "symbol": symbol,
            "decision_date": decision_date.isoformat(),
            "decision_id": output.decision_id,
            "decision": str(output.signal.decision),
            "action_signal": int(output.signal.action_signal),
            "reason": output.signal.reason,
            "cited_memory_ids": cited_memory_ids(output),
            "context_hash": output.context_hash,
            "eligible_memory_ids": eligible,
            "eligible_memory_set_hash": eligible_memory_set_hash(eligible),
            **dict(runtime_contract),
            "returned_model": provenance.get("returned_model"),
            "response_hash": provenance.get("response_hash"),
            "retry_count": int(provenance.get("retry_count", 0)),
            "provenance": provenance,
            "agent_output": output.model_dump(mode="json"),
        }
        payload["record_hash"] = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return self._validate_record(payload, f"{decision_date}__{symbol}.json")

    @staticmethod
    def _normalize_eligible(
        eligible_memory_ids: Mapping[str, list[int]] | Any,
    ) -> dict[str, list[int]]:
        if not isinstance(eligible_memory_ids, Mapping):
            raise CausalDecisionJournalError("Eligible memory IDs must be a mapping")
        normalized: dict[str, list[int]] = {}
        for layer in _CITATION_FIELDS:
            raw_values = eligible_memory_ids.get(layer, [])
            if not isinstance(raw_values, list):
                raise CausalDecisionJournalError(
                    f"Eligible {layer} memory IDs must be a list"
                )
            values = []
            for value in raw_values:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise CausalDecisionJournalError(
                        f"Eligible {layer} memory ID must be an integer"
                    )
                values.append(value)
            normalized[layer] = sorted(set(values))
        return normalized

    def _path(self, decision_date: dt.date, symbol: str) -> Path:
        if symbol not in self.symbols:
            raise CausalDecisionJournalError(f"Unknown journal symbol: {symbol}")
        return self.directory / f"{decision_date.isoformat()}__{symbol}.json"
