from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from puppy.data_engine.normalization import load_stage4_config
from puppy.graph.graph_dataset import CANONICAL_FEATURE_NAMES
from puppy.swarm.psychology import build_psychology_state

from .rollout_state import atomic_write_json, atomic_write_text, sha256_path
from .temporal_assertions import FORBIDDEN_GRAPH_FIELDS


REQUIRED_FILES = [
    "text_samples.parquet", "agent_outputs.parquet", "observed_features.parquet",
    "outcomes.parquet", "pending_decisions.parquet", "reflection_updates.parquet",
    "graph_samples.npz", "feature_schema.json", "data_manifest.json",
    "rollout_manifest.json", "temporal_quality.json", "temporal_quality.md",
    "state/checkpoint.json",
]


def _records(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _parse_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _parse_datetime(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _forbidden_context_fields(value: Any) -> int:
    forbidden = {
        "target",
        "label",
        "future_return",
        "asset_return",
        "strategy_return",
        "direction_label",
        "current_decision_result",
    }
    if isinstance(value, dict):
        return sum(
            int(str(key).lower() in forbidden) + _forbidden_context_fields(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_forbidden_context_fields(item) for item in value)
    return 0


def _secret_material_count(value: Any) -> int:
    text = json.dumps(value, sort_keys=True, default=str).lower()
    markers = (
        "finmem_llm_api_key",
        "openai_api_key",
        '"api_key"',
        '"authorization"',
        "bearer sk-",
        '"headers"',
    )
    return sum(text.count(marker) for marker in markers)


def validate_training_artifacts(
    config_path: str,
    run_id: str,
    *,
    require_final: bool = True,
) -> dict[str, Any]:
    config = load_stage4_config(config_path)
    output_dir = config.repo_root / "data/training" / run_id
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 5 artifact(s): {missing}")
    data_manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
    current_hashes = {
        config.relative(path): sha256_path(path)
        for path in (config.canonical_prices, config.canonical_news, config.canonical_filings)
    }
    if data_manifest["input_hashes"] != current_hashes:
        raise ValueError("Stage 4 canonical input hashes no longer match artifact manifest")
    text_rows = _records(output_dir / "text_samples.parquet")
    output_rows = _records(output_dir / "agent_outputs.parquet")
    observed_rows = _records(output_dir / "observed_features.parquet")
    outcome_rows = _records(output_dir / "outcomes.parquet")
    pending_rows = _records(output_dir / "pending_decisions.parquet")
    reflection_rows = _records(output_dir / "reflection_updates.parquet")
    rollout_manifest = json.loads(
        (output_dir / "rollout_manifest.json").read_text(encoding="utf-8")
    )
    feature_schema = json.loads((output_dir / "feature_schema.json").read_text(encoding="utf-8"))
    graph = np.load(output_dir / "graph_samples.npz", allow_pickle=False)

    duplicate_ids = sum((
        _duplicate_count([row["text_sample_id"] for row in text_rows]),
        _duplicate_count([row["decision_id"] for row in output_rows]),
        _duplicate_count([row["decision_id"] for row in pending_rows]),
        _duplicate_count([row["outcome_id"] for row in outcome_rows]),
        _duplicate_count([row["reflection_id"] for row in reflection_rows]),
        _duplicate_count(graph["sample_ids"].astype(str).tolist()),
    ))
    pending_by_id = {row["decision_id"]: row for row in pending_rows}
    text_by_decision = {row["decision_id"]: row for row in text_rows}
    output_by_decision = {row["decision_id"]: row for row in output_rows}
    context_mismatches = 0
    prompt_version_mismatches = 0
    future_event_violations = 0
    same_day_reflections = 0
    for decision_id, pending in pending_by_id.items():
        linked = [pending.get("context_hash")]
        if decision_id in text_by_decision:
            linked.append(text_by_decision[decision_id].get("context_hash"))
        if decision_id in output_by_decision:
            linked.append(output_by_decision[decision_id].get("context_hash"))
        context_mismatches += int(len(set(linked)) != 1)
        prompt_versions = [pending.get("prompt_version"), data_manifest.get("prompt_version")]
        if decision_id in text_by_decision:
            prompt_versions.append(text_by_decision[decision_id].get("prompt_version"))
        if decision_id in output_by_decision:
            prompt_versions.append(output_by_decision[decision_id].get("prompt_version"))
        prompt_version_mismatches += int(
            None in prompt_versions or len(set(prompt_versions)) != 1
        )
        cutoff = pending["decision_cutoff"]
        refs = _parse_json(pending.get("memory_refs")) or []
        for ref in refs:
            available = ref.get("available_at")
            if available:
                future_event_violations += int(_parse_datetime(available) > cutoff)
            if str(ref.get("layer")) == "reflection" and ref.get("date"):
                same_day_reflections += int(dt.date.fromisoformat(ref["date"]) >= pending["decision_date"])
    future_price_violations = 0
    for row in observed_rows:
        dates = _parse_json(row["source_dates"]) or []
        future_price_violations += sum(dt.date.fromisoformat(value) > row["date"] for value in dates)
        metadata = _parse_json(row["metadata"]) or {}
        if metadata.get("max_source_date"):
            future_price_violations += int(dt.date.fromisoformat(metadata["max_source_date"]) > row["date"])
    alignment = 0
    for outcome in outcome_rows:
        pending = pending_by_id.get(outcome["decision_id"])
        alignment += int(pending is None or pending["expected_outcome_date"] != outcome["outcome_date"])
        alignment += int(outcome["available_at"] <= (pending["decision_cutoff"] if pending else outcome["available_at"]))
    graph_names = feature_schema["graph_feature_names"]
    if tuple(graph_names) != CANONICAL_FEATURE_NAMES:
        raise ValueError(
            "Stage 5 graph artifacts require regeneration: expected canonical "
            "market/action/text_score/prior_shadow/prior_psychology feature order"
        )
    forbidden = sorted(FORBIDDEN_GRAPH_FIELDS.intersection(graph_names))
    graph_violations = len(forbidden)
    graph_violations += int(graph["feature_matrix"].shape[-1] != len(graph_names))
    graph_violations += int(not feature_schema.get("graph_targets_separate"))
    graph_future = sum(
        dt.date.fromisoformat(source) > dt.date.fromisoformat(decision)
        for source, decision in zip(graph["max_source_dates"].astype(str), graph["decision_dates"].astype(str))
    )
    shadow_violations = 0
    psychology_violations = 0
    current_outcome_context_violations = 0
    for pending in pending_rows:
        snapshot = _parse_json(pending["context_snapshot"])
        agent_context = snapshot.get("agent_context") or {}
        state = agent_context.get("shadow_state")
        if state and state.get("last_resolved_decision_date"):
            shadow_violations += int(dt.date.fromisoformat(state["last_resolved_decision_date"]) >= pending["decision_date"])
        expected_psychology = build_psychology_state(state)
        psychology_violations += int(
            agent_context.get("psychology_state") != expected_psychology
        )
        current_outcome_context_violations += _forbidden_context_fields(snapshot)
        outcome = next(
            (
                row
                for row in outcome_rows
                if row["decision_id"] == pending["decision_id"]
            ),
            None,
        )
        if outcome is not None:
            serialized_snapshot = json.dumps(snapshot, sort_keys=True, default=str)
            current_outcome_context_violations += int(
                str(outcome["outcome_id"]) in serialized_snapshot
            )
    non_boundary_missing = sum(
        row["status"] not in {"resolved", "unresolved_end_of_data"}
        for row in pending_rows
    )
    required_symbols = set(config.symbols)
    output_symbols = {str(row["symbol"]) for row in output_rows}
    missing_symbol_count = len(required_symbols - output_symbols)
    decisions_by_date: dict[dt.date, set[str]] = {}
    for row in output_rows:
        decisions_by_date.setdefault(row["decision_date"], set()).add(str(row["symbol"]))
    missing_symbol_count += sum(
        len(required_symbols - symbols) for symbols in decisions_by_date.values()
    )
    parse_failures = sum(
        not bool(row.get("parse_ok"))
        or str(row.get("decision", "")).lower() not in {"buy", "hold", "sell"}
        or int(row.get("action_signal", 99)) not in {-1, 0, 1}
        or not row.get("decision_id")
        or not row.get("context_hash")
        or not row.get("prompt_version")
        or (
            require_final
            and (
                not row.get("model_name")
                or bool((_parse_json(row.get("metadata")) or {}).get("bootstrap", True))
                or bool((_parse_json(row.get("metadata")) or {}).get("synthetic", True))
                or not (_parse_json(row.get("metadata")) or {}).get(
                    "backend_identity"
                )
            )
        )
        for row in output_rows
    )
    provenance = data_manifest.get("generation_provenance") or {}
    final_contract = {
        "bootstrap_false": data_manifest.get("bootstrap") is False,
        "bootstrap_policy_absent": data_manifest.get("bootstrap_policy") is None,
        "real_action_source": data_manifest.get("action_source")
        == "real_causal_sector_agent",
        "external_llm_used": data_manifest.get("external_llm_used") is True,
        "non_synthetic_provenance": not bool(provenance.get("synthetic", True)),
        "final_training_eligible": data_manifest.get("final_training_eligible")
        is True,
        "full_interval": data_manifest.get("full_interval") is True,
        "rollout_final_training_eligible": rollout_manifest.get(
            "final_training_eligible"
        )
        is True,
        "rollout_full_interval": rollout_manifest.get("full_interval") is True,
    }
    final_contract_violations = (
        sum(not value for value in final_contract.values()) if require_final else 0
    )
    test_seal_violations = sum(
        str(row.get("split")) == "test"
        and bool((_parse_json(row.get("metadata")) or {}).get("used_for_training"))
        for row in text_rows + output_rows + observed_rows
    )
    secret_violations = _secret_material_count(
        {
            "data_manifest": data_manifest,
            "rollout_manifest": rollout_manifest,
            "feature_schema": feature_schema,
            "text_rows": text_rows,
            "output_rows": output_rows,
            "pending_rows": pending_rows,
        }
    )
    critical = {
        "duplicate_artifact_ids": duplicate_ids,
        "future_price_violations": future_price_violations + graph_future,
        "future_event_violations": future_event_violations,
        "same_day_reflection_violations": same_day_reflections,
        "shadow_timing_violations": shadow_violations,
        "psychology_prior_state_violations": psychology_violations,
        "current_outcome_context_violations": current_outcome_context_violations,
        "graph_forbidden_feature_violations": graph_violations,
        "outcome_alignment_violations": alignment,
        "context_hash_mismatches": context_mismatches,
        "prompt_version_mismatches": prompt_version_mismatches,
        "non_boundary_missing_outcomes": non_boundary_missing,
        "missing_required_symbols": missing_symbol_count,
        "invalid_final_decisions": parse_failures,
        "final_action_contract_violations": final_contract_violations,
        "test_split_seal_violations": test_seal_violations,
        "secret_material_violations": secret_violations,
    }
    previous = json.loads((output_dir / "temporal_quality.json").read_text(encoding="utf-8"))
    report = {
        **previous,
        "status": "PASS" if not any(critical.values()) else "FAIL",
        "critical_metrics": critical,
        "validated_artifact_counts": {
            "text_samples": len(text_rows), "agent_outputs": len(output_rows),
            "observed_features": len(observed_rows), "outcomes": len(outcome_rows),
            "pending_decisions": len(pending_rows), "reflection_updates": len(reflection_rows),
            "graph_samples": len(graph["sample_ids"]),
        },
        "graph_audit": {
            "feature_names": graph_names,
            "forbidden_fields_found": forbidden,
            "maximum_source_date_leq_decision_date": graph_future == 0,
            "targets_separate": graph_violations == 0,
        },
        "action_audit": {
            "source": data_manifest.get("action_source")
            or data_manifest.get("bootstrap_policy")
            or "unknown",
            "bootstrap": bool(data_manifest.get("bootstrap")),
            "external_llm_used": bool(data_manifest.get("external_llm_used")),
            "synthetic": bool(provenance.get("synthetic", True)),
            "contract_checks": final_contract,
            "final_training_eligible": (
                not final_contract_violations
                and not parse_failures
                and not missing_symbol_count
                and not secret_violations
            ),
        },
        "test_split_sealed": test_seal_violations == 0,
        "secrets_absent": secret_violations == 0,
    }
    atomic_write_json(output_dir / "temporal_quality.json", report)
    lines = ["# FinSwarm Stage 5 Temporal Quality", "", f"Status: **{report['status']}**", "", "## Critical metrics", ""]
    lines.extend(f"- `{name}`: {value}" for name, value in critical.items())
    lines.append("")
    atomic_write_text(output_dir / "temporal_quality.md", "\n".join(lines))
    if report["status"] != "PASS":
        raise ValueError(f"Stage 5 temporal gate failed: {critical}")
    return report
