from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.common.schemas import CausalRunContext
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.normalization import load_stage4_config
from puppy.swarm.causal_agent_factory import build_causal_agent_bundle
from puppy.swarm.causal_orchestrator import CausalOrchestrator
from puppy.swarm.legacy_finmem_adapter import ADAPTER_VERSION
from puppy.swarm.psychology import build_psychology_state


def _common_dates(path: Path) -> list[dt.date]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        {
            dt.date.fromisoformat(value)
            for split in raw["splits"].values()
            for value in split["dates"]
        }
    )


def _backend_preflight(bundle: dict) -> dict:
    first = bundle["agents"]["TSLA"].llm_agent
    backend = first.chat_backend
    endpoint = backend.guardrail_endpoint()
    response = endpoint(
        'Return exactly this JSON object: {"status":"ok"}. Do not add text.'
    )
    return {
        "status": "PASS",
        "provider": getattr(backend, "service_provider", "unknown"),
        "requested_model": getattr(backend, "model", "unknown"),
        "returned_model": getattr(backend, "last_returned_model", None),
        "response_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "api_call_count": int(getattr(backend, "api_call_count", 0)),
    }


def _causal_diagnostics(
    bundle: dict,
    *,
    config_path: str,
    decision_date: dt.date,
    symbols: tuple[str, ...],
) -> dict:
    config = load_stage4_config(config_path)
    dates = _common_dates(config.split_manifest)
    if decision_date not in dates:
        raise ValueError("Diagnostic date is not a common trading date")
    index = dates.index(decision_date)
    expected_outcome_date = dates[index + 1] if index + 1 < len(dates) else None
    store = DuckDBMarketStore(
        str(config.duckdb_path),
        market_timezone=config.timezone.key,
        decision_cutoff=config.decision_cutoff,
    )
    store.connect()
    try:
        step = store.get_point_in_time_market_step(
            decision_date, list(config.symbols), dates
        )
    finally:
        store.close()
    run_context = CausalRunContext(
        run_id=f"diagnostic-{decision_date.isoformat()}",
        config_path=config.relative(config.config_path),
        current_date=decision_date,
        current_cutoff=step.as_of,
        bootstrap=False,
        prompt_version=bundle["prompt_version"],
        encoder_version=f"transformers-frozen:{bundle['encoder'].revision}",
        adapter_version=ADAPTER_VERSION,
        git_commit="diagnostic",
        metadata={
            "generation_parameters": {
                "temperature": 0.0,
                "reasoning_effort": "low",
            },
            "force_llm": True,
            "diagnostic": True,
        },
    )
    orchestrator = CausalOrchestrator(bundle["encoder"])
    results = []
    for symbol in symbols:
        agent = bundle["agents"][symbol]
        agent.causal_adapter.advance_memory_clock(decision_date, run_context)
        agent.causal_adapter.ingest_market_events(step, run_context)
        result = orchestrator.decide(
            agent=agent,
            market_step=step,
            run_context=run_context,
            expected_outcome_date=expected_outcome_date,
            split="diagnostic",
            psychology_state=build_psychology_state(None),
        )
        provenance = dict(result.output.metadata.get("causal_provenance", {}))
        cited_memory_ids = {
            layer: [
                item["memory_index"]
                for item in result.output.raw_reflection.get(field, [])
            ]
            for layer, field in {
                "short": "short_memory_index",
                "middle": "middle_memory_index",
                "long": "long_memory_index",
                "reflection": "reflection_memory_index",
            }.items()
        }
        results.append(
            {
                "symbol": symbol,
                "decision": str(result.output.signal.decision).upper(),
                "action_signal": result.output.signal.action_signal,
                "parse_ok": result.output.parse_ok,
                "decision_id": result.output.decision_id,
                "context_hash": result.output.context_hash,
                "prompt_version": result.output.prompt_version,
                "schema_version": result.output.metadata.get("schema_version"),
                "cited_memory_ids": cited_memory_ids,
                "retry_count": int(provenance.get("retry_count", 0)),
                "bootstrap": agent.causal_adapter.bootstrap,
                "cache_hit": bool(result.output.metadata.get("cache_hit")),
            }
        )
    return {
        "status": "PASS",
        "date": decision_date.isoformat(),
        "synthetic": False,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="User-run live diagnostics around the canonical causal factory"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("backend", "one-agent", "three-agent"), required=True
    )
    parser.add_argument("--date", help="Common trading date for causal diagnostics")
    args = parser.parse_args()
    if args.mode != "backend" and not args.date:
        parser.error("--date is required for one-agent and three-agent modes")
    bundle = build_causal_agent_bundle(args.config)
    if args.mode == "backend":
        result = _backend_preflight(bundle)
    else:
        symbols = ("TSLA",) if args.mode == "one-agent" else ("TSLA", "NVDA", "AAPL")
        result = _causal_diagnostics(
            bundle,
            config_path=args.config,
            decision_date=dt.date.fromisoformat(args.date),
            symbols=symbols,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
