from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puppy.training.artifact_generator import generate_training_artifacts


def _load_agent_factory(spec: str):
    if ":" not in spec:
        raise ValueError("--agent-factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("Configured causal agent factory is not callable")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal FinSwarm training artifacts without training models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--agent-factory",
        help=(
            "Explicit module:function returning agents, encoder and optional "
            "text_score_head/prompt_version for real causal generation"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Bypass the causal decision cache without permitting bootstrap fallback",
    )
    args = parser.parse_args()
    if args.bootstrap and args.agent_factory:
        parser.error("--bootstrap and --agent-factory are mutually exclusive")
    if (
        args.agent_factory
        and args.start is None
        and args.end is None
        and os.environ.get("FINMEM_FULL_ROLLOUT_CONFIRMATION") != "YES"
    ):
        parser.error(
            "Full real causal rollout requires FINMEM_FULL_ROLLOUT_CONFIRMATION=YES"
        )
    injected = {}
    if args.agent_factory:
        injected = dict(_load_agent_factory(args.agent_factory)(args.config))
    result = generate_training_artifacts(
        args.config,
        start=args.start,
        end=args.end,
        bootstrap=args.bootstrap,
        resume=args.resume,
        overwrite=args.overwrite,
        force_llm=args.force_llm,
        causal_agents=injected.get("agents"),
        encoder=injected.get("encoder"),
        text_score_head=injected.get("text_score_head"),
        prompt_version=injected.get("prompt_version"),
        generation_provenance=injected.get("provenance"),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
