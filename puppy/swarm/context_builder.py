import json
from datetime import date, datetime
from typing import Any

from puppy.common.enums import MemoryLayer
from puppy.common.schemas import AgentContext, MemoryRef


def build_agent_context(
    sector_agent: Any,
    cur_date: date,
    query_text: str | None = None,
    top_k: int | None = None,
    latest_reason: str | None = None,
    master_portfolio_state: dict[str, Any] | None = None,
    psychology_state: dict[str, Any] | None = None,
) -> AgentContext:
    top_k = top_k if top_k is not None else getattr(sector_agent.llm_agent, "top_k", 3)
    query_text = (
        query_text
        or getattr(sector_agent.llm_agent, "character_string", None)
        or sector_agent.symbol
    )
    memories, refs = _query_layered_memory(
        sector_agent=sector_agent,
        query_text=query_text,
        top_k=top_k,
        cur_date=cur_date,
    )

    return AgentContext(
        agent_id=sector_agent.agent_id,
        symbol=sector_agent.symbol,
        date=cur_date,
        latest_reason=latest_reason,
        short_memory=memories["short"],
        mid_memory=memories["mid"],
        long_memory=memories["long"],
        reflection_memory=memories["reflection"],
        memory_refs=refs,
        shared_context=(
            list(sector_agent.eligible_external_context(cur_date))
            if hasattr(sector_agent, "eligible_external_context")
            else [
                item
                for item in sector_agent.external_context_buffer
                if item.created_at is None or item.created_at < cur_date
            ]
        ),
        master_portfolio_state=master_portfolio_state,
        psychology_state=psychology_state,
    )


def build_raw_context(context: AgentContext) -> str:
    sections = [
        f"Agent: {context.agent_id}",
        f"Symbol: {context.symbol}",
        f"Date: {context.date}",
    ]
    if context.latest_reason:
        sections.append(f"Prior decision reason:\n{context.latest_reason}")
    sections.extend(
        [
            _format_text_list("Short memory", context.short_memory),
            _format_text_list("Mid memory", context.mid_memory),
            _format_text_list("Long memory", context.long_memory),
            _format_text_list("Reflection memory", context.reflection_memory),
        ]
    )
    shared_text = []
    for item in context.shared_context:
        provenance = {
            "source_agent_id": item.source_agent_id,
            "target_agent_id": item.target_agent_id,
            "source_symbol": item.source_symbol,
            "target_symbol": item.target_symbol,
            "alpha": item.alpha,
            "created_at": item.created_at,
            "reason": item.reason,
            "memory_refs": [
                ref.model_dump(mode="json") for ref in item.memory_refs
            ],
        }
        shared_text.append(
            f"Provenance: {json.dumps(provenance, default=_json_default, sort_keys=True)}\n"
            f"Content: {item.text}"
        )
    sections.append(_format_text_list("Shared context", shared_text))
    if context.master_portfolio_state is not None:
        sections.append(
            _format_structured(
                "Master portfolio state",
                context.master_portfolio_state,
            )
        )
    if context.psychology_state is not None:
        sections.append(_format_structured("Psychology state", context.psychology_state))
    return "\n\n".join(sections)


def _format_text_list(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}:\nNone."
    return f"{title}:\n" + "\n".join(f"- {item}" for item in items)


def _format_structured(title: str, value: dict[str, Any]) -> str:
    return f"{title}:\n" + json.dumps(
        value,
        default=_json_default,
        indent=2,
        sort_keys=True,
    )


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _query_layered_memory(
    sector_agent: Any,
    query_text: str,
    top_k: int,
    cur_date: date,
) -> tuple[dict[str, list[str]], list[MemoryRef]]:
    empty = {"short": [], "mid": [], "long": [], "reflection": []}
    brain = getattr(getattr(sector_agent, "llm_agent", None), "brain", None)
    if brain is None:
        return empty, []

    query_plan = [
        ("short", MemoryLayer.SHORT, getattr(brain, "query_short", None)),
        ("mid", MemoryLayer.MID, getattr(brain, "query_mid", None)),
        ("long", MemoryLayer.LONG, getattr(brain, "query_long", None)),
        ("reflection", MemoryLayer.REFLECTION, getattr(brain, "query_reflection", None)),
    ]
    refs: list[MemoryRef] = []
    for key, layer, query_func in query_plan:
        if query_func is None:
            continue
        texts, ids = query_func(
            query_text=query_text,
            top_k=top_k,
            symbol=sector_agent.symbol,
        )
        empty[key] = list(texts)
        refs.extend(
            MemoryRef(
                memory_id=memory_id,
                layer=layer,
                symbol=sector_agent.symbol,
                date=cur_date,
                source_agent_id=sector_agent.agent_id,
                text_preview=text[:160],
            )
            for text, memory_id in zip(texts, ids)
            if memory_id != -1
        )
    return empty, refs
