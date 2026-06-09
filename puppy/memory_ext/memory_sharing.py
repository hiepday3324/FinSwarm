from datetime import date
from typing import Any

from puppy.common.enums import MemoryLayer
from puppy.common.schemas import (
    MemoryRef,
    MemoryShareEvent,
    MemoryShareRoute,
    SharedContext,
)


class MemorySharingService:
    def __init__(self) -> None:
        self._applied_keys: set[tuple[str, str, tuple[int | str, ...]]] = set()

    def apply_memory_share_route(
        self,
        route: MemoryShareRoute,
        source_memory_store: Any,
        target_agent: Any,
    ) -> MemoryShareEvent:
        memories = self._search_memory(route=route, source_memory_store=source_memory_store)
        memory_refs = [
            self._memory_to_ref(memory=memory, route=route) for memory in memories
        ]
        memory_ids = [ref.memory_id for ref in memory_refs]
        shared_context = SharedContext(
            source_agent_id=route.source_agent_id,
            target_agent_id=route.target_agent_id,
            source_symbol=route.source_symbol,
            target_symbol=route.target_symbol,
            text=self._summarize(route=route, memories=memories),
            reason=route.reason,
            alpha=route.alpha,
            created_at=route.date,
            ttl_days=route.ttl_days,
            memory_refs=memory_refs,
        )

        dedupe_key = (
            route.source_agent_id,
            route.target_agent_id,
            tuple(memory_ids),
        )
        deduped = dedupe_key in self._applied_keys
        if not deduped:
            target_agent.inject_external_context(shared_context)
            self._applied_keys.add(dedupe_key)

        return MemoryShareEvent(
            date=route.date,
            route=route,
            shared_context=shared_context,
            shared_memory_ids=memory_ids,
            deduped=deduped,
        )

    def _search_memory(
        self, route: MemoryShareRoute, source_memory_store: Any
    ) -> list[Any]:
        if hasattr(source_memory_store, "search_memory"):
            return list(
                source_memory_store.search_memory(
                    agent_id=route.source_agent_id,
                    symbol=route.source_symbol,
                    layer=route.layer,
                    top_k=route.top_k,
                )
            )

        layer = MemoryLayer(route.layer).value
        query_func_name = {
            "short": "query_short",
            "mid": "query_mid",
            "long": "query_long",
            "reflection": "query_reflection",
        }.get(layer)
        if query_func_name and hasattr(source_memory_store, query_func_name):
            texts, ids = getattr(source_memory_store, query_func_name)(
                query_text=route.query_text or route.reason or route.source_symbol,
                top_k=route.top_k,
                symbol=route.source_symbol,
            )
            return [
                {
                    "memory_id": memory_id,
                    "text": text,
                    "layer": layer,
                    "symbol": route.source_symbol,
                }
                for text, memory_id in zip(texts, ids)
            ]

        if isinstance(source_memory_store, dict):
            memories = source_memory_store.get(layer, source_memory_store.get("memories", []))
            return list(memories)[: route.top_k]

        return []

    def _memory_to_ref(self, memory: Any, route: MemoryShareRoute) -> MemoryRef:
        text = self._memory_value(memory, "text", "")
        return MemoryRef(
            memory_id=self._memory_value(memory, "memory_id", self._memory_value(memory, "id", "")),
            layer=self._memory_value(memory, "layer", route.layer),
            symbol=self._memory_value(memory, "symbol", route.source_symbol),
            date=self._memory_value(memory, "date", route.date),
            score=self._memory_value(memory, "score", None),
            source_agent_id=route.source_agent_id,
            text_preview=text[:160] if isinstance(text, str) else None,
        )

    def _summarize(self, route: MemoryShareRoute, memories: list[Any]) -> str:
        memory_text = "\n".join(
            f"- {self._memory_value(memory, 'text', str(memory))}" for memory in memories
        )
        if not memory_text:
            memory_text = "- No source memory returned."
        return (
            f"Shared from {route.source_symbol} to {route.target_symbol}. "
            f"Graph influence alpha={route.alpha:.3f}. Reason: {route.reason}\n"
            f"{memory_text}"
        )

    @staticmethod
    def _memory_value(memory: Any, key: str, default: Any = None) -> Any:
        if isinstance(memory, dict):
            return memory.get(key, default)
        return getattr(memory, key, default)


def apply_memory_share_route(
    route: MemoryShareRoute,
    source_memory_store: Any,
    target_agent: Any,
) -> MemoryShareEvent:
    return MemorySharingService().apply_memory_share_route(
        route=route,
        source_memory_store=source_memory_store,
        target_agent=target_agent,
    )
