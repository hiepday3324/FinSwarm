from collections.abc import Iterable, Mapping
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
        memories = self._eligible_memories(route, memories)
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

    def _eligible_memories(
        self,
        route: MemoryShareRoute,
        memories: list[Any],
    ) -> list[Any]:
        eligible = []
        for memory in memories:
            memory_id = self._memory_value(
                memory, "memory_id", self._memory_value(memory, "id", None)
            )
            if route.memory_id is not None and memory_id != route.memory_id:
                continue
            importance = self._memory_value(
                memory, "importance", route.memory_importance
            )
            try:
                routing_score = float(route.alpha) * float(importance)
            except (TypeError, ValueError):
                continue
            if routing_score > float(route.routing_threshold):
                eligible.append(memory)
        return eligible

    def apply_memory_share_routes(
        self,
        routes: Iterable[MemoryShareRoute],
        source_memory_stores: Any,
        target_agents: Mapping[str, Any] | Iterable[Any],
        event_store: Any | None = None,
    ) -> list[MemoryShareEvent]:
        events: list[MemoryShareEvent] = []
        for route in routes:
            target_agent = _resolve_target_agent(target_agents, route)
            if target_agent is None:
                raise KeyError(
                    f"No target agent for route target_agent_id={route.target_agent_id} "
                    f"target_symbol={route.target_symbol}."
                )
            source_memory_store = _resolve_source_memory_store(
                source_memory_stores=source_memory_stores,
                route=route,
            )
            event = self.apply_memory_share_route(
                route=route,
                source_memory_store=source_memory_store,
                target_agent=target_agent,
            )
            events.append(event)
            if event_store is not None:
                if hasattr(event_store, "append_memory_share_event"):
                    event_store.append_memory_share_event(route.date, event)
                else:
                    event_store.append_event("MemoryShareEvent", route.date, event)
        return events

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
            metadata={
                "importance": self._memory_value(
                    memory, "importance", route.memory_importance
                ),
                "routing_score": float(route.alpha)
                * float(
                    self._memory_value(
                        memory, "importance", route.memory_importance
                    )
                ),
                "routing_threshold": route.routing_threshold,
            },
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


def apply_memory_share_routes(
    routes: Iterable[MemoryShareRoute],
    source_memory_stores: Any,
    target_agents: Mapping[str, Any] | Iterable[Any],
    event_store: Any | None = None,
) -> list[MemoryShareEvent]:
    return MemorySharingService().apply_memory_share_routes(
        routes=routes,
        source_memory_stores=source_memory_stores,
        target_agents=target_agents,
        event_store=event_store,
    )


def _resolve_target_agent(
    target_agents: Mapping[str, Any] | Iterable[Any],
    route: MemoryShareRoute,
) -> Any | None:
    if isinstance(target_agents, Mapping):
        for key in (route.target_agent_id, route.target_symbol):
            if key in target_agents:
                return target_agents[key]
        return None

    for agent in target_agents:
        if getattr(agent, "agent_id", None) == route.target_agent_id:
            return agent
        if getattr(agent, "symbol", None) == route.target_symbol:
            return agent
    return None


def _resolve_source_memory_store(
    source_memory_stores: Any,
    route: MemoryShareRoute,
) -> Any:
    if not isinstance(source_memory_stores, Mapping):
        return source_memory_stores

    for key in (route.source_agent_id, route.source_symbol, "default"):
        if key in source_memory_stores:
            return source_memory_stores[key]

    layer_value = MemoryLayer(route.layer).value
    if layer_value in source_memory_stores or "memories" in source_memory_stores:
        return source_memory_stores
    return source_memory_stores
