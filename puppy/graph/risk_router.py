"""Risk router that emits memory sharing routes from graph attention."""

from __future__ import annotations

from collections.abc import Mapping

from puppy.common.enums import MemoryLayer
from puppy.common.schemas import GraphOutput, MemoryShareRoute


def build_memory_share_routes(
    graph_output: GraphOutput,
    agent_by_symbol: dict[str, str],
    threshold: float = 0.5,
    top_k: int = 3,
    layer: MemoryLayer | str = MemoryLayer.SHORT,
    importance_by_source: Mapping[str, float] | None = None,
) -> list[MemoryShareRoute]:
    """Emit routes only when ``Alpha[target, source] * Importance > theta``."""

    routes: list[MemoryShareRoute] = []
    for target_symbol, sources in graph_output.attention.items():
        target_agent_id = agent_by_symbol.get(target_symbol)
        if target_agent_id is None:
            continue
        for source_symbol, alpha in sources.items():
            importance = float(
                (importance_by_source or {}).get(source_symbol, 1.0)
            )
            routing_score = float(alpha) * importance
            if source_symbol == target_symbol or routing_score <= threshold:
                continue
            source_agent_id = agent_by_symbol.get(source_symbol)
            if source_agent_id is None:
                continue
            routes.append(
                MemoryShareRoute(
                    date=graph_output.date,
                    source_agent_id=source_agent_id,
                    target_agent_id=target_agent_id,
                    source_symbol=source_symbol,
                    target_symbol=target_symbol,
                    alpha=float(alpha),
                    memory_importance=importance,
                    routing_score=routing_score,
                    routing_threshold=float(threshold),
                    reason=(
                        f"{source_symbol} influences {target_symbol} through "
                        f"alpha={float(alpha):.4f} * importance={importance:.4f} "
                        f"= {routing_score:.4f} > theta={float(threshold):.4f}."
                    ),
                    layer=layer,
                    top_k=top_k,
                    query_text=(
                        f"{source_symbol} risk and market memory relevant to "
                        f"{target_symbol}"
                    ),
                    ttl_days=1,
                )
            )

    return sorted(
        routes,
        key=lambda route: float(route.routing_score or 0.0),
        reverse=True,
    )
