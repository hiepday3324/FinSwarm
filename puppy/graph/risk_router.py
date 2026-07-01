"""Risk router that emits memory sharing routes from graph attention."""

from __future__ import annotations

from puppy.common.enums import MemoryLayer
from puppy.common.schemas import GraphOutput, MemoryShareRoute


def build_memory_share_routes(
    graph_output: GraphOutput,
    agent_by_symbol: dict[str, str],
    threshold: float = 0.5,
    top_k: int = 3,
    layer: MemoryLayer | str = MemoryLayer.SHORT,
) -> list[MemoryShareRoute]:
    """Emit memory sharing routes from graph attention weights."""

    routes: list[MemoryShareRoute] = []
    for target_symbol, sources in graph_output.attention.items():
        target_agent_id = agent_by_symbol.get(target_symbol)
        if target_agent_id is None:
            continue
        for source_symbol, alpha in sources.items():
            if source_symbol == target_symbol or alpha < threshold:
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
                    reason=(
                        f"{source_symbol} influences {target_symbol} through "
                        f"graph attention alpha={float(alpha):.4f}."
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

    return sorted(routes, key=lambda route: route.alpha, reverse=True)
