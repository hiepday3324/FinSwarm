from collections.abc import Mapping
from datetime import date
from typing import Any

from puppy.common.schemas import AgentContext, AgentOutput, TextFeature
from puppy.run_type import RunMode

from .allocator_agent import AllocatorAgent
from .context_builder import build_agent_context
from .psychology import build_psychology_state


class SwarmOrchestrator:
    def __init__(
        self,
        agents: list[Any],
        text_encoder: Any | None = None,
        allocator: AllocatorAgent | None = None,
    ) -> None:
        self.agents = agents
        self.text_encoder = text_encoder
        self.allocator = allocator or AllocatorAgent()

    def run_day(
        self,
        cur_date: date,
        market_info_by_symbol: dict[str, tuple],
        run_mode: RunMode,
        shadow_state_by_agent: Mapping[str, Any] | None = None,
        psychology_state_by_agent: Mapping[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        outputs: list[AgentOutput] = []
        contexts: list[AgentContext] = []
        text_features: list[TextFeature] = []

        for agent in self.agents:
            output = agent.step(
                market_info=market_info_by_symbol[agent.symbol],
                run_mode=run_mode,
            )

            if self.text_encoder is not None:
                shadow_state = _lookup_agent_state(shadow_state_by_agent, agent)
                psychology_state = _lookup_agent_state(psychology_state_by_agent, agent)
                if psychology_state is None and shadow_state is not None:
                    psychology_state = build_psychology_state(shadow_state)
                context = build_agent_context(
                    sector_agent=agent,
                    cur_date=cur_date,
                    latest_reason=output.signal.reason,
                    shadow_state=_to_context_dict(shadow_state),
                    psychology_state=psychology_state,
                )
                contexts.append(context)
                text_feature = self.text_encoder.encode(context)
                output.h_text = text_feature.h_text
                output.raw_context = text_feature.raw_context
                text_features.append(text_feature)

            outputs.append(output)

        allocation = self.allocator.aggregate(
            signals=[output.signal for output in outputs],
            cur_date=cur_date,
        )
        return {
            "date": cur_date,
            "agent_outputs": outputs,
            "contexts": contexts,
            "text_features": text_features,
            "allocation": allocation,
        }


def _lookup_agent_state(
    state_by_agent: Mapping[str, Any] | None,
    agent: Any,
) -> Any | None:
    if state_by_agent is None:
        return None
    agent_id = getattr(agent, "agent_id", None)
    symbol = getattr(agent, "symbol", None)
    for key in (agent_id, symbol):
        if key is not None and key in state_by_agent:
            return state_by_agent[key]
    return None


def _to_context_dict(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in (
            "date",
            "agent_id",
            "symbol",
            "value",
            "roi",
            "rolling_return",
            "sharpe",
            "drawdown",
            "win_rate",
            "metadata",
        )
        if hasattr(value, key)
    }
