from datetime import date
from typing import Any

from puppy.common.schemas import AgentOutput, AllocationDecision, TextFeature
from puppy.run_type import RunMode

from .allocator_agent import AllocatorAgent
from .context_builder import build_agent_context


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
    ) -> dict[str, Any]:
        outputs: list[AgentOutput] = []
        text_features: list[TextFeature] = []

        for agent in self.agents:
            output = agent.step(
                market_info=market_info_by_symbol[agent.symbol],
                run_mode=run_mode,
            )

            if self.text_encoder is not None:
                context = build_agent_context(
                    sector_agent=agent,
                    cur_date=cur_date,
                    latest_reason=output.signal.reason,
                )
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
            "text_features": text_features,
            "allocation": allocation,
        }
