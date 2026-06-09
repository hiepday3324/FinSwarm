from datetime import date
from typing import Any

import numpy as np

from puppy.common.schemas import MemoryShareRoute
from puppy.debate.conflict_detector import build_debate_request
from puppy.debate.debate_loop import DebateLoop
from puppy.debate.judge import DebateJudge
from puppy.fusion.text_encoder import TextStreamEncoder
from puppy.memory_ext.memory_sharing import MemorySharingService
from puppy.run_type import RunMode

from .orchestrator import SwarmOrchestrator
from .sector_agent import SectorAgent


class MockLLMAgent:
    def __init__(self, symbol: str, decision: str, reason: str) -> None:
        self.symbol = symbol
        self.decision = decision
        self.reason = reason
        self.model_name = "mock-llm"
        self.top_k = 1
        self.character_string = f"Mock sector profile for {symbol}"
        self.reflection_result_series_dict: dict[date, dict[str, Any]] = {}
        self.brain = None

    def step(self, market_info: tuple, run_mode: RunMode) -> None:
        cur_date = market_info[0]
        self.reflection_result_series_dict[cur_date] = {
            "investment_decision": self.decision,
            "summary_reason": self.reason,
            "short_memory_index": [{"memory_index": f"{self.symbol}-short-1"}],
            "middle_memory_index": [{"memory_index": f"{self.symbol}-mid-1"}],
            "long_memory_index": [{"memory_index": f"{self.symbol}-long-1"}],
            "reflection_memory_index": [
                {"memory_index": f"{self.symbol}-reflection-1"}
            ],
        }


def deterministic_embedding(text: str) -> np.ndarray:
    length_feature = len(text) / 1000.0
    checksum_feature = sum(ord(char) for char in text) % 997 / 997.0
    line_feature = text.count("\n") / 100.0
    return np.array([[length_feature, checksum_feature, line_feature]], dtype=np.float32)


def build_mock_agents() -> list[SectorAgent]:
    return [
        SectorAgent(
            agent_id="sector_tsla",
            symbol="TSLA",
            llm_agent=MockLLMAgent("TSLA", "buy", "EV demand remains strong."),
        ),
        SectorAgent(
            agent_id="sector_nvda",
            symbol="NVDA",
            llm_agent=MockLLMAgent("NVDA", "hold", "AI demand offsets valuation risk."),
        ),
        SectorAgent(
            agent_id="sector_aapl",
            symbol="AAPL",
            llm_agent=MockLLMAgent("AAPL", "sell", "Hardware demand looks weak."),
        ),
    ]


def run_mock_swarm(cur_date: date | None = None) -> dict[str, Any]:
    cur_date = cur_date or date(2026, 1, 2)
    agents = build_mock_agents()
    market_info_by_symbol = {
        agent.symbol: (cur_date, 100.0, None, None, [], 0.0, False)
        for agent in agents
    }
    text_encoder = TextStreamEncoder(
        embedding_func=deterministic_embedding,
        embedding_model="deterministic-mock",
    )
    orchestrator = SwarmOrchestrator(agents=agents, text_encoder=text_encoder)
    result = orchestrator.run_day(
        cur_date=cur_date,
        market_info_by_symbol=market_info_by_symbol,
        run_mode=RunMode.Test,
    )

    memory_route = MemoryShareRoute(
        date=cur_date,
        source_agent_id="sector_tsla",
        target_agent_id="sector_nvda",
        source_symbol="TSLA",
        target_symbol="NVDA",
        alpha=0.82,
        reason="TSLA supply-chain pressure can affect semiconductor demand.",
        layer="short",
        top_k=1,
    )
    source_memory_store = {
        "short": [
            {
                "memory_id": "TSLA-shared-1",
                "text": "TSLA production guidance mentions chip supply constraints.",
                "layer": "short",
                "symbol": "TSLA",
            }
        ]
    }
    memory_event = MemorySharingService().apply_memory_share_route(
        route=memory_route,
        source_memory_store=source_memory_store,
        target_agent=agents[1],
    )

    tsla_signal = result["agent_outputs"][0].signal
    debate_request = build_debate_request(
        agent_signal=tsla_signal,
        fusion_score=0.42,
        graph_signal=-1,
        cur_date=cur_date,
        reason="Graph risk is negative while local TSLA signal is buy.",
    )
    transcript = DebateLoop().run_debate(
        request=debate_request,
        sector_agent=agents[0],
    )
    verdict = DebateJudge().make_verdict(transcript)

    result["memory_share_event"] = memory_event
    result["debate_request"] = debate_request
    result["debate_transcript"] = transcript
    result["debate_verdict"] = verdict
    return result
