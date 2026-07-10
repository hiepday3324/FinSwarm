import datetime as dt
import os
import tempfile
import unittest

from puppy.common.schemas import AgentSignal, FusionOutput, MemoryRef, SharedContext
from puppy.data_engine.event_store import EventStore
from puppy.data_engine.result_store import ResultStore
from puppy.debate.coordinator import (
    DebateCoordinator,
    build_debate_requests_from_fusion,
    graph_score_to_signal,
)
from puppy.run_type import RunMode
from puppy.swarm.run_swarm import build_mock_agents
from scripts.run_milestone_7_debate import run_milestone_7_debate


class Milestone7DebateTest(unittest.TestCase):
    def test_graph_score_to_signal_uses_symmetric_threshold(self) -> None:
        self.assertEqual(graph_score_to_signal(0.20, threshold=0.10), 1)
        self.assertEqual(graph_score_to_signal(-0.20, threshold=0.10), -1)
        self.assertEqual(graph_score_to_signal(0.05, threshold=0.10), 0)
        self.assertEqual(graph_score_to_signal(None), 0)

    def test_debate_requests_from_fusion_use_grey_zone_and_signal_conflict(
        self,
    ) -> None:
        date = dt.date(2026, 1, 9)
        signals = [
            AgentSignal(
                agent_id="sector_tsla",
                symbol="TSLA",
                date=date,
                decision="buy",
                action_signal=1,
                reason="local buy",
            ),
            AgentSignal(
                agent_id="sector_nvda",
                symbol="NVDA",
                date=date,
                decision="hold",
                action_signal=0,
                reason="local hold",
            ),
            AgentSignal(
                agent_id="sector_aapl",
                symbol="AAPL",
                date=date,
                decision="sell",
                action_signal=-1,
                reason="local sell",
            ),
        ]
        fusion_output = FusionOutput(
            date=date,
            symbols=["TSLA", "NVDA", "AAPL"],
            graph_scores={"TSLA": -0.60, "NVDA": 0.20, "AAPL": -0.40},
            fusion_scores={"TSLA": 0.90, "NVDA": 0.58, "AAPL": 0.18},
            conflict_flags={"TSLA": False, "NVDA": False, "AAPL": False},
        )

        requests = build_debate_requests_from_fusion(signals, fusion_output)

        self.assertEqual([request.symbol for request in requests], ["TSLA", "NVDA"])
        self.assertEqual(requests[0].graph_signal, -1)
        self.assertEqual(requests[1].mode, "debate")

    def test_debate_coordinator_writes_verdict_artifacts_and_shared_citations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")
            event_store = EventStore(event_path)
            date = dt.date(2026, 1, 9)
            agents = build_mock_agents()
            agents[0].inject_external_context(
                SharedContext(
                    source_agent_id="sector_nvda",
                    target_agent_id="sector_tsla",
                    source_symbol="NVDA",
                    target_symbol="TSLA",
                    text="NVDA shared supply-chain pressure.",
                    reason="graph route",
                    created_at=date,
                    memory_refs=[
                        MemoryRef(
                            memory_id="NVDA-shared-1",
                            layer="short",
                            symbol="NVDA",
                            source_agent_id="sector_nvda",
                        )
                    ],
                )
            )
            for agent in agents:
                agent.step((date, 100.0, None, None, [], 0.0, False), RunMode.Test)

            fusion_output = FusionOutput(
                date=date,
                symbols=["TSLA"],
                graph_scores={"TSLA": -0.65},
                fusion_scores={"TSLA": 0.42},
                conflict_flags={"TSLA": True},
            )
            verdicts = DebateCoordinator().run_for_fusion(
                fusion_output=fusion_output,
                agent_signals=[agent.get_signal(date) for agent in agents],
                sector_agents=agents,
                event_store=event_store,
            )
            result_store = ResultStore(event_store)

            self.assertEqual(len(verdicts), 1)
            self.assertEqual(len(result_store.read_debate_requests()), 1)
            self.assertEqual(len(result_store.read_debate_transcripts()), 1)
            self.assertEqual(len(result_store.read_debate_verdicts()), 1)
            transcript = result_store.read_debate_transcripts()[0]
            cited_ids = {ref.memory_id for ref in transcript.cited_memory_refs}
            self.assertIn("NVDA-shared-1", cited_ids)
            self.assertIn("shared:NVDA-shared-1", transcript.answer)

    def test_milestone_7_debate_runner_writes_tien_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = os.path.join(temp_dir, "events.jsonl")
            summary = run_milestone_7_debate(event_store_path=event_path)
            event_store = EventStore(event_path)
            result_store = ResultStore(event_store)

            self.assertEqual(summary["num_agent_outputs"], 3)
            self.assertEqual(summary["num_text_features"], 3)
            self.assertEqual(summary["num_fusion_outputs"], 1)
            self.assertEqual(summary["num_debate_requests"], 2)
            self.assertEqual(summary["num_debate_transcripts"], 2)
            self.assertEqual(summary["num_debate_verdicts"], 2)
            self.assertEqual(summary["debated_symbols"], ["TSLA", "NVDA"])
            self.assertEqual(len(result_store.read_debate_verdicts()), 2)
            self.assertEqual(event_store.read_events("AllocationWeights"), [])
            self.assertEqual(event_store.read_events("PortfolioSnapshot"), [])


if __name__ == "__main__":
    unittest.main()
