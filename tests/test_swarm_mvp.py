import unittest
from datetime import date

import numpy as np

from puppy.common.enums import Decision
from puppy.common.schemas import AgentContext, MemoryShareRoute
from puppy.debate.conflict_detector import build_debate_request, detect_conflict
from puppy.debate.debate_loop import DebateLoop
from puppy.debate.judge import DebateJudge
from puppy.fusion.text_encoder import TextStreamEncoder
from puppy.memory_ext.memory_sharing import MemorySharingService
from puppy.run_type import RunMode
from puppy.swarm.protocols import decision_to_action
from puppy.swarm.run_swarm import run_mock_swarm
from puppy.swarm.sector_agent import SectorAgent


class FakeLLMAgent:
    def __init__(self, raw_reflection):
        self.raw_reflection = raw_reflection
        self.reflection_result_series_dict = {}
        self.model_name = "fake"
        self.top_k = 1
        self.character_string = "fake profile"
        self.brain = None

    def step(self, market_info, run_mode):
        self.reflection_result_series_dict[market_info[0]] = self.raw_reflection


class SwarmMvpTest(unittest.TestCase):
    def test_decision_to_action(self):
        self.assertEqual(decision_to_action("buy"), 1)
        self.assertEqual(decision_to_action("hold"), 0)
        self.assertEqual(decision_to_action("sell"), -1)
        self.assertEqual(decision_to_action("unknown"), 0)

    def test_sector_agent_extracts_signal_and_memory_refs(self):
        cur_date = date(2026, 1, 2)
        raw = {
            "investment_decision": "buy",
            "summary_reason": "Strong demand.",
            "short_memory_index": [{"memory_index": 1}],
            "middle_memory_index": [{"memory_index": 2}],
            "long_memory_index": [{"memory_index": 3}],
            "reflection_memory_index": [{"memory_index": 4}],
        }
        agent = SectorAgent("sector_tsla", "TSLA", FakeLLMAgent(raw))
        output = agent.step((cur_date, 100.0, None, None, [], 0.0, False), RunMode.Test)

        self.assertTrue(output.parse_ok)
        self.assertEqual(output.signal.decision, Decision.BUY.value)
        self.assertEqual(output.signal.action_signal, 1)
        self.assertEqual(len(output.signal.memory_refs), 4)
        self.assertEqual(agent.get_signal(cur_date).reason, "Strong demand.")

    def test_sector_agent_fallbacks_on_empty_reflection(self):
        cur_date = date(2026, 1, 2)
        agent = SectorAgent("sector_tsla", "TSLA", FakeLLMAgent({}))
        output = agent.step((cur_date, 100.0, None, None, [], 0.0, False), RunMode.Test)

        self.assertFalse(output.parse_ok)
        self.assertEqual(output.signal.decision, Decision.HOLD.value)
        self.assertEqual(output.signal.action_signal, 0)

    def test_text_stream_encoder_returns_float32_feature(self):
        cur_date = date(2026, 1, 2)
        context = AgentContext(agent_id="sector_tsla", symbol="TSLA", date=cur_date)
        encoder = TextStreamEncoder(
            embedding_func=lambda text: np.array([[1.0, 2.0]], dtype=np.float64),
            embedding_model="fake-embedding",
        )
        vector, _ = encoder.encode_array(context)
        feature = encoder.encode(context)

        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(feature.symbol, "TSLA")
        self.assertEqual(feature.date, cur_date)
        self.assertEqual(feature.dtype, "float32")
        self.assertIn("Symbol: TSLA", feature.raw_context)

    def test_memory_sharing_injects_context_and_dedupes(self):
        cur_date = date(2026, 1, 2)
        target = SectorAgent("sector_nvda", "NVDA", FakeLLMAgent({}))
        route = MemoryShareRoute(
            date=cur_date,
            source_agent_id="sector_tsla",
            target_agent_id="sector_nvda",
            source_symbol="TSLA",
            target_symbol="NVDA",
            alpha=0.82,
            reason="Supply-chain risk.",
        )
        store = {
            "short": [
                {
                    "memory_id": "m1",
                    "text": "TSLA mentions chip supply pressure.",
                    "symbol": "TSLA",
                    "layer": "short",
                }
            ]
        }
        service = MemorySharingService()
        event = service.apply_memory_share_route(route, store, target)
        duplicate = service.apply_memory_share_route(route, store, target)

        self.assertFalse(event.deduped)
        self.assertTrue(duplicate.deduped)
        self.assertEqual(len(target.external_context_buffer), 1)
        self.assertEqual(event.shared_memory_ids, ["m1"])

    def test_conflict_detector_and_debate(self):
        self.assertTrue(detect_conflict(text_signal=1, graph_signal=-1, fusion_score=0.9))
        self.assertTrue(detect_conflict(text_signal=1, graph_signal=1, fusion_score=0.5))

        cur_date = date(2026, 1, 2)
        raw = {"investment_decision": "buy", "summary_reason": "Strong demand."}
        agent = SectorAgent("sector_tsla", "TSLA", FakeLLMAgent(raw))
        output = agent.step((cur_date, 100.0, None, None, [], 0.0, False), RunMode.Test)
        request = build_debate_request(
            agent_signal=output.signal,
            fusion_score=0.42,
            graph_signal=-1,
        )
        transcript = DebateLoop().run_debate(request, agent)
        verdict = DebateJudge().make_verdict(transcript)

        self.assertEqual(verdict.verdict, "reduce")
        self.assertIn("Graph risk conflicts", transcript.question)

    def test_mock_integration_pipeline(self):
        result = run_mock_swarm(date(2026, 1, 2))

        self.assertEqual(len(result["agent_outputs"]), 3)
        self.assertEqual(len(result["text_features"]), 3)
        self.assertIsNotNone(result["allocation"])
        self.assertEqual(result["debate_verdict"].verdict, "reduce")


if __name__ == "__main__":
    unittest.main()
