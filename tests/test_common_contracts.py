import datetime as dt
import unittest

from puppy.common import (
    AgentContext,
    AgentOutput,
    AgentSignal,
    AllocationDecision,
    AllocationWeights,
    DebateRequest,
    DebateTranscript,
    DebateVerdict,
    FilingEvent,
    FusionOutput,
    GraphOutput,
    MarketStep,
    MemoryRef,
    MemoryShareEvent,
    MemoryShareRoute,
    NewsEvent,
    PortfolioSnapshot,
    PriceBar,
    QuantFeatureTable,
    SharedContext,
    ShadowPortfolioState,
    TextFeature,
)


class CommonContractsTest(unittest.TestCase):
    def test_new_m0_m2_contracts_instantiate_from_puppy_common(self) -> None:
        date = dt.date(2026, 1, 2)
        as_of = dt.datetime(2026, 1, 2, 12, 0, 0)
        market_step = MarketStep(
            date=date,
            symbols=["TSLA"],
            prices={"TSLA": PriceBar(symbol="TSLA", date=date, close=100.0)},
            news={
                "TSLA": [
                    NewsEvent(
                        event_id="n1",
                        symbol="TSLA",
                        published_at=as_of,
                        available_at=as_of,
                        text="news",
                    )
                ]
            },
            filings={
                "TSLA": [
                    FilingEvent(
                        filing_id="f1",
                        symbol="TSLA",
                        filing_type="10-Q",
                        filing_date=date,
                        available_at=as_of,
                        text="filing",
                    )
                ]
            },
            as_of=as_of,
        )
        quant_table = QuantFeatureTable(
            date=date,
            symbols=["TSLA"],
            action_signals={"TSLA": 1},
            returns={"TSLA": 0.01},
        )
        graph_output = GraphOutput(date=date, symbols=["TSLA"])
        fusion_output = FusionOutput(date=date, symbols=["TSLA"])
        allocation = AllocationWeights(date=date, weights={"TSLA": 0.4}, cash_weight=0.6)
        snapshot = PortfolioSnapshot(
            date=date,
            cash=60000.0,
            positions={"TSLA": 400.0},
            weights={"TSLA": 0.4},
            equity=100000.0,
        )
        shadow = ShadowPortfolioState(
            date=date,
            agent_id="fake_tsla",
            symbol="TSLA",
            value=1.01,
            roi=0.01,
        )

        self.assertEqual(market_step.prices["TSLA"].close, 100.0)
        self.assertEqual(quant_table.action_signals["TSLA"], 1)
        self.assertEqual(graph_output.symbols, ["TSLA"])
        self.assertEqual(fusion_output.symbols, ["TSLA"])
        self.assertAlmostEqual(allocation.cash_weight, 0.6)
        self.assertEqual(snapshot.equity, 100000.0)
        self.assertEqual(shadow.symbol, "TSLA")

    def test_existing_tien_contracts_still_import_and_instantiate(self) -> None:
        date = dt.date(2026, 1, 2)
        ref = MemoryRef(memory_id="m1", layer="short", symbol="TSLA")
        shared = SharedContext(
            source_agent_id="a1",
            target_agent_id="a2",
            text="shared",
            reason="test",
            memory_refs=[ref],
        )
        context = AgentContext(agent_id="a1", symbol="TSLA", date=date, shared_context=[shared])
        text_feature = TextFeature(
            agent_id="a1",
            symbol="TSLA",
            date=date,
            h_text=[0.1, 0.2],
            raw_context="ctx",
            embedding_model="fake",
            dim=2,
        )
        signal = AgentSignal(
            agent_id="a1",
            symbol="TSLA",
            date=date,
            decision="buy",
            action_signal=1,
            reason="reason",
        )
        output = AgentOutput(signal=signal)
        allocation_decision = AllocationDecision(
            date=date,
            action="increase_exposure",
            score=1,
            reason="reason",
            signals=[signal],
        )
        route = MemoryShareRoute(
            date=date,
            source_agent_id="a1",
            target_agent_id="a2",
            source_symbol="TSLA",
            target_symbol="NVDA",
            alpha=0.5,
            reason="route",
        )
        event = MemoryShareEvent(date=date, route=route, shared_context=shared)
        request = DebateRequest(
            date=date,
            agent_id="a1",
            symbol="TSLA",
            local_decision="buy",
            text_signal=1,
            graph_signal=-1,
            fusion_score=0.4,
            reason="conflict",
            agent_signal=signal,
        )
        transcript = DebateTranscript(
            date=date,
            agent_id="a1",
            symbol="TSLA",
            request=request,
            question="q",
            answer="a",
        )
        verdict = DebateVerdict(
            date=date,
            agent_id="a1",
            symbol="TSLA",
            verdict="reduce",
            reason="rule",
            transcript=transcript,
        )

        self.assertEqual(context.shared_context[0].text, "shared")
        self.assertEqual(text_feature.dim, 2)
        self.assertEqual(output.signal.action_signal, 1)
        self.assertEqual(allocation_decision.score, 1)
        self.assertEqual(event.route.alpha, 0.5)
        self.assertEqual(verdict.verdict, "reduce")


if __name__ == "__main__":
    unittest.main()
