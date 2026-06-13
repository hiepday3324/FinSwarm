from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.common.schemas import AgentOutput, AgentSignal
from puppy.data_engine.duckdb_store import DuckDBMarketStore
from puppy.data_engine.event_store import EventStore
from puppy.data_engine.multi_asset_env import MultiAssetEnvironment
from puppy.portfolio_ext.allocation_engine import signals_to_target_weights
from puppy.portfolio_ext.master_portfolio import MasterPortfolio
from puppy.quant.return_utils import build_quant_feature_table


def _fake_agent_outputs(date: dt.date, symbols: list[str]) -> list[AgentOutput]:
    outputs = []
    for symbol in symbols:
        decision = "buy" if symbol == "TSLA" else "hold"
        action_signal = 1 if decision == "buy" else 0
        outputs.append(
            AgentOutput(
                signal=AgentSignal(
                    agent_id=f"fake_{symbol.lower()}",
                    symbol=symbol,
                    date=date,
                    decision=decision,
                    action_signal=action_signal,
                    reason=f"Deterministic M1 fake signal for {symbol}.",
                ),
                raw_reflection={"source": "m1_fake_agent"},
                model_name="deterministic-fake",
                parse_ok=True,
            )
        )
    return outputs


def _seed_market_store() -> tuple[DuckDBMarketStore, list[dt.date], list[str]]:
    symbols = ["TSLA", "NVDA"]
    dates = [dt.date(2026, 1, 2), dt.date(2026, 1, 3)]
    store = DuckDBMarketStore(db_path=":memory:")
    store.connect()
    store.ingest_prices(
        [
            {"symbol": "TSLA", "date": dates[0], "open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1000, "source": "m1"},
            {"symbol": "NVDA", "date": dates[0], "open": 200.0, "high": 202.0, "low": 198.0, "close": 200.0, "adj_close": 200.0, "volume": 2000, "source": "m1"},
            {"symbol": "TSLA", "date": dates[1], "open": 100.0, "high": 106.0, "low": 100.0, "close": 105.0, "adj_close": 105.0, "volume": 1100, "source": "m1"},
            {"symbol": "NVDA", "date": dates[1], "open": 200.0, "high": 203.0, "low": 199.0, "close": 202.0, "adj_close": 202.0, "volume": 2100, "source": "m1"},
        ]
    )
    return store, dates, symbols


def run_milestone_1_mock(event_store_path: str | None = None) -> dict[str, Any]:
    store, dates, symbols = _seed_market_store()
    env = MultiAssetEnvironment(market_store=store, symbols=symbols, dates=dates)
    event_store = EventStore(event_store_path or "data/event_store/milestone_1_mock.jsonl")
    portfolio = MasterPortfolio(initial_cash=100000.0)

    snapshots = []
    for index, cur_date in enumerate(dates):
        market_step = env.step(date=cur_date)
        next_step = (
            env.get_market_step(dates[index + 1])
            if index + 1 < len(dates)
            else None
        )
        agent_outputs = _fake_agent_outputs(cur_date, symbols)
        quant_table = build_quant_feature_table(
            date=cur_date,
            symbols=symbols,
            agent_outputs=agent_outputs,
            current_market_step=market_step,
            next_market_step=next_step,
        )
        allocation = signals_to_target_weights(
            date=cur_date,
            symbols=symbols,
            action_signals=quant_table.action_signals,
            max_weight=0.4,
        )
        snapshot = portfolio.apply_target_weights(
            date=cur_date,
            prices=market_step.prices,
            target_weights=allocation,
        )
        snapshots.append(snapshot)

        event_store.append_event("MarketStep", cur_date, market_step)
        for output in agent_outputs:
            event_store.append_event("FakeAgentOutput", cur_date, output)
        event_store.append_event("QuantFeatureTable", cur_date, quant_table)
        event_store.append_event("AllocationWeights", cur_date, allocation)
        event_store.append_event("PortfolioSnapshot", cur_date, snapshot)

    events_written = len(event_store.read_events())
    final_snapshot = snapshots[-1]
    return {
        "dates_processed": len(dates),
        "symbols": symbols,
        "final_equity": final_snapshot.equity,
        "final_weights": final_snapshot.weights,
        "events_written": events_written,
        "last_market_step": market_step,
        "last_quant_table": quant_table,
        "last_portfolio_snapshot": final_snapshot,
    }


def main() -> None:
    summary = run_milestone_1_mock()
    printable = {
        key: value
        for key, value in summary.items()
        if key not in {"last_market_step", "last_quant_table", "last_portfolio_snapshot"}
    }
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
