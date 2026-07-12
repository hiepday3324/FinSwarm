from __future__ import annotations

import math
import statistics
from typing import Any

from puppy.common.schemas import MarketStep, ObservedFeatureTable, PriceBar


OBSERVED_FEATURE_NAMES = [
    "observed_close_return_1d",
    "observed_intraday_return_1d",
    "momentum_5d",
    "momentum_20d",
    "rolling_volatility_5d",
    "rolling_volatility_20d",
    "volume_ratio_20d",
    "range_pct",
    "price_history_count",
    "feature_missing_count",
]

FEATURE_DEFINITIONS = {
    "observed_close_return_1d": "close[t] / close[previous_common_date] - 1",
    "observed_intraday_return_1d": "close[t] / open[t] - 1",
    "momentum_5d": "close[t] / close[t-5 common trading observations] - 1",
    "momentum_20d": "close[t] / close[t-20 common trading observations] - 1",
    "rolling_volatility_5d": "population standard deviation of last 5 observed close returns",
    "rolling_volatility_20d": "population standard deviation of last 20 observed close returns",
    "volume_ratio_20d": "volume[t] / mean(volume over last 20 observed bars)",
    "range_pct": "(high[t] - low[t]) / close[t]",
    "price_history_count": "number of exact observed bars through t",
    "feature_missing_count": "count of unavailable numeric features excluding counters",
}


def _safe_return(first: float | int | None, second: float | int | None) -> float | None:
    if first is None or second is None or float(first) <= 0:
        return None
    return float(second) / float(first) - 1.0


def _momentum(history: list[PriceBar], periods: int) -> float | None:
    if len(history) <= periods:
        return None
    return _safe_return(history[-periods - 1].close, history[-1].close)


def _volatility(history: list[PriceBar], periods: int) -> float | None:
    if len(history) <= periods:
        return None
    returns = [
        _safe_return(history[index - 1].close, history[index].close)
        for index in range(len(history) - periods, len(history))
    ]
    values = [float(value) for value in returns if value is not None and math.isfinite(value)]
    return statistics.pstdev(values) if len(values) == periods else None


def build_observed_feature_table(
    market_step: MarketStep,
    price_history: dict[str, list[PriceBar]],
) -> ObservedFeatureTable:
    features: dict[str, dict[str, float | int | None]] = {}
    source_dates: dict[str, list[Any]] = {}
    for symbol in market_step.symbols:
        history = sorted(
            [bar for bar in price_history.get(symbol, []) if bar.date <= market_step.date],
            key=lambda bar: bar.date,
        )
        current = history[-1] if history and history[-1].date == market_step.date else None
        previous = history[-2] if len(history) >= 2 and current is not None else None
        observed: dict[str, float | int | None] = {
            "observed_close_return_1d": _safe_return(previous.close, current.close) if previous and current else None,
            "observed_intraday_return_1d": _safe_return(current.open, current.close) if current else None,
            "momentum_5d": _momentum(history, 5) if current else None,
            "momentum_20d": _momentum(history, 20) if current else None,
            "rolling_volatility_5d": _volatility(history, 5) if current else None,
            "rolling_volatility_20d": _volatility(history, 20) if current else None,
            "volume_ratio_20d": None,
            "range_pct": (
                (float(current.high) - float(current.low)) / float(current.close)
                if current and current.high is not None and current.low is not None and current.close > 0
                else None
            ),
            "price_history_count": len(history),
            "feature_missing_count": 0,
        }
        if current and len(history) >= 20 and current.volume is not None:
            volumes = [float(bar.volume) for bar in history[-20:] if bar.volume is not None]
            mean_volume = sum(volumes) / len(volumes) if len(volumes) == 20 else 0.0
            observed["volume_ratio_20d"] = float(current.volume) / mean_volume if mean_volume > 0 else None
        observed["feature_missing_count"] = sum(
            observed[name] is None
            for name in OBSERVED_FEATURE_NAMES
            if name not in {"price_history_count", "feature_missing_count"}
        )
        features[symbol] = observed
        source_dates[symbol] = [bar.date for bar in history[-21:]]
    return ObservedFeatureTable(
        date=market_step.date,
        as_of=market_step.as_of,
        symbols=list(market_step.symbols),
        features=features,
        feature_names=list(OBSERVED_FEATURE_NAMES),
        source_dates=source_dates,
        metadata={"definitions": FEATURE_DEFINITIONS, "max_source_date": market_step.date.isoformat()},
    )
