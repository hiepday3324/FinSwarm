from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYMBOLS = ("TSLA", "NVDA", "AAPL")

RAW_ROOT = Path("data/raw/prices/massive")
OUTPUT_PATH = Path("data/manifests/price_split_manifest.json")

WARMUP_DAYS = 40
TRAIN_DAYS = 252
VALIDATION_DAYS = 63


def read_symbol_dates(symbol: str) -> set[str]:
    symbol_dir = RAW_ROOT / symbol
    files = sorted(symbol_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No Massive JSONL file found for {symbol}: {symbol_dir}"
        )

    dates: set[str] = set()

    for path in files:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                record: dict[str, Any] = json.loads(line)

                market_date = record.get("market_date")
                if not market_date:
                    raise ValueError(
                        f"Missing market_date in {path}, line {line_number}"
                    )

                dates.add(str(market_date))

    return dates


def make_period(dates: list[str]) -> dict[str, Any]:
    if not dates:
        raise ValueError("Cannot create a period from an empty date list.")

    return {
        "start_date": dates[0],
        "end_date": dates[-1],
        "trading_day_count": len(dates),
        "dates": dates,
    }


def main() -> None:
    dates_by_symbol = {
        symbol: read_symbol_dates(symbol)
        for symbol in SYMBOLS
    }

    common_dates = sorted(
        set.intersection(*dates_by_symbol.values())
    )

    required_days = (
        WARMUP_DAYS
        + TRAIN_DAYS
        + VALIDATION_DAYS
        + 1
    )

    if len(common_dates) < required_days:
        raise RuntimeError(
            f"Not enough common trading days. "
            f"Found {len(common_dates)}, require at least {required_days}."
        )

    warmup_end = WARMUP_DAYS
    train_end = warmup_end + TRAIN_DAYS
    validation_end = train_end + VALIDATION_DAYS

    warmup_dates = common_dates[:warmup_end]
    train_dates = common_dates[warmup_end:train_end]
    validation_dates = common_dates[train_end:validation_end]
    test_dates = common_dates[validation_end:]

    manifest = {
        "provider": "massive",
        "symbols": list(SYMBOLS),
        "common_start_date": common_dates[0],
        "common_end_date": common_dates[-1],
        "common_trading_day_count": len(common_dates),
        "split_policy": {
            "method": "chronological_trading_days",
            "warmup_days": WARMUP_DAYS,
            "train_days": TRAIN_DAYS,
            "validation_days": VALIDATION_DAYS,
            "test_days": len(test_dates),
            "shuffle": False,
        },
        "splits": {
            "warmup": make_period(warmup_dates),
            "train": make_period(train_dates),
            "validation": make_period(validation_dates),
            "test": make_period(test_dates),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT_PATH}")
    print(
        f"Warm-up:    {warmup_dates[0]} -> {warmup_dates[-1]} "
        f"({len(warmup_dates)} days)"
    )
    print(
        f"Train:      {train_dates[0]} -> {train_dates[-1]} "
        f"({len(train_dates)} days)"
    )
    print(
        f"Validation: {validation_dates[0]} -> {validation_dates[-1]} "
        f"({len(validation_dates)} days)"
    )
    print(
        f"Test:       {test_dates[0]} -> {test_dates[-1]} "
        f"({len(test_dates)} days)"
    )


if __name__ == "__main__":
    main()