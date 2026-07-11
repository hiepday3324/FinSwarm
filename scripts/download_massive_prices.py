from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv
from massive import RESTClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily Massive OHLCV data for FinSwarm."
    )
    parser.add_argument(
        "--config",
        default="config/swarm/data_config.yaml",
        help="Path to data configuration YAML.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing downloaded files.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "price_data" not in config:
        raise ValueError("Missing 'price_data' section in config.")

    return config


def aggregate_to_raw_record(
    symbol: str,
    aggregate: Any,
    market_timezone: ZoneInfo,
) -> dict[str, Any]:
    timestamp_ms = int(aggregate.timestamp)

    timestamp_utc = datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    )
    timestamp_market = timestamp_utc.astimezone(market_timezone)

    return {
        # Preserve Massive field names.
        "ticker": symbol,
        "v": float(aggregate.volume),
        "vw": (
            float(aggregate.vwap)
            if getattr(aggregate, "vwap", None) is not None
            else None
        ),
        "o": float(aggregate.open),
        "c": float(aggregate.close),
        "h": float(aggregate.high),
        "l": float(aggregate.low),
        "t": timestamp_ms,
        "n": (
            int(aggregate.transactions)
            if getattr(aggregate, "transactions", None) is not None
            else None
        ),
        "otc": getattr(aggregate, "otc", None),

        # Derived audit fields.
        "market_date": timestamp_market.date().isoformat(),
        "timestamp_utc": timestamp_utc.isoformat(),
        "timestamp_market": timestamp_market.isoformat(),
    }


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_dates: set[str] = set()

    for index, row in enumerate(records):
        market_date = row["market_date"]

        if market_date in seen_dates:
            errors.append(
                f"duplicate market date at row {index}: {market_date}"
            )
        seen_dates.add(market_date)

        open_price = row["o"]
        high_price = row["h"]
        low_price = row["l"]
        close_price = row["c"]
        volume = row["v"]

        if min(open_price, high_price, low_price, close_price) <= 0:
            errors.append(f"non-positive price at {market_date}")

        if not low_price <= open_price <= high_price:
            errors.append(f"open outside OHLC range at {market_date}")

        if not low_price <= close_price <= high_price:
            errors.append(f"close outside OHLC range at {market_date}")

        if volume < 0:
            errors.append(f"negative volume at {market_date}")

    timestamps = [row["t"] for row in records]

    if timestamps != sorted(timestamps):
        errors.append("records are not sorted by timestamp")

    return errors


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_symbol(
    client: RESTClient,
    symbol: str,
    price_config: dict[str, Any],
    raw_root: Path,
    force: bool,
) -> tuple[dict[str, Any], set[str]]:
    start_date = price_config["requested_start"]
    end_date = price_config["requested_end"]

    output_path = (
        raw_root
        / "prices"
        / "massive"
        / symbol
        / f"{start_date}_{end_date}.jsonl"
    )

    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Use --force to overwrite."
        )

    timezone_name = price_config["market_timezone"]
    market_timezone = ZoneInfo(timezone_name)

    records: list[dict[str, Any]] = []

    for aggregate in client.list_aggs(
        symbol,
        int(price_config["multiplier"]),
        str(price_config["timespan"]),
        start_date,
        end_date,
        adjusted=bool(price_config["adjusted"]),
        sort=str(price_config["sort"]),
        limit=int(price_config["limit"]),
    ):
        records.append(
            aggregate_to_raw_record(
                symbol=symbol,
                aggregate=aggregate,
                market_timezone=market_timezone,
            )
        )

    records.sort(key=lambda row: row["t"])
    validation_errors = validate_records(records)
    checksum = write_jsonl(output_path, records)

    dates = {row["market_date"] for row in records}

    summary = {
        "symbol": symbol,
        "requested_start": start_date,
        "requested_end": end_date,
        "record_count": len(records),
        "first_market_date": (
            records[0]["market_date"] if records else None
        ),
        "last_market_date": (
            records[-1]["market_date"] if records else None
        ),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "sha256": checksum,
        "output_path": str(output_path),
    }

    return summary, dates


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    price_config = config["price_data"]
    paths_config = config["paths"]

    api_key_env = price_config["api_key_env"]
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not configured."
        )

    client = RESTClient(api_key)

    raw_root = Path(paths_config["raw"])
    reports_root = Path(paths_config["reports"])
    reports_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    symbol_dates: dict[str, set[str]] = {}

    for symbol in price_config["symbols"]:
        print(f"Downloading {symbol}...")

        summary, dates = download_symbol(
            client=client,
            symbol=symbol,
            price_config=price_config,
            raw_root=raw_root,
            force=args.force,
        )

        summaries.append(summary)
        symbol_dates[symbol] = dates

        print(
            f"  {summary['record_count']} bars: "
            f"{summary['first_market_date']} → "
            f"{summary['last_market_date']}"
        )

    common_dates: set[str]

    if symbol_dates:
        common_dates = set.intersection(*symbol_dates.values())
    else:
        common_dates = set()

    sorted_common_dates = sorted(common_dates)

    report = {
        "provider": "massive",
        "adjusted": bool(price_config["adjusted"]),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(price_config["symbols"]),
        "summaries": summaries,
        "common_trading_day_count": len(sorted_common_dates),
        "common_first_date": (
            sorted_common_dates[0] if sorted_common_dates else None
        ),
        "common_last_date": (
            sorted_common_dates[-1] if sorted_common_dates else None
        ),
    }

    report_path = reports_root / "massive_price_coverage.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    critical_errors = [
        summary
        for summary in summaries
        if summary["record_count"] == 0
        or summary["validation_error_count"] > 0
    ]

    print(f"\nCoverage report: {report_path}")
    print(
        f"Common trading days: "
        f"{report['common_trading_day_count']}"
    )
    print(
        f"Common range: "
        f"{report['common_first_date']} → "
        f"{report['common_last_date']}"
    )

    if critical_errors:
        raise RuntimeError(
            "Price download completed with critical validation errors. "
            "Inspect massive_price_coverage.json."
        )


if __name__ == "__main__":
    main()