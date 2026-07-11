from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import httpx
import yaml


CONFIG_PATH = Path("config/swarm/data_config.yaml")
BASE_URL = "https://api.massive.com/v2/reference/news"

REQUEST_INTERVAL_SECONDS = 15.0
MAX_RETRIES = 10
MAX_BACKOFF_SECONDS = 120.0


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "data_config.yaml must contain a YAML object."
        )

    if "news_data" not in config:
        raise ValueError(
            "Missing 'news_data' section in data_config.yaml."
        )

    news_config = config["news_data"]

    required_fields = [
        "api_key",
        "symbols",
        "requested_start",
        "requested_end",
        "sort",
        "order",
        "limit",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in news_config
    ]

    if missing_fields:
        raise ValueError(
            f"Missing news_data fields: {missing_fields}"
        )

    if not str(news_config["api_key"]).strip():
        raise ValueError(
            "news_data.api_key is empty in data_config.yaml."
        )

    if not news_config["symbols"]:
        raise ValueError(
            "news_data.symbols must not be empty."
        )

    return config


def parse_retry_after(
    response: httpx.Response,
) -> float | None:
    retry_after = response.headers.get("Retry-After")

    if not retry_after:
        return None

    try:
        return max(1.0, float(retry_after))
    except ValueError:
        return None


def merge_url_query(
    url: str,
    extra_params: dict[str, Any],
) -> str:
    """
    Preserve every query parameter already present in next_url,
    especially Massive's pagination cursor, then add or replace
    the supplied parameters.
    """
    absolute_url = urljoin(BASE_URL, url)
    parts = urlsplit(absolute_url)

    existing_pairs = parse_qsl(
        parts.query,
        keep_blank_values=True,
    )

    replacement_keys = {
        str(key).lower()
        for key in extra_params
    }

    merged_pairs = [
        (key, value)
        for key, value in existing_pairs
        if key.lower() not in replacement_keys
    ]

    for key, value in extra_params.items():
        merged_pairs.append(
            (str(key), str(value))
        )

    new_query = urlencode(
        merged_pairs,
        doseq=True,
    )

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            new_query,
            parts.fragment,
        )
    )


def get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None,
    description: str,
) -> httpx.Response:
    last_network_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if params is None:
                response = client.get(url)
            else:
                response = client.get(
                    url,
                    params=params,
                )

        except httpx.RequestError as exc:
            last_network_error = exc

            wait_seconds = min(
                MAX_BACKOFF_SECONDS,
                REQUEST_INTERVAL_SECONDS * attempt,
            )

            print(
                f"  Network error for {description}. "
                f"Retry {attempt}/{MAX_RETRIES} after "
                f"{wait_seconds:.0f} seconds."
            )

            time.sleep(wait_seconds)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            retry_after = parse_retry_after(response)

            if retry_after is not None:
                wait_seconds = retry_after
            else:
                wait_seconds = min(
                    MAX_BACKOFF_SECONDS,
                    REQUEST_INTERVAL_SECONDS * attempt,
                )

            print(
                f"  Massive rate limit 429 for {description}. "
                f"Retry {attempt}/{MAX_RETRIES} after "
                f"{wait_seconds:.0f} seconds."
            )

            time.sleep(wait_seconds)
            continue

        if 500 <= response.status_code < 600:
            wait_seconds = min(
                MAX_BACKOFF_SECONDS,
                REQUEST_INTERVAL_SECONDS * attempt,
            )

            print(
                f"  Massive server error "
                f"{response.status_code} for {description}. "
                f"Retry {attempt}/{MAX_RETRIES} after "
                f"{wait_seconds:.0f} seconds."
            )

            time.sleep(wait_seconds)
            continue

        raise RuntimeError(
            f"Massive request failed for {description}. "
            f"HTTP status={response.status_code}. "
            f"Response={response.text[:1000]}"
        )

    if last_network_error is not None:
        raise RuntimeError(
            f"Network request failed for {description} "
            f"after {MAX_RETRIES} retries."
        ) from last_network_error

    raise RuntimeError(
        f"Massive request failed for {description} "
        f"after {MAX_RETRIES} retries."
    )


def article_identity(
    article: dict[str, Any],
) -> str:
    article_id = article.get("id")

    if article_id:
        return f"id:{article_id}"

    article_url = str(
        article.get("article_url", "")
    )
    published_utc = str(
        article.get("published_utc", "")
    )
    title = str(
        article.get("title", "")
    )

    fallback = "|".join(
        [
            article_url,
            published_utc,
            title,
        ]
    )

    if fallback != "||":
        return f"fallback:{fallback}"

    return (
        "json:"
        + json.dumps(
            article,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def write_jsonl(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(output_path)


def download_symbol(
    client: httpx.Client,
    symbol: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    api_key = str(
        config["api_key"]
    ).strip()

    start = str(
        config["requested_start"]
    )

    end = str(
        config["requested_end"]
    )

    limit = int(
        config["limit"]
    )

    first_page_params = {
        "ticker": symbol,
        "published_utc.gte": (
            f"{start}T00:00:00Z"
        ),
        "published_utc.lte": (
            f"{end}T23:59:59Z"
        ),
        "sort": str(config["sort"]),
        "order": str(config["order"]),
        "limit": limit,
        "apiKey": api_key,
    }

    records: list[dict[str, Any]] = []
    seen_article_ids: set[str] = set()
    seen_page_urls: set[str] = set()

    duplicate_count = 0
    page_number = 0
    next_url: str | None = BASE_URL

    while next_url:
        page_number += 1

        if page_number == 1:
            request_url = BASE_URL
            request_params: dict[str, Any] | None = (
                first_page_params
            )
            page_identity = str(
                httpx.URL(
                    BASE_URL,
                    params=first_page_params,
                )
            )
        else:
            request_url = merge_url_query(
                next_url,
                {
                    "apiKey": api_key,
                    "limit": limit,
                },
            )

            request_params = None
            page_identity = request_url

        if page_identity in seen_page_urls:
            raise RuntimeError(
                f"Pagination loop detected for {symbol}. "
                f"The same page URL was requested twice."
            )

        seen_page_urls.add(page_identity)

        response = get_with_retry(
            client=client,
            url=request_url,
            params=request_params,
            description=(
                f"{symbol} news page {page_number}"
            ),
        )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Massive returned invalid JSON for "
                f"{symbol}, page {page_number}."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Massive returned a non-object response "
                f"for {symbol}, page {page_number}."
            )

        status = payload.get("status")

        if status not in (None, "OK"):
            raise RuntimeError(
                f"Massive returned API status={status} "
                f"for {symbol}, page {page_number}. "
                f"Response={str(payload)[:1000]}"
            )

        page_results = payload.get(
            "results",
            [],
        )

        if not isinstance(page_results, list):
            raise RuntimeError(
                f"Massive results is not a list for "
                f"{symbol}, page {page_number}."
            )

        new_article_count = 0

        for article in page_results:
            if not isinstance(article, dict):
                continue

            identity = article_identity(article)

            if identity in seen_article_ids:
                duplicate_count += 1
                continue

            seen_article_ids.add(identity)

            records.append(
                {
                    "queried_symbol": symbol,
                    **article,
                }
            )

            new_article_count += 1

        print(
            f"  Page {page_number}: "
            f"{len(page_results)} received, "
            f"{new_article_count} new, "
            f"{len(records)} total"
        )

        next_url_value = payload.get(
            "next_url"
        )

        if not next_url_value:
            next_url = None
            continue

        if new_article_count == 0:
            raise RuntimeError(
                f"Pagination made no progress for {symbol} "
                f"at page {page_number}. "
                f"No new articles were returned."
            )

        next_url = str(
            next_url_value
        )

        print(
            f"  Waiting "
            f"{REQUEST_INTERVAL_SECONDS:.0f} seconds "
            f"before next page..."
        )

        time.sleep(
            REQUEST_INTERVAL_SECONDS
        )

    records.sort(
        key=lambda row: (
            str(
                row.get(
                    "published_utc",
                    "",
                )
            ),
            str(
                row.get(
                    "id",
                    "",
                )
            ),
        )
    )

    output_path = (
        Path("data/raw/news/massive")
        / symbol
        / f"{start}_{end}.jsonl"
    )

    write_jsonl(
        output_path=output_path,
        records=records,
    )

    publishers = Counter(
        str(
            (
                record.get("publisher")
                or {}
            ).get(
                "name",
                "UNKNOWN",
            )
        )
        for record in records
    )

    published_values = [
        str(record["published_utc"])
        for record in records
        if record.get("published_utc")
    ]

    empty_title_count = sum(
        not str(
            record.get(
                "title",
                "",
            )
        ).strip()
        for record in records
    )

    empty_description_count = sum(
        not str(
            record.get(
                "description",
                "",
            )
        ).strip()
        for record in records
    )

    empty_published_utc_count = sum(
        not record.get("published_utc")
        for record in records
    )

    return {
        "symbol": symbol,
        "requested_start": start,
        "requested_end": end,
        "record_count": len(records),
        "page_count": page_number,
        "duplicate_count": duplicate_count,
        "first_published_utc": (
            published_values[0]
            if published_values
            else None
        ),
        "last_published_utc": (
            published_values[-1]
            if published_values
            else None
        ),
        "empty_title_count": (
            empty_title_count
        ),
        "empty_description_count": (
            empty_description_count
        ),
        "empty_published_utc_count": (
            empty_published_utc_count
        ),
        "publisher_count": len(publishers),
        "top_publishers": (
            publishers.most_common(10)
        ),
        "output_path": str(output_path),
    }


def main() -> None:
    config = load_config()
    news_config = config["news_data"]

    symbols = [
        str(symbol)
        for symbol in news_config["symbols"]
    ]

    summaries: list[dict[str, Any]] = []

    with httpx.Client(
        timeout=httpx.Timeout(
            connect=30.0,
            read=60.0,
            write=30.0,
            pool=30.0,
        ),
        follow_redirects=True,
        headers={
            "Accept": "application/json",
            "User-Agent": "FinSwarm-Research/1.0",
        },
    ) as client:
        for index, symbol in enumerate(symbols):
            print(
                f"Downloading news for {symbol}..."
            )

            summary = download_symbol(
                client=client,
                symbol=symbol,
                config=news_config,
            )

            summaries.append(summary)

            print(
                f"  Completed {symbol}: "
                f"{summary['record_count']} articles, "
                f"{summary['page_count']} pages"
            )

            print(
                f"  Coverage: "
                f"{summary['first_published_utc']} -> "
                f"{summary['last_published_utc']}"
            )

            if index < len(symbols) - 1:
                print(
                    f"  Waiting "
                    f"{REQUEST_INTERVAL_SECONDS:.0f} seconds "
                    f"before next symbol..."
                )

                time.sleep(
                    REQUEST_INTERVAL_SECONDS
                )

    report = {
        "provider": "massive",
        "requested_start": (
            news_config["requested_start"]
        ),
        "requested_end": (
            news_config["requested_end"]
        ),
        "symbols": symbols,
        "summaries": summaries,
        "total_symbol_article_records": sum(
            item["record_count"]
            for item in summaries
        ),
        "total_duplicate_count": sum(
            item["duplicate_count"]
            for item in summaries
        ),
        "total_page_count": sum(
            item["page_count"]
            for item in summaries
        ),
    }

    report_path = Path(
        "data/reports/massive_news_coverage.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Created report: {report_path}"
    )
    print(
        "Total article-symbol records: "
        f"{report['total_symbol_article_records']}"
    )
    print(
        "Total pages: "
        f"{report['total_page_count']}"
    )


if __name__ == "__main__":
    main()