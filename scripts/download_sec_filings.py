from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx
import yaml


DEFAULT_CONFIG_PATH = Path("config/swarm/data_config.yaml")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_HISTORY_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
    "{cik_no_zeroes}/{accession_compact}/index.json"
)
ARCHIVE_FILE_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
    "{cik_no_zeroes}/{accession_compact}/{filename}"
)

MAX_RETRIES = 8
REQUEST_INTERVAL_SECONDS = 0.25
MAX_BACKOFF_SECONDS = 60.0


@dataclass
class RateLimiter:
    interval_seconds: float
    last_request_at: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        remaining = self.interval_seconds - elapsed

        if remaining > 0:
            time.sleep(remaining)

        self.last_request_at = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download complete SEC filing directories, including the "
            "primary filing, exhibits, XBRL files, images, and index.json."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config/swarm/data_config.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when a valid local copy already exists.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML root must be an object.")

    if "filing_data" not in config:
        raise ValueError("Missing 'filing_data' in data_config.yaml.")

    filing_config = config["filing_data"]

    required_fields = [
        "requested_start",
        "requested_end",
        "user_agent",
        "forms",
        "companies",
    ]

    missing = [
        field
        for field in required_fields
        if field not in filing_config
    ]

    if missing:
        raise ValueError(f"Missing filing_data fields: {missing}")

    user_agent = str(filing_config["user_agent"]).strip()

    if not user_agent or "@" not in user_agent:
        raise ValueError(
            "filing_data.user_agent should identify the project and include "
            "a contact email, for example: "
            "'FinSwarm Research your_email@example.com'."
        )

    if not filing_config["forms"]:
        raise ValueError("filing_data.forms must not be empty.")

    if not filing_config["companies"]:
        raise ValueError("filing_data.companies must not be empty.")

    return config


def normalize_cik(value: Any) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())

    if not digits:
        raise ValueError(f"Invalid CIK: {value!r}")

    return digits.zfill(10)


def parse_retry_after(response: httpx.Response) -> float | None:
    retry_after = response.headers.get("Retry-After")

    if not retry_after:
        return None

    try:
        return max(1.0, float(retry_after))
    except ValueError:
        return None


def request_with_retry(
    client: httpx.Client,
    limiter: RateLimiter,
    url: str,
    description: str,
) -> httpx.Response:
    last_network_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()

        try:
            response = client.get(url)
        except httpx.RequestError as exc:
            last_network_error = exc
            wait_seconds = min(
                MAX_BACKOFF_SECONDS,
                2.0 * attempt,
            )

            print(
                f"    Network error for {description}. "
                f"Retry {attempt}/{MAX_RETRIES} after "
                f"{wait_seconds:.0f}s."
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            retry_after = parse_retry_after(response)
            wait_seconds = (
                retry_after
                if retry_after is not None
                else min(MAX_BACKOFF_SECONDS, 5.0 * attempt)
            )

            print(
                f"    SEC rate limit 429 for {description}. "
                f"Retry {attempt}/{MAX_RETRIES} after "
                f"{wait_seconds:.0f}s."
            )
            time.sleep(wait_seconds)
            continue

        if 500 <= response.status_code < 600:
            wait_seconds = min(
                MAX_BACKOFF_SECONDS,
                3.0 * attempt,
            )

            print(
                f"    SEC server error {response.status_code} for "
                f"{description}. Retry {attempt}/{MAX_RETRIES} after "
                f"{wait_seconds:.0f}s."
            )
            time.sleep(wait_seconds)
            continue

        raise RuntimeError(
            f"SEC request failed for {description}. "
            f"HTTP status={response.status_code}. "
            f"URL={url}. Response={response.text[:1000]}"
        )

    if last_network_error is not None:
        raise RuntimeError(
            f"Network request failed for {description} after "
            f"{MAX_RETRIES} retries."
        ) from last_network_error

    raise RuntimeError(
        f"SEC request failed for {description} after "
        f"{MAX_RETRIES} retries."
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    content = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    atomic_write_bytes(path, content)


def parse_json_response(
    response: httpx.Response,
    description: str,
) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"SEC returned invalid JSON for {description}."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"SEC returned a non-object JSON payload for {description}."
        )

    return payload


def get_column(
    table: dict[str, Any],
    name: str,
    index: int,
    default: Any = None,
) -> Any:
    values = table.get(name, [])

    if not isinstance(values, list):
        return default

    if index >= len(values):
        return default

    return values[index]


def extract_filing_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        isinstance(payload.get("filings"), dict)
        and isinstance(payload["filings"].get("recent"), dict)
    ):
        table = payload["filings"]["recent"]
    else:
        table = payload

    accession_numbers = table.get("accessionNumber", [])

    if not isinstance(accession_numbers, list):
        return []

    fields = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "act",
        "form",
        "fileNumber",
        "filmNumber",
        "items",
        "size",
        "isXBRL",
        "isInlineXBRL",
        "primaryDocument",
        "primaryDocDescription",
    ]

    rows: list[dict[str, Any]] = []

    for index in range(len(accession_numbers)):
        row = {
            field: get_column(table, field, index)
            for field in fields
        }
        rows.append(row)

    return rows


def history_file_overlaps(
    entry: dict[str, Any],
    requested_start: str,
    requested_end: str,
) -> bool:
    filing_from = str(entry.get("filingFrom") or "")
    filing_to = str(entry.get("filingTo") or "")

    if filing_to and filing_to < requested_start:
        return False

    if filing_from and filing_from > requested_end:
        return False

    return True


def load_all_submission_rows(
    client: httpx.Client,
    limiter: RateLimiter,
    company_root: Path,
    cik: str,
    requested_start: str,
    requested_end: str,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    submissions_url = SUBMISSIONS_URL.format(cik=cik)
    submissions_path = company_root / "submissions.json"

    response = request_with_retry(
        client=client,
        limiter=limiter,
        url=submissions_url,
        description=f"CIK {cik} submissions",
    )

    atomic_write_bytes(submissions_path, response.content)
    submissions = parse_json_response(
        response,
        description=f"CIK {cik} submissions",
    )

    all_rows = extract_filing_rows(submissions)

    history_entries = (
        submissions.get("filings", {}).get("files", [])
        if isinstance(submissions.get("filings"), dict)
        else []
    )

    history_root = company_root / "submissions_history"

    if isinstance(history_entries, list):
        for entry in history_entries:
            if not isinstance(entry, dict):
                continue

            name = str(entry.get("name") or "").strip()

            if not name:
                continue

            if not history_file_overlaps(
                entry,
                requested_start=requested_start,
                requested_end=requested_end,
            ):
                continue

            history_path = history_root / name
            history_url = SUBMISSIONS_HISTORY_URL.format(
                name=quote(name, safe="._-")
            )

            if (
                history_path.exists()
                and history_path.stat().st_size > 0
                and not force
            ):
                try:
                    history_payload = json.loads(
                        history_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    history_payload = None
            else:
                history_response = request_with_retry(
                    client=client,
                    limiter=limiter,
                    url=history_url,
                    description=f"historical submissions {name}",
                )
                atomic_write_bytes(
                    history_path,
                    history_response.content,
                )
                history_payload = parse_json_response(
                    history_response,
                    description=f"historical submissions {name}",
                )

            if isinstance(history_payload, dict):
                all_rows.extend(
                    extract_filing_rows(history_payload)
                )

    unique_rows: dict[str, dict[str, Any]] = {}

    for row in all_rows:
        accession_number = str(
            row.get("accessionNumber") or ""
        ).strip()

        if accession_number:
            unique_rows[accession_number] = row

    return submissions, list(unique_rows.values())


def validate_archive_filename(name: str) -> PurePosixPath:
    path = PurePosixPath(name)

    if path.is_absolute():
        raise ValueError(f"Absolute archive path is not allowed: {name}")

    if not path.parts:
        raise ValueError("Empty archive filename.")

    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe archive filename: {name}")

    return path


def expected_size_from_item(item: dict[str, Any]) -> int | None:
    value = item.get("size")

    if value in (None, ""):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def local_file_is_complete(
    path: Path,
    expected_size: int | None,
) -> bool:
    if not path.exists() or not path.is_file():
        return False

    actual_size = path.stat().st_size

    if expected_size is not None:
        return actual_size == expected_size

    return actual_size > 0


def download_archive_file(
    client: httpx.Client,
    limiter: RateLimiter,
    url: str,
    destination: Path,
    description: str,
    expected_size: int | None,
    force: bool,
) -> str:
    if (
        not force
        and local_file_is_complete(
            destination,
            expected_size=expected_size,
        )
    ):
        return "skipped_existing"

    response = request_with_retry(
        client=client,
        limiter=limiter,
        url=url,
        description=description,
    )

    atomic_write_bytes(destination, response.content)

    if expected_size is not None:
        actual_size = destination.stat().st_size

        if actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {description}. "
                f"Expected {expected_size}, downloaded {actual_size}."
            )

    return "downloaded"


def download_filing_directory(
    client: httpx.Client,
    limiter: RateLimiter,
    symbol: str,
    cik: str,
    filing_row: dict[str, Any],
    company_root: Path,
    force: bool,
) -> dict[str, Any]:
    accession_number = str(
        filing_row.get("accessionNumber") or ""
    ).strip()

    if not accession_number:
        raise ValueError("Missing accession number.")

    accession_compact = accession_number.replace("-", "")
    cik_no_zeroes = str(int(cik))
    filing_root = company_root / accession_number
    filing_root.mkdir(parents=True, exist_ok=True)

    index_url = ARCHIVE_INDEX_URL.format(
        cik_no_zeroes=cik_no_zeroes,
        accession_compact=accession_compact,
    )
    index_path = filing_root / "index.json"

    index_response = request_with_retry(
        client=client,
        limiter=limiter,
        url=index_url,
        description=f"{symbol} {accession_number} index.json",
    )

    atomic_write_bytes(index_path, index_response.content)
    index_payload = parse_json_response(
        index_response,
        description=f"{symbol} {accession_number} index.json",
    )

    directory = index_payload.get("directory", {})

    if not isinstance(directory, dict):
        raise RuntimeError(
            f"Invalid SEC directory index for {symbol} "
            f"{accession_number}: missing directory object."
        )

    items = directory.get("item", [])

    if not isinstance(items, list):
        raise RuntimeError(
            f"Invalid SEC directory index for {symbol} "
            f"{accession_number}: directory.item is not a list."
        )

    downloaded_count = 0
    skipped_existing_count = 0
    failed_count = 0
    file_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "").lower()
        name = str(item.get("name") or "").strip()

        if not name:
            continue

        if item_type in {"dir", "directory"}:
            continue

        if name == "index.json":
            continue

        try:
            safe_relative_path = validate_archive_filename(name)
        except ValueError as exc:
            failed_count += 1
            errors.append(
                {
                    "filename": name,
                    "error": str(exc),
                }
            )
            continue

        expected_size = expected_size_from_item(item)
        destination = filing_root.joinpath(
            *safe_relative_path.parts
        )

        encoded_name = quote(
            safe_relative_path.as_posix(),
            safe="/._-()",
        )

        file_url = ARCHIVE_FILE_URL.format(
            cik_no_zeroes=cik_no_zeroes,
            accession_compact=accession_compact,
            filename=encoded_name,
        )

        try:
            status = download_archive_file(
                client=client,
                limiter=limiter,
                url=file_url,
                destination=destination,
                description=(
                    f"{symbol} {accession_number} {name}"
                ),
                expected_size=expected_size,
                force=force,
            )

            if status == "downloaded":
                downloaded_count += 1
            else:
                skipped_existing_count += 1

            file_records.append(
                {
                    "name": name,
                    "type": item.get("type"),
                    "expected_size": expected_size,
                    "local_size": (
                        destination.stat().st_size
                        if destination.exists()
                        else None
                    ),
                    "status": status,
                    "url": file_url,
                    "local_path": str(destination),
                }
            )

        except Exception as exc:
            failed_count += 1
            error_text = str(exc)

            print(
                f"    Failed {symbol} {accession_number} "
                f"{name}: {error_text}"
            )

            errors.append(
                {
                    "filename": name,
                    "error": error_text,
                }
            )

    primary_document = str(
        filing_row.get("primaryDocument") or ""
    ).strip()
    primary_path = (
        filing_root / primary_document
        if primary_document
        else None
    )

    metadata = {
        "symbol": symbol,
        "cik": cik,
        "accession_number": accession_number,
        "accession_compact": accession_compact,
        "form": filing_row.get("form"),
        "filing_date": filing_row.get("filingDate"),
        "report_date": filing_row.get("reportDate"),
        "acceptance_datetime": filing_row.get(
            "acceptanceDateTime"
        ),
        "act": filing_row.get("act"),
        "file_number": filing_row.get("fileNumber"),
        "film_number": filing_row.get("filmNumber"),
        "items": filing_row.get("items"),
        "reported_size": filing_row.get("size"),
        "is_xbrl": filing_row.get("isXBRL"),
        "is_inline_xbrl": filing_row.get("isInlineXBRL"),
        "primary_document": primary_document,
        "primary_document_description": filing_row.get(
            "primaryDocDescription"
        ),
        "primary_document_local_path": (
            str(primary_path)
            if primary_path is not None
            else None
        ),
        "primary_document_exists": (
            primary_path.exists()
            if primary_path is not None
            else False
        ),
        "index_url": index_url,
        "index_local_path": str(index_path),
        "directory_file_count": len(file_records),
        "downloaded_file_count": downloaded_count,
        "skipped_existing_file_count": (
            skipped_existing_count
        ),
        "failed_file_count": failed_count,
        "files": file_records,
        "errors": errors,
    }

    atomic_write_json(
        filing_root / "filing_metadata.json",
        metadata,
    )

    return metadata


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(path)


def download_company(
    client: httpx.Client,
    limiter: RateLimiter,
    symbol: str,
    cik: str,
    config: dict[str, Any],
    raw_root: Path,
    force: bool,
) -> dict[str, Any]:
    requested_start = str(config["requested_start"])
    requested_end = str(config["requested_end"])
    allowed_forms = {
        str(form)
        for form in config["forms"]
    }

    company_root = raw_root / symbol
    company_root.mkdir(parents=True, exist_ok=True)

    _, all_submission_rows = load_all_submission_rows(
        client=client,
        limiter=limiter,
        company_root=company_root,
        cik=cik,
        requested_start=requested_start,
        requested_end=requested_end,
        force=force,
    )

    selected_rows = [
        row
        for row in all_submission_rows
        if (
            str(row.get("form") or "") in allowed_forms
            and requested_start
            <= str(row.get("filingDate") or "")
            <= requested_end
        )
    ]

    selected_rows.sort(
        key=lambda row: (
            str(row.get("acceptanceDateTime") or ""),
            str(row.get("filingDate") or ""),
            str(row.get("accessionNumber") or ""),
        )
    )

    filing_records: list[dict[str, Any]] = []
    failed_filings: list[dict[str, str]] = []

    for index, filing_row in enumerate(
        selected_rows,
        start=1,
    ):
        accession_number = str(
            filing_row.get("accessionNumber") or ""
        )
        form = str(filing_row.get("form") or "")

        print(
            f"  [{index}/{len(selected_rows)}] "
            f"{form} {accession_number}"
        )

        try:
            filing_metadata = download_filing_directory(
                client=client,
                limiter=limiter,
                symbol=symbol,
                cik=cik,
                filing_row=filing_row,
                company_root=company_root,
                force=force,
            )
            filing_records.append(filing_metadata)

        except Exception as exc:
            error_text = str(exc)

            print(
                f"    Filing failed: {error_text}"
            )

            failed_filings.append(
                {
                    "accession_number": accession_number,
                    "form": form,
                    "error": error_text,
                }
            )

    metadata_path = company_root / (
        f"{requested_start}_{requested_end}_metadata.jsonl"
    )

    write_jsonl(
        metadata_path,
        filing_records,
    )

    form_counts: dict[str, int] = {}

    for record in filing_records:
        form = str(record.get("form") or "UNKNOWN")
        form_counts[form] = form_counts.get(form, 0) + 1

    return {
        "symbol": symbol,
        "cik": cik,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "selected_filing_count": len(selected_rows),
        "completed_filing_count": len(filing_records),
        "failed_filing_count": len(failed_filings),
        "form_counts": form_counts,
        "downloaded_file_count": sum(
            int(record["downloaded_file_count"])
            for record in filing_records
        ),
        "skipped_existing_file_count": sum(
            int(record["skipped_existing_file_count"])
            for record in filing_records
        ),
        "failed_file_count": sum(
            int(record["failed_file_count"])
            for record in filing_records
        ),
        "first_acceptance_datetime": (
            filing_records[0].get("acceptance_datetime")
            if filing_records
            else None
        ),
        "last_acceptance_datetime": (
            filing_records[-1].get("acceptance_datetime")
            if filing_records
            else None
        ),
        "metadata_path": str(metadata_path),
        "failed_filings": failed_filings,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    filing_config = config["filing_data"]
    paths_config = config.get("paths", {})

    raw_base = Path(
        paths_config.get("raw", "data/raw")
    )
    reports_root = Path(
        paths_config.get("reports", "data/reports")
    )

    raw_root = raw_base / "filings" / "sec"
    raw_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": str(filing_config["user_agent"]),
        "Accept-Encoding": "gzip, deflate",
        "Accept": (
            "application/json,text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
    }

    limiter = RateLimiter(
        interval_seconds=REQUEST_INTERVAL_SECONDS
    )

    summaries: list[dict[str, Any]] = []

    with httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(
            connect=30.0,
            read=120.0,
            write=30.0,
            pool=30.0,
        ),
        follow_redirects=True,
    ) as client:
        for symbol, cik_value in filing_config[
            "companies"
        ].items():
            symbol = str(symbol)
            cik = normalize_cik(cik_value)

            print(
                f"Downloading complete SEC filings for "
                f"{symbol} (CIK {cik})..."
            )

            summary = download_company(
                client=client,
                limiter=limiter,
                symbol=symbol,
                cik=cik,
                config=filing_config,
                raw_root=raw_root,
                force=args.force,
            )

            summaries.append(summary)

            print(
                f"  Completed {summary['completed_filing_count']}/"
                f"{summary['selected_filing_count']} filings; "
                f"downloaded files="
                f"{summary['downloaded_file_count']}; "
                f"skipped existing="
                f"{summary['skipped_existing_file_count']}; "
                f"file errors={summary['failed_file_count']}."
            )

    report = {
        "provider": "sec_edgar",
        "downloaded_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "requested_start": filing_config[
            "requested_start"
        ],
        "requested_end": filing_config[
            "requested_end"
        ],
        "forms": list(filing_config["forms"]),
        "summaries": summaries,
        "total_selected_filings": sum(
            summary["selected_filing_count"]
            for summary in summaries
        ),
        "total_completed_filings": sum(
            summary["completed_filing_count"]
            for summary in summaries
        ),
        "total_failed_filings": sum(
            summary["failed_filing_count"]
            for summary in summaries
        ),
        "total_downloaded_files": sum(
            summary["downloaded_file_count"]
            for summary in summaries
        ),
        "total_skipped_existing_files": sum(
            summary["skipped_existing_file_count"]
            for summary in summaries
        ),
        "total_failed_files": sum(
            summary["failed_file_count"]
            for summary in summaries
        ),
    }

    report_path = reports_root / "sec_filing_coverage.json"
    atomic_write_json(report_path, report)

    print()
    print(f"Created report: {report_path}")
    print(
        f"Completed filings: "
        f"{report['total_completed_filings']}/"
        f"{report['total_selected_filings']}"
    )
    print(
        f"Downloaded files: "
        f"{report['total_downloaded_files']}"
    )
    print(
        f"Skipped existing files: "
        f"{report['total_skipped_existing_files']}"
    )
    print(
        f"Failed files: "
        f"{report['total_failed_files']}"
    )

    if (
        report["total_failed_filings"] > 0
        or report["total_failed_files"] > 0
    ):
        raise RuntimeError(
            "SEC download finished with errors. Inspect "
            "data/reports/sec_filing_coverage.json."
        )


if __name__ == "__main__":
    main()