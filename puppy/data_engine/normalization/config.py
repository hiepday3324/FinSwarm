from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


NORMALIZER_VERSION = "stage4-v1"
UTC = dt.timezone.utc


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: Any, field_name: str) -> dt.datetime:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required")
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def cutoff_utc(date: dt.date, cutoff: dt.time, timezone: ZoneInfo) -> dt.datetime:
    return dt.datetime.combine(date, cutoff, timezone).astimezone(UTC)


@dataclass(frozen=True)
class Stage4Config:
    config_path: Path
    repo_root: Path
    run_id: str
    symbols: tuple[str, ...]
    start: dt.date
    end: dt.date
    timezone: ZoneInfo
    decision_cutoff: dt.time
    split_manifest: Path
    raw_prices: Path
    raw_news: Path
    raw_filings: Path
    canonical_prices: Path
    canonical_news: Path
    canonical_filings: Path
    duckdb_path: Path
    include_8k_exhibits: tuple[str, ...]

    def relative(self, path: str | Path) -> str:
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return candidate.as_posix()


def load_stage4_config(path: str | Path) -> Stage4Config:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage 4 config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Stage 4 config must be a mapping: {config_path}")
    repo_root = config_path.parents[2]

    def required(*keys: str) -> Any:
        value: Any = raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"Missing config field: {'.'.join(keys)}")
            value = value[key]
        return value

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else repo_root / candidate

    cutoff_text = str(required("time", "decision_cutoff"))
    return Stage4Config(
        config_path=config_path,
        repo_root=repo_root,
        run_id=str(required("run", "run_id")),
        symbols=tuple(str(item) for item in required("assets", "symbols")),
        start=dt.date.fromisoformat(str(required("data", "start"))),
        end=dt.date.fromisoformat(str(required("data", "end"))),
        timezone=ZoneInfo(str(required("time", "timezone"))),
        decision_cutoff=dt.time.fromisoformat(cutoff_text),
        split_manifest=resolve(str(required("data", "split_manifest"))),
        raw_prices=resolve(str(required("data", "raw", "prices"))),
        raw_news=resolve(str(required("data", "raw", "news"))),
        raw_filings=resolve(str(required("data", "raw", "filings"))),
        canonical_prices=resolve(str(required("data", "canonical", "prices"))),
        canonical_news=resolve(str(required("data", "canonical", "news"))),
        canonical_filings=resolve(str(required("data", "canonical", "filings"))),
        duckdb_path=resolve(str(required("data", "duckdb"))),
        include_8k_exhibits=tuple(
            str(item).upper()
            for item in required("normalization", "filings", "include_8k_exhibits")
        ),
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records
