from __future__ import annotations

import hashlib
import html
import mimetypes
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "caption", "div", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "p",
    "section", "table", "td", "th", "tr", "ul", "ol",
}
_SKIP_TAGS = {"script", "style", "noscript", "template", "ix:hidden"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_SUPPORT_EXTENSIONS = {
    ".css", ".js", ".xsd", ".xml", ".json", ".zip", ".xlsx", ".xls", ".png",
    ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".woff", ".woff2", ".ttf",
}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if self.skip_depth:
            if normalized not in _VOID_TAGS:
                self.skip_depth += 1
            return
        attrs_dict = {key.lower(): (value or "").lower() for key, value in attrs}
        hidden = (
            normalized in _SKIP_TAGS
            or "hidden" in attrs_dict
            or attrs_dict.get("aria-hidden") == "true"
            or "display:none" in attrs_dict.get("style", "").replace(" ", "")
        )
        if hidden:
            self.skip_depth += 1
            return
        if self.skip_depth == 0 and normalized in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_depth == 0 and tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if normalized in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)


def _normalize_text(text: str) -> str:
    lines: list[str] = []
    previous = None
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", html.unescape(raw_line)).strip()
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines).strip()


def html_to_text(source: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(source)
    parser.close()
    return _normalize_text("".join(parser.parts))


def _is_exhibit_99(name: str, document_type: str) -> bool:
    normalized_type = re.sub(r"[^A-Z0-9]", "", document_type.upper())
    normalized_name = re.sub(r"[^A-Z0-9]", "", Path(name).stem.upper())
    return normalized_type.startswith("EX99") or "EX99" in normalized_name


def _decision(
    name: str,
    document_type: str,
    filing_type: str,
    primary_document: str,
) -> tuple[bool, str]:
    suffix = Path(name).suffix.lower()
    if Path(name).name.lower() == Path(primary_document).name.lower():
        if suffix in _SUPPORT_EXTENSIONS:
            return False, "primary_document_has_unsupported_type"
        return True, "primary_document"
    if suffix in _SUPPORT_EXTENSIONS:
        return False, "support_or_binary_document"
    if filing_type.upper() in {"8-K", "8-K/A"} and _is_exhibit_99(name, document_type):
        if suffix in {".htm", ".html", ".txt"}:
            return True, "material_exhibit_99"
        return False, "material_exhibit_has_unsupported_type"
    return False, "not_primary_or_material_exhibit"


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8", errors="replace")
    if path.suffix.lower() in {".htm", ".html"}:
        return html_to_text(decoded)
    return _normalize_text(decoded)


def extract_filing_documents(
    symbol: str,
    accession_number: str,
    filing_type: str,
    primary_document: str,
    documents: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    included_documents: list[dict[str, Any]] = []
    excluded_documents: list[dict[str, Any]] = []
    sections: list[str] = []
    ordered = sorted(documents, key=lambda item: str(item.get("name") or item.get("local_path") or "").lower())
    ordered.sort(key=lambda item: Path(str(item.get("name") or "")).name.lower() != Path(primary_document).name.lower())
    for document in ordered:
        name = str(document.get("name") or Path(str(document.get("local_path") or "")).name)
        document_type = str(document.get("type") or "")
        local_path = Path(str(document.get("local_path") or ""))
        include, reason = _decision(name, document_type, filing_type, primary_document)
        extracted = ""
        digest = ""
        if local_path.is_file():
            digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        elif include:
            include = False
            reason = "selected_document_missing"
        if include:
            try:
                extracted = _read_text(local_path)
            except (OSError, UnicodeError) as exc:
                include = False
                reason = f"extraction_failed:{type(exc).__name__}"
            if include and not extracted:
                include = False
                reason = "empty_extracted_text"
        record = {
            "name": name,
            "type": document_type,
            "local_path": local_path.as_posix(),
            "sha256": digest,
            "decision_reason": reason,
        }
        if include:
            included_documents.append(record)
            sections.append(f"===== SEC DOCUMENT: {name} [{document_type or 'UNKNOWN'}] =====\n{extracted}")
        else:
            excluded_documents.append(record)
        audit.append(
            {
                "symbol": symbol,
                "accession_number": accession_number,
                "filing_type": filing_type,
                "document_name": name,
                "document_type": document_type,
                "local_path": local_path.as_posix(),
                "included": include,
                "decision_reason": reason,
                "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
                "extracted_character_count": len(extracted) if include else 0,
                "sha256": digest,
            }
        )
    return "\n\n".join(sections), audit, included_documents, excluded_documents
