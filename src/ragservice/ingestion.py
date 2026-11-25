"""Multi-format document ingestion pipeline: Markdown, HTML, CSV, and table-aware chunking.

Enterprise corpora rarely consist of clean prose alone. HTML pages contain navigation
chrome and scripts, Markdown docs contain pipe tables that get mangled by naive paragraph
splitting, and CSV exports lose column meaning if rows are split without repeating the
header row.

This module provides a unified ``ingest_document`` entrypoint that:
1. Detects the document format (from ``metadata["format"]``, file extension, or content).
2. Parses HTML using stdlib ``html.parser`` (stripping ``<script>``, ``<style>``, ``<nav>``,
   converting headings to Markdown levels and ``<table>`` elements to pipe tables).
3. Chunks CSV files in row batches while repeating the header row on every chunk.
4. Splits Markdown sections into prose and pipe-table blocks, preserving table headers
   when large tables span multiple chunks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

from ragservice.chunking import Chunk, Document, _sections, chunk_document

FormatType = Literal["markdown", "html", "csv", "text"]

_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_HTML_TAG_RE = re.compile(r"<\s*(html|body|div|p|h[1-6]|table|article|section)\b", re.IGNORECASE)


def detect_format(doc_id: str, text: str, explicit_format: str | None = None) -> FormatType:
    """Infer document format from explicit hint, file extension, or content sniffing."""
    if explicit_format:
        fmt = explicit_format.lower().strip()
        if fmt in ("markdown", "md"):
            return "markdown"
        if fmt in ("html", "htm"):
            return "html"
        if fmt == "csv":
            return "csv"
        return "text"

    suffix = Path(doc_id).suffix.lower()
    if suffix in (".html", ".htm"):
        return "html"
    if suffix == ".csv":
        return "csv"
    if suffix in (".md", ".markdown"):
        return "markdown"

    stripped = text.lstrip()
    if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html") or _HTML_TAG_RE.search(stripped[:512]):
        return "html"
    return "markdown" if re.search(r"^#{1,6}\s+", text, re.MULTILINE) else "text"


class _HTMLToMarkdownParser(HTMLParser):
    """Minimal dependency-free HTML-to-Markdown converter preserving headings and tables."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "nav", "footer", "svg"})
    _BLOCK_TAGS = frozenset({"p", "div", "section", "article", "li", "blockquote", "pre", "br", "hr"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._heading_level = 0
        self._in_table = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._table_rows: list[list[str]] = []
        self._in_cell = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._heading_level = int(tag[1])
            self._parts.append(f"\n\n{'#' * self._heading_level} ")
        elif tag == "table":
            self._in_table = True
            self._table_rows = []
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._current_cell = []
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._heading_level = 0
            self._parts.append("\n\n")
        elif tag in ("td", "th") and self._in_table and self._in_cell:
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._in_cell = False
        elif tag == "tr" and self._in_table:
            if any(c for c in self._current_row):
                self._table_rows.append(self._current_row)
            self._current_row = []
        elif tag == "table" and self._in_table:
            self._flush_table()
            self._in_table = False
        elif tag in ("p", "div", "section", "article"):
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_table and self._in_cell:
            self._current_cell.append(data)
        elif not self._in_table:
            self._parts.append(data)

    def _flush_table(self) -> None:
        if not self._table_rows:
            return
        width = max(len(r) for r in self._table_rows)
        norm = [r + [""] * (width - len(r)) for r in self._table_rows]
        header = "| " + " | ".join(norm[0]) + " |"
        sep = "| " + " | ".join(["---"] * width) + " |"
        body = ["| " + " | ".join(r) + " |" for r in norm[1:]]
        self._parts.append("\n\n" + "\n".join([header, sep, *body]) + "\n\n")

    def get_markdown(self) -> str:
        raw = "".join(self._parts)
        lines = [line.rstrip() for line in raw.splitlines()]
        cleaned = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML into clean Markdown preserving headings, lists, and tables."""
    parser = _HTMLToMarkdownParser()
    parser.feed(html)
    parser.close()
    return parser.get_markdown()


def _make_structured_chunk(
    doc: Document,
    heading: str,
    text: str,
    index: int,
    extra_metadata: dict | None = None,
) -> Chunk:
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    meta = dict(doc.metadata)
    if extra_metadata:
        meta.update(extra_metadata)
    return Chunk(
        chunk_id=f"{doc.doc_id}#{index}-{digest}",
        doc_id=doc.doc_id,
        text=text,
        heading=heading,
        metadata=meta,
    )


def chunk_csv(doc: Document, max_rows_per_chunk: int = 20) -> list[Chunk]:
    """Chunk CSV data into row batches, repeating the header row on every chunk."""
    if max_rows_per_chunk < 1:
        raise ValueError("max_rows_per_chunk must be >= 1")
    reader = csv.reader(io.StringIO(doc.text.strip()))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return []
    header = [c.strip() for c in rows[0]]
    data_rows = rows[1:]
    heading = str(doc.metadata.get("title") or Path(doc.doc_id).stem)

    if not data_rows:
        md_table = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |"
        return [_make_structured_chunk(doc, heading, md_table, 0, {"content_type": "table", "format": "csv", "row_range": "0-0"})]

    chunks: list[Chunk] = []
    width = len(header)
    sep_line = "| " + " | ".join(["---"] * width) + " |"
    header_line = "| " + " | ".join(header) + " |"
    for start in range(0, len(data_rows), max_rows_per_chunk):
        batch = data_rows[start : start + max_rows_per_chunk]
        formatted_rows = []
        for row in batch:
            padded = [c.strip() for c in row] + [""] * max(0, width - len(row))
            formatted_rows.append("| " + " | ".join(padded[:width]) + " |")
        table_text = "\n".join([header_line, sep_line, *formatted_rows])
        end = start + len(batch)
        chunks.append(
            _make_structured_chunk(
                doc,
                heading,
                table_text,
                len(chunks),
                {"content_type": "table", "format": "csv", "row_range": f"{start + 1}-{end}"},
            )
        )
    return chunks


def _split_prose_and_tables(body: str) -> list[tuple[Literal["prose", "table"], str]]:
    """Separate Markdown pipe tables from prose blocks within a section."""
    lines = body.splitlines()
    blocks: list[tuple[Literal["prose", "table"], str]] = []
    prose_buf: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEP_RE.match(lines[i + 1]):
            if prose_buf:
                prose_text = "\n".join(prose_buf).strip()
                if prose_text:
                    blocks.append(("prose", prose_text))
                prose_buf = []
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
        else:
            prose_buf.append(lines[i])
            i += 1
    if prose_buf:
        prose_text = "\n".join(prose_buf).strip()
        if prose_text:
            blocks.append(("prose", prose_text))
    return blocks


def chunk_markdown_table_aware(
    doc: Document,
    max_words: int = 200,
    overlap_words: int = 40,
    max_table_rows: int = 20,
) -> list[Chunk]:
    """Chunk Markdown while keeping pipe tables structured and repeating table headers on splits."""
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    if max_table_rows < 1:
        raise ValueError("max_table_rows must be >= 1")

    chunks: list[Chunk] = []
    for heading, body in _sections(doc.text):
        for kind, block in _split_prose_and_tables(body):
            if kind == "prose":
                sub_doc = Document(doc.doc_id, block, doc.metadata)
                for sub_chunk in chunk_document(sub_doc, max_words=max_words, overlap_words=overlap_words):
                    chunks.append(
                        _make_structured_chunk(
                            doc,
                            heading,
                            sub_chunk.text,
                            len(chunks),
                            {"content_type": "prose"},
                        )
                    )
            else:
                table_lines = block.splitlines()
                header_lines = table_lines[:2]
                data_lines = table_lines[2:]
                if not data_lines:
                    chunks.append(_make_structured_chunk(doc, heading, block, len(chunks), {"content_type": "table"}))
                else:
                    for start in range(0, len(data_lines), max_table_rows):
                        batch = data_lines[start : start + max_table_rows]
                        table_text = "\n".join([*header_lines, *batch])
                        chunks.append(
                            _make_structured_chunk(
                                doc,
                                heading,
                                table_text,
                                len(chunks),
                                {"content_type": "table", "row_range": f"{start + 1}-{start + len(batch)}"},
                            )
                        )
    return chunks


def ingest_document(
    doc: Document,
    max_words: int = 200,
    overlap_words: int = 40,
    max_table_rows: int = 20,
) -> list[Chunk]:
    """Parse and chunk a document according to its detected format (Markdown, HTML, CSV, text)."""
    fmt = detect_format(doc.doc_id, doc.text, doc.metadata.get("format"))
    if fmt == "csv":
        return chunk_csv(doc, max_rows_per_chunk=max_table_rows)
    if fmt == "html":
        md = html_to_markdown(doc.text)
        md_doc = Document(doc.doc_id, md, {**doc.metadata, "format": "html"})
        return chunk_markdown_table_aware(md_doc, max_words=max_words, overlap_words=overlap_words, max_table_rows=max_table_rows)
    if fmt == "markdown" and _TABLE_SEP_RE.search(doc.text):
        return chunk_markdown_table_aware(doc, max_words=max_words, overlap_words=overlap_words, max_table_rows=max_table_rows)
    return chunk_document(doc, max_words=max_words, overlap_words=overlap_words)


def load_documents_from_directory(
    directory: str | Path,
    patterns: tuple[str, ...] = ("*.md", "*.txt", "*.html", "*.htm", "*.csv"),
) -> list[Document]:
    """Load all supported documents under ``directory`` with relative paths as doc_ids."""
    root = Path(directory)
    seen: set[Path] = set()
    docs: list[Document] = []
    for pat in patterns:
        for path in sorted(root.rglob(pat)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            rel = str(path.relative_to(root)).replace("\\", "/")
            fmt = detect_format(rel, "")
            docs.append(Document(rel, path.read_text(encoding="utf-8", errors="ignore"), {"format": fmt}))
    return docs
