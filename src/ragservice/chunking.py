"""Split documents into retrieval-sized chunks.

Chunking is the most under-rated RAG hyper-parameter:

* Too large: one chunk covers several topics. Its embedding is a blurry
  average, and it wastes the LLM's context on irrelevant text.
* Too small: the answer gets split across chunks, and each chunk lacks the
  context (e.g. which product a sentence talks about) needed to match a query.

The strategy here is *structure-aware*:

1. Split on Markdown headings first, and keep the heading path
   ("Install > Linux") as a prefix of every chunk, so each chunk carries its context.
2. Within a section, pack whole paragraphs, then sentences, up to ``max_words``.
3. Overlap consecutive chunks by ``overlap_words``, so a fact that straddles a
   boundary appears whole in at least one chunk.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    """A source document. ``doc_id`` should be stable (path or URL) so re-ingesting updates in place."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text, plus where it came from (for citations)."""

    chunk_id: str
    doc_id: str
    text: str
    heading: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def text_for_index(self) -> str:
        """What gets embedded and BM25-indexed: the heading path + body, so context is searchable."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading path, body) pairs. Text before any heading gets an empty path."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    last_end, last_heading = 0, ""
    for m in _HEADING.finditer(text):
        body = text[last_end : m.start()].strip()
        if body:
            sections.append((last_heading, body))
        level, title = len(m.group(1)), m.group(2).strip()
        # A level-2 heading replaces any previous level-2+ headings in the path.
        path = path[: level - 1] + [title]
        last_heading = " > ".join(path)
        last_end = m.end()
    tail = text[last_end:].strip()
    if tail:
        sections.append((last_heading, tail))
    return sections


def _units(body: str, max_words: int) -> list[str]:
    """Paragraphs, but break a paragraph longer than ``max_words`` into sentences (and long sentences into word runs)."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        for sent in _SENTENCE_END.split(para):
            words = sent.split()
            for i in range(0, len(words), max_words):
                units.append(" ".join(words[i : i + max_words]))
    return units


def chunk_document(doc: Document, max_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    """Split one document into chunks (see module docstring for the strategy).

    Args:
        doc: The document to split.
        max_words: Target chunk size. 200 words is about 260 tokens, a common sweet spot.
        overlap_words: Words repeated from the end of the previous chunk.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    chunks: list[Chunk] = []
    for heading, body in _sections(doc.text):
        current: list[str] = []
        # Units are at most (max_words - overlap_words) long, so carried overlap + one unit never exceeds max_words.
        for unit in _units(body, max_words - overlap_words):
            words = unit.split()
            if current and len(current) + len(words) > max_words:
                chunks.append(_make_chunk(doc, heading, current, len(chunks)))
                current = current[-overlap_words:] if overlap_words else []  # carry the overlap forward
            current.extend(words)
        if current:
            chunks.append(_make_chunk(doc, heading, current, len(chunks)))
    return chunks


def _make_chunk(doc: Document, heading: str, words: list[str], index: int) -> Chunk:
    text = " ".join(words)
    # Content hash in the id: an unchanged chunk keeps its id across re-ingestion, so caches stay valid.
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    return Chunk(
        chunk_id=f"{doc.doc_id}#{index}-{digest}",
        doc_id=doc.doc_id,
        text=text,
        heading=heading,
        metadata=dict(doc.metadata),
    )
