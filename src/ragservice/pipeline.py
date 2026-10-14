"""The end-to-end RAG pipeline, shared by the API, the CLI and the evaluation runner.

Keeping one implementation means the system that gets evaluated is exactly the
system that gets served. There is no "eval-only" code path that can drift.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ragservice.chunking import Document, chunk_document
from ragservice.embeddings import Embedder
from ragservice.generation import (
    ABSTAIN,
    LLM,
    build_prompt,
    is_abstention,
    parse_citations,
)
from ragservice.guardrails import screen_hits, should_abstain
from ragservice.index import KnowledgeBase
from ragservice.retrieval import Hit, Mode, Retriever


@dataclass
class Answer:
    """Everything the caller needs: the answer, which sources back it, and timings."""

    question: str
    answer: str
    citations: list[Hit]  # only the sources the answer actually cites
    retrieved: list[Hit]  # everything shown to the LLM
    abstained: bool
    blocked_chunks: int = 0  # retrieved chunks removed by the injection screen
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """Ingest documents, then answer questions with citations.

    Args:
        embedder: Embedding backend for the dense index.
        llm: Chat model used to write answers.
        mode: Retrieval mode (``bm25`` | ``dense`` | ``hybrid``).
        top_k: Chunks shown to the LLM. More context helps recall but adds cost and distraction.
        chunk_words / overlap_words: Chunking parameters (see ``chunking.py``).
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: LLM,
        mode: Mode = "hybrid",
        top_k: int = 5,
        chunk_words: int = 200,
        overlap_words: int = 40,
        reranker=None,
    ):
        self.kb = KnowledgeBase(embedder)
        self.llm = llm
        self.top_k = top_k
        self.chunk_words, self.overlap_words = chunk_words, overlap_words
        self.retriever = Retriever(self.kb, mode=mode, reranker=reranker)

    # ------------------------------------------------------------- ingestion
    def ingest(self, docs: list[Document]) -> int:
        """Chunk and index documents; returns the number of *new* chunks."""
        chunks = [c for d in docs for c in chunk_document(d, self.chunk_words, self.overlap_words)]
        return self.kb.add(chunks)

    def ingest_directory(self, directory: str | Path, patterns: tuple[str, ...] = ("*.md", "*.txt")) -> int:
        """Ingest every Markdown/text file under ``directory`` (the path is the document id)."""
        root = Path(directory)
        docs = [Document(str(p.relative_to(root)), p.read_text(errors="ignore")) for pat in patterns for p in sorted(root.rglob(pat))]
        return self.ingest(docs)

    # --------------------------------------------------------------- answering
    def retrieve(self, question: str) -> tuple[list[Hit], int, float]:
        """Search + injection screen. Returns (kept hits, number dropped, elapsed ms)."""
        t = time.perf_counter()
        screened = screen_hits(self.retriever.search(question, self.top_k))
        return screened.kept, len(screened.dropped), (time.perf_counter() - t) * 1000

    def finalize(
        self,
        question: str,
        text: str,
        hits: list[Hit],
        dropped: int,
        timings: dict[str, float],
    ) -> Answer:
        """Turn raw LLM text into an :class:`Answer` (parse citations, detect abstention)."""
        cited = [hits[n - 1] for n in parse_citations(text, len(hits))]
        return Answer(question, text.strip(), cited, hits, is_abstention(text), dropped, timings)

    def answer(self, question: str) -> Answer:
        hits, dropped, t_ret = self.retrieve(question)
        if should_abstain(hits):
            return Answer(question, ABSTAIN, [], hits, True, dropped, {"retrieval": t_ret})
        t = time.perf_counter()
        text = self.llm.complete(build_prompt(question, hits))
        return self.finalize(
            question,
            text,
            hits,
            dropped,
            {"retrieval": t_ret, "generation": (time.perf_counter() - t) * 1000},
        )

    def answer_stream(self, question: str) -> tuple[list[Hit], Iterator[str]]:
        """Retrieve first (so sources can be sent to the client immediately), then stream the answer."""
        hits, _, _ = self.retrieve(question)
        if should_abstain(hits):
            return hits, iter([ABSTAIN])
        return hits, self.llm.stream(build_prompt(question, hits))
