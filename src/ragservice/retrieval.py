"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, optionally reranked.

**Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009): each retriever votes
for a document with ``1 / (k + rank)``, and the votes are summed. It uses only
*ranks*, so it needs no calibration between BM25 scores (unbounded) and cosine
similarities (in [-1, 1]), and it is robust: a document ranked highly by
either retriever surfaces.

**Reranking**: a cross-encoder reads (query, passage) *together* and scores
relevance far more accurately than comparing two independent embeddings. It is
too slow to run over the whole corpus, so it reorders only the fused top-N.
Whether it helps must be measured (it hurt in the retrieval study in
generative-ai-engineering), so it is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ragservice.chunking import Chunk
from ragservice.index import KnowledgeBase

Mode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its score and the per-retriever ranks (for debugging and evaluation)."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked lists of ids into one list sorted by RRF score.

    ``k = 60`` is the constant from the original paper. Larger k flattens the
    advantage of the top ranks.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class CrossEncoderReranker:
    """Rerank with a ``sentence-transformers`` CrossEncoder, e.g. ``BAAI/bge-reranker-base``."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text_for_index) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        return [Hit(hits[i].chunk, float(scores[i]), hits[i].bm25_rank, hits[i].dense_rank) for i in order]


class Retriever:
    """Query the knowledge base with BM25, dense or hybrid retrieval.

    Args:
        kb: The knowledge base to search.
        mode: ``bm25``, ``dense`` or ``hybrid`` (RRF of both).
        candidates: How many results each retriever contributes before fusion.
        reranker: Optional cross-encoder applied to the fused candidates.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        mode: Mode = "hybrid",
        candidates: int = 50,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.kb, self.mode, self.candidates, self.reranker = (
            kb,
            mode,
            candidates,
            reranker,
        )

    def search(self, query: str, k: int = 5) -> list[Hit]:
        bm25 = self.kb.bm25.search(query, self.candidates) if self.mode in ("bm25", "hybrid") else []
        dense = self.kb.dense.search(self.kb.embedder.embed_query(query), self.candidates) if self.mode in ("dense", "hybrid") else []
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25)}
        dense_rank = {i: r for r, (i, _) in enumerate(dense)}

        if self.mode == "bm25":
            fused = bm25
        elif self.mode == "dense":
            fused = dense
        else:
            fused = reciprocal_rank_fusion([[i for i, _ in bm25], [i for i, _ in dense]])

        hits = [Hit(self.kb.chunks[i], s, bm25_rank.get(i), dense_rank.get(i)) for i, s in fused]
        if self.reranker is not None:
            # Rerank a bit more than k: reordering only the top-k could never promote rank k+1.
            hits = self.reranker.rerank(query, hits[: max(k * 4, 20)])
        return hits[:k]
