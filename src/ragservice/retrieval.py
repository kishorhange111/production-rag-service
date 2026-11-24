"""Hybrid retrieval engine: BM25 + dense vector search with RRF and convex score fusion.

**Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009): each retriever votes
for a document with ``weight / (k + rank)``, and the votes are summed. It uses only
*ranks*, so it needs no calibration between BM25 scores (unbounded) and cosine
similarities (in [-1, 1]), and it is robust: a document ranked highly by
either retriever surfaces.

**Linear (convex) score fusion**: min-max normalizes BM25 and dense scores into
``[0, 1]`` and blends them via ``alpha * dense + (1 - alpha) * bm25``. Useful when
score magnitudes carry strong confidence signal.

**Diversity capping** (``max_per_doc``): prevents a single long document from
monopolizing all top-k slots.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from ragservice.chunking import Chunk
from ragservice.index import KnowledgeBase

Mode = Literal["bm25", "dense", "hybrid"]
FusionStrategy = Literal["rrf", "linear"]


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its fused score and per-retriever ranks/scores."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Observability metadata for a single retrieval call."""

    mode: Mode
    fusion: FusionStrategy
    bm25_candidates: int
    dense_candidates: int
    overlap_count: int
    overlap_ratio: float
    returned: int


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse several ranked lists of ids into one list sorted by weighted RRF score.

    ``k = 60`` is the constant from the original paper. Larger k flattens the
    advantage of the top ranks.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")
    w_list = weights if weights is not None else [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, w_list):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _min_max_normalize(results: list[tuple[int, float]]) -> dict[int, float]:
    if not results:
        return {}
    vals = [s for _, s in results]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {doc: 1.0 for doc, _ in results}
    return {doc: (s - lo) / (hi - lo) for doc, s in results}


def linear_score_fusion(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Fuse min-max normalized BM25 and dense scores: ``alpha * dense + (1 - alpha) * bm25``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    bm25_norm = _min_max_normalize(bm25_results)
    dense_norm = _min_max_normalize(dense_results)
    all_ids = set(bm25_norm) | set(dense_norm)
    fused = {doc: alpha * dense_norm.get(doc, 0.0) + (1.0 - alpha) * bm25_norm.get(doc, 0.0) for doc in all_ids}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


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
        return [
            Hit(
                hits[i].chunk,
                float(scores[i]),
                hits[i].bm25_rank,
                hits[i].dense_rank,
                hits[i].bm25_score,
                hits[i].dense_score,
            )
            for i in order
        ]


def _apply_doc_diversity(hits: list[Hit], max_per_doc: int) -> list[Hit]:
    counts: Counter[str] = Counter()
    kept: list[Hit] = []
    for h in hits:
        if counts[h.chunk.doc_id] < max_per_doc:
            kept.append(h)
            counts[h.chunk.doc_id] += 1
    return kept


class Retriever:
    """Query the knowledge base with BM25, dense, or hybrid (RRF / linear) retrieval."""

    def __init__(
        self,
        kb: KnowledgeBase,
        mode: Mode = "hybrid",
        candidates: int = 50,
        reranker: Any | None = None,
        fusion: FusionStrategy = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        alpha: float = 0.5,
        min_score: float | None = None,
        max_per_doc: int | None = None,
    ):
        self.kb = kb
        self.mode = mode
        self.candidates = candidates
        self.reranker = reranker
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.alpha = alpha
        self.min_score = min_score
        self.max_per_doc = max_per_doc

    def search_with_diagnostics(self, query: str, k: int = 5) -> tuple[list[Hit], RetrievalDiagnostics]:
        bm25 = self.kb.bm25.search(query, self.candidates) if self.mode in ("bm25", "hybrid") else []
        dense = self.kb.dense.search(self.kb.embedder.embed_query(query), self.candidates) if self.mode in ("dense", "hybrid") else []
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25)}
        dense_rank = {i: r for r, (i, _) in enumerate(dense)}
        bm25_scores = dict(bm25)
        dense_scores = dict(dense)

        if self.mode == "bm25":
            fused = bm25
        elif self.mode == "dense":
            fused = dense
        elif self.fusion == "linear":
            fused = linear_score_fusion(bm25, dense, alpha=self.alpha)
        else:
            fused = reciprocal_rank_fusion(
                [[i for i, _ in bm25], [i for i, _ in dense]],
                k=self.rrf_k,
                weights=[self.bm25_weight, self.dense_weight],
            )

        hits = [
            Hit(
                self.kb.chunks[i],
                s,
                bm25_rank.get(i),
                dense_rank.get(i),
                bm25_scores.get(i),
                dense_scores.get(i),
            )
            for i, s in fused
        ]
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits[: max(k * 4, 20)])
        if self.min_score is not None:
            hits = [h for h in hits if h.score >= self.min_score]
        if self.max_per_doc is not None:
            hits = _apply_doc_diversity(hits, self.max_per_doc)
        hits = hits[:k]

        bm25_set, dense_set = set(bm25_rank), set(dense_rank)
        union_len = len(bm25_set | dense_set)
        overlap = len(bm25_set & dense_set)
        diag = RetrievalDiagnostics(
            mode=self.mode,
            fusion=self.fusion,
            bm25_candidates=len(bm25),
            dense_candidates=len(dense),
            overlap_count=overlap,
            overlap_ratio=overlap / union_len if union_len else 0.0,
            returned=len(hits),
        )
        return hits, diag

    def search(self, query: str, k: int = 5) -> list[Hit]:
        hits, _ = self.search_with_diagnostics(query, k=k)
        return hits
