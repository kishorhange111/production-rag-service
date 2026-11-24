"""Reranker interface, CPU-friendly lexical-semantic reranker, and score calibration.

Cross-encoders output unbounded logits (e.g. ``[-10, +10]``) whose scale varies by
model checkpoint. Raw logits cannot be safely compared against a fixed abstention
threshold across models. This module provides:

* ``Reranker`` protocol shared by neural cross-encoders and lightweight rerankers.
* ``sigmoid_calibrate`` (temperature-scaled Platt/logistic mapping to ``(0, 1)``) and
  ``minmax_calibrate`` for list-wise normalization.
* ``LexicalSemanticReranker``: a deterministic, zero-download reranker combining
  query-token coverage, bigram/phrase match, heading alignment, and optional dense
  cosine similarity.
* ``CalibratedReranker``: decorator that calibrates any reranker's scores and
  optionally prunes candidates below a confidence floor.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol

from ragservice.embeddings import Embedder
from ragservice.index import tokenize
from ragservice.retrieval import Hit

CalibrationMode = Literal["sigmoid", "minmax"]


class Reranker(Protocol):
    """Protocol implemented by all rerankers."""

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


def sigmoid_calibrate(score: float, temperature: float = 1.0, bias: float = 0.0) -> float:
    """Map an unbounded logit to ``(0, 1)`` via ``1 / (1 + exp(-(score + bias) / temperature))``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = (score + bias) / temperature
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def minmax_calibrate(scores: list[float]) -> list[float]:
    """Normalize a list of scores into ``[0, 1]``."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


class LexicalSemanticReranker:
    """Fast CPU reranker blending token coverage, phrase bigrams, heading match, and dense similarity."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        w_coverage: float = 0.45,
        w_bigram: float = 0.20,
        w_heading: float = 0.15,
        w_semantic: float = 0.20,
    ) -> None:
        self.embedder = embedder
        self.w_coverage = w_coverage
        self.w_bigram = w_bigram
        self.w_heading = w_heading
        self.w_semantic = w_semantic

    def _score_one(self, q_tokens: list[str], q_set: set[str], q_bi: set[tuple[str, str]], hit: Hit, sem: float) -> float:
        doc_tokens = tokenize(hit.chunk.text_for_index)
        doc_set = set(doc_tokens)
        coverage = len(q_set & doc_set) / len(q_set) if q_set else 0.0
        bigram_overlap = len(q_bi & _bigrams(doc_tokens)) / len(q_bi) if q_bi else coverage
        heading_tokens = set(tokenize(hit.chunk.heading))
        heading_overlap = len(q_set & heading_tokens) / len(q_set) if q_set and heading_tokens else 0.0
        raw = self.w_coverage * coverage + self.w_bigram * bigram_overlap + self.w_heading * heading_overlap + self.w_semantic * sem
        return max(0.0, min(1.0, raw))

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        q_bi = _bigrams(q_tokens)

        if self.embedder is not None:
            q_vec = self.embedder.embed_query(query)
            d_vecs = self.embedder.embed_documents([h.chunk.text_for_index for h in hits])
            sem_scores = [max(0.0, float(v @ q_vec)) for v in d_vecs]
        else:
            sem_scores = [max(0.0, min(1.0, h.dense_score if h.dense_score is not None else 0.0)) for h in hits]

        scored = [
            Hit(
                h.chunk,
                self._score_one(q_tokens, q_set, q_bi, h, sem),
                h.bm25_rank,
                h.dense_rank,
                h.bm25_score,
                h.dense_score,
            )
            for h, sem in zip(hits, sem_scores)
        ]
        return sorted(scored, key=lambda h: -h.score)


class CalibratedReranker:
    """Wrap any ``Reranker`` to calibrate scores into ``[0, 1]`` and optionally filter low-confidence hits."""

    def __init__(
        self,
        inner: Reranker,
        mode: CalibrationMode = "sigmoid",
        temperature: float = 1.0,
        bias: float = 0.0,
        min_score: float | None = None,
    ) -> None:
        self.inner = inner
        self.mode = mode
        self.temperature = temperature
        self.bias = bias
        self.min_score = min_score

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        reranked = self.inner.rerank(query, hits)
        if not reranked:
            return []
        if self.mode == "sigmoid":
            cal_scores = [sigmoid_calibrate(h.score, self.temperature, self.bias) for h in reranked]
        else:
            cal_scores = minmax_calibrate([h.score for h in reranked])
        out = [
            Hit(h.chunk, cs, h.bm25_rank, h.dense_rank, h.bm25_score, h.dense_score)
            for h, cs in zip(reranked, cal_scores)
            if self.min_score is None or cs >= self.min_score
        ]
        return sorted(out, key=lambda h: -h.score)
