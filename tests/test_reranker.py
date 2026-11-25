"""Unit tests for reranker interface, lexical-semantic reranker, and score calibration."""

from __future__ import annotations

import pytest

from ragservice.chunking import Chunk, Document
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline
from ragservice.reranker import (
    CalibratedReranker,
    LexicalSemanticReranker,
    minmax_calibrate,
    sigmoid_calibrate,
)
from ragservice.retrieval import Hit


def test_sigmoid_and_minmax_calibration():
    assert sigmoid_calibrate(0.0) == pytest.approx(0.5)
    assert sigmoid_calibrate(1000.0) == pytest.approx(1.0)
    assert sigmoid_calibrate(-1000.0) == pytest.approx(0.0)
    assert sigmoid_calibrate(2.0, temperature=2.0) < sigmoid_calibrate(2.0, temperature=0.5)
    with pytest.raises(ValueError):
        sigmoid_calibrate(1.0, temperature=0.0)

    assert minmax_calibrate([]) == []
    assert minmax_calibrate([3.0, 3.0]) == [1.0, 1.0]
    assert minmax_calibrate([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]


def test_lexical_semantic_reranker_promotes_exact_phrase_and_heading_match():
    emb = HashingEmbedder()
    reranker = LexicalSemanticReranker(embedder=emb)
    hits = [
        Hit(Chunk("c1", "d1", "General networking overview and routing.", heading="Network"), 0.9),
        Hit(Chunk("c2", "d2", "PgBouncer connection pooling listens on port 6432.", heading="Database > PgBouncer"), 0.4),
    ]
    out = reranker.rerank("PgBouncer connection pooling port", hits)
    assert out[0].chunk.chunk_id == "c2"
    assert 0.0 <= out[0].score <= 1.0


class _StubLogitReranker:
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        logits = [-4.0, 3.5]
        return [Hit(h.chunk, logit) for h, logit in zip(hits, logits)]


def test_calibrated_reranker_maps_logits_and_filters_low_scores():
    hits = [Hit(Chunk("low", "d1", "unrelated"), 0.1), Hit(Chunk("high", "d2", "relevant"), 0.2)]
    cal = CalibratedReranker(_StubLogitReranker(), mode="sigmoid", min_score=0.5)
    out = cal.rerank("q", hits)
    assert len(out) == 1 and out[0].chunk.chunk_id == "high" and out[0].score > 0.95


def test_pipeline_with_lexical_semantic_reranker():
    emb = HashingEmbedder()
    pipe = RAGPipeline(emb, ExtractiveLLM(), top_k=2, reranker=LexicalSemanticReranker(embedder=emb))
    pipe.ingest(
        [
            Document("a.md", "# Deploy\n\nCanary rollouts take 15 minutes."),
            Document("b.md", "# Backups\n\nBackups are encrypted with AES-256."),
        ]
    )
    ans = pipe.answer("How are backups encrypted?")
    assert "AES-256" in ans.answer and not ans.abstained
