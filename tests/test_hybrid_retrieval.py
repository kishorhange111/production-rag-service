"""Unit tests for the hybrid retrieval engine: weighted RRF, linear fusion, diversity, diagnostics."""

from __future__ import annotations

import pytest

from ragservice.chunking import Chunk
from ragservice.embeddings import HashingEmbedder
from ragservice.index import KnowledgeBase
from ragservice.retrieval import Retriever, linear_score_fusion, reciprocal_rank_fusion


def test_weighted_rrf_respects_retriever_weights():
    # Doc 1 is top in list A, doc 2 is top in list B
    fused_a = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60, weights=[3.0, 1.0])
    fused_b = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60, weights=[1.0, 3.0])
    assert fused_a[0][0] == 1
    assert fused_b[0][0] == 2


def test_weighted_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])


def test_linear_score_fusion_blends_normalized_scores():
    bm25 = [(1, 10.0), (2, 2.0)]
    dense = [(2, 0.95), (1, 0.10)]
    pure_bm25 = linear_score_fusion(bm25, dense, alpha=0.0)
    pure_dense = linear_score_fusion(bm25, dense, alpha=1.0)
    assert pure_bm25[0][0] == 1
    assert pure_dense[0][0] == 2

    with pytest.raises(ValueError, match="alpha"):
        linear_score_fusion(bm25, dense, alpha=1.5)


def test_retriever_doc_diversity_and_min_score_and_diagnostics():
    kb = KnowledgeBase(HashingEmbedder())
    kb.add(
        [
            Chunk("docA#0", "docA", "postgres database connection pool port 6432"),
            Chunk("docA#1", "docA", "postgres database failover replica port 6432"),
            Chunk("docA#2", "docA", "postgres database backup schedule port 6432"),
            Chunk("docB#0", "docB", "postgres database monitoring alerts"),
        ]
    )
    ret = Retriever(kb, mode="hybrid", fusion="linear", alpha=0.5, max_per_doc=1)
    hits, diag = ret.search_with_diagnostics("postgres database", k=3)
    assert len(hits) == 2
    assert {h.chunk.doc_id for h in hits} == {"docA", "docB"}
    assert hits[0].bm25_score is not None and hits[0].dense_score is not None
    assert diag.fusion == "linear" and diag.bm25_candidates == 4 and diag.overlap_ratio == 1.0

    strict = Retriever(kb, mode="hybrid", fusion="linear", min_score=0.99)
    strict_hits = strict.search("postgres database", k=5)
    assert all(h.score >= 0.99 for h in strict_hits)
