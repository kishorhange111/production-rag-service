"""Unit tests for semantic query cache with TTL, LRU eviction, and tenant isolation."""

from __future__ import annotations

from ragservice.acl import AccessContext
from ragservice.cache import SemanticQueryCache
from ragservice.chunking import Document
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline


def test_semantic_cache_exact_and_near_duplicate_hits():
    emb = HashingEmbedder()
    cache = SemanticQueryCache(emb, similarity_threshold=0.80, ttl_seconds=60.0)
    cache.put("How long are database backups kept?", "35 days")

    assert cache.get("  how long are database backups kept? ") == "35 days"
    # Very close wording should hit semantically
    assert cache.get("How long are database backups kept") == "35 days"
    assert cache.get("Completely unrelated question about kubernetes pods") is None
    stats = cache.stats()
    assert stats.hits == 2 and stats.misses == 1 and stats.hit_rate > 0.6


def test_semantic_cache_strict_tenant_and_role_isolation():
    emb = HashingEmbedder()
    cache = SemanticQueryCache(emb, similarity_threshold=0.85, ttl_seconds=60.0)
    t1 = AccessContext("tenant-a", roles=frozenset({"eng"}))
    t2 = AccessContext("tenant-b", roles=frozenset({"eng"}))
    t1_admin = AccessContext("tenant-a", roles=frozenset({"admin"}))

    cache.put("What is the API key?", "SECRET-A", access=t1)
    assert cache.get("What is the API key?", access=t1) == "SECRET-A"
    assert cache.get("What is the API key?", access=t2) is None
    assert cache.get("What is the API key?", access=t1_admin) is None

    assert cache.invalidate_tenant("tenant-a") == 1
    assert cache.get("What is the API key?", access=t1) is None


def test_semantic_cache_ttl_and_lru_eviction():
    now = [100.0]
    cache = SemanticQueryCache(HashingEmbedder(), ttl_seconds=10.0, max_entries=2, clock=lambda: now[0])
    cache.put("q1", "a1")
    cache.put("q2", "a2")
    cache.put("q3", "a3")  # evicts q1
    assert cache.get("q1") is None
    assert cache.get("q2") == "a2"
    assert cache.stats().evictions == 1

    now[0] += 11.0  # expire all
    assert cache.get("q2") is None
    assert cache.stats().expirations >= 1


def test_pipeline_uses_cache_and_invalidates_on_ingest():
    emb = HashingEmbedder()
    cache = SemanticQueryCache(emb, ttl_seconds=60.0)
    pipe = RAGPipeline(emb, ExtractiveLLM(), top_k=2, cache=cache)
    pipe.ingest([Document("b.md", "# Backups\n\nBackups are kept for 35 days.")])

    a1 = pipe.answer("How long are backups kept?")
    a2 = pipe.answer("How long are backups kept?")
    assert a1 is a2 and cache.stats().hits == 1

    pipe.ingest([Document("c.md", "# New\n\nNew doc invalidates cache.")])
    assert cache.stats().size == 0
