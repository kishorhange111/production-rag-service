"""Unit tests for incremental CDC deduplication and re-indexing queue."""

from __future__ import annotations

from ragservice.cdc import CDCReindexQueue, compute_content_hash
from ragservice.chunking import Document
from ragservice.embeddings import HashingEmbedder
from ragservice.index import KnowledgeBase


def test_content_hash_sensitive_to_text_and_metadata():
    d1 = Document("a.md", "hello", {"tenant": "acme"})
    d2 = Document("a.md", "hello", {"tenant": "acme"})
    d3 = Document("a.md", "hello!", {"tenant": "acme"})
    d4 = Document("a.md", "hello", {"tenant": "beta"})
    assert compute_content_hash(d1) == compute_content_hash(d2)
    assert compute_content_hash(d1) != compute_content_hash(d3)
    assert compute_content_hash(d1) != compute_content_hash(d4)


def test_cdc_skips_unchanged_and_replaces_modified_docs():
    kb = KnowledgeBase(HashingEmbedder())
    q = CDCReindexQueue()

    doc_v1 = Document("db.md", "# DB\n\nPostgres runs on port 5432.")
    q.enqueue_upsert(doc_v1)
    s1 = q.process_batch(kb)
    assert s1.upserted == 1 and s1.chunks_added == 1 and s1.skipped_unchanged == 0

    # Same content again -> skipped via content hash
    q.enqueue_upsert(Document("db.md", "# DB\n\nPostgres runs on port 5432."))
    s2 = q.process_batch(kb)
    assert s2.skipped_unchanged == 1 and s2.upserted == 0

    # Updated content -> old chunk removed, new chunk indexed
    q.enqueue_upsert(Document("db.md", "# DB\n\nPgBouncer listens on port 6432."))
    s3 = q.process_batch(kb)
    assert s3.upserted == 1 and s3.chunks_removed == 1 and s3.chunks_added == 1
    assert kb.bm25.search("5432", 5) == []
    assert len(kb.bm25.search("6432", 5)) == 1


def test_cdc_delete_removes_chunks_and_keeps_indexes_aligned():
    kb = KnowledgeBase(HashingEmbedder())
    q = CDCReindexQueue()
    q.enqueue_upsert(Document("a.md", "# A\n\nAlpha service port 8001."))
    q.enqueue_upsert(Document("b.md", "# B\n\nBeta service port 8002."))
    q.process_batch(kb)
    assert len(kb.chunks) == 2 and kb.dense.vectors.shape[0] == 2

    q.enqueue_delete("a.md")
    s = q.process_batch(kb)
    assert s.deleted == 1 and s.chunks_removed == 1
    assert len(kb.chunks) == 1 and kb.chunks[0].doc_id == "b.md"
    assert kb.dense.vectors.shape[0] == 1
    assert kb.bm25.search("alpha", 5) == []
    assert kb.bm25.search("beta", 5)[0][0] == 0


def test_cdc_dead_letters_failing_events_and_respects_max_events():
    kb = KnowledgeBase(HashingEmbedder())
    q = CDCReindexQueue()
    q.enqueue_upsert(Document("bad.md", "fail"))
    q.enqueue_upsert(Document("good.md", "hello world"))
    q.enqueue_upsert(Document("later.md", "queued for next batch"))

    def flaky_chunker(doc: Document):
        if doc.doc_id == "bad.md":
            raise RuntimeError("corrupt payload")
        from ragservice.ingestion import ingest_document

        return ingest_document(doc)

    s = q.process_batch(kb, chunker=flaky_chunker, max_events=2)
    assert s.processed == 2 and s.failed == 1 and s.upserted == 1
    assert len(q.dead_letters) == 1 and q.dead_letters[0][0].doc_id == "bad.md"
    assert q.pending_count == 1
