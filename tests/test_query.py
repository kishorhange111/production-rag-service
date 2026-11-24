"""Unit tests for query rewriting, HyDE, sub-query decomposition, and routing."""

from __future__ import annotations

from ragservice.chunking import Document
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline
from ragservice.query import QueryProcessor, decompose_query, generate_hyde, rewrite_query, route_query


def test_rewrite_query_strips_filler_and_expands_acronyms():
    out = rewrite_query("Can you please explain how k8s db auth works?")
    assert not out.lower().startswith("can you")
    assert "kubernetes" in out and "database" in out and "authentication" in out


def test_route_query_classifies_intents():
    assert route_query("What causes ERR_SSL_PROTOCOL?") == ("exact_lookup", "bm25")
    assert route_query("Compare Postgres vs Redis") == ("comparison", "hybrid")
    assert route_query("How to deploy a canary release?") == ("procedural", "hybrid")
    assert route_query("List all production clusters") == ("aggregate", "hybrid")
    assert route_query("What is the backup retention window?") == ("factual", "hybrid")


def test_decompose_query_handles_comparison_and_compound():
    assert decompose_query("Compare backup retention and on-call rotation") == [
        "backup retention",
        "on-call rotation",
    ]
    assert decompose_query("Postgres vs PgBouncer") == ["Postgres", "PgBouncer"]
    assert decompose_query("What is the backup port; how long are logs kept?") == [
        "What is the backup port",
        "how long are logs kept",
    ]
    assert decompose_query("Single simple question?") == ["Single simple question"]


def test_generate_hyde_offline_and_with_llm():
    offline = generate_hyde("How long are database backups kept?")
    assert "database backups kept" in offline.lower()

    class _StubLLM:
        def complete(self, messages, max_tokens=512):
            return "Backups are retained for 35 days in encrypted object storage."

        def stream(self, messages, max_tokens=512):
            yield self.complete(messages, max_tokens)

    online = generate_hyde("How long are backups kept?", llm=_StubLLM())
    assert "35 days" in online


def test_multi_query_search_fuses_evidence_from_multiple_docs():
    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3)
    pipe.ingest(
        [
            Document("backup.md", "# Backups\n\nBackups are kept for 35 days."),
            Document("oncall.md", "# On-call\n\nOn-call shifts rotate every Monday."),
            Document("unrelated.md", "# Cafeteria\n\nLunch is served at noon."),
        ]
    )
    qp = QueryProcessor(enable_hyde=True)
    hits, plan = qp.multi_query_search(pipe.retriever, "Compare backup retention and on-call rotation", k=2)
    assert plan.route == "comparison" and len(plan.sub_queries) == 2 and plan.hyde_document is not None
    assert {h.chunk.doc_id for h in hits} == {"backup.md", "oncall.md"}
