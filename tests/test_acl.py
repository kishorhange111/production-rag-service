"""Unit and API tests for multi-tenant ACL and metadata filter enforcement."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ragservice.acl import AccessContext, MetadataFilter, is_chunk_authorized
from ragservice.api import create_app
from ragservice.chunking import Chunk, Document
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline


def test_acl_tenant_roles_clearance_and_explicit_deny():
    chunk = Chunk(
        "c1",
        "d1",
        "secret",
        metadata={
            "tenant_id": "acme",
            "allowed_roles": ["finance", "admin"],
            "min_clearance": 2,
            "denied_principals": ["mallory"],
        },
    )
    assert is_chunk_authorized(chunk, AccessContext("acme", "alice", frozenset({"finance"}), clearance_level=2))
    assert not is_chunk_authorized(chunk, AccessContext("beta", "alice", frozenset({"finance"}), clearance_level=3))
    assert not is_chunk_authorized(chunk, AccessContext("acme", "bob", frozenset({"eng"}), clearance_level=2))
    assert not is_chunk_authorized(chunk, AccessContext("acme", "carol", frozenset({"finance"}), clearance_level=1))
    assert not is_chunk_authorized(chunk, AccessContext("acme", "mallory", frozenset({"admin"}), clearance_level=3))

    public_chunk = Chunk("pub", "p", "public info", metadata={"visibility": "public"})
    assert is_chunk_authorized(public_chunk, AccessContext("any-tenant"))
    untagged_chunk = Chunk("raw", "r", "untagged", metadata={})
    assert not is_chunk_authorized(untagged_chunk, AccessContext("any-tenant"))


def test_metadata_filter_equals_and_in_values():
    mf = MetadataFilter(equals={"region": "us-east"}, in_values={"env": frozenset({"prod", "stage"})})
    assert mf.matches({"region": "us-east", "env": "prod"})
    assert not mf.matches({"region": "eu-west", "env": "prod"})
    assert not mf.matches({"region": "us-east", "env": "dev"})


def test_pipeline_enforces_cross_tenant_isolation_and_filters():
    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3)
    pipe.ingest(
        [
            Document("acme_payroll.md", "# Payroll\n\nAcme payroll closes on the 25th.", {"tenant_id": "acme", "dept": "hr"}),
            Document("globex_payroll.md", "# Payroll\n\nGlobex payroll closes on the 28th.", {"tenant_id": "globex", "dept": "hr"}),
            Document("acme_eng.md", "# Deploy\n\nAcme deploys on Tuesdays.", {"tenant_id": "acme", "dept": "eng"}),
        ]
    )
    ans_acme = pipe.answer("When does payroll close?", access=AccessContext("acme"))
    assert "25th" in ans_acme.answer and "28th" not in ans_acme.answer
    assert all(h.chunk.metadata["tenant_id"] == "acme" for h in ans_acme.retrieved)

    ans_globex = pipe.answer("When does payroll close?", access=AccessContext("globex"))
    assert "28th" in ans_globex.answer and "25th" not in ans_globex.answer

    ans_filtered = pipe.answer(
        "When does payroll close?",
        access=AccessContext("acme"),
        metadata_filter=MetadataFilter(equals={"dept": "eng"}),
    )
    assert all(h.chunk.metadata["dept"] == "eng" for h in ans_filtered.retrieved)


def test_api_query_enforces_tenant_isolation(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    app = create_app(RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3))
    client = TestClient(app)
    client.post(
        "/ingest",
        json={
            "documents": [
                {"doc_id": "t1.md", "text": "# Secret\n\nTenant1 key is ALPHA-1.", "metadata": {"tenant_id": "t1"}},
                {"doc_id": "t2.md", "text": "# Secret\n\nTenant2 key is BETA-2.", "metadata": {"tenant_id": "t2"}},
            ]
        },
    )
    r1 = client.post("/query", json={"question": "What is the key?", "tenant_id": "t1"}).json()
    assert "ALPHA-1" in r1["answer"] and "BETA-2" not in r1["answer"]
    r3 = client.post("/query", json={"question": "What is the key?", "tenant_id": "t3"}).json()
    assert r3["abstained"] is True and r3["sources"] == []
