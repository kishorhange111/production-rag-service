"""API tests with FastAPI's TestClient (no server, no network, no model)."""

import pytest
from fastapi.testclient import TestClient

from ragservice.api import create_app
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    app = create_app(RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3))
    return TestClient(app)


DOCS = {
    "documents": [
        {
            "doc_id": "backup.md",
            "text": "# Backups\n\nFull backups run nightly at 02:00 UTC and are kept for 35 days.",
        },
        {
            "doc_id": "oncall.md",
            "text": "# On-call\n\nShifts last one week, Monday to Monday.",
        },
    ]
}


def test_health_ingest_query_flow(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}
    r = client.post("/ingest", json=DOCS)
    assert r.status_code == 200 and r.json()["added_chunks"] == 2
    assert client.post("/ingest", json=DOCS).json()["added_chunks"] == 0  # idempotent

    r = client.post("/query", json={"question": "How long are backups kept?"})
    body = r.json()
    assert r.status_code == 200 and "35 days" in body["answer"] and not body["abstained"]
    assert body["cited"] and body["sources"][body["cited"][0] - 1]["doc_id"] == "backup.md"
    assert "X-Request-ID" in r.headers


def test_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_validation_errors_are_422(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/ingest", json={"documents": []}).status_code == 422


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "s3cret")
    assert client.post("/query", json={"question": "x"}).status_code == 401
    assert client.post("/query", json={"question": "x"}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for load balancers


def test_streaming_sends_sources_then_tokens_then_done(client):
    client.post("/ingest", json=DOCS)
    with client.stream("POST", "/query/stream", json={"question": "How long are on-call shifts?"}) as r:
        events = [line.split(": ", 1)[1] for line in r.iter_lines() if line.startswith("event:")]
        assert r.status_code == 200
    assert events[0] == "sources" and events[-1] == "done" and "token" in events


def test_metrics_are_prometheus_text(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert 'rag_requests_total{path="/health",status="200"} 1' in body
    assert 'rag_request_seconds_bucket{le="+Inf"}' in body and "rag_request_seconds_count" in body
