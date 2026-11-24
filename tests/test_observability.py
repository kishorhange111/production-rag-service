"""Unit tests for OpenTelemetry-compatible tracing, Prometheus metrics, and Grafana dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragservice.api import create_app
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.observability import MetricsRegistry, RAGTracer
from ragservice.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_tracer_records_nested_spans_and_errors():
    tracer = RAGTracer()
    with tracer.span("parent", {"tenant": "acme"}) as parent, tracer.span("child") as child:
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id

    with pytest.raises(RuntimeError), tracer.span("failing"):
        raise RuntimeError("boom")

    spans = tracer.recorded_spans
    assert len(spans) == 3
    assert spans[0].name == "child" and spans[0].duration_ms >= 0
    assert spans[2].status == "ERROR" and spans[2].error == "boom"


def test_metrics_registry_and_api_observability(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    app = create_app(RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3))
    client = TestClient(app)

    client.post("/ingest", json={"documents": [{"doc_id": "b.md", "text": "# Backups\n\nBackups run at 02:00 UTC."}]})
    r = client.post("/query", json={"question": "When do backups run?"})
    assert r.status_code == 200 and "X-Trace-ID" in r.headers

    metrics_text = client.get("/metrics").text
    assert "rag_indexed_chunks 1" in metrics_text
    assert 'rag_stage_duration_seconds_count{stage="retrieval"} 1' in metrics_text
    assert "rag_abstentions_total 0" in metrics_text
    assert len(app.state.tracer.recorded_spans) >= 2

    reg = MetricsRegistry()
    reg.inc_cache("hit")
    assert 'rag_cache_events_total{outcome="hit"} 1' in reg.render_prometheus()


def test_grafana_dashboard_json_is_valid():
    path = ROOT / "observability" / "grafana" / "rag-service-dashboard.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["uid"] == "production-rag-service"
    assert len(data["panels"]) >= 5
    exprs = " ".join(t["expr"] for p in data["panels"] for t in p.get("targets", []))
    assert "rag_requests_total" in exprs
    assert "rag_request_seconds_bucket" in exprs
    assert "rag_stage_duration_seconds_sum" in exprs
