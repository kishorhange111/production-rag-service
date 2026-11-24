"""Unit and API tests for typed schemas and streaming SSE generation."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ragservice.api import create_app
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline
from ragservice.schemas import SSEDonePayload, format_sse


def test_format_sse_with_id_and_model():
    frame = format_sse("done", SSEDonePayload(cited=[1], abstained=False, token_count=4, elapsed_ms=1.5), event_id="7")
    assert "id: 7\n" in frame
    assert "event: done\n" in frame
    assert '"token_count":4' in frame
    assert frame.endswith("\n\n")


def _parse_sse(lines: list[str]) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    cur_event = None
    for line in lines:
        if line.startswith("event: "):
            cur_event = line.split(": ", 1)[1]
        elif line.startswith("data: ") and cur_event is not None:
            events.append((cur_event, json.loads(line.split(": ", 1)[1])))
            cur_event = None
    return events


def test_streaming_sse_headers_grounding_and_done_metadata(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    app = create_app(RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3))
    client = TestClient(app)
    client.post(
        "/ingest",
        json={"documents": [{"doc_id": "b.md", "text": "# Backups\n\nBackups are kept for 35 days."}]},
    )

    with client.stream(
        "POST",
        "/query/stream",
        json={"question": "How long are backups kept?", "verify_grounding": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"
        parsed = _parse_sse(list(resp.iter_lines()))

    names = [name for name, _ in parsed]
    assert names[0] == "sources"
    assert "token" in names
    assert "grounding" in names
    assert names[-1] == "done"

    done_payload = parsed[-1][1]
    assert done_payload["cited"] == [1]
    assert done_payload["abstained"] is False
    assert done_payload["token_count"] > 0

    grounding_payload = dict(parsed)["grounding"]
    assert grounding_payload["is_grounded"] is True


def test_query_endpoint_grounding_and_schema_validation(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    app = create_app(RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3))
    client = TestClient(app)
    client.post(
        "/ingest",
        json={"documents": [{"doc_id": "b.md", "text": "# Backups\n\nBackups are kept for 35 days."}]},
    )
    res = client.post("/query", json={"question": "How long are backups kept?", "verify_grounding": True}).json()
    assert res["grounding"]["is_grounded"] is True
    assert res["grounding"]["faithfulness_score"] == 1.0
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422
