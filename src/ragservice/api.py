"""HTTP API (FastAPI).

Endpoints:
    GET  /health         liveness/readiness, plus index size
    POST /ingest         add documents (JSON), idempotent by chunk content
    POST /query          answer with citations (JSON)
    POST /query/stream   Server-Sent Events: a ``sources`` event first, then ``token`` events, then ``done``
    GET  /metrics        request counts and latency percentiles (a stand-in for a Prometheus exporter)

Production touches:
    * a request id on every response (``X-Request-ID``), echoed from the client if provided, for log correlation
    * an optional API key (``RAG_API_KEY`` env var); if set, every call except /health must send ``X-API-Key``
    * request validation via Pydantic: bad input gets a 422 with details, never a stack trace
    * configuration only through environment variables (12-factor), see :func:`pipeline_from_env`
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragservice.chunking import Document
from ragservice.embeddings import build_embedder
from ragservice.generation import ExtractiveLLM, OpenAICompatibleLLM
from ragservice.pipeline import RAGPipeline


# ------------------------------------------------------------------ schemas
class DocumentIn(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=512)
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=1000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    n: int
    doc_id: str
    heading: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
    cited: list[int]
    timings_ms: dict[str, float]


# ------------------------------------------------------------------ factory
def pipeline_from_env() -> RAGPipeline:
    """Build the pipeline from environment variables (defaults run with no GPU or model download).

    RAG_EMBEDDER   sentence-transformers model name, or "hashing" (default)
    RAG_LLM_URL    OpenAI-compatible base URL; unset means use the extractive fallback
    RAG_LLM_MODEL  model name for that server
    RAG_MODE       bm25 | dense | hybrid (default)
    RAG_TOP_K      chunks per answer (default 5)
    RAG_DATA_DIR   folder of .md/.txt files ingested at start-up (optional)
    """
    llm_url = os.environ.get("RAG_LLM_URL")
    llm = OpenAICompatibleLLM(os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"), llm_url) if llm_url else ExtractiveLLM()
    pipe = RAGPipeline(
        build_embedder(os.environ.get("RAG_EMBEDDER", "hashing")),
        llm,
        mode=os.environ.get("RAG_MODE", "hybrid"),  # type: ignore[arg-type]
        top_k=int(os.environ.get("RAG_TOP_K", "5")),
    )
    if data_dir := os.environ.get("RAG_DATA_DIR"):
        pipe.ingest_directory(data_dir)
    return pipe


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """App factory: tests inject a pipeline; production builds one from the environment."""
    app = FastAPI(title="RAG service", version="0.1.0")
    app.state.pipeline = pipeline or pipeline_from_env()
    # Rolling window of recent latencies: bounded memory, recent behaviour.
    app.state.latencies = deque(maxlen=1000)
    app.state.requests = 0

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("RAG_API_KEY")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id and record latency for every call."""
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        app.state.requests += 1
        app.state.latencies.append((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = rid
        return response

    def to_sources(hits) -> list[Source]:
        return [Source(n=i, doc_id=h.chunk.doc_id, heading=h.chunk.heading, text=h.chunk.text) for i, h in enumerate(hits, start=1)]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(app.state.pipeline.kb.chunks)}

    @app.post("/ingest", dependencies=[Depends(require_key)])
    def ingest(req: IngestRequest) -> dict:
        added = app.state.pipeline.ingest([Document(d.doc_id, d.text, d.metadata) for d in req.documents])
        return {
            "added_chunks": added,
            "total_chunks": len(app.state.pipeline.kb.chunks),
        }

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_key)])
    def query(req: QueryRequest) -> QueryResponse:
        ans = app.state.pipeline.answer(req.question)
        cited_ids = {id(h) for h in ans.citations}
        return QueryResponse(
            answer=ans.answer,
            abstained=ans.abstained,
            sources=to_sources(ans.retrieved),
            cited=[i for i, h in enumerate(ans.retrieved, start=1) if id(h) in cited_ids],
            timings_ms=ans.timings_ms,
        )

    @app.post("/query/stream", dependencies=[Depends(require_key)])
    def query_stream(req: QueryRequest) -> StreamingResponse:
        hits, tokens = app.state.pipeline.answer_stream(req.question)

        def events():
            # Sources go first, so the UI can render them while the answer is still being written.
            yield f"event: sources\ndata: {json.dumps([s.model_dump() for s in to_sources(hits)])}\n\n"
            for tok in tokens:
                yield f"event: token\ndata: {json.dumps(tok)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/metrics")
    def metrics() -> dict:
        lat = sorted(app.state.latencies)

        def pct(p: float) -> float:
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2) if lat else 0.0

        return {
            "requests_total": app.state.requests,
            "latency_ms_p50": pct(0.5),
            "latency_ms_p95": pct(0.95),
            "latency_ms_p99": pct(0.99),
        }

    return app


# `uvicorn ragservice.api:app` entry point. It's created lazily, so importing this module in tests has no side effects.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
