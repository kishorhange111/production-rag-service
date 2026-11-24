"""HTTP API (FastAPI).

Endpoints:
    GET  /health         liveness/readiness, plus index size
    POST /ingest         add documents (JSON), idempotent by chunk content
    POST /query          answer with citations (JSON)
    POST /query/stream   Server-Sent Events: a ``sources`` event first, then ``token`` events, then ``done``
    GET  /metrics        Prometheus text format: request counter and a latency histogram

Production touches:
    * a request id on every response (``X-Request-ID``), echoed from the client if provided, for log correlation
    * an optional API key (``RAG_API_KEY`` env var); if set, every call except /health must send ``X-API-Key``
    * request validation via Pydantic: bad input gets a 422 with details, never a stack trace
    * configuration only through environment variables (12-factor), see :func:`pipeline_from_env`

Known limitation: the index lives in process memory, so run ONE worker per
container (the Dockerfile does) and scale with replicas that each load the same
saved index. Documents sent to /ingest reach only the replica that served the call;
a shared vector store (pgvector, Qdrant) is the fix if ingestion must be live.
"""

from __future__ import annotations

import hmac
import os
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from ragservice.acl import AccessContext, MetadataFilter
from ragservice.chunking import Document
from ragservice.citations import verify_grounding
from ragservice.embeddings import build_embedder
from ragservice.generation import ExtractiveLLM, OpenAICompatibleLLM, is_abstention, parse_citations
from ragservice.pipeline import RAGPipeline
from ragservice.schemas import (
    DocumentIn,
    GroundingSummary,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Source,
    SSEDonePayload,
    format_sse,
)

# Histogram bucket upper bounds in seconds (Prometheus default-style spacing).
LATENCY_BUCKETS_S = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

__all__ = [
    "DocumentIn",
    "GroundingSummary",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "Source",
    "create_app",
    "pipeline_from_env",
]


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
    # Cumulative histogram counts per upper bound (Prometheus convention), plus a sum for computing the mean.
    app.state.buckets = {b: 0 for b in LATENCY_BUCKETS_S}
    app.state.latency_sum = 0.0
    app.state.requests = {}

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("RAG_API_KEY")
        # compare_digest takes the same time whatever the input, so the key can't be guessed byte by byte from response timings.
        if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id and record latency for every call."""
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        key = (request.url.path, response.status_code)
        app.state.requests[key] = app.state.requests.get(key, 0) + 1
        app.state.latency_sum += elapsed
        for b in LATENCY_BUCKETS_S:
            if elapsed <= b:
                app.state.buckets[b] += 1
        response.headers["X-Request-ID"] = rid
        return response

    def to_sources(hits) -> list[Source]:
        return [
            Source(n=i, doc_id=h.chunk.doc_id, heading=h.chunk.heading, text=h.chunk.text, score=round(h.score, 6))
            for i, h in enumerate(hits, start=1)
        ]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(app.state.pipeline.kb.chunks)}

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_key)])
    def ingest(req: IngestRequest) -> IngestResponse:
        added = app.state.pipeline.ingest([Document(d.doc_id, d.text, d.metadata) for d in req.documents])
        return IngestResponse(
            added_chunks=added,
            total_chunks=len(app.state.pipeline.kb.chunks),
        )

    def _build_access_and_filter(req: QueryRequest) -> tuple[AccessContext | None, MetadataFilter | None]:
        access = (
            AccessContext(
                tenant_id=req.tenant_id,
                principal_id=req.principal_id,
                roles=frozenset(req.roles),
                clearance_level=req.clearance_level,
            )
            if req.tenant_id is not None
            else None
        )
        mfilter = MetadataFilter(equals=req.filters) if req.filters else None
        return access, mfilter

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_key)])
    def query(req: QueryRequest) -> QueryResponse:
        access, mfilter = _build_access_and_filter(req)
        ans = app.state.pipeline.answer(req.question, access=access, metadata_filter=mfilter)
        cited_ids = {id(h) for h in ans.citations}
        grounding = None
        if req.verify_grounding:
            rep = verify_grounding(ans.answer, ans.retrieved)
            grounding = GroundingSummary(
                is_grounded=rep.is_grounded,
                faithfulness_score=rep.faithfulness_score,
                citation_precision=rep.citation_precision,
                claim_coverage=rep.claim_coverage,
                invalid_citations=rep.invalid_citation_indices,
            )
        return QueryResponse(
            answer=ans.answer,
            abstained=ans.abstained,
            sources=to_sources(ans.retrieved),
            cited=[i for i, h in enumerate(ans.retrieved, start=1) if id(h) in cited_ids],
            timings_ms=ans.timings_ms,
            grounding=grounding,
        )

    @app.post("/query/stream", dependencies=[Depends(require_key)])
    def query_stream(req: QueryRequest) -> StreamingResponse:
        access, mfilter = _build_access_and_filter(req)
        t0 = time.perf_counter()
        hits, tokens = app.state.pipeline.answer_stream(req.question, access=access, metadata_filter=mfilter)

        def events():
            seq = 1
            # Sources go first, so the UI can render them while the answer is still being written.
            yield format_sse("sources", [s.model_dump() for s in to_sources(hits)], event_id=str(seq))
            seq += 1
            chunks_text: list[str] = []
            tok_count = 0
            for tok in tokens:
                chunks_text.append(tok)
                tok_count += 1
                yield format_sse("token", tok, event_id=str(seq))
                seq += 1
            full_text = "".join(chunks_text).strip()
            if req.verify_grounding:
                rep = verify_grounding(full_text, hits)
                summary = GroundingSummary(
                    is_grounded=rep.is_grounded,
                    faithfulness_score=rep.faithfulness_score,
                    citation_precision=rep.citation_precision,
                    claim_coverage=rep.claim_coverage,
                    invalid_citations=rep.invalid_citation_indices,
                )
                yield format_sse("grounding", summary, event_id=str(seq))
                seq += 1
            done = SSEDonePayload(
                cited=parse_citations(full_text, len(hits)),
                abstained=is_abstention(full_text),
                token_count=tok_count,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
            yield format_sse("done", done, event_id=str(seq))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """Prometheus exposition format, so a standard scraper and Grafana dashboards work unchanged."""
        lines = ["# HELP rag_requests_total HTTP requests by path and status.", "# TYPE rag_requests_total counter"]
        for (path, status), n in sorted(app.state.requests.items()):
            lines.append(f'rag_requests_total{{path="{path}",status="{status}"}} {n}')
        lines += ["# HELP rag_request_seconds Request latency.", "# TYPE rag_request_seconds histogram"]
        total = sum(app.state.requests.values())
        for b in LATENCY_BUCKETS_S:
            lines.append(f'rag_request_seconds_bucket{{le="{b}"}} {app.state.buckets[b]}')
        lines.append(f'rag_request_seconds_bucket{{le="+Inf"}} {total}')
        lines.append(f"rag_request_seconds_sum {app.state.latency_sum:.6f}")
        lines.append(f"rag_request_seconds_count {total}")
        return "\n".join(lines) + "\n"

    return app


# `uvicorn ragservice.api:app` entry point. It's created lazily, so importing this module in tests has no side effects.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
