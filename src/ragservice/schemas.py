"""Typed Pydantic request/response schemas and Server-Sent Events (SSE) framing helpers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ragservice.retrieval import Mode


class DocumentIn(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=512)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=1000)


class IngestResponse(BaseModel):
    added_chunks: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Mode | None = None
    tenant_id: str | None = None
    principal_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    clearance_level: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)
    verify_grounding: bool = False


class Source(BaseModel):
    n: int
    doc_id: str
    heading: str
    text: str
    score: float | None = None


class GroundingSummary(BaseModel):
    is_grounded: bool
    faithfulness_score: float
    citation_precision: float
    claim_coverage: float
    invalid_citations: list[int] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
    cited: list[int]
    timings_ms: dict[str, float]
    grounding: GroundingSummary | None = None


class SSEDonePayload(BaseModel):
    cited: list[int] = Field(default_factory=list)
    abstained: bool = False
    token_count: int = 0
    elapsed_ms: float = 0.0


def format_sse(event: str, data: BaseModel | dict[str, Any] | list[Any] | str, event_id: str | None = None) -> str:
    """Serialize one Server-Sent Events frame with optional ``id:`` line."""
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"
