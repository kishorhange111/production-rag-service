"""OpenTelemetry-compatible span tracing and Prometheus metrics registry for RAG serving.

Uses ``opentelemetry.trace`` automatically when installed, and falls back to a lightweight
in-memory span recorder so distributed trace instrumentation is always exercisable in unit
tests and minimal containers without extra dependencies.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

LATENCY_BUCKETS_S = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


@dataclass
class SpanRecord:
    """Recorded execution span compatible with OpenTelemetry span semantics."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ns: int
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return round((self.end_ns - self.start_ns) / 1_000_000.0, 4) if self.end_ns else 0.0


class RAGTracer:
    """Lightweight OTel-compatible tracer with optional ``opentelemetry`` bridge."""

    def __init__(self, service_name: str = "production-rag-service", max_recorded: int = 500) -> None:
        self.service_name = service_name
        self._spans: deque[SpanRecord] = deque(maxlen=max_recorded)
        self._stack: list[SpanRecord] = []
        self._otel_tracer = None
        try:
            from opentelemetry import trace as otel_trace

            self._otel_tracer = otel_trace.get_tracer(service_name)
        except Exception:
            self._otel_tracer = None

    @property
    def recorded_spans(self) -> list[SpanRecord]:
        return list(self._spans)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None, trace_id: str | None = None):
        parent = self._stack[-1] if self._stack else None
        tid = trace_id or (parent.trace_id if parent else uuid.uuid4().hex)
        rec = SpanRecord(
            trace_id=tid,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
            name=name,
            start_ns=time.perf_counter_ns(),
            attributes=dict(attributes or {}),
        )
        self._stack.append(rec)
        otel_cm = self._otel_tracer.start_as_current_span(name) if self._otel_tracer is not None else None
        otel_span = otel_cm.__enter__() if otel_cm is not None else None
        try:
            if otel_span is not None:
                for k, v in rec.attributes.items():
                    otel_span.set_attribute(k, str(v))
            yield rec
        except Exception as exc:
            rec.status = "ERROR"
            rec.error = str(exc)
            raise
        finally:
            rec.end_ns = time.perf_counter_ns()
            self._stack.pop()
            self._spans.append(rec)
            if otel_cm is not None:
                otel_cm.__exit__(None, None, None)


class MetricsRegistry:
    """Prometheus text-format registry for HTTP and RAG-specific SLIs."""

    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS_S) -> None:
        self.buckets = buckets
        self.requests: dict[tuple[str, int], int] = {}
        self.latency_buckets: dict[float, int] = {b: 0 for b in buckets}
        self.latency_sum: float = 0.0
        self.stage_seconds_sum: dict[str, float] = {}
        self.stage_count: dict[str, int] = {}
        self.cache_events: dict[str, int] = {"hit": 0, "miss": 0}
        self.guardrail_blocks: dict[str, int] = {"query": 0, "context": 0}
        self.abstentions_total: int = 0
        self.indexed_chunks: int = 0

    def observe_request(self, path: str, status: int, elapsed_s: float) -> None:
        key = (path, status)
        self.requests[key] = self.requests.get(key, 0) + 1
        self.latency_sum += elapsed_s
        for b in self.buckets:
            if elapsed_s <= b:
                self.latency_buckets[b] += 1

    def observe_stage(self, stage: str, elapsed_ms: float) -> None:
        self.stage_seconds_sum[stage] = self.stage_seconds_sum.get(stage, 0.0) + (elapsed_ms / 1000.0)
        self.stage_count[stage] = self.stage_count.get(stage, 0) + 1

    def inc_cache(self, outcome: str) -> None:
        self.cache_events[outcome] = self.cache_events.get(outcome, 0) + 1

    def inc_guardrail_block(self, stage: str, count: int = 1) -> None:
        self.guardrail_blocks[stage] = self.guardrail_blocks.get(stage, 0) + count

    def inc_abstention(self) -> None:
        self.abstentions_total += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP rag_requests_total HTTP requests by path and status.",
            "# TYPE rag_requests_total counter",
        ]
        for (path, status), n in sorted(self.requests.items()):
            lines.append(f'rag_requests_total{{path="{path}",status="{status}"}} {n}')

        lines += [
            "# HELP rag_request_seconds Request latency.",
            "# TYPE rag_request_seconds histogram",
        ]
        total = sum(self.requests.values())
        for b in self.buckets:
            lines.append(f'rag_request_seconds_bucket{{le="{b}"}} {self.latency_buckets[b]}')
        lines.append(f'rag_request_seconds_bucket{{le="+Inf"}} {total}')
        lines.append(f"rag_request_seconds_sum {self.latency_sum:.6f}")
        lines.append(f"rag_request_seconds_count {total}")

        lines += [
            "# HELP rag_stage_duration_seconds Cumulative RAG pipeline stage latency in seconds.",
            "# TYPE rag_stage_duration_seconds summary",
        ]
        for stage in sorted(self.stage_seconds_sum):
            lines.append(f'rag_stage_duration_seconds_sum{{stage="{stage}"}} {self.stage_seconds_sum[stage]:.6f}')
            lines.append(f'rag_stage_duration_seconds_count{{stage="{stage}"}} {self.stage_count[stage]}')

        lines += [
            "# HELP rag_cache_events_total Semantic query cache hits and misses.",
            "# TYPE rag_cache_events_total counter",
        ]
        for outcome, count in sorted(self.cache_events.items()):
            lines.append(f'rag_cache_events_total{{outcome="{outcome}"}} {count}')

        lines += [
            "# HELP rag_guardrail_blocks_total Requests or chunks blocked by prompt-injection defense.",
            "# TYPE rag_guardrail_blocks_total counter",
        ]
        for stage, count in sorted(self.guardrail_blocks.items()):
            lines.append(f'rag_guardrail_blocks_total{{stage="{stage}"}} {count}')

        lines += [
            "# HELP rag_abstentions_total Total answers where the pipeline abstained.",
            "# TYPE rag_abstentions_total counter",
            f"rag_abstentions_total {self.abstentions_total}",
            "# HELP rag_indexed_chunks Current number of indexed chunks in the knowledge base.",
            "# TYPE rag_indexed_chunks gauge",
            f"rag_indexed_chunks {self.indexed_chunks}",
        ]
        return "\n".join(lines) + "\n"
