"""Concurrent load-test and latency benchmark harness for ``production-rag-service``.

Supports two modes:
1. **In-process benchmark** (default): exercises ``RAGPipeline.answer`` concurrently
   across threads with zero network setup, suitable for CI and local regression checks.
2. **HTTP benchmark** (``--url http://localhost:8080/query``): drives real HTTP traffic
   against a running FastAPI / Docker / Kubernetes deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline

DEFAULT_QUERIES = [
    "Which port do applications use to connect to the database through PgBouncer?",
    "How long are full nightly database backups kept?",
    "Which environment receives automatic deployments on every merge to main?",
    "Where must rollbacks be announced with the release id?",
    "When do new engineers join the on-call rotation?",
]


@dataclass(frozen=True)
class BenchmarkSummary:
    mode: str
    concurrency: int
    total_requests: int
    errors: int
    error_rate: float
    wall_time_s: float
    throughput_rps: float
    min_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile for ``p`` in ``[0, 100]``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize_latencies(
    latencies_ms: list[float],
    errors: int,
    wall_time_s: float,
    mode: str,
    concurrency: int,
    total_requests: int,
) -> BenchmarkSummary:
    wall = max(wall_time_s, 1e-6)
    return BenchmarkSummary(
        mode=mode,
        concurrency=concurrency,
        total_requests=total_requests,
        errors=errors,
        error_rate=round(errors / total_requests, 4) if total_requests else 0.0,
        wall_time_s=round(wall, 4),
        throughput_rps=round(total_requests / wall, 2),
        min_ms=round(min(latencies_ms), 3) if latencies_ms else 0.0,
        mean_ms=round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else 0.0,
        p50_ms=round(percentile(latencies_ms, 50), 3),
        p95_ms=round(percentile(latencies_ms, 95), 3),
        p99_ms=round(percentile(latencies_ms, 99), 3),
        max_ms=round(max(latencies_ms), 3) if latencies_ms else 0.0,
    )


def run_inprocess_benchmark(
    pipeline: RAGPipeline,
    queries: list[str] = DEFAULT_QUERIES,
    concurrency: int = 4,
    total_requests: int = 100,
) -> BenchmarkSummary:
    """Benchmark ``pipeline.answer`` under concurrent worker threads."""
    latencies: list[float] = []
    errors = 0

    def _one(idx: int) -> tuple[float, bool]:
        q = queries[idx % len(queries)]
        t0 = time.perf_counter()
        try:
            pipeline.answer(q)
            return (time.perf_counter() - t0) * 1000.0, True
        except Exception:
            return (time.perf_counter() - t0) * 1000.0, False

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for ms, ok in pool.map(_one, range(total_requests)):
            latencies.append(ms)
            if not ok:
                errors += 1
    wall = time.perf_counter() - t_start
    return summarize_latencies(latencies, errors, wall, "inprocess", concurrency, total_requests)


def run_http_benchmark(
    url: str,
    queries: list[str] = DEFAULT_QUERIES,
    concurrency: int = 4,
    total_requests: int = 100,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> BenchmarkSummary:
    """Benchmark a live ``POST /query`` HTTP endpoint."""
    latencies: list[float] = []
    errors = 0
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    def _one(idx: int) -> tuple[float, bool]:
        q = queries[idx % len(queries)]
        req = urllib.request.Request(url, data=json.dumps({"question": q}).encode(), headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
                ok = 200 <= resp.status < 300
        except Exception:
            ok = False
        return (time.perf_counter() - t0) * 1000.0, ok

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for ms, ok in pool.map(_one, range(total_requests)):
            latencies.append(ms)
            if not ok:
                errors += 1
    wall = time.perf_counter() - t_start
    return summarize_latencies(latencies, errors, wall, "http", concurrency, total_requests)


def format_markdown_table(summary: BenchmarkSummary) -> str:
    return (
        "| Mode | Concurrency | Requests | RPS | Error Rate | p50 (ms) | p95 (ms) | p99 (ms) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {summary.mode} | {summary.concurrency} | {summary.total_requests} | {summary.throughput_rps:.1f} "
        f"| {summary.error_rate:.2%} | {summary.p50_ms:.2f} | {summary.p95_ms:.2f} | {summary.p99_ms:.2f} |\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load-test and latency benchmark for production-rag-service.")
    parser.add_argument("--url", default=None, help="Target POST /query URL; omit for in-process benchmark")
    parser.add_argument("--corpus", default="data/sample")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    if args.url:
        summary = run_http_benchmark(args.url, concurrency=args.concurrency, total_requests=args.requests, api_key=args.api_key)
    else:
        pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), mode="hybrid", top_k=3)
        pipe.ingest_directory(args.corpus)
        summary = run_inprocess_benchmark(pipe, concurrency=args.concurrency, total_requests=args.requests)

    print(format_markdown_table(summary))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
