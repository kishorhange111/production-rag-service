"""Unit tests for the load-test and latency benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("load_test", ROOT / "benchmarks" / "load_test.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["load_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_inprocess_benchmark_and_percentiles(tmp_path):
    mod = _load_bench_module()
    assert mod.percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0

    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), mode="hybrid", top_k=3)
    pipe.ingest_directory(ROOT / "data" / "sample")
    summary = mod.run_inprocess_benchmark(pipe, concurrency=2, total_requests=20)
    assert summary.total_requests == 20
    assert summary.errors == 0 and summary.error_rate == 0.0
    assert summary.throughput_rps > 0
    assert summary.min_ms <= summary.p50_ms <= summary.p95_ms <= summary.p99_ms <= summary.max_ms

    md = mod.format_markdown_table(summary)
    assert "| inprocess |" in md

    out_json = tmp_path / "bench.json"
    assert mod.main(["--corpus", str(ROOT / "data" / "sample"), "--requests", "10", "--concurrency", "2", "--out", str(out_json)]) == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["total_requests"] == 10 and data["errors"] == 0


def test_k6_script_present_with_thresholds():
    k6 = (ROOT / "benchmarks" / "k6_load_test.js").read_text(encoding="utf-8")
    assert "http_req_duration" in k6 and "/query" in k6
