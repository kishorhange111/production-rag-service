"""Tests for the automated RAG evaluation suite and golden QA fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from ragservice.embeddings import HashingEmbedder
from ragservice.eval.suite import load_golden_cases, run_eval_suite
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
GOLDEN_PATH = ROOT / "data" / "eval" / "golden_qa.json"


def test_golden_eval_suite_meets_quality_baseline():
    cases = load_golden_cases(GOLDEN_PATH)
    assert len(cases) == 10

    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), mode="bm25", top_k=3)
    pipe.ingest_directory(SAMPLE_DIR)
    report = run_eval_suite(pipe, cases, k=3)

    assert report.total_cases == 10
    assert report.answerable_cases == 8 and report.unanswerable_cases == 2
    assert report.context_recall_at_k >= 0.85
    assert report.faithfulness >= 0.85
    assert report.citation_precision >= 0.85
    assert report.answer_match_rate >= 0.75
    assert report.abstention_recall == 1.0
    assert report.false_abstention_rate == 0.0

    serialized = json.dumps(report.to_dict())
    assert "db-pgbouncer-port" in serialized


def test_eval_suite_handles_empty_case_list():
    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3)
    report = run_eval_suite(pipe, [])
    assert report.total_cases == 0 and report.faithfulness == 0.0
