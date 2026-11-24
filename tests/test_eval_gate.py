"""Unit tests for the CI evaluation regression gate CLI and threshold checks."""

from __future__ import annotations

import json
from pathlib import Path

from ragservice.eval.gate import EvalThresholds, evaluate_thresholds, format_markdown_summary, main
from ragservice.eval.suite import EvalSuiteReport

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_thresholds_detects_regressions():
    report = EvalSuiteReport(
        total_cases=10,
        answerable_cases=8,
        unanswerable_cases=2,
        context_recall_at_k=0.70,
        mrr=0.65,
        faithfulness=0.95,
        citation_precision=0.90,
        answer_match_rate=0.60,
        abstention_recall=1.0,
        false_abstention_rate=0.20,
        avg_latency_ms=1.2,
        cases=[],
    )
    res = evaluate_thresholds(report, EvalThresholds())
    assert not res.passed
    assert any("context_recall_at_k" in f for f in res.failures)
    assert any("answer_match_rate" in f for f in res.failures)
    assert any("false_abstention_rate" in f for f in res.failures)
    md = format_markdown_summary(res, EvalThresholds())
    assert "**FAIL**" in md and "Context Recall@k" in md


def test_gate_cli_passes_on_golden_fixtures_and_writes_outputs(tmp_path):
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "summary.md"
    code = main(
        [
            "--corpus",
            str(ROOT / "data" / "sample"),
            "--golden",
            str(ROOT / "data" / "eval" / "golden_qa.json"),
            "--out",
            str(out_json),
            "--summary-md",
            str(out_md),
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "**PASS**" in out_md.read_text(encoding="utf-8")


def test_gate_cli_fails_when_threshold_impossible():
    code = main(
        [
            "--corpus",
            str(ROOT / "data" / "sample"),
            "--golden",
            str(ROOT / "data" / "eval" / "golden_qa.json"),
            "--min-answer-match",
            "1.01",
        ]
    )
    assert code == 1
