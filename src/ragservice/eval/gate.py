"""CI evaluation regression gate CLI.

Runs the deterministic offline RAG evaluation suite against golden fixtures and exits
with code 1 if any quality metric regresses below configured thresholds. Designed to
run in GitHub Actions (``.github/workflows/eval-gate.yml``) on every pull request.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ragservice.embeddings import HashingEmbedder
from ragservice.eval.suite import EvalSuiteReport, load_golden_cases, run_eval_suite
from ragservice.generation import ExtractiveLLM
from ragservice.pipeline import RAGPipeline


@dataclass(frozen=True)
class EvalThresholds:
    min_context_recall: float = 0.85
    min_faithfulness: float = 0.85
    min_citation_precision: float = 0.85
    min_answer_match: float = 0.75
    min_abstention_recall: float = 0.90
    max_false_abstention_rate: float = 0.10


@dataclass(frozen=True)
class GateCheckResult:
    passed: bool
    failures: list[str]
    report: EvalSuiteReport


def evaluate_thresholds(report: EvalSuiteReport, thresholds: EvalThresholds) -> GateCheckResult:
    failures: list[str] = []
    checks = [
        ("context_recall_at_k", report.context_recall_at_k, ">=", thresholds.min_context_recall),
        ("faithfulness", report.faithfulness, ">=", thresholds.min_faithfulness),
        ("citation_precision", report.citation_precision, ">=", thresholds.min_citation_precision),
        ("answer_match_rate", report.answer_match_rate, ">=", thresholds.min_answer_match),
        ("abstention_recall", report.abstention_recall, ">=", thresholds.min_abstention_recall),
        ("false_abstention_rate", report.false_abstention_rate, "<=", thresholds.max_false_abstention_rate),
    ]
    for name, actual, op, bound in checks:
        ok = actual >= bound if op == ">=" else actual <= bound
        if not ok:
            failures.append(f"{name}={actual:.4f} failed threshold {op} {bound:.4f}")
    return GateCheckResult(passed=not failures, failures=failures, report=report)


def format_markdown_summary(result: GateCheckResult, thresholds: EvalThresholds) -> str:
    r = result.report
    status = "PASS" if result.passed else "FAIL"
    rows = [
        ("Context Recall@k", f"{r.context_recall_at_k:.3f}", f">= {thresholds.min_context_recall:.3f}"),
        ("MRR", f"{r.mrr:.3f}", "info"),
        ("Faithfulness", f"{r.faithfulness:.3f}", f">= {thresholds.min_faithfulness:.3f}"),
        ("Citation Precision", f"{r.citation_precision:.3f}", f">= {thresholds.min_citation_precision:.3f}"),
        ("Answer Match", f"{r.answer_match_rate:.3f}", f">= {thresholds.min_answer_match:.3f}"),
        ("Abstention Recall", f"{r.abstention_recall:.3f}", f">= {thresholds.min_abstention_recall:.3f}"),
        ("False Abstention Rate", f"{r.false_abstention_rate:.3f}", f"<= {thresholds.max_false_abstention_rate:.3f}"),
        ("Avg Latency (ms)", f"{r.avg_latency_ms:.2f}", "info"),
    ]
    lines = [
        f"## RAG Eval Regression Gate: **{status}** ({r.total_cases} cases)",
        "",
        "| Metric | Score | Threshold |",
        "| --- | ---: | ---: |",
        *[f"| {name} | {score} | {thr} |" for name, score, thr in rows],
    ]
    if result.failures:
        lines += ["", "### Failures", *[f"- {f}" for f in result.failures]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAG eval suite and enforce regression thresholds.")
    parser.add_argument("--corpus", default="data/sample")
    parser.add_argument("--golden", default="data/eval/golden_qa.json")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--min-faithfulness", type=float, default=0.85)
    parser.add_argument("--min-citation-precision", type=float, default=0.85)
    parser.add_argument("--min-answer-match", type=float, default=0.75)
    parser.add_argument("--min-abstention-recall", type=float, default=0.90)
    parser.add_argument("--max-false-abstention", type=float, default=0.10)
    parser.add_argument("--out", default=None, help="Optional path to write JSON eval report")
    parser.add_argument("--summary-md", default=None, help="Optional path to write Markdown summary")
    args = parser.parse_args(argv)

    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), mode=args.mode, top_k=args.top_k)
    pipe.ingest_directory(args.corpus)
    cases = load_golden_cases(args.golden)
    report = run_eval_suite(pipe, cases, k=args.top_k)

    thresholds = EvalThresholds(
        min_context_recall=args.min_recall,
        min_faithfulness=args.min_faithfulness,
        min_citation_precision=args.min_citation_precision,
        min_answer_match=args.min_answer_match,
        min_abstention_recall=args.min_abstention_recall,
        max_false_abstention_rate=args.max_false_abstention,
    )
    result = evaluate_thresholds(report, thresholds)
    summary = format_markdown_summary(result, thresholds)
    print(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"passed": result.passed, "failures": result.failures, "report": report.to_dict()}, indent=2))
    if args.summary_md:
        md_path = Path(args.summary_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(summary, encoding="utf-8")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
