"""Automated offline RAG evaluation suite: faithfulness, context recall, citation precision, and abstention.

Runs deterministically on CPU with zero paid API keys so it can execute in CI on every
pull request against versioned golden fixtures (``data/eval/golden_qa.json``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragservice.citations import verify_grounding
from ragservice.eval.metrics import abstention_scores, answer_contains_reference, reciprocal_rank
from ragservice.pipeline import RAGPipeline


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_doc_ids: list[str]
    reference_answers: list[str]
    answerable: bool = True


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    question: str
    answer: str
    abstained: bool
    answerable: bool
    context_recall: float
    mrr: float
    faithfulness: float
    citation_precision: float
    answer_match: float
    latency_ms: float


@dataclass(frozen=True)
class EvalSuiteReport:
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    context_recall_at_k: float
    mrr: float
    faithfulness: float
    citation_precision: float
    answer_match_rate: float
    abstention_recall: float
    false_abstention_rate: float
    avg_latency_ms: float
    cases: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenCase(
            id=item["id"],
            question=item["question"],
            expected_doc_ids=list(item.get("expected_doc_ids", [])),
            reference_answers=list(item.get("reference_answers", [])),
            answerable=bool(item.get("answerable", True)),
        )
        for item in raw
    ]


def _mean(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def run_eval_suite(pipeline: RAGPipeline, cases: list[GoldenCase], k: int | None = None) -> EvalSuiteReport:
    """Evaluate ``pipeline`` over ``cases`` and aggregate retrieval, citation, and faithfulness metrics."""
    orig_k = pipeline.top_k
    if k is not None:
        pipeline.top_k = k

    results: list[CaseResult] = []
    try:
        for case in cases:
            t0 = time.perf_counter()
            ans = pipeline.answer(case.question)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            retrieved_docs = [h.chunk.doc_id for h in ans.retrieved]
            if case.expected_doc_ids:
                found = sum(1 for doc_id in case.expected_doc_ids if doc_id in retrieved_docs)
                ctx_recall = found / len(case.expected_doc_ids)
                mrr = max(reciprocal_rank(retrieved_docs, doc_id) for doc_id in case.expected_doc_ids)
            else:
                ctx_recall = 1.0
                mrr = 1.0

            grounding = verify_grounding(ans.answer, ans.retrieved)
            ans_match = (
                answer_contains_reference(ans.answer, case.reference_answers)
                if case.answerable and case.reference_answers
                else float(ans.abstained)
            )
            results.append(
                CaseResult(
                    case_id=case.id,
                    question=case.question,
                    answer=ans.answer,
                    abstained=ans.abstained,
                    answerable=case.answerable,
                    context_recall=round(ctx_recall, 4),
                    mrr=round(mrr, 4),
                    faithfulness=grounding.faithfulness_score,
                    citation_precision=grounding.citation_precision,
                    answer_match=round(ans_match, 4),
                    latency_ms=round(elapsed_ms, 3),
                )
            )
    finally:
        pipeline.top_k = orig_k

    ans_results = [r for r in results if r.answerable]
    unans_results = [r for r in results if not r.answerable]
    abst = abstention_scores([r.abstained for r in results], [r.answerable for r in results])

    return EvalSuiteReport(
        total_cases=len(results),
        answerable_cases=len(ans_results),
        unanswerable_cases=len(unans_results),
        context_recall_at_k=_mean([r.context_recall for r in ans_results]),
        mrr=_mean([r.mrr for r in ans_results]),
        faithfulness=_mean([r.faithfulness for r in results]),
        citation_precision=_mean([r.citation_precision for r in results]),
        answer_match_rate=_mean([r.answer_match for r in ans_results]),
        abstention_recall=round(abst["abstain_recall"], 4),
        false_abstention_rate=round(abst["false_abstention_rate"], 4),
        avg_latency_ms=_mean([r.latency_ms for r in results]),
        cases=results,
    )
