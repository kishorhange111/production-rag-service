"""Metrics for the two halves of RAG: did we *find* the evidence, and did we *use* it correctly?

Retrieval (needs the id of the gold passage for each question):
    recall@k   gold passage is among the top-k chunks
    mrr        mean of 1/rank of the first gold chunk (0 if not retrieved)
    ndcg@k     rank-discounted gain; rewards putting the gold chunk first

Answers:
    exact_match / f1     SQuAD-style overlap with the reference answer(s)
    abstention metrics   on questions that have *no* answer in the corpus, does the system say "I don't know"?
    citation_validity    share of answers whose citations all point at real sources
    faithfulness         (LLM judge) is every claim in the answer supported by the cited sources?
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter


def recall_at_k(ranked_doc_ids: list[str], gold: str, k: int) -> float:
    return float(gold in ranked_doc_ids[:k])


def reciprocal_rank(ranked_doc_ids: list[str], gold: str) -> float:
    for i, d in enumerate(ranked_doc_ids, start=1):
        if d == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], gold: str, k: int) -> float:
    """With a single relevant document the ideal DCG is 1, so nDCG = 1 / log2(rank + 1)."""
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if d == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


def _normalize(s: str) -> str:
    """Official SQuAD normalisation: lowercase, strip punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def strip_citations(answer: str) -> str:
    """'Paris is the capital [1][2].' -> 'Paris is the capital.' (citations are not part of the answer text)."""
    return re.sub(r"\s*\[\d+\]", "", answer).strip()


def exact_match(prediction: str, references: list[str]) -> float:
    p = _normalize(strip_citations(prediction))
    return float(any(p == _normalize(r) for r in references))


def f1(prediction: str, references: list[str]) -> float:
    """Best token-overlap F1 over the reference answers."""
    pred = _normalize(strip_citations(prediction)).split()
    best = 0.0
    for ref in references:
        gold = _normalize(ref).split()
        common = sum((Counter(pred) & Counter(gold)).values())
        if common == 0:
            continue
        p, r = common / len(pred), common / len(gold)
        best = max(best, 2 * p * r / (p + r))
    return best


def answer_contains_reference(prediction: str, references: list[str]) -> float:
    """1 if any reference answer appears verbatim (normalised) in the prediction.

    LLM answers are full sentences ("The Normans were in Normandy [1]."), so
    exact match against a 1-3 word SQuAD span is unfairly strict. Containment
    is the standard "lenient" correctness signal for generative QA.
    """
    p = _normalize(strip_citations(prediction))
    return float(any(_normalize(r) and _normalize(r) in p for r in references))


def abstention_scores(abstained: list[bool], answerable: list[bool]) -> dict[str, float]:
    """Treat "abstain" as the positive class for unanswerable questions.

    precision: when the system abstains, was the question really unanswerable?
    recall:    of the unanswerable questions, how many did it abstain on?
    false_abstention_rate: answerable questions it wrongly refused (lost helpfulness)
    """
    tp = sum(a and not ans for a, ans in zip(abstained, answerable))
    fp = sum(a and ans for a, ans in zip(abstained, answerable))
    fn = sum((not a) and (not ans) for a, ans in zip(abstained, answerable))
    n_ans = sum(answerable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_abstention_rate": fp / n_ans if n_ans else 0.0,
    }


FAITHFULNESS_PROMPT = """You are checking an AI answer for hallucinations.
Sources:
{sources}

Answer: {answer}

Is every factual claim in the answer directly supported by the sources? Ignore citation formatting.
Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


def parse_faithfulness(judge_output: str) -> float | None:
    """1.0 for SUPPORTED, 0.0 for UNSUPPORTED, None if the judge didn't follow the format."""
    text = judge_output.strip().upper()
    if text.startswith("UNSUPPORTED"):
        return 0.0
    if text.startswith("SUPPORTED"):
        return 1.0
    return None
