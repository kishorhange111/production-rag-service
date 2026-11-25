"""Citation extraction and deterministic faithfulness / grounding verification.

Even when an LLM emits inline citations like ``[1]``, two failure modes remain common:
1. **Fabricated references**: citing ``[7]`` when only 5 sources were provided.
2. **Unsupported or misattributed claims**: stating a number or fact not present in the
   cited source (e.g. "Backups are kept for 99 days [1]" when source [1] says "35 days").

This module extracts sentence-level claims and citations and verifies grounding against
the cited chunks using content-token overlap plus strict numeric/identifier consistency,
running deterministically on CPU with zero external API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragservice.generation import is_abstention
from ragservice.index import tokenize
from ragservice.retrieval import Hit

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:]\d+)*\b")


@dataclass(frozen=True)
class ClaimCitation:
    """Verification result for one sentence-level claim in a generated answer."""

    sentence: str
    clean_sentence: str
    cited_indices: list[int]
    invalid_indices: list[int]
    grounded: bool
    support_score: float
    supporting_sources: list[int]


@dataclass(frozen=True)
class GroundingReport:
    """Aggregate citation and faithfulness report for an answer."""

    claims: list[ClaimCitation]
    valid_citation_indices: list[int]
    invalid_citation_indices: list[int]
    citation_precision: float
    claim_coverage: float
    faithfulness_score: float
    is_grounded: bool


def extract_claim_citations(answer: str, num_sources: int) -> list[tuple[str, str, list[int], list[int]]]:
    """Split ``answer`` into claims and extract valid vs out-of-range citation indices."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
    out: list[tuple[str, str, list[int], list[int]]] = []
    for sent in sentences:
        valid: list[int] = []
        invalid: list[int] = []
        for m in _CITATION_RE.finditer(sent):
            idx = int(m.group(1))
            if 1 <= idx <= num_sources:
                if idx not in valid:
                    valid.append(idx)
            elif idx not in invalid:
                invalid.append(idx)
        clean = _CITATION_RE.sub("", sent).strip()
        clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
        out.append((sent, clean, valid, invalid))
    return out


def _claim_support_against_text(claim: str, source_text: str) -> float:
    """Score how well ``source_text`` supports ``claim`` in ``[0, 1]`` with strict numeric check."""
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 1.0
    source_tokens = set(tokenize(source_text))

    # Strict numeric grounding: every number asserted in the claim must appear in the source.
    claim_numbers = set(_NUMBER_RE.findall(claim))
    if claim_numbers:
        source_numbers = set(_NUMBER_RE.findall(source_text))
        if not claim_numbers.issubset(source_numbers):
            return 0.0

    return len(claim_tokens & source_tokens) / len(claim_tokens)


def verify_grounding(
    answer: str,
    hits: list[Hit],
    support_threshold: float = 0.35,
    min_faithfulness: float = 0.7,
) -> GroundingReport:
    """Verify every claim in ``answer`` against its cited sources (and full retrieved set)."""
    if not answer.strip() or is_abstention(answer):
        return GroundingReport(
            claims=[],
            valid_citation_indices=[],
            invalid_citation_indices=[],
            citation_precision=1.0,
            claim_coverage=1.0,
            faithfulness_score=1.0,
            is_grounded=True,
        )

    extracted = extract_claim_citations(answer, len(hits))
    claims: list[ClaimCitation] = []
    all_valid: list[int] = []
    all_invalid: list[int] = []

    cited_claim_count = 0
    cited_supported_count = 0
    grounded_claim_count = 0

    for sent, clean, valid, invalid in extracted:
        for idx in valid:
            if idx not in all_valid:
                all_valid.append(idx)
        for idx in invalid:
            if idx not in all_invalid:
                all_invalid.append(idx)

        # Evaluate support against cited sources if present, else against all retrieved sources.
        candidate_indices = valid if valid else list(range(1, len(hits) + 1))
        combined_text = " ".join(hits[i - 1].chunk.text_for_index for i in candidate_indices)
        score = _claim_support_against_text(clean, combined_text) if candidate_indices else 0.0

        supporting_sources = [
            i for i in range(1, len(hits) + 1) if _claim_support_against_text(clean, hits[i - 1].chunk.text_for_index) >= support_threshold
        ]
        grounded = score >= support_threshold and not invalid
        if grounded:
            grounded_claim_count += 1
        if valid:
            cited_claim_count += 1
            if score >= support_threshold:
                cited_supported_count += 1

        claims.append(
            ClaimCitation(
                sentence=sent,
                clean_sentence=clean,
                cited_indices=valid,
                invalid_indices=invalid,
                grounded=grounded,
                support_score=round(score, 4),
                supporting_sources=supporting_sources,
            )
        )

    n_claims = len(claims)
    faithfulness = grounded_claim_count / n_claims if n_claims else 1.0
    coverage = cited_claim_count / n_claims if n_claims else 1.0
    precision = cited_supported_count / cited_claim_count if cited_claim_count else (1.0 if not all_invalid else 0.0)
    is_grounded = faithfulness >= min_faithfulness and not all_invalid

    return GroundingReport(
        claims=claims,
        valid_citation_indices=all_valid,
        invalid_citation_indices=all_invalid,
        citation_precision=round(precision, 4),
        claim_coverage=round(coverage, 4),
        faithfulness_score=round(faithfulness, 4),
        is_grounded=is_grounded,
    )
