"""Unit tests for citation extraction and faithfulness grounding verification."""

from __future__ import annotations

from ragservice.chunking import Chunk
from ragservice.citations import extract_claim_citations, verify_grounding
from ragservice.generation import ABSTAIN
from ragservice.retrieval import Hit


def _sample_hits() -> list[Hit]:
    return [
        Hit(Chunk("b#0", "backup.md", "Full backups run nightly at 02:00 UTC and are kept for 35 days.", "Backups"), 0.9),
        Hit(Chunk("o#0", "oncall.md", "On-call shifts last one week from Monday to Monday.", "On-call"), 0.8),
    ]


def test_grounded_answer_scores_high():
    hits = _sample_hits()
    ans = "Backups are kept for 35 days [1]. On-call shifts last one week [2]."
    report = verify_grounding(ans, hits)
    assert report.is_grounded is True
    assert report.faithfulness_score == 1.0
    assert report.citation_precision == 1.0
    assert report.claim_coverage == 1.0
    assert report.valid_citation_indices == [1, 2]
    assert report.invalid_citation_indices == []


def test_hallucinated_number_fails_grounding():
    hits = _sample_hits()
    ans = "Backups are kept for 99 days [1]."
    report = verify_grounding(ans, hits)
    assert report.is_grounded is False
    assert report.faithfulness_score == 0.0
    assert report.claims[0].grounded is False


def test_out_of_range_and_misattributed_citations_detected():
    hits = _sample_hits()
    extracted = extract_claim_citations("Backups last 35 days [2][9].", num_sources=2)
    assert extracted[0][2] == [2] and extracted[0][3] == [9]

    report = verify_grounding("Backups are kept for 35 days [2][9].", hits)
    assert report.is_grounded is False
    assert report.invalid_citation_indices == [9]
    assert report.citation_precision == 0.0
    assert 1 in report.claims[0].supporting_sources


def test_abstention_is_grounded():
    report = verify_grounding(ABSTAIN, _sample_hits())
    assert report.is_grounded is True
    assert report.faithfulness_score == 1.0
    assert report.claims == []
