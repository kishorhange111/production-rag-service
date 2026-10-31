"""Safety checks around retrieval and generation.

1. **Indirect prompt injection.** Retrieved documents are untrusted input. A
   web page or uploaded PDF can contain "Ignore previous instructions and ...".
   Because RAG pastes that text into the LLM prompt, it can hijack the model.
   We screen retrieved chunks with high-precision patterns and drop matches.
   Pattern matching is a first line of defence, not a guarantee. A trained
   classifier (see the prompt-injection-guardrail project in
   generative-ai-engineering) catches more, at higher cost.
2. **Abstention.** If retrieval found nothing relevant, the right answer is "I
   don't know", not a fluent guess. The generator is also told to abstain when
   the context doesn't contain the answer. That check is evaluated separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragservice.retrieval import Hit

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts|messages)",
    r"disregard (the |all )?(previous|prior|above|system)",
    r"you are now (a|an|in) ",
    r"new instructions?:",
    r"system prompt",
    r"reveal (your|the) (instructions|prompt|system)",
    r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    """True if the text contains a known prompt-injection phrase."""
    return bool(_INJECTION.search(text))


@dataclass(frozen=True)
class ScreenResult:
    kept: list[Hit]
    dropped: list[Hit]


def screen_hits(hits: list[Hit]) -> ScreenResult:
    """Drop retrieved chunks that look like injection attempts (they are logged, not shown to the LLM)."""
    kept, dropped = [], []
    for h in hits:
        (dropped if looks_like_injection(h.chunk.text) else kept).append(h)
    return ScreenResult(kept, dropped)


def should_abstain(hits: list[Hit], min_hits: int = 1) -> bool:
    """Abstain before calling the LLM when retrieval returned (almost) nothing.

    BM25 returns nothing when no query term occurs in the corpus, and the
    screen may have removed everything. Then an answer can only be a hallucination.
    """
    return len(hits) < min_hits
