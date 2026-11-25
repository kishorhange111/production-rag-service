"""Layered prompt-injection and context-poisoning defense middleware.

Defends both the **query ingress** (direct prompt injection from callers) and
**retrieved context** (indirect prompt injection / context poisoning embedded in
untrusted documents):

1. Strips invisible zero-width characters and applies Unicode NFKC normalization
   before scanning so obfuscated payloads cannot bypass regex rules.
2. Inspects base64-encoded blobs embedded in text for hidden instructions.
3. Categorizes findings (``prompt_injection``, ``delimiter_escape``,
   ``exfiltration``, ``context_poisoning``, ``obfuscated_payload``) with a
   calibrated risk score in ``[0, 1]``.
4. Provides a Starlette/FastAPI HTTP middleware (``PromptDefenseMiddleware``)
   that rejects high-risk query payloads with HTTP 400 before they reach the LLM.
"""

from __future__ import annotations

import base64
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

ThreatCategory = Literal[
    "prompt_injection",
    "delimiter_escape",
    "exfiltration",
    "context_poisoning",
    "obfuscated_payload",
]
Severity = Literal["low", "medium", "high"]

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_BASE64_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

_RULES: list[tuple[ThreatCategory, Severity, float, re.Pattern[str]]] = [
    (
        "prompt_injection",
        "high",
        0.9,
        re.compile(r"ignore (all |any )?(previous|prior|above) (instructions|prompts|messages)", re.IGNORECASE),
    ),
    (
        "prompt_injection",
        "high",
        0.85,
        re.compile(r"disregard (the |all )?(previous|prior|above|system)", re.IGNORECASE),
    ),
    (
        "prompt_injection",
        "high",
        0.8,
        re.compile(r"reveal (your|the) (instructions|prompt|system|secrets)", re.IGNORECASE),
    ),
    (
        "prompt_injection",
        "medium",
        0.65,
        re.compile(r"(you are now (a|an|in) |new instructions?:|system prompt)", re.IGNORECASE),
    ),
    (
        "delimiter_escape",
        "high",
        0.9,
        re.compile(r"(<\s*/?\s*(system|assistant|im_start|im_end)\s*>|<\|im_(?:start|end)\|>|###\s*system\s*:)", re.IGNORECASE),
    ),
    (
        "context_poisoning",
        "high",
        0.9,
        re.compile(r"(\[system override\]|begin hidden instructions|end hidden instructions)", re.IGNORECASE),
    ),
    (
        "exfiltration",
        "high",
        0.85,
        re.compile(r"!\[[^\]]*\]\(https?://[^)]+\?(?:[^)]*(?:secret|token|key|data|q)=)[^)]*\)", re.IGNORECASE),
    ),
]


@dataclass(frozen=True)
class ThreatFinding:
    category: ThreatCategory
    severity: Severity
    weight: float
    snippet: str


@dataclass(frozen=True)
class InspectionResult:
    blocked: bool
    risk_score: float
    findings: list[ThreatFinding]
    normalized_text: str


def normalize_untrusted_text(text: str) -> str:
    """Strip zero-width obfuscation characters and apply Unicode NFKC normalization."""
    stripped = _ZERO_WIDTH_RE.sub("", text)
    return unicodedata.normalize("NFKC", stripped)


def _scan_rules(text: str) -> list[ThreatFinding]:
    findings: list[ThreatFinding] = []
    for category, severity, weight, pattern in _RULES:
        m = pattern.search(text)
        if m:
            findings.append(ThreatFinding(category=category, severity=severity, weight=weight, snippet=m.group(0)[:80]))
    return findings


def _scan_base64_blobs(text: str) -> list[ThreatFinding]:
    findings: list[ThreatFinding] = []
    for m in _BASE64_BLOB_RE.finditer(text):
        blob = m.group(0)
        try:
            decoded = base64.b64decode(blob, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if decoded and _scan_rules(normalize_untrusted_text(decoded)):
            findings.append(
                ThreatFinding(
                    category="obfuscated_payload",
                    severity="high",
                    weight=0.95,
                    snippet=f"base64:{decoded[:60]}",
                )
            )
    return findings


def inspect_text(text: str, block_threshold: float = 0.6) -> InspectionResult:
    """Inspect untrusted query or chunk text for prompt injection and context poisoning."""
    normalized = normalize_untrusted_text(text)
    findings = _scan_rules(normalized) + _scan_base64_blobs(normalized)
    risk = min(1.0, sum(f.weight for f in findings))
    return InspectionResult(
        blocked=risk >= block_threshold,
        risk_score=round(risk, 4),
        findings=findings,
        normalized_text=normalized,
    )


class PromptDefenseMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware blocking direct prompt-injection attempts on query endpoints."""

    def __init__(self, app, protected_paths: tuple[str, ...] = ("/query", "/query/stream"), block_threshold: float = 0.6) -> None:
        super().__init__(app)
        self.protected_paths = protected_paths
        self.block_threshold = block_threshold

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in self.protected_paths:
            body = await request.body()
            if body:
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
                if isinstance(payload, dict) and isinstance(payload.get("question"), str):
                    result = inspect_text(payload["question"], block_threshold=self.block_threshold)
                    if result.blocked:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "detail": "request blocked by prompt-injection defense",
                                "risk_score": result.risk_score,
                                "categories": sorted({f.category for f in result.findings}),
                            },
                            headers={"X-Threat-Blocked": "true"},
                        )
        return await call_next(request)
