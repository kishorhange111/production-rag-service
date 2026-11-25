# ADR-003: Layered Prompt-Injection Defense and Deterministic Grounding Verification

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
RAG services face two distinct trust boundaries: untrusted caller queries (direct prompt injection) and untrusted retrieved documents (indirect prompt injection / context poisoning). Additionally, LLMs can hallucinate numbers or cite out-of-range sources even when instructed to ground answers.

## Decision
1. **Ingress & Context Defense** (`src/ragservice/security.py`, `src/ragservice/guardrails.py`): strip zero-width obfuscation characters, apply Unicode NFKC normalization, inspect embedded base64 payloads, and run high-precision rules across query ingress (`PromptDefenseMiddleware` -> HTTP 400) and retrieved chunks (`screen_hits`).
2. **Deterministic Grounding Verifier** (`src/ragservice/citations.py`): extract sentence-level claims and `[n]` citations, flag out-of-range references, and verify content-token + strict numeric consistency against cited chunks.

## Consequences
- Catches obfuscated and base64-encoded injection payloads and numeric hallucinations deterministically on CPU with <1 ms overhead.
- Deterministic lexical/numeric checks complement (rather than replace) an LLM-as-judge for nuanced semantic entailment, which remains supported via `ragservice.eval.runner`.
