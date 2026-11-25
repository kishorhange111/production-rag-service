# ADR-004: Deterministic CI Evaluation Regression Gate

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
RAG quality degrades silently when chunking, tokenization, fusion weights, or guardrails change. GPU-only benchmarks (e.g. SQuAD 2.0 with 7B LLMs) are too slow and expensive to run on every pull request.

## Decision
Maintain a two-tier evaluation strategy:
1. **PR-blocking CPU regression gate** (`.github/workflows/eval-gate.yml`, `src/ragservice/eval/gate.py`): runs `run_eval_suite` on versioned golden QA fixtures (`data/eval/golden_qa.json`) on every PR and fails CI if context recall, faithfulness, citation precision, answer match, or abstention metrics drop below thresholds.
2. **Offline GPU benchmark** (`src/ragservice/eval/runner.py`, `scripts/colab_job.sh`): runs full SQuAD 2.0 ablations with BGE embeddings, cross-encoder reranking, and LLM-as-judge faithfulness with bootstrap 95% CIs.

## Consequences
- Every pull request is continuously verified for retrieval, citation, grounding, and abstention regressions in under 5 seconds with zero paid API keys.
