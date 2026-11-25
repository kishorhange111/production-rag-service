# ADR-001: Hybrid BM25 + Dense Retrieval with Weighted RRF and Linear Fusion

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
Enterprise technical corpora mix exact identifiers (ports like `6432`, error codes like `ERR_SSL_PROTOCOL`, tool names like `Alembic`) with conceptual questions ("how do we safely release schema changes?"). Pure dense retrieval misses rare exact tokens; pure BM25 misses paraphrases.

## Decision
1. Maintain aligned `BM25Index` and `DenseIndex` inside `KnowledgeBase` (`src/ragservice/index.py`).
2. Default to **Reciprocal Rank Fusion (RRF)** (`k=60`, configurable per-retriever weights) because rank fusion needs no cross-index score calibration.
3. Provide **min-max normalized linear fusion** (`alpha * dense + (1 - alpha) * bm25`), per-document diversity capping (`max_per_doc`), and an optional calibrated reranker (`LexicalSemanticReranker` / `CrossEncoderReranker` + `CalibratedReranker`).

## Consequences
- Strong recall across both exact-token and semantic queries with zero external vector DB required for single-node or replicated read-only deployments.
- Single-process memory index scales horizontally via read-only replicas loading a shared saved index; live multi-replica writes require swapping `KnowledgeBase` for pgvector/Qdrant behind the same `Retriever` interface.
