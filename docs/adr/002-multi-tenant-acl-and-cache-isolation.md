# ADR-002: Fail-Closed Multi-Tenant ACLs and Isolated Semantic Caching

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
Relying on LLM system prompts to withhold unauthorized documents fails under indirect prompt injection and leaks sensitive tenant data. Similarly, sharing a semantic query cache across tenants or roles would serve Tenant A's answers to Tenant B.

## Decision
1. Enforce **retrieval-time, fail-closed authorization** (`src/ragservice/acl.py`) checking `tenant_id`, `allowed_roles`, `allowed_principals`, `denied_principals`, and `min_clearance` *before* chunks enter the prompt. When ACLs or metadata filters are active, candidate fetch expands so unauthorized high-scoring chunks never crowd out authorized ones.
2. Partition `SemanticQueryCache` (`src/ragservice/cache.py`) by a canonical namespace hash of `(tenant_id, principal_id, sorted(roles), clearance_level, metadata_filter)`, with TTL expiration, LRU eviction, and automatic invalidation on corpus ingestion.

## Consequences
- Zero cross-tenant leakage in both live retrieval and cached responses.
- Slightly higher per-query filtering cost on large in-memory corpora when ACLs are active, acceptable for single-node scale and directly pushable into SQL/vector pre-filters when backed by pgvector/Qdrant.
