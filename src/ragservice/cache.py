"""Semantic query cache with TTL, LRU eviction, and strict tenant/ACL isolation.

LLM generation is the dominant latency and cost driver in RAG. Caching answers by
embedding similarity cuts repeat-query latency to sub-millisecond, *provided* that
isolation is airtight: a cached answer generated for Tenant A (or role ``admin``)
must never be served to Tenant B (or role ``guest``).

Every cache lookup is partitioned by a canonical ``namespace`` derived from
``(tenant_id, sorted(roles), clearance_level, metadata_filter)``, and entries
expire automatically after ``ttl_seconds`` or on corpus invalidation.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragservice.acl import AccessContext, MetadataFilter
from ragservice.embeddings import Embedder


def build_cache_namespace(
    access: AccessContext | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> str:
    """Deterministic isolation key combining tenant, roles, clearance, and metadata filters."""
    payload: dict[str, Any] = {
        "tenant": access.tenant_id if access else "__default__",
        "principal": access.principal_id if access else None,
        "roles": sorted(access.roles) if access else [],
        "clearance": access.clearance_level if access else 0,
        "eq": dict(sorted(metadata_filter.equals.items())) if metadata_filter else {},
        "in": {k: sorted(v) for k, v in sorted(metadata_filter.in_values.items())} if metadata_filter else {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass
class _CacheEntry:
    namespace: str
    tenant_id: str
    normalized_question: str
    embedding: np.ndarray
    value: Any
    created_at: float
    expires_at: float
    hit_count: int = 0


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    expirations: int
    size: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SemanticQueryCache:
    """Tenant-isolated semantic cache with TTL and bounded LRU capacity."""

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _prune_expired(self, now: float) -> None:
        expired_keys = [k for k, e in self._entries.items() if e.expires_at <= now]
        for k in expired_keys:
            del self._entries[k]
            self._expirations += 1

    @staticmethod
    def _norm(question: str) -> str:
        return " ".join(question.strip().lower().split())

    def get(
        self,
        question: str,
        access: AccessContext | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> Any | None:
        now = self._clock()
        self._prune_expired(now)
        ns = build_cache_namespace(access, metadata_filter)
        norm_q = self._norm(question)

        # Fast path: exact normalized question match inside the same namespace.
        exact_key = f"{ns}::{norm_q}"
        if exact_key in self._entries:
            entry = self._entries[exact_key]
            self._entries.move_to_end(exact_key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

        # Semantic match within the same namespace only.
        q_vec = self.embedder.embed_query(question)
        best_key: str | None = None
        best_sim = -1.0
        for key, entry in self._entries.items():
            if entry.namespace != ns:
                continue
            sim = float(entry.embedding @ q_vec)
            if sim >= self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None:
            entry = self._entries[best_key]
            self._entries.move_to_end(best_key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

        self._misses += 1
        return None

    def put(
        self,
        question: str,
        value: Any,
        access: AccessContext | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> None:
        now = self._clock()
        self._prune_expired(now)
        ns = build_cache_namespace(access, metadata_filter)
        norm_q = self._norm(question)
        key = f"{ns}::{norm_q}"
        entry = _CacheEntry(
            namespace=ns,
            tenant_id=access.tenant_id if access else "__default__",
            normalized_question=norm_q,
            embedding=self.embedder.embed_query(question),
            value=value,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = entry
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def invalidate_tenant(self, tenant_id: str) -> int:
        keys = [k for k, e in self._entries.items() if e.tenant_id == tenant_id]
        for k in keys:
            del self._entries[k]
        return len(keys)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> CacheStats:
        self._prune_expired(self._clock())
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            expirations=self._expirations,
            size=len(self._entries),
        )
