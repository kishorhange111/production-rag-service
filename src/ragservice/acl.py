"""Multi-tenant ACL and metadata filter enforcement at retrieval time.

Enterprise RAG systems cannot rely on post-hoc LLM prompting to hide restricted
documents: if an unauthorized chunk enters the context window, the model can leak it.
This module enforces fail-closed authorization and metadata filtering *before* chunks
reach the generator:

1. **Tenant isolation**: chunks tagged with a ``tenant_id`` are visible only to matching
   tenants (unless explicitly marked ``tenant_id == "*"`` or ``visibility == "public"``).
2. **Role & principal ACLs**: ``allowed_roles`` / ``allowed_principals`` restrict access;
   ``denied_principals`` always takes precedence.
3. **Clearance levels**: ``min_clearance`` blocks principals below the required tier.
4. **Metadata filters**: exact-match (``equals``) and set-membership (``in_values``) filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragservice.chunking import Chunk


@dataclass(frozen=True)
class AccessContext:
    """Caller identity and security attributes evaluated during retrieval."""

    tenant_id: str
    principal_id: str | None = None
    roles: frozenset[str] = frozenset()
    clearance_level: int = 0


@dataclass(frozen=True)
class MetadataFilter:
    """Declarative metadata predicate applied to retrieved chunks."""

    equals: dict[str, Any] = field(default_factory=dict)
    in_values: dict[str, frozenset[Any]] = field(default_factory=dict)

    def matches(self, metadata: dict[str, Any]) -> bool:
        if any(metadata.get(key) != expected for key, expected in self.equals.items()):
            return False
        return all(metadata.get(key) in allowed for key, allowed in self.in_values.items())


def is_chunk_authorized(chunk: Chunk, access: AccessContext) -> bool:
    """Fail-closed authorization check for a single chunk."""
    meta = chunk.metadata

    denied = meta.get("denied_principals")
    if denied and access.principal_id and access.principal_id in denied:
        return False

    chunk_tenant = meta.get("tenant_id")
    visibility = meta.get("visibility")
    if chunk_tenant is not None:
        if chunk_tenant != "*" and chunk_tenant != access.tenant_id:
            return False
    elif visibility != "public":
        # Fail closed when caller supplies a tenant context and chunk has no tenant or public marker
        return False

    min_clearance = meta.get("min_clearance")
    if min_clearance is not None and access.clearance_level < int(min_clearance):
        return False

    allowed_roles = meta.get("allowed_roles")
    allowed_principals = meta.get("allowed_principals")
    if allowed_roles or allowed_principals:
        role_ok = bool(allowed_roles and (access.roles & set(allowed_roles)))
        principal_ok = bool(allowed_principals and access.principal_id and access.principal_id in allowed_principals)
        if not (role_ok or principal_ok):
            return False

    return True


def chunk_allowed(
    chunk: Chunk,
    access: AccessContext | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> bool:
    """Combined ACL + metadata filter check."""
    if access is not None and not is_chunk_authorized(chunk, access):
        return False
    return not (metadata_filter is not None and not metadata_filter.matches(chunk.metadata))
