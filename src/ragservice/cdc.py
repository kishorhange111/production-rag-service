"""Incremental CDC (Change Data Capture) deduplication and re-indexing queue.

Re-ingesting an entire corpus on every edit is expensive and wastes embedding compute.
This module tracks per-document content hashes and buffers upsert/delete change events
in a FIFO queue so a background worker (or API batch handler) can:

1. Skip unchanged documents in O(1) using SHA-256 content hashes.
2. Replace modified documents cleanly by removing stale chunks before indexing new ones.
3. Process tombstones (deletes) so deleted documents disappear from retrieval immediately.
4. Isolate per-document failures into a dead-letter queue instead of stalling the batch.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ragservice.chunking import Chunk, Document
from ragservice.index import KnowledgeBase
from ragservice.ingestion import ingest_document

ChangeOp = Literal["upsert", "delete"]


def compute_content_hash(doc: Document) -> str:
    """Deterministic SHA-256 digest of a document's text and metadata."""
    payload = json.dumps({"text": doc.text, "metadata": doc.metadata}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChangeEvent:
    """A single document mutation queued for incremental indexing."""

    event_id: str
    op: ChangeOp
    doc_id: str
    document: Document | None = None
    content_hash: str | None = None
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class ReindexStats:
    """Summary of one ``process_batch`` run."""

    processed: int = 0
    upserted: int = 0
    deleted: int = 0
    skipped_unchanged: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    failed: int = 0


class CDCReindexQueue:
    """In-memory CDC queue with content-hash deduplication and dead-letter capture."""

    def __init__(self, initial_hashes: dict[str, str] | None = None) -> None:
        self._queue: deque[ChangeEvent] = deque()
        self.doc_hashes: dict[str, str] = dict(initial_hashes or {})
        self.dead_letters: list[tuple[ChangeEvent, str]] = []

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def enqueue_upsert(self, doc: Document) -> ChangeEvent:
        event = ChangeEvent(
            event_id=uuid.uuid4().hex[:12],
            op="upsert",
            doc_id=doc.doc_id,
            document=doc,
            content_hash=compute_content_hash(doc),
        )
        self._queue.append(event)
        return event

    def enqueue_delete(self, doc_id: str) -> ChangeEvent:
        event = ChangeEvent(
            event_id=uuid.uuid4().hex[:12],
            op="delete",
            doc_id=doc_id,
        )
        self._queue.append(event)
        return event

    def process_batch(
        self,
        kb: KnowledgeBase,
        chunker: Callable[[Document], list[Chunk]] = ingest_document,
        max_events: int | None = None,
    ) -> ReindexStats:
        """Drain up to ``max_events`` from the queue and apply mutations to ``kb``."""
        stats = ReindexStats()
        limit = len(self._queue) if max_events is None else min(max_events, len(self._queue))
        for _ in range(limit):
            event = self._queue.popleft()
            stats.processed += 1
            try:
                if event.op == "delete":
                    removed = kb.remove_doc(event.doc_id)
                    self.doc_hashes.pop(event.doc_id, None)
                    stats.deleted += 1
                    stats.chunks_removed += removed
                elif event.op == "upsert":
                    if event.document is None:
                        raise ValueError("upsert event missing document payload")
                    if event.content_hash and self.doc_hashes.get(event.doc_id) == event.content_hash:
                        stats.skipped_unchanged += 1
                        continue
                    chunks = chunker(event.document)
                    removed, added = kb.upsert_doc(event.doc_id, chunks)
                    if event.content_hash:
                        self.doc_hashes[event.doc_id] = event.content_hash
                    stats.upserted += 1
                    stats.chunks_removed += removed
                    stats.chunks_added += added
            except Exception as exc:
                stats.failed += 1
                self.dead_letters.append((event, str(exc)))
        return stats
