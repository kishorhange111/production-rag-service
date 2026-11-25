"""Query rewriting, HyDE (Hypothetical Document Embeddings), sub-query decomposition, and routing.

Raw user questions are often conversational, ambiguous, or multi-hop ("Compare backup
retention and on-call rotation"). Improving the query before retrieval raises recall
without touching the index:

1. **Query rewriting**: strips conversational filler and expands domain acronyms.
2. **Routing**: classifies query intent (``exact_lookup``, ``comparison``, ``procedural``,
   ``aggregate``, ``factual``) and picks the optimal retrieval mode (e.g. ``bm25`` for
   error codes / quoted identifiers, ``hybrid`` otherwise).
3. **Sub-query decomposition**: splits compound or comparative questions into focused
   atomic sub-queries, then fuses their results via Reciprocal Rank Fusion.
4. **HyDE**: synthesizes a short hypothetical answer passage (via an LLM when available,
   or a deterministic declarative transform offline) to bridge question-vs-passage
   embedding asymmetry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ragservice.acl import AccessContext, MetadataFilter
from ragservice.generation import LLM
from ragservice.retrieval import Hit, Mode, Retriever, reciprocal_rank_fusion

QueryRoute = Literal["factual", "comparison", "procedural", "aggregate", "exact_lookup"]

DEFAULT_ACRONYMS: dict[str, str] = {
    "k8s": "kubernetes",
    "db": "database",
    "pg": "postgres",
    "auth": "authentication",
    "authz": "authorization",
    "ssl": "tls ssl",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "slo": "service level objective",
    "sla": "service level agreement",
}

_FILLER_RE = re.compile(
    r"^(?:can you please|could you please|please|can you|could you|i want to know|tell me|explain to me)\s+",
    re.IGNORECASE,
)
_EXACT_CODE_RE = re.compile(r'(".*?"|`[^`]+`|\b[A-Z]{2,}_[A-Z0-9_]+\b|\b(?:ERR|HTTP|RFC)[-_ ]?\d+\b)')
_COMPARE_RE = re.compile(r"\b(compare|versus|vs\.?|difference between|differences between)\b", re.IGNORECASE)
_PROCEDURAL_RE = re.compile(r"\b(how (?:do|to|can)|steps to|procedure|runbook|deploy|configure|rollback)\b", re.IGNORECASE)
_AGGREGATE_RE = re.compile(r"\b(list all|summarize|summary|overview|all the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryPlan:
    """Structured query transformation plan produced before retrieval."""

    original: str
    rewritten: str
    sub_queries: list[str]
    hyde_document: str | None
    route: QueryRoute
    recommended_mode: Mode


def rewrite_query(question: str, acronyms: dict[str, str] | None = None) -> str:
    """Normalize whitespace, strip conversational filler, and expand known acronyms."""
    cleaned = " ".join(question.strip().split())
    cleaned = _FILLER_RE.sub("", cleaned).strip()
    mapping = {**DEFAULT_ACRONYMS, **(acronyms or {})}
    tokens = cleaned.split()
    expanded: list[str] = []
    for tok in tokens:
        bare = re.sub(r"[^a-zA-Z0-9]", "", tok).lower()
        expanded.append(tok)
        if bare in mapping and mapping[bare] not in cleaned.lower():
            expanded.append(mapping[bare])
    return " ".join(expanded)


def route_query(question: str) -> tuple[QueryRoute, Mode]:
    """Classify query intent and recommend a retrieval mode."""
    if _EXACT_CODE_RE.search(question):
        return "exact_lookup", "bm25"
    if _COMPARE_RE.search(question):
        return "comparison", "hybrid"
    if _PROCEDURAL_RE.search(question):
        return "procedural", "hybrid"
    if _AGGREGATE_RE.search(question):
        return "aggregate", "hybrid"
    return "factual", "hybrid"


def decompose_query(question: str, max_sub_queries: int = 4) -> list[str]:
    """Break compound or comparative questions into focused sub-queries."""
    q = " ".join(question.strip().split()).rstrip("?")
    if not q:
        return []

    # Pattern 1: "Compare X and Y" / "Difference between X and Y" / "X vs Y"
    m = re.search(r"(?:compare|differences? between)\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+)", q, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()][:max_sub_queries]

    m_vs = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", q, re.IGNORECASE)
    if m_vs:
        return [m_vs.group(1).strip(), m_vs.group(2).strip()][:max_sub_queries]

    # Pattern 2: Multi-clause "X and also Y" / "X; Y"
    parts = [p.strip() for p in re.split(r"\s*(?:;|\band also\b|\band what about\b|\band how\b)\s*", q, flags=re.IGNORECASE) if p.strip()]
    if len(parts) > 1:
        return parts[:max_sub_queries]

    return [q]


def generate_hyde(question: str, llm: LLM | None = None) -> str:
    """Produce a hypothetical answer passage for dense retrieval (LLM-backed or deterministic fallback)."""
    if llm is not None:
        messages = [
            {
                "role": "system",
                "content": "Write a concise, factual 2-sentence documentation excerpt that would directly answer the user's question.",
            },
            {"role": "user", "content": question},
        ]
        return llm.complete(messages, max_tokens=96).strip()

    # Deterministic offline HyDE: turn interrogative phrasing into a declarative documentation statement.
    core = re.sub(
        r"^(?:what|which|how|when|where|who|why)\s+(?:is|are|does|do|can|should|long are|many)\s+",
        "",
        question.strip().rstrip("?"),
        flags=re.IGNORECASE,
    )
    return f"Documentation summary: {core}."


class QueryProcessor:
    """Orchestrates query rewriting, routing, decomposition, HyDE, and multi-query retrieval."""

    def __init__(
        self,
        llm: LLM | None = None,
        enable_hyde: bool = False,
        acronyms: dict[str, str] | None = None,
        max_sub_queries: int = 4,
    ) -> None:
        self.llm = llm
        self.enable_hyde = enable_hyde
        self.acronyms = acronyms
        self.max_sub_queries = max_sub_queries

    def plan(self, question: str) -> QueryPlan:
        rewritten = rewrite_query(question, acronyms=self.acronyms)
        route, rec_mode = route_query(question)
        sub_queries = decompose_query(rewritten, max_sub_queries=self.max_sub_queries)
        hyde_doc = generate_hyde(rewritten, llm=self.llm) if self.enable_hyde else None
        return QueryPlan(
            original=question,
            rewritten=rewritten,
            sub_queries=sub_queries or [rewritten],
            hyde_document=hyde_doc,
            route=route,
            recommended_mode=rec_mode,
        )

    def multi_query_search(
        self,
        retriever: Retriever,
        question: str,
        k: int = 5,
        access: AccessContext | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[list[Hit], QueryPlan]:
        """Run all sub-queries (and optional HyDE) and fuse hits with RRF."""
        plan = self.plan(question)
        queries = list(plan.sub_queries)
        if plan.hyde_document:
            queries.append(plan.hyde_document)

        chunk_by_id: dict[str, Hit] = {}
        id_to_int: dict[str, int] = {}
        int_to_id: dict[int, str] = {}
        rankings: list[list[int]] = []

        for q in queries:
            hits = retriever.search(q, k=max(k * 2, 10), access=access, metadata_filter=metadata_filter)
            ranking: list[int] = []
            for h in hits:
                cid = h.chunk.chunk_id
                if cid not in id_to_int:
                    idx = len(id_to_int)
                    id_to_int[cid] = idx
                    int_to_id[idx] = cid
                    chunk_by_id[cid] = h
                ranking.append(id_to_int[cid])
            rankings.append(ranking)

        fused = reciprocal_rank_fusion(rankings, k=retriever.rrf_k)
        out = [
            Hit(
                chunk_by_id[int_to_id[idx]].chunk,
                score,
                chunk_by_id[int_to_id[idx]].bm25_rank,
                chunk_by_id[int_to_id[idx]].dense_rank,
                chunk_by_id[int_to_id[idx]].bm25_score,
                chunk_by_id[int_to_id[idx]].dense_score,
            )
            for idx, score in fused[:k]
        ]
        return out, plan
