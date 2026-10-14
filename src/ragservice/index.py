"""Search indexes: BM25 for keywords and a dense vector index for meaning.

Why both? They fail in different ways:

* **BM25** matches exact terms. It is great for names, error codes and IDs
  ("ERR_SSL_PROTOCOL", "RFC 9110"), but blind to paraphrase ("car" vs "automobile").
* **Dense** retrieval matches meaning, but can miss rare exact tokens the
  embedding model never learned.

Hybrid retrieval (see ``retrieval.py``) combines them.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ragservice.chunking import Chunk
from ragservice.embeddings import Embedder

# A small English stop-word list: these words carry no topical signal and only add noise to BM25.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stop words; keeps numbers and identifiers like 'v2' or 'http2'."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25, the classic lexical ranking function behind Elasticsearch and Lucene.

    score(q, d) = sum over query terms t of
        IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    * IDF: rare terms count more than common ones.
    * tf saturation (k1): the 10th occurrence of a word adds much less than the 1st.
    * length normalisation (b): long documents don't win just by containing more words.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tfs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()  # number of documents containing each term

    def add(self, texts: list[str]) -> None:
        for text in texts:
            tf = Counter(tokenize(text))
            self.doc_tfs.append(tf)
            self.doc_lens.append(sum(tf.values()))
            self.df.update(tf.keys())

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (position, score) pairs for ``query``; positions index the chunks added so far."""
        n = len(self.doc_tfs)
        if n == 0:
            return []
        avgdl = sum(self.doc_lens) / n
        scores = np.zeros(n, dtype=np.float32)
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # the "+1" keeps IDF positive for very common terms
            for i, tf in enumerate(self.doc_tfs):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / avgdl)
                    scores[i] += idf * f * (self.k1 + 1) / denom
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class DenseIndex:
    """Exact cosine-similarity search over normalised embeddings (a matrix product).

    Brute force is exact and fast enough up to ~1M chunks on one machine (1M x
    768 floats = 3 GB). Beyond that, swap in FAISS/HNSW; the interface stays the same.
    """

    def __init__(self, dim: int):
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.vectors) == 0:
            return []
        sims = self.vectors @ query_vec  # cosine similarity, because both sides are unit vectors
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]  # O(n) selection, then sort only the k winners
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]


class KnowledgeBase:
    """Chunks plus both indexes, kept aligned by position, with save/load to a directory."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.bm25 = BM25Index()
        self.dense = DenseIndex(embedder.dim)

    def add(self, chunks: list[Chunk]) -> int:
        """Index new chunks; chunks whose id is already present are skipped (idempotent ingestion)."""
        known = {c.chunk_id for c in self.chunks}
        new = [c for c in chunks if c.chunk_id not in known]
        if new:
            texts = [c.text_for_index for c in new]
            self.chunks.extend(new)
            self.bm25.add(texts)
            self.dense.add(self.embedder.embed_documents(texts))
        return len(new)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in self.chunks))
        np.save(d / "vectors.npy", self.dense.vectors)

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> KnowledgeBase:
        """Reload chunks and vectors. BM25 statistics are cheap to rebuild, so they aren't stored."""
        d = Path(directory)
        kb = cls(embedder)
        kb.chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text().splitlines() if line]
        kb.bm25.add([c.text_for_index for c in kb.chunks])
        kb.dense.vectors = np.load(d / "vectors.npy")
        if kb.dense.vectors.shape[1] != embedder.dim:
            raise ValueError("index was built with a different embedding model (dimension mismatch)")
        return kb
