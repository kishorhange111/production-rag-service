"""Text embedding backends.

An embedder maps text to a vector such that semantically similar texts get
nearby vectors (high cosine similarity). All vectors returned here are
L2-normalised, so cosine similarity is just a dot product.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything with ``embed_documents`` / ``embed_query`` returning normalised float32 arrays."""

    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model, e.g. ``BAAI/bge-base-en-v1.5``.

    BGE models are trained with an instruction prefix on the *query* side
    only. Leaving it out costs a few points of retrieval quality, so the prefix
    is applied automatically for BGE.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
    ):
        from sentence_transformers import (
            SentenceTransformer,  # heavy import, only when used
        )

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self.model.encode([self.query_prefix + text], convert_to_numpy=True))[0]


class HashingEmbedder:
    """A dependency-free embedder: hashed bag of words and word bigrams.

    It has no semantic understanding (synonyms don't match), but it is
    deterministic, instant and needs no model download, so unit tests and the
    Docker smoke test can exercise the whole dense-retrieval path.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:4], "little")
            # The sign bit reduces the bias from hash collisions (the "hashing trick").
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vector(text)[None])[0]


def build_embedder(name: str) -> Embedder:
    """``"hashing"`` gives the test embedder; anything else is a sentence-transformers model name."""
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)
