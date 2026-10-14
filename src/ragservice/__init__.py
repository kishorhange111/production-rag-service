"""ragservice: a production-style retrieval-augmented generation (RAG) service.

Request flow::

    question ──► Retriever ──► top-k chunks ──► Guardrails ──► Generator ──► answer + citations
                  │  BM25 (lexical)                (injection      (LLM with a
                  │  Dense (embeddings)             filter,         cite-your-sources
                  └► Reciprocal Rank Fusion          abstain if      prompt)
                     └► optional cross-encoder       low evidence)
                        reranker

Modules:
    chunking    split documents into overlapping, heading-aware chunks
    embeddings  embedding backends (sentence-transformers, or a hashing embedder for tests)
    index       BM25 and dense vector indexes, saved to and loaded from disk
    retrieval   hybrid retrieval with RRF fusion and reranking
    generation  prompt construction, LLM clients, citation parsing
    guardrails  prompt-injection screening of retrieved text, abstention
    pipeline    ties everything together (used by the API, CLI and evaluation)
    api         FastAPI app: /ingest, /query, /query/stream, /health, /metrics
    eval        retrieval and answer-quality metrics, plus the evaluation runner
"""

__version__ = "0.1.0"
