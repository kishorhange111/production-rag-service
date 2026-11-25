"""Unit tests for multi-format ingestion: HTML, CSV, and table-aware Markdown chunking."""

from __future__ import annotations

import pytest

from ragservice.chunking import Document
from ragservice.embeddings import HashingEmbedder
from ragservice.generation import ExtractiveLLM
from ragservice.ingestion import (
    chunk_csv,
    chunk_markdown_table_aware,
    detect_format,
    html_to_markdown,
    ingest_document,
    load_documents_from_directory,
)
from ragservice.pipeline import RAGPipeline


def test_detect_format_by_hint_extension_and_content():
    assert detect_format("doc", "plain text", explicit_format="csv") == "csv"
    assert detect_format("page.html", "hello") == "html"
    assert detect_format("metrics.csv", "a,b\n1,2") == "csv"
    assert detect_format("guide.md", "plain") == "markdown"
    assert detect_format("blob", "<html><body><h1>Title</h1></body></html>") == "html"
    assert detect_format("blob", "# Heading\n\nBody") == "markdown"
    assert detect_format("blob", "just plain text") == "text"


def test_html_to_markdown_strips_scripts_and_preserves_headings_and_tables():
    html = """
    <html>
      <head><style>.x{color:red}</style><script>alert('xss')</script></head>
      <body>
        <nav>Home | Pricing</nav>
        <h1>Service Ports</h1>
        <p>Core services run behind Envoy.</p>
        <table>
          <tr><th>Service</th><th>Port</th></tr>
          <tr><td>Postgres</td><td>5432</td></tr>
          <tr><td>Redis</td><td>6379</td></tr>
        </table>
      </body>
    </html>
    """
    md = html_to_markdown(html)
    assert "alert" not in md and "color:red" not in md and "Pricing" not in md
    assert "# Service Ports" in md
    assert "| Service | Port |" in md
    assert "| Postgres | 5432 |" in md


def test_csv_chunker_repeats_headers_across_batches():
    rows = ["service,port,owner"] + [f"svc-{i},{8000 + i},team-{i % 3}" for i in range(12)]
    doc = Document("inventory.csv", "\n".join(rows))
    chunks = chunk_csv(doc, max_rows_per_chunk=5)
    assert len(chunks) == 3
    for chunk in chunks:
        assert "| service | port | owner |" in chunk.text
        assert chunk.metadata["content_type"] == "table"
        assert chunk.metadata["format"] == "csv"
    assert chunks[0].metadata["row_range"] == "1-5"
    assert chunks[2].metadata["row_range"] == "11-12"


def test_csv_chunker_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        chunk_csv(Document("a.csv", "a,b\n1,2"), max_rows_per_chunk=0)


def test_markdown_table_aware_chunking_splits_large_tables_with_headers():
    table_rows = "\n".join(f"| svc-{i} | {9000 + i} |" for i in range(7))
    md = f"# Infra\n\nIntro paragraph about services.\n\n| Service | Port |\n| --- | --- |\n{table_rows}\n\nClosing note."
    chunks = chunk_markdown_table_aware(Document("infra.md", md), max_words=200, overlap_words=40, max_table_rows=3)
    table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
    prose_chunks = [c for c in chunks if c.metadata.get("content_type") == "prose"]
    assert len(table_chunks) == 3
    assert all("| Service | Port |" in c.text for c in table_chunks)
    assert all(c.heading == "Infra" for c in chunks)
    assert len(prose_chunks) == 2


def test_ingest_document_and_directory_end_to_end(tmp_path):
    (tmp_path / "runbook.html").write_text("<h1>Backups</h1><p>Snapshots are retained for 90 days.</p>", encoding="utf-8")
    (tmp_path / "ports.csv").write_text("service,port\npgbouncer,6432\nredis,6379\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes\n\nAlways verify checksums.", encoding="utf-8")

    docs = load_documents_from_directory(tmp_path)
    assert {d.doc_id for d in docs} == {"runbook.html", "ports.csv", "notes.md"}

    pipe = RAGPipeline(HashingEmbedder(), ExtractiveLLM(), top_k=3)
    added = pipe.ingest_directory(tmp_path)
    assert added == 3
    ans = pipe.answer("How long are snapshots retained?")
    assert "90 days" in ans.answer and not ans.abstained

    csv_chunks = ingest_document(Document("ports.csv", "service,port\npgbouncer,6432\n"))
    assert len(csv_chunks) == 1 and "pgbouncer" in csv_chunks[0].text
