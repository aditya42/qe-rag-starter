from pathlib import Path
import pytest
from rag_core import build_prompt, chunk_text, iter_chunks, make_chunk, source_label


def test_chunking_has_overlap_without_infinite_tail():
    words = " ".join(str(x) for x in range(11))
    result = chunk_text(words, size=5, overlap=2)
    assert result == ["0 1 2 3 4", "3 4 5 6 7", "6 7 8 9 10"]


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("one two", size=4, overlap=4)


def test_empty_document_makes_no_chunks():
    assert chunk_text("   ") == []


def test_stable_ids_change_with_text():
    original = make_chunk("runbook.md", 0, 1, "check traces")
    assert original.id == make_chunk("runbook.md", 0, 1, "check traces").id
    assert original.id != make_chunk("runbook.md", 0, 1, "check logs").id


def test_markdown_ingestion_keeps_relative_source(tmp_path: Path):
    nested = tmp_path / "qa"
    nested.mkdir()
    (nested / "guide.md").write_text("playwright locator auto wait", encoding="utf-8")
    chunks = list(iter_chunks(tmp_path))
    assert len(chunks) == 1
    assert chunks[0].source == "qa/guide.md"
    assert chunks[0].page == 0


def test_source_label_handles_pdf_pages():
    assert source_label("guide.pdf", 3) == "guide.pdf, page 3"
    assert source_label("guide.md", 0) == "guide.md"


def test_prompt_includes_citation_reference_and_fallback():
    p = build_prompt("What now?", [{"source":"runbook.md", "page":0, "text":"Check traces"}])
    assert "[1] runbook.md" in p
    assert "I don't know based on the provided documents" in p
    assert "Check traces" in p
