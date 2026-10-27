"""Mocked pipeline contract checks; live Ollama/Chroma integration is a local smoke test."""
from pathlib import Path
import sys
from types import SimpleNamespace

import rag


class FakeCollection:
    def __init__(self):
        self.items = []

    def count(self):
        return len(self.items)

    def add(self, ids, documents, metadatas, embeddings):
        self.items.extend(zip(ids, documents, metadatas, embeddings))

    def peek(self, limit=1):
        return {"metadatas": [t[2] for t in self.items[:limit]]}

    def query(self, query_embeddings, n_results, include):
        assert query_embeddings == [[0.1, 0.2]]
        selected = self.items[:n_results]
        return {"documents": [[t[1] for t in selected]],
                "metadatas": [[t[2] for t in selected]],
                "distances": [[0.11 for _ in selected]]}


class FakeClient:
    collections = {}

    def __init__(self, path):
        self.path = path

    def list_collections(self):
        return [SimpleNamespace(name=k) for k in self.collections]

    def delete_collection(self, name):
        del self.collections[name]

    def create_collection(self, name, configuration):
        assert configuration["hnsw"]["space"] == "cosine"
        collection = FakeCollection()
        self.collections[name] = collection
        return collection

    def get_collection(self, name):
        return self.collections[name]


def test_ingest_retrieve_and_answer_with_mocked_models(tmp_path: Path, monkeypatch):
    FakeClient.collections = {}
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "quality.md").write_text("Use Playwright traces to debug flaky tests", encoding="utf-8")
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])
    count = rag.ingest(docs, tmp_path / "db", chunk_size=20, overlap=3)
    assert count == 1
    retrieved = rag.retrieve("How to debug flaky tests?", tmp_path / "db")
    assert retrieved[0]["source"] == "quality.md"
    assert retrieved[0]["score"] > 0.8
    monkeypatch.setattr(rag, "post_ollama", lambda endpoint, data: {
        "message": {"content": "Inspect the trace [1]."}
    })
    assert rag.ask("How to debug flaky tests?", retrieved) == "Inspect the trace [1]."


def test_rejection_when_no_match_passes_threshold(tmp_path: Path, monkeypatch):
    FakeClient.collections = {}
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "quality.md").write_text("Check traces", encoding="utf-8")
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])
    rag.ingest(docs, tmp_path / "db", 20, 3)
    retrieved = rag.retrieve("Unrelated query", tmp_path / "db", min_score=0.95)
    assert retrieved == []
    assert rag.ask("Unrelated query", retrieved) == "I don't know based on the provided documents."
