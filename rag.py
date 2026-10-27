"""Local RAG demo: Ollama embeddings -> Chroma -> Ollama grounded answer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from rag_core import build_prompt, iter_chunks, source_label

COLLECTION = "qe_knowledge"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def post_ollama(endpoint: str, payload: dict) -> dict:
    import requests
    try:
        response = requests.post(f"{OLLAMA_URL}{endpoint}", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Ollama is unavailable. Start Ollama (or run `ollama serve`).") from exc
    except requests.exceptions.HTTPError as exc:
        msg = response.text[:500]
        raise RuntimeError(f"Ollama request failed: {msg}. Did you `ollama pull` the models?") from exc
    except (ValueError, requests.exceptions.RequestException) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    result = post_ollama("/api/embed", {"model": EMBED_MODEL, "input": texts})
    vectors = result["embeddings"]
    if len(vectors) != len(texts):
        raise RuntimeError("Ollama returned an unexpected embedding count")
    return vectors


def collection_at(db_path: Path, reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(db_path))
    if reset:
        names = {getattr(c, "name", c) for c in client.list_collections()}
        if COLLECTION in names:
            client.delete_collection(name=COLLECTION)
        return client.create_collection(
            name=COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
        )
    try:
        return client.get_collection(name=COLLECTION)
    except Exception as exc:
        raise RuntimeError("No index yet. First run: python rag.py ingest") from exc


def ingest(docs: Path, db_path: Path, chunk_size: int, overlap: int) -> int:
    chunks = list(iter_chunks(docs, size=chunk_size, overlap=overlap))
    if not chunks:
        raise RuntimeError(f"No extractable text in {docs}. Use PDF, .md or .txt files.")

    # Embed before replacing the old index, so a model error does not erase it.
    vectors = []
    batch_size = 12
    for offset in range(0, len(chunks), batch_size):
        group = chunks[offset:offset + batch_size]
        vectors.extend(embed_texts([c.text for c in group]))
        print(f"Embedded {len(vectors)}/{len(chunks)} chunks", file=sys.stderr)

    collection = collection_at(db_path, reset=True)
    for offset in range(0, len(chunks), 100):
        group = chunks[offset:offset + 100]
        collection.add(
            ids=[c.id for c in group],
            documents=[c.text for c in group],
            metadatas=[{"source": c.source, "page": c.page, "embed_model": EMBED_MODEL}
                       for c in group],
            embeddings=vectors[offset:offset + 100],
        )
    return len(chunks)


def retrieve(question: str, db_path: Path, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
    if top_k < 1 or not 0.0 <= min_score <= 1.0:
        raise ValueError("top-k must be >= 1 and min-score must be between 0 and 1")
    collection = collection_at(db_path)
    count = collection.count()
    if count == 0:
        raise RuntimeError("The index is empty; re-run ingestion")
    stored = collection.peek(limit=1)["metadatas"]
    model_at_index = stored[0].get("embed_model") if stored else None
    if model_at_index != EMBED_MODEL:
        raise RuntimeError(
            f"Index uses {model_at_index}; current embedding model is {EMBED_MODEL}. "
            "Re-index before querying."
        )
    embedding = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        if score >= min_score:
            matches.append({"source": metadata["source"], "page": metadata["page"],
                            "text": text, "score": score})
    return matches


def ask(question: str, matches: list[dict]) -> str:
    if not matches:
        return "I don't know based on the provided documents."
    result = post_ollama("/api/chat", {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful Quality Engineering knowledge assistant."},
            {"role": "user", "content": build_prompt(question, matches)},
        ],
        "think": False,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
    })
    return result["message"]["content"].strip()


def run_eval(path: Path, db: Path, top_k: int) -> int:
    cases = json.loads(path.read_text(encoding="utf-8"))
    hits = 0
    for case in cases:
        found = retrieve(case["question"], db, top_k=top_k)
        sources = {match["source"] for match in found}
        hit = any(expected in sources for expected in case["expected_sources"])
        hits += int(hit)
        print(f"{'PASS' if hit else 'FAIL'} | {case['question']}")
        print(f"   Retrieved: {sorted(sources)}")
    print(f"Recall@{top_k} on labelled queries: {hits}/{len(cases)} = {hits/len(cases):.1%}")
    return 0 if hits == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Small, local RAG pipeline")
    parser.add_argument("--db", type=Path, default=Path(".chroma"))
    sub = parser.add_subparsers(dest="command", required=True)
    index_cmd = sub.add_parser("ingest", help="Index documents; replaces this demo index")
    index_cmd.add_argument("--docs", type=Path, default=Path("data"))
    index_cmd.add_argument("--chunk-size", type=int, default=160)
    index_cmd.add_argument("--overlap", type=int, default=30)
    for name in ("ask", "search"):
        cmd = sub.add_parser(name)
        cmd.add_argument("question")
        cmd.add_argument("--top-k", type=int, default=3)
        cmd.add_argument("--min-score", type=float, default=0.0,
                         help="Experimental cosine score cutoff; calibrate with evaluation data")
    eval_cmd = sub.add_parser("eval", help="Evaluate retrieval on labelled questions")
    eval_cmd.add_argument("--cases", type=Path, default=Path("eval_questions.json"))
    eval_cmd.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            print(f"Indexed {ingest(args.docs, args.db, args.chunk_size, args.overlap)} chunks.")
        elif args.command == "eval":
            return run_eval(args.cases, args.db, args.top_k)
        else:
            matches = retrieve(args.question, args.db, args.top_k, args.min_score)
            if args.command == "ask":
                print("\nAnswer:\n" + ask(args.question, matches) + "\n")
            print("Retrieved sources:")
            if not matches:
                print("  None above the configured similarity threshold")
            for index, m in enumerate(matches, 1):
                print(f"  [{index}] {source_label(m['source'], m['page'])} "
                      f"(cosine similarity ~{m['score']:.3f})")
                if args.command == "search":
                    print(f"      {m['text'][:250]}...")
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
