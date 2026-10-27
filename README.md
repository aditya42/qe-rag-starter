# QE RAG Starter: 100% local Retrieval-Augmented Generation

A minimal, reproducible local RAG pipeline for QE runbooks, testing standards, and engineering docs.

## Architecture

Documents (.md/.txt/.pdf) → word-based overlapping chunks → Ollama Qwen3 Embedding 0.6B
→ ChromaDB cosine vector search → top-k supporting excerpts → Ollama Qwen3 4B → answer with references.

This is a learning/demo project, **not yet a security-hardened production service**.

## 1. Requirements

- Python 3.11 or 3.12, git, and local Ollama: https://ollama.com/download
- Enough RAM and disk to run the Qwen3 4B model (~2.5 GB model download) and
  the 0.6B embedding model (~639 MB download). Runtime memory is higher.

Install Ollama; make sure its local service is running. If necessary, run `ollama serve` in another terminal.

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
ollama list
```

## 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## 3. Ingest the sample documents

```bash
python rag.py ingest --docs data
```

The program reads supported document types recursively, extracts text, chunks it with
30-word overlap, generates embeddings with Ollama, and stores vectors and metadata
in `.chroma/`. Re-running `ingest` **replaces** this demo collection so deleted
files do not leave stale results.

To ingest your own documents:

```bash
mkdir -p my-docs
# Copy your .md, .txt, and text-based .pdf files into my-docs/
python rag.py ingest --docs my-docs
```

pypdf cannot read text from scanned/image-only PDFs without an external OCR step.

## 4. Retrieve and ask

```bash
python rag.py search "What do we do when a checkout fails?"
python rag.py ask "How do we handle flaky Playwright tests?"
python rag.py ask "What are the release quality gates?"
```

The answer is generated with instructions to cite numbered retrieved excerpts. The
CLI also prints the actual retrieved file sources as an audit trail. Note that **a
citation in the LLM's answer is not independently verified**; for high-assurance
systems add programmatic citation verification and human review.

Top-k and score filtering are configurable:

```bash
python rag.py search "checkout failure" --top-k 5 --min-score 0.30
```

Do not assume `0.30` is universally suitable. Evaluate retrieval using real
questions before selecting a score threshold.

## 5. Test and evaluate

```bash
python -m pytest -q
python rag.py eval --cases eval_questions.json
```

- Unit tests validate chunk boundaries, deterministic IDs, document paths, and prompt construction.
- `eval` checks **document-level Recall@k** on labelled queries. This is not a full
  answer-faithfulness or safety evaluation. Edit `eval_questions.json` with your
  own labelled examples for regression testing.

## 6. Optional model settings

```bash
export RAG_CHAT_MODEL=qwen3:4b
export RAG_EMBED_MODEL=qwen3-embedding:0.6b
export OLLAMA_URL=http://127.0.0.1:11434
```

**If you change the embedding model, run `ingest` again** before querying. The CLI
checks for this mismatch and refuses to search an incompatible index.

## What is deliberately out of scope?

- ACL-aware document retrieval and tenant isolation
- Encrypted secrets and private data governance
- Incremental indexing, distributed workers, OCR, rich tables, and versioned re-indexing
- Reranking, hybrid BM25+dense retrieval, and tuned evaluation/rejection thresholds
- Production health checks, API authentication, observability, and automatic citation verification

Before processing confidential client data, review the libraries/model licenses,
model downloads, logs, disk permissions, data retention, and who can access the local
service. Only ingest data you are permitted to process.

## AWS migration option

1. S3 replaces the local `data/` directory; configure access controls and encryption.
2. EventBridge/SQS coordinates upload events; Lambda or ECS/Fargate runs ingestion.
3. Use Amazon Bedrock embedding and generation models **or** keep self-hosted models on EC2/ECS.
4. Use a suitable persistent vector index, such as pgvector on RDS or OpenSearch,
   with metadata-based authorization checks.
5. Instrument relevance, grounding, latency, access decisions, model cost, and
   failure rates; gate release through evaluation tests.

AWS services have their own pricing; this architecture is **not guaranteed to fit AWS Free Tier**.

## References

- https://docs.ollama.com/api/chat
- https://ollama.com/library/qwen3:4b
- https://ollama.com/library/qwen3-embedding:0.6b
- https://docs.trychroma.com/reference/python/collection
- https://pypdf.readthedocs.io/en/stable/user/extract-text.html
