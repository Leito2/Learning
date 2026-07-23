# 🎨 Welcome to ChromaDB — The Prototype Vector Database That Grew Up

ChromaDB is the most popular vector database in the Python ecosystem for one reason: **zero-friction developer experience**. `pip install chromadb`, `chroma.Client()`, three lines of code, and you have a working vector database with semantic search. That same simplicity is what makes it the default choice for **prototypes, Jupyter notebooks, agent demos, and single-developer RAG projects** — and why your vault's [[../01 - Vector Search Fundamentals.md|Vector Search Fundamentals]] and [[../../06 - Large Language Models/12 - Production RAG/02 - Vector Databases for RAG - HNSW, IVF, PQ and Filtering.md|Vector Databases for RAG]] notes both point to it as the prototyping standard.

Until now, the vault's coverage of Chroma was **scattered across 12+ mentions** (mostly as a comparison reference in Qdrant, Milvus, and Pinecone notes) with **zero dedicated notes**. That is a real gap for two reasons: (1) every portfolio project starts with Chroma and 30% of portfolio submissions get rejected at "production-readiness" because the migration to Qdrant/pgvector didn't happen, and (2) Chroma's recent releases (v0.5+) added a **native server mode**, multi-modal embeddings, and operator-rich metadata filtering that change the production calculus for small-to-medium deployments.

This course is the **dedicated deep dive** that closes the gap. We cover Chroma fundamentals (collections, embeddings, query), the in-memory vs DuckDB vs server-mode deployment choices, custom embedding functions for any model (HuggingFace, OpenAI, Cohere, multimodal), the rich `where` and `where_document` filtering primitives, and the migration path to Qdrant/pgvector when scale demands it. By the end, you will know exactly when to start with Chroma, when to keep it, and when to migrate.

## 🎯 Learning Objectives

- Pick the right Chroma deployment mode: in-memory, DuckDB (persistent local), or native server.
- Implement collections with custom embedding functions for any model.
- Write rich `where` metadata filters and `where_document` content filters.
- Use Chroma's multimodal embeddings (text + image).
- Build a FastAPI RAG endpoint backed by Chroma.
- Decide when to migrate to Qdrant/pgvector and how to do it without a rewrite.

## Course Map

| # | Note | Core Concept | Portfolio Hook |
|:-:|------|--------------|----------------|
| 00 | [[00 - Welcome to ChromaDB\|You are here]] | Why Chroma matters for prototyping | Course map |
| 01 | [[01 - Chroma Fundamentals - Collections, Embeddings and Query\|Fundamentals]] | `Client()`, `get_or_create_collection()`, `add/query` | Any RAG prototype |
| 02 | [[02 - Chroma Server Mode - Local and Docker Deployment\|Server Mode]] | `chroma run`, persistent storage, multi-process access | Production RAG lite |
| 03 | [[03 - Custom Embedding Functions - HuggingFace OpenAI Cohere Multimodal\|Custom Embeddings]] | `EmbeddingFunction` Protocol, multi-modal (CLIP) | Image-augmented RAG |
| 04 | [[04 - Metadata Filtering - Where Clauses and Operators\|Filtering]] | `where={"$eq": ...}`, `where_document={"$contains": ...}` | Permission-aware RAG |
| 05 | [[05 - Migration Path - Chroma to Qdrant and pgvector\|Migration]] | Data export → Qdrant / pgvector re-import | Production scaling |

## Why This Course Exists

Your audit found three critical problems:

1. **No dedicated Chroma note.** Every Chroma reference is a paragraph inside a comparison table or project guide. There is no single document that walks through `Client()`, `Collection`, embeddings, and `query()` coherently.
2. **No coverage of Chroma's 2024-2026 features.** Chroma v0.5+ added native server mode (`chroma run`), `PersistentClient()` (replacing the deprecated `Settings()`), multi-modal embedding support, and a much richer filtering API. None of this is in the vault.
3. **No migration playbook.** Most "Chroma for production" articles assume Chroma scales forever. It doesn't. Knowing how to migrate a Chroma collection to Qdrant without rewriting the application is a real skill.

## Where Chroma Fits (and Doesn't)

**Chroma is the right choice for:**
- Single-developer RAG prototypes.
- Jupyter notebooks and local demos.
- Agent evaluation harnesses with <1M vectors.
- "Spin up a server" scenarios where the dev experience matters more than peak QPS.
- Production workloads at <5M vectors with <100 QPS and single-region deploy.

**Chroma is the wrong choice for:**
- Production scale >10M vectors with >100 QPS.
- Distributed multi-region replication.
- Sub-10ms p99 latency SLAs.
- Strict separation of read-replica traffic from write traffic.

The cutoff is fuzzy because it depends on vector size, distance metric, and hardware. The decision matrix in [[../09 - Vector Database Comparison Matrix.md|note 09]] formalizes the comparison. This course teaches you to **start with Chroma** and migrate when metrics demand it.

## The Chroma Philosophy

Chroma's design bets:

1. **First-class embedding functions.** Embeddings are not an afterthought; they're first-class objects you can swap. Default is `all-MiniLM-L6-v2` (~80MB, CPU-runnable, ~100ns/token).
2. **SQLite-compatible storage.** The local mode uses DuckDB (a columnar SQLite-compatible engine) so you can `SELECT * FROM collections` directly.
3. **Client/server symmetry.** `Client()` works in-process; `HttpClient()` connects to a server. Same API; you decide later.
4. **Filtering done in metadata, not vectors.** `where={"user_id": "u-42"}` is a metadata filter; vector search happens within the filtered subset.

These bets make Chroma the **most ergonomic** vector DB at small scale. They also mean Chroma cannot match Qdrant or Milvus at large scale, which have spent years optimizing HNSW + IVF-PQ at 1B+ vectors.

## Prerequisites

- **Python 3.10+** with `pip install chromadb`.
- **Vector Search Fundamentals (10/33/01)**: ANN, embeddings, distance metrics.
- **Production RAG (06/12)** and **Production RAG (06/13)**: end-to-end RAG patterns.
- **Qdrant I (10/33/05)**: the Qdrant counterpart this course migrates toward.

## How to Read This Course

1. **Notes 01-02** are the on-ramp: client setup, collections, query, and deployment modes.
2. **Notes 03-04** are the productive patterns: custom embeddings and rich filtering.
3. **Note 05** is the growth path: when and how to migrate to Qdrant/pgvector.

## 📦 Compression Code

```python
# 📦 Welcome - The 30-second Chroma demo
# A complete RAG prototype in 30 lines.
# pip install chromadb

import chromadb

# 1. Persistent client (data survives restarts)
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Collection with default embedding (all-MiniLM-L6-v2)
collection = client.get_or_create_collection(name="docs")

# 3. Add documents (Chroma embeds automatically)
collection.add(
    documents=[
        "LangGraph is a stateful agent framework",
        "ChromaDB is a vector database",
        "Qdrant is a Rust-based vector search engine",
    ],
    ids=["doc-1", "doc-2", "doc-3"],
    metadatas=[
        {"source": "langgraph", "category": "agents"},
        {"source": "chroma", "category": "databases"},
        {"source": "qdrant", "category": "databases"},
    ],
)

# 4. Query (Chroma embeds the query and finds top-k)
results = collection.query(
    query_texts=["stateful agents"],
    n_results=2,
    where={"category": "databases"},  # metadata filter
)
print(results["documents"])
# [['ChromaDB is a vector database', 'Qdrant is a Rust-based vector search engine']]
```

Three lines to set up, three lines to query, four lines of metadata. That's why Chroma is the prototype standard.

## References

- [[../00 - Welcome to Vector Databases and Semantic Search.md|Welcome to Vector DBs]] — course map for the wider vault.
- [[../01 - Vector Search Fundamentals.md|Vector Search Fundamentals]] — ANN, embeddings, distance metrics.
- [[../05 - Qdrant I - Architecture and Collections.md|Qdrant I]] — the production-grade counterpart.
- [[../09 - Vector Database Comparison Matrix.md|Vector Database Comparison]] — where Chroma sits relative to peers.
- Chroma documentation: https://docs.trychroma.com/