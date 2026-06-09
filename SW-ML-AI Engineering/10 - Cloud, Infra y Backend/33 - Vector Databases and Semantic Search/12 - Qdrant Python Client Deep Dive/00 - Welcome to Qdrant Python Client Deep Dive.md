# 🐍 Welcome to Qdrant Python Client Deep Dive

## 🎯 Learning Objectives

- Diagnose which **Qdrant client API surface** the existing vault notes do not cover, and explain why those gaps matter for production
- Master the **advanced client APIs** that the basics course skipped: local mode, named vectors, quantization config, scroll, batch search, count, facet, recommend
- Build **production-grade async** Qdrant integrations with FastAPI, retries, batching, and observability hooks
- Compose a **full RAG production stack** that ties `AsyncQdrantClient` together with LiteLLM routing and Phoenix traces
- Distinguish the **sync `QdrantClient`**, the **`AsyncQdrantClient`**, and the **`QdrantClient(path=...)` local mode** — and pick the right one for each context (production, FastAPI, tests/edge)
- Identify the **Qdrant 1.10+ v1.12+ API changes** that break older code (`query_points` supersedes `search`, new `models` namespace)

---

## Why This Course Exists

The vault's existing Qdrant coverage in `05 - Qdrant I - Architecture and Collections` and `06 - Qdrant II - Distributed and Cloud Deployment` is the **fundamentals layer**: collections, points, basic upsert and search, payload filters, snapshots, and a 3-node cluster. That is exactly what you need to *understand* Qdrant. It is not enough to *use* it in production at the levels expected in 2025-2026.

A production engineer working on a RAG stack reaches for capabilities those notes never cover:

- **Local mode** — running Qdrant in-process for unit tests, CI pipelines, or single-tenant edge deployments (`QdrantClient(path="./qdrant_data")`).
- **Named vectors** — storing multiple embedding models per point (e.g., dense + ColBERT + sparse) under typed vector names, instead of cramming them into a single concatenated vector.
- **Quantization configuration** — turning on `ScalarQuantization` or `ProductQuantization` at collection creation to cut memory by 4× with <2% recall loss.
- **Scroll, batch, count, facet, recommend** — the rest of the client API that the basics course treats as "out of scope".
- **Production async patterns** — `AsyncQdrantClient` connection pooling, retry-with-backoff, batch ingestion at 10K+ points/sec, OpenTelemetry instrumentation, graceful shutdown, health checks.

This course closes those gaps. We assume you have read [[05 - Qdrant I - Architecture and Collections]] and [[06 - Qdrant II - Distributed and Cloud Deployment]] (or have equivalent production experience). We do not re-explain HNSW, distance metrics, or what a payload is — that is settled.

The capstone in note `03` builds the **Qdrant layer of the LLM Gateway capstone** (`[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/06 - Capstone - Multi-Provider RAG Gateway with LiteLLM]]`) with the deeper patterns it deserves: multi-tenant via Qdrant payloads, hybrid search, query rewriting, response streaming, and Phoenix traces end-to-end.

---

## Course Map

| Note | Title | What You Will Master |
|------|-------|----------------------|
| `00` | **Welcome** *(this note)* | Why this course exists, what is covered, prerequisites |
| `01` | **Advanced Client APIs** | Local mode, named vectors, quantization, scroll, batch search, count, facet, recommend |
| `02` | **Production Async Patterns** | `AsyncQdrantClient` for FastAPI, connection pooling, retry, batch ingestion, OTel, health checks, graceful shutdown |
| `03` | **Capstone — Production RAG with Qdrant + LiteLLM + Phoenix** | Full stack: FastAPI + AsyncQdrantClient + LiteLLM router + Phoenix traces; multi-tenant payloads, hybrid search, query rewriting, response streaming |

---

## Prerequisites

| Concept | Where to Review If Needed |
|---------|--------------------------|
| Qdrant fundamentals (collections, points, HNSW, payloads) | [[../05 - Qdrant I - Architecture and Collections]] |
| Qdrant cluster ops (Raft, sharding, snapshots) | [[../06 - Qdrant II - Distributed and Cloud Deployment]] |
| Hybrid search (sparse + dense) | [[../10 - Advanced Patterns and Observability]] |
| Async Python (`async`/`await`, `asyncio.gather`, `httpx`) | `03 - Python Avanzado/06 - Concurrencia - Threading y Asyncio` |
| FastAPI dependency injection, lifespan, Pydantic | [[../../../13 - Go Engineering]] is a different stack — use the FastAPI docs; the vault has FastAPI examples in `10/31/01` |
| OpenTelemetry basics | `09 - MLOps y Produccion/31 - Evidently AI and Phoenix` |
| LiteLLM router | `[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/02 - LiteLLM Core - Unified Multi-Provider Interface]]` |

> 💡 **Tip**: if you can write `from qdrant_client import QdrantClient; QdrantClient("localhost:6333").search(...)` and understand what it returns, you have the prerequisites. This course extends that surface.

---

## Project Outcome

By the end of this course you will have a `qdrant-python-deep-dive/` repository with:

- A **local-mode test harness** for CI (`QdrantClient(path=...)`, no Docker required)
- A **multi-vector collection** with named dense + sparse + ColBERT vectors
- A **production async service** that ingests 10K points/sec with retry, OTel, and graceful shutdown
- A **FastAPI RAG service** that uses the Qdrant collection, rewrites queries, calls LiteLLM, streams responses, and ships Phoenix traces

All four components are runnable locally with `uv sync && docker compose up` and reusable as drop-in libraries.

---

## 🔗 Related Vault Modules

| Module | Connection |
|--------|-----------|
| [[../05 - Qdrant I - Architecture and Collections]] | The fundamentals we build on |
| [[../06 - Qdrant II - Distributed and Cloud Deployment]] | Production deployment context |
| [[../10 - Advanced Patterns and Observability]] | Hybrid search baseline |
| [[../11 - Capstone Project - Multi-DB Semantic Search Platform]] | The async + FastAPI pattern we extend |
| [[../../35 - Vector Quantization/01 - Product Quantization]] | Quantization internals referenced in note 01 |
| `[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/06 - Capstone - Multi-Provider RAG Gateway with LiteLLM]]` | The peer capstone we deepen in note 03 |
| `[[../../../09 - MLOps y Produccion/31 - Evidently AI and Phoenix/03 - Phoenix by Arize - LLM Observability, Traces and Embedding Drift]]` | Observability layer in note 03 |
| `[[../../../14 - Rust Engineering/04 - Rust for ML and AI/07 - Vector Databases in Rust (Qdrant, pgvector)]]` | The Rust client peer (different language) |

## 📝 How to Use These Notes

Each note follows the **Deep Format**: Learning Objectives, Introduction, four numbered sections (Problem → Conceptual Deep Dive → Production Reality → Code in Practice), Key Takeaways, Compression Code, References. Theory always precedes code. Every `code` block is runnable against `qdrant-client>=1.12`. Every note ends with a **Compression Code** block — a single script that exercises all the key APIs in the note.

Set up your environment with:

```bash
uv add qdrant-client[fastembed]>=1.12.0 fastapi uvicorn httpx openinference-instrumentation-qdrant \
       opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
       litellm arize-phoenix
```

Pull the Qdrant image for the production parts: `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.12.0`. For the local-mode parts, no Docker is required — Qdrant runs in-process.
