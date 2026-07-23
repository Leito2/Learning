# 🏷️ Welcome — Production RAG Frameworks: Haystack and txtai

## 🎯 Learning Objectives
- Distinguish the four major RAG frameworks (LangChain, LlamaIndex, Haystack, txtai) and pick the right one per scenario
- Build production pipelines with Haystack 2.x: components, pipelines, retrievers, generators
- Use txtai for all-in-one semantic search + RAG with graphs and workflows
- Implement hybrid retrieval (BM25 + dense + reranking) on Haystack
- Design production RAG services with FastAPI, LangFuse, and Phoenix observability
- Migrate between frameworks and integrate with the broader vault (LangChain, LlamaIndex, Instructor)

## Introduction

The RAG framework ecosystem has crystallized into **four serious contenders** by 2026. **LangChain** dominates the open-source Python world with the broadest integration surface. **LlamaIndex** owns the data-heavy indexing use case with the most sophisticated ingestion pipelines. **Haystack** by [deepset](https://www.deepset.ai) is the enterprise alternative with explicit production hardening — explicit DAG pipelines, deepset Cloud deployment, and the kind of testing rigor that Fortune 500 companies require. **txtai** by [neuml](https://github.com/neuml/txtai) is the all-in-one semantic search + RAG framework with the smallest surface area and the fastest cold-start.

This course is the missing piece in the vault's RAG track. You already have:

- **Production RAG** ([[06 - Large Language Models/12 - Production RAG]]) — the canonical patterns
- **vLLM and Advanced RAG** ([[06 - Large Language Models/13 - vLLM and Advanced RAG]]) — retrieval + serving
- **ColBERT, SGLang and Next-Gen Inference** ([[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference]]) — token-level retrieval
- **Advanced RAG** (covered in the Production RAG course)
- **LangGraph + RAG** ([[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]])

What you do not yet have is **enterprise-grade RAG pipelines** with Haystack's explicit DAG semantics, deepset Cloud deployment, and the kind of testing/CI rigor that distinguishes a real production system from a research demo. And you do not yet have **txtai's all-in-one** semantic search + RAG pattern that some teams prefer for its simplicity.

By the end of these six notes you will have built a production RAG service that uses Haystack 2.x for the core pipeline, txtai for graph-based multi-hop retrieval, and integrates with LangChain/LlamaIndex retrievers via the OpenAI-compatible API. The capstone is the **eighth portfolio project**.

![Haystack 2.x architecture](https://haystack.deepset.ai/assets/img/og_image.png)

---

## 1. The Four Major RAG Frameworks in 2026

| Framework | License | Sweet spot | Strength | Weakness |
|-----------|---------|------------|----------|----------|
| **LangChain** | MIT | Composition via chains | Ubiquitous, broad integrations | Abstraction overhead, complex RAG via LCEL |
| **LlamaIndex** | MIT | Data-heavy indexing | Best ingestion, multi-modal | Less flexible for non-RAG agents |
| **Haystack** | Apache 2.0 | Enterprise pipelines | Explicit DAGs, deepset Cloud, production testing | Steeper learning curve, smaller community |
| **txtai** | Apache 2.0 | All-in-one semantic search | Smallest API, fastest cold-start | Less flexible for complex multi-step agents |

For an AI/ML Engineer targeting enterprise roles, **Haystack is the signal-skill**. Hiring managers at Fortune 500 recognize it immediately as the production-grade choice. For teams that want minimal surface area, **txtai** is the alternative.

---

## 2. Why Haystack, Why Now

Three reasons:

1. **Enterprise adoption.** Haystack is the foundation of deepset Cloud (the managed RAG-as-a-service) and several Fortune 500 internal knowledge systems. If you work in financial services, healthcare, or defense — Haystack is the standard.
2. **Explicit DAG pipelines.** Haystack 2.x (released 2024) treats pipelines as **typed DAGs** with components, connections, and validation. Compare to LangChain's `|` operator — Haystack's `Pipeline` is debuggable, serializable, and testable in isolation.
3. **Production testing.** Haystack ships with `pytest` integration for component-level testing, evaluation pipelines for offline RAGAS-style evals, and CI/CD patterns that map directly to enterprise GitOps workflows.

The framework has 22k+ GitHub stars, 5k+ active contributors, and a stable 2.x release that ships minor versions monthly. The companion product, **deepset Cloud**, offers managed deployment for teams that don't want to operate Kubernetes.

---

## 3. Why txtai

txtai occupies a different niche: **all-in-one simplicity**. The library combines embeddings index, vector search, graph database, and LLM orchestration in a single Python package with a small API surface.

For teams that want:
- Embeddings index without the complexity of Qdrant/Milvus
- Semantic search + RAG in one library
- Graph-based multi-hop retrieval
- Low-latency (typically <10ms cold-start)

txtai is the right choice. It is not as flexible as Haystack for complex multi-step agents, but it is dramatically simpler for the canonical RAG use case.

---

## 4. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why Production RAG Frameworks | This overview |
| 01 | Haystack Fundamentals — Pipelines, Components, Retrievers | 2.x architecture, Component DSL, Document Stores |
| 02 | Haystack Advanced Pipelines | Hybrid retrieval, reranking, agentic loops, evaluation |
| 03 | txtai Fundamentals — Semantic Search + RAG | Embeddings index, graphs, workflows |
| 04 | Production RAG Patterns — Comparison and Integration | Framework selection, integration with LangFuse/Phoenix |
| 05 | Capstone — Production Hybrid RAG Service | FastAPI + Haystack + txtai + LiteLLM + observability |

---

## 5. Prerequisites

You should already be comfortable with:

- **RAG fundamentals** — retrieval, chunking, embedding, reranking from [[06 - Large Language Models/12 - Production RAG]]
- **Vector databases** — Qdrant, Milvus, Pinecone from [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search]]
- **Pydantic models** — for schemas from [[03 - Advanced Python/06 - Pydantic Deep Dive]]
- **FastAPI service patterns** — async, lifespan from [[10 - Cloud, Infra y Backend/31 - FastAPI for ML]]
- **LLM serving** — vLLM, LiteLLM from [[06 - Large Language Models/13 - vLLM and Advanced RAG]] and [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]]
- **Instructor + structured outputs** — for typed RAG responses from [[06 - Large Language Models/22 - Instructor and Structured Generation]]

💡 If you have not yet read [[06 - Large Language Models/12 - Production RAG]], skim it before Note 01 — the canonical RAG patterns are assumed.

---

## 6. Cross-Module Connections

This course is part of the RAG spine across modules 06, 07, 09:

| Vault Module | Connection to This Course |
|--------------|---------------------------|
| [[06 - Large Language Models/12 - Production RAG\|Production RAG]] | Foundational RAG patterns |
| [[06 - Large Language Models/13 - vLLM and Advanced RAG\|vLLM and Advanced RAG]] | vLLM as the LLM backend in the capstone |
| [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference\|ColBERT/SGLang]] | Token-level retrieval integration |
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns]] | LiteLLM routing across providers |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor and Structured Generation]] | Structured RAG responses |
| [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization\|Serverless LLM Platforms]] | Modal for custom reranking, Together/Fireworks for LLM |
| [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns\|LangGraph Deep Patterns]] | Agentic RAG patterns |
| [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix\|Evidently AI and Phoenix]] | RAG observability |
| [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability\|LangFuse Deep Dive]] | Cost attribution per tenant |
| [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search\|Vector Databases]] | Qdrant/Milvus/Pinecone as Haystack document stores |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | Capstone service architecture |

---

## 7. What You Will Build

By Note 05, you will have constructed a **production hybrid RAG service** with:

- A Haystack 2.x pipeline that ingests documents, performs hybrid retrieval (BM25 + dense), and reranks via a cross-encoder
- A txtai embeddings index for semantic search with graph-based multi-hop retrieval
- A FastAPI service exposing `/query`, `/ingest`, and `/evaluate` endpoints
- LiteLLM routing across OpenAI, Anthropic, and Together AI for the LLM backend
- Instructor + Pydantic for structured RAG responses with citations
- LangFuse traces for every retrieval + generation, with Phoenix spans for the pipeline DAG
- Offline evaluation via RAGAS-style metrics on a held-out golden dataset
- Docker Compose stack that brings up the whole system with one command

This is the **eighth portfolio project**: a production hybrid RAG service that demonstrates enterprise-grade engineering rigor.

---

## 8. Reading Order for Vault Natives

If you already have RAG mastery:

1. Note 00 (this file) — landscape and decision framework
2. Note 01 (Haystack Fundamentals) — required foundation
3. Note 02 (Haystack Advanced) — production patterns
4. Note 03 (txtai) — alternative approach
5. Note 04 (Production Patterns) — comparison and integration
6. Note 05 (Capstone) — synthesis

If you are new to RAG:

1. Read [[06 - Large Language Models/12 - Production RAG]] first
2. Then return here for Note 01

---

## 9. The Cutting Edge in 2026

Three frontiers are emerging:

1. **Deepset Cloud + Haystack 2.5.** Managed Haystack deployment with auto-scaling, observability, and SSO. The deepset Cloud team ships monthly.
2. **txtai's workflow DSL.** Beyond embeddings + graphs, txtai 7+ ships a workflow primitive similar to LangGraph but in pure Python.
3. **Cross-framework retrievers.** Haystack 2.x supports retrievers from any framework via the `RunnableInterface` — LangChain retrievers, LlamaIndex query engines, custom PyTorch models all integrate.

These frontiers map directly onto the user's portfolio: the **LLM Edge Gateway** (Go/Fiber/Redis) can integrate a Haystack retrieval endpoint; the **Automated LLM Evaluation Suite** runs RAGAS evals against Haystack outputs; the **Multi-Agent Research System** (LangGraph) uses Haystack retrievers as tools; the **StayBot** uses txtai for graph-based Airbnb listing recommendations.

---

⚠️ The RAG framework ecosystem evolves fast. Haystack ships minor versions monthly (2.x), LangChain ships breaking changes quarterly, and LlamaIndex ships major versions yearly. The **patterns** in this course are stable; the **library APIs and version pins** in the code examples will need updating. Always cross-check the official docs ([haystack.deepset.ai](https://haystack.deepset.ai), [neuml.github.io/txtai](https://neuml.github.io/txtai/)) before deploying.