# 🏷️ Welcome to vLLM and Advanced RAG

## 🎯 Learning Objectives

- Deploy production-grade LLM inference servers using **vLLM** with PagedAttention and continuous batching
- Benchmark and compare vLLM against SGLang, TGI, and TensorRT-LLM for throughput, latency, and memory efficiency
- Architect **hybrid search pipelines** combining BM25 lexical retrieval and dense vector search for RAG systems
- Implement fusion strategies (Reciprocal Rank Fusion, convex combination, late-interaction) in Python with Redis
- Integrate RedisSearch indexes with HuggingFace embeddings behind a FastAPI layer
- Design **auto-scaling inference APIs** with Docker Compose and Kubernetes GPU scheduling
- Understand real-world deployment decisions at Anthropic, OpenAI, Notion, and Coda scale

---

## Why vLLM + Advanced RAG Is the 2025 Standard

By 2025, the LLM serving landscape has consolidated around a few battle-tested engines, and vLLM has emerged as the dominant open-source choice for production inference. Simultaneously, naive RAG pipelines that rely solely on cosine similarity over dense embeddings have proven brittle in production: lexical gaps cause missed retrievals, semantic gaps cause irrelevant results, and latency budgets collapse under naive batching. This course teaches the **engineered intersection** of these two pillars.

If you have completed [[../../02 - Large Language Models/09 - Sistemas de LLMs en Produccion/05 - Caso Practico - API de LLM Escalable|the scalable LLM API case study]], you already understand the FastAPI + Redis + vLLM architecture at a high level. If you have worked through [[../04 - Production RAG System.md|the Production RAG System course]], you know how chunking, vector indexing, and reranking shape retrieval quality. This course goes deeper into both topics — and merges them.

When you are interviewing for your first AI/ML Engineer role, the ability to discuss **PagedAttention memory management** and **Reciprocal Rank Fusion** sets you apart from candidates who only know `model.generate()`. ML platform teams at companies like Anthropic, Notion, and Stripe build exactly these systems, and they test for this knowledge. [[../17 - ML Platform Engineering/05 - ML Platform Engineering.md|ML platform engineering]] roles increasingly expect hands-on inference serving expertise.

---

## Course Map

| # | Note | What You Build | Time |
|---|------|---------------|------|
| 01 | vLLM: Production-Grade LLM Serving | Auto-scaling inference API with Docker Compose | 4-6 h |
| 02 | Hybrid Search: BM25 + Dense + Redis | FastAPI hybrid search endpoint with RedisSearch | 4-6 h |

Each note is **self-contained** — you can work through them in any order. Note 01 focuses on the serving layer; Note 02 focuses on the retrieval layer. Together they form a complete production RAG backend.

---

## Prerequisites

| Concept | Expected Knowledge | Review If Needed |
|---------|-------------------|-----------------|
| LLM fundamentals | Transformer architecture, tokenization, attention | [[../../02 - Large Language Models/06 - Fundamentos de LLMs/05 - Evaluacion de LLMs.md\|LLM Evaluation notes]] |
| Python async | `async def`, `await`, `asyncio.gather` | FastAPI documentation |
| Docker | Dockerfile, docker-compose, GPU passthrough | Docker Profesional course |
| Vector search | Embedding models, cosine similarity, HNSW/IVF | [[../04 - Production RAG System.md\|Production RAG System]] |
| Redis basics | Data structures, TTL, pub/sub | [[../../06 - Cloud, Infra y Backend/25 - Bases de Datos y Message Queues/03 - Redis y Caching.md\|Redis y Caching]] |
| GPU concepts | CUDA cores, VRAM, tensor parallelism | NVIDIA developer docs |
| Kubernetes (optional) | Pods, deployments, GPU scheduling | ML Platform Engineering course |

> 💡 **Tip**: If you are shaky on Redis, skim the [[../../06 - Cloud, Infra y Backend/25 - Bases de Datos y Message Queues/03 - Redis y Caching.md|Redis y Caching]] note before Note 02. Hybrid search assumes basic Redis fluency.

---

## Project Outcome

By the end of this course, you will have two deployable components that can be combined into a single production-stack repository:

```
rag-backend/
├── llm-serving/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── vllm_api.py
├── hybrid-search/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── search_api.py
└── README.md
```

This repository is **interview-ready**: push it to GitHub and reference it when asked "Tell me about a production ML system you built."

---

## How to Use These Notes

1. **Read the theory section first.** Every module explains *why* before showing *how*.
2. **Copy the compression code.** Each note includes a complete, runnable block that you can paste into a file and execute.
3. **Study the documented project.** The project walkthrough explains every design decision.
4. **Link to your other vault notes.** When you see `[[...]]` links, click them to reinforce connections across courses.
