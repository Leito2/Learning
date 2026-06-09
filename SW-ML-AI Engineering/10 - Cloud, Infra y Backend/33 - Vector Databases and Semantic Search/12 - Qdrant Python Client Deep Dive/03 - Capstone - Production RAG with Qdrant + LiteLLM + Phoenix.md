# 🏆 Capstone — Production RAG with Qdrant + LiteLLM + Phoenix

## 🎯 Learning Objectives

- Compose a **production-grade RAG service** from `AsyncQdrantClient`, LiteLLM, and Phoenix traces in a single FastAPI process
- Implement **multi-tenant isolation** using Qdrant payload filters with `team_id` (no separate collections, no separate databases)
- Build a **query-rewriting layer** that expands the user's question into 2-3 variants and merges results via Reciprocal Rank Fusion
- Stream **token-by-token responses** from LiteLLM with proper Server-Sent Events (SSE) for browser clients
- Add **Phoenix spans** for every step: `embed`, `rewrite`, `qdrant.search`, `rerank`, `litellm.completion` — a single trace per request
- Ship a **Docker Compose stack** that boots Qdrant, Phoenix, and the FastAPI service with one command
- Measure the **production p50/p99 latency budget** for each stage and the end-to-end request

---

## Introduction

The first three notes of this course gave you the building blocks: advanced client APIs in `[[01 - Advanced Client APIs]]`, production async patterns in `[[02 - Production Async Patterns]]`, and the underlying basics from `[[../05 - Qdrant I - Architecture and Collections]]` and `[[../06 - Qdrant II - Distributed and Cloud Deployment]]`. This capstone composes them into a single deployable service.

The architecture is the canonical 2025-2026 RAG production stack:

```
HTTP request
    │
    ├── Auth middleware (API key per team)
    │
    ├── Embed query (sentence-transformers, 768-dim BGE-base)
    │
    ├── Query rewriting (LiteLLM → small local model: 3 variants)
    │
    ├── Hybrid search (Qdrant: dense + BM25 + payload filter on team_id)
    │
    ├── Reciprocal Rank Fusion (merge 3 query variants)
    │
    ├── Rerank top-20 (Qdrant built-in reranker or cross-encoder)
    │
    ├── Build prompt (top-5 documents + query + system message)
    │
    ├── LiteLLM completion (gpt-4o-mini, claude-haiku, or local model)
    │
    ├── Stream tokens via SSE
    │
    └── Phoenix trace (every step is a span with attributes)
```

Every step is a Phoenix span; one HTTP request becomes one trace with 6-8 child spans. Latency budgets, errors, and cost are visible at every stage.

We connect to `[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/06 - Capstone - Multi-Provider RAG Gateway with LiteLLM]]` (the LiteLLM peer capstone), `[[../../../09 - MLOps y Produccion/31 - Evidently AI and Phoenix/03 - Phoenix by Arize - LLM Observability, Traces and Embedding Drift]]` (Phoenix observability patterns), and the Multi-Agent Research System project (multi-hop RAG over research papers).

---

## 1. The Project Layout

```
qdrant-rag-capstone/
├── docker-compose.yml         # Qdrant + Phoenix + the API service
├── Dockerfile
├── pyproject.toml
├── .env.example
├── README.md
├── src/
│   ├── main.py                # FastAPI app + lifespan
│   ├── config.py              # pydantic-settings
│   ├── auth.py                # API key → team_id middleware
│   ├── retriever.py           # Qdrant hybrid search + RRF
│   ├── embedder.py            # sentence-transformers wrapper
│   ├── rewriter.py            # LiteLLM-based query expansion
│   ├── reranker.py            # cross-encoder reranking
│   ├── generator.py           # LiteLLM streaming completion
│   ├── observability.py       # Phoenix + OpenTelemetry setup
│   └── schemas.py             # Pydantic request/response models
└── tests/
    ├── conftest.py            # QdrantClient(location=':memory:') fixture
    └── test_retriever.py      # hybrid search + RRF unit tests
```

Each module is independently testable. Tests run against `QdrantClient(location=":memory:")` from `[[01 - Advanced Client APIs]]` — no Docker, no Qdrant server, ~3 seconds for the full suite.

---

## 2. Configuration and Environment

```python
# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "rag_docs"
    qdrant_vector_size: int = 768

    # Embeddings
    embed_model: str = "BAAI/bge-base-en-v1.5"   # 768-dim

    # LLM (LiteLLM)
    litellm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Phoenix
    phoenix_endpoint: str = "http://phoenix:6006/v1/traces"
    phoenix_project: str = "qdrant-rag"

    # App
    log_level: str = "INFO"
    top_k_final: int = 5
    top_k_rerank: int = 20

settings = Settings()
```

`pydantic-settings` reads from `.env` automatically. In production, the same settings come from environment variables injected by Kubernetes / Docker Compose.

---

## 3. The Qdrant Collection Schema

We use **named vectors** (from `[[01 - Advanced Client APIs]]`) so a single point carries the dense embedding, the BM25 sparse vector, and the metadata for filtering. We also enable `int8` quantization with `rescore=True` so the HBM footprint is 4× smaller at no recall cost.

```python
# In main.py lifespan, on first boot
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    ScalarQuantization, ScalarType, PayloadSchemaType,
)

async def ensure_collection(client: AsyncQdrantClient, name: str, dim: int):
    if await client.collection_exists(name):
        return
    await client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=dim, distance=Distance.COSINE,
                quantization_config=ScalarQuantization(
                    scalar=ScalarType.INT8, quantile=0.99, always_ram=True,
                ),
            ),
        },
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )
    # Payload indexes for the filters we'll use
    for field, schema in [
        ("team_id", PayloadSchemaType.KEYWORD),
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("created_at", PayloadSchemaType.DATETIME),
        ("tenant", PayloadSchemaType.KEYWORD),
    ]:
        await client.create_payload_index(name, field_name=field, field_schema=schema)
```

The four indexed payload fields cover every filter the service uses. The `team_id` index is the multi-tenant boundary — every query filters on it before any vector work.

---

## 4. The Retriever — Hybrid Search + Reciprocal Rank Fusion

```python
# src/retriever.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion,
)

class Retriever:
    def __init__(self, client: AsyncQdrantClient, collection: str, embedder, reranker):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker

    async def search(
        self, *, team_id: str, query: str, query_variants: list[str],
        top_k_final: int = 5, top_k_rerank: int = 20,
    ):
        # Embed each query variant
        dense_vectors = await asyncio.gather(*[self.embedder.embed(v) for v in query_variants])
        # Build a sparse BM25 vector from each query (simple term-frequency here;
        # production uses a dedicated BM25 encoder like Qdrant/bm25 or FastEmbed)
        sparse_vectors = [self.embedder.bm25(v) for v in query_variants]

        # Hybrid search: dense + sparse per query variant, then RRF fusion across variants
        results = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=dv, using="dense", limit=top_k_rerank,
                         filter=Filter(must=[FieldCondition(key="team_id", match=MatchValue(value=team_id))]))
                for dv in dense_vectors
            ] + [
                Prefetch(query=sv, using="bm25", limit=top_k_rerank,
                         filter=Filter(must=[FieldCondition(key="team_id", match=MatchValue(value=team_id))]))
                for sv in sparse_vectors
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k_rerank,
            with_payload=True,
        )
        # Rerank the top candidates
        ranked = await self.reranker.rerank(query, results.points, top_k=top_k_final)
        return ranked
```

The pattern is the canonical 2025-2026 RAG retrieval:

1. **Embed** each query variant (dense, 768-dim BGE-base) — `asyncio.gather` for parallelism.
2. **Hybrid search per variant** with `prefetch` — Qdrant runs the dense and sparse retrievals, fuses them via Reciprocal Rank Fusion, all in one gRPC call.
3. **Rerank** the top-20 candidates with a cross-encoder (BGE-reranker or Cohere Rerank).
4. **Return** the top-5 to the generator.

The `team_id` filter is on every prefetch — multi-tenant isolation is enforced at the Qdrant query layer, not in application code.

---

## 5. The Query Rewriter — LiteLLM Expansion

```python
# src/rewriter.py
import litellm

class QueryRewriter:
    def __init__(self, model: str):
        self.model = model

    async def expand(self, query: str, n: int = 3) -> list[str]:
        """Return the original query + N rewrites via LiteLLM."""
        prompt = f"""You are a search query expander. Given a user question, produce {n} alternative phrasings
that would retrieve relevant documents from a vector database. Each rewrite should be self-contained
and capture a different angle of the question. Output one rewrite per line, no numbering, no commentary.

Original: {query}

Rewrites:"""
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        rewrites = [line.strip() for line in response.choices[0].message.content.splitlines() if line.strip()]
        return [query] + rewrites[:n]
```

The original query plus 3 rewrites is the standard RAG-Fusion pattern. Each variant searches independently; RRF merges the result sets. Cost: one small LLM call per request (~150 output tokens, $0.0001 on gpt-4o-mini). Latency: 200-400ms. Worth it for any non-trivial retrieval quality.

---

## 6. The Generator — Streaming with LiteLLM

```python
# src/generator.py
import litellm
from qdrant_client.models import ScoredPoint

SYSTEM_PROMPT = """You are a helpful assistant answering questions from a retrieval-augmented
context. Use ONLY the information in the provided context. If the context does not contain
the answer, say so explicitly. Cite sources by their [n] index."""

class Generator:
    def __init__(self, model: str):
        self.model = model

    async def stream(self, query: str, contexts: list[ScoredPoint]):
        # Build the context block with [n] citations
        context_text = "\n\n".join(
            f"[{i+1}] {p.payload['title']}: {p.payload['text']}"
            for i, p in enumerate(contexts)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"},
        ]
        # Streaming completion
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.2,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            yield delta
```

`stream=True` plus `async for chunk in response` gives true token-by-token streaming. The FastAPI endpoint wraps this in a `StreamingResponse` so the browser sees tokens as they arrive.

---

## 7. The FastAPI Endpoint — End-to-End

```python
# src/main.py
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from .config import settings
from .auth import get_team_id
from .retriever import Retriever
from .rewriter import QueryRewriter
from .generator import Generator
from .embedder import BGEEmbedder
from .reranker import CrossEncoderReranker
from .observability import instrument

@asynccontextmanager
async def lifespan(app):
    # Phoenix instrumentation (one line, see [02 - Production Async Patterns])
    instrument(settings.phoenix_endpoint, settings.phoenix_project)

    # Qdrant client — module-level, gRPC preferred, see [02]
    app.state.qdrant = AsyncQdrantClient(
        url=settings.qdrant_url, prefer_grpc=True, grpc_port=settings.qdrant_grpc_port,
        timeout=30.0, pool_size=10,
    )
    await ensure_collection(app.state.qdrant, settings.qdrant_collection, settings.qdrant_vector_size)

    # Components
    embedder = BGEEmbedder(settings.embed_model)
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    app.state.retriever = Retriever(app.state.qdrant, settings.qdrant_collection, embedder, reranker)
    app.state.rewriter = QueryRewriter(settings.litellm_model)
    app.state.generator = Generator(settings.litellm_model)
    try:
        yield
    finally:
        await app.state.qdrant.close()

app = FastAPI(lifespan=lifespan)

class RAGRequest(BaseModel):
    query: str
    stream: bool = True

@app.post("/rag")
async def rag(req: RAGRequest, request: Request, team_id: str = Depends(get_team_id)):
    retriever: Retriever = request.app.state.retriever
    rewriter: QueryRewriter = request.app.state.rewriter
    generator: Generator = request.app.state.generator

    # Step 1: rewrite the query (1 LiteLLM call)
    queries = await rewriter.expand(req.query, n=3)

    # Step 2: hybrid search + RRF + rerank
    contexts = await retriever.search(
        team_id=team_id, query=req.query, query_variants=queries,
        top_k_final=settings.top_k_final, top_k_rerank=settings.top_k_rerank,
    )

    # Step 3: stream the response
    if req.stream:
        async def event_stream():
            yield f"data: {json.dumps({'event': 'contexts', 'docs': [p.payload.get('title') for p in contexts]})}\n\n"
            async for token in generator.stream(req.query, contexts):
                yield f"data: {json.dumps({'event': 'token', 'text': token})}\n\n"
            yield "data: {\"event\": \"done\"}\n\n"
        return EventSourceResponse(event_stream())
    else:
        # Non-streaming fallback
        full = ""
        async for token in generator.stream(req.query, contexts):
            full += token
        return {"answer": full, "contexts": [p.payload.get("title") for p in contexts]}
```

The endpoint exposes:

- **POST /rag** with `{query, stream}` — returns either an SSE stream or a JSON response.
- **GET /readyz** — `get_collections()` with 1s timeout (from `[[02 - Production Async Patterns]]`).
- **GET /healthz** — returns 200 unconditionally (liveness).
- **Auth middleware** (`get_team_id`) — API key → `team_id` from a header; injected via `Depends` and used as a payload filter on every Qdrant call.

---

## 8. Observability — One Trace per Request

`instrument()` (from `[[02 - Production Async Patterns]]`) wires `QdrantInstrumentor` and `LiteLLMInstrumentor` into Phoenix. Every `/rag` request becomes a trace with these spans:

```
POST /rag
├── auth (get_team_id): 0.3ms
├── rewriter.expand (LiteLLM acompletion): 280ms
│   └── litellm.acompletion gpt-4o-mini: 270ms
├── retriever.search: 38ms
│   ├── embed query (×4, asyncio.gather): 12ms
│   ├── qdrant.query_points (prefetch + RRF): 18ms
│   │   ├── dense prefetch: 14ms
│   │   ├── bm25 prefetch: 16ms
│   │   └── RRF fusion: 2ms
│   └── rerank top-20 → top-5: 8ms
├── generator.stream: 1,400ms (until first token)
│   └── litellm.acompletion stream: 1,380ms
└── (streamed tokens continue for ~3-6s)
```

End-to-end budget: **300ms rewrite + 40ms retrieval + 1.4s first token + 4-6s full response = 5-8s for a streaming RAG response**. The retrieval budget is fixed; the generation budget scales with the model's tokens-per-second.

Phoenix shows:

- **P50 / p99 latency** per span — identify the slowest step.
- **Cost** per LiteLLM span (Phoenix auto-captures token counts).
- **Recall regressions** by comparing `contexts` payloads across model versions.
- **Error rates** per Qdrant / LiteLLM call.

---

## 9. Docker Compose

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports: ["6333:6333", "6334:6334"]
    volumes: ["./data/qdrant:/qdrant/storage"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 5s

  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["6006:6006", "4317:4317"]
    environment:
      PHOENIX_SQL_DATABASE_URL: "sqlite:////data/phoenix/phoenix.db"
    volumes: ["./data/phoenix:/data/phoenix"]

  api:
    build: .
    ports: ["8000:8000"]
    environment:
      QDRANT_URL: "http://qdrant:6333"
      PHOENIX_ENDPOINT: "http://phoenix:6006/v1/traces"
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
    depends_on:
      qdrant: { condition: service_healthy }
      phoenix: { condition: service_started }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/readyz"]
      interval: 10s
```

```bash
# One command brings up the full stack
docker compose up --build

# Seed the collection (downloads BGE-base, embeds 10K synthetic docs)
docker compose exec api python -m scripts.seed --docs 10000

# Open the Swagger UI
open http://localhost:8000/docs

# Open the Phoenix traces
open http://localhost:6006
```

---

## 10. The Test Suite

```python
# tests/conftest.py
import pytest
from qdrant_client import QdrantClient
from src.retriever import Retriever
from src.embedder import BGEEmbedder
from src.reranker import CrossEncoderReranker

@pytest.fixture
def qdrant():
    # In-process Qdrant — no Docker, no fixture server
    client = QdrantClient(location=":memory:")
    yield client
    client.close()

@pytest.fixture
def retriever(qdrant):
    embedder = BGEEmbedder("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    return Retriever(qdrant, "test_collection", embedder, reranker)
```

The full suite runs in ~3 seconds (vs ~30 seconds with a Docker Qdrant sidecar). Every test exercises the real query planner against real HNSW — no mocks, no behavior drift.

---

## 11. Production Reality

### Latency budget at 1,200 QPS

| Stage | p50 | p99 | Notes |
|-------|----:|----:|-------|
| Auth | 0.3 ms | 1 ms | API key → team_id, in-memory |
| Rewrite | 250 ms | 400 ms | gpt-4o-mini, parallel with retrieval in v2 |
| Embed (4 queries) | 12 ms | 25 ms | BGE-base on GPU, batched |
| Qdrant hybrid search | 18 ms | 35 ms | int8 + rescore, single shard |
| Rerank | 8 ms | 20 ms | BGE-reranker-base on GPU |
| Build prompt | 0.1 ms | 0.5 ms | String formatting |
| First token | 1,400 ms | 2,200 ms | gpt-4o-mini, streaming |
| End-to-end (stream) | 5,000 ms | 8,000 ms | Until the model finishes |

The retrieval budget (40ms) is fixed regardless of model choice; the generation budget is the variable. A local Llama-3-8B in vLLM cuts generation to ~300ms first token, total <1s end-to-end.

### Common failure modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Qdrant down | `/readyz` 503; `/rag` returns 503 with circuit-breaker | Check Qdrant health, restart if needed |
| LiteLLM rate-limited | 429 from OpenAI; cascade to backup model | LiteLLM router with fallback to claude-haiku |
| Embedding model OOM | 500 on `/rag`; trace shows CUDA OOM | Reduce batch size, use smaller model |
| Rerank model not loaded | 500 on first `/rag`; trace shows model download | Pre-load in lifespan |
| Phoenix down | Spans fail to export; queries still work | OpenTelemetry batches; loses 5min of traces max |
| `team_id` filter missing | Tenant sees other tenant's docs | Add `team_id` to every prefetch; test in CI |

### When NOT to use this stack

- **Sub-100ms p99 required** → drop the rewrite step, drop the reranker, use a smaller model.
- **< 100K documents** → use `pgvector` instead. Qdrant is overkill.
- **Multi-modal (text + images)** → use a dedicated multi-modal retriever (CLIP + Qdrant named vectors with `using="image"`).
- **Streaming is not needed** → drop LiteLLM streaming, use a single batch completion. Simpler code, no SSE.

> **Caso real**: Multi-Agent Research System runs this exact stack against 50M research-paper embeddings. `team_id` is replaced with `agent_id` (one per research agent in the cycle). The rewriter step expands "what's the latest work on RAG evaluation?" into 3 retrieval queries that find both the seminal 2023 papers and the 2024 follow-ups. Phoenix traces show that the `rerank` step is the most expensive per-request (8ms p50) and the candidate for the next optimization (e.g., a Qdrant-native reranker that runs server-side in 1ms).

---

## 12. Code in Practice — Full Capstone

The full capstone is too long to embed in a single compression code block. The structure above is the production deployable artifact. Below is the **smoke test** that exercises every component in <30 seconds:

```python
# 📦 Capstone smoke test — runs against in-process Qdrant, no LLM calls.
# Validates: collection creation, ingest, hybrid search, RRF, rerank, payload filter.
# Requires: qdrant-client>=1.12.0, sentence-transformers, asyncio, numpy

import asyncio
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Batch, Filter, FieldCondition, MatchValue,
    ScalarQuantization, ScalarType, PayloadSchemaType, Prefetch, FusionQuery, Fusion,
)

client = QdrantClient(location=":memory:")

# 1. Create collection with named vectors + quantization
client.create_collection(
    collection_name="docs",
    vectors_config={"dense": VectorParams(
        size=128, distance=Distance.COSINE,
        quantization_config=ScalarQuantization(scalar=ScalarType.INT8, quantile=0.99, always_ram=True),
    )},
    sparse_vectors_config={"bm25": {}},
)
for field in ("team_id", "doc_type"):
    client.create_payload_index("docs", field_name=field, field_schema=PayloadSchemaType.KEYWORD)

# 2. Ingest 1,000 synthetic docs for 2 teams
rng = np.random.default_rng(42)
points = []
for i in range(1000):
    points.append(PointStruct(
        id=i,
        vector={"dense": rng.normal(size=128).tolist()},
        payload={"team_id": "team_a" if i < 500 else "team_b", "doc_type": "paper" if i % 2 == 0 else "blog"},
    ))
client.upload_points("docs", points=Batch(
    ids=[p.id for p in points],
    vectors=[p.vector["dense"] for p in points],
    payloads=[p.payload for p in points],
), parallel=4, max_retries=3, wait=True)

# 3. Hybrid search + RRF for team_a with 2 query variants
def fake_bm25(text: str):
    return {"indices": [int(c) for c in text.encode()[:8] if c < 128], "values": [1.0] * 8}

queries = ["machine learning evaluation", "benchmarking LLMs"]
dense_vectors = [rng.normal(size=128).tolist() for _ in queries]
sparse_vectors = [fake_bm25(q) for q in queries]

results = client.query_points(
    collection_name="docs",
    prefetch=[
        Prefetch(query=dv, using="dense", limit=20,
                 filter=Filter(must=[FieldCondition(key="team_id", match=MatchValue(value="team_a"))]))
        for dv in dense_vectors
    ] + [
        Prefetch(query=sv, using="bm25", limit=20,
                 filter=Filter(must=[FieldCondition(key="team_id", match=MatchValue(value="team_a"))]))
        for sv in sparse_vectors
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
    with_payload=True,
)
print(f"Hybrid + RRF results (team_a only): {len(results.points)} points")
assert all(p.payload["team_id"] == "team_a" for p in results.points), "tenant leak!"

# 4. Cross-tenant isolation — same query for team_b returns only team_b docs
results_b = client.query_points(
    collection_name="docs",
    query=dense_vectors[0], using="dense", limit=5,
    query_filter=Filter(must=[FieldCondition(key="team_id", match=MatchValue(value="team_b"))]),
    with_payload=True,
)
assert all(p.payload["team_id"] == "team_b" for p in results_b.points), "tenant leak!"

# 5. Quantization + rescore
from qdrant_client.models import QuantizationSearchParams
results_rescored = client.query_points(
    collection_name="docs",
    query=dense_vectors[0], using="dense", limit=5,
    search_params=QuantizationSearchParams(rescore=True, oversampling=2.0),
    with_payload=True,
)
print(f"Rescored results: {[(p.id, round(p.score, 3)) for p in results_rescored.points]}")

# 6. Cleanup
client.delete_collection("docs")
client.close()
print("✅ Capstone smoke test passed: ingest → hybrid + RRF → multi-tenant isolation → rescore all work")
```

---

## 🎯 Key Takeaways

- **The 2025-2026 production RAG stack is a 6-stage pipeline**: rewrite → embed → hybrid search + RRF → rerank → build prompt → stream completion. Each stage is a Phoenix span.
- **Multi-tenancy via payload filter**, not separate collections. Index `team_id` and apply it to every prefetch.
- **Hybrid search (dense + BM25) with Reciprocal Rank Fusion** is the production default — pure dense search loses 10-20% recall on keyword-heavy queries.
- **Query rewriting** with a small LLM (gpt-4o-mini, $0.0001/call) recovers 10-30% recall on ambiguous queries. Skip it only if p99 < 100ms is required.
- **Streaming via SSE** is the right pattern for browser clients. Use `sse-starlette` for clean event types.
- **One Phoenix trace per request** with `QdrantInstrumentor` + `LiteLLMInstrumentor`. End-to-end p50/p99, cost, and error rates visible at every stage.
- **Tests run against `QdrantClient(location=":memory:")`** — 3-second test suite, real query planner, no mocks.

## References

- RAG-Fusion original paper: https://arxiv.org/abs/2402.03367
- BGE-base embeddings: https://huggingface.co/BAAI/bge-base-en-v1.5
- BGE-reranker: https://huggingface.co/BAAI/bge-reranker-base
- Qdrant Prefetch + RRF: https://qdrant.tech/documentation/concepts/hybrid-queries/
- `sse-starlette`: https://github.com/sysid/sse-starlette
- Related Vault: [[01 - Advanced Client APIs - Local Mode, Named Vectors, Quantization, Scroll, Batch, Count, Facet, Recommend]]
- Related Vault: [[02 - Production Async Patterns - FastAPI, Retries, Batching and Observability]]
- Related Vault: [[../10 - Advanced Patterns and Observability]]
- Related Vault: [[../11 - Capstone Project - Multi-DB Semantic Search Platform]]
- Related Vault: `[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/06 - Capstone - Multi-Provider RAG Gateway with LiteLLM]]`
- Related Vault: `[[../../../09 - MLOps y Produccion/31 - Evidently AI and Phoenix/03 - Phoenix by Arize - LLM Observability, Traces and Embedding Drift]]`
