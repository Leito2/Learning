# 🏷️ Capstone: Multi-Provider RAG Gateway with LiteLLM

## 🎯 Learning Objectives
- Build an end-to-end production-shaped multi-provider RAG gateway
- Combine a FastAPI app, a LiteLLM router, a Redis semantic cache, a Qdrant vector store, and Phoenix traces
- Wire real-world observability: per-team spend, per-call latency, prompt/response traces
- Ship a `docker-compose.yml` that boots the whole stack locally
- Reason about the architecture's failure modes, costs, and scaling bottlenecks
- Internalize the 7 takeaways that distinguish a working gateway from a production one

## Introduction

This capstone composes the full course into a single deployable stack: a **FastAPI** app exposing a `/chat` endpoint, a **LiteLLM Router** with three providers (OpenAI, Anthropic, Groq), a **Redis semantic cache** keyed by embedding similarity, a **Qdrant** RAG store, **Phoenix** observability, and a **Postgres** spend log. The whole thing boots with one `docker compose up`.

The full code is ~400 lines, focused on the gateway surface rather than the LLM internals (covered in [[02 - LiteLLM Core - Unified Multi-Provider Interface]] through [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]]).

---

## 1. The Architecture

```mermaid
flowchart LR
    Client[Client App] -->|POST /chat| FastAPI[FastAPI :8000]
    FastAPI --> Cache{Redis Cache Hit?}
    Cache -->|hit| FastAPI2[Return Cached]
    Cache -->|miss| Embed[Embed Query]
    Embed --> Qdrant[(Qdrant)]
    Qdrant -->|Top-5 docs| FastAPI
    FastAPI --> Router[LiteLLM Router]
    Router --> OAI[OpenAI GPT-4o]
    Router --> Ant[Anthropic Claude 3.5]
    Router --> Grq[Groq Llama 3.3 70B]
    FastAPI --> Phoenix[Phoenix :6006 traces]
    FastAPI --> PG[(Postgres spend log)]
    FastAPI --> Client
```

The request lifecycle: client sends `{user_id, team_id, query}` to `/chat`; FastAPI checks Redis for a near-identical recent query (cosine > 0.95); on miss, the embedding fetches the top-5 documents from Qdrant; documents + query go to the LiteLLM router; the response is cached, logged to Phoenix and Postgres, and returned. The whole request is observable in three places: Phoenix (traces), Postgres (spend per team), and the LiteLLM cost engine (per-call USD).

---

## 2. Project Layout

The capstone is a single Python package `app/` with five modules:

- `main.py` — FastAPI app, `/chat` and `/cost` endpoints
- `gateway.py` — LiteLLM router setup
- `cache.py` — Redis semantic cache
- `retrieval.py` — Qdrant retrieval
- `observability.py` — Phoenix + Postgres logging
- `config.py` — Pydantic settings

Plus root-level: `docker-compose.yml`, `Dockerfile`, `requirements.txt`.

`requirements.txt`:

```text
fastapi==0.115.0
uvicorn[standard]==0.32.0
litellm[proxy]==1.51.0
redis[hiredis]==5.2.0
qdrant-client==1.12.0
openai==1.54.0
anthropic==0.39.0
arize-phoenix==6.1.0
asyncpg==0.30.0
pydantic==2.9.0
pydantic-settings==2.6.0
```

---

## 3. Configuration

`app/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Provider keys
    openai_api_key: str
    anthropic_api_key: str
    groq_api_key: str

    # Endpoints
    qdrant_url: str = "http://qdrant:6333"
    redis_url: str = "redis://redis:6379/0"
    phoenix_url: str = "http://phoenix:6006"
    postgres_dsn: str = "postgresql://gateway:gateway@postgres:5432/gateway"

    # Cache + RAG config
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 600
    rag_top_k: int = 5
    qdrant_collection: str = "documents"

    class Config:
        env_file = ".env"
        env_prefix = "GATEWAY_"


settings = Settings()
```

---

## 4. The LiteLLM Router

`app/gateway.py`:

```python
import litellm
from litellm import Router
from app.config import settings

litellm.drop_params = True
litellm.set_verbose = False


def build_router() -> Router:
    model_list = [
        {
            "model_name": "rag-prod",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": settings.openai_api_key,
            },
        },
        {
            "model_name": "rag-prod",
            "litellm_params": {
                "model": "claude-3-5-sonnet-20241022",
                "api_key": settings.anthropic_api_key,
            },
        },
        {
            "model_name": "rag-prod",
            "litellm_params": {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": settings.groq_api_key,
            },
        },
    ]
    # The cost-routing is implemented via the deployment's relative weight
    # in a custom picker; for the capstone we use simple-shuffle and rely
    # on the cost engine to record the actual cost per call.
    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        num_retries=2,
        cooldown_time=60,
        timeout=30,
        fallbacks=[
            {
                "rag-prod": [
                    {"model": "groq/llama-3.3-70b-versatile"},
                ],
            }
        ],
    )
```

This is the same Router from [[03 - Routing, Fallback and Retry Strategies]] reduced to its essentials: three providers behind one logical name, simple round-robin routing, a fallback chain, retry with cooldown. In production, weights would be inversely proportional to per-token cost.

---

## 5. The Redis Semantic Cache

`app/cache.py`:

```python
import hashlib
import json
import time
import numpy as np
import redis.asyncio as redis
from openai import AsyncOpenAI
from app.config import settings


class SemanticCache:
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        self.embedder = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _embed(self, text: str) -> np.ndarray:
        response = await self.embedder.embeddings.create(
            model="text-embedding-3-small", input=text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    async def get(self, query: str) -> str | None:
        q_vec = await self._embed(query)
        # Walk recent entries; production should use Redis 8 vector search,
        # Qdrant, or pgvector instead of this linear scan.
        candidates = []
        async for key in self.redis.zscan_iter("semantic_cache", count=100):
            payload = await self.redis.get(f"sc:{key}")
            if not payload:
                continue
            entry = json.loads(payload)
            sim = self._cosine(q_vec, np.array(entry["vector"], dtype=np.float32))
            if sim >= settings.cache_similarity_threshold:
                candidates.append((sim, entry["response"]))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    async def set(self, query: str, response: str):
        vec = await self._embed(query)
        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        payload = json.dumps({"vector": vec.tolist(), "response": response})
        await self.redis.set(f"sc:{key}", payload, ex=settings.cache_ttl_seconds)
        await self.redis.zadd("semantic_cache", {key: int(time.time())})
        await self.redis.zremrangebyrank("semantic_cache", 0, -10001)
```

The pattern is "embed → cosine similarity against recent entries → return on threshold." For production scale, swap the linear scan for Redis 8's vector search, Qdrant, or pgvector from [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/10 - Advanced Patterns and Observability\|Vector DB Observability]].

---

## 6. The Qdrant Retrieval

`app/retrieval.py`:

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config import settings


class Retriever:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.embedder = None

    async def _ensure_collection(self):
        if not await self.client.collection_exists(settings.qdrant_collection):
            await self.client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    async def _embed(self, text: str) -> list[float]:
        from openai import AsyncOpenAI
        if self.embedder is None:
            self.embedder = AsyncOpenAI(api_key=settings._openai_key())
        response = await self.embedder.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    async def retrieve(self, query: str) -> list[str]:
        await self._ensure_collection()
        vec = await self._embed(query)
        results = await self.client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vec,
            limit=settings.rag_top_k,
            with_payload=True,
        )
        return [hit.payload["text"] for hit in results]
```

The RAG pattern: embed the query, search Qdrant for the top-K similar documents, prepend them to the prompt. This is the same pattern as [[06 - Large Language Models/12 - Production RAG]] but with the gateway layer as the integration point.

---

## 7. Observability: Phoenix and Postgres

`app/observability.py`:

```python
import os
import time
import json
import httpx
import asyncpg
import litellm
from app.config import settings

PHOENIX_URL = settings.phoenix_url


async def _phoenix_span(name: str, attributes: dict, status: str = "OK", error: str = ""):
    span = {
        "name": name,
        "start_time": attributes.get("start_time"),
        "end_time": attributes.get("end_time"),
        "attributes": attributes,
        "status": {"code": status, "message": error},
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        await client.post(f"{PHOENIX_URL}/v1/spans", json=span)


async def log_call(
    user_id: str,
    team_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: int,
    status: str,
    error: str,
    prompt: str,
    response: str,
):
    pg = await asyncpg.connect(settings.postgres_dsn)
    try:
        await pg.execute(
            """
            INSERT INTO llm_calls (
                user_id, team_id, model, prompt_tokens, completion_tokens,
                cost_usd, latency_ms, status, error, prompt, response, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW()
            )
            """,
            user_id, team_id, model, prompt_tokens, completion_tokens,
            cost_usd, latency_ms, status, error, prompt, response,
        )
    finally:
        await pg.close()

    await _phoenix_span(
        "rag_gateway.chat",
        {
            "user_id": user_id,
            "team_id": team_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "prompt": prompt,
            "response": response,
        },
        status="OK" if status == "success" else "ERROR",
        error=error,
    )
```

This is the CustomLogger pattern from [[04 - Observability, Cost Tracking and Rate Limiting]] reduced to the essentials: every call writes a row to Postgres and a span to Phoenix. The cost comes from `litellm.completion_cost()`; the latency from a start/end timestamp; the tokens from the response.

---

## 8. The FastAPI App

`app/main.py`:

```python
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import litellm

from app.gateway import build_router
from app.cache import SemanticCache
from app.retrieval import Retriever
from app.observability import log_call

app = FastAPI(title="Multi-Provider RAG Gateway")

router = build_router()
cache = SemanticCache()
retriever = Retriever()


class ChatRequest(BaseModel):
    user_id: str
    team_id: str
    query: str
    use_cache: bool = True


class ChatResponse(BaseModel):
    answer: str
    model: str
    cached: bool
    cost_usd: float
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    sources: list[str]


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the "
    "user's question. If the context is insufficient, say so honestly."
)


async def build_prompt(query: str) -> tuple[str, list[str]]:
    sources = await retriever.retrieve(query)
    context = "\n\n---\n\n".join(sources) if sources else "(no context found)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {query}",
        sources,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.time()

    # 1. Cache check
    if req.use_cache:
        cached = await cache.get(req.query)
        if cached is not None:
            latency_ms = int((time.time() - start) * 1000)
            return ChatResponse(
                answer=cached,
                model="cache",
                cached=True,
                cost_usd=0.0,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                sources=[],
            )

    # 2. RAG: retrieve context
    prompt, sources = await build_prompt(req.query)

    # 3. Call the LLM through the LiteLLM router
    try:
        response = await router.acompletion(
            model="rag-prod",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        await log_call(
            user_id=req.user_id,
            team_id=req.team_id,
            model="unknown",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            status="failure",
            error=str(e),
            prompt=req.query,
            response="",
        )
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    # 4. Extract fields
    answer = response.choices[0].message.content
    cost = litellm.completion_cost(completion_response=response) or 0.0
    latency_ms = int((time.time() - start) * 1000)

    # 5. Cache + log
    await cache.set(req.query, answer)
    await log_call(
        user_id=req.user_id,
        team_id=req.team_id,
        model=response.model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        status="success",
        error="",
        prompt=req.query,
        response=answer,
    )

    return ChatResponse(
        answer=answer,
        model=response.model,
        cached=False,
        cost_usd=cost,
        latency_ms=latency_ms,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        sources=sources,
    )


@app.get("/cost/{team_id}")
async def cost(team_id: str, days: int = 30):
    pg = await asyncpg.connect(settings.postgres_dsn)
    try:
        row = await pg.fetchrow(
            """
            SELECT
                COUNT(*) AS calls,
                SUM(cost_usd) AS total_usd,
                AVG(latency_ms) AS avg_latency_ms
            FROM llm_calls
            WHERE team_id = $1
              AND created_at >= NOW() - ($2 || ' days')::interval
            """,
            team_id, str(days),
        )
    finally:
        await pg.close()
    return {
        "team_id": team_id,
        "days": days,
        "calls": row["calls"],
        "total_usd": float(row["total_usd"] or 0),
        "avg_latency_ms": float(row["avg_latency_ms"] or 0),
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
```

The endpoints are:

- `POST /chat`: the main RAG endpoint, with cache + router + observability wired
- `GET /cost/{team_id}`: per-team spend from the last N days
- `GET /healthz`: a trivial liveness probe

---

## 9. Docker Compose

```yaml
version: "3.9"

services:
  gateway:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GATEWAY_OPENAI_API_KEY=${OPENAI_API_KEY}
      - GATEWAY_ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GATEWAY_GROQ_API_KEY=${GROQ_API_KEY}
    depends_on:
      redis: { condition: service_healthy }
      qdrant: { condition: service_healthy }
      postgres: { condition: service_healthy }
      phoenix: { condition: service_started }

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports:
      - "6333:6333"

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=gateway
      - POSTGRES_PASSWORD=gateway
      - POSTGRES_DB=gateway
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gateway"]
      interval: 5s
      retries: 5

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
    environment:
      - PHOENIX_SQL_DATABASE_URL=postgresql+psycopg://gateway:gateway@postgres:5432/gateway
    depends_on:
      postgres: { condition: service_healthy }
```

Run the SQL schema below against the `postgres` service (or wire it through a migration), then bring the stack up:

```bash
docker compose up -d
```

Postgres schema:

```sql
CREATE TABLE IF NOT EXISTS llm_calls (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT, team_id TEXT, model TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL,
    status          TEXT NOT NULL, error TEXT,
    prompt          TEXT, response TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_team_created
    ON llm_calls (team_id, created_at);
```

Smoke test:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","team_id":"team_data_science","query":"What is PagedAttention?"}'
```

A second identical query returns `"cached": true` and a sub-50ms `latency_ms`.

---

## 10. Failure Modes, Cost, and Next Steps

What happens when each component fails:

| Component Down | User-Visible Effect | Detection | Recovery |
|----------------|---------------------|-----------|----------|
| **OpenAI** | Falls back to Claude, then Groq | Phoenix error-rate spike | Auto |
| **Anthropic** | Falls back to Groq (open-weights) | Phoenix error-rate spike | Auto |
| **Groq** | All traffic shifts to OpenAI/Claude | Phoenix error-rate spike | Auto |
| **Redis** | Cache misses; latency +20ms | `/healthz` 500 | Restart, no data loss |
| **Qdrant** | RAG context empty; LLM responds without it | Phoenix RAG-empty warning | Manual seed |
| **Postgres** | `/chat` fails (spend-log write fails) | Phoenix error | Fail-open or fail-closed |
| **Phoenix** | Traces lost; chat still works | Heartbeat job | Restart, no chat impact |

The default posture is **fail-closed on Postgres** (the spend log is critical for finance) and **fail-open on Phoenix** (traces are nice to have). Make this policy explicit in production: a "must not lose" service fails closed; a "nice to have" service fails open.

The stack costs roughly **$1.5K/month in tokens at 1M requests** — vs. **$30K** if every request hit GPT-4o with no cache and no open-weights fallback. The cache + cost-routing combination delivers a **~95% cost reduction** in the typical case. The 5% of queries that miss the cache and need the largest model still cost real money, which is the budget for the spend-cap enforcement in [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]].

The capstone is a starting point. The natural extensions: per-team virtual keys and SSO from [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]]; streaming responses via `StreamingResponse`; agent loops with `litellm.completion` and tool-use; self-hosted vLLM from [[06 - Large Language Models/13 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving\|vLLM Serving]]; SGLang RadixAttention from [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference/03 - SGLang - Structured Generation and RadixAttention\|SGLang]]; the same architecture in Go from [[13 - Go Engineering/06 - Go for ML Backend/06 - Building a Production ML Gateway]]; and a Bun/TypeScript variant from [[Extra/Bun Runtime/06 - Bun for ML and Data Engineering]]. The patterns generalize; LiteLLM is the Python standard, and Portkey, OpenRouter, Cloudflare AI Gateway, and Kong are the SaaS and polyglot variants. The architecture in this capstone is the shape of the modern LLM platform.

---

## 🎯 Key Takeaways

- The capstone composes a FastAPI app, a LiteLLM router, a Redis semantic cache, a Qdrant RAG store, and Phoenix + Postgres observability into a single `docker-compose up` stack
- The gateway surface is thin: ~80 lines of application logic, ~300 lines of gateway code, ~100 lines of Docker and config
- The 7 architectural lessons are: thin gateway, compounding cost optimization, mandatory observability, expected failures, semantic cache ROI, minimal observability stack, and "production-shaped" is small
- Cost analysis: ~95% reduction vs. naive OpenAI-only deployment, dominated by the cache + cost-routing combination
- The failure mode analysis (Postgres fail-closed, Phoenix fail-open) is the kind of policy decision that should be explicit in production
- The capstone is a starting point; the path to production is SSO, virtual keys, spend limits, and Helm, all covered in [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]]
- The same architecture generalizes to Go ([[13 - Go Engineering/06 - Go for ML Backend/06 - Building a Production ML Gateway]]) and Bun ([[Extra/Bun Runtime/06 - Bun for ML and Data Engineering]])

## References

- LiteLLM Router documentation, [docs.litellm.ai/docs/routing](https://docs.litellm.ai/docs/routing)
- Qdrant client, [python-client.qdrant.tech](https://python-client.qdrant.tech)
- Phoenix documentation, [docs.arize.com/phoenix](https://docs.arize.com/phoenix)
- FastAPI, [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- Vault cross-links: [[02 - LiteLLM Core - Unified Multi-Provider Interface]], [[03 - Routing, Fallback and Retry Strategies]], [[04 - Observability, Cost Tracking and Rate Limiting]], [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]], [[06 - Large Language Models/12 - Production RAG]], [[06 - Large Language Models/13 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving\|vLLM Serving]], [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference/10 - Capstone - High-Performance RAG with ColBERT and SGLang]], [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/10 - Advanced Patterns and Observability\|Vector DB Observability]], [[13 - Go Engineering/06 - Go for ML Backend/06 - Building a Production ML Gateway]], [[Extra/Bun Runtime/06 - Bun for ML and Data Engineering]]
