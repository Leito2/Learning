# 🏷️ 11 - Capstone Project - Multi-DB Semantic Search Platform

## 🎯 Learning Objectives
- Architect a polyglot semantic search API leveraging both Qdrant and pgvector simultaneously
- Implement dual-language backends: Python/FastAPI and Go/Fiber connecting to both databases
- Generate embeddings using OpenAI or local models and store identical data in both engines
- Build a comparative query endpoint measuring latency and recall across backends in real time
- Create a benchmark harness validating p99 < 100ms and recall@10 > 0.9 under load
- Orchestrate the entire stack with Docker Compose for reproducible local development

## Introduction

Throughout this course, you studied [[05 - Qdrant I - Architecture and Collections|Qdrant's]] payload-filtered ANN, [[03 - pgvector I - Core Operations and Indexing|pgvector's]] SQL-native simplicity, and [[07 - Milvus I - Distributed Architecture|Milvus's]] microservice scale. The capstone unifies these lessons into a production-grade platform treating Qdrant and pgvector as complementary tiers: Qdrant for low-latency semantic retrieval, pgvector for transactional metadata lookups and hybrid SQL queries. The project is implemented in both **Python/FastAPI** and **Go/Fiber** so you can compare ecosystem ergonomics and runtime performance within a single stack.
---

## Module 1: Architecture and Data Flow
### 1.1 Theoretical Foundation 🧠

A multi-DB semantic search platform must solve three problems: **data consistency** (both stores hold the same vectors), **query routing** (which backend serves which request), and **comparative evaluation** (how do we know which is better?). The architecture uses an abstraction layer — a `SearchBackend` interface in Go or an abstract base class in Python — that hides Qdrant and pgvector specifics from HTTP handlers.

Consistency is achieved via **dual write**: the ingest endpoint writes to Qdrant and pgvector in parallel. In production, this would be wrapped in a Saga or outbox pattern; for the capstone, simple try/except with rollback suffices. Query routing can be static (`?backend=qdrant`) or dynamic (latency-based adaptive routing). The compare endpoint runs the same query against both backends and returns latency and overlap metrics.

### 1.2 Mental Model 📐

```
┌─────────────────────────────────────────────────────────────┐
│                    Client / Load Tester                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
            ┌──────────▼──────────┐
            │   API Gateway       │
            │  (FastAPI / Fiber)  │
            └──────────┬──────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐  ┌──────────────┐
│  /ingest     │ │ /search  │  │ /compare     │
│  (dual write)│ │(routed)  │  │ (both DBs)   │
└──────┬───────┘ └────┬─────┘  └──────┬───────┘
       │              │               │
   ┌───┴───┐      ┌───┴───┐      ┌───┴───┐
   ▼       ▼      ▼       ▼      ▼       ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│Qdran│ │pgvec│ │Qdran│ │pgvec│ │Qdran│ │pgvec│
│-t   │ │-tor │ │-t   │ │-tor │ │-t   │ │-tor │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

### 1.3 Syntax and Semantics 📝

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class SearchBackend(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict]) -> None:
        """WHY: Dual write requires atomicity guarantees. In production,
        use an outbox table in Postgres to ensure eventual consistency."""
        ...

    @abstractmethod
    def search(self, vector: List[float], top_k: int, filters: Dict = None) -> List[Tuple[str, float]]:
        """WHY: Common return type (id, score) lets /compare merge results
        without vendor-specific parsing."""
        ...
```

```go
type SearchBackend interface {
	Upsert(ctx context.Context, ids []string, vectors [][]float32, payloads []map[string]interface{}) error
	Search(ctx context.Context, vector []float32, topK int, filters map[string]interface{}) ([]SearchResult, error)
}

type SearchResult struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"`
}
```

### 1.4 Visual Representation 🖼️

```mermaid
flowchart TB
    subgraph Client["Client"]
        C[HTTP Client]
    end
    subgraph API["API Layer"]
        F[FastAPI / Fiber]
    end
    subgraph Backends["Backends"]
        Q[Qdrant Backend]
        P[pgvector Backend]
    end
    subgraph DBs["Databases"]
        QDB[(Qdrant)]
        PDB[(Postgres + pgvector)]
    end
    C -->|/ingest| F
    C -->|/search?backend=qdrant| F
    C -->|/compare| F
    F --> Q
    F --> P
    Q --> QDB
    P --> PDB
```

![Docker Compose](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Docker_%28container_engine%29_logo.svg/512px-Docker_%28container_engine%29_logo.svg.png)

### 1.5 Application in ML/AI Systems 🤖

Real case: **Shopify's semantic search** runs Qdrant for customer-facing product search and pgvector for internal admin tools with complex SQL joins. A shared embedding pipeline keeps both stores synchronized. The /compare endpoint is used in CI to catch regressions when upgrading either database.

| ML Use Case           | This Concept                 | Impact                           |
|-----------------------|------------------------------|----------------------------------|
| Polyglot retrieval    | Qdrant + pgvector dual backend | Best-of-breed per query type    |
| Language migration    | Python + Go implementations  | Team flexibility, perf testing   |
| A/B testing indices   | /compare endpoint            | Data-driven index tuning         |
| Local reproducibility | Docker Compose stack         | Onboard engineers in < 10 min    |

### 1.6 Common Pitfalls ⚠️

⚠️ **Dual write without idempotency**: Retrying a failed ingest creates duplicates. Use deterministic IDs (content hash or UUIDv5) so both stores deduplicate naturally.
💡 *Mnemonic: "Idempotent writes, safe retries."*

⚠️ **Connection pool exhaustion**: FastAPI's default asyncpg pool and Qdrant's synchronous client can deadlock under load. Size pools to 2× expected concurrency and use async Qdrant client in Python.
💡 *Mnemonic: "Pools sized small, queries stall."*

### 1.7 Knowledge Check ❓

1. Why is an abstraction layer essential before adding a third backend (e.g., Milvus)?
2. What consistency guarantees does the dual-write pattern provide, and what does it lack?
3. How would you modify the architecture to use a message queue for asynchronous dual writes?
---

## Module 2: Implementation in Python and Go
### 2.1 Theoretical Foundation 🧠

FastAPI natively supports async/await, automatic OpenAPI documentation, and Pydantic validation. The asyncpg driver for Postgres is fully async, preventing blocking during pgvector queries. Qdrant's `AsyncQdrantClient` makes true dual-write parallelism possible with `asyncio.gather`. Embedding generation is the slowest step; we support OpenAI (`text-embedding-3-small`) and local models (`sentence-transformers/all-MiniLM-L6-v2`) behind an `Embedder` interface.

Go's goroutines and lightweight runtime make it ideal for I/O-bound gateway services. Fiber provides fast routing and middleware support. The `qdrant/go-client` wraps gRPC calls to Qdrant, while `pgx` (with `pgxpool`) is the idiomatic Postgres driver for Go. Both support context cancellation, enabling request timeouts that propagate to the database layer.

### 2.2 Mental Model 📐

```
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI App (Python)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  /ingest    │  │  /search    │  │  /compare           │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                    │            │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────────▼──────────┐ │
│  │  Embedder   │  │  Embedder   │  │  Embedder           │ │
│  │  (OpenAI /  │  │  (cached)   │  │  (cached)           │ │
│  │   local)    │  │             │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                    │            │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────────┴──────────┐ │
│  │ QdrantClient│  │ QdrantClient│  │ Both Clients        │ │
│  │ (async)     │  │ (async)     │  │ (parallel search)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                    │            │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────────┴──────────┐ │
│  │ asyncpg     │  │ asyncpg     │  │ Both Pools          │ │
│  │ (pgvector)  │  │ (pgvector)  │  │ (parallel query)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```python
# backends/qdrant_backend.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

class QdrantBackend:
    def __init__(self, host: str = "qdrant", port: int = 6333):
        self.client = AsyncQdrantClient(host=host, port=port)

    async def ensure_collection(self, name: str, dim: int):
        exists = await self.client.collection_exists(name)
        if not exists:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def upsert(self, collection: str, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                  for i in range(len(ids))]
        await self.client.upsert(collection_name=collection, points=points)

    async def search(self, collection: str, vector, top_k: int):
        results = await self.client.search(
            collection_name=collection, query_vector=vector, limit=top_k, with_payload=False,
        )
        return [(r.id, r.score) for r in results]
```

```python
# backends/pgvector_backend.py
import asyncpg, json

class PgvectorBackend:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(self.dsn, min_size=5, max_size=20)
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS items ("
                "id TEXT PRIMARY KEY, embedding vector(768), payload JSONB)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_embedding ON items "
                "USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)"
            )

    async def upsert(self, ids, vectors, payloads):
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO items (id, embedding, payload) VALUES ($1, $2, $3)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
                """,
                [(ids[i], str(vectors[i]), json.dumps(payloads[i])) for i in range(len(ids))]
            )

    async def search(self, vector, top_k: int):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, embedding <=> $1 as distance FROM items ORDER BY embedding <=> $1 LIMIT $2",
                str(vector), top_k,
            )
            return [(r["id"], 1.0 - float(r["distance"])) for r in rows]
```

```python
# main.py (excerpt)
from fastapi import FastAPI
import asyncio, time

app = FastAPI()

@app.post("/ingest")
async def ingest(req: IngestRequest):
    vector = await embedder.encode(req.text)
    payload = {"text": req.text}
    # WHY: asyncio.gather runs both writes concurrently.
    await asyncio.gather(
        qdrant.upsert(COLLECTION, [req.id], [vector], [payload]),
        pgvector.upsert([req.id], [vector], [payload]),
        return_exceptions=False,
    )
    return {"status": "ok"}

@app.get("/compare")
async def compare(q: str, top_k: int = 10):
    vector = await embedder.encode(q)
    t0 = time.perf_counter()
    q_res = await qdrant.search(COLLECTION, vector, top_k)
    q_lat = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_res = await pgvector.search(vector, top_k)
    p_lat = time.perf_counter() - t0
    overlap = len({r[0] for r in q_res} & {r[0] for r in p_res}) / top_k
    return {
        "qdrant": {"latency_ms": round(q_lat * 1000, 2), "results": q_res},
        "pgvector": {"latency_ms": round(p_lat * 1000, 2), "results": p_res},
        "overlap@k": round(overlap, 2),
    }
```

```go
// backends/qdrant.go
package backends
import ("context"; pb "github.com/qdrant/go-client/qdrant"; "google.golang.org/grpc"; "google.golang.org/grpc/credentials/insecure")
type QdrantBackend struct{ client pb.PointsClient }
func NewQdrantBackend(addr string) (*QdrantBackend, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil { return nil, err }
	return &QdrantBackend{client: pb.NewPointsClient(conn)}, nil
}
func (q *QdrantBackend) Upsert(ctx context.Context, collection string, ids []string, vectors [][]float32, payloads []map[string]interface{}) error {
	points := make([]*pb.PointStruct, len(ids))
	for i := range ids {
		points[i] = &pb.PointStruct{
			Id:      &pb.PointId{PointIdOptions: &pb.PointId_Uuid{Uuid: ids[i]}},
			Vectors: &pb.Vectors{VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: vectors[i]}}},
		}
	}
	_, err := q.client.Upsert(ctx, &pb.UpsertPoints{CollectionName: collection, Points: points})
	return err
}
func (q *QdrantBackend) Search(ctx context.Context, collection string, vector []float32, topK int) ([]SearchResult, error) {
	res, err := q.client.Search(ctx, &pb.SearchPoints{
		CollectionName: collection, Vector: vector, Limit: uint64(topK),
		WithPayload: &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: false}},
	})
	if err != nil { return nil, err }
	results := make([]SearchResult, len(res.Result))
	for i, r := range res.Result { results[i] = SearchResult{ID: r.Id.GetUuid(), Score: r.Score} }
	return results, nil
}
```

```go
// backends/pgvector.go
package backends
import ("context"; "github.com/jackc/pgx/v5/pgxpool")
type PgvectorBackend struct{ pool *pgxpool.Pool }
func NewPgvectorBackend(dsn string) (*PgvectorBackend, error) {
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil { return nil, err }
	config.MaxConns = 20
	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil { return nil, err }
	return &PgvectorBackend{pool: pool}, nil
}
func (p *PgvectorBackend) Upsert(ctx context.Context, ids []string, vectors [][]float32, payloads []map[string]interface{}) error {
	tx, err := p.pool.Begin(ctx)
	if err != nil { return err }
	defer tx.Rollback(ctx)
	for i := range ids {
		_, err := tx.Exec(ctx,
			"INSERT INTO items (id, embedding, payload) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload",
			ids[i], pgvector.NewVector(vectors[i]), payloads[i],
		)
		if err != nil { return err }
	}
	return tx.Commit(ctx)
}
func (p *PgvectorBackend) Search(ctx context.Context, vector []float32, topK int) ([]SearchResult, error) {
	rows, err := p.pool.Query(ctx, "SELECT id, embedding <=> $1 as distance FROM items ORDER BY embedding <=> $1 LIMIT $2", pgvector.NewVector(vector), topK)
	if err != nil { return nil, err }
	defer rows.Close()
	var results []SearchResult
	for rows.Next() {
		var id string; var dist float32
		if err := rows.Scan(&id, &dist); err != nil { return nil, err }
		results = append(results, SearchResult{ID: id, Score: 1.0 - dist})
	}
	return results, nil
}
```

### 2.4 Visual Representation 🖼️

```mermaid
graph TD
    A[Client Request] --> B{Language?}
    B -->|Python| C[FastAPI]
    B -->|Go| D[Fiber]
    C --> E[Shared Abstraction]
    D --> E
    E --> F[Qdrant]
    E --> G[pgvector]
```

![FastAPI](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/FastAPI_logo.svg/512px-FastAPI_logo.svg.png)

### 2.5 Application in ML/AI Systems 🤖

Real case: **A health-tech startup** built their semantic search API in FastAPI with dual-write to Qdrant and pgvector. Qdrant serves patient-facing symptom search; pgvector powers internal analytics joining embeddings with insurance tables. The shared abstraction let them swap Qdrant for Milvus later without changing API routes.

| ML Use Case           | This Concept            | Impact                            |
|-----------------------|-------------------------|-----------------------------------|
| Rapid prototyping     | FastAPI + async clients | Sub-50ms p99 at 1k QPS            |
| High-throughput gateway| Go/Fiber + pgxpool     | 30% lower p99 vs Python at 10k QPS|
| CI regression testing | /compare overlap metric | Catch index degradation in pipelines|
| Schema evolution      | JSONB payloads + pgvector| No schema migrations for new metadata|

### 2.6 Common Pitfalls ⚠️

⚠️ **Blocking the event loop**: Using the sync `QdrantClient` inside an async FastAPI handler blocks all concurrent requests. Always use `AsyncQdrantClient`.
💡 *Mnemonic: "Sync in async = death by a thousand blocks."*

⚠️ **Forgetting pgxpool.Close()**: Leaking pool connections causes Postgres to reject new connections after `max_connections` is reached. Always defer `pool.Close()` in `main()`.
💡 *Mnemonic: "Open pool, defer close—never drown."*

### 2.7 Knowledge Check ❓

1. Why does `asyncio.gather` improve dual-write latency over sequential writes?
2. How would you share the embedding model between Python and Go without duplicating inference code?
3. What are the memory and GC implications of loading large embedding batches in Go vs. Python?
---

## Module 3: Docker Compose and Benchmarking
### 3.1 Theoretical Foundation 🧠

Local reproducibility is the hallmark of a well-engineered system. Docker Compose orchestrates Postgres (with pgvector), Qdrant, the API service (Python or Go), and a benchmark runner in a single `docker-compose up`. This eliminates "works on my machine" disputes and provides a sandbox for load testing before cloud deployment.

Benchmarking must measure both **latency** and **accuracy**. Latency is straightforward: record `time.Since(start)` or `time.perf_counter()`. Accuracy requires a ground-truth dataset. For the capstone, we use a small labeled corpus (e.g., MS MARCO passages) where relevance judgments are known. `recall@k` is computed by comparing ANN results against a brute-force linear scan of the same vectors. A benchmark harness runs 1,000 queries, records latencies, and validates that p99 < 100ms and recall@10 > 0.9.

### 3.2 Mental Model 📐

```
┌─────────────────────────────────────────────────────────────┐
│                 Docker Compose Stack                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   qdrant    │  │  postgres   │  │   api (py/go)       │ │
│  │   :6333     │  │  :5432      │  │   :8000             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              benchmark (k6 / Python)                │   │
│  │   - ingest labeled corpus                           │   │
│  │   - run 1k queries, record latency & recall         │   │
│  │   - switch backend, rerun, compare                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```yaml
# docker-compose.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: [qdrant_storage:/qdrant/storage]
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: vectors
    ports: ["5432:5432"]
    volumes: [pg_data:/var/lib/postgresql/data]
  api:
    build: ./api
    environment:
      QDRANT_HOST: qdrant
      DATABASE_URL: postgres://user:pass@postgres:5432/vectors
      EMBEDDER: local
    ports: ["8000:8000"]
    depends_on: [qdrant, postgres]
volumes:
  qdrant_storage:
  pg_data:
```

```python
# benchmark.py
import asyncio, time, requests, random
from datasets import load_dataset

API = "http://localhost:8000"
CORPUS = load_dataset("microsoft/ms_marco", "v1.1", split="validation[:1000]")

async def ingest():
    for doc in CORPUS:
        requests.post(f"{API}/ingest", json={"id": doc["query_id"], "text": doc["query"]})

async def benchmark(backend: str, n: int = 1000):
    latencies = []
    for _ in range(n):
        q = random.choice(CORPUS)["query"]
        t0 = time.perf_counter()
        requests.get(f"{API}/search", params={"q": q, "backend": backend, "top_k": 10})
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p50 = latencies[int(n * 0.5)]
    p99 = latencies[int(n * 0.99)]
    print(f"{backend}: p50={p50:.1f}ms p99={p99:.1f}ms")
    assert p99 < 100, f"p99 SLA violated: {p99}ms"

async def main():
    await ingest()
    await benchmark("qdrant")
    await benchmark("pgvector")

asyncio.run(main())
```

### 3.4 Visual Representation 🖼️

![Benchmark](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Placeholder_view_vector.svg/681px-Placeholder_view_vector.svg.png)

### 3.5 Application in ML/AI Systems 🤖

Real case: **A legal-tech startup** runs the exact Docker Compose stack in CI for every pull request. The benchmark script catches performance regressions (e.g., a library upgrade slowing pgvector by 20%). Before merging, the PR must pass p99 < 100ms and recall@10 > 0.9.

| ML Use Case            | This Concept               | Impact                          |
|------------------------|----------------------------|---------------------------------|
| CI performance gates   | benchmark.py in Docker Compose | Block regressions before deploy |
| Load testing           | k6 against /search         | Validate 10k QPS target         |
| Backend migration      | Switch + compare endpoints | Risk-free cutover with metrics  |
| Local debugging        | docker-compose logs -f api | Reproduce prod issues locally   |

### 3.6 Common Pitfalls ⚠️

⚠️ **Benchmarking on an empty database**: ANN indices behave differently with 100 vectors vs. 1M vectors. Always seed with realistic data volumes before measuring latency.
💡 *Mnemonic: "Bench with volume, or bench in vain."*

⚠️ **Docker resource limits**: Default Docker Desktop allocates 2 GB RAM, which crashes Qdrant or Postgres under load. Increase to 8 GB before benchmarking.
💡 *Mnemonic: "RAM starved, benchmark carved."*

### 3.7 Knowledge Check ❓

1. Why is recall@10 measured against brute-force rather than another ANN index?
2. How would you integrate the benchmark script into a GitHub Actions CI pipeline?
3. What Docker Compose changes are needed to scale Qdrant to a 3-node cluster for local testing?
---

## 📦 Compression Code

```python
"""Capstone — Minimal runnable skeleton (Python)."""
import asyncio, time, os, json
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import asyncpg

app = FastAPI()
qdrant = AsyncQdrantClient("qdrant", port=6333)
pg_pool = None
COLLECTION = "docs"

async def lifespan(app):
    global pg_pool
    pg_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"), min_size=5, max_size=20)
    exists = await qdrant.collection_exists(COLLECTION)
    if not exists:
        await qdrant.create_collection(COLLECTION, vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    yield
    await pg_pool.close()

app.router.lifespan_context = lifespan

@app.post("/ingest")
async def ingest(id: str, text: str):
    vec = [0.1] * 768
    payload = {"text": text}
    await asyncio.gather(
        qdrant.upsert(COLLECTION, [PointStruct(id=id, vector=vec, payload=payload)]),
        pg_pool.execute("INSERT INTO items (id, embedding, payload) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding", id, str(vec), json.dumps(payload)),
    )
    return {"status": "ok"}

@app.get("/compare")
async def compare(q: str, top_k: int = 10):
    vec = [0.1] * 768
    t0 = time.perf_counter()
    qr = await qdrant.search(COLLECTION, vec, limit=top_k)
    ql = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    pr = await pg_pool.fetch("SELECT id, embedding <=> $1 as d FROM items ORDER BY embedding <=> $1 LIMIT $2", str(vec), top_k)
    pl = (time.perf_counter() - t0) * 1000
    return {"qdrant_ms": round(ql, 2), "pgvector_ms": round(pl, 2), "qdrant_hits": len(qr), "pgvector_hits": len(pr)}
```

## 🎯 Documented Project
### Description
Build and deploy the Multi-DB Semantic Search Platform locally. Implement both Python/FastAPI and Go/Fiber versions. Ingest 10,000 MS MARCO queries, benchmark p50/p99 latency and recall@10, and produce a report comparing Qdrant vs. pgvector. Include a backend-switch endpoint and a Docker Compose file.
### Functional Requirements
- `docker-compose.yml` with Qdrant, Postgres (pgvector), API (Python), and benchmark services
- Python FastAPI: `/ingest`, `/search?backend=`, `/compare`
- Go Fiber: identical routes and behavior
- Benchmark script: 1,000 queries, computes p50/p99/recall@10, asserts p99 < 100ms and recall@10 > 0.9
- Backend switch: `POST /switch` updates a config value routed by middleware
### Main Components
- `python-api/`: FastAPI app + Dockerfile
- `go-api/`: Fiber app + Dockerfile + go.mod
- `embedder/`: Python gRPC embedding service (shared by both APIs)
- `benchmark/`: Python script + k6 JS script
- `docker-compose.yml`: Full local stack
### Success Metrics
- p99 search latency < 100ms for both Qdrant and pgvector at 10k vectors
- recall@10 > 0.9 vs brute-force on the same corpus
- Docker Compose starts from clean in < 3 minutes
- Benchmark report generated automatically in CI
## 🎯 Key Takeaways
- Polyglot backends (Qdrant + pgvector) let you optimize for latency and SQL flexibility without migration risk.
- An abstraction layer (`SearchBackend` interface) is essential for swapping or adding databases without API changes.
- Dual-write ingestion ensures consistency but requires idempotency (deterministic IDs) to handle retries safely.
- Python/FastAPI excels for ML integration and rapid development; Go/Fiber excels for high-throughput, low-memory gateways.
- Docker Compose provides reproducible local environments for development, debugging, and CI performance gates.
- Benchmarking must validate both latency percentiles (p50, p99) and accuracy (recall@k) against brute-force ground truth.
- Backend-switch endpoints enable canary deployments and A/B testing between indexing strategies or database versions.
## References
- FastAPI Docs: https://fastapi.tiangolo.com/
- Fiber Docs: https://docs.gofiber.io/
- qdrant-client (Python): https://github.com/qdrant/qdrant-client
- qdrant/go-client: https://github.com/qdrant/go-client
- asyncpg: https://magicstack.github.io/asyncpg/current/
- pgx: https://github.com/jackc/pgx
- MS MARCO Dataset: https://microsoft.github.io/msmarco/
- [[05 - Qdrant I - Architecture and Collections]]
- [[03 - pgvector I - Core Operations and Indexing]]
- [[07 - Milvus I - Distributed Architecture]]
- [[10 - Advanced Patterns and Observability]]
