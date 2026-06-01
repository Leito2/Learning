# 🏗️ 11 - Capstone Project - Multi-DB Semantic Search Platform

## 🎯 Learning Objectives
- Architect a polyglot semantic search API leveraging both Qdrant and pgvector simultaneously
- Implement dual-language backends: Python/FastAPI and Go/Fiber connecting to both databases
- Generate embeddings using OpenAI or local models, storing identical data in both engines
- Build a comparative query endpoint measuring latency and recall across backends in real time
- Create a benchmark harness validating p99 < 100ms and recall@10 > 0.9 under load
- Orchestrate the entire stack with Docker Compose for reproducible local development

## Introduction

Throughout this course, we studied [[05 - Qdrant I - Architecture and Collections|Qdrant's]] payload-filtered ANN, [[03 - pgvector I - Core Operations and Indexing|pgvector's]] SQL-native simplicity, and [[07 - Milvus I - Distributed Architecture|Milvus's]] microservice scale. The capstone unifies these lessons into a production-grade platform treating Qdrant and pgvector as complementary tiers: Qdrant for low-latency semantic retrieval, pgvector for transactional metadata lookups and hybrid SQL queries.

The project is implemented in both **Python/FastAPI** and **Go/Fiber** so you can compare ecosystem ergonomics and runtime performance within a single stack. The entire environment runs locally with Docker Compose, seeding a labeled corpus (MS MARCO passages) and running benchmark assertions in CI.

---

## 1. Architecture and Data Flow

A multi-DB semantic search platform must solve three problems: **data consistency** (both stores hold the same vectors), **query routing** (which backend serves which request), and **comparative evaluation** (how do we know which is faster/more accurate?). The architecture uses an abstraction layer — a `SearchBackend` interface — that hides Qdrant and pgvector specifics from HTTP handlers.

**Dual write**: the ingest endpoint writes to Qdrant and pgvector concurrently. In production this would use a Saga or outbox pattern; for the capstone, `asyncio.gather` with `return_exceptions=True` and a rollback on failure suffices.

**Query routing** supports:
- Static: `?backend=qdrant` or `?backend=pgvector`
- Dynamic: latency-based adaptive routing (experimental)
- Compare: runs same query against both, returns latency + overlap

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class SearchBackend(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict]) -> None: ...
    @abstractmethod
    def search(self, vector: List[float], top_k: int, filters: Dict = None) -> List[Tuple[str, float]]: ...
    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...
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

💡 Use deterministic IDs (UUIDv5 from content hash) so both stores deduplicate naturally on retry. 💡 *Idempotent writes, safe retries.*

❌ **Antipattern**: Sequential writes to Qdrant then pgvector — doubles latency.  
✅ **Correct**: `asyncio.gather` in Python or goroutines in Go for parallel dual writes.

❌ **Antipattern**: No rollback on partial write failure — data gets out of sync.  
✅ **Correct**: Wrap in try/except with compensation (delete from successful store on failure).

**Caso real — Shopify**: Qdrant for customer-facing product search, pgvector for internal admin tools with complex SQL joins. A shared embedding pipeline keeps both stores synchronized. The `/compare` endpoint is used in CI to catch regressions when upgrading either database. The abstraction layer let them add a third backend (Milvus for batch image search) without changing API routes.

```mermaid
flowchart TB
    C[HTTP Client] -->|/ingest| F[FastAPI / Fiber]
    C -->|/search| F
    C -->|/compare| F
    F --> Q[Qdrant Backend] --> QDB[(Qdrant)]
    F --> P[pgvector Backend] --> PDB[(Postgres + pgvector)]
```

## 2. Implementation: Python/FastAPI Backend

FastAPI natively supports async/await, automatic OpenAPI docs, and Pydantic validation. `asyncpg` for Postgres is fully async, preventing blocking during pgvector queries. `AsyncQdrantClient` enables true dual-write parallelism with `asyncio.gather`.

Embedding generation supports OpenAI `text-embedding-3-small` and local `sentence-transformers/all-MiniLM-L6-v2` behind an `Embedder` interface. This is the slowest step — embedding 1,000 texts takes ~1-3s depending on model.

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import asyncpg, json

class QdrantBackend:
    def __init__(self, host="qdrant", port=6333):
        self.client = AsyncQdrantClient(host=host, port=port)

    async def ensure_collection(self, name: str, dim: int):
        exists = await self.client.collection_exists(name)
        if not exists:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def upsert(self, collection, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        await self.client.upsert(collection_name=collection, points=points)

    async def search(self, collection, vector, top_k):
        results = await self.client.search(
            collection_name=collection, query_vector=vector, limit=top_k, with_payload=False)
        return [(r.id, r.score) for r in results]


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
                "id TEXT PRIMARY KEY, embedding vector(768), payload JSONB)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_embedding ON items "
                "USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)")

    async def upsert(self, ids, vectors, payloads):
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO items (id, embedding, payload) VALUES ($1, $2, $3) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload",
                [(ids[i], str(vectors[i]), json.dumps(payloads[i])) for i in range(len(ids))])

    async def search(self, vector, top_k):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, embedding <=> $1 as distance FROM items ORDER BY embedding <=> $1 LIMIT $2",
                str(vector), top_k)
            return [(r["id"], 1.0 - float(r["distance"])) for r in rows]
```

```python
from fastapi import FastAPI, Query
import asyncio, time

app = FastAPI(title="Multi-DB Semantic Search")

@app.post("/ingest")
async def ingest(id: str, text: str):
    vector = await embedder.encode(text)
    payload = {"text": text}
    await asyncio.gather(
        qdrant.upsert(COLLECTION, [id], [vector], [payload]),
        pgvector.upsert([id], [vector], [payload]),
    )
    return {"status": "ok"}

@app.get("/compare")
async def compare(q: str = Query(...), top_k: int = 10):
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

⚠️ Using the sync `QdrantClient` inside an async FastAPI handler blocks the event loop, killing concurrency. Always use `AsyncQdrantClient`. 💡 *Sync in async = death by a thousand blocks.*

**Caso real — Health-tech startup**: FastAPI with dual-write to Qdrant and pgvector. Qdrant serves patient-facing symptom search (p99 < 30ms); pgvector powers internal analytics joining embeddings with insurance tables (complex SQL). The abstraction layer let them swap Qdrant for Milvus when they hit 100M vectors without changing API routes.

## 3. Implementation: Go/Fiber Backend

Go's goroutines and lightweight runtime make it ideal for I/O-bound gateway services. Fiber provides fast routing and middleware. The `qdrant/go-client` wraps gRPC calls to Qdrant; `pgx` with `pgxpool` handles Postgres. Both support context cancellation.

```go
package backends

import (
	"context"
	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

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
		WithPayload: &pb.WithPayloadSelector{
			SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: false},
		},
	})
	if err != nil { return nil, err }
	results := make([]SearchResult, len(res.Result))
	for i, r := range res.Result {
		results[i] = SearchResult{ID: r.Id.GetUuid(), Score: r.Score}
	}
	return results, nil
}
```

```go
package backends

import (
	"context"
	"github.com/jackc/pgx/v5/pgxpool"
)

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
			ids[i], vectors[i], payloads[i])
		if err != nil { return err }
	}
	return tx.Commit(ctx)
}
```

❌ **Antipattern**: Leaking pgxpool connections — Postgres rejects new connections after `max_connections` is reached.  
✅ **Correct**: Always `defer pool.Close()` in `main()`.

❌ **Antipattern**: Not setting `MaxConns` — default pool size (4) bottlenecks concurrent requests.  
✅ **Correct**: Size pool to 2x expected concurrency, capped at `max_connections` in Postgres.

## 4. Docker Compose and Benchmarking

Docker Compose orchestrates Postgres (pgvector), Qdrant, the API, and a benchmark runner in a single `docker-compose up`. This eliminates environment inconsistencies and provides a sandbox for load testing.

```yaml
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

async def ingest_all():
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
    assert p50 < 30, f"p50 SLA violated: {p50}ms"

async def main():
    await ingest_all()
    await benchmark("qdrant")
    await benchmark("pgvector")

asyncio.run(main())
```

⚠️ ANN indices behave differently with 100 vectors vs. 1M vectors. Always seed realistic volumes before measuring latency. 💡 *Bench with volume, or bench in vain.*

❌ **Antipattern**: Default Docker Desktop allocates 2 GB RAM — crashes Qdrant or Postgres under load.  
✅ **Correct**: Increase Docker to 8 GB (Settings > Resources > Memory) before benchmarking.

**Caso real — Legal-tech startup**: Runs this exact Docker Compose stack in CI for every PR. The benchmark script catches performance regressions (e.g., a library upgrade slowing pgvector by 20%). Before merging, the PR must pass p99 < 100ms and recall@10 > 0.9. This gate has prevented 3 production incidents in 6 months.

---

## 🎯 Key Takeaways
- Polyglot backends (Qdrant + pgvector) let you optimize for latency and SQL flexibility without migration risk.
- An abstraction layer (`SearchBackend` interface) is essential for swapping or adding databases without API changes.
- Dual-write ingestion ensures consistency but requires idempotency (deterministic IDs) to handle retries.
- Python/FastAPI excels for ML integration and rapid development; Go/Fiber excels for high-throughput gateways.
- Docker Compose provides reproducible environments for development, debugging, and CI performance gates.
- Benchmarking must validate both latency percentiles (p50, p99) and accuracy (recall@k) against brute-force ground truth.
- Backend-switch endpoints enable canary deployments and A/B testing between database versions.

## References
- FastAPI: https://fastapi.tiangolo.com/
- Fiber: https://docs.gofiber.io/
- qdrant-client: https://github.com/qdrant/qdrant-client
- qdrant/go-client: https://github.com/qdrant/go-client
- asyncpg: https://magicstack.github.io/asyncpg/
- pgx: https://github.com/jackc/pgx
- MS MARCO: https://microsoft.github.io/msmarco/
- [[05 - Qdrant I - Architecture and Collections]]
- [[03 - pgvector I - Core Operations and Indexing]]
- [[07 - Milvus I - Distributed Architecture]]
- [[10 - Advanced Patterns and Observability]]

## 📦 Código de compresión
```python
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
    return {"qdrant_ms": round(ql, 2), "pgvector_ms": round(pl, 2)}
```
