# 🚀 pgvector II - Production and Hybrid Search

## 🎯 Learning Objectives

- Partition vector tables for horizontal scalability and query isolation
- Analyze vector query performance using `pg_stat_statements` and execution plans
- Implement hybrid search combining `tsvector` full-text ranking with `vector` similarity
- Configure connection pooling with PgBouncer for high-throughput vector workloads
- Design backup and recovery strategies for vector-indexed PostgreSQL databases
- Monitor index bloat and vacuum health for `hnsw` and `ivfflat` indexes

## Introduction

Running `pgvector` on a development laptop with 10,000 vectors is trivial. Running it in production with 100 million vectors, thousands of queries per second, strict latency SLAs, and hybrid ranking requirements is a different discipline entirely. This note bridges the gap between "it works" and "it scales." We cover table partitioning (to isolate hot data and parallelize scans), hybrid search (to combine semantic and lexical signals), operational tooling (pooling, monitoring, backups), and index maintenance.

Hybrid search is particularly important for ML engineers because pure vector search can miss exact keyword matches that users expect. A query for "OpenAI GPT-4o release date" may return semantically related blog posts but miss the exact announcement page. Combining `tsvector` full-text search with `vector` similarity — using weighted sums or Reciprocal Rank Fusion (RRF) — solves this.

This module builds on [[03 - pgvector I - Core Operations and Indexing]] and connects to [[15 - Docker and Kubernetes]] (containerized Postgres operations) and [[18 - MLOps and Model Serving]] (production SLAs and observability).

---

## Module 1: Partitioning for Scale

### 1.1 Theoretical Foundation 🧠

PostgreSQL declarative partitioning splits a logical table into physical child tables (partitions) based on a partition key. For vector workloads, the most common strategies are:

1. **Range partitioning by time**: Segregate embeddings by ingestion date. Recent partitions are hot (frequently queried) and can reside on faster storage; old partitions are cold and can be archived or compressed.
2. **List partitioning by tenant/category**: In multi-tenant SaaS applications, partition by `tenant_id` or `category`. This enables partition pruning — the query planner skips entire partitions that do not match the `WHERE` clause.
3. **Hash partitioning by ID**: Distribute vectors evenly across partitions to parallelize sequential scans and index builds.

Partitioning improves vector search in three ways. First, **pruning** reduces the dataset scanned when filters are present. Second, **maintenance** becomes localized: you can rebuild an `hnsw` index on one partition without affecting others. Third, **I/O isolation** prevents a bulk load into a new partition from thrashing the cache of partitions serving live queries.

### 1.2 Mental Model 📐

```
┌─────────────────────────────────────────────┐
│  Logical Table: embeddings                  │
│  Partitioned by RANGE (created_at)          │
│                                             │
│  ┌─────────────────┐                        │
│  │  embeddings_p1  │ 2023 data (cold)       │
│  │  embeddings_p2  │ 2024 data (warm)       │
│  │  embeddings_p3  │ 2025 data (hot)        │
│  └─────────────────┘                        │
│                                             │
│  Query: WHERE created_at > '2025-01-01'     │
│  ──► Planner skips p1, p2                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Multi-tenant Partitioning by tenant_id     │
│                                             │
│  ┌─────────────────┐                        │
│  │  embeddings_t1  │ tenant A               │
│  │  embeddings_t2  │ tenant B               │
│  │  embeddings_t3  │ tenant C               │
│  └─────────────────┘                        │
│                                             │
│  Query: WHERE tenant_id = 'B'               │
│  ──► Only scan embeddings_t2                │
│  ──► Index on t2 is much smaller            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Partition Pruning + Vector Index           │
│                                             │
│  WHERE category = 'ML' AND embedding <=> q  │
│       │                    │                │
│       ▼                    ▼                │
│  prune to ML          hnsw scan on ML       │
│  partition            partition only        │
└─────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```sql
-- WHY: Range partitioning by month is ideal for time-series embeddings
--      such as news articles, logs, or social media posts.
CREATE TABLE embeddings (
    id          BIGINT GENERATED ALWAYS AS IDENTITY,
    content_id  BIGINT NOT NULL,
    tenant_id   INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(768),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions.
CREATE TABLE embeddings_y2025m01 PARTITION OF embeddings
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE embeddings_y2025m02 PARTITION OF embeddings
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- WHY: Indexes must be created on each partition individually
--      (or use template index in Postgres 17+).
CREATE INDEX ON embeddings_y2025m01
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX ON embeddings_y2025m02
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Insert routes automatically to the correct partition.
INSERT INTO embeddings (content_id, tenant_id, created_at, embedding)
VALUES (1, 42, '2025-01-15', ARRAY(SELECT random() FROM generate_series(1, 768))::vector);

-- Query with partition pruning.
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, embedding <=> query_vec AS dist
FROM embeddings, (SELECT '[0.1, -0.2, ...]'::vector AS query_vec) q
WHERE created_at >= '2025-01-01' AND created_at < '2025-02-01'
ORDER BY dist
LIMIT 10;
-- Look for: "Partition Ref" showing only the relevant child table.
```

### 1.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A[Logical Table: embeddings] --> B[Partition Selector]
    B --> C[embeddings_y2025m01]
    B --> D[embeddings_y2025m02]
    B --> E[embeddings_y2025m03]
    C --> F[hnsw index on C]
    D --> G[hnsw index on D]
    E --> H[hnsw index on E]
```

```mermaid
graph LR
    Q[Query + Time Filter] --> P{Partition Pruner}
    P -->|skip| C1[Old Partitions]
    P -->|scan| C2[Matching Partition]
    C2 --> I[hnsw Index]
    I --> R[Results]
```

![Partitioning](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Postgresql_elephant.svg/440px-Postgresql_elephant.svg.png)

### 1.5 Application in ML/AI Systems 🤖

Real case: **A news aggregation platform** partitions their article embeddings by publication month. During major events (elections, sports finals), query load spikes on the current month's partition. By isolating partitions, they can scale the hot partition to a dedicated tablespace on NVMe while keeping historical partitions on cheaper SSD. They also detach old partitions for archival without affecting query performance.

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| Time-series RAG | Range partition by ingestion week | Query only recent documents; archive old ones cheaply |
| Multi-tenant embeddings | List partition by tenant_id | Enforce data isolation and per-tenant index sizing |
| Batch re-embedding | Hash partition by content_id | Parallelize index rebuilds across partitions |

### 1.6 Common Pitfalls ⚠️

⚠️ **Pitfall: Forgetting to create indexes on every partition.** Root cause: `pgvector` indexes are not automatically inherited by partitions in PostgreSQL. If you create an index only on the parent table (pre-17), child partitions will do sequential scans. Always create indexes per partition, or use the `PARTITION BY` template index feature in PostgreSQL 17+.

💡 **Mnemonic: "Partitions share schema, not indexes."** — After adding a new partition, run a check script that verifies all expected indexes exist.

### 1.7 Knowledge Check ❓

1. You have 500M embeddings ingested over 3 years. Design a partitioning strategy that minimizes query latency for "last 30 days" searches and explain how partition pruning works in the planner.
2. Why must the partition key appear in the primary key (or a unique index) for partitioned tables?
3. Write the SQL to detach a partition named `embeddings_y2023m01` and attach a new `embeddings_y2025m06` partition.

---

## Module 2: Query Analysis and Performance Tuning

### 2.1 Theoretical Foundation 🧠

Production vector databases require the same observability as any other OLTP or OLAP system. PostgreSQL provides `pg_stat_statements` to track query frequency, latency, and I/O. For vector queries, you must look beyond average time: a single unindexed `ORDER BY embedding <=> ... LIMIT 10` on a 100M-row table can take minutes and saturate I/O.

Key metrics to monitor:
- **Mean and p99 latency** of top-k vector queries
- **Shared buffer hit ratio**: Vector scans are memory-bound; if vectors are evicted from shared_buffers, performance collapses.
- **Index usage ratio**: From `pg_stat_user_indexes`, verify that `hnsw`/`ivfflat` indexes are being used.
- **Temporary file usage**: Large sorts or joins may spill to disk.

`EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` is the definitive tool. For vector queries, check whether the planner chose an index scan, how many buffers were read, and whether the `Limit` node reduced work early.

### 2.2 Mental Model 📐

```
┌─────────────────────────────────────────────┐
│  pg_stat_statements Vector for pgvector     │
│                                             │
│  queryid │ calls │ mean_time │ shared_blks  │
│  ────────┼───────┼───────────┼───────────── │
│  12345   │ 10k   │ 45ms      │ 1200         │
│  12346   │ 500   │ 8500ms    │ 500000       │
│           ^            ^            ^        │
│        suspicious   disaster   buffer miss   │
│                                             │
│  → Investigate queryid 12346 immediately    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  EXPLAIN ANALYZE Readout for Vector Search  │
│                                             │
│  Limit                                      │
│    -> Index Scan using hnsw_idx             │
│         Index Cond: (embedding <=> q)       │
│         Rows Removed by Index Recheck: 0    │
│         Buffers: shared hit=45              │
│                                             │
│  ✅ Good: index used, few buffers           │
└─────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```sql
-- WHY: pg_stat_statements must be added to shared_preload_libraries
--      and requires a restart. It is essential for production.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Identify slow vector queries.
SELECT queryid, query, calls, mean_exec_time, shared_blks_hit, shared_blks_read
FROM pg_stat_statements
WHERE query LIKE '%embedding%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Reset stats after tuning.
SELECT pg_stat_statements_reset();

-- Detailed execution plan for a vector query.
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, embedding <=> '[0.1, -0.2, ...]'::vector AS dist
FROM embeddings
ORDER BY dist
LIMIT 10;

-- Check index usage statistics.
SELECT schemaname, relname, indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%embedding%';

-- WHY: If idx_scan is 0 but the table is queried, the planner is
--      either not finding the index useful or the operator class mismatches.
```

### 2.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A[Application Query] --> B[PostgreSQL]
    B --> C[pg_stat_statements]
    C --> D[Slow Query Log]
    D --> E[EXPLAIN ANALYZE]
    E --> F{Index Used?}
    F -->|Yes| G[Tune ef_search / probes]
    F -->|No| H[Fix Index / Operator Class]
```

```mermaid
xychart-beta
    title "Query Latency vs. Collection Size (hnsw)"
    x-axis [10k, 100k, 1M, 10M, 100M]
    y-axis "Latency (ms)" 0 --> 200
    line [2, 4, 8, 15, 35]
```

![PostgreSQL stats](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Postgresql_elephant.svg/440px-Postgresql_elephant.svg.png)

### 2.5 Application in ML/AI Systems 🤖

Real case: **A financial fraud detection team** stores transaction embeddings in `pgvector`. During a production incident, `pg_stat_statements` revealed that a new batch job was running unfiltered `ORDER BY embedding <=> ... LIMIT 50` on a 200M-row table without an index — causing 30-second queries and cache eviction. They added an `hnsw` index and a `WHERE transaction_date > now() - interval '7 days'` filter, reducing latency to 12ms.

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| Latency regression detection | `pg_stat_statements` | Catch missing indexes before users complain |
| Capacity planning | Buffer hit ratio | Size `shared_buffers` to fit hot vector partitions |
| Query optimization | `EXPLAIN (BUFFERS)` | Verify that LIMIT is pushed into the index scan |

### 2.6 Common Pitfalls ⚠️

⚠️ **Pitfall: Trusting `mean_exec_time` when vector queries have bimodal latency.** Root cause: Cached queries (vectors in shared_buffers) may take 5ms; cold queries (disk read) may take 500ms. The mean hides the tail. Always look at `stddev_exec_time` and percentiles, or use `pg_stat_statements` with extension wrappers like `pg_stat_kcache` for I/O latency breakdown.

💡 **Mnemonic: "Mean lies; histograms don't."** — Export `pg_stat_statements` into a metrics system (Prometheus/Grafana) and visualize latency distributions, not averages.

### 2.7 Knowledge Check ❓

1. You observe `shared_blks_read` is high for a vector query that should be indexed. What three configuration or schema issues could cause this?
2. How would you calculate the total memory required to keep a 100M-vector `hnsw` index and its vectors in `shared_buffers`?
3. Write a query that finds the top-5 most frequent vector search queries and their average latency.

---

## Module 3: Hybrid Search, Pooling, Backups, and Monitoring

### 3.1 Theoretical Foundation 🧠

**Hybrid search** combines lexical matching (full-text search via `tsvector`) with semantic matching (vector similarity). Neither alone is sufficient: keyword search excels at exact entity names, dates, and IDs; vector search excels at conceptual similarity and paraphrase. Two common fusion strategies are:

1. **Weighted linear combination**: `score = α * norm_ts_rank + (1-α) * norm_vector_score`. Simple and fast, but requires careful normalization and the weight α is task-dependent.
2. **Reciprocal Rank Fusion (RRF)**: `score = Σ 1 / (k + rank_i)` for each result's rank in each ranking stream. RRF requires no hyperparameter tuning (beyond `k=60` as a robust default) and handles heterogeneous score scales naturally. It is the recommended approach for most hybrid pipelines.

**Connection pooling** (PgBouncer) is critical because vector queries can be long-running compared to typical OLTP queries. Without pooling, a burst of concurrent vector searches can exhaust Postgres's `max_connections`, causing cascading failures.

**Backups** must capture both the table data and the index structures. `pgvector` indexes are stored in the same physical files as standard Postgres indexes, so `pg_basebackup` and `pg_dump` handle them transparently. However, `pg_dump` outputs the index definition as SQL (`CREATE INDEX ...`), meaning restore will rebuild the index from scratch — potentially taking hours for large `hnsw` indexes. For large databases, physical backups (`pg_basebackup`, WAL archiving, snapshots) are preferred.

**Index bloat** occurs when updates and deletes leave dead tuples in the index. `pgvector` `hnsw` indexes do not currently support VACUUM-driven deduplication as effectively as B-trees, so heavy churn can inflate index size. Monitor `pgstattuple` and schedule `REINDEX` during maintenance windows if bloat exceeds 30%.

### 3.2 Mental Model 📐

```
┌─────────────────────────────────────────────┐
│  Hybrid Search: Two Streams, One Ranking    │
│                                             │
│  Text Query: "Python asyncio tutorial"      │
│                                             │
│  Stream A (tsvector):                       │
│    rank 1: doc #42  (ts_rank = 0.95)        │
│    rank 2: doc #7   (ts_rank = 0.80)        │
│                                             │
│  Stream B (vector):                         │
│    rank 1: doc #7   (cos_dist = 0.12)       │
│    rank 2: doc #42  (cos_dist = 0.20)       │
│                                             │
│  RRF(k=60):                                 │
│    doc #42: 1/(60+1) + 1/(60+2) = 0.0325    │
│    doc #7:  1/(60+2) + 1/(60+1) = 0.0325    │
│    (tie; break by secondary signal)         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  PgBouncer Pooling Architecture             │
│                                             │
│  App ──► PgBouncer ──► PostgreSQL           │
│  500        20              20              │
│  conns    pooled           backends         │
│                                             │
│  WHY: Vector queries are longer than OLTP;  │
│       without pooling, burst traffic exhausts │
│       Postgres connections.                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Backup Strategy Matrix                     │
│                                             │
│  Small DB (<100GB):  pg_dump + restore      │
│       → index rebuild acceptable            │
│                                             │
│  Large DB (>1TB):    pg_basebackup + WAL    │
│       → physical copy, fast restore         │
│       → index files included verbatim       │
└─────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```sql
-- Hybrid Search Schema
CREATE TABLE documents (
    id      SERIAL PRIMARY KEY,
    title   TEXT,
    body    TEXT,
    embedding vector(768),
    -- WHY: tsvector is generated automatically to stay in sync with body.
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);

-- Full-text index
CREATE INDEX idx_fts ON documents USING GIN (search_vector);

-- Vector index
CREATE INDEX idx_vec ON documents USING hnsw (embedding vector_cosine_ops);

-- Hybrid query with RRF
-- WHY: CTEs let us compute ranks independently before fusion.
WITH vector_results AS (
    SELECT id,
           embedding <=> '[0.1, -0.2, ...]'::vector AS dist,
           row_number() OVER (ORDER BY embedding <=> '[0.1, -0.2, ...]'::vector) AS v_rank
    FROM documents
    ORDER BY dist
    LIMIT 100
),
text_results AS (
    SELECT id,
           ts_rank_cd(search_vector, plainto_tsquery('english', 'asyncio tutorial')) AS t_score,
           row_number() OVER (ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', 'asyncio tutorial')) DESC) AS t_rank
    FROM documents
    WHERE search_vector @@ plainto_tsquery('english', 'asyncio tutorial')
    ORDER BY t_score DESC
    LIMIT 100
)
SELECT
    COALESCE(v.id, t.id) AS id,
    COALESCE(1.0 / (60 + v.v_rank), 0) + COALESCE(1.0 / (60 + t.t_rank), 0) AS rrf_score
FROM vector_results v
FULL OUTER JOIN text_results t ON v.id = t.id
ORDER BY rrf_score DESC
LIMIT 10;

-- Monitoring index bloat with pgstattuple
CREATE EXTENSION IF NOT EXISTS pgstattuple;
SELECT * FROM pgstattuple('idx_vec');
-- Look at: tuple_percent vs dead_tuple_percent
```

```ini
; PgBouncer configuration snippet (pgbouncer.ini)
; WHY: transaction pooling is safest for Postgres features
;      but may conflict with prepared statements.
[databases]
vectordb = host=localhost port=5432 dbname=vectordb

[pgbouncer]
listen_port = 6432
listen_addr = 0.0.0.0
auth_type = md5
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
reserve_pool_size = 5
```

### 3.4 Visual Representation 🖼️

```mermaid
flowchart TD
    Q[Query: "Python asyncio"] --> A[Text Search Stream]
    Q --> B[Vector Search Stream]
    A --> C[Ranked Text Results]
    B --> D[Ranked Vector Results]
    C --> E[RRF Fusion]
    D --> E
    E --> F[Final Top-10]
```

```mermaid
graph LR
    A[Application] --> B[PgBouncer]
    B --> C1[Postgres Backend 1]
    B --> C2[Postgres Backend 2]
    B --> C3[Postgres Backend N]
```

![RRF formula](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Postgresql_elephant.svg/440px-Postgresql_elephant.svg.png)

### 3.5 Application in ML/AI Systems 🤖

Real case: **Supabase** (which offers managed `pgvector`) recommends hybrid search for their AI assistant use cases. Users often query with specific product names that semantic search alone misranks. By combining `tsvector` keyword matching with `hnsw` vector retrieval via RRF, they improved top-5 accuracy by 23% on their benchmark query set.

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| RAG with exact citations | Hybrid RRF | Retrieve both conceptually similar and keyword-matching chunks |
| E-commerce search | tsvector + vector fusion | Match "Nike Air Max" exactly while also showing similar styles |
| High-throughput API | PgBouncer pooling | Serve 1000+ concurrent embedding lookups without connection exhaustion |
| Disaster recovery | pg_basebackup + WAL | Restore terabyte-scale vector DB in minutes, not days |

### 3.6 Common Pitfalls ⚠️

⚠️ **Pitfall: Using `pg_dump` to back up a multi-terabyte vector database.** Root cause: `pg_dump` outputs schema + data as SQL. Restoring requires re-inserting all vectors and rebuilding all `hnsw` indexes — a process that can take days for large collections and may fail midway. For production vector databases >100GB, always use physical backups (`pg_basebackup`, filesystem snapshots, or managed backup services).

💡 **Mnemonic: "pg_dump is for logic; pg_basebackup is for scale."** — Treat vector indexes like large B-trees: their physical structure is worth preserving.

### 3.7 Knowledge Check ❓

1. Explain why RRF does not require normalizing the scores from the text and vector streams. What mathematical property makes this possible?
2. You have 500 concurrent vector search clients. PostgreSQL `max_connections` is 100. What `pool_mode` in PgBouncer should you use, and what is the trade-off versus `session` pooling?
3. Write a shell command using `pg_basebackup` to create a streaming replication base backup suitable for point-in-time recovery of a `pgvector` database.

---

## 📦 Compression Code

```python
"""
pgvector II — Compression Script
Summarizes: partitioning, hybrid RRF, PgBouncer, monitoring, backups.
"""
import psycopg2
from psycopg2.extras import RealDictCursor

DSN = "postgresql://user:pass@localhost:6432/vectordb"  # PgBouncer port

def hybrid_search_rrf(conn, query_vec: list[float], text_query: str, k: int = 10):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            WITH vec AS (
                SELECT id,
                       embedding <=> %s::vector AS dist,
                       row_number() OVER (ORDER BY embedding <=> %s::vector) AS r
                FROM documents
                ORDER BY dist
                LIMIT 100
            ),
            txt AS (
                SELECT id,
                       ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS score,
                       row_number() OVER (ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC) AS r
                FROM documents
                WHERE search_vector @@ plainto_tsquery('english', %s)
                LIMIT 100
            )
            SELECT COALESCE(v.id, t.id) AS id,
                   COALESCE(1.0/(60+v.r), 0) + COALESCE(1.0/(60+t.r), 0) AS rrf
            FROM vec v
            FULL OUTER JOIN txt t ON v.id = t.id
            ORDER BY rrf DESC
            LIMIT %s;
        """, (query_vec, query_vec, text_query, text_query, text_query, k))
        return cur.fetchall()

def check_bloat(conn, index_name: str):
    with conn.cursor() as cur:
        cur.execute("SELECT tuple_percent, dead_tuple_percent FROM pgstattuple(%s);", (index_name,))
        return cur.fetchone()

if __name__ == "__main__":
    conn = psycopg2.connect(DSN)
    print(hybrid_search_rrf(conn, [0.1]*768, "machine learning"))
    print(check_bloat(conn, "idx_vec"))
    conn.close()
```

## 🎯 Documented Project

**Project: Production-Ready Hybrid RAG Backend**

- **Description**: A backend service that ingests Markdown documentation, chunks it, embeds it, and serves hybrid semantic + full-text search over a partitioned `pgvector` table.
- **Functional Requirements**:
  - Partition the `chunks` table by `source_id` (list partitioning) for multi-repo isolation.
  - Build `hnsw` indexes per partition.
  - Expose `/search?q=...&mode=hybrid|vector|text` endpoint.
  - Hybrid mode uses RRF with `k=60`; pure vector uses `ef_search=128`.
  - Deploy behind PgBouncer with `transaction` pooling.
- **Main Components**:
  - `ingest.py`: chunk Markdown, embed with `sentence-transformers`, bulk-insert with `COPY`.
  - `search.py`: SQL generation for vector, text, and hybrid modes.
  - `api.py`: FastAPI with connection pooling via `psycopg2.pool.ThreadedConnectionPool`.
  - `ops/`: PgBouncer config, `pg_basebackup` cron script, Grafana dashboard JSON.
- **Success Metrics**:
  - p99 hybrid search latency <80ms on 10M chunks.
  - Recall@10 ≥92% on labeled benchmark vs. exact KNN baseline.
  - Zero-downtime backup and restore validated quarterly.

## 🎯 Key Takeaways

- **Partitioning** (range, list, hash) is essential for scaling `pgvector` beyond single-table limits; it enables pruning and parallel maintenance.
- **Hybrid search** via RRF combines lexical and semantic signals without fragile score normalization; it is the production standard for RAG and e-commerce.
- **PgBouncer** in `transaction` mode prevents connection exhaustion from concurrent long-running vector queries.
- **Physical backups** (`pg_basebackup`) are mandatory for large vector databases because index rebuild times are prohibitive.
- Monitor **index bloat** with `pgstattuple` and schedule `REINDEX` during maintenance windows.
- Always validate index usage with `EXPLAIN (ANALYZE, BUFFERS)` after schema or parameter changes.

## References

- pgvector documentation: https://github.com/pgvector/pgvector
- PostgreSQL Partitioning: https://www.postgresql.org/docs/current/ddl-partitioning.html
- PgBouncer: https://www.pgbouncer.org/
- Cormack, Clarke, and Buettcher. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR, 2009.
- [[03 - pgvector I - Core Operations and Indexing]] — Foundation operations and index creation.
- [[15 - Docker and Kubernetes]] — Containerized deployment patterns.
