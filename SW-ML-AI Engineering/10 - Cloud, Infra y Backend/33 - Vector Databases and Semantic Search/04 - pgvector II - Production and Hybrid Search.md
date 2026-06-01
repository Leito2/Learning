# 🔍 4 - pgvector II - Production and Hybrid Search

## 🎯 Learning Objectives
- Partition vector tables for horizontal scalability, query isolation, and partition pruning — essential for multi-tenant and time-series workloads
- Analyze vector query performance using `pg_stat_statements` and `EXPLAIN (ANALYZE, BUFFERS)` to identify missing indexes and tuning opportunities
- Implement hybrid search combining `tsvector` full-text ranking with `vector` similarity via Reciprocal Rank Fusion (RRF)
- Configure connection pooling with PgBouncer for high-throughput vector workloads without exhausting PostgreSQL connections
- Design backup and recovery strategies for vector-indexed PostgreSQL databases — physical backups for large, logical for small
- Monitor index bloat for `hnsw` and `ivfflat` indexes using `pgstattuple` and schedule `REINDEX CONCURRENTLY` proactively

## Introduction

Running `pgvector` on a development laptop with 10,000 vectors is trivial. Running it in production with 100 million vectors, thousands of queries per second, strict latency SLAs, and hybrid ranking requirements is a different discipline entirely. This note bridges the gap between "it works" and "it scales." We cover table partitioning (to isolate hot partitions and parallelize scans), hybrid search (to combine semantic and lexical signals), operational tooling (connection pooling, monitoring, backups), and index maintenance under churn.

Hybrid search is particularly important for ML engineers because pure vector search can miss exact keyword matches that users expect. A query for "OpenAI GPT-4o release date" may return semantically related blog posts but miss the exact announcement page because the semantic embedding of the announcement is similar to other AI news, not specifically about "release date." Combining `tsvector` full-text search with `vector` similarity — using Reciprocal Rank Fusion (RRF) — solves this by giving weight to both semantic and lexical signals.

This module builds on [[03 - pgvector I - Core Operations and Indexing]] and connects to [[15 - Docker and Kubernetes]] (containerized Postgres operations) and [[18 - MLOps and Model Serving]] (production SLAs and observability). The partitioning and monitoring patterns here also apply to any large PostgreSQL deployment, not just vector workloads.

---

## 1. Partitioning for Scale

### Theory

PostgreSQL declarative partitioning splits a logical table into physical child tables (partitions) based on a partition key. For vector workloads, the three most common strategies are:

1. **Range partitioning by time**: Segregate embeddings by ingestion date. Recent partitions are hot (frequently queried) and can reside on faster storage; old partitions are cold and can be archived or detached. This is ideal for time-series embedding collections like news articles, social media posts, or logs.

2. **List partitioning by tenant/category**: In multi-tenant SaaS applications, partition by `tenant_id` or `category`. This enables partition pruning — the query planner skips entire partitions that do not match the `WHERE` clause — dramatically reducing the effective search space.

3. **Hash partitioning by ID**: Distribute vectors evenly across partitions to parallelize sequential scans, index builds, and bulk loads. Useful when no natural partition key exists and you want to bound partition size.

Partitioning improves vector search in three ways. First, **pruning** reduces the dataset scanned when filters are present — a `WHERE created_at >= '2025-01-01'` skips months of irrelevant data entirely. Second, **maintenance isolation** lets you rebuild an `hnsw` index on one partition without affecting queries on others — critical for large tables where `REINDEX` can take hours. Third, **I/O isolation** prevents a bulk load into a new partition from thrashing the buffer cache of partitions serving live traffic. Each partition can also have different storage parameters: the hot partition on NVMe with a low `ef_search`, cold partitions on HDD with a higher `ef_search`.

**Partitioning strategy decisions:**

| Factor | Range (by time) | List (by tenant) | Hash (by ID) |
|---|---|---|---|
| Pruning | Excellent with time filters | Excellent if tenant always filtered | None |
| Data distribution | Even (if uniform ingestion) | Possibly skewed (big tenants) | Even |
| Old data archival | Natural (detach old partition) | Manual | Manual |
| Best for | News, logs, time-series | Multi-tenant SaaS | Workload balancing |

For vector workloads, **range partitioning by time** is the most common and most effective. It naturally supports the hot/cold data pattern and enables simple archival policies.

```sql
-- Range partitioning by month — ideal for time-series embeddings.
CREATE TABLE embeddings (
    id          BIGINT GENERATED ALWAYS AS IDENTITY,
    content_id  BIGINT NOT NULL,
    tenant_id   INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(768),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Monthly partitions.
CREATE TABLE embeddings_y2025m01 PARTITION OF embeddings
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE embeddings_y2025m02 PARTITION OF embeddings
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE embeddings_y2025m03 PARTITION OF embeddings
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Create hnsw indexes on each partition individually.
CREATE INDEX ON embeddings_y2025m01
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX ON embeddings_y2025m02
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX ON embeddings_y2025m03
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);

-- Insert routes to correct partition automatically.
INSERT INTO embeddings (content_id, tenant_id, created_at, embedding)
VALUES (1, 42, '2025-01-15', '[0.1, -0.2, ...]'::vector);

-- Query with partition pruning.
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, embedding <=> '[0.1, -0.2, ...]'::vector AS dist
FROM embeddings
WHERE created_at >= '2025-01-01' AND created_at < '2025-02-01'
ORDER BY dist
LIMIT 10;
-- Look for: "Partition Ref" showing only the relevant child table.
```

⚠️ **Forgetting to create indexes on every partition is a silent performance killer.** PostgreSQL does not automatically inherit indexes on partitions (pre-17). If you create an index only on the parent table, child partitions fall back to sequential scans. Postgres 17+ supports template indexes with `PARTITION BY`, but for older versions, always create indexes per partition.

💡 **Use a check script to verify index coverage after adding partitions.** Query `pg_indexes` for each partition and assert the expected indexes exist. This catches the "forgot to index the new partition" class of production incidents.

**Caso real — News aggregation platform:** A news aggregation service partitions article embeddings by publication month. During major events (elections, sports finals), query load spikes on the current month's partition. By isolating partitions, they can move the hot partition to a dedicated tablespace on NVMe storage while keeping historical partitions on cheaper SSD. They also detach old partitions (`ALTER TABLE ... DETACH PARTITION`) for archival without affecting query performance on the remaining partitions. The cold partitions are stored in compressed format and re-attached only if queried.

❌ **Antipattern: Using hash partitioning when range or list is more natural.** Hash partitioning scatters related data across partitions, defeating the purpose of partition pruning for most filtered queries. Use hash only when the partition key is a high-cardinality column with no natural ordering (e.g., a random UUID) and you need to bound individual partition size.

✅ **Always include the partition key in your `ORDER BY` and `WHERE` clauses.** The planner cannot prune partitions without a filter on the partition key. If you always query by `tenant_id`, use list partitioning by `tenant_id` — the pruning is automatic and dramatically reduces query cost.

---

## 2. Query Analysis and Performance Tuning

### Theory

Production vector databases require the same observability as any other OLTP or OLAP system. PostgreSQL provides `pg_stat_statements` to track query frequency, latency, and I/O metrics. For vector queries, you must look beyond average latency: a single unindexed `ORDER BY embedding <=> ... LIMIT 10` on a 100M-row table can take minutes and saturate I/O, evicting hot data from `shared_buffers` and degrading all concurrent queries.

Key metrics to monitor:
- **Mean and p99 latency** of top-k vector queries — track these over time to detect regression
- **Shared buffer hit ratio**: Vector scans are memory-bound; if `shared_buffers` is too small or the working set exceeds it, performance collapses
- **Index usage ratio**: From `pg_stat_user_indexes`, verify that `hnsw`/`ivfflat` indexes are being used for the expected queries
- **Temporary file usage**: Large sorts or hash joins spilling to disk indicate insufficient `work_mem` or a poorly planned query

`EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` is the definitive tool for vector query debugging. Check three things: (1) did the planner choose an index scan?, (2) how many buffers were read (should be low for an index scan), and (3) did the `Limit` node stop early (a small number of rows removed by recheck means the index is working efficiently)?

```sql
-- Enable pg_stat_statements (requires shared_preload_libraries + restart).
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Identify slow vector queries.
SELECT queryid, query, calls, mean_exec_time, shared_blks_hit, shared_blks_read
FROM pg_stat_statements
WHERE query LIKE '%embedding%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Detailed execution plan analysis.
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT id, embedding <=> '[0.1, -0.2, ...]'::vector AS dist
FROM embeddings
WHERE created_at >= '2025-01-01'
ORDER BY dist
LIMIT 10;

-- Check index usage statistics.
SELECT schemaname, relname, indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%embedding%';
-- If idx_scan is 0 but the table is queried often, the planner is not using the index.

-- Reset stats after tuning.
SELECT pg_stat_statements_reset();
```

⚠️ **Trusting `mean_exec_time` when vector queries have bimodal latency is dangerous.** Cached queries (vectors in `shared_buffers`) may take 5ms; cold queries (disk read) may take 500ms. The mean hides the tail. Always look at `stddev_exec_time` and consider using `pg_stat_kcache` for I/O breakdown.

💡 **Export `pg_stat_statements` to Prometheus/Grafana and visualize latency histograms, not just averages.** Set up an alert when p99 latency exceeds 2× the baseline for any vector query pattern.

**Caso real — Financial fraud detection:** A financial fraud detection team stored transaction embeddings in `pgvector`. During a production incident, `pg_stat_statements` revealed that a new batch job was running unfiltered `ORDER BY embedding <=> ... LIMIT 50` on a 200M-row table without an index — causing 30-second queries and cache eviction that degraded all other queries. They added an `hnsw` index and a `WHERE transaction_date > now() - interval '7 days'` filter, reducing latency from 30s to 12ms. The fix was identified from `pg_stat_statements` data within minutes of the incident.

❌ **Antipattern: Only checking average latency for capacity planning.** Vector queries often have a long tail — p99 can be 10× p50 due to buffer cache misses, partition skew, or concurrent load. Always dimension your cluster for p99, not p50.

✅ **Use `EXPLAIN (ANALYZE, BUFFERS)` after every schema or parameter change.** Before/after comparison of buffer reads is the fastest way to confirm that a new index or tuning parameter is working as expected.

---

## 3. Hybrid Search, Pooling, Backups, and Monitoring

### Theory

**Hybrid search** combines lexical matching (full-text search via `tsvector`) with semantic matching (vector similarity). Neither alone is sufficient: keyword search excels at exact entity names, dates, and IDs; vector search excels at conceptual similarity and paraphrase. A query for "crash report iOS 18.4" should match both documents containing those exact words AND conceptually similar documents about iOS stability, even if they use different terminology. The two most common fusion strategies are:

1. **Weighted linear combination**: $\text{score} = \alpha \cdot \text{norm\_ts\_rank} + (1-\alpha) \cdot \text{norm\_vector\_score}$. Simple but fragile — requires careful score normalization (z-score or min-max) and the weight $\alpha$ is task-dependent (tune on a validation set). A poor $\alpha$ choice can make results worse than either single stream.

2. **Reciprocal Rank Fusion (RRF)**: $\text{RRF}(d) = \sum_{r \in R(d)} \frac{1}{k + r}$, where $R(d)$ is the set of ranks document $d$ received from each ranking stream. RRF requires no hyperparameter tuning beyond $k=60$ (a robust default from the SIGIR 2009 paper) and handles heterogeneous score scales naturally because it operates on ranks rather than raw scores. This is the recommended fusion strategy for most production systems.

The RRF formula has an intuitive interpretation: a document ranked #1 in one stream contributes $\frac{1}{61} \approx 0.0164$ to the RRF score; a document ranked #10 contributes $\frac{1}{70} \approx 0.0143$. The difference between ranks shrinks as rank increases, meaning RRF is robust to outliers in one ranking stream — if a document is ranked #500 in one stream but #2 in another, it still scores well. The $k$ parameter acts as a dampening factor: higher $k$ reduces the influence of high rankings and produces more balanced fusion.

**Connection pooling** (PgBouncer) is critical because vector queries can be long-running compared to typical OLTP queries. Without pooling, a burst of concurrent vector searches can exhaust `max_connections`, causing cascading failures. PgBouncer's `transaction` pooling mode is the safest for PostgreSQL features.

**Backups** must capture both table data and index structures. `pg_dump` outputs `CREATE INDEX` as SQL, meaning restore rebuilds indexes from scratch — potentially taking hours for large `hnsw` indexes. For databases >100GB, physical backups (`pg_basebackup`, WAL archiving, filesystem snapshots) are mandatory.

**Index bloat** occurs when updates and deletes leave dead tuples in the index. Monitor `pgstattuple` and schedule `REINDEX` during maintenance windows if dead tuple percentage exceeds 30%.

```sql
-- Hybrid search schema with auto-generated tsvector.
CREATE TABLE documents (
    id      SERIAL PRIMARY KEY,
    title   TEXT,
    body    TEXT,
    embedding vector(768),
    search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);

-- Indexes for both search streams.
CREATE INDEX idx_fts ON documents USING GIN (search_vector);
CREATE INDEX idx_vec ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Hybrid query with Reciprocal Rank Fusion (RRF).
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
SELECT COALESCE(v.id, t.id) AS id,
       COALESCE(1.0 / (60 + v.v_rank), 0) + COALESCE(1.0 / (60 + t.t_rank), 0) AS rrf_score
FROM vector_results v
FULL OUTER JOIN text_results t ON v.id = t.id
ORDER BY rrf_score DESC
LIMIT 10;

-- Monitor index bloat.
CREATE EXTENSION IF NOT EXISTS pgstattuple;
SELECT * FROM pgstattuple('idx_vec');
-- tuple_percent vs dead_tuple_percent — schedule REINDEX if >30% dead.

-- REINDEX CONCURRENTLY (Postgres 12+) avoids table lock.
REINDEX INDEX CONCURRENTLY idx_vec;
```

```ini
; PgBouncer configuration (pgbouncer.ini)
; transaction pooling is safest — session pooling may break some features.
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

```bash
# Physical backup for large vector databases.
pg_basebackup -h localhost -D /backup/$(date +%Y%m%d) -X stream -P -U replicator

# Restore from physical backup.
# Copy the backup directory to the target server's data directory,
# then start PostgreSQL. Indexes are restored in their built state — no rebuild needed.
```

⚠️ **Using `pg_dump` to back up a multi-terabyte vector database is a disaster waiting to happen.** `pg_dump` serializes all data as SQL `INSERT` statements and then replays them, requiring index rebuild from scratch. For large `hnsw` indexes (>10M vectors), this can take days. Always use `pg_basebackup` or filesystem snapshots for databases >100GB.

💡 **"`pg_dump` is for schema and small data; `pg_basebackup` is for scale."** — Physically backing up index files preserves the built HNSW graph structure, which took hours to construct. Rebuilding from SQL dump discards that investment.

**Caso real — Supabase hybrid search:** Supabase (which offers managed `pgvector`) recommends hybrid search for their AI assistant use cases. Users often query with specific product names that pure semantic search misranks. By combining `tsvector` keyword matching with `hnsw` vector retrieval via RRF ($k=60$), they improved top-5 accuracy by 23% on their benchmark query set. Their implementation uses the `FULL OUTER JOIN` pattern above, with a fallback: if the hybrid CTE returns fewer than 5 results, they re-run with pure vector search and no limit to ensure coverage.

❌ **Antipattern: Ignoring `dead_tuple_percent` on `hnsw` indexes under heavy write load.** Unlike B-tree indexes, `hnsw` does not reclaim space efficiently from dead tuples. Heavy UPDATE/DELETE cycles can inflate index size by 2–3×. Monitor `pgstattuple` weekly and `REINDEX CONCURRENTLY` during maintenance windows.

✅ **Use `GENERATED ALWAYS AS ... STORED` for `tsvector` columns.** This guarantees `search_vector` stays in sync with `body` without application-level logic. Any change to `body` automatically updates the `tsvector` behind the scenes.

---

## 🎯 Key Takeaways

- **Partitioning** (range by time, list by tenant, hash by ID) is essential for scaling `pgvector` beyond single-table limits; partition pruning reduces the effective search space dramatically
- **Hybrid search** via RRF combines lexical and semantic signals without fragile score normalization — use $k=60$ as a robust default
- **PgBouncer** in `transaction` mode prevents connection exhaustion from concurrent long-running vector queries; set `default_pool_size` to match your CPU core count
- **Physical backups** (`pg_basebackup`) are mandatory for large vector databases because index rebuild times from `pg_dump` are prohibitive
- Monitor **index bloat** with `pgstattuple` weekly; `REINDEX CONCURRENTLY` when dead tuple percentage exceeds 30%
- Always validate index usage with `EXPLAIN (ANALYZE, BUFFERS)` after schema or parameter changes — a single missing index can cause 1000× latency degradation
- Use `GENERATED ALWAYS AS ... STORED` for `tsvector` to keep full-text indexes in sync automatically
- For backups: `pg_dump` for schema + small data (<100GB), `pg_basebackup` for large data

**Backup strategy comparison for vector databases:**

| Strategy | Size Limit | Restore Speed | Index State After Restore | Best For |
|---|---|---|---|---|
| `pg_dump` logical | <100GB | Slow (rebuilds indexes) | Needs rebuild | Dev/staging, schema-only |
| `pg_basebackup` physical | Any | Fast | Exact copy as backup time | Production >100GB |
| Filesystem snapshot (LVM/ZFS) | Any | Instant (copy-on-write) | Exact copy at snapshot time | Zero-downtime backups |
| WAL archiving + PITR | Any | Fast with replay | Point-in-time recoverable | DR with RPO < 1 hour |

For production vector databases, prefer `pg_basebackup` (simple, standard) or filesystem snapshots (faster, but require compatible storage). Always validate backups with a quarterly restore drill.

---

## References

- pgvector documentation: https://github.com/pgvector/pgvector
- PostgreSQL Partitioning: https://www.postgresql.org/docs/current/ddl-partitioning.html
- PgBouncer: https://www.pgbouncer.org/
- Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR, 2009
- [[03 - pgvector I - Core Operations and Indexing]] — Foundation operations and index creation
- [[15 - Docker and Kubernetes]] — Containerized deployment patterns
- [[18 - MLOps and Model Serving]] — Production SLAs and observability

## 📦 Código de compresión

```python
"""
pgvector II — Compression Script
Summarizes: partitioning, hybrid RRF, PgBouncer config, monitoring, backups.
"""
import psycopg2
from psycopg2.extras import RealDictCursor

DSN = "postgresql://user:pass@localhost:6432/vectordb"

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
            FROM vec v FULL OUTER JOIN txt t ON v.id = t.id
            ORDER BY rrf DESC
            LIMIT %s;
        """, (query_vec, query_vec, text_query, text_query, text_query, k))
        return cur.fetchall()

def check_bloat(conn, index_name: str):
    with conn.cursor() as cur:
        cur.execute("SELECT tuple_percent, dead_tuple_percent FROM pgstattuple(%s);", (index_name,))
        return cur.fetchone()

def get_top_slow_queries(conn):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT queryid, LEFT(query, 80) AS query_short,
                   calls, mean_exec_time, shared_blks_hit, shared_blks_read
            FROM pg_stat_statements
            WHERE query LIKE '%embedding%'
            ORDER BY mean_exec_time DESC
            LIMIT 5;
        """)
        return cur.fetchall()

if __name__ == "__main__":
    conn = psycopg2.connect(DSN)
    print("Hybrid search:", hybrid_search_rrf(conn, [0.1]*768, "machine learning"))
    print("Bloat:", check_bloat(conn, "idx_vec"))
    print("Slow queries:", get_top_slow_queries(conn))
    conn.close()
```
