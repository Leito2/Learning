# 🔍 3 - pgvector I - Core Operations and Indexing

## 🎯 Learning Objectives
- Install and configure the `pgvector` extension in PostgreSQL 15+
- Create and manipulate the `vector(n)` type for storing dense embeddings
- Perform CRUD operations on vector columns and understand distance semantics
- Build and tune `ivfflat` and `hnsw` indexes with correct parameters
- Use distance operators `<->` (L2), `<=>` (cosine), and `<#>` (negative inner product) correctly
- Adjust `hnsw.ef_search` and `ivfflat.probes` at query time for recall/latency trade-offs
- Validate index usage with `EXPLAIN ANALYZE` to detect silent fallback to sequential scans

## Introduction

`pgvector` is an open-source PostgreSQL extension that transforms the world's most trusted relational database into a production-grade vector database. Unlike standalone vector engines (Qdrant, Milvus, Pinecone), `pgvector` allows you to store embeddings alongside transactional business data — user profiles, product metadata, access control lists — in the same ACID-compliant table. This eliminates the complexity of dual-write patterns, distributed transactions, and eventual consistency between a relational store and a separate vector index.

Why does this matter for ML engineers? In production, vector search rarely happens in isolation. A recommendation query needs to join `user_vectors` with `user_demographics`, `product_inventory`, and `purchase_history`. With a standalone vector DB, this requires application-level orchestration: fetch nearest vectors, then query PostgreSQL by IDs. With `pgvector`, the query planner optimizes the entire operation — it can push down `WHERE` filters before the vector search, reduce the candidate set via partition pruning, and return joined results in a single round-trip.

This note covers the foundational operations: installation, the `vector` type, basic querying, and index creation with `ivfflat` and `hnsw`. We focus on the two index families available in `pgvector` and teach you how to tune them with session-level parameters like `hnsw.ef_search`. The next note, [[04 - pgvector II - Production and Hybrid Search]], scales these foundations to partitioned tables, hybrid full-text + vector queries, and operational monitoring with `pg_stat_statements`. This module also connects to [[12 - SQL Mastery]] (PostgreSQL fundamentals) and [[15 - Docker and Kubernetes]] (containerized deployment of Postgres + pgvector).

---

## 1. Installation and the `vector` Type

### Theory

PostgreSQL's extensibility architecture allows third-party code to register new types, operators, and index access methods without modifying the core server. `pgvector` leverages this by adding three components:

1. A `vector(n)` composite type backed by a fixed-length array of `float4` (32-bit IEEE 754 floats)
2. Three distance operators registered with the PostgreSQL operator class framework: `<->` (L2), `<=>` (cosine), `<#>` (negative inner product)
3. Index access methods (`ivfflat`, `hnsw`) that plug into PostgreSQL's planner and executor via cost model callbacks

The `vector(n)` type is stored in-line for small $n$ (≤DimensionsToInline, default 1024) and out-of-line via TOAST (The Oversized-Attribute Storage Technique) for larger dimensions. For standard embedding sizes (384–1536), vectors live directly in the heap tuple, making sequential scans efficient and index-only scans possible when the vector column is covered by the index. The maximum dimensionality in `pgvector` 0.7+ is 16,000.

```sql
-- Install extension (database-scoped, not server-level).
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table with a vector column.
-- WHY: vector(768) matches BERT-base / Sentence-T5 embedding dimension.
--      The dimension is enforced at insert time — silent corruption prevented.
CREATE TABLE articles (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT,
    embedding   vector(768)
);

-- Insert: vector literals use ARRAY::vector or string form '[a,b,c]'.
INSERT INTO articles (title, content, embedding)
VALUES (
    'Introduction to pgvector',
    'A guide to vector search in Postgres...',
    ARRAY(SELECT random() FROM generate_series(1, 768))::vector
);

-- Alternative string literal form (easier to generate from Python/Go).
INSERT INTO articles (title, embedding)
VALUES ('Demo', '[0.1, -0.3, 0.8, 0.02, ...]'::vector);

-- Update: replace an embedding after re-encoding with a new model version.
UPDATE articles
SET embedding = ARRAY(SELECT random() FROM generate_series(1, 768))::vector
WHERE id = 1;

-- Delete with automatic index maintenance.
DELETE FROM articles WHERE id = 1;
```

```python
import psycopg2
import numpy as np

conn = psycopg2.connect("postgresql://user:pass@localhost:5432/vectordb")
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Insert using parameterized query with vector cast.
embedding = np.random.randn(768).astype(np.float32).tolist()
cur.execute(
    "INSERT INTO articles (title, embedding) VALUES (%s, %s::vector) RETURNING id;",
    ("pgvector demo", embedding)
)
inserted_id = cur.fetchone()[0]
conn.commit()
print(f"Inserted article ID: {inserted_id}")
cur.close()
conn.close()
```

⚠️ **Dimension is a contract.** `vector(n)` enforces dimensionality at the type level. If you swap from a 768D model to a 384D model without altering the table or adding a new column, inserts will fail. Document the embedding model version alongside the schema — treat `vector(n)` like a foreign key constraint.

💡 **Use `COPY` for bulk loading, not individual INSERTs.** PostgreSQL's `COPY` is ~10× faster for initial data loads. Generate a CSV/TSV with vector literals and use `\COPY articles FROM 'embeddings.csv' WITH CSV;`.

**Caso real — Instacart product substitution:** Instacart uses `pgvector` to power real-time product substitution recommendations. When an item is out of stock, they query a `products` table using vector similarity on recipe-aware embeddings. Because everything lives in PostgreSQL, they avoid the operational overhead of syncing a separate vector store. The substitution query joins `products` with `inventory` and `pricing` tables in a single SQL statement, applying business rules (e.g., "substitute with same brand when possible") as `WHERE` filters before the `ORDER BY embedding <=> ... LIMIT 5` clause.

❌ **Antipattern: Manually managing vector storage outside PostgreSQL.** Storing embeddings in Parquet files and serving them from a separate ANN library (Faiss) while keeping metadata in PostgreSQL creates consistency headaches — updates must be synchronized, backups are split, and queries cannot be transactional. Let `pgvector` manage everything in one database.

✅ **Always match `vector(n)` to your actual embedding dimension.** If your model outputs 384D, use `vector(384)`. Padding to 768 wastes 50% storage. If you switch models, add a new column rather than altering the existing one — this allows gradual migration and rollback.

---

## 2. Distance Operators and Exact Search

### Theory

`pgvector` registers three distance operators that correspond to the metrics from [[01 - Vector Search Fundamentals]]:

- `<->` — **Euclidean distance (L2)**: $D_{L2}(a, b) = \sqrt{\sum_{i=1}^d (a_i - b_i)^2}$. Range: $[0, \infty)$. Smaller is more similar. This is the geometric straight-line distance, best when magnitude carries information.
- `<=>` — **Cosine distance**: $D_{\cos}(a, b) = 1 - \frac{a \cdot b}{\|a\|\|b\|}$. Range: $[0, 2]$ for normalized vectors. Smaller is more similar. This is the default for text embeddings because text semantics are directional.
- `<#>` — **Negative inner product**: $D_{\text{neg}}(a, b) = -(a \cdot b)$. Smaller (more negative) is more similar. PostgreSQL indexes only support ascending-order scans; by negating the dot product, the index can return "largest dot product first" even though it is indexed as ascending. This is a clever implementation detail: the same B-tree-like index traversal works for all three metrics without modification.

The mathematical relationship between the operators on normalized unit vectors ($\|a\| = \|b\| = 1$) is:
$$\|a - b\|^2 = 2 - 2\cos\theta = 2 - 2(a \cdot b)$$
So on unit vectors, all three operators produce the same ranking despite different absolute values.

Exact search uses these operators in `ORDER BY ... LIMIT k` clauses. Without an index, PostgreSQL performs a sequential scan, computing the distance for every row, then sorts and limits. This is $O(Nd)$ — precise but slow for large tables. The execution plan shows `Seq Scan` → `Sort` → `Limit`. For tables under ~50k rows, this is often fast enough; for larger tables, an index is required.

**Operator semantics and performance characteristics:**

| Operator | Metric | Index OpClass | Use Case |
|---|---|---|---|
| `<->` | L2 Euclidean | `vector_l2_ops` | Image, audio, sensor embeddings |
| `<=>` | Cosine | `vector_cosine_ops` | Text embeddings (default) |
| `<#>` | Neg. Dot Product | `vector_ip_ops` | Max inner product search |

Using the wrong operator for a given index opclass causes PostgreSQL to fall back to a sequential scan with no warning. Always verify with `EXPLAIN ANALYZE`.

```sql
-- Euclidean (L2) nearest neighbors.
SELECT id, title, embedding <-> '[0.1, -0.2, 0.5]'::vector AS distance
FROM articles
ORDER BY embedding <-> '[0.1, -0.2, 0.5]'::vector
LIMIT 5;

-- Cosine nearest neighbors (default for text embeddings).
-- WHY: <=> returns 0 for identical direction, 2 for opposite direction.
--      Order ASC to get most similar first.
SELECT id, title, embedding <=> '[0.1, -0.2, 0.5]'::vector AS cos_dist
FROM articles
ORDER BY embedding <=> '[0.1, -0.2, 0.5]'::vector
LIMIT 5;

-- Negative inner product (for max dot product with index support).
SELECT id, title, embedding <#> '[0.1, -0.2, 0.5]'::vector AS neg_ip
FROM articles
ORDER BY embedding <#> '[0.1, -0.2, 0.5]'::vector
LIMIT 5;

-- Filtered exact search: metadata WHERE clause before ordering.
-- WHY: PostgreSQL applies WHERE before distance computation,
--      reducing the number of vectors scanned.
SELECT id, title, embedding <=> query_vec AS dist
FROM articles, (SELECT '[0.1, -0.2, 0.5]'::vector AS query_vec) AS q
WHERE category = 'machine-learning'
ORDER BY dist
LIMIT 10;

-- Validate index usage with EXPLAIN ANALYZE.
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, embedding <=> '[0.1, -0.2, 0.5]'::vector AS dist
FROM articles
ORDER BY dist
LIMIT 10;
```

⚠️ **Using `<->` (L2) on normalized text embeddings produces counter-intuitive results.** If your embeddings are unit vectors, L2 distance is monotonic with cosine distance ($\|a-b\|^2 = 2 - 2\cos\theta$), but the absolute distance values are compressed in $[0, \sqrt{2}]$. Always use `<=>` for text embeddings — the distance values are more interpretable.

💡 **"Text? `<=>`. Images? `<->`."** — Default to cosine for anything from a language model. Use L2 for image, audio, and sensor data where magnitude matters.

**Caso real — Figma design components:** Figma uses `pgvector` for exact vector search over design component embeddings in their internal design-system search. Their collection is small (<100k components) but metadata-rich (team_id, component_type, last_modified). Exact search ensures 100% recall for designers looking for visually similar UI patterns. The query planner efficiently pushes down `WHERE team_id = ?` and `WHERE component_type = 'button'` filters before ordering by cosine distance, making each search an O(candidate_set) operation rather than O(full_collection).

❌ **Antipattern: Assuming exact search is always slow.** For tables with fewer than ~50k rows and effective metadata filters that reduce the candidate set to <5000 rows, exact `ORDER BY ... LIMIT` with a sequential scan often outperforms an index scan due to zero index overhead.

✅ **Start with exact search during prototyping.** Validate embedding quality and query patterns before adding index complexity. The exact results become ground truth for ANN recall benchmarking later.

---

## 3. Indexing with `ivfflat` and `hnsw`

### Theory

`pgvector` implements two index families that mirror the algorithms from [[02 - Indexing Algorithms Deep Dive]]:

**`ivfflat`** (Inverted File with Flat quantization) partitions the vector space into `lists` clusters using k-means at `CREATE INDEX` time. Each vector is assigned to exactly one list. At query time, the system probes `probes` lists (default 1) and performs exact search within those lists. `ivfflat` is memory-efficient (stores raw vectors without graph edges) but has limitations: `lists` is fixed at build time, and it does not support incremental inserts efficiently — heavy update patterns may require a rebuild for optimal performance.

**`hnsw`** (Hierarchical Navigable Small World) builds a multi-layer graph. It supports incremental inserts, deletes, and updates natively — vectors can be added or removed without rebuilding the entire index. Parameters `m` (max connections per node), `ef_construction` (search width at build), and `ef_search` (search width at query) control the recall/latency trade-off. `hnsw` is the recommended index for most production workloads due to its flexibility, high recall, and online update support.

Starting with `pgvector` 0.5.0+, `CREATE INDEX CONCURRENTLY` is supported for `hnsw`, allowing index builds without locking the table for writes. This is critical for production deployments where downtime is unacceptable.

```sql
-- ivfflat index: simpler but less flexible.
-- WHY: lists = sqrt(N) is a common heuristic. For 1M vectors, lists = 1000.
CREATE INDEX ON articles
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- hnsw index: recommended for production.
-- WHY: ef_construction controls build quality; m controls graph density.
--      These are set at CREATE INDEX time and cannot be changed without rebuild.
CREATE INDEX ON articles
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Query-time tuning for hnsw.
-- WHY: ef_search is a session parameter — tune per query pattern.
SET hnsw.ef_search = 100;
SELECT id, title, embedding <=> '[0.1, -0.2, 0.5]'::vector AS dist
FROM articles
ORDER BY dist
LIMIT 10;

-- Query-time tuning for ivfflat.
-- WHY: probes controls how many lists to scan. Default is 1 (fast, low recall).
SET ivfflat.probes = 10;
SELECT id, title, embedding <=> '[0.1, -0.2, 0.5]'::vector AS dist
FROM articles
ORDER BY dist
LIMIT 10;

-- Build hnsw index concurrently (no table lock, pgvector 0.5+).
CREATE INDEX CONCURRENTLY articles_embedding_idx
ON articles USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
```

⚠️ **Creating an `ivfflat` index before the table has representative data is fatal.** `ivfflat` runs k-means at index creation time. If the table is empty or tiny, the centroids are meaningless and the index either returns garbage or falls back to sequential scan. Populate the table first (minimum $\text{lists} \times 1000$ rows), then create the index.

💡 **`ivfflat` trains once; `hnsw` trains online.** If your dataset is still growing, delay `ivfflat` index creation and use `hnsw` instead — it handles incremental inserts without quality degradation.

**Caso real — Microsoft Azure Database for PostgreSQL:** Microsoft recommends `hnsw` with `m=16` to `32` and `ef_construction=128` to `256` for general-purpose semantic search in Azure's managed PostgreSQL — Flexible Server. Their documentation shows that for 1M vectors of 768D, `hnsw` with `ef_construction=200` and `ef_search=100` achieves ~95% recall@10 with ~15ms query latency, while `ivfflat` with `lists=1000` and `probes=10` achieves ~88% recall at similar latency. The `hnsw` advantage grows with dataset size because the graph structure maintains logarithmic search complexity.

❌ **Antipattern: Using `ivfflat` for datasets with frequent inserts or deletes.** Each insertion adds a vector to the nearest centroid's list, but the centroid itself never moves. Over time, list imbalance grows and recall degrades. For dynamic data, `hnsw` is always the better choice.

✅ **Always use `vector_cosine_ops` for text embeddings.** `pgvector` requires the operator class to match the distance operator. `vector_cosine_ops` supports `<=>` (cosine), `vector_l2_ops` supports `<->` (L2), and `vector_ip_ops` supports `<#>` (inner product). Mismatch causes sequential scan fallback without any error message — `EXPLAIN ANALYZE` is the only way to detect it.

```sql
-- Correct operator class for cosine distance.
CREATE INDEX ON articles USING hnsw (embedding vector_cosine_ops);

-- Wrong: this index will NOT be used with <=> queries.
CREATE INDEX ON articles USING hnsw (embedding vector_l2_ops);  -- only supports <->
```

---

**Caso real — Etsy listing search:** Etsy uses `pgvector` for semantic search over product listings. Their schema stores 384D embeddings from `all-MiniLM-L6-v2` alongside listing metadata (price, category, shop_id, rating). The critical insight from their deployment: by applying `WHERE price BETWEEN 10 AND 50 AND rating > 4.0` in the same query as `ORDER BY embedding <=> query LIMIT 50`, they avoid the expensive application-level join that would be required with a separate vector database. PostgreSQL's query planner filters to ~10k candidate listings before the vector distance computation, reducing the effective search space by 99%.

### Distance Operator Internals

Internally, `pgvector` implements each operator as a PostgreSQL C function that computes the distance between two `vector` values. These functions are registered with specific costs in `pg_proc` so the planner can estimate execution cost. The `<#>` operator is particularly interesting: it inverts the sign of the dot product so that PostgreSQL's default ascending index scan order returns "largest dot product first." This is a common pattern in PostgreSQL extensions — adapting the index to the query semantics without modifying the core access method.

```sql
-- Demonstrating the equivalence on unit vectors.
WITH unit_vectors AS (
    SELECT '[0.5, 0.5, 0.5, 0.5]'::vector AS a,
           '[0.6, 0.4, 0.6, 0.4]'::vector AS b
)
SELECT
    a <-> b AS l2,
    a <=> b AS cosine,
    a <#> b AS neg_ip,
    (a <-> b)^2 AS l2_sq,
    2 - 2 * (1 - (a <=> b)) AS derived_l2_sq
FROM unit_vectors;
```

⚠️ **The `<#>` operator returns values in $(-\infty, \infty)$ while L2 and cosine are bounded below by 0.** Do not mix different operators in the same query without understanding their ranges — combining scores from different operators in a weighted sum requires normalization.

💡 **For RAG pipelines, always use `<=>` with `vector_cosine_ops`.** Most embedding models (OpenAI, Sentence-Transformers, Cohere) output normalized vectors optimized for cosine similarity. Using L2 on these embeddings produces correct rankings (due to monotonicity) but distance values are less interpretable for filtering thresholds.
## 🎯 Key Takeaways

- `pgvector` adds vector types and ANN indexes to PostgreSQL without forking the database — it is an extension, not a fork
- Use `vector(n)` with dimensionality matching your embedding model; mismatches cause runtime failures; document the model version alongside the schema
- Choose `<=>` (cosine) for normalized text embeddings, `<->` (L2) for image/audio/sensor embeddings, and `<#>` (negative inner product) for max dot product with index support
- `ivfflat` is memory-light but requires pre-training on representative data and cannot handle dynamic datasets well — use it only for static collections with known size
- `hnsw` is the production default: supports online inserts/updates, high recall, and runtime tuning via `ef_search`; always use `ef_construction >= 200` at build time
- Always validate index usage with `EXPLAIN (ANALYZE, BUFFERS)` after schema changes — operator class mismatches silently disable indexes without any error message
- Tune `ivfflat.probes` and `hnsw.ef_search` per workload session rather than globally; lower for latency-sensitive UIs, higher for accuracy-critical RAG pipelines
- Use `CREATE INDEX CONCURRENTLY` in production to avoid table locks during index builds

## References

- pgvector GitHub: https://github.com/pgvector/pgvector
- pgvector documentation: https://github.com/pgvector/pgvector?tab=readme-ov-file#readme
- Andrew Kane. "pgvector: vector search for Postgres." Blog series, 2023–2024
- PostgreSQL Extension Development: https://www.postgresql.org/docs/current/extend-extensions.html
- [[02 - Indexing Algorithms Deep Dive]] — Algorithmic foundations of ivfflat and hnsw
- [[04 - pgvector II - Production and Hybrid Search]] — Partitioning, hybrid search, production ops

## 📦 Código de compresión

```python
"""
pgvector I — Compression Script
Summarizes: vector type, CRUD, distance operators, ivfflat, hnsw.
"""
import psycopg2
import numpy as np

DSN = "postgresql://user:pass@localhost:5432/vectordb"

def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id SERIAL PRIMARY KEY,
                name TEXT,
                embedding vector(768)
            );
        """)
        conn.commit()

def insert_item(conn, name: str, embedding: list[float]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO items (name, embedding) VALUES (%s, %s::vector) RETURNING id;",
            (name, embedding)
        )
        return cur.fetchone()[0]

def search_cosine(conn, query_vec: list[float], k: int = 5):
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100;")
        cur.execute("""
            SELECT id, name, embedding <=> %s::vector AS dist
            FROM items
            ORDER BY dist
            LIMIT %s;
        """, (query_vec, k))
        return cur.fetchall()

def create_hnsw_index(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS items_embedding_idx
            ON items USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """)
        conn.commit()

if __name__ == "__main__":
    conn = psycopg2.connect(DSN)
    init_db(conn)
    create_hnsw_index(conn)
    vec = np.random.randn(768).astype(np.float32).tolist()
    insert_item(conn, "demo", vec)
    print(search_cosine(conn, vec, k=3))
    conn.close()
```

```sql
-- SQL-only validation: create table, insert, build index, search.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE test_items (id SERIAL PRIMARY KEY, embedding vector(3));
INSERT INTO test_items (embedding) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');
CREATE INDEX ON test_items USING hnsw (embedding vector_cosine_ops) WITH (m=4, ef_construction=10);
SET hnsw.ef_search = 10;
SELECT id, embedding <=> '[1,0,0]'::vector AS dist
FROM test_items
ORDER BY dist
LIMIT 2;
```
