# 🐘 pgvector I - Core Operations and Indexing

## 🎯 Learning Objectives

- Install and configure the `pgvector` extension in PostgreSQL 15+
- Create and manipulate the `vector(n)` type for storing dense embeddings
- Perform CRUD operations on vector columns and understand distance semantics
- Build and tune `ivfflat` and `hnsw` indexes with correct parameters
- Use distance operators `<->`, `<=>`, and `<#>` and explain their geometric meanings
- Adjust `hnsw.ef_search` and `ivfflat.probes` at query time for recall/latency trade-offs

## Introduction

`pgvector` is an open-source PostgreSQL extension that transforms the world's most trusted relational database into a production-grade vector database. Unlike standalone vector engines, `pgvector` allows you to store embeddings alongside transactional business data — user profiles, product metadata, access control lists — in the same ACID-compliant table. This eliminates the complexity of dual-write patterns, distributed transactions, and eventual consistency between a relational store and a separate vector index.

This note covers the foundational operations: installation, the `vector` type, basic querying, and index creation. We focus on `ivfflat` and `hnsw`, the two index families available in `pgvector`, and teach you how to tune them with session-level parameters. The next note, [[04 - pgvector II - Production and Hybrid Search]], scales these foundations to partitioned tables, hybrid full-text + vector queries, and operational monitoring.

This module connects to [[12 - SQL Mastery]] (PostgreSQL fundamentals) and [[15 - Docker and Kubernetes]] (containerized deployment of Postgres + pgvector).

---

## Module 1: pgvector Installation and the vector Type

### 1.1 Theoretical Foundation 🧠

PostgreSQL's extensibility architecture allows third-party code to register new types, operators, and index access methods without modifying the core server. `pgvector` leverages this by adding:

1. A `vector(n)` composite type backed by a fixed-length array of `float4` (32-bit floats)
2. Three distance operators registered with the operator class framework
3. Index access methods (`ivfflat`, `hnsw`) that plug into PostgreSQL's planner and executor

The `vector(n)` type is stored in-line for small n (≤DimensionsToInline, typically 1024) and out-of-line via TOAST for larger dimensions. This means that for standard embedding sizes (384–1536), vectors live directly in the heap tuple, making sequential scans fast and index-only scans possible when the vector column is covered by the index.

Why use `pgvector` instead of a dedicated vector database? The answer is **data locality**. In production ML systems, embeddings are rarely queried in isolation. A recommendation query might join `user_vectors` with `user_demographics` and `product_inventory`. Doing this across two databases requires application-level joins or denormalization; `pgvector` lets the query planner optimize the entire operation.

### 1.2 Mental Model 📐

```
┌─────────────────────────────────────────────┐
│  PostgreSQL + pgvector Architecture         │
│                                             │
│  ┌──────────────┐                           │
│  │  SQL Parser  │                           │
│  └──────┬───────┘                           │
│         ▼                                   │
│  ┌──────────────┐                           │
│  │   Planner    │◄── pgvector cost model    │
│  └──────┬───────┘                           │
│         ▼                                   │
│  ┌──────────────┐                           │
│  │  Executor    │──► vector ops (<-> etc.)  │
│  └──────┬───────┘                           │
│         ▼                                   │
│  ┌──────────────┐  ┌──────────┐             │
│  │  Heap/Toast  │  │  ivfflat │  hnsw       │
│  │   Storage    │  │  Index   │  Index      │
│  └──────────────┘  └──────────┘             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  vector(n) Storage Layout                   │
│                                             │
│  [dim: uint16] [float4] [float4] ...        │
│       │          │        │                 │
│       │          └─ x1    └─ x2             │
│       └─ dimensionality check at insert     │
│                                             │
│  Max dim: 16,000 (pgvector 0.5+)            │
└─────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```sql
-- WHY: pgvector must be installed in each database where it is used.
--      It is not a server-level extension because types are database-scoped.
CREATE EXTENSION IF NOT EXISTS vector;

-- WHY: vector(n) enforces dimensionality at the type level.
--      This prevents silent corruption from mismatched embedding models.
CREATE TABLE articles (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT,
    embedding   vector(768)  -- matches e.g. BERT-base embeddings
);

-- Insert: vector literals use ARRAY::vector syntax or string form.
-- WHY: The string form '[a,b,c]' is often easier to generate from Python/Go.
INSERT INTO articles (title, content, embedding)
VALUES (
    'Introduction to pgvector',
    'A guide to vector search in Postgres...',
    ARRAY(SELECT random() FROM generate_series(1, 768))::vector
);

-- Alternative string literal form.
INSERT INTO articles (title, embedding)
VALUES ('Demo', '[0.1, -0.3, 0.8, ...]');  -- truncated for brevity

-- Update: replace an embedding after re-encoding.
UPDATE articles
SET embedding = ARRAY(SELECT random() FROM generate_series(1, 768))::vector
WHERE id = 1;

-- Delete with cascading index maintenance.
DELETE FROM articles WHERE id = 1;
```

### 1.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A[Client Application] --> B[PostgreSQL Server]
    B --> C[SQL Parser]
    C --> D[Query Planner]
    D --> E{Use Index?}
    E -->|Yes| F[ivfflat / hnsw Scan]
    E -->|No| G[Sequential Scan]
    F --> H[Distance Operator]
    G --> H
    H --> I[Result Set]
```

```mermaid
erDiagram
    ARTICLES {
        int id PK
        text title
        text content
        vector768 embedding
    }
    INDEX_HNSW {
        vector vector "graph layers"
    }
    ARTICLES ||--o{ INDEX_HNSW : indexed_by
```

![PostgreSQL logo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Postgresql_elephant.svg/440px-Postgresql_elephant.svg.png)

### 1.5 Application in ML/AI Systems 🤖

Real case: **Instacart** uses `pgvector` to power real-time product substitution recommendations. When an item is out of stock, they query a `products` table (joined with `inventory` and `pricing`) using vector similarity on recipe-aware embeddings. Because everything lives in PostgreSQL, they avoid the operational overhead of syncing a separate vector store.

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| RAG document stores | `vector` column in `documents` table | Join metadata filters with semantic retrieval in one query |
| Product search | `vector` + `tsvector` hybrid | Combine semantic and lexical matching ([[04 - pgvector II - Production and Hybrid Search]]) |
| Entity resolution | L2 distance on record embeddings | Deduplicate customer profiles using vector similarity |

### 1.6 Common Pitfalls ⚠️

⚠️ **Pitfall: Mixing embedding models with different dimensions in the same column.** Root cause: `pgvector` enforces `vector(n)` at the type level, but if you swap from a 768D model to a 384D model without altering the table or creating a new column, inserts will fail. Worse, some ORMs silently truncate or pad. Document your model version alongside the schema.

💡 **Mnemonic: "Dimension is a contract."** — Treat `vector(n)` like a foreign key constraint: changing it requires a migration plan.

### 1.7 Knowledge Check ❓

1. Write the SQL to create a `users` table with a 512-dimensional `profile_embedding` column and a uniqueness constraint on `email`.
2. Explain why `pgvector` stores vectors as `float4` instead of `float8`. What is the accuracy vs. storage trade-off?
3. You have 10 million 1536D vectors. Will they be stored in-line or TOAST? How does this affect sequential scan performance?

---

## Module 2: Distance Operators and Exact Search

### 2.1 Theoretical Foundation 🧠

`pgvector` registers three distance operators that correspond to the metrics introduced in [[01 - Vector Search Fundamentals]]:

- `<->` — **Euclidean distance (L2)**: `sqrt(sum((a_i - b_i)^2))`. Smaller is more similar. This is the default operator for many `pgvector` examples.
- `<=>` — **Cosine distance**: `1 - cosine_similarity`. Range is [0, 2] for unit vectors. Smaller is more similar. This is the correct operator for normalized text embeddings.
- `<#>` — **Negative inner product**: `-(a · b)`. Smaller (more negative) is more similar. This operator exists because PostgreSQL indexes only support ascending-order scans; by negating the dot product, the B-tree-like index traversal can return "largest dot product first" as "smallest negative inner product first."

Exact search uses these operators in `ORDER BY ... LIMIT k` clauses. Without an index, PostgreSQL performs a sequential scan, computes the distance for every row, sorts, and returns the top k. This is precise but slow for large tables. The execution plan will show `Seq Scan` followed by `Sort` and `Limit`.

### 2.2 Mental Model 📐

```
┌─────────────────────────────────────────────┐
│  Distance Operators in pgvector             │
│                                             │
│  <->  L2 Distance                           │
│       d = sqrt(sum((a_i - b_i)^2))          │
│       Use when: image/audio, magnitude      │
│       matters, vectors not normalized       │
│                                             │
│  <=>  Cosine Distance                       │
│       d = 1 - (a·b)/(||a|| ||b||)          │
│       Use when: text embeddings, normalized │
│       vectors, semantic similarity          │
│                                             │
│  <#>  Negative Inner Product                │
│       d = -(a·b)                            │
│       Use when: you need max dot product    │
│       and want index support                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Exact Search Execution Plan                │
│                                             │
│  Seq Scan on articles                       │
│    Filter: ...                              │
│  Sort                                       │
│    Sort Key: (embedding <-> query_vec)      │
│  Limit                                      │
│    -> ...                                   │
│                                             │
│  Cost: O(N) distance computations           │
└─────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```sql
-- WHY: Exact search is the baseline. Use it to validate index correctness
--      and for small tables (<50k rows) where index overhead isn't worth it.

-- Euclidean (L2) nearest neighbors
SELECT id, title, embedding <-> '[0.1, -0.2, 0.5, ...]'::vector AS distance
FROM articles
ORDER BY embedding <-> '[0.1, -0.2, 0.5, ...]'::vector
LIMIT 5;

-- Cosine nearest neighbors (text semantics)
-- WHY: <=> returns 0 for identical direction, 2 for opposite direction.
--      Order ASC to get most similar first.
SELECT id, title, embedding <=> '[0.1, -0.2, 0.5, ...]'::vector AS cos_dist
FROM articles
ORDER BY embedding <=> '[0.1, -0.2, 0.5, ...]'::vector
LIMIT 5;

-- Negative inner product (for max dot product)
SELECT id, title, embedding <#> '[0.1, -0.2, 0.5, ...]'::vector AS neg_ip
FROM articles
ORDER BY embedding <#> '[0.1, -0.2, 0.5, ...]'::vector
LIMIT 5;

-- Filtering before ordering: combine metadata + vector search
-- WHY: PostgreSQL applies WHERE before distance computation when possible.
SELECT id, title, embedding <=> query_vec AS dist
FROM articles, (SELECT '[0.1, -0.2, ...]'::vector AS query_vec) AS q
WHERE category = 'machine-learning'
ORDER BY dist
LIMIT 10;
```

### 2.4 Visual Representation 🖼️

```mermaid
graph LR
    Q[Query Vector] --> E[L2 <->]
    Q --> C[Cosine <=>]
    Q --> I[Neg IP <#>]
    E --> R1[Geometric Distance]
    C --> R2[Angular Distance]
    I --> R3[Orientation + Magnitude]
```

```mermaid
flowchart TD
    A[SELECT ... ORDER BY embedding <=> q LIMIT 10] --> B[Planner]
    B --> C{Index on <=>?}
    C -->|No| D[Seq Scan all rows]
    D --> E[Compute <=> for each]
    E --> F[Sort]
    F --> G[Limit 10]
```

![Distance metrics](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Euclidean_vs_Cosine.png/500px-Euclidean_vs_Cosine.png)

### 2.5 Application in ML/AI Systems 🤖

Real case: **Figma** uses `pgvector` for exact vector search over design component embeddings in their internal design-system search. Their collection is small (<100k components) but metadata-rich. Exact search ensures 100% recall for designers looking for visually similar UI patterns, and the query planner efficiently pushes down `WHERE team_id = ...` filters before sorting.

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| Small-catalog semantic search | Exact ORDER BY ... LIMIT | No index build time, perfect recall |
| Prototype RAG pipeline | Exact cosine search | Validate embedding quality before adding index complexity |
| Filtered similarity | WHERE + ORDER BY | Combine business rules with vector ranking in one transaction |

### 2.6 Common Pitfalls ⚠️

⚠️ **Pitfall: Using `<->` (L2) on normalized embeddings and expecting semantic results.** Root cause: If your embeddings are unit vectors (common for text), L2 distance is monotonic with cosine distance, but the absolute values are compressed near [0, √2]. Small numerical differences become exaggerated. Always use `<=>` for normalized text embeddings to get intuitive distance values in [0, 2].

💡 **Mnemonic: "Text? <=>. Images? <->."** — Default to cosine for anything that came out of a language model.

### 2.7 Knowledge Check ❓

1. Given two identical unit vectors, what are the results of `<->`, `<=>`, and `<#>`? Prove it algebraically.
2. Write a query that finds the 10 most similar articles using cosine distance, but excludes articles with `is_draft = true`.
3. Why does `pgvector` use `<#>` (negative inner product) instead of providing a direct "dot product" ordering operator?

---

## Module 3: Indexing with ivfflat and hnsw

### 3.1 Theoretical Foundation 🧠

`pgvector` implements two index families that mirror the algorithms from [[02 - Indexing Algorithms Deep Dive]]:

**`ivfflat`** (Inverted File with Flat quantization) partitions the vector space into `lists` clusters using k-means. Each vector is assigned to exactly one list. At query time, the system probes `probes` lists (default 1) and performs exact search within those lists. It is the older index type, memory-efficient, but requires `lists` to be set at build time and does not support incremental updates efficiently (rebuilds may be needed for optimal performance after heavy updates).

**`hnsw`** (Hierarchical Navigable Small World) builds a multi-layer graph. It supports incremental inserts, deletes, and updates natively — vectors can be added or removed without rebuilding the entire index. Its parameters `m` (max connections per node), `ef_construction` (search width at build), and `ef_search` (search width at query) control the recall/latency trade-off. `hnsw` is the recommended index for most production workloads in `pgvector` because of its flexibility and high recall.

The index build is **blocking** by default in older versions; `pgvector` 0.5+ supports `CREATE INDEX CONCURRENTLY` for `hnsw`, allowing builds without locking the table.

### 3.2 Mental Model 📐

```
┌─────────────────────────────────────────────┐
│  ivfflat Index Structure                    │
│                                             │
│  lists = 100                                │
│                                             │
│  Centroid C1 ──► [v3, v7, v12, ...]         │
│  Centroid C2 ──► [v1, v5, v9, ...]          │
│  ...                                        │
│  Centroid C100 ──► [v2, v4, ...]            │
│                                             │
│  Query: nearest centroids ──► probe lists   │
│         ──► exact scan within lists         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  hnsw Index Structure                       │
│                                             │
│  Layer 2:  ●────●                           │
│  Layer 1:  ●─●──●─●                         │
│  Layer 0:  ●─●─●─●─●─●─●─● ... (all nodes)  │
│                                             │
│  Supports: INSERT, UPDATE, DELETE online    │
│  No full rebuild required for new data      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Query-Time Tuning Parameters               │
│                                             │
│  ivfflat: SET ivfflat.probes = 10;          │
│            ↑ more probes = higher recall    │
│                                             │
│  hnsw: SET hnsw.ef_search = 128;            │
│         ↑ wider beam = higher recall        │
│         ↑ slower query                      │
└─────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```sql
-- WHY: ivfflat is simpler but less flexible. Good for static datasets
--      where you know the final size upfront.
CREATE INDEX ON articles
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);   -- number of clusters; tune to ~sqrt(N) as heuristic

-- WHY: hnsw is the default recommendation for production.
--      ef_construction controls build quality; m controls graph density.
CREATE INDEX ON articles
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Query-time tuning for hnsw
-- WHY: ef_search is a session parameter. Tune per query pattern.
SET hnsw.ef_search = 100;

SELECT id, title, embedding <=> '[0.1, -0.2, ...]'::vector AS dist
FROM articles
ORDER BY dist
LIMIT 10;

-- Query-time tuning for ivfflat
-- WHY: probes controls how many lists to scan. Default is 1 (fast, low recall).
SET ivfflat.probes = 10;

SELECT id, title, embedding <=> '[0.1, -0.2, ...]'::vector AS dist
FROM articles
ORDER BY dist
LIMIT 10;

-- WHY: EXPLAIN ANALYZE reveals whether the index is used.
EXPLAIN ANALYZE
SELECT id, title, embedding <=> '[0.1, -0.2, ...]'::vector AS dist
FROM articles
ORDER BY dist
LIMIT 10;
-- Look for: "Index Scan using articles_embedding_idx"
-- If you see "Seq Scan", the index is not being used (check opclass match).
```

### 3.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A[CREATE INDEX ... USING ivfflat] --> B[K-Means Clustering]
    B --> C[Assign vectors to lists]
    C --> D[Store lists in index pages]
    E[Query] --> F[Find nearest centroids]
    F --> G[Scan probed lists]
    G --> H[Exact distance + Limit K]
```

```mermaid
flowchart TD
    A[CREATE INDEX ... USING hnsw] --> B[Build multi-layer graph]
    B --> C[Insert vectors incrementally]
    D[Query] --> E[Set ef_search]
    E --> F[Greedy descent from entry point]
    F --> G[Refine in Layer 0]
    G --> H[Return approximate KNN]
```

![HNSW layers](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Small_world_network_example.png/500px-Small_world_network_example.png)

### 3.5 Application in ML/AI Systems 🤖

Real case: **Microsoft** uses `pgvector` with `hnsw` indexes in Azure Database for PostgreSQL — Flexible Server to provide managed vector search. Their recommendation is `m = 16` to `32` and `ef_construction = 128` to `256` for general-purpose semantic search, with `ef_search` tuned per workload (lower for exploratory UI, higher for RAG retrieval).

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| Real-time RAG retrieval | hnsw with ef_search=128 | <20ms retrieval from millions of chunks |
| Batch catalog embedding | ivfflat with lists=1000 | Fast bulk-load with moderate query latency |
| Multi-tenant SaaS | hnsw per tenant table | Isolated index tuning and access control |

### 3.6 Common Pitfalls ⚠️

⚠️ **Pitfall: Creating an `ivfflat` index before the table has representative data.** Root cause: `ivfflat` runs k-means at `CREATE INDEX` time. If the table is empty or tiny, the centroids are meaningless, and the index returns garbage or falls back to sequential scan. Populate the table first, then build the index.

💡 **Mnemonic: "ivfflat trains once; hnsw trains online."** — If your data is still loading, delay `ivfflat` index creation until you have at least `lists × 1000` vectors.

### 3.7 Knowledge Check ❓

1. You have 1 million vectors. Using the heuristic `lists = sqrt(N)`, how many `lists` should you set for `ivfflat`? If `probes = 5`, approximately what fraction of the data is scanned?
2. Explain the difference between `ef_construction` and `ef_search`. Which one affects index build time, and which affects query time?
3. You run `EXPLAIN ANALYZE` and see a sequential scan despite having an `hnsw` index. List three possible reasons why the planner chose not to use it.

---

## 📦 Compression Code

```python
"""
pgvector I — Compression Script
Summarizes: vector type, CRUD, distance operators, ivfflat, hnsw.
"""
import psycopg2

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

def insert_item(conn, name: str, embedding: list[float]):
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
    # Example insert
    vec = [0.1] * 768
    insert_item(conn, "demo", vec)
    print(search_cosine(conn, vec, k=3))
    conn.close()
```

## 🎯 Documented Project

**Project: PostgreSQL Semantic FAQ Search**

- **Description**: A minimal FAQ system where questions are embedded and stored in `pgvector`. Users ask natural-language questions; the system returns the most similar stored FAQ answer.
- **Functional Requirements**:
  - Schema: `faqs(id, question, answer, embedding vector(384))`.
  - Populate with 1,000 synthetic FAQ pairs and real embeddings from `all-MiniLM-L6-v2`.
  - Build an `hnsw` index on `embedding` using `vector_cosine_ops`.
  - API endpoint `POST /search` accepts a question, embeds it, and returns top-3 FAQs.
- **Main Components**:
  - `embedder.py`: wraps `sentence-transformers` to produce 384D vectors.
  - `db.py`: `psycopg2` connection pool with `pgvector` initialization.
  - `api.py`: FastAPI endpoint executing `ORDER BY embedding <=> %s LIMIT 3`.
- **Success Metrics**:
  - p99 query latency <50ms with `hnsw.ef_search = 64` on 100k vectors.
  - Human-evaluated relevance: top-1 answer correct ≥80% of time.

## 🎯 Key Takeaways

- `pgvector` adds vector types and ANN indexes to PostgreSQL without forking the database.
- Use `vector(n)` with dimensionality matching your embedding model; mismatches cause runtime failures.
- Choose `<=>` for normalized text embeddings, `<->` for general L2, and `<#>` for max dot product.
- `ivfflat` is memory-light but requires pre-training on representative data and is less dynamic.
- `hnsw` is the production default: supports online updates, high recall, and runtime tuning via `ef_search`.
- Always validate index usage with `EXPLAIN ANALYZE`; operator class mismatches silently disable indexes.
- Tune `ivfflat.probes` and `hnsw.ef_search` per workload rather than globally.

## References

- pgvector GitHub: https://github.com/pgvector/pgvector
- pgvector documentation: https://github.com/pgvector/pgvector?tab=readme-ov-file#readme
- Andrew Kane, "pgvector: vector search for Postgres." Blog series, 2023–2024.
- PostgreSQL Extension Development: https://www.postgresql.org/docs/current/extend-extensions.html
- [[02 - Indexing Algorithms Deep Dive]] — Algorithmic foundations of ivfflat and hnsw.
