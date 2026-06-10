# 🟦 AlloyDB, BigQuery `VECTOR_SEARCH()`, and the GCP Decision Framework

## 🎯 Learning Objectives

- Use **AlloyDB AI** for transactional + vector workloads in a single Postgres-compatible database
- Run **BigQuery `VECTOR_SEARCH()`** for SQL-native batch and ad-hoc vector retrieval on existing BigQuery tables
- Predict the **monthly cost** of each option at 1M, 10M, and 100M vector scale
- Apply the **decision framework** when starting a new GCP project, choosing between Vertex AI Vector Search, AlloyDB AI, and BigQuery `VECTOR_SEARCH()`
- Use the **`google-cloud-alloydb-connector`** Python SDK to connect from Cloud Run, GKE, or local development
- Migrate workloads **between options** without vendor lock-in (the GCP ecosystem is designed for this)
- Implement **multi-region replication** for AlloyDB AI and **cross-region BigQuery** for global RAG

---

## Introduction

Note `01` covered Vertex AI Vector Search — the **managed, low-latency, GCP-native** answer for dedicated vector retrieval. But Vertex AI is not always the right answer. Two scenarios push you toward alternatives:

1. **Transactional + vector in one database**: you already run Postgres on GCP, you have 50M rows of business data, and you want to add vector search without introducing a new system. AlloyDB AI (Postgres-compatible, with pgvector++) is the answer.
2. **Vector data already lives in BigQuery**: you ETL embeddings into a BigQuery table for analytics, and you want to query them without exporting to a separate system. BigQuery `VECTOR_SEARCH()` is the answer.

This note covers both alternatives, then closes with a **decision framework** for picking the right option. The matrix is not "which is best" — it is "which matches your specific workload, team, and existing GCP footprint".

We connect to `[[../01 - Vertex AI Vector Search - Architecture and Python Client]]` (the managed alternative), `[[../../../10 - Cloud, Infra y Backend/36 - PostgreSQL for AI-ML Workloads/01 - pgvector Production Tuning|10/36/01]]` (the pgvector tuning fundamentals), and `[[../../../10 - Cloud, Infra y Backend/28 - BigQuery for ML/01 - BigQuery Architecture|10/28/01]]` (the BigQuery architecture context).

---

## 1. The Problem and Why This Solution Exists

### The "I already have Postgres" use case

Vertex AI Vector Search is a **separate system**. It requires a separate deployment, separate IAM, separate monitoring, separate billing. For teams that already operate Postgres on GCP — whether Cloud SQL for PostgreSQL, self-managed on GCE, or the new AlloyDB — the cost of adding a second system is real:

- **Data duplication**: embeddings live in Postgres tables; they would have to be mirrored to Vertex AI Vector Search via a sync job. Each row, two storage systems.
- **Consistency**: every upsert in Postgres needs a matching upsert in Vertex AI. Failures in the sync job cause drift.
- **Joins**: if the business logic needs `SELECT products WHERE price < 100 ORDER BY embedding <=> $1 LIMIT 10` (semantic search joined with a price filter), that query is trivial in Postgres and impossible across two systems.

**AlloyDB AI** solves this by adding the `vector` extension and Google's pgvector optimizations to a Postgres-compatible engine. One database, ACID transactions, foreign keys, and vector search.

> **Caso real**: **Home Depot's product search ranking** runs on AlloyDB AI (Postgres-compatible) with pgvector + AlloyDB's columnar engine. The query is `SELECT * FROM products WHERE in_stock = true AND embedding <=> $1 < 0.3 ORDER BY embedding <=> $1 LIMIT 50`. The whole pipeline — semantic search, in-stock filter, price range, vendor tier — runs as a single SQL query with a 30 ms p99.

### The "I already have BigQuery" use case

Most GCP-based ML pipelines ETL data into BigQuery for analytics. The embeddings, if computed in the same pipeline, end up as a column in a BigQuery table. Exposing them to a separate vector search service means either:

- Re-ETL the embeddings from BigQuery to Vertex AI (extra cost, extra lag)
- Compute embeddings in a separate pipeline and write to Vertex AI directly (duplicates the BigQuery work)

**BigQuery `VECTOR_SEARCH()`** solves this by adding vector search as a **SQL function** on existing tables. The query is `SELECT * FROM VECTOR_SEARCH(TABLE my_docs, "embedding", query_embedding, top_k => 10)`. The data stays in BigQuery; the search is native SQL.

The trade-off: `VECTOR_SEARCH()` is **batch-optimized**, not online-retrieval-optimized. Latency is 100-500 ms for small tables, seconds for billion-row tables. For RAG with a 5-second end-to-end budget, that is fine. For real-time recommendation at <50 ms, you want Vertex AI or AlloyDB.

### The 2019-2026 evolution

| Era | AlloyDB | BigQuery `VECTOR_SEARCH()` |
|-----|---------|---------------------------|
| 2019-2021 | Cloud SQL for PostgreSQL with pgvector (limited) | Not available |
| 2022-2023 | AlloyDB GA with pgvector (basic) | Not available |
| 2024 (preview) | AlloyDB AI with columnar engine, 4× faster pgvector | `VECTOR_SEARCH()` preview |
| 2025 (GA) | AlloyDB AI GA, ScaNN-based Tree-AH | `VECTOR_SEARCH()` GA, with `VECTOR_INDEX` |
| 2026 | Auto-scaling storage, multi-region | `VECTOR_SEARCH()` + `VECTOR_INDEX` GA, cross-region |

Both products are recent — AlloyDB AI GA in 2025, BigQuery `VECTOR_SEARCH()` GA in 2025. Mature production deployments are just beginning.

---

## 2. Conceptual Deep Dive

### 2.1 AlloyDB AI — Postgres-compatible vector search

```sql
-- 1. Create the database (via Cloud Console or gcloud)
-- gcloud alloydb clusters create rag-cluster --region=us-central1

-- 2. Connect and enable the extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS google_ml_integration;

-- 3. Create the table with a vector column
CREATE TABLE products (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    price       NUMERIC(10, 2),
    in_stock    BOOLEAN DEFAULT true,
    embedding   vector(768),            -- 768-dim, matches bge-base
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Create an index (AlloyDB's ScaNN-based Tree-AH is the recommended default)
CREATE INDEX products_embedding_idx ON products
    USING scann (embedding cosine_distance)
    WITH (num_leaves = 100, num_leaves_to_search = 10);

-- 5. Query — the entire retrieval + filter in one SQL statement
SELECT id, name, price,
       embedding <=> $1 AS distance
FROM products
WHERE in_stock = true
  AND price < 100.0
  AND embedding <=> $1 < 0.3          -- pre-filter by relevance
ORDER BY embedding <=> $1
LIMIT 50;
```

The `embedding <=> $1` operator is the cosine distance. The same query with `embedding <-> $1` (L2) or `embedding <#> $1` (inner product) for different metrics. The `<=> < 0.3` filter is the pre-filter that combines with the index for sub-linear search.

### 2.2 AlloyDB's columnar engine and auto-scaling

Two AlloyDB features that make it 4× faster than vanilla pgvector:

- **Columnar engine**: column-oriented storage for analytical queries. For vector search, this means the `embedding` column is stored separately from `price`, `in_stock`, etc. A query that only reads `name` and `embedding` touches 2 columns instead of the whole row.
- **Auto-scaling storage**: the database layer scales compute and storage independently. You can add a read replica in 60 seconds; you can resize the primary without downtime. Compare to Cloud SQL for PostgreSQL, which requires a maintenance window for resize.

```python
# Add a read replica for read scaling (Python SDK)
from google.cloud import alloydb_v1

client = alloydb_v1.AlloyDBAdminClient()
instance = alloydb_v1.Instance(
    name="projects/PROJECT/locations/REGION/clusters/CLUSTER/instances/READ-REPLICA",
    instance_type=alloydb_v1.Instance.InstanceType.READ_POOL,
    node_count=2,
)
client.create_instance(parent="projects/PROJECT/locations/REGION/clusters/CLUSTER", instance_id="read-replica", instance=instance)
```

### 2.3 BigQuery `VECTOR_SEARCH()` — SQL-native

```sql
-- 1. Create or assume a table with an embedding column
CREATE TABLE rag_docs.embeddings (
    doc_id     STRING NOT NULL,
    chunk      STRING,
    embedding  ARRAY<FLOAT64>,
    metadata   JSON
);

-- 2. Insert (typically via ETL, not manual INSERT)
INSERT INTO rag_docs.embeddings (doc_id, chunk, embedding, metadata)
VALUES
    ('doc-1', 'What is RAG?', [0.1, 0.2, ...], JSON '{"source": "wiki"}'),
    ('doc-2', 'How does RAG work?', [0.15, 0.25, ...], JSON '{"source": "blog"}');

-- 3. Create a vector index (optional but recommended for >1M rows)
CREATE OR REPLACE VECTOR INDEX rag_docs_embeddings_idx
ON rag_docs.embeddings(embedding)
OPTIONS (distance_type = 'COSINE', index_type = 'TREE_AH', num_leaves = 100);

-- 4. Query with VECTOR_SEARCH — SQL-native semantic search
DECLARE query_embedding ARRAY<FLOAT64> DEFAULT [0.12, 0.22, ...];
SELECT
    base.doc_id,
    base.chunk,
    distance,
    base.metadata
FROM VECTOR_SEARCH(
    TABLE rag_docs.embeddings,
    'embedding',
    query_embedding,
    top_k => 10,
    distance_type => 'COSINE'
) AS results
JOIN rag_docs.embeddings AS base ON results.id = base.doc_id
ORDER BY distance
LIMIT 10;
```

The `VECTOR_SEARCH()` function returns a result table with `id` and `distance` columns; you join back to the source table for the actual data. The `top_k` parameter caps the result count; `distance_type` controls the metric (COSINE, EUCLIDEAN, DOT_PRODUCT).

### 2.4 When the index matters

| Table size | VECTOR_INDEX needed? | Query latency |
|-----------:|---------------------:|---------------|
| <100K rows | No — full table scan is fast | 50-200 ms |
| 100K-1M | Recommended (TREE_AH, low num_leaves) | 100-500 ms |
| 1M-100M | Required (TREE_AH, tuned num_leaves) | 200-1500 ms |
| >100M | Required + partitioning + cluster pruning | 1-10 seconds |

The index has a build cost (10-30 min for 10M rows) and storage cost (~30% of the embedding column size). Skip it for ad-hoc exploration; enable it for production.

### 2.5 Hybrid: BigQuery for batch + Vertex AI for online

A common production pattern:

```python
# BigQuery: nightly batch re-embeddings and bulk reindexing
bq_client.query("""
    INSERT INTO rag_docs.embeddings (doc_id, chunk, embedding)
    SELECT doc_id, chunk, ML.GENERATE_EMBEDDING(chunk, model_id => 'text-embedding-005') AS embedding
    FROM rag_docs.new_chunks
    WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
""")

# Vertex AI: stream the new embeddings to the online index
# (a Cloud Run job runs every 5 minutes, queries BQ for new rows, upserts to Vertex)
```

BigQuery handles the **batch + analytics** workload (compute embeddings at scale over 1M docs). Vertex AI handles the **online retrieval** workload (sub-50ms p99 for the live RAG service). A scheduled job keeps them in sync.

> 💡 **Tip**: the `ML.GENERATE_EMBEDDING()` BigQuery function calls Vertex AI's embedding models in SQL. You can compute embeddings for millions of documents without orchestrating a separate embedding pipeline.

---

## 3. Production Reality

### Decision matrix

| Workload | Best option | Reasoning |
|----------|-------------|----------|
| **Transactional + vector in one DB** (e.g., products + search) | **AlloyDB AI** | One ACID transaction, joins with business data, pgvector++ 4× faster than vanilla |
| **Embeddings already in BigQuery** (ETL pipeline feeds BQ) | **BigQuery `VECTOR_SEARCH()`** | No duplication, SQL-native, no extra infra |
| **Online retrieval at <50ms p99** (RAG, real-time recommendation) | **Vertex AI Vector Search** | Dedicated service, streaming update, public/private endpoints |
| **Multi-region with global reads** | **Vertex AI Vector Search** (multi-region) or **AlloyDB cross-region replicas** | Both support, but Vertex is simpler |
| **<1M vectors, dev/staging, sporadic queries** | **BigQuery `VECTOR_SEARCH()`** | Free tier, no extra infra |
| **Hybrid batch + online** | BigQuery for batch, Vertex AI for online | The standard GCP production pattern |
| **Hybrid search (BM25 + dense)** | Vertex AI Vector Search (2024+) or AlloyDB AI (manual RRF) | Vertex has it native; AlloyDB needs manual fusion |
| **Compliance / VPC-SC / CMEK** | **Vertex AI Vector Search** (CMEK) or **AlloyDB AI** (CMEK + VPC-SC) | Both are enterprise-ready |

### Cost comparison (10M vectors, 768-dim, 500 QPS, 24/7)

| Cost component | AlloyDB AI | BigQuery `VECTOR_SEARCH()` | Vertex AI Vector Search |
|----------------|-----------:|--------------------------:|----------------------:|
| Storage | $400 | $300 (Bifrost, on-demand) | $500-800 |
| Compute (steady state) | $800 (2× HA primary + 1 read replica) | $1,200 (slot reservation) | $1,200-1,800 |
| Query cost | included | included (slot-based) | $1,500-2,500 |
| **Total** | **~$1,200** | **~$1,500** | **~$3,300-5,300** |

AlloyDB AI is the **cheapest** of the three for a transactional workload (1/3 the cost of Vertex). BigQuery `VECTOR_SEARCH()` is in the middle. Vertex AI is the most expensive but the only one that hits <50ms p99 reliably.

### Spotify-on-GCP case study

The vault's existing Feast course (`[[../../../09 - MLOps y Produccion/27 - Feast and Feature Stores/03 - Feast in Production AWS and GCP]]`) covers Spotify's home-screen personalization stack on GCP. The vector retrieval layer uses:

- **Vertex AI Vector Search** for the online ranking (sub-20 ms p99)
- **BigQuery** for the offline batch re-embedding pipeline (10M tracks re-embedded nightly)
- **AlloyDB** is not used in this stack — Spotify's serving layer is purpose-built, not a transactional DB

This is the canonical pattern: **dedicated vector service for online + BigQuery for offline**. AlloyDB is the right answer when the vector workload is part of a transactional pipeline (e.g., product search where price, stock, and location are in the same query).

### Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Skipping VECTOR_INDEX in BigQuery | Slow queries on >1M rows | Always create `VECTOR_INDEX` for production tables |
| Using `<=>` without index in pgvector | O(N) linear scan | Always create the index for the column you query |
| Mixing distance operators in one index | Wrong results | Pick one operator per column (typically cosine for normalized) |
| Forgetting to set `num_leaves` correctly | Slow queries (too low) or high memory (too high) | `num_leaves ≈ sqrt(rows) / 10` is a good starting point |
| Treating BigQuery `VECTOR_SEARCH()` as a real-time API | 1-5 second latency on large tables | Use Vertex AI Vector Search for real-time; BigQuery for batch + ad-hoc |
| Connecting to AlloyDB without the connector | Connection failures from non-VPC environments | Use `google-cloud-alloydb-connector` Python SDK with IAM auth |
| Not enabling columnar engine | Slower than expected for analytical queries | Set `google_columnar_engine_enabled = on` at cluster level |

---

## 4. Code in Practice

```python
# 📦 AlloyDB AI + BigQuery VECTOR_SEARCH — single file, two options side by side.
# Requires: google-cloud-aiplatform>=1.50.0, google-cloud-alloydb-connector-py, sqlalchemy,
#           google-cloud-bigquery, db-dtypes, pg8000, numpy

import os, asyncio
import numpy as np
from google.cloud import bigquery
from google.cloud.alloydb.connector import Connector
import sqlalchemy

PROJECT = os.environ["GCP_PROJECT"]
REGION = "us-central1"
CLUSTER = "rag-cluster"
INSTANCE = "rag-primary"
DB = "postgres"
USER = "postgres"
DIM = 768

# ──── OPTION 1: AlloyDB AI ────

async def alloydb_demo():
    """AlloyDB AI with pgvector++ + ScaNN-based Tree-AH index."""
    connector = Connector()
    def getconn():
        return connector.connect(
            f"projects/{PROJECT}/locations/{REGION}/clusters/{CLUSTER}/instances/{INSTANCE}",
            "pg8000", user=USER, db=DB, enable_iam_auth=True,
        )
    pool = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)

    # 1. Create extension + table + index
    with pool.connect() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS products (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                price NUMERIC(10, 2),
                in_stock BOOLEAN DEFAULT true,
                embedding vector(768)
            )
        """))
        conn.execute(sqlalchemy.text("""
            CREATE INDEX IF NOT EXISTS products_embedding_idx
            ON products USING scann (embedding cosine_distance)
            WITH (num_leaves = 100, num_leaves_to_search = 10)
        """))
        conn.commit()
        print("AlloyDB: schema + index created")

    # 2. Insert 10,000 synthetic products
    rng = np.random.default_rng(42)
    with pool.connect() as conn:
        for i in range(0, 10_000, 500):
            rows = [
                {"name": f"Product {j}", "description": f"Desc {j}",
                 "price": float(rng.uniform(5, 500)), "in_stock": True,
                 "embedding": rng.normal(size=DIM).tolist()}
                for j in range(i, min(i + 500, 10_000))
            ]
            conn.execute(sqlalchemy.text("""
                INSERT INTO products (name, description, price, in_stock, embedding)
                VALUES (:name, :description, :price, :in_stock, :embedding)
            """), rows)
        conn.commit()
        print(f"AlloyDB: inserted 10,000 products")

    # 3. Query: semantic search + business filter in ONE SQL statement
    query_embedding = rng.normal(size=DIM).tolist()
    with pool.connect() as conn:
        results = conn.execute(sqlalchemy.text("""
            SELECT id, name, price, embedding <=> :q AS distance
            FROM products
            WHERE in_stock = true AND price < 100 AND embedding <=> :q < 0.4
            ORDER BY embedding <=> :q LIMIT 10
        """), {"q": query_embedding}).fetchall()
    print(f"AlloyDB: top-10 (price<100, in_stock, distance<0.4): {[(r.name, float(r.price), round(r.distance, 3)) for r in results]}")

    connector.close()

# ──── OPTION 2: BigQuery VECTOR_SEARCH ────

def bigquery_demo():
    """BigQuery VECTOR_SEARCH with TREE_AH vector index."""
    bq = bigquery.Client(project=PROJECT)
    rng = np.random.default_rng(42)

    # 1. Create dataset + table with embedding column
    dataset_id = "rag_docs_demo"
    table_id = f"{PROJECT}.{dataset_id}.embeddings"
    bq.create_dataset(dataset_id, exists_ok=True)
    schema = [
        bigquery.SchemaField("doc_id", "STRING"),
        bigquery.SchemaField("chunk", "STRING"),
        bigquery.SchemaField("embedding", "REPEATED", field_type="FLOAT64"),
    ]
    bq.create_table(f"{table_id}", schema=schema, exists_ok=True)

    # 2. Insert 10,000 synthetic docs via load table
    rows = [{
        "doc_id": f"doc-{i}",
        "chunk": f"This is document {i} about topic {i % 10}.",
        "embedding": rng.normal(size=DIM).tolist(),
    } for i in range(10_000)]
    errors = bq.insert_rows_json(f"{table_id}.{dataset_id}.embeddings", rows)
    print(f"BigQuery: inserted 10,000 docs (errors: {len(errors)})")

    # 3. Create VECTOR_INDEX (TREE_AH)
    bq.query(f"""
        CREATE OR REPLACE VECTOR INDEX `{dataset_id}.embeddings_embedding_idx`
        ON `{table_id}`(embedding)
        OPTIONS (distance_type = 'COSINE', index_type = 'TREE_AH', num_leaves = 100)
    """).result()
    print("BigQuery: VECTOR_INDEX created (TREE_AH)")

    # 4. Query with VECTOR_SEARCH — SQL-native
    query_embedding = rng.normal(size=DIM).tolist()
    sql = f"""
        SELECT base.doc_id, base.chunk, distance
        FROM VECTOR_SEARCH(
            TABLE `{table_id}`,
            'embedding',
            {query_embedding},
            top_k => 10,
            distance_type => 'COSINE'
        ) AS results
        JOIN `{table_id}` AS base ON results.id = base.doc_id
        ORDER BY distance
    """
    results = bq.query(sql).result()
    rows_list = list(results)
    print(f"BigQuery: top-10 (TREE_AH): {[(r['doc_id'], round(r['distance'], 3)) for r in rows_list]}")

# ──── Run both ────
asyncio.run(alloydb_demo())
bigquery_demo()
```

```bash
export GCP_PROJECT=your-gcp-project
python scratch.py
```

---

## 🎯 Key Takeaways

- **AlloyDB AI** is the right answer for transactional + vector workloads. Postgres-compatible, ACID, joins, 4× faster than vanilla pgvector via columnar engine and ScaNN-based index.
- **BigQuery `VECTOR_SEARCH()`** is the right answer when embeddings already live in BigQuery. SQL-native, batch-optimized, no extra infra. Skip the VECTOR_INDEX for <100K rows; enable for production tables >1M.
- **Vertex AI Vector Search** (note `01`) is the right answer for online retrieval at <50ms p99. The 2-3× cost premium buys unified IAM, compliance, and dedicated throughput.
- **The standard production pattern** is BigQuery for batch + offline analytics, Vertex AI for online retrieval, AlloyDB for transactional-with-vector. Many production stacks use all three.
- **`ML.GENERATE_EMBEDDING()` in BigQuery** calls Vertex AI's embedding models in SQL — compute embeddings for millions of documents without orchestrating a separate pipeline.
- **Cost at 10M vectors, 500 QPS**: AlloyDB ~$1,200/month, BigQuery ~$1,500/month, Vertex AI ~$3,300-5,300/month. Premium for Vertex buys latency, not just features.
- **Decision criterion**: where does the vector data already live? BigQuery → `VECTOR_SEARCH()`. Postgres → AlloyDB AI. RAG corpus with no other home → Vertex AI.

## References

- AlloyDB AI docs: https://cloud.google.com/alloydb/docs/ai
- BigQuery `VECTOR_SEARCH()` docs: https://cloud.google.com/bigquery/docs/vector-search
- `google-cloud-alloydb-connector`: https://github.com/GoogleCloudPlatform/alloydb-python-connector
- BigQuery `ML.GENERATE_EMBEDDING()`: https://cloud.google.com/bigquery/docs/generate-embedding
- AlloyDB ScaNN index: https://cloud.google.com/alloydb/docs/ai/vector-index
- Spotify on GCP (Feast course): `[[../../../09 - MLOps y Produccion/27 - Feast and Feature Stores/03 - Feast in Production AWS and GCP]]`
- Home Depot on AlloyDB: https://cloud.google.com/blog/topics/developers-practitioners/alloydb-ai-now-generally-available
- Related Vault: `[[../01 - Vertex AI Vector Search - Architecture and Python Client]]`
- Related Vault: `[[../../../10 - Cloud, Infra y Backend/36 - PostgreSQL for AI-ML Workloads/01 - pgvector Production Tuning - HNSW, Quantization and Hybrid Search]]`
- Related Vault: `[[../../../10 - Cloud, Infra y Backend/28 - BigQuery for ML/01 - BigQuery Architecture]]`
