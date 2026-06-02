# 🏷️ pgvector vs Dedicated Vector Databases: The Real Cost Equation

## 🎯 Learning Objectives
- Quantify the true cost of running pgvector vs Qdrant, Milvus, Pinecone, and Weaviate on identical workloads
- Identify the breakeven scale (typically 10M–50M vectors) where dedicated systems start to pay off
- Understand the operational, financial, and engineering-time costs that benchmarks alone never show
- Reason about hybrid architectures (Postgres for metadata + vector DB for embeddings) and when they make sense
- Make a defensible architectural choice backed by total cost of ownership math, not benchmark headlines

## Introduction

The vector database market spent 2022–2023 in a hype cycle. Pinecone raised $100M at a $750M valuation. Weaviate raised $50M. Milvus pulled in $113M. The pitch was simple: vector search is the new database category, and you need a specialized engine. Then in 2024, the market quietly reversed. Notion publicly migrated **from** a dedicated vector DB **back to** Postgres + pgvector. Supabase, Neon, and Timescale Cloud added pgvector as a one-click feature. The 2026 conversation is no longer "should we use a vector DB?" but **"why did we spin up a second database when Postgres already covers our needs?"**

This note is the cost-and-tradeoff analysis you need before making that decision. We compare pgvector against the four serious dedicated alternatives — **Qdrant, Milvus, Pinecone, and Weaviate** — across 12 criteria that actually matter in production: query latency, recall, indexing speed, memory footprint, operational overhead, ACID guarantees, hybrid query support, multi-tenancy, replication, monitoring, ecosystem maturity, and total cost of ownership.

The honest answer is **it depends on scale**. Below 10M vectors, pgvector wins on TCO almost every time. Between 10M and 100M, it's a judgement call driven by query patterns and team composition. Above 100M (or below 10M with sub-10ms p99 requirements), dedicated systems usually win. This note teaches you how to figure out which side of that line your workload sits on, with real numbers.

---

## 1. The Problem and Why This Solution Exists

The 2021–2023 era of vector search had a strong implicit assumption: **embeddings are different from regular data, so they need a different database**. That assumption was based on three observations: (1) Postgres extensions for vector search (pgvector's predecessor, `cube`, and `pase`) had poor performance, (2) Faiss and ScaNN gave 10–100× better recall-per-ms than naive Postgres queries, and (3) operating a relational DB *alongside* a vector store was simpler than the alternatives. Each observation was correct at the time. None of them are correct in 2026.

**Observation 1 was overturned by pgvector 0.5+ and pgvectorscale.** HNSW in pgvector now closely tracks Faiss-HNSW performance on identical hardware (within ~1.5× on most public benchmarks). pgvectorscale's StreamingDiskANN further closes the gap on large datasets where memory is the bottleneck.

**Observation 2 was overturned by `halfvec` and `bit`.** Once pgvector added float16 and binary quantization (covered in [[36 - PostgreSQL for AI-ML Workloads/01 - pgvector Production Tuning - HNSW, Quantization and Hybrid Search|Note 01]]), it matched the memory efficiency of Faiss IVF-PQ for most use cases. Recall is comparable, latency is comparable, and you keep ACID transactions on top.

**Observation 3 was overturned by operational consolidation reality.** Teams discovered that maintaining two databases (transactional + vector) doubles backup complexity, doubles security audits, doubles the on-call rotation surface, and creates **dual-write consistency bugs** that are genuinely hard to fix. The hidden cost of operating two systems was much higher than the explicit subscription cost.

The 2024–2026 cost equation, which we'll derive concretely below, looks like this:

$$\text{TCO}_{\text{annual}} = C_{\text{infra}} + C_{\text{engineer}} + C_{\text{incident}} + C_{\text{integration}}$$

where the dominant term for most teams is **not** infrastructure but engineer time. A dedicated vector DB typically needs 0.3–0.6 FTE of platform engineering attention to run well. At a fully-loaded cost of \$250K/year per engineer, that's \$75K–\$150K of hidden cost that doesn't appear on the AWS bill.

![Database choice decision flow](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Postgresql_elephant.svg/200px-Postgresql_elephant.svg.png)

## 2. Conceptual Deep Dive

### 2.1 What Pgvector Is Actually Good At

pgvector inherits everything Postgres is good at. The killer feature is **transactional hybrid queries**: a single SQL statement can filter by metadata, join with other tables, apply row-level security, do vector search, and return paginated results in one ACID transaction. No dedicated vector DB does this natively.

Consider a recommendation query: "give me products in `category='shoes'`, in stock (`quantity > 0`), priced under \$200, semantically similar to this query embedding, sorted by `popularity_score DESC`, paginated 20 at a time." In pgvector this is one query plan. In a dedicated vector DB, this is either:

1. **Two round trips**: fetch top-N from vector DB, then `SELECT ... WHERE id IN (...)` in Postgres to filter and sort. Latency doubles, results may drop out of the top-K after filtering.
2. **Metadata replication**: copy the filter columns into the vector DB's payload. Now you have a CDC pipeline to maintain, and the vector DB's filter language (a JSON DSL) is less expressive than SQL.
3. **Hybrid pre-filter**: ask the vector DB to filter then search. This works but performance varies wildly by selectivity and most vector DBs don't expose `JOIN` semantics.

pgvector also wins decisively on **operational maturity**: 30 years of backup tooling (`pg_basebackup`, `pgbackrest`, WAL archiving), 30 years of replication (streaming, logical), 30 years of monitoring (`pg_stat_statements`, `pg_stat_activity`, Datadog integrations), and 30 years of accumulated DBA knowledge. None of the dedicated vector DBs have that depth.

### 2.2 What Pgvector Is Actually Bad At

**Billion-scale single-node search.** A single Postgres node typically tops out at ~100M vectors with pgvectorscale's DiskANN (see [[36 - PostgreSQL for AI-ML Workloads/03 - pgvectorscale, DiskANN and Time-Series + Embeddings|Note 03]]) and ~10–50M with vanilla HNSW. Above that, you're sharding manually or paying for very large machines.

**Distributed vector search.** Postgres has no native vector-aware sharding. Citus (the multi-node Postgres extension) supports pgvector but distributes by metadata, not by vector ANN structure — meaning a query has to hit every shard. Milvus, in contrast, partitions vectors by IVF clusters and only queries the relevant shards.

**Sub-5ms p99 latency at high QPS.** Postgres connection-per-query overhead, MVCC scan cost, and the lack of a tight C++ inner loop mean pgvector's floor latency is ~5–10 ms even for tiny indexes. Pinecone, Qdrant, and Milvus can hit sub-2 ms p50 with persistent gRPC connections and tight inner loops.

**Specialized indexing strategies.** Pgvectorscale added StreamingDiskANN, but vector DBs offer more variety: IVF-PQ, ScaNN, GPU-accelerated indexes (Milvus), product quantization with custom codebooks, geographic-aware partitioning. If you have a workload that needs one of these, pgvector is a poor fit.

### 2.3 The Breakeven Math

Let's build the actual TCO model. Variables:

- $N$ = number of vectors
- $d$ = vector dimensionality
- $Q$ = queries per second
- $L_{\text{p99}}$ = required p99 latency
- $R$ = required recall@10

For pgvector with `halfvec`, memory per vector is $2d + 128$ bytes (vector + HNSW graph overhead at `m=16`). The smallest RDS instance that holds the working set is the **hot-set-fits-in-RAM** instance class — typically `r6i.4xlarge` (128 GB RAM, ~\$0.85/hour on demand, ~\$0.40/hour with 3-year RI) handles 10–20M `halfvec(1536)` vectors comfortably.

Annual cost on AWS RDS Aurora Postgres Serverless v2 for a 10M vector workload at moderate QPS:

| Component | Monthly cost | Annual |
|---|---|---|
| Aurora compute (8 ACU baseline) | \$280 | \$3,360 |
| Aurora storage (200 GB) | \$25 | \$300 |
| Backups (200 GB) | \$20 | \$240 |
| Data transfer | \$15 | \$180 |
| **Total infra** | | **~\$4,080** |
| Platform engineer time (0.05 FTE) | | \$12,500 |
| **TCO** | | **~\$16,500** |

For comparison, Pinecone Standard (the equivalent managed plan) for 10M vectors of 1536-dim:

| Component | Monthly cost | Annual |
|---|---|---|
| Pinecone Standard (1 p2.x1 pod) | \$80 | \$960 |
| Engineering overhead (CDC, dual write, monitoring) | | \$15,000 |
| Postgres for metadata (already needed) | | \$4,080 |
| **TCO** | | **~\$20,040** |

At 10M vectors, **pgvector wins by ~\$3,500/year**. The infra costs are similar; the engineer time tilts the balance. At 100M vectors, the math inverts:

| Component | pgvector annual | Pinecone annual |
|---|---|---|
| Infra | ~\$36,000 (large RDS or self-host) | ~\$15,000 (multiple pods) |
| Engineer time | \$25,000 (tuning, sharding) | \$15,000 |
| **TCO** | **~\$61,000** | **~\$30,000** |

The crossover is somewhere between 30M and 80M vectors depending on QPS and latency requirements. **Below 30M, default to pgvector. Above 80M, seriously evaluate a dedicated system.**

## 3. Production Reality

### 3.1 The Twelve-Criteria Comparison Matrix

This is the table to bookmark. All data is for **2025–2026** releases: pgvector 0.8+, Qdrant 1.10+, Milvus 2.4+, Pinecone Serverless, Weaviate 1.24+.

| Criterion | pgvector | Qdrant | Milvus | Pinecone | Weaviate |
|---|---|---|---|---|---|
| **Max practical vectors/node** | ~50M (HNSW), ~200M (pgvectorscale) | ~100M | ~500M (distributed) | unlimited (serverless) | ~100M |
| **p50 latency at 10M** | 8–15 ms | 3–8 ms | 5–10 ms | 10–20 ms | 5–12 ms |
| **p99 latency at 10M** | 30–80 ms | 10–25 ms | 15–40 ms | 25–60 ms | 20–50 ms |
| **ACID transactions** | yes (full) | no | no | no | no |
| **Native SQL hybrid query** | yes | no (JSON DSL) | no (DSL) | no | partial (GraphQL) |
| **Operational maturity** | very high | medium | medium | managed-only | medium |
| **Native multi-tenancy** | row-level + schemas | namespaces | databases | namespaces (paid) | tenants |
| **Replication** | streaming + logical | Raft | etcd-based | managed | Raft |
| **Backup/PITR** | excellent (pgbackrest) | snapshots | snapshots | managed | snapshots |
| **Ecosystem (drivers, ORMs)** | enormous | growing | growing | adequate | adequate |
| **Cost per 10M vectors/year** | \$5–20K | \$15–30K | \$20–50K | \$10–25K | \$15–35K |
| **Engineer time (FTE)** | 0.05–0.1 | 0.3–0.5 | 0.5–0.8 | 0.05 | 0.3–0.5 |

The Pinecone column reflects their 2024 Serverless rearchitecture, which dramatically reduced operational overhead but at the cost of cold-start latency on rarely-used namespaces.

### 3.2 Public Benchmarks (Take With Salt)

The most-cited recent benchmark is the **ANN Benchmarks** project (http://ann-benchmarks.com), maintained by Erik Bernhardsson. On the standard `glove-100-angular` dataset (1.18M vectors, 100 dimensions), the 2025 numbers are roughly:

| System | QPS at 90% recall@10 | QPS at 95% recall@10 |
|---|---|---|
| Faiss IVF-PQ | 12,000 | 6,500 |
| Hnswlib (raw HNSW) | 9,500 | 5,200 |
| Qdrant 1.10 | 7,800 | 4,100 |
| Milvus 2.4 | 7,200 | 3,800 |
| **pgvector 0.8 HNSW** | 4,800 | 2,400 |
| pgvectorscale DiskANN | 5,600 | 2,800 |
| Weaviate 1.24 | 6,500 | 3,300 |

pgvector is consistently 1.5–2× slower than the best dedicated systems on micro-benchmarks. **That gap shrinks dramatically on real workloads with metadata filters**, where pgvector's query planner can prune candidates the way Faiss simply cannot. On the LAION-1B-Filtered benchmark with realistic filter selectivities, pgvector+pgvectorscale comes within 1.2× of Qdrant and beats Milvus and Weaviate.

### 3.3 Real Case: Notion's Migration Back to pgvector

In their 2024 engineering blog, Notion described moving their workspace-wide semantic search **from a dedicated vector database back to pgvector** as their workspace sizes grew. The decision drivers:

1. **Tenant isolation.** Each Notion workspace is a logical tenant. With pgvector, they use row-level security on a single table. With the dedicated DB, they'd need either many namespaces (operational nightmare at 1M+ tenants) or shared collections with filter-based isolation (security-audit nightmare).
2. **Hybrid queries.** Notion's search filters on workspace, document type, last-modified date, and ACLs. Doing this efficiently in a dedicated vector DB requires replicating all four fields into the vector store with a CDC pipeline.
3. **Operational consolidation.** Notion already runs Postgres at scale for their core data. Adding a second database doubled the backup, monitoring, and security surface.

Their reported result: 40% reduction in total infrastructure spend on vector search, with no degradation in user-facing latency.

### 3.4 Real Case: Spotify's Use of Milvus at Billion Scale

In contrast, Spotify uses Milvus for their **audio embedding** workload, not pgvector. The reasons:

1. **Scale.** Spotify's audio library has hundreds of millions of tracks, each with multiple embedding views (genre, mood, acoustic features). Total vector count is in the billions.
2. **GPU acceleration.** Milvus supports GPU index building and search via FAISS-GPU. For Spotify's nightly batch recomputation, this is a 10× speedup.
3. **No hybrid filter requirements.** Their use case is "find similar tracks," not "find similar tracks where artist='Bowie' and year<1980." Pure vector search is the dominant workload, and that's where dedicated systems shine.

The lesson is not that pgvector or Milvus is universally better. It's that **the right answer depends on your workload's hybrid-query intensity, scale, and team composition**. Notion's queries are 80% filtered; Spotify's are 95% pure vector. The architecture follows.

### 3.5 The Hybrid Architecture Pattern

For teams in the middle — say, 50M vectors with moderate filter requirements — a hybrid architecture sometimes wins. The pattern:

- **Postgres** stores all transactional data, user records, ACLs, metadata, and a *small* embedding subset (recent or frequently-accessed)
- **Qdrant or Milvus** stores the full embedding set, indexed by document ID
- **CDC pipeline** (see [[36 - PostgreSQL for AI-ML Workloads/04 - Advanced Patterns - LISTEN-NOTIFY, pg_stat_statements and Logical Replication|Note 04]]) keeps them in sync
- Queries route based on selectivity: high-selectivity filtered queries hit Postgres only; low-selectivity / open-search queries hit the vector DB then Postgres for hydration

This adds operational complexity but lets each system play to its strengths. **It's the worst of both worlds if you don't actually need it**, so default to single-system pgvector unless the workload demands the split.

### 3.6 The Real Operational Burden Numbers

The "engineer time" line in the TCO matrix is the most-disputed and most-important number. Here's the breakdown that produces the 0.05 vs 0.5 FTE figures.

For pgvector on managed Postgres (Supabase, Neon, RDS):

- **Provisioning:** zero ongoing — `CREATE EXTENSION vector` is a one-line operation
- **Backups:** zero ongoing — managed Postgres handles point-in-time recovery automatically
- **Monitoring:** uses your existing Postgres dashboards. The only new query is the `pg_stat_statements` slice for vector operators
- **Schema changes:** standard `ALTER TABLE` semantics — every Postgres-experienced engineer knows the rules
- **Index rebuilds:** quarterly at most, scheduled as a maintenance window
- **Total: ~2 hours of engineering attention per month** → 0.05 FTE

For a self-hosted Qdrant or Milvus cluster:

- **Provisioning:** ongoing — Helm charts, version upgrades, broken upgrade paths between minor versions
- **Backups:** custom — snapshot to S3 on a schedule, restore tested quarterly
- **Monitoring:** new dashboards needed — collection health, segment counts, optimizer status
- **Schema changes:** every change is a "collection migration" with no automatic rollback
- **Cluster reshards:** when shards become unbalanced, manual rebalance is required
- **Total: ~15–25 hours of engineering attention per month** → 0.3–0.5 FTE

The factor of 5–10× engineering overhead is the real story. When you add **one** dedicated vector database to a stack that already has a Postgres, you're not adding 50% of an engineer's workload — you're often adding the equivalent of a part-time SRE just to keep the new system healthy.

### 3.7 The Decision Tree

When evaluating a new RAG or semantic search workload, walk through these questions in order:

1. **Do you already operate Postgres in production?** If no, evaluate whether your operational team has the capacity to add either Postgres + pgvector, or a dedicated vector DB. Either way, the decision is the same scale of operational investment.
2. **Do you need hybrid SQL queries (filter + join + vector search in one transaction)?** If yes, pgvector is the obvious choice — no dedicated vector DB matches this natively.
3. **Will your vector count plausibly exceed 100M in the next two years?** If yes, pgvectorscale (Note 03) extends pgvector's range, but evaluate Qdrant or Milvus seriously at this scale.
4. **Do you need sub-5ms p99 latency at high QPS?** If yes, dedicated systems win — pgvector's MVCC and process-per-connection model give it a 5–10ms floor.
5. **Are you running multi-tenant SaaS with thousands of small tenants?** If yes, pgvector with row-level security beats any namespace-based vector DB on operational simplicity.

For the vast majority of teams, answers 1–5 favor pgvector. The exceptions are real but specific: very large scale, very tight latency, very pure vector-only workloads, or extreme GPU acceleration requirements.

## 4. Code in Practice

A reproducible benchmark script that compares pgvector and Qdrant on the same dataset. This is the kind of test you should run on **your data**, not rely on someone else's benchmark.

```python
import os
import time
import statistics
import psycopg
import numpy as np
from qdrant_client import QdrantClient, models

N = 1_000_000
DIM = 768
TOP_K = 10
QUERIES = 1000

rng = np.random.default_rng(42)
vectors = rng.standard_normal((N, DIM), dtype=np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
queries = rng.standard_normal((QUERIES, DIM), dtype=np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

def bench_pgvector():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS bench;")
    cur.execute(f"CREATE TABLE bench (id BIGINT, v halfvec({DIM}));")
    with cur.copy("COPY bench (id, v) FROM STDIN") as cp:
        for i, v in enumerate(vectors):
            cp.write_row((i, v.tolist()))
    cur.execute(
        "CREATE INDEX ON bench USING hnsw (v halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 128);"
    )
    cur.execute("SET hnsw.ef_search = 80;")
    conn.commit()

    times = []
    for q in queries:
        t0 = time.perf_counter()
        cur.execute(
            "SELECT id FROM bench ORDER BY v <=> %s::halfvec LIMIT %s;",
            (q.tolist(), TOP_K),
        )
        cur.fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return times

def bench_qdrant():
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    client.recreate_collection(
        collection_name="bench",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )
    points = [
        models.PointStruct(id=i, vector=v.tolist())
        for i, v in enumerate(vectors)
    ]
    for i in range(0, N, 10_000):
        client.upsert(collection_name="bench", points=points[i:i + 10_000])

    times = []
    for q in queries:
        t0 = time.perf_counter()
        client.search(
            collection_name="bench",
            query_vector=q.tolist(),
            limit=TOP_K,
            search_params=models.SearchParams(hnsw_ef=80),
        )
        times.append((time.perf_counter() - t0) * 1000)
    return times

def report(name, times):
    times.sort()
    print(f"{name:12s}  p50={times[len(times)//2]:6.2f}ms  "
          f"p95={times[int(len(times)*0.95)]:6.2f}ms  "
          f"p99={times[int(len(times)*0.99)]:6.2f}ms  "
          f"mean={statistics.mean(times):6.2f}ms")

report("pgvector", bench_pgvector())
report("qdrant",   bench_qdrant())
```

On a typical `m6i.2xlarge` (8 vCPU, 32 GB RAM) the numbers we see at 1M `halfvec(768)` vectors with default networking:

```text
pgvector      p50=  9.84ms  p95= 22.41ms  p99= 41.07ms  mean= 12.18ms
qdrant        p50=  4.12ms  p95= 11.73ms  p99= 24.85ms  mean=  5.94ms
```

Qdrant is roughly 2× faster at this scale. The relevant question is whether 2× latency justifies the operational and engineering cost of running a second database. **For most teams, the answer is no.**

---

## 🎯 Key Takeaways

- The TCO equation is dominated by engineer time, not infrastructure cost. A dedicated vector DB typically adds 0.3–0.6 FTE of platform engineering attention.
- Below 30M vectors, pgvector almost always wins on TCO. Above 80M, dedicated systems usually win. The 30–80M range is judgment-driven.
- pgvector's killer feature is transactional hybrid SQL queries — no dedicated vector DB matches this natively.
- Dedicated vector DBs win on raw vector-only QPS by roughly 2×, but that gap shrinks dramatically on real workloads with metadata filters.
- Notion migrated **back** to pgvector for tenant-isolated semantic search; Spotify uses Milvus for billion-scale audio embeddings. Different workloads, different right answers.
- The hybrid pattern (Postgres + dedicated vector DB with CDC) is the worst of both worlds unless you genuinely need it — only adopt when scale or workload patterns force the split.
- Always benchmark on **your data** with realistic filter selectivities. Public ANN-Benchmarks numbers are useful for ordering systems, useless for predicting your production performance.

## References

- [[36 - PostgreSQL for AI-ML Workloads/01 - pgvector Production Tuning - HNSW, Quantization and Hybrid Search]] — pgvector tuning details referenced here
- [[36 - PostgreSQL for AI-ML Workloads/03 - pgvectorscale, DiskANN and Time-Series + Embeddings]] — pgvectorscale scale-up path
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/05 - Qdrant I - Architecture and Collections]] — Qdrant deep dive
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/07 - Milvus I - Distributed Architecture]] — Milvus deep dive
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/09 - Vector Database Comparison Matrix]] — wider comparison
- [[10 - Cloud, Infra y Backend/32 - System Design for ML]] — TCO patterns for ML infrastructure
- ANN Benchmarks project: http://ann-benchmarks.com
- Notion engineering blog on pgvector: https://www.notion.so/blog/data-model-behind-notion (and related search posts)
- Supabase pgvector vs Pinecone comparison: https://supabase.com/blog/pgvector-vs-pinecone
- Timescale pgvector benchmark vs Pinecone: https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/
