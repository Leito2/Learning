# ☁️ Welcome to Vector Search on Google Cloud

## 🎯 Learning Objectives

- Map the **three GCP-native options** for vector search (Vertex AI Vector Search, AlloyDB AI, BigQuery `VECTOR_SEARCH()`) and the workloads each one wins
- Trace the **architectural evolution** from Matching Engine (2019) to Vector Search (2023 rebrand) to the 2024 hybrid search addition
- Distinguish the **Index**, **IndexEndpoint**, and **DeployedIndex** abstractions in Vertex AI Vector Search and the public-vs-private endpoint trade-off
- Predict **latency, cost, and ops overhead** for each option at 1M, 10M, and 100M vector scale
- Use the **`google-cloud-aiplatform` Python SDK** to create indexes, deploy endpoints, upsert datapoints, and query
- Apply the **decision framework** when starting a new GCP-based project, and the migration path when outgrowing one option for another

---

## Why This Course Exists

GCP offers **three production-ready ways** to do vector search:

1. **Vertex AI Vector Search** — the GCP-native managed service, formerly "AI Platform Matching Engine". Tree-AH index (ScaNN-derived), public or private endpoints, streaming update.
2. **AlloyDB AI** — pgvector++ on Google Cloud's Postgres-compatible engine. Auto-scaling storage, columnar engine, 4× faster than vanilla pgvector.
3. **BigQuery `VECTOR_SEARCH()`** (2024+) — SQL-native vector search on Google Cloud's serverless data warehouse. Best for batch + ad-hoc, not for sub-100ms online retrieval.

The vault already covers the **non-vector** parts of the GCP ML stack: `[[../../../10 - Cloud, Infra y Backend/28 - BigQuery for ML]]` (BigQuery architecture, BigQuery ML, Vertex AI Feature Store, Vertex AI Pipelines), `[[../../../09 - MLOps y Produccion/27 - Feast and Feature Stores/03 - Feast in Production AWS and GCP|09/27/03 Feast on GCP]]` (Feast + Vertex AI), and the portfolio context in `[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]]` (LiteLLM as a unified interface over 100+ providers including Vertex AI). What was missing: **the dedicated vector search path**. This course fills that gap.

The user's **Automated LLM Evaluation Suite** portfolio project runs on AWS SageMaker + GCP Vertex AI. A natural extension: store the eval corpus embeddings in Vertex AI Vector Search for semantic drift detection across model versions. This course makes that extension concrete.

---

## Course Map

| Note | Title | What You Will Master |
|------|-------|----------------------|
| `00` | **Welcome** *(this note)* | The 3 GCP options, decision tree, prerequisites |
| `01` | **Vertex AI Vector Search — Architecture and Python Client** | Index types (Tree-AH, brute force), IndexEndpoint, deployed index, gRPC vs REST, hybrid search, IAM, VPC |
| `02` | **AlloyDB, BigQuery `VECTOR_SEARCH()` and the GCP Decision Framework** | AlloyDB AI auto-scale, BigQuery SQL vector search, Spotify-on-GCP case study, decision matrix |

---

## Prerequisites

| Concept | Where to Review |
|---------|----------------|
| BigQuery architecture and SQL | `[[../../../10 - Cloud, Infra y Backend/28 - BigQuery for ML/01 - BigQuery Architecture|10/28/01]]` |
| Vertex AI ecosystem (Feature Store, Endpoints, Pipelines) | `[[../../../10 - Cloud, Infra y Backend/28 - BigQuery for ML/02 - BigQuery ML and GCP Ecosystem|10/28/02]]` |
| Vector search theory (ANN, HNSW, distance metrics) | `[[../../33 - Vector Databases and Semantic Search/01 - Vector Search Fundamentals|10/33/01]]` |
| pgvector production tuning | `[[../../../10 - Cloud, Infra y Backend/36 - PostgreSQL for AI-ML Workloads/01|10/36/01]]` |
| Qdrant / Pinecone as comparison peers | `[[../05 - Qdrant I|10/33/05]]`, `[[../12 - Pinecone|10/33/12]]` |

---

## The Decision Tree (Quick Reference)

```
Where does the vector data live?
│
├── Inside BigQuery tables (you already ETL to BQ)
│   └── BigQuery VECTOR_SEARCH() — SQL-native, no extra infra
│       Best for: batch + ad-hoc + dashboards
│
├── In a Postgres-compatible store (you already use Postgres)
│   └── AlloyDB AI with pgvector — 4× faster than vanilla pgvector
│       Best for: transactional + vector in one DB
│
└── In a dedicated vector store (RAG corpus, semantic search, retrieval)
    └── Vertex AI Vector Search — managed, low-latency, public or VPC endpoint
        Best for: <50ms p99 online retrieval, multi-region, GCP-native RAG
```

The deeper matrix in note `02` covers cost, latency, ops, and migration paths between options.

---

## Project Outcome

By the end of this course you will have:

- A **Vertex AI Vector Search index** deployed to a public endpoint, queryable via the `google-cloud-aiplatform` Python SDK
- An **AlloyDB cluster** with the `vector` extension and `google_ml_integration` enabled, indexed with `IVF` or `HNSW`
- A **BigQuery table with `VECTOR_SEARCH()` queries** for batch and ad-hoc retrieval
- A **decision document** that picks the right option for your specific workload (1M, 10M, 100M vectors, batch vs online, single-region vs multi-region)

---

## 🔗 Related Vault Modules

| Module | Connection |
|--------|-----------|
| `[[../../../10 - Cloud, Infra y Backend/28 - BigQuery for ML]]` | BigQuery architecture, BigQuery ML, Vertex AI Feature Store context |
| `[[../../../10 - Cloud, Infra y Backend/36 - PostgreSQL for AI-ML Workloads]]` | pgvector production tuning, pgvectorscale, AlloyDB peer |
| `[[../../33 - Vector Databases and Semantic Search]]` | Cross-engine comparison, Qdrant / Milvus / pgvector deep dives |
| `[[../12 - Pinecone Architecture and Python Client]]` | The non-GCP managed alternative for comparison |
| `[[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]]` | LiteLLM unifies Vertex AI as one of 100+ LLM providers |
| `[[../../../09 - MLOps y Produccion/27 - Feast and Feature Stores/03 - Feast in Production AWS and GCP]]` | Feast on GCP, Vertex AI as feature store + online serving |

## 📝 How to Use These Notes

Each note follows the **Deep Format**: Learning Objectives, Introduction, four numbered sections (Problem → Conceptual Deep Dive → Production Reality → Code in Practice), Key Takeaways, Compression Code, References. Theory precedes code. Every code block is runnable against the live `google-cloud-aiplatform` SDK (≥1.50).

Set up your environment with:

```bash
uv add google-cloud-aiplatform>=1.50.0 google-cloud-alloydb-connector-py sqlalchemy \
       google-cloud-bigquery db-dtypes pandas
gcloud auth application-default login    # one-time, for local dev
gcloud config set project YOUR_GCP_PROJECT
```
