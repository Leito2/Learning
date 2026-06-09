# 🔍 Welcome to Vector Databases and Semantic Search

Welcome to the **Vector Databases and Semantic Search** course. This course covers the theory, implementation, and production deployment of vector search systems that power modern AI applications — from RAG pipelines to semantic caching and recommendation systems.

## 📚 Course Map

| Note | Title | What You'll Master |
|------|-------|-------------------|
| `00` | **Welcome** | Course overview, prerequisites, setup |
| `01` | **Vector Search Fundamentals** | Embeddings, distance metrics, ANN vs KNN, curse of dimensionality |
| `02` | **Indexing Algorithms Deep Dive** | IVF, HNSW, PQ, OPQ, DiskANN, ScaNN, trade-offs |
| `03` | **pgvector I — Core Operations and Indexing** | PostgreSQL extension CRUD, indexing, distance operators |
| `04` | **pgvector II — Production and Hybrid Search** | Partitioning, hybrid search, pooling, monitoring, backups |
| `05` | **Qdrant I — Architecture and Collections** | Rust-based engine, collections, points, payload indexing, APIs |
| `06` | **Qdrant II — Distributed and Cloud Deployment** | Cluster mode, shards, replicas, cloud, LangChain integration |
| `07` | **Milvus I — Distributed Architecture** | Cloud-native design, data model, index building, consistency |
| `08` | **Milvus II — Kubernetes and Multi-tenancy** | K8s deployment, operator, multi-tenancy, tiered storage |
| `09` | **Vector Database Comparison Matrix** | Feature-by-feature comparison: pgvector vs Qdrant vs Milvus |
| `10` | **Advanced Patterns and Observability** | Hybrid search, late interaction, RAG patterns, monitoring |
| `11` | **Capstone — Multi-DB Semantic Search Platform** | Full stack search platform with Go API and Python integration |
| `12` | **Qdrant Python Client Deep Dive** | Advanced Python client APIs, async patterns, production RAG capstone with LiteLLM + Phoenix |

## 🎯 Prerequisites

- **SQL**: `SELECT`, `INSERT`, `CREATE INDEX`, query tuning
- **Python/Go basics**: Reading and writing code in either language
- **Docker**: Running containers and composing services
- **Linear algebra**: Vectors, norms, dot products

## 🔗 Related Vault Modules

- [[06 - Large Language Models]] — Vector DBs are the memory layer for RAG pipelines
- [[13 - Go Engineering]] — Client implementations for Go/Python
- [[09 - MLOps y Produccion]] — Deploying vector pipelines in production
- [[10 - Cloud, Infra y Backend]] — Serving infrastructure, Docker, K8s

## 🛠️ Setup

```bash
pip install pgvector qdrant-client numpy sentence-transformers
docker pull qdrant/qdrant:latest
```

Review [[01 - Vector Search Fundamentals]] to start.
