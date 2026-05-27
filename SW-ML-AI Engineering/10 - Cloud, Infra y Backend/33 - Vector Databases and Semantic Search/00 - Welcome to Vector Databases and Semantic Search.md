# 🏷️ Welcome to Vector Databases and Semantic Search

Welcome to the **Vector Databases and Semantic Search** course module. This course covers the theory, implementation, and production deployment of vector search systems that power modern AI applications.

## 📚 Course Roadmap

This course contains the following notes:

1. [[00 - Welcome to Vector Databases and Semantic Search]] — Course overview and roadmap (this note)
2. [[01 - Vector Search Fundamentals]] — Embeddings, distance metrics, ANN vs KNN, curse of dimensionality
3. [[02 - Indexing Algorithms Deep Dive]] — IVF, HNSW, PQ, OPQ, DiskANN, ScaNN, and trade-offs
4. [[03 - pgvector I - Core Operations and Indexing]] — PostgreSQL extension basics, CRUD, indexing, distance operators
5. [[04 - pgvector II - Production and Hybrid Search]] — Partitioning, hybrid search, pooling, monitoring, backups
6. [[05 - Qdrant I - Architecture and Collections]] — Rust-based engine, collections, points, payload indexing, APIs
7. [[06 - Qdrant II - Distributed and Cloud Deployment]] — Cluster mode, shards, replicas, cloud, LangChain integration

## 🎯 Prerequisites

Before starting this course, you should be comfortable with:

- **SQL**: Writing `SELECT`, `INSERT`, `CREATE INDEX`, and basic query tuning ([[12 - SQL Mastery]])
- **Python/Go basics**: Reading and writing code in either language ([[13 - Go Engineering]], [[14 - Python for AI]])
- **Docker**: Running containers and composing services ([[15 - Docker and Kubernetes]])
- **Linear algebra basics**: Vectors, norms, dot products ([[03 - Mathematics for ML]])

## 🔗 Related Vault Modules

- [[06 - Large Language Models]] — Vector DBs are the memory layer for RAG pipelines
- [[13 - Go Engineering]] — Qdrant is written in Rust; clients exist in Go and Python
- [[14 - Rust Engineering]] — Understanding Qdrant internals and high-performance systems
- [[18 - MLOps and Model Serving]] — Deploying vector pipelines in production
- [[22 - NLP and Transformers]] — Where embeddings are generated
- [[12 - SQL Mastery]] — PostgreSQL fundamentals required for pgvector modules

## 🛠️ Setup Checklist

- [ ] Install PostgreSQL 15+ with `pgvector` extension
- [ ] Install Docker Desktop or Docker Engine
- [ ] Pull `qdrant/qdrant:latest` image
- [ ] Install Python dependencies: `pip install pgvector qdrant-client numpy sentence-transformers`
- [ ] Clone the course companion repo (if available) for starter code and datasets
- [ ] Verify installations by running the compression scripts at the end of each note
- [ ] Configure your Obsidian graph view to show backlinks for easy navigation between modules
- [ ] Review the [[01 - Vector Search Fundamentals]] learning objectives before starting

## 📖 How to Use These Notes

Each note is designed to be read in sequence, but you can also jump to specific topics based on your current project needs. Core notes (01–06) follow a consistent structure: theoretical foundations first, mental models and visual diagrams second, code and syntax third, and production considerations last. Always run the **Compression Code** scripts to validate your understanding before moving to the next module.

If you are new to vector search, start with [[01 - Vector Search Fundamentals]] and progress linearly. If you are evaluating databases for a production RAG system, skim [[01]] and [[02]], then focus on [[03]]–[[06]] for implementation details. Each note ends with a Documented Project section that you can use as a portfolio piece or team onboarding exercise.

We recommend maintaining a local sandbox database while reading. Execute the SQL and Python snippets as you encounter them — active experimentation solidifies abstract concepts far better than passive reading. Keep a running glossary of new terms in your Obsidian vault and link them back to these notes.

## 🤝 Contributing and Feedback

Found an error or want to suggest an improvement? Update the relevant note and cross-reference any changes in related modules. Keep the Obsidian link syntax (`[[...]]`) consistent so the knowledge graph remains navigable. If you extend a project into a working prototype, link back to the original note so future readers can find your implementation.

---

> **Next**: Start with [[01 - Vector Search Fundamentals]] to build the theoretical foundation for everything that follows.
