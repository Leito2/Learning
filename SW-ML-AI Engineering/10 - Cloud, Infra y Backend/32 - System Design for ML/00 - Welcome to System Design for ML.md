# 🏗️ 00 - Welcome to System Design for ML

## 🎯 Learning Objectives

- Articulate why ML system design is the #1 differentiator in FAANG+ ML engineer interviews and how it diverges from traditional software system design
- Map the five pillars of the course: CAP theorem for ML, caching/storage architectures, load balancing/sharding, back-of-envelope estimation, and end-to-end interview walkthroughs
- Identify prerequisites and internal vault connections that form the foundation for ML system design reasoning
- Recognize that ML systems are designed for training, inference, feature serving, and data pipelines — not CRUD APIs

## Introduction

System design interviews dominate FAANG+ hiring for ML engineer roles. The format is deceptively simple: the interviewer says "design a recommendation system" or "design a real-time fraud detector," and you have 45 minutes to sketch the architecture, estimate scale, and defend every tradeoff. What trips up most candidates is that ML system design is fundamentally different from software system design. You are not designing a Twitter clone or a URL shortener. You are designing for model training pipelines, inference latency budgets, feature freshness SLAs, GPU allocation, embedding caches, and data distribution — concepts that never appear in the classic system design primer.

The etymology is instructive. *System* comes from Greek *systēma* (organized whole, from *syn-* "together" + *histanai* "to set up"). *Design* comes from Latin *designare* (to mark out, from *de-* "out" + *signare* "to mark"). ML system design is the art of marking out an organized whole where every component — feature store, model server, cache layer, load balancer — fits together under mathematical constraints of latency, throughput, and cost. A software engineer designs for correctness under concurrency. An ML engineer designs for correctness under staleness, GPU saturation, and embedding dimension explosion.

This course bridges the gap between knowing ML algorithms and demonstrating senior engineering judgment in an interview. You will learn to classify every component by its CAP profile, estimate GPU requirements with back-of-envelope math, design multi-tiered caching strategies, and walk through complete system designs the way FAANG interviewers expect. The patterns here connect deeply to the FastAPI serving layer ([[../../31 - FastAPI for ML/00 - Welcome to FastAPI for ML|FastAPI for ML]]), cloud infrastructure provisioning ([[../../23 - Infrastructure as Code/00 - Welcome to Infrastructure as Code|IaC]]), and distributed training ([[../../29 - Distributed ML Infrastructure/00 - Welcome|Distributed ML]]).

---

## 📋 Course Map

| # | Note | Description | Lines |
|---|------|-------------|-------|
| 01 | CAP Theorem and Consistency Models in ML Workloads | CAP trilemma with ML-specific tradeoffs: fraud (CP), recsys (AP), training (CP/AP split). Consistency model deep-dive with LaTeX, feature store classification, DoorDash case study | ~420 |
| 02 | Caching, CDNs, and Storage Architectures for ML | Embedding caches, Redis semantic cache, prediction caches, model artifact CDN, tiered storage economics, KV cache reuse (SGLang RadixAttention), LLM Edge Gateway case study | ~420 |
| 03 | Load Balancing, Sharding, and Scaling ML Systems | Consistent hashing for KV cache locality, GPU-aware load balancing, feature store sharding, model parallelism as load distribution, custom-metric HPA for GPU services, Notion case study | ~400 |
| 04 | Back-of-Envelope Estimation — Throughput, Latency, Storage, and Cost | Numbers to memorize (HBM bandwidth, network latencies, model sizes). LaTeX estimation templates for storage, throughput, latency, and cost. Three graded practice problems with full solutions. Python utility library | ~380 |
| 05 | Interview Walkthrough — Design Systems End-to-End | Full walkthroughs: YouTube-style recommendation system (500M DAU) and real-time fraud detection (10K TPS). Requirements → back-of-envelope → API design → architecture → scaling plan → tradeoffs discussion | ~420 |

## 🧱 Prerequisites

| Topic | Required Proficiency | Related Vault Note |
|-------|---------------------|--------------------|
| Cloud computing fundamentals | Regions, VMs, object storage, networking | [[../../22 - Cloud Computing/00 - Bienvenida|Cloud Computing]] |
| Backend API design | REST, FastAPI, async patterns, middleware | [[../../31 - FastAPI for ML/00 - Welcome to FastAPI for ML|FastAPI for ML]] |
| ML deployment patterns | Containerization, serving, model artifacts | [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/00 - Deployment y Serving|Deployment and Serving]] |
| Vector databases and embeddings | ANN search, embedding storage, approximate retrieval | [[../../33 - Vector Databases and Semantic Search/00 - Welcome|Vector Databases]] |
| Basic Python | Data structures, async/await, type hints | [[../../24 - Backend para ML/00 - Backend para ML|Backend for ML]] |

---

## 📦 What You Will Build

By the end of this course, you will produce a **system design portfolio** consisting of:

- A CAP classification matrix for ML components, mapping each component to its consistency and availability requirements
- A multi-tiered caching strategy document with Redis semantic cache, prediction cache, and feature cache designs (extending the LLM Edge Gateway pattern)
- A consistent-hashing-based load balancing design with GPU-aware routing and custom-metric autoscaling
- A back-of-envelope estimation worksheet with LaTeX derivations for storage, throughput, latency, and cost
- Two complete end-to-end system design documents: YouTube-style recommendation system and real-time fraud detection system

These artifacts collectively demonstrate the senior engineering judgment that FAANG+ interviewers evaluate: you don't just know algorithms — you know how to deploy them at scale under real-world constraints.

---

## 🔗 Vault Connections

This course sits at the intersection of several knowledge domains:

- **Cloud & Infrastructure**: [[../../23 - Infrastructure as Code/00 - Welcome to Infrastructure as Code|IaC]] — provisioning GPU clusters, networking, and storage programmatically
- **Serving & APIs**: [[../../31 - FastAPI for ML/00 - Welcome to FastAPI for ML|FastAPI]] — async inference servers, streaming, middleware patterns
- **Real-Time Systems**: [[../../30 - WebSockets and Real-Time ML/00 - WebSockets and Real-Time ML|WebSockets]] — bidirectional streaming, low-latency prediction endpoints
- **Distributed Training**: [[../../29 - Distributed ML Infrastructure/00 - Welcome|Distributed ML]] — model parallelism, data parallelism, pipeline parallelism
- **Feature Stores**: [[../../25 - Bases de Datos y Message Queues/20 - Feature Stores y ML Metadata/20 - Feature Stores y ML Metadata|Feature Stores]] — online/offline feature serving, consistency guarantees
- **Vector Search**: [[../../33 - Vector Databases and Semantic Search/00 - Welcome|Vector Databases]] — ANN indices, embedding retrieval, semantic caching
- **MLOps**: [[../../../05 - MLOps y Produccion/20 - Deployment y Serving/00 - Deployment y Serving|Deployment]] — CI/CD for ML, model versioning, A/B testing infrastructure

---

## References

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. — The canonical system design text. Chapters 5 (Replication), 6 (Partitioning), and 9 (Consistency and Consensus) are essential.
- Donnemartin. (2024). *System Design Primer*. https://github.com/donnemartin/system-design-primer — Open-source system design resource with flashcard-style concepts.
- AWS Well-Architected Framework. (2024). *Machine Learning Lens*. https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/ — Production ML architecture patterns.
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly. — ML-specific system design patterns, feature stores, and serving architectures.
- [[../../31 - FastAPI for ML/00 - Welcome to FastAPI for ML|FastAPI for ML]]
- [[../../23 - Infrastructure as Code/00 - Welcome to Infrastructure as Code|IaC]]
- [[../../29 - Distributed ML Infrastructure/00 - Welcome|Distributed ML Infrastructure]]
