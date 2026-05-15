# 🌳 The AI/ML Engineer Growth Map

> **Alternative Organization.** Instead of portfolio sections, this document organizes every skill in the Learning vault as a **progressive growth map** — from fundamentals to specialization. Each tier builds on the previous, forming a complete career development tree for the AI/ML Engineer.

---

```
                                     ┌──────────────────────┐
                                     │   THE COMPLETE       │
                                     │   AI/ML ENGINEER     │
                                     │                      │
                                     │   🌟 Senior Impact   │
                                     │   🧠 Research+Build  │
                                     │   ⚙️ Production      │
                                     │   🌐 Cloud+Infra     │
                                     │   📐 Engineering     │
                                     └──────────────────────┘
```

---

## 🌱 ROOTS — Foundations (What Everything Grows From)

> *Without strong roots, the tree collapses under production weight.*

### 🧮 Mathematical Thinking
| Topic | Course | Status |
|-------|--------|--------|
| Linear Algebra (matrices, eigenvalues, SVD) | M00 > 01 - Matematicas para ML | ✅ |
| Calculus & Gradients (backprop, optimization) | M00 > 01 > 02 | ✅ |
| Probability & Statistics (Bayes, distributions) | M00 > 01 > 03 | ✅ |
| Optimization (SGD, Adam, convex optimization) | M00 > 01 > 04 | ✅ |
| Information Theory (entropy, KL divergence) | M00 > 01 > 05 | ✅ |
| Advanced Statistics (causal inference, A/B tests) | M07 > 29 | ✅ |

### 🗄️ Data Structures & Algorithms
| Topic | Course | Status |
|-------|--------|--------|
| Algorithmic Complexity (Big O, NP-completeness) | M00 > 02 > 01 | ✅ |
| Arrays, Lists, Hash Tables | M00 > 02 > 02 | ✅ |
| Trees, Heaps, Priority Queues | M00 > 02 > 03 | ✅ |
| Graph Algorithms (BFS, DFS, Dijkstra) | M00 > 02 > 04 | ✅ |
| Searching & Sorting | M00 > 02 > 05 | ✅ |
| Dynamic Programming | M00 > 02 > 06 | ✅ |

### 🐍 Core Programming
| Topic | Course | Status |
|-------|--------|--------|
| Python (Basic → Advanced) | [[Advanced Python]] (01-05) | ✅ |
| Go (Fundamentals → ML Backend) | [[Go Engineering]] (01-06) | ✅ |
| Rust (Fundamentals → Agentic AI) | [[Rust Engineering]] (01-06) | ✅ |
| SQL (PostgreSQL Fundamentals → Advanced) | [[Curso SQL con PostgreSQL]] | ✅ |
| Git & Version Control | All courses (distributed) | ✅ |
| Markdown & Technical Documentation | Curso Markdown | ✅ |

### 🏗️ Software Engineering
| Topic | Course | Status |
|-------|--------|--------|
| System Design for ML | [[SE extra]] > 02 - System Design for ML | ✅ |
| Testing (unit, integration, property-based) | [[Rust Engineering/02]] > 05, [[SE extra]] > 03 | ✅ |
| CI/CD Pipelines | M06 > 23, [[SE extra]] > 04, [[Go Engineering/04]] | ✅ |
| Docker & Containerization | [[01 - Docker Profesional]], M05 > 20 | ✅ |
| FastAPI (Python backends) | M06 > 24 > 01, [[SE extra]] > 01 | ✅ |
| gRPC & Protocol Buffers | M06 > 24 > 02, [[Go Engineering/03]] > 03 | ✅ |
| Message Queues & Streaming | M06 > 25 > 04 | ✅ |
| Design Patterns (builder, factory, observer) | [[Go Engineering/04]], [[Rust Engineering/02]] > 03 | ✅ |

---

## 🪵 TRUNK — Core AI/ML Competencies

> *The thick trunk that supports everything above. Every AI/ML Engineer must master these.*

### 🔥 Deep Learning
| Topic | Course | Status |
|-------|--------|--------|
| Neural Networks Fundamentals | M01 > 03 > 01 | ✅ |
| CNNs & Modern Architectures (ResNet, EfficientNet) | M01 > 03 > 02 | ✅ |
| RNNs, LSTMs, GRUs | M01 > 03 > 03 | ✅ |
| Training Strategies (LR scheduling, regularization) | M01 > 03 > 04 | ✅ |
| Transfer Learning | M01 > 03 > 05 | ✅ |
| Vision Transformers (ViT, Swin, DINO) | M01 > 04 > 03 | ✅ |
| Object Detection (YOLO, DETR, Faster R-CNN) | M01 > 04 > 01 | ✅ |
| Segmentation (UNet, DeepLab, SAM) | M01 > 04 > 02 | ✅ |
| Multimodal Models (CLIP, LLaVA, ImageBind) | M01 > 05 | ✅ |
| OCR & Document Understanding | M01 > 04 > 04 | ✅ |
| Gorgonia (Deep Learning in Go) | [[Go Engineering/07 - Gorgonia]] | ✅ |
| Candle (ML in Rust — HuggingFace) | [[Rust Engineering/04]] > 02 | ✅ |

### 🧠 Large Language Models
| Topic | Course | Status |
|-------|--------|--------|
| Transformer Architecture (attention, positional encoding) | M02 > 06 > 01 | ✅ |
| Tokenization (BPE, WordPiece, SentencePiece) | M02 > 06 > 02 | ✅ |
| Pretraining & Self-Supervised Learning | M02 > 06 > 03 | ✅ |
| Scaling Laws & Emergent Abilities | M02 > 06 > 04 | ✅ |
| Encoder-only (BERT, RoBERTa) | M04 > 16 > 01 | ✅ |
| Decoder-only (GPT, Llama, Mistral) | M04 > 16 > 02 | ✅ |
| Encoder-Decoder (T5, BART) | M04 > 16 > 03 | ✅ |
| Mixture of Experts (MoE — Mixtral, DeepSeek) | M02 > 10 > 01 | ✅ |
| State Space Models (Mamba, S4) | M02 > 10 > 02 | ✅ |
| Model Quantization (INT8, GPTQ, GGUF) | M02 > 09 > 02, [[Rust Engineering/04]] > 06 | ✅ |
| Model Distillation | M02 > 09 > 02 | ✅ |
| LLM Evaluation (MMLU, HumanEval, HELM) | M02 > 06 > 05 | ✅ |

### 🎯 LLM Engineering
| Topic | Course | Status |
|-------|--------|--------|
| Prompt Engineering (few-shot, CoT, ReAct) | M02 > 07 > 03 | ✅ |
| Fine-Tuning: SFT, LoRA, QLoRA, PEFT | M02 > 07, M09 extra > 03 | ✅ |
| RAG (Retrieval-Augmented Generation) | M02 > 07 > 04, M09 extra > 04 | ✅ |
| Decoding (beam search, nucleus, contrastive) | M02 > 08 > 01 | ✅ |
| Hallucination Mitigation | M02 > 08 > 03 | ✅ |
| LLM Security & Alignment (RLHF, DPO) | M02 > 09 > 04 | ✅ |
| Scalable LLM Serving (vLLM, continuous batching) | M02 > 09 > 03 | ✅ |
| Semantic Search & Hybrid Retrieval | M04 > 17 > 04 | ✅ |
| Knowledge Graphs for LLMs | M04 > 17 > 03 | ✅ |
| Multilingual LLMs | M04 > 17 > 01 | ✅ |

---

## 🌿 BRANCHES — Specializations

> *Choose your path. Each branch represents a career direction within AI/ML Engineering.*

### 🤖 Branch A: Agentic AI Engineer

*Build autonomous systems that reason, plan, and act.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | Agent Fundamentals (perception, reasoning, action loops) | M03 > 11 > 01 |
| 🌿 | Tool Use & Function Calling | M03 > 11 > 02 |
| 🌿 | Agent Memory Architectures | M03 > 11 > 03 |
| 🌿 | Planning & Reasoning (ReAct, ToT) | M03 > 11 > 04 |
| 🌿 | LangChain Ecosystem | M03 > 12 > 01 |
| 🌿 | LlamaIndex & Advanced RAG | M03 > 12 > 02 |
| 🌿 | CrewAI & AutoGen Orchestration | M03 > 12 > 03 |
| 🌿 | Multi-Agent Architectures | M03 > 13 |
| 🌿 | Autonomous Agents (AutoGPT-style) | M03 > 14 |
| 🌿 | Agent Reflection & Self-Improvement | M03 > 14 > 02 |
| 🌿 | MCP — Model Context Protocol (Anthropic) | [[Rust Engineering/06]] > 01 |
| 🌿 | Goose Agent (Block) | [[Rust Engineering/06]] > 02 |
| 🌿 | Autonomous OS Interaction | [[Rust Engineering/06]] > 03 |
| 🌿 | Agent Orchestration & Reasoning Loops | [[Rust Engineering/06]] > 04 |
| 🌿 | Agent Security & Sandboxing | [[Rust Engineering/06]] > 05 |
| 🌿 | Production AI Agents in Rust | [[Rust Engineering/06]] > 06 |
| 🌿 | Agentic AI Projects (MCP Agent, Goose) | [[Rust projects]] > 05 |
| 🌿 | StayBot — Airbnb Agent (portfolio project) | Portfolio |
| 🌿 | Multi-Agent Research System (portfolio project) | Portfolio |

### ⚙️ Branch B: MLOps & Production Engineer

*Ship, scale, and maintain ML systems in production.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | Experiment Tracking (MLflow, W&B) | M05 > 18 > 01 |
| 🌿 | Data Versioning (DVC) | M05 > 18 > 02 |
| 🌿 | Model Registry & Lifecycle | M05 > 18 > 03 |
| 🌿 | Feature Engineering & Feature Stores (Feast) | M05 > 19 |
| 🌿 | Docker for ML (multi-stage, GPU images) | M05 > 20 > 01, [[01 - Docker Profesional]] |
| 🌿 | Model Serving Patterns | M05 > 20 > 02 |
| 🌿 | Kubernetes for ML (K8s, KubeFlow) | M05 > 20 > 03 |
| 🌿 | A/B Testing & Canary Deployments | M05 > 20 > 04 |
| 🌿 | Data Drift & Concept Drift Monitoring | M05 > 21 > 01 |
| 🌿 | Model Monitoring & Alerting | M05 > 21 > 02 |
| 🌿 | Auto-Retraining Pipelines | M05 > 21 > 03 |
| 🌿 | Explainability (SHAP, LIME) & Fairness | M05 > 21 > 04 |
| 🌿 | Testing ML Systems | [[SE extra]] > 03 |
| 🌿 | CI/CD for ML | [[SE extra]] > 04 |
| 🌿 | High-Throughput Inference (Go) | [[Go Engineering/06]] > 02, 04 |
| 🌿 | High-Throughput Inference (Rust) | [[Rust Engineering/04]] > 04 |
| 🌿 | Go ML Serving Gateway | [[Go projects]] > 05 |
| 🌿 | Rust Inference Server (PyO3) | [[Rust projects]] > 02 |
| 🌿 | LLM Edge Gateway (portfolio project) | Portfolio |
| 🌿 | Automated LLM Evaluation Suite (portfolio project) | Portfolio |

### 🌐 Branch C: Cloud & Infrastructure Engineer

*Architect the platform that AI runs on.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | Cloud Computing (AWS, GCP) | M06 > 22 |
| 🌿 | Infrastructure as Code (Terraform) | M06 > 23 > 01 |
| 🌿 | IaC with Go (Pulumi) | [[Go Engineering/02]] > 05 |
| 🌿 | Configuration Management (Ansible) | M06 > 23 > 02 |
| 🌿 | CI/CD & GitOps (ArgoCD, Flux) | M06 > 23 > 03 |
| 🌿 | Observability (Prometheus, Grafana, OpenTelemetry) | M06 > 23 > 04 |
| 🌿 | Go Cloud Native (Docker, K8s internals) | [[Go Engineering/02]] |
| 🌿 | Service Discovery & Load Balancing | [[Go Engineering/02]] > 04 |
| 🌿 | Distributed Tracing | [[Go Engineering/03]] > 06 |
| 🌿 | PostgreSQL & NoSQL Databases | M06 > 25 |
| 🌿 | Redis & Caching Strategies | M06 > 25 > 03 |
| 🌿 | Message Queues & Event Streaming | M06 > 25 > 04 |
| 🌿 | WebAssembly & Edge AI | [[Rust Engineering/05]] |
| 🌿 | WASI & Serverless Edge | [[Rust Engineering/05]] > 03 |
| 🌿 | Edge AI Deployment | [[Rust Engineering/05]] > 06 |
| 🌿 | Go and eBPF for Observability | [[Go extra]] > 06 |
| 🌿 | Rust Cryptography & Security | [[Rust extra]] > 03 |
| 🌿 | Rust in the Linux Kernel | [[Rust extra]] > 05 |
| 🌿 | Blockchain (Solana, Polkadot) in Rust | [[Rust extra]] > 06 |

### 📊 Branch D: Data & Feature Engineer

*Build the data foundations that ML models consume.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | ETL Pipeline Design | M07 > 28 > 01 |
| 🌿 | Apache Spark (distributed processing) | M07 > 28 > 02 |
| 🌿 | Data Warehousing (Snowflake, BigQuery) | M07 > 28 > 03 |
| 🌿 | Data Quality & Governance | M07 > 28 > 04 |
| 🌿 | Polars — Rust DataFrame (10-100x Pandas) | [[Rust Engineering/03]] > 01 |
| 🌿 | Apache Arrow & Zero-Copy Data | [[Rust Engineering/03]] > 02 |
| 🌿 | High-Performance ETL in Rust | [[Rust Engineering/03]] > 03 |
| 🌿 | Polars Internals (lazy, streaming, Arrow) | [[07 - Polars Internals and Advanced]] |
| 🌿 | DataFusion (Arrow query engine) | [[Rust Engineering/03]] > 03 |
| 🌿 | uv — Python Package Manager (Rust) | [[Rust Engineering/03]] > 04 |
| 🌿 | Ruff — Python Linter (Rust) | [[Rust Engineering/03]] > 05 |
| 🌿 | Data Processing CLIs (clap, indicatif) | [[Rust Engineering/03]] > 06 |
| 🌿 | Time Series Analysis | M07 > 29 > 04 |
| 🌿 | Causal Inference | M07 > 29 > 02 |
| 🌿 | Design of Experiments | M07 > 29 > 03 |
| 🌿 | Data Visualization & Storytelling | M07 > 27 |
| 🌿 | Vector Databases (Qdrant, pgvector) | [[Rust Engineering/04]] > 05 |
| 🌿 | Polars Data Pipeline Project | [[Rust projects]] > 01 |
| 🌿 | CLI Tool Projects (Go + Rust) | [[Go projects]] > 02, [[Rust projects]] |

### 🔬 Branch E: Research & Applied Science

*Push the boundaries — read, reproduce, and publish.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | How to Read ML Papers | M07 > 26 > 01 |
| 🌿 | Reproducibility & Experiment Design | M07 > 26 > 02 |
| 🌿 | Benchmarking & Competitions (Kaggle) | M07 > 26 > 03, M09 extra > 01 |
| 🌿 | Technical Writing & Papers | M07 > 26 > 04 |
| 🌿 | ONNX Runtime (cross-framework inference) | [[Rust Engineering/04]] > 03 |
| 🌿 | Model Quantization Research | [[Rust Engineering/04]] > 06 |
| 🌿 | Custom Models with Candle (Rust) | [[07 - Candle Advanced Patterns]] |
| 🌿 | Gorgonia — Computational Graphs & Autodiff | [[Go Engineering/07 - Gorgonia]] |
| 🌿 | LocalAI — Local LLM Server Architecture | [[Go Engineering/07 - LocalAI]] |
| 🌿 | WASM ML Pipelines (Browser + Edge) | [[Rust Engineering/05]] > 05 |
| 🌿 | Embedded ML (RTIC, TinyML, microcontrollers) | [[Rust extra]] > 04 |
| 🌿 | Async Rust Internals (Tokio, custom runtimes) | [[Rust extra]] > 02 |
| 🌿 | Rust Memory Model Deep Dive | [[Rust extra]] > 01 |
| 🌿 | Paper Reproduction (end-to-end) | M09 extra > 07, [[ML projects]] > 07 |
| 🌿 | End-to-End ML Project | M09 extra > 02, [[ML projects]] > 02 |

### 🏢 Branch F: Product & Business AI

*Bridge the gap between ML research and business value.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | AI Product Design & Strategy | M08 > 30 > 01 |
| 🌿 | ML Roadmap & Strategy | M08 > 30 > 02 |
| 🌿 | AI Ethics & Responsibility | M08 > 30 > 03 |
| 🌿 | Legal & Compliance (GDPR, EU AI Act) | M08 > 30 > 04 |
| 🌿 | ROI of ML Projects | M08 > 31 > 01 |
| 🌿 | Business Metrics vs Technical Metrics | M08 > 31 > 02 |
| 🌿 | ML Cost Management & Budgeting | M08 > 31 > 03 |
| 🌿 | Stakeholder Communication | M08 > 31 > 04 |
| 🌿 | Open Source Contribution | M08 > 32 > 01, [[Rust projects]] > 04 |
| 🌿 | Building Technical Community | M08 > 32 > 02 |
| 🌿 | Publishing Libraries & Papers | M08 > 32 > 03 |
| 🌿 | Career Growth & Professional Development | M08 > 32 > 04 |
| 🌿 | Contributing to Ruff/Polars (real OSS) | [[Rust projects]] > 04 |

---

## 🍃 LEAVES — Transversal Skills (Apply to Every Branch)

| Skill | Course | Why Universal |
|-------|--------|---------------|
| **Technical English** | [[Extra Skills/01]] | All papers, docs, and interviews are in English |
| **Technical Communication & Storytelling** | [[Extra Skills/02]] | Explain complex ideas to non-technical stakeholders |
| **Technical Leadership** | [[Extra Skills/03]] | Lead projects, mentor juniors, drive architecture |
| **System Design for ML** | [[SE extra]] > 02 | Design interviews at FAANG+ companies |
| **Testing in ML Systems** | [[SE extra]] > 03 | Production reliability |
| **CI/CD for ML** | [[SE extra]] > 04 | Automate everything |
| **FastAPI for ML** | [[SE extra]] > 01 | The standard Python ML serving framework |
| **Rust for ML Infrastructure** | [[SE extra]] > 05 | When Python isn't fast enough |
| **Go for ML Backend** | [[Go Engineering/06]] | High-concurrency serving and infrastructure |
| **Wails — Desktop Apps with Go** | [[Go Engineering/07 - Wails]] | Ship ML tools as native apps |
| **WASM ML in Browser** | [[Rust projects]] > 03 | Client-side ML, privacy-preserving inference |

---

## 🌟 FRUITS — Portfolio-Ready Projects

> *The evidence that proves you can do everything above.*

| Project | Technology | Vault Location | Portfolio Status |
|---------|-----------|----------------|-----------------|
| LLM Edge Gateway | Go, Fiber, Redis, HuggingFace | [[Go projects]], Portfolio | ✅ Live |
| Sudoku Together | JS, WebSockets, Node.js, Redis | Portfolio | ✅ Live |
| Automated LLM Evaluation Suite | Python, SageMaker, Vertex AI, Gemma | Portfolio | ✅ Live |
| Multi-Agent Research System | LangGraph, Gemma, Tavily API | Portfolio | ✅ Live |
| StayBot — Airbnb Agent | LangGraph, CrewAI, FastAPI, Supabase | Portfolio | ✅ Live |
| Go ML Serving Gateway | Go, gRPC, ONNX, K8s | [[Go projects]] > 05 | 🚧 Ready to build |
| Local RAG with Go | Go, Vector DB, Ollama | [[Go projects]] > 04 | 🚧 Ready to build |
| Polars Data Pipeline | Rust, Polars, Parquet, Arrow | [[Rust projects]] > 01 | 🚧 Ready to build |
| Rust Inference Server | Rust, PyO3, Candle, ONNX | [[Rust projects]] > 02 | 🚧 Ready to build |
| WASM ML in Browser | Rust, WASM, Candle, WebGPU | [[Rust projects]] > 03 | 🚧 Ready to build |
| MCP Agent in Rust | Rust, MCP, Goose | [[Rust projects]] > 05 | 🚧 Ready to build |
| End-to-End ML (Kaggle) | Python, PyTorch, MLflow, AWS | [[ML projects]] > 01-02 | 🚧 Ready to build |
| Fine-Tuned LLM Service | Python, LoRA, FastAPI, Docker | [[ML projects]] > 03 | 🚧 Ready to build |
| Production RAG System | Python, LangChain, Qdrant, GCP | [[ML projects]] > 04 | 🚧 Ready to build |
| CV Pipeline | Python, PyTorch, ONNX, Triton | [[ML projects]] > 05 | 🚧 Ready to build |
| Advanced MLOps System | MLflow, DVC, K8s, Prometheus | [[ML projects]] > 06 | 🚧 Ready to build |
| Paper Reproduction | Variable (reproduce a NeurIPS paper) | [[ML projects]] > 07 | 🚧 Ready to build |

---

## 📐 The Complete Skill Matrix

```
                         ┌──────────────────────────────────────┐
  🌟 SENIOR              │ Product Strategy · Ethics · Research  │
                         │ Technical Leadership · OSS · Papers  │
                         ├──────────────────────────────────────┤
  🧠 RESEARCH+BUILD      │ LLM Fine-Tuning · RAG · Agents · MoE  │
                         │ CV · NLP · Multimodal · Quantization │
                         │ Mamba · GNN · RL · Custom Models     │
                         ├──────────────────────────────────────┤
  ⚙️ PRODUCTION           │ MLOps · K8s · CI/CD · Monitoring     │
                         │ Feature Stores · A/B Testing · gRPC  │
                         │ Docker · FastAPI · Go/Rust Serving   │
                         ├──────────────────────────────────────┤
  🌐 CLOUD+INFRA         │ AWS · GCP · Terraform · WASM · Edge  │
                         │ PostgreSQL · Redis · Kafka · NATS    │
                         │ Linux · eBPF · Cross-compilation     │
                         ├──────────────────────────────────────┤
  📐 ENGINEERING         │ Python · Go · Rust · SQL · Git       │
                         │ Algorithms · Data Structures · Math  │
                         │ System Design · Testing · CI/CD      │
                         └──────────────────────────────────────┘
```

| Layer | Courses in Learning | Skills Count |
|-------|-------------------|:-----------:|
| 🌟 Senior Impact | M08 (Product/Negocio/OSS), Extra Skills, M07 (Research) | 18 |
| 🧠 Research & Build | M01-M04 (DL, CV, LLMs, NLP, Agents), M09 extra | 47 |
| ⚙️ Production | M05 (MLOps), M06 (Cloud/Backend), Go 03-06, Rust 02-04 | 34 |
| 🌐 Cloud + Infra | M06, Go 02, Rust 05, Go extra, Rust extra | 22 |
| 📐 Engineering Foundations | M00, Advanced Python, Docker, SE extra, Go 01, Rust 01 | 28 |
| **TOTAL** | **All courses in Learning vault** | **149** |

---

## 🗺️ Recommended Growth Paths

### Path 1: The Generalist (First Job → Junior ML Engineer)
```
Roots → Trunk → Sample 2-3 branches → Build 5 projects → Apply
Timeline: 6-12 months full-time study
```

### Path 2: The MLOps Specialist (ML Infrastructure Engineer)
```
Roots → Trunk → Branch B (MLOps) + Branch C (Cloud) → 8 projects → Apply
Timeline: Build on existing SWE experience
```

### Path 3: The Agentic AI Engineer (Hot market 2025-2026)
```
Roots → Trunk → Branch A (Agents) + LLMs deep → 6 agent projects → Apply
Key: Master MCP, Goose, LangGraph, AutoGen — highest demand right now
```

### Path 4: The Research-to-Production Engineer (Startup-ready)
```
Roots → Trunk → Branch E (Research) + Branch B (MLOps) → Paper repro + prod → Apply
Key: Can read papers AND ship them
```

---

---

## 🎯 Portfolio Hiring Analysis — Skills to Add

> **Current state:** 34 skills on portfolio. **Learning vault:** 139 skills mastered.  
> **Objective:** Select the highest-impact skills that make recruiters say "we need this person."

---

### 🤖 Machine Learning *(Portfolio: 4 skills → Target: 10)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **ONNX Runtime** | Industry standard for cross-framework deployment. You have it in Go and Rust — extremely rare combo | [[06 - Go for ML Backend]], [[04 - Rust for ML and AI]] > 03 |
| **Transfer Learning & Fine-Tuning** | 90% of production ML is fine-tuning, not training from scratch | M01 > 03 > 05 |
| **Model Quantization (INT8, GPTQ, GGUF)** | Reduces GPU costs 4-8x. Every company serving models needs this | M02 > 09 > 02, [[04 - Rust for ML and AI]] > 06 |
| **Multimodal Models (CLIP, ImageBind)** | The frontier: text+image+audio. All new models are multimodal | M01 > 05 |
| **Vision Transformers (ViT, DINOv2)** | Replaced CNNs. If you do CV in 2026, you use ViT | M01 > 04 > 03 |
| **Object Detection (YOLO, DETR)** | Most common industrial CV use case | M01 > 04 > 01 |

---

### 🧠 LLMs & NLP *(Portfolio: 3 skills → Target: 10)*

> **Highest-impact section right now.** This is what hiring managers search for first.

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **RAG — Retrieval-Augmented Generation** | The #1 enterprise LLM pattern. Every job posting mentions it | M02 > 07 > 04, M09 extra > 04 |
| **Fine-Tuning: LoRA, QLoRA, PEFT** | Difference between "API user" and "ML Engineer" | M02 > 07, M09 extra > 03 |
| **Prompt Engineering (CoT, ReAct, Few-shot)** | Foundation for agents and production LLM systems | M02 > 07 > 03 |
| **LLM Evaluation (HELM, MMLU, LLM-as-Judge)** | Without evaluation there is no production. Companies desperate for this | M02 > 06 > 05 |
| **High-Throughput LLM Serving (vLLM, continuous batching)** | Reduce inference costs 10x. Infrastructure skill with direct ROI | M02 > 09 > 03, [[04 - Rust for ML and AI]] > 04 |
| **Mixture of Experts (MoE)** | Architecture behind Mistral, DeepSeek, Mixtral. Dominant in 2026 | M02 > 10 > 01 |
| **Hallucination Mitigation** | Problem #1 of LLMs in production. Mastering this differentiates you | M02 > 08 > 03 |

---

### 🎭 AI Agents *(Portfolio: 6 skills → Target: 11)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **MCP — Model Context Protocol** | Anthropic's open standard for agent-tool communication. Early adopter = competitive advantage | [[06 - Agentic AI with Rust]] > 01 |
| **Multi-Agent Systems & Orchestration** | Companies are migrating from single-agent to swarms | M03 > 13, [[06 - Agentic AI with Rust]] > 04 |
| **Agent Memory Architectures** | Memory is the difference between a useful agent and a dumb one | M03 > 11 > 03 |
| **Function Calling / Tool Use** | 80% of agent value comes from integrating external APIs | M03 > 11 > 02 |
| **Agent Security & Sandboxing** | Nobody mentions it. If you master it, you stand out immediately | [[06 - Agentic AI with Rust]] > 05 |

---

### 📊 Data Engineering *(Portfolio: 8 skills → Target: 14)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **Rust** | "I build data pipelines 100x faster than Python" — that line lands jobs | [[Rust Engineering]] (complete 01-07) |
| **Polars (Rust/Python)** | Successor to Pandas. Companies migrating seek people who already know it | [[03 - Rust for Data Engineering]] > 01, [[07 - Polars Internals]] |
| **Apache Arrow** | Columnar format standard. Snowflake, Databricks, InfluxDB use it | [[03 - Rust for Data Engineering]] > 02 |
| **Vector Databases (Qdrant, pgvector)** | No vectors = no RAG. Indispensable 2025-2026 skill | [[04 - Rust for ML and AI]] > 05 |
| **PostgreSQL Advanced** | Window functions, recursive CTEs, indexes, pgvector — not just "I know SQL" | [[Curso SQL con PostgreSQL]], M06 > 25 > 01 |
| **High-Performance ETL (Rust/Go)** | Spark is fine. Building ETL in Rust/Go is staff-engineer level | [[03 - Rust for Data Engineering]] > 03, 06 |

---

### ⚙️ MLOps & Deploy *(Portfolio: 8 skills → Target: 13)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **Model Monitoring (Data Drift, Concept Drift)** | No monitoring = no real production. Every team needs this | M05 > 21 > 01-02 |
| **gRPC & Protocol Buffers** | ML service communication. Faster than REST | M06 > 24 > 02, [[03 - Microservices with Go]] > 03 |
| **DVC — Data Versioning** | Git for datasets. Reproducibility = requirement in serious teams | M05 > 18 > 02 |
| **Feature Stores (Feast, Tecton)** | Online/offline feature serving. Enterprise-scale teams need this | M05 > 19 > 02 |
| **Docker Multi-stage (images < 20 MB)** | Container optimization. Senior detail, not junior | [[01 - Docker Profesional]], [[02 - Advanced Rust]] > 06 |

---

### ☁️ Cloud & Infra *(Portfolio: 4 skills → Target: 8)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **Terraform / Pulumi (IaC)** | Without IaC you don't scale. SRE skill the modern ML Engineer needs | M06 > 23 > 01, [[02 - Go for Cloud Native]] > 05 |
| **WebAssembly & Edge Computing** | Inference at the edge, Cloudflare Workers, Fastly. Hot topic | [[05 - WebAssembly and Edge AI]] (complete) |
| **Kubernetes for ML (GPU scheduling, tolerations)** | Generic K8s anyone can list. GPU workload K8s is valuable niche | [[02 - Go for Cloud Native]] > 02, M05 > 20 > 03 |
| **Observability (Prometheus + Grafana + OpenTelemetry)** | Tracing distributed ML pipelines requires this | M06 > 23 > 04, [[03 - Microservices with Go]] > 06 |

---

### 🆕 NEW SECTION 1: Software Engineering for ML

> *Does NOT exist on your current portfolio. This is what separates a "notebook engineer" from a real ML Engineer.*

| Skill | Why It Gets You Hired | Learning Course |
|-------|----------------------|-----------------|
| **System Design for ML** | Asked in EVERY ML Engineer interview (FAANG and startups) | [[SE extra]] > 02 |
| **Testing — Unit, Integration, Property-Based** | ML code without tests = technical debt. Extremely rare in candidates | [[02 - Advanced Rust]] > 05, [[SE extra]] > 03 |
| **CI/CD for ML Pipelines** | Automate training, evaluation, and deployment end-to-end | [[SE extra]] > 04, [[04 - DevSecOps and CLI Tools]] |
| **Go for ML Backend** | Fiber, Gin, gRPC. High-concurrency serving without the Python GIL | [[Go Engineering/03]], [[Go Engineering/06]] |
| **Rust for ML Infrastructure** | PyO3, Candle, ONNX. Critical performance without sacrificing safety | [[04 - Rust for ML and AI]], [[SE extra]] > 05 |

---

### 🆕 NEW SECTION 2: Research & Communication

> *Replaces your current "Core Competencies" (generic soft traits → concrete vetted skills).*

| Skill | Why It Gets You Hired | Learning Course |
|-------|----------------------|-----------------|
| **Paper Reading & Reproduction** | Reading a NeurIPS paper and implementing it = signal of intellectual autonomy | M07 > 26, M09 extra > 07 |
| **Technical Writing (Papers, ADRs, Design Docs)** | Documenting architecture = senior/staff skill. Scarce in ML | M07 > 26 > 04, [[Extra Skills/02]] |
| **Data Storytelling & Visualization** | Explain models to non-technical stakeholders. Difference between approval and rejection | M07 > 27 |
| **Open Source Contribution** | Commits to Ruff, Polars, Candle = immediate social proof | M08 > 32, [[Rust projects]] > 04 |
| **AI Ethics, Fairness & EU AI Act** | Regulation is coming. Knowing compliance makes you "future-proof" | M08 > 30 > 03-04 |

---


## ⚡ Top 7 — If You Can Only Add These

> *Highest hiring impact. Start here.*

| # | Skill | Section | Why #1 Priority |
|---|-------|---------|-----------------|
| 1 | **RAG** | LLMs & NLP | Every enterprise LLM job requires it |
| 2 | **Fine-Tuning (LoRA, QLoRA)** | LLMs & NLP | Separates API users from ML Engineers |
| 3 | **Model Monitoring (Drift)** | MLOps | Production = monitoring. No exceptions |
| 4 | **Rust** | Data Engineering | Differentiator no other candidate has |
| 5 | **MCP (Model Context Protocol)** | AI Agents | Emerging standard — be the early expert |
| 6 | **System Design for ML** | SWE for ML 🆕 | Every interview. Every senior role |
| 7 | **ONNX Runtime** | Machine Learning | Cross-framework deployment standard |

---

*This hiring analysis maps all 139 Learning vault skills against the portfolio at [white-portfolio-ia-ml-engineer.netlify.app](https://white-portfolio-ia-ml-engineer.netlify.app/). Generated from 459 notes, ~95,000 lines of compressed ML/AI knowledge.*
