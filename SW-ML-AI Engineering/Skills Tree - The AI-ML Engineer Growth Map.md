# 🌳 The AI/ML Engineer Growth Map

> **Alternative Organization.** Instead of portfolio sections, this document organizes every skill in the Learning vault as a **progressive growth map** — from fundamentals to specialization. Each tier builds on the previous, forming a complete career development tree for the AI/ML Engineer.


## 🌱 ROOTS — Foundations (What Everything Grows From)

> *Without strong roots, the tree collapses under production weight.*

### 🧮 Mathematical Thinking
| Topic | Course | Status |
|-------|--------|--------|
| Linear Algebra (matrices, eigenvalues, SVD) | 00 > 01 - Matematicas para ML | ✅ |
| Calculus & Gradients (backprop, optimization) | 00 > 01 > 02 | ✅ |
| Probability & Statistics (Bayes, distributions) | 00 > 01 > 03 | ✅ |
| Optimization (SGD, Adam, convex optimization) | 00 > 01 > 04 | ✅ |
| Information Theory (entropy, KL divergence) | 00 > 01 > 05 | ✅ |
| Advanced Statistics (causal inference, A/B tests) | 07 > 29 | ✅ |

### 🗄️ Data Structures & Algorithms
| Topic | Course | Status |
|-------|--------|--------|
| Algorithmic Complexity (Big O, NP-completeness) | 00 > 02 > 01 | ✅ |
| Arrays, Lists, Hash Tables | 00 > 02 > 02 | ✅ |
| Trees, Heaps, Priority Queues | 00 > 02 > 03 | ✅ |
| Graph Algorithms (BFS, DFS, Dijkstra) | 00 > 02 > 04 | ✅ |
| Searching & Sorting | 00 > 02 > 05 | ✅ |
| Dynamic Programming | 00 > 02 > 06 | ✅ |

### 🐍 Core Programming
| Topic | Course | Status |
|-------|--------|--------|
| Python (Basic → Advanced) | 09 - Advanced Python (01-05) | ✅ |
| Go (Fundamentals → ML Backend) | 13 - Go Engineering (01-07) | ✅ |
| Rust (Fundamentals → Agentic AI) | 14 - Rust Engineering (01-07) | ✅ |
| SQL (PostgreSQL Fundamentals → Advanced) | 11 - Curso SQL con PostgreSQL | ✅ |
| Git & Version Control | All courses (distributed) | ✅ |
| Markdown & Technical Documentation | 12 - Curso Markdown | ✅ |

### 🏗️ Software Engineering
| Topic | Course | Status |
|-------|--------|--------|
| System Design for ML | 06 > 32, [[SE extra]] > 02 | ✅ |
| Testing (unit, integration, property-based) | 14 > 02 > 05, 05 > 28, [[SE extra]] > 03 | ✅ |
| CI/CD Pipelines | 06 > 23, 05 > 29, [[SE extra]] > 04, 13 > 04 | ✅ |
| Docker & Containerization | 10 - Docker Profesional, 05 > 20 | ✅ |
| FastAPI (Python backends) | 06 > 24 > 01, 06 > 31, [[SE extra]] > 01 | ✅ |
| gRPC & Protocol Buffers | 06 > 24 > 02, 13 > 03 > 03 | ✅ |
| Message Queues & Streaming | 06 > 25 > 04 | ✅ |
| Design Patterns (builder, factory, observer) | 13 > 04, 14 > 02 > 03 | ✅ |

---

## 🪵 TRUNK — Core AI/ML Competencies

> *The thick trunk that supports everything above. Every AI/ML Engineer must master these.*

### 🔥 Deep Learning
| Topic | Course | Status |
|-------|--------|--------|
| Neural Networks Fundamentals | 01 > 03 > 01 | ✅ |
| CNNs & Modern Architectures (ResNet, EfficientNet) | 01 > 03 > 02 | ✅ |
| RNNs, LSTMs, GRUs | 01 > 03 > 03 | ✅ |
| Training Strategies (LR scheduling, regularization) | 01 > 03 > 04 | ✅ |
| Transfer Learning | 01 > 03 > 05 | ✅ |
| Vision Transformers (ViT, Swin, DINO) | 01 > 04 > 03 | ✅ |
| Object Detection (YOLO, DETR, Faster R-CNN) | 01 > 04 > 01 | ✅ |
| Segmentation (UNet, DeepLab, SAM) | 01 > 04 > 02 | ✅ |
| Multimodal Models (CLIP, LLaVA, ImageBind) | 01 > 05 | ✅ |
| OCR & Document Understanding | 01 > 04 > 04 | ✅ |
| CV Pipeline (preprocessing, augmentation, deployment) | 01 > 06 | ✅ |
| Reinforcement Learning (Q-Learning, PPO, DQN) | 01 > 07 | ✅ |
| Graph Neural Networks (GCN, GAT, Message Passing) | 01 > 08 | ✅ |
| Gorgonia (Deep Learning in Go) | 13 > 07 - Gorgonia | ✅ |
| Candle (ML in Rust — HuggingFace) | 14 > 04 > 02 | ✅ |
| TensorFlow / Keras (tf.keras, tf.data, TPU training) | 01 > 09 | ✅ |
| JAX (jit, vmap, grad, pmap, XLA, Flax, Optax) | 01 > 10 | ✅ |

### 🧠 Large Language Models
| Topic | Course | Status |
|-------|--------|--------|
| Transformer Architecture (attention, positional encoding) | 02 > 06 > 01 | ✅ |
| Tokenization (BPE, WordPiece, SentencePiece) | 02 > 06 > 02 | ✅ |
| Pretraining & Self-Supervised Learning | 02 > 06 > 03 | ✅ |
| Scaling Laws & Emergent Abilities | 02 > 06 > 04 | ✅ |
| Encoder-only (BERT, RoBERTa) | 04 > 16 > 01 | ✅ |
| Decoder-only (GPT, Llama, Mistral) | 04 > 16 > 02 | ✅ |
| Encoder-Decoder (T5, BART) | 04 > 16 > 03 | ✅ |
| Mixture of Experts (MoE — Mixtral, DeepSeek) | 02 > 10 > 01 | ✅ |
| State Space Models (Mamba, S4) | 02 > 10 > 02 | ✅ |
| Model Quantization (INT8, GPTQ, GGUF) | 02 > 09 > 02, 14 > 04 > 06 | ✅ |
| Model Distillation | 02 > 09 > 02 | ✅ |
| LLM Evaluation (MMLU, HumanEval, HELM) | 02 > 06 > 05 | ✅ |

### 🎯 LLM Engineering
| Topic | Course | Status |
|-------|--------|--------|
| Prompt Engineering (few-shot, CoT, ReAct) | 02 > 07 > 03 | ✅ |
| Fine-Tuning: SFT, LoRA, QLoRA, PEFT | 02 > 07, 02 > 11 | ✅ |
| RAG (Retrieval-Augmented Generation) | 02 > 07 > 04, 02 > 12 | ✅ |
| Decoding (beam search, nucleus, contrastive) | 02 > 08 > 01 | ✅ |
| Hallucination Mitigation | 02 > 08 > 03 | ✅ |
| LLM Security & Alignment (RLHF, DPO, jailbreaking) | 02 > 09 > 04, 02 > 15 | ✅ |
| Scalable LLM Serving (vLLM, continuous batching) | 02 > 09 > 03, 02 > 13 | ✅ |
| Unsloth (accelerated fine-tuning, quantization-aware) | 02 > 14 | ✅ |
| HuggingFace Transformers Deep Dive (library mastery) | 02 > 16 | ✅ |
| ColBERT (late interaction retrieval) | 02 > 17 | ✅ |
| SGLang (structured generation, RadixAttention) | 02 > 17 | ✅ |
| Inference-Time Scaling (Self-Refinement, Self-Consistency) | 02 > 17 | ✅ |
| Speculative Decoding 2.0 (Draft → Eagle → MTP) | 02 > 17 | ✅ |
| KV Cache Compression (MLA, H2O, StreamingLLM, SnapKV) | 02 > 17 | ✅ |
| FP8 Hybrid Precision (E4M3/E5M2, Transformer Engine) | 02 > 17 | ✅ |
| Disaggregated Serving and Edge Inference (NPU, ExecuTorch) | 02 > 17 | ✅ |
| GraphRAG (Knowledge Graph + RAG for LLMs) | 02 > 13, 04 > 17 > 03 | ✅ |
| RAGAS (RAG evaluation framework) | 02 > 12 | ✅ |
| Semantic Search & Hybrid Retrieval | 04 > 17 > 04 | ✅ |
| Knowledge Graphs for LLMs | 04 > 17 > 03 | ✅ |
| Multilingual LLMs | 04 > 17 > 01 | ✅ |

---

## 🌿 BRANCHES — Specializations

> *Choose your path. Each branch represents a career direction within AI/ML Engineering.*

### 🤖 Branch A: Agentic AI Engineer

*Build autonomous systems that reason, plan, and act.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | Agent Fundamentals (perception, reasoning, action loops) | 03 > 11 > 01 |
| 🌿 | Tool Use & Function Calling | 03 > 11 > 02 |
| 🌿 | Agent Memory Architectures | 03 > 11 > 03 |
| 🌿 | Planning & Reasoning (ReAct, ToT) | 03 > 11 > 04 |
| 🌿 | LangChain Ecosystem | 03 > 12 > 01 |
| 🌿 | LlamaIndex & Advanced RAG | 03 > 12 > 02 |
| 🌿 | CrewAI & AutoGen Orchestration | 03 > 12 > 03 |
| 🌿 | Multi-Agent Architectures | 03 > 13 |
| 🌿 | Autonomous Agents (AutoGPT-style) | 03 > 14 |
| 🌿 | Agent Reflection & Self-Improvement | 03 > 14 > 02 |
| 🌿 | MCP — Model Context Protocol (Anthropic) | 03 > 15, 14 > 06 > 01 |
| 🌿 | LangGraph + MCP Integration | 03 > 15 |
| 🌿 | Computer Use (agent GUI interaction) | 03 > 15 |
| 🌿 | Goose Agent (Block) | 14 > 06 > 02 |
| 🌿 | Autonomous OS Interaction | 14 > 06 > 03 |
| 🌿 | Agent Orchestration & Reasoning Loops | 14 > 06 > 04 |
| 🌿 | Agent Security & Sandboxing | 14 > 06 > 05, 02 > 15 |
| 🌿 | Production AI Agents in Rust | 14 > 06 > 06 |
| 🌿 | Agentic AI Projects (MCP Agent, Goose) | [[Rust projects]] > 05 |
| 🌿 | StayBot — Airbnb Agent (portfolio project) | Portfolio |
| 🌿 | Multi-Agent Research System (portfolio project) | Portfolio |

### ⚙️ Branch B: MLOps & Production Engineer

*Ship, scale, and maintain ML systems in production.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | Experiment Tracking (MLflow, W&B) | 05 > 18 > 01, 05 > 24 |
| 🌿 | Data Versioning (DVC) | 05 > 18 > 02 |
| 🌿 | Model Registry & Lifecycle | 05 > 18 > 03 |
| 🌿 | Feature Engineering & Feature Stores (Feast) | 05 > 19, 05 > 27 |
| 🌿 | Docker for ML (multi-stage, GPU images) | 05 > 20 > 01, 10 - Docker Profesional |
| 🌿 | Model Serving Patterns | 05 > 20 > 02 |
| 🌿 | TorchServe (PyTorch serving, MAR files, handlers, K8s) | 05 > 30 |
| 🌿 | Kubernetes for ML (K8s, KubeFlow) | 05 > 20 > 03 |
| 🌿 | A/B Testing & Canary Deployments | 05 > 20 > 04 |
| 🌿 | Data Drift & Concept Drift Monitoring | 05 > 21 > 01 |
| 🌿 | Evidently AI + Phoenix (drift detection, LLM obs) | 05 > 31 |
| 🌿 | Model Monitoring & Alerting | 05 > 21 > 02 |
| 🌿 | Auto-Retraining Pipelines | 05 > 21 > 03 |
| 🌿 | Explainability (SHAP, LIME) & Fairness | 05 > 21 > 04 |
| 🌿 | End-to-End ML (data → deploy → monitor) | 05 > 22 |
| 🌿 | Advanced MLOps (multi-cloud, GPUs, SLAs) | 05 > 23 |
| 🌿 | Weights & Biases (advanced experiment tracking) | 05 > 24 |
| 🌿 | MLOps Tooling Comparison (Kubeflow vs MLflow vs SageMaker) | 05 > 25 |
| 🌿 | ML Platform Engineering | 05 > 26 |
| 🌿 | Feast — Feature Store in Production | 05 > 27 |
| 🌿 | Testing ML Systems (data validation, model validation) | 05 > 28, [[SE extra]] > 03 |
| 🌿 | CI/CD for ML (GitHub Actions, Argo, model pipelines) | 05 > 29, [[SE extra]] > 04 |
| 🌿 | High-Throughput Inference (Go) | 13 > 06 > 02, 04 |
| 🌿 | High-Throughput Inference (Rust) | 14 > 04 > 04 |
| 🌿 | Go ML Serving Gateway | [[Go projects]] > 05 |
| 🌿 | Rust Inference Server (PyO3) | [[Rust projects]] > 02 |
| 🌿 | LLM Edge Gateway (portfolio project) | Portfolio |
| 🌿 | Automated LLM Evaluation Suite (portfolio project) | Portfolio |

### 🌐 Branch C: Cloud & Infrastructure Engineer

*Architect the platform that AI runs on.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | Cloud Computing (AWS, GCP) | 06 > 22 |
| 🌿 | Infrastructure as Code (Terraform) | 06 > 23 > 01 |
| 🌿 | IaC with Go (Pulumi) | 13 > 02 > 05 |
| 🌿 | Configuration Management (Ansible) | 06 > 23 > 02 |
| 🌿 | CI/CD & GitOps (ArgoCD, Flux) | 06 > 23 > 03 |
| 🌿 | Observability (Prometheus, Grafana, OpenTelemetry) | 06 > 23 > 04 |
| 🌿 | Databricks — Unified Analytics Platform | 06 > 26 |
| 🌿 | Apache Spark — Distributed Data Processing | 06 > 27 |
| 🌿 | BigQuery — Serverless Data Warehouse | 06 > 28 |
| 🌿 | Distributed ML Infrastructure (GPU clusters, scheduling) | 06 > 29 |
| 🌿 | WebSockets (real-time ML serving, bidirectional streaming) | 06 > 30 |
| 🌿 | FastAPI for ML (production serving patterns) | 06 > 31 |
| 🌿 | System Design for ML (interviews + architecture) | 06 > 32 |
| 🌿 | Go Cloud Native (Docker, K8s internals) | 13 > 02 |
| 🌿 | Service Discovery & Load Balancing | 13 > 02 > 04 |
| 🌿 | Distributed Tracing | 13 > 03 > 06 |
| 🌿 | PostgreSQL & NoSQL Databases | 06 > 25 |
| 🌿 | Redis & Caching Strategies | 06 > 25 > 03 |
| 🌿 | Message Queues & Event Streaming | 06 > 25 > 04 |
| 🌿 | WebAssembly & Edge AI | 14 > 05 |
| 🌿 | WASI & Serverless Edge | 14 > 05 > 03 |
| 🌿 | Edge AI Deployment | 14 > 05 > 06 |
| 🌿 | Go and eBPF for Observability | [[Go extra]] > 06 |
| 🌿 | Rust Cryptography & Security | [[Rust extra]] > 03 |
| 🌿 | Rust in the Linux Kernel | [[Rust extra]] > 05 |
| 🌿 | Blockchain (Solana, Polkadot) in Rust | [[Rust extra]] > 06 |

### 📊 Branch D: Data & Feature Engineer

*Build the data foundations that ML models consume.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | ETL Pipeline Design | 07 > 28 > 01 |
| 🌿 | Apache Spark (distributed processing) | 07 > 28 > 02, 06 > 27 |
| 🌿 | Data Warehousing (Snowflake, BigQuery) | 07 > 28 > 03, 06 > 28 |
| 🌿 | DuckDB (in-process OLAP, Python/Go integration) | 06 > 34 |
| 🌿 | Data Quality & Governance | 07 > 28 > 04 |
| 🌿 | Polars — Rust DataFrame (10-100x Pandas) | 14 > 03 > 01 |
| 🌿 | Apache Arrow & Zero-Copy Data | 14 > 03 > 02 |
| 🌿 | High-Performance ETL in Rust | 14 > 03 > 03 |
| 🌿 | Polars Internals (lazy, streaming, Arrow) | [[07 - Polars Internals and Advanced]] |
| 🌿 | DataFusion (Arrow query engine) | 14 > 03 > 03 |
| 🌿 | uv — Python Package Manager (Rust) | 14 > 03 > 04 |
| 🌿 | Ruff — Python Linter (Rust) | 14 > 03 > 05 |
| 🌿 | Data Processing CLIs (clap, indicatif) | 14 > 03 > 06 |
| 🌿 | Time Series Analysis | 07 > 29 > 04 |
| 🌿 | Causal Inference | 07 > 29 > 02 |
| 🌿 | Design of Experiments | 07 > 29 > 03 |
| 🌿 | Data Visualization & Storytelling | 07 > 27 |
| 🌿 | Vector Databases (Qdrant, pgvector, Milvus) | 10 > 33, 14 > 04 > 05 |
| 🌿 | Polars Data Pipeline Project | [[Rust projects]] > 01 |
| 🌿 | CLI Tool Projects (Go + Rust) | [[Go projects]] > 02, [[Rust projects]] |

### 🔬 Branch E: Research & Applied Science

*Push the boundaries — read, reproduce, and publish.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | How to Read ML Papers | 07 > 26 > 01 |
| 🌿 | Reproducibility & Experiment Design | 07 > 26 > 02 |
| 🌿 | Benchmarking & Competitions (Kaggle) | 07 > 26 > 03, 07 > 30 |
| 🌿 | Technical Writing & Papers | 07 > 26 > 04 |
| 🌿 | Kaggle Competitions (end-to-end, top-tier) | 07 > 30 |
| 🌿 | Paper Reproduction (full implementation) | 07 > 31 |
| 🌿 | Advanced ML Topics (SSL, GANs, Diffusion) | 07 > 32 |
| 🌿 | ONNX Runtime (cross-framework inference) | 14 > 04 > 03 |
| 🌿 | Model Quantization Research | 14 > 04 > 06 |
| 🌿 | Custom Models with Candle (Rust) | [[07 - Candle Advanced Patterns]] |
| 🌿 | Gorgonia — Computational Graphs & Autodiff | 13 > 07 - Gorgonia |
| 🌿 | LocalAI — Local LLM Server Architecture | 13 > 07 - LocalAI |
| 🌿 | WASM ML Pipelines (Browser + Edge) | 14 > 05 > 05 |
| 🌿 | Embedded ML (RTIC, TinyML, microcontrollers) | [[Rust extra]] > 04 |
| 🌿 | Async Rust Internals (Tokio, custom runtimes) | [[Rust extra]] > 02 |
| 🌿 | Rust Memory Model Deep Dive | [[Rust extra]] > 01 |
| 🌿 | Paper Reproduction (end-to-end) | 07 > 31, [[ML projects]] > 07 |
| 🌿 | End-to-End ML Project | 05 > 22, [[ML projects]] > 02 |

### 🏢 Branch F: Product & Business AI

*Bridge the gap between ML research and business value.*

| Level | Skill | Course |
|-------|-------|--------|
| 🌿 | AI Product Design & Strategy | 08 > 30 > 01 |
| 🌿 | ML Roadmap & Strategy | 08 > 30 > 02 |
| 🌿 | AI Ethics & Responsibility | 08 > 30 > 03 |
| 🌿 | Legal & Compliance (GDPR, EU AI Act) | 08 > 30 > 04 |
| 🌿 | ROI of ML Projects | 08 > 31 > 01 |
| 🌿 | Business Metrics vs Technical Metrics | 08 > 31 > 02 |
| 🌿 | ML Cost Management & Budgeting | 08 > 31 > 03 |
| 🌿 | Stakeholder Communication | 08 > 31 > 04 |
| 🌿 | Open Source Contribution | 08 > 32 > 01, [[Rust projects]] > 04 |
| 🌿 | Building Technical Community | 08 > 32 > 02 |
| 🌿 | Publishing Libraries & Papers | 08 > 32 > 03 |
| 🌿 | Career Growth & Professional Development | 08 > 32 > 04 |
| 🌿 | Contributing to Ruff/Polars (real OSS) | [[Rust projects]] > 04 |

---

## 🍃 LEAVES — Transversal Skills (Apply to Every Branch)

| Skill | Course | Why Universal |
|-------|--------|---------------|
| **Technical English** | 15 > 01 | All papers, docs, and interviews are in English |
| **Technical Communication & Storytelling** | 15 > 02 | Explain complex ideas to non-technical stakeholders |
| **Technical Leadership** | 15 > 03 | Lead projects, mentor juniors, drive architecture |
| **System Design for ML** | 06 > 32, [[SE extra]] > 02 | Design interviews at FAANG+ companies |
| **Testing in ML Systems** | 05 > 28, [[SE extra]] > 03 | Production reliability |
| **CI/CD for ML** | 05 > 29, [[SE extra]] > 04 | Automate everything |
| **FastAPI for ML** | 06 > 31, [[SE extra]] > 01 | The standard Python ML serving framework |
| **Rust for ML Infrastructure** | 14 > 04, [[SE extra]] > 05 | When Python isn't fast enough |
| **Go for ML Backend** | 13 > 06 | High-concurrency serving and infrastructure |
| **Wails — Desktop Apps with Go** | 13 > 07 - Wails | Ship ML tools as native apps |
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
| 🌟 Senior Impact | 08 (Producto/Negocio/OSS), 15 (Transversal Skills), 07 (Research) | 18 |
| 🧠 Research & Build | 01-04 (DL, CV, LLMs, NLP, Agents) | 47 |
| ⚙️ Production | 05 (MLOps), 06 (Cloud/Backend), 13 > 03-06, 14 > 02-04 | 34 |
| 🌐 Cloud + Infra | 06, 13 > 02, 14 > 05, Go extra, Rust extra | 22 |
| 📐 Engineering Foundations | 00, 09 (Python), 10 (Docker), 11 (SQL), 12 (Markdown), 13 > 01, 14 > 01 | 28 |
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
| **ONNX Runtime** | Industry standard for cross-framework deployment. You have it in Go and Rust — extremely rare combo | 13 > 06, 14 > 04 > 03 |
| **Transfer Learning & Fine-Tuning** | 90% of production ML is fine-tuning, not training from scratch | 01 > 03 > 05 |
| **Model Quantization (INT8, GPTQ, GGUF)** | Reduces GPU costs 4-8x. Every company serving models needs this | 02 > 09 > 02, 14 > 04 > 06 |
| **Multimodal Models (CLIP, ImageBind)** | The frontier: text+image+audio. All new models are multimodal | 01 > 05 |
| **Vision Transformers (ViT, DINOv2)** | Replaced CNNs. If you do CV in 2026, you use ViT | 01 > 04 > 03 |
| **Object Detection (YOLO, DETR)** | Most common industrial CV use case | 01 > 04 > 01 |
| **TensorFlow / Keras for Training** | Enterprise and Google Cloud teams still run on TF. Dual-framework literacy = staff engineer signal | 01 > 09 |

---

### 🧠 LLMs & NLP *(Portfolio: 3 skills → Target: 10)*

> **Highest-impact section right now.** This is what hiring managers search for first.

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **RAG — Retrieval-Augmented Generation** | The #1 enterprise LLM pattern. Every job posting mentions it | 02 > 07 > 04, 02 > 12 |
| **Fine-Tuning: LoRA, QLoRA, PEFT** | Difference between "API user" and "ML Engineer" | 02 > 07, 02 > 11 |
| **Prompt Engineering (CoT, ReAct, Few-shot)** | Foundation for agents and production LLM systems | 02 > 07 > 03 |
| **LLM Evaluation (HELM, MMLU, LLM-as-Judge)** | Without evaluation there is no production. Companies desperate for this | 02 > 06 > 05 |
| **High-Throughput LLM Serving (vLLM, continuous batching)** | Reduce inference costs 10x. Infrastructure skill with direct ROI | 02 > 09 > 03, 02 > 13, 14 > 04 > 04 |
| **HuggingFace Transformers Deep Dive** | The ecosystem behind 90% of open-source LLMs. Library literacy separates users from engineers | 02 > 16 |
| **Mixture of Experts (MoE)** | Architecture behind Mistral, DeepSeek, Mixtral. Dominant in 2026 | 02 > 10 > 01 |
| **Hallucination Mitigation** | Problem #1 of LLMs in production. Mastering this differentiates you | 02 > 08 > 03 |
| **ColBERT (late interaction retrieval)** | Token-level retrieval outperforms dense embeddings. Only method with cross-encoder quality at bi-encoder speed | 02 > 17 |
| **SGLang / Speculative Decoding** | Cut inference costs 2-5x with RadixAttention and draft-model verification. Directly reduces cloud GPU bills | 02 > 17 |

---

### 🎭 AI Agents *(Portfolio: 6 skills → Target: 11)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **MCP — Model Context Protocol** | Anthropic's open standard for agent-tool communication. Early adopter = competitive advantage | 03 > 15, 14 > 06 > 01 |
| **Multi-Agent Systems & Orchestration** | Companies are migrating from single-agent to swarms | 03 > 13, 14 > 06 > 04 |
| **Agent Memory Architectures** | Memory is the difference between a useful agent and a dumb one | 03 > 11 > 03 |
| **Function Calling / Tool Use** | 80% of agent value comes from integrating external APIs | 03 > 11 > 02 |
| **Agent Security & Sandboxing** | Nobody mentions it. If you master it, you stand out immediately | 14 > 06 > 05, 02 > 15 |

---

### 📊 Data Engineering *(Portfolio: 8 skills → Target: 14)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **Rust** | "I build data pipelines 100x faster than Python" — that line lands jobs | 14 - Rust Engineering (01-07) |
| **Polars (Rust/Python)** | Successor to Pandas. Companies migrating seek people who already know it | 14 > 03 > 01, [[07 - Polars Internals]] |
| **Apache Arrow** | Columnar format standard. Snowflake, Databricks, InfluxDB use it | 14 > 03 > 02 |
| **Vector Databases (Qdrant, pgvector, Milvus)** | No vectors = no RAG. Indispensable 2025-2026 skill | 10 > 33, 14 > 04 > 05 |
| **PostgreSQL Advanced** | Window functions, recursive CTEs, indexes, pgvector — not just "I know SQL" | 11 - Curso SQL con PostgreSQL, 06 > 25 > 01 |
| **High-Performance ETL (Rust/Go)** | Spark is fine. Building ETL in Rust/Go is staff-engineer level | 14 > 03 > 03, 06 |

---

### ⚙️ MLOps & Deploy *(Portfolio: 8 skills → Target: 13)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **Model Monitoring (Data Drift, Concept Drift)** | No monitoring = no real production. Every team needs this | 05 > 21 > 01-02 |
| **gRPC & Protocol Buffers** | ML service communication. Faster than REST | 06 > 24 > 02, 13 > 03 > 03 |
| **DVC — Data Versioning** | Git for datasets. Reproducibility = requirement in serious teams | 05 > 18 > 02 |
| **Feature Stores (Feast, Tecton)** | Online/offline feature serving. Enterprise-scale teams need this | 05 > 27 |
| **Docker Multi-stage (images < 20 MB)** | Container optimization. Senior detail, not junior | 10 - Docker Profesional, 14 > 02 > 06 |

---

### ☁️ Cloud & Infra *(Portfolio: 4 skills → Target: 8)*

| Add to Portfolio | Why It Gets You Hired | Learning Course |
|------------------|----------------------|-----------------|
| **Terraform / Pulumi (IaC)** | Without IaC you don't scale. SRE skill the modern ML Engineer needs | 06 > 23 > 01, 13 > 02 > 05 |
| **WebAssembly & Edge Computing** | Inference at the edge, Cloudflare Workers, Fastly. Hot topic | 14 > 05 (complete) |
| **Kubernetes for ML (GPU scheduling, tolerations)** | Generic K8s anyone can list. GPU workload K8s is valuable niche | 13 > 02 > 02, 05 > 20 > 03 |
| **Observability (Prometheus + Grafana + OpenTelemetry)** | Tracing distributed ML pipelines requires this | 06 > 23 > 04, 13 > 03 > 06 |

---

### 🆕 NEW SECTION 1: Software Engineering for ML

> *Does NOT exist on your current portfolio. This is what separates a "notebook engineer" from a real ML Engineer.*

| Skill | Why It Gets You Hired | Learning Course |
|-------|----------------------|-----------------|
| **System Design for ML** | Asked in EVERY ML Engineer interview (FAANG and startups) | 06 > 32, [[SE extra]] > 02 |
| **Testing — Unit, Integration, Property-Based** | ML code without tests = technical debt. Extremely rare in candidates | 14 > 02 > 05, 05 > 28, [[SE extra]] > 03 |
| **CI/CD for ML Pipelines** | Automate training, evaluation, and deployment end-to-end | 05 > 29, [[SE extra]] > 04, 14 > 04 |
| **Go for ML Backend** | Fiber, Gin, gRPC. High-concurrency serving without the Python GIL | 13 > 03, 13 > 06 |
| **Rust for ML Infrastructure** | PyO3, Candle, ONNX. Critical performance without sacrificing safety | 14 > 04, [[SE extra]] > 05 |

---

### 🆕 NEW SECTION 2: Research & Communication

> *Replaces your current "Core Competencies" (generic soft traits → concrete vetted skills).*

| Skill | Why It Gets You Hired | Learning Course |
|-------|----------------------|-----------------|
| **Paper Reading & Reproduction** | Reading a NeurIPS paper and implementing it = signal of intellectual autonomy | 07 > 26, 07 > 31 |
| **Technical Writing (Papers, ADRs, Design Docs)** | Documenting architecture = senior/staff skill. Scarce in ML | 07 > 26 > 04, 15 > 02 |
| **Data Storytelling & Visualization** | Explain models to non-technical stakeholders. Difference between approval and rejection | 07 > 27 |
| **Open Source Contribution** | Commits to Ruff, Polars, Candle = immediate social proof | 08 > 32, [[Rust projects]] > 04 |
| **AI Ethics, Fairness & EU AI Act** | Regulation is coming. Knowing compliance makes you "future-proof" | 08 > 30 > 03-04 |


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
