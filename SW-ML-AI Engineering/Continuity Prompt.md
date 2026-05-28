# Continuity Prompt for Next Session

Copy and paste this entire file into a new session to continue the project without losing context.

---

## ⚠️ CRITICAL: CONTEXT MANAGEMENT

This vault is **~600+ notes**. Bulk content creation in the **main thread** WILL saturate context.

### Mandatory Subagent Workflow

| Task Type | Rule |
|-----------|------|
| Writing courses (3+ notes) | `task` tool, **max 2 subagents parallel**, **max 7 notes each** |
| Research / exploring codebase | `task` tool with `subagent_type: explore` |
| Single-note edit | Main thread (Write/Edit) — OK for 1-2 files |
| Post-subagent verification | `bash` — check file counts and line counts |

**Never:** write 5+ notes in main thread, launch >2 subagents, skip verification.

**Filesystem is the single source of truth.** Subagents may silently complete. Verify with `ls`/`wc -l` immediately after launch.

---

## ⚠️ DEEP FORMAT SPECIFICATION — DEFAULT STANDARD FOR ALL NOTES

> **This is the DEFAULT and MANDATORY format for every note in the vault.** No exceptions.
>
> **Core philosophy:** What matters most is **deep conceptual and functional theory** — WHY something exists, HOW it works under the hood, and WHAT problems it solves. Every note must build complete technical and practical mastery of its topic. Theory always precedes code. Code demonstrates understanding, it does not replace it.

Applies to **all new notes** and **all expansions**. No exceptions.

### Required Sections

```markdown
# 🏷️ Title (with emoji)

## 🎯 Learning Objectives
(Bullet list of what the reader will master)

## Introduction
(2-3 paragraphs. Deep context: WHY this matters for ML/AI engineering.
Connect to other vault modules via [[...]] links.)

---

## Module X: <Concept Name>

### X.1 Theoretical Foundation 🧠
(3-4 paragraphs: WHY the concept exists, historical context, design motivation.
Show the problem BEFORE the solution.)

### X.2 Mental Model 📐
(ASCII diagram using ┌─│└├┘┤┬┴┼ — at least 3 per note)

### X.3 Syntax and Semantics 📝
(Exact syntax with line-by-line WHY comments, not just WHAT)
```go / rust / python
// code with inline comments explaining WHY
```

### X.4 Visual Representation 🖼️
(Mermaid diagram — at least 2 per note, varied types)
```mermaid
<diagram>
```
(Wikimedia image URL: ![Alt](https://upload.wikimedia.org/...))

### X.5 Application in ML/AI Systems 🤖
(Specific real-world case: "Real case: <company>" or usage table)

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|

### X.6 Common Pitfalls ⚠️
⚠️ <Warning with root cause explanation>
💡 <Tip with mnemonic>

### X.7 Knowledge Check ❓
(2-3 questions or mini-exercises)

---

## 📦 Compression Code
(Complete, production-ready script summarizing ALL concepts)

## 🎯 Documented Project
### Description, Functional Requirements, Main Components, Success Metrics

## 🎯 Key Takeaways
(5-7 bullet points)

## References
```

### Line Targets

| Note Type | Lines |
|-----------|:-----:|
| Course welcome (00) | 60-90 |
| Core course note | 400-600 |
| Quick-reference | 200-350 |
| Tool comparison | 200-400 |

### Key Rules
- **Theory BEFORE code** in every section. Conceptual and functional theory is the HIGHEST priority.
- Every note must deliver COMPLETE technical and practical mastery: theoretical foundation → mental model → syntax/semantics → visual representation → real-world application → pitfalls → knowledge check → compression code → project.
- **English** for all new content; **Spanish** only for existing M00-M08 modules.
- Code blocks MUST have language tag: ` ```python `, ` ```go `, ` ```rust `, ` ```sql `, ` ```bash `, ` ```yaml `, ` ```json `.
- Tables MUST use aligned columns.
- Mermaid MUST use ` ```mermaid ` wrapper.
- Use `[[...]]` for internal Obsidian links.
- File names: `## - Descriptive Name.md`. **Never use `/`** (Windows restriction).
- One H1 per note only.
- **THIS DEEP FORMAT IS THE DEFAULT STANDARD.** All notes must follow it. All subagents must be instructed to follow it.

---

## Project Context

We are building an **Obsidian vault** inside `/home/white/Learning`, a **Git repo** connected to `https://github.com/Leito2/Learning.git`. The vault contains compressed markdown courses for an **AI/ML Engineer** learning path (~600 notes).

**Current Goal:** Maintain and deepen the vault as a job-ready knowledge base. The user is seeking their first ML/AI Engineer role.

**Gap-fill initiative:** ✅ COMPLETE — 22 original gaps filled (50 notes across 12 courses).

**Current total: ~603 notes** — all redistributed into logical numbered modules (00-16).

## User Profile (Leandro Cataño Cardeño)

| Field | Detail |
|-------|--------|
| **Role** | AI & ML Engineer — LLMs, RAG Systems, Agentic AI |
| **Location** | Medellín, Colombia |
| **Languages** | English B2/Advanced, Spanish Native |
| **Portfolio** | https://white-portfolio-ia-ml-engineer.netlify.app/ |
| **GitHub** | https://github.com/Leito2 |
| **Resume keywords** | Python, Go, SQL, Redis, Linux, Git, Docker, Kubernetes, PyTorch, LangGraph, Hugging Face, FastAPI, Fiber, AWS, GCP |

### Portfolio Projects (4 core)
1. **LLM Edge Gateway** — Go/Fiber + Redis semantic caching + Gemma 4 failover. 30-40% API cost reduction, <10ms cached responses, 99.9% uptime.
2. **Automated LLM Evaluation Suite** — Python asyncio + Gemma 4 as Golden Judge + AWS SageMaker + GCP Vertex AI. 90% audit automation, real-time semantic drift detection.
3. **Multi-Agent Research System** — LangGraph cyclic agents (Research → Fact-Audit → Synthesis) + Gemma 4 function calling + Tavily API. 85%+ accuracy on multi-hop tasks.
4. **StayBot — Airbnb Property Agent** — LangGraph + CrewAI + FastAPI + Supabase. Multi-agent property management automation.

### Vault Skills Coverage
**Strong (have courses + portfolio projects):** Python, Go, LangGraph, FastAPI, Fiber, Docker, Redis, AWS, GCP, PyTorch, RAG
**Present (have course notes):** Rust, Kubernetes, HuggingFace, TensorFlow, MLOps, CI/CD, SQL
**Gaps (need new courses):** See "High-Value Technologies" section below

---

## Vault Structure (Condensed)

All courses distributed into logical numbered modules (00-16). Markdown, SQL, Docker moved before Advanced Python as tooling foundations. Bun Runtime relocated to `Extra/`. New module 16 added.

```
SW-ML-AI Engineering/
├── Welcome.md
├── 00 - Indice Maestro de Cursos.md
├── Skills Tree - AI-ML Engineer Growth Map.md
│
├── 00 - Curso Markdown/                (5 notes, Spanish)
│
├── 01 - Curso SQL con PostgreSQL/      (8 notes, Spanish)
│
├── 02 - Docker Profesional/            (7 notes, Spanish)
│
├── 03 - Advanced Python/               (62 notes, Spanish)
│   ├── 01 - Python Basico
│   ├── 02 - Python Intermedio
│   ├── 03 - Python Avanzado
│   ├── 04 - Librerias Basicas de Python
│   └── 05 - Librerias Especificas
│
├── 04 - Engineering Fundamentals/      (23 notes, Spanish)
│   ├── 00 - Python Avanzado para ML
│   ├── 01 - Matematicas para ML
│   └── 02 - Estructuras de Datos
│
├── 05 - Deep Learning y CV/            (38 notes: 18 Spanish + 20 English)
│   ├── 03 - DL con PyTorch         (7)
│   ├── 04 - CV Avanzada            (6)
│   ├── 05 - Multimodal AI          (5)
│   ├── 06 - CV Pipeline            (1 EN)
│   ├── 07 - Reinforcement Learning (7 EN)
│   ├── 08 - Graph Neural Networks  (5 EN)
│   └── 09 - Deep Learning with TF  (7 EN)
│
├── 06 - Large Language Models/         (63 notes: 30 Spanish + 33 English)
│   ├── 06 - Fundamentos de LLMs   (7)
│   ├── 07 - Fine-Tuning            (6)
│   ├── 08 - Gen de Texto           (6)
│   ├── 09 - LLMs en Produccion     (6)
│   ├── 10 - Arq Avanzadas/MoE      (5)
│   ├── 11 - Fine-Tuning LLMs       (1 EN)
│   ├── 12 - Production RAG         (1 EN)
│   ├── 13 - vLLM and Advanced RAG  (6 EN)
│   ├── 14 - Unsloth                (5 EN)
│   ├── 15 - LLM Security           (5 EN)
│   ├── 16 - HuggingFace Transformers Deep Dive (10 EN)
│   └── 17 - ColBERT, SGLang and Advanced Inference (7 EN)
│
├── 07 - AI Agents/                     (28 notes: 22 Spanish + 6 English)
│   ├── 11 - Fundamentos Agentes    (6)
│   ├── 12 - Frameworks             (6)
│   ├── 13 - Multi-Agente           (5)
│   ├── 14 - Agentes Autonomos      (5)
│   └── 15 - MCP and Agentic Prot   (6 EN)
│
├── 08 - NLP Avanzado/                  (17 notes, Spanish)
│
├── 09 - MLOps/                         (50 notes: 25 Spanish + 25 English)
│   ├── 18 - Experiment Tracking    (7)
│   ├── 19 - Feature Engineering    (6)
│   ├── 20 - Deployment             (6)
│   ├── 21 - Monitoreo              (6)
│   ├── 22 - End-to-End ML          (1 EN)
│   ├── 23 - Advanced MLOps         (1 EN)
│   ├── 24 - Weights and Biases     (5 EN)
│   ├── 25 - Tooling Comparison     (1 EN)
│   ├── 26 - ML Platform Eng        (6 EN)
│   ├── 27 - Feast                  (5 EN)
│   ├── 28 - Testing in ML          (1 EN)
│   └── 29 - CI-CD for ML           (1 EN)
│
├── 10 - Cloud, Infra/                  (52 notes: 24 Spanish + 28 English)
│   ├── 22 - Cloud Computing        (6)
│   ├── 23 - Infra como Codigo      (6)
│   ├── 24 - Backend para ML        (6)
│   ├── 25 - BD y Message Queues    (6)
│   ├── 26 - Databricks for ML      (5 EN)
│   ├── 27 - Apache Spark           (6 EN)
│   ├── 28 - BigQuery               (3 EN)
│   ├── 29 - Distributed ML Infra   (7 EN)
│   ├── 30 - WebSockets             (5 EN)
│   ├── 31 - FastAPI for ML         (6 EN)
│   ├── 32 - System Design for ML   (1 EN)
│   └── 33 - Vector Databases and Semantic Search (12 EN)
│
├── 11 - Research/                      (33 notes: 24 Spanish + 9 English)
│   ├── 26 - Metodologia            (6)
│   ├── 27 - Visualizacion          (6)
│   ├── 28 - ETL                    (6)
│   ├── 29 - Estadistica Avanzada   (6)
│   ├── 30 - Kaggle Competitions    (1 EN)
│   ├── 31 - Paper Reproduction     (1 EN)
│   └── 32 - Advanced ML Topics     (7 EN)
│
├── 12 - Producto y Negocio/            (18 notes, Spanish)
│
├── 13 - Go Engineering/                (73 notes, English)
├── 14 - Rust Engineering/              (73 notes, English)
├── 15 - Transversal Skills/            (4 notes, English)
│
├── 16 - SDD and Harness Engineering/   (11 notes, English)
│   ├── 00 - Welcome to SDD and Harness Engineering
│   ├── 01 - Harness Engineering Fundamentals
│   ├── 02 - SDD: The Specification-First Workflow
│   ├── 03 - Agent Loop Architecture: Building the Core
│   ├── 04 - Multi-Agent Orchestration and Roles
│   ├── 05 - External Memory and Context Management
│   ├── 06 - The 20 Harnesses: Phase Control and Contracts
│   ├── 07 - Tools and Provider Abstraction
│   ├── 08 - File Structures and Repository Harnesses
│   ├── 09 - Verification and Quality Gates
│   └── 10 - End-to-End Workflow and Technical Summary
│
├── Extra/                              # Bun Runtime + cross-cutting topics
└── projects/                           (15+ guides)
```

---

## Language Policy

- **Modules 00-02 (Markdown, SQL, Docker), 03 (Advanced Python), 04-12:** Spanish. **Do NOT modify** unless requested.
- **New courses integrated into 05-12 (English-labeled sub-modules), Go Engineering (13), Rust Engineering (14), Transversal Skills (15), SDD and Harness Engineering (16), and projects:** **English**.
- **Welcome/indices:** May contain both languages to bridge old and new.

---

## Git Commands

```bash
# Linux paths (current environment)
git -C /home/white/Learning status
git -C /home/white/Learning add -A
git -C /home/white/Learning commit -m "message"
git -C /home/white/Learning push origin master

# Verify course files after subagents
find /home/white/Learning/PATH/TO/COURSE -type f | sort
wc -l /home/white/Learning/PATH/TO/COURSE/*.md
```

## Repository
- **GitHub:** https://github.com/Leito2/Learning.git
- **Branch:** master
- **Linux path:** /home/white/Learning
- **Actual vault path:** /home/white/Learning/SW-ML-AI Engineering/

---

## 🎯 High-Value Technologies for This Profile (2025-2026)

Filtered from a broader tech scan — only technologies that directly complement the user's existing Go/LangGraph/Redis/RAG/Agentic/AWS/GCP profile. 🚨 = gap (no dedicated notes yet), ✅ = covered.

### Production LLM Serving & Inference
| Tech | Status | Why |
|------|:------:|-----|
| **vLLM** | ✅ (06/13) | PagedAttention, continuous batching. Standard for production LLM APIs. |
| **SGLang** | ✅ (06/17) | RadixAttention for structured output. Faster than vLLM for LLM-as-a-Judge. Deep course with speculative decoding, quantization. |
| **TensorRT-LLM** | 🚨 | NVIDIA's max-throughput engine for GPU-heavy deployments. |
| **Unsloth** | ✅ (06/14) | Fine-tuning 2-5x faster, 80% less memory. Deep course covering QLoRA, SFT, DPO, deployment. |

### Advanced RAG (partially created)
| Tech | Status | Why |
|------|:------:|-----|
| **Hybrid Search (BM25 + Dense)** | ✅ (06/13) | Redis + vector search. Already used in LLM Gateway project. |
| **Reranking (Cohere, bge-reranker)** | ✅ (06/13) | Second-stage precision. Critical for production RAG accuracy. |
| **GraphRAG** | ✅ (06/13) | Microsoft. Multi-hop reasoning over knowledge graphs. |
| **RAGAS / DeepEval** | ✅ (06/13) | RAG quality evaluation. Bridges to LLM Evaluation Suite project. |
| **ColBERT (late interaction)** | ✅ (06/17) | Token-level retrieval. State-of-the-art for passage search. Deep course with PLAID indexing, hybrid RAG capstone. |
| **Late Chunking (Jina)** | 🚨 | Context-aware chunking that keeps surrounding text in embeddings. |

### Agentic AI & Protocols
| Tech | Status | Why |
|------|:------:|-----|
| **MCP (Model Context Protocol)** | ✅ (07/15) | Anthropic standard for agent-to-tool communication. Deep course with LangGraph integration. |
| **A2A (Agent-to-Agent)** | ✅ (07/15) | Google protocol. Multi-agent enterprise communication standard. |
| **LangGraph Deep Patterns** | ✅ (07/15) | Subgraphs, dynamic routing, human-in-the-loop. MCP integration with LangGraph. |
| **Computer Use / Browser Agents** | ✅ (07/15) | Claude Computer Use, Browser-use. Agent interacts with real UIs. |

### MLOps & Infrastructure
| Tech | Status | Why |
|------|:------:|-----|
| **Feast (Feature Store)** | ✅ (09/27) | Online/offline feature serving, point-in-time joins, Redis integration. Deep course covering AWS/GCP deployment. |
| **Evidently AI / Phoenix** | 🚨 | Drift detection + LLM observability. Connects to Evaluation Suite project. |
| **LLM Guard / Guardrails AI** | ✅ (06/15) | Prompt injection defense, PII redaction, content safety. Deep course covering NeMo, Guardrails AI, Presidio, Lakera. |
| **Knative / KServe** | 🚨 | Serverless model serving on Kubernetes. Production deployment. |
| **Temporal** | 🚨 | Durable execution for long-running ML pipelines with retries and state. |

### Real-Time ML & Streaming
| Tech | Status | Why |
|------|:------:|-----|
| **WebSockets for ML Serving** | ✅ (10/30) | Deep course: protocol internals, real-time inference, scaling, ML Gateway. Complements Go/Fiber/Sudoku Together. |
| **Redis Pub/Sub + WebSocket scaling** | ✅ (10/30) | Horizontal scaling of WS connections. Natural extension of LLM Gateway. |
| **Server-Sent Events (SSE)** | ✅ (Go Local AI) | Token streaming. Covered but could deepen. |

### Data & Feature Engineering
| Tech | Status | Why |
|------|:------:|-----|
| **DuckDB** | 🚨 | OLAP in-process. Fast analytics in Python/Go without Spark. |
| **dbt** | ✅ (10/29) | Data transformations. Already covered. |

### Vector Databases
| Tech | Status | Why |
|------|:------:|-----|
| **pgvector** | ✅ (10/33) | PostgreSQL extension for vector search. Deep 2-note course: core ops, hybrid search, production tuning. |
| **Qdrant** | ✅ (10/33) | Rust-based vector DB. Deep 2-note course: architecture, collections, distributed deployment, Go/Python clients. |
| **Milvus** | ✅ (10/33) | Distributed GPU-accelerated vector DB. Deep 2-note course: architecture, Kubernetes, multi-tenancy, tiered storage. |

> **Removed from original scan:** Rust-exclusive tech (already covered but not user's core focus), Spark/BigQuery (covered, not profile-aligned), GNN/RL (covered, niche), MoE/SSM/Jamba (too research-stage), Mojo/MAX (pre-release), Iceberg/Delta Lake (irrelevant to RAG/Agentic profile).

---

## Next Course Candidates (prioritized for this profile)

| # | Course | Notes | Justification |
|:--:|--------|:-----:|---------------|
| ~~1~~ | ~~Unsloth and Efficient Fine-Tuning~~ | ~~5~~ | ✅ CREATED (06/14) |
| ~~2~~ | ~~MCP, A2A and Agentic Protocols~~ | ~~6~~ | ✅ CREATED (07/15) |
| ~~3~~ | ~~WebSockets and Real-Time ML Serving~~ | ~~5~~ | ✅ CREATED (10/30) |
| ~~4~~ | ~~LLM Security and Guardrails~~ | ~~5~~ | ✅ CREATED (06/15) |
| ~~5~~ | ~~Feast and Feature Stores for MLOps~~ | ~~5~~ | ✅ CREATED (09/27) |
| ~~7~~ | ~~Bun Runtime~~ | ~~7~~ | ✅ CREATED (10/33) |
| ~~8~~ | ~~ONNX Runtime Rust — Deepened + 2 New Notes~~ | ~~3~~ | ✅ DONE (14/04) |
| ~~6~~ | ~~ColBERT, SGLang and Advanced Inference~~ | ~~7~~ | ✅ CREATED (06/17). ColBERT late interaction, PLAID, SGLang RadixAttention, speculative decoding, quantization. Capstone: hybrid RAG with ColBERT reranking + SGLang judge.
| ~~9~~ | ~~SDD: AI-Assisted Project Architecture~~ | ~~11~~ | ✅ CREATED AS **SDD and Harness Engineering** (16/00-10). Harness Engineering + SDD workflow + 20 harnesses + 4 file structure alternatives.
| 10 | **Vector Databases and Semantic Search** | 12 | ✅ CREATED (10/33). pgvector, Qdrant, Milvus deep courses + comparison + capstone with Go and Python.
| 11 | **HuggingFace Transformers Deep Dive** | 10 | ✅ CREATED (06/16). from_pretrained, tokenizers, Trainer, generation, vision/audio/multimodal, optimum, Diffusers (2 notes), capstone.
| 12 | **Deep Learning with TensorFlow** | 7 | ✅ CREATED (05/09). tf.keras architectures, tf.data/TFRecord, distributed training (TPU/GPU), TensorBoard/callbacks/tuning, SavedModel/TF Serving/TFLite, CV capstone with EfficientNet.

### Remaining High-Priority Gaps (4 courses)

| # | Course | Est. Notes | Justification |
|:--:|--------|:-----:|---------------|
| 13 | **JAX Deep Dive** | 5-6 | Google's high-performance ML framework. DeepMind uses it for AlphaFold, Gemini. XLA compilation, `pmap`/`vmap`, Flax `linen`, TPU training. Fills "Present but shallow" gap. |
| 14 | **TorchServe** | 3-4 | PyTorch-native model serving. MAR files, custom handlers, model archiver, multi-model endpoints. Bridges PyTorch training (M05/03) to production (M09). |
| 15 | **Evidently AI / Phoenix** | 4-5 | Drift detection + LLM observability. Connects directly to Evaluation Suite portfolio project. Critical for production ML monitoring. |
| 16 | **DuckDB** | 3-4 | OLAP in-process analytics. Fast SQL analytics in Python/Go without Spark. Complements RAG preprocessing and data exploration workflows. |

---

> **⚠️ MANDATORY for next session:**
> 1. Read the "CRITICAL: CONTEXT MANAGEMENT" section.
> 2. ALWAYS use subagents (`task` tool) for bulk creation — max 2 parallel, max 7 notes each.
> 3. Never write 5+ full course notes in the main thread.
> 4. Verify filesystem state after every subagent batch.
> 5. **Reorganized:** 00=Markdown, 01=SQL, 02=Docker, 03=Advanced Python, 04=Engineering Fundamentals, 05-12=ML core, 13=Go, 14=Rust, 15=Transversal, 16=SDD and Harness Engineering, Extra/ at end.
> 6. Next priority: **JAX Deep Dive** (5-6 notes) OR **TorchServe** (3-4 notes) OR **Evidently AI / Phoenix** (drift detection).
> 7. **Completed this session:** ColBERT, SGLang and Advanced Inference course (7 notes, ~3,373 lines, 2 subagents). Deep format applied. Module 06/17. Etymology explained (ColBERT = Contextualized Late Interaction over BERT; SGLang = Structured Generation Language). Covers token-level late interaction, PLAID indexing, RadixAttention, structured LLM programs, speculative decoding, quantization, and capstone hybrid RAG.
> 8. Portfolio URL: https://white-portfolio-ia-ml-engineer.netlify.app/
