# Continuity Prompt for Next Kimi Session

Copy and paste this entire file into a new Kimi session to continue the project without losing context.

---

## Project Context

We are building an **Obsidian vault** inside `C:\Users\Leito\Documents\Learning`, which is a **Git repository** connected to `https://github.com/Leito2/Learning.git`. The vault contains compressed markdown courses for an **AI/ML Engineer** learning path. The vault is also used to document portfolio projects for landing a first job as an ML/AI Engineer.

**Current Goal:** Build the vault into a comprehensive, job-ready knowledge base. All **new** courses and notes are written in **English** with deep theoretical and practical depth. The user is looking for their first ML/AI Engineer job.

---

## Vault Structure

```
Learning/
├── .obsidian/                          # Obsidian configuration
├── Welcome.md                          # Main vault index (Spanish)
│
├── Curso Markdown/                     # Complete (5 notes)
├── Curso SQL con PostgreSQL/           # Complete (7 notes + assets)
│
├── Advanced Python/                    # Complete (62 notes + images)
│   ├── 01 - Python Basico/             # 13 notes
│   ├── 02 - Python Intermedio/         # 13 notes
│   ├── 03 - Python Avanzado/           # 13 notes
│   ├── 04 - Librerias Basicas de Python/# 8 notes
│   └── 05 - Librerias Especificas/     # 7 notes
│
├── Software Engineering/               # Active area
│   ├── 01 - Docker Profesional/        # Complete (7 notes)
│   ├── extra/                          # NEW (6 notes - English)
│   │   ├── 00 - Welcome to SE Extra.md
│   │   ├── 01 - FastAPI for ML.md
│   │   ├── 02 - System Design for ML.md
│   │   ├── 03 - Testing in ML Systems.md
│   │   ├── 04 - CI-CD for ML.md
│   │   └── 05 - Rust for ML Infra.md
│   └── projects/                       # NEW (6 notes - English)
│       ├── 00 - Project Planning Guide for SE.md
│       ├── 01 - FastAPI for ML - Project Guide.md
│       ├── 02 - System Design for ML - Project Guide.md
│       ├── 03 - Testing in ML Systems - Project Guide.md
│       ├── 04 - CI-CD for ML - Project Guide.md
│       └── 05 - Rust for ML Infra - Project Guide.md
│
├── SW-ML-AI Engineering/               # Main vault for SW + ML + AI courses
│   ├── Welcome.md                      # Sub-vault index (Spanish + English sections)
│   ├── 00 - Indice Maestro de Cursos.md # Master course index
│   │
│   ├── M00 - Fundamentos de Ingenieria/           # Complete + images (23 notes)
│   ├── M01 - Deep Learning y Computer Vision/     # Complete + images (18 notes)
│   ├── M02 - Large Language Models/               # Complete + images (30 notes)
│   ├── M03 - AI Agents y Agentic Systems/         # Complete + images (22 notes)
│   ├── M04 - NLP Avanzado/                        # Complete + images (17 notes)
│   ├── M05 - MLOps y Produccion/                  # Complete + images (24 notes)
│   ├── M06 - Cloud, Infra y Backend/              # Complete + images (24 notes)
│   ├── M07 - Research y Ciencia de Datos/         # Complete + images (24 notes)
│   ├── M08 - Producto, Negocio y Open Source/     # Complete + images (18 notes)
│   │
│   ├── extra/                          # NEW (8 notes - English)
│   │   ├── 00 - Welcome to ML Extra Courses.md
│   │   ├── 01 - Kaggle Competitions.md
│   │   ├── 02 - End-to-End ML Project.md
│   │   ├── 03 - Fine-Tuning LLMs.md
│   │   ├── 04 - Production RAG System.md
│   │   ├── 05 - Computer Vision Pipeline.md
│   │   ├── 06 - Advanced MLOps.md
│   │   └── 07 - Paper Reproduction.md
│   └── projects/                       # NEW (8 notes - English)
│       ├── 00 - Project Planning Guide for ML.md
│       ├── 01 - Kaggle Competitions - Project Guide.md
│       ├── 02 - End-to-End ML Project - Project Guide.md
│       ├── 03 - Fine-Tuning LLMs - Project Guide.md
│       ├── 04 - Production RAG System - Project Guide.md
│       ├── 05 - Computer Vision Pipeline - Project Guide.md
│       ├── 06 - Advanced MLOps - Project Guide.md
│       └── 07 - Paper Reproduction - Project Guide.md
│
├── Transversal Skills/                 # NEW (4 notes - English)
│   ├── 00 - Welcome to Transversal Skills.md
│   ├── 01 - Technical English.md
│   ├── 02 - Technical Communication and Storytelling.md
│   └── 03 - Technical Leadership.md
│
├── Software Engineering/
│   ├── Go Engineering/                 # COMPLETE (49 notes - English)
│   │   ├── 00 - Welcome to Go Engineering.md
│   │   ├── 01 - Go Fundamentals/       # ✅ IMPROVED — Deep theory, ASCII, ML context (4179 lines)
│   │   ├── 02 - Go for Cloud Native/   # 7 notes
│   │   ├── 03 - Microservices with Go/ # 7 notes
│   │   ├── 04 - DevSecOps and CLI Tools/ # 7 notes
│   │   ├── 05 - Local AI with Go/      # 7 notes
│   │   ├── 06 - Go for ML Backend/     # 7 notes
│   │   ├── 07 - Gorgonia - Deep Learning in Go/  # ✅ NEW (6 notes - English, deep format)
│   │   ├── 07 - LocalAI - Local LLM Server/      # ✅ NEW (6 notes - English, deep format)
│   │   ├── 07 - Wails - Desktop Apps with Go/    # ✅ NEW (6 notes - English, deep format)
│   │   ├── projects/                   # 6 notes
│   │   └── extra/                      # 7 notes
│   │
│   └── Rust Engineering/               # ✅ COMPLETE (49 notes + 12 new = 61 notes - English)
│       ├── 00 - Welcome to Rust Engineering.md
│       ├── 01 - Rust Fundamentals/     # 7 notes
│       ├── 02 - Advanced Rust/         # 7 notes
│       ├── 03 - Rust for Data Engineering/ # 7 notes
│       ├── 04 - Rust for ML and AI/    # 7 notes
│       ├── 05 - WebAssembly and Edge AI/ # 7 notes
│       ├── 06 - Agentic AI with Rust/  # 7 notes
│       ├── 07 - Polars Internals and Advanced/   # ✅ NEW (6 notes - English, deep format)
│       ├── 07 - Candle Advanced Patterns/        # ✅ NEW (6 notes - English, deep format)
│       ├── projects/                   # 6 notes
│       └── extra/                      # 7 notes
```

**Current total: ~513 notes created** (294 original + 42 SE/ML/Transversal + 67 Go + 61 Rust + 49 restructuring)

**Format Improvement Progress:**
| Course | Status | Avg Lines | Notes |
|--------|--------|-----------|-------|
| Go 01 - Fundamentals | ✅ IMPROVED | ~600 | 7/7 |
| Go 02 - Cloud Native | ✅ IMPROVED | ~500 | 7/7 |
| Go 03 - Microservices | ✅ IMPROVED | ~500 | 7/7 |
| Go 04 - DevSecOps | ✅ IMPROVED | ~520 | 7/7 |
| Go 05 - Local AI | ✅ IMPROVED | ~550 | 7/7 |
| Go 06 - ML Backend | ✅ IMPROVED | ~530 | 7/7 |
| Go 07 - Gorgonia | ✅ CREATED (deep format) | ~380 | 6/6 |
| Go 07 - LocalAI | ✅ CREATED (deep format) | ~410 | 6/6 |
| Go 07 - Wails | ✅ CREATED (deep format) | ~380 | 6/6 |
| Go extra | ✅ IMPROVED | ~770 | 7/7 |
| Go projects | ✅ IMPROVED | ~770 | 6/6 |
| Rust 01 - Fundamentals | ✅ IMPROVED | ~510 | 7/7 |
| Rust 02 - Advanced Rust | ✅ IMPROVED | ~770 | 7/7 |
| Rust 03 - Data Engineering | ✅ IMPROVED | ~640 | 7/7 |
| Rust 04 - Rust for ML/AI | ✅ IMPROVED | ~760 | 7/7 |
| Rust 05 - WebAssembly | ✅ IMPROVED | ~790 | 7/7 |
| Rust 06 - Agentic AI | ✅ IMPROVED | ~730 | 7/7 |
| Rust 07 - Polars | ✅ CREATED (deep format) | ~360 | 6/6 |
| Rust 07 - Candle | ✅ CREATED (deep format) | ~290 | 6/6 |
| Rust extra | ✅ IMPROVED | ~940 | 7/7 |
| Rust projects | ✅ IMPROVED | ~1190 | 6/6 |

**Git Checkpoints:**
- *(next)* — Folder rename ML and IA Engineering → SW-ML-AI Engineering + skills maps + gap analysis
- `38a4a25` — Continuity Prompt updated with improvement progress
- `888e6a6` — Deep expansion courses + Go 02-06 + Rust 01 improved
- `ef57e18` — Go Fundamentals improved with new deep format
- `6de9e0e` — Rust Engineering complete (revert point)

---

## Language Policy

- **Existing modules (M00-M08, Advanced Python, Docker, Markdown, SQL):** Written in Spanish. **DO NOT modify** unless explicitly requested.
- **All new content (SE extra/projects, ML extra/projects, Transversal, Go Engineering, future Rust Engineering):** Written entirely in **English**.
- **Welcome.md and indices:** May contain both Spanish and English sections to bridge old and new content.

---

## Note Style and Format

### ⭐ NEW IMPROVED Format (applied to Go Fundamentals 01)

The course `01 - Go Fundamentals` has been upgraded with a **deeper, modular format** that must be used for ALL future note improvements. This format produced notes of **400-1200+ lines** (vs 200-370 original).

```markdown
# <Title with emoji>

## 🎯 Learning Objectives
<Bullet list of what the reader will master in this specific module>

---

## Introduction
<2-3 paragraphs. Deep context about WHY this matters for ML/AI engineering.
Connect to other vault modules via [[...]] links.>

---

## Module X: <Concept Name>

### X.1 Theoretical Foundation 🧠
<3-4 paragraphs explaining WHY this concept exists.
Historical context, computer science theory, design motivation.
Show the problem this concept solves before showing the solution.>

### X.2 Mental Model 📐
<ASCII diagram or conceptual table that builds intuition>
```
┌─────────────────────────────────────────────┐
│  Concept Visualization (ASCII)              │
├─────────────────────────────────────────────┤
│  Component A ──────► Component B            │
│       │                    │                │
│       └────────────────────┘                │
└─────────────────────────────────────────────┘
```

### X.3 Syntax and Semantics 📝
<Exact syntax with line-by-line comments explaining what each line does>
```go / rust / python
// code with inline comments explaining WHY, not just WHAT
```

### X.4 Visual Representation 🖼️
<Mermaid diagram AND reference to Wikimedia image URL>
```mermaid
<diagram>
```
![Description](https://upload.wikimedia.org/wikipedia/commons/thumb/...)

### X.5 Application in ML/AI Systems 🤖
<Specific real-world ML example showing this concept in production.
Include a mini-case study with company name, problem, solution, and impact.>

| ML Use Case | This Concept | Impact |
|-------------|-------------|--------|
| ... | ... | ... |

### X.6 Common Pitfalls ⚠️
⚠️ <Warning with explanation of WHY it happens>
⚠️ <Warning>
💡 <Tip with mnemonic or mental shortcut>

### X.7 Knowledge Check ❓
<2-3 questions or mini-exercises that test understanding>

---

## 📦 Compression Code
<Complete, production-ready code summarizing ALL concepts in the module>

## 🎯 Documented Project

### Description
<2-3 sentences about a real-world ML project using these concepts>

### Functional Requirements
1. ...
2. ...
3. ...
4. ...
5. ...

### Main Components
- ...
- ...
- ...

### Success Metrics
- ...
- ...
- ...

### References
- Official docs: <URL>
- Paper/library: <URL>
```

### Key Rules for IMPROVED Format
- **Theory BEFORE code** in EVERY section — never show code without first explaining WHY the concept exists
- **At least 3 ASCII diagrams** per note — use `┌─│└├┘┤┬┴┼` characters for structural diagrams
- **At least 2 Mermaid diagrams** per note — varied types (flowchart, sequence, state, class)
- **At least 2 Wikimedia image URLs** per note — reference the URL even if not yet downloaded
- **Each concept has ML/AI application** — show "Real case: <company>" or a usage table
- **Target 400-600 lines** (welcome notes: 200-300)
- **Modular organization** — each concept is a self-contained "Module" with sub-sections X.1-X.7
- **Code has WHY comments** — not just WHAT, but WHY each pattern is used

### Assets Directory
Each course should have an `assets/` folder for images and diagrams:
```
<Course>/assets/
├── README.md                 # Asset inventory
└── module-X-topic.png        # Images referenced in notes
```
Reference images as: `![Description](assets/module-X-topic.png)`
Keep Wikimedia URLs as fallback: `![Description](https://upload.wikimedia.org/...)`

---

### Original Format (Legacy courses - M00-M08, SE extra/projects)

### For Project Guide Notes (projects/ folder)

```markdown
# <Title with emoji>

## Overview
Why this matters for a first job.

## Prerequisites

## Learning Objectives

## Official Resources & Links
| Resource | Type | URL | Why It Matters |
|----------|------|-----|----------------|
(At least 5 links per note)

## Architecture & Planning
Mermaid diagram + key decisions.

## Step-by-Step Implementation Guide
5-10 numbered steps with what, why, code, expected output.

## Guide Class / Example
Complete copy-pasteable code.

## Common Pitfalls & Checklist
⚠️ At least 3 mistakes
✅ Checklist table

## Deployment & Portfolio Integration
How to deploy and present for recruiters.

## Next Steps
[[...]] internal links.
```

### Formatting Rules
- Use `[[...]]` for internal Obsidian links.
- File names: `## - Descriptive Name.md` (with spaces).
- **IMPORTANT:** Do NOT use `/` in file names. Use `-` instead (Windows restriction).
- One H1 (`#`) per note only.
- Blank lines between paragraphs.
- Markdown tables for comparisons.
- Emojis for sections: 📦 (code), 🎯 (project), 💡 (tip), ⚠️ (warning).

---

## CRITICAL RULE: Subagent (Task Tool) Usage

### Identified Problem
Subagents sometimes complete work (files created on disk) but do not return a result message, appearing "hung." This happens when:
- More than 3 subagents are launched in parallel.
- A subagent must create more than 7 notes at once.
- Prompts are extremely long.

### MANDATORY Workflow for Creating Courses

1. **ALWAYS** use subagents to create batches of notes (creating one by one consumes too much main model context).
2. **Maximum 2 subagents in parallel per batch.**
3. **Maximum 7 notes per subagent.** If a course has more notes, split into 2 sequential subagents (verify the first's filesystem before launching the second).
4. **IMMEDIATELY** after launching subagents, verify the filesystem with bash:
   ```powershell
   Get-ChildItem -Recurse -File "C:\Users\Leito\Documents\Learning\PATH\TO\COURSE" | ForEach-Object { $_.FullName }
   ```
5. If files exist, consider the task completed even if the subagent has not reported anything.
6. If files are missing, re-launch ONLY the missing ones.
7. **The filesystem is the single source of truth.**

---

## Upcoming Work Queue

### 1. Go Engineering (✅ COMPLETE — 73 notes)
- **01 - Go Fundamentals:** ✅ IMPROVED to new deep format (~600 lines/note)
- **02 - Go for Cloud Native:** ✅ IMPROVED to new deep format (~500 lines/note)
- **03 - Microservices with Go:** ✅ IMPROVED to new deep format (~500 lines/note)
- **04 - DevSecOps and CLI Tools:** ✅ IMPROVED to new deep format (~520 lines/note)
- **05 - Local AI with Go:** ✅ IMPROVED to new deep format (~550 lines/note)
- **06 - Go for ML Backend:** ✅ IMPROVED to new deep format (~530 lines/note)
- **07 - Gorgonia:** ✅ CREATED (deep format, 6 notes)
- **07 - LocalAI:** ✅ CREATED (deep format, 6 notes)
- **07 - Wails:** ✅ CREATED (deep format, 6 notes)
- **Projects:** ✅ IMPROVED (6 notes, ~770 avg lines)
- **Extra:** ✅ IMPROVED (7 notes, ~770 avg lines)

### 2. Rust Engineering (✅ COMPLETE — 73 notes)
- **01 - Rust Fundamentals:** ✅ IMPROVED to new deep format (~510 lines/note)
- **02 - Advanced Rust:** ✅ IMPROVED to new deep format (~770 lines/note)
- **03 - Rust for Data Engineering:** ✅ IMPROVED to new deep format (~640 lines/note)
- **04 - Rust for ML and AI:** ✅ IMPROVED to new deep format (~760 lines/note)
- **05 - WebAssembly and Edge AI:** ✅ IMPROVED to new deep format (~790 lines/note)
- **06 - Agentic AI with Rust:** ✅ IMPROVED to new deep format (~730 lines/note)
- **07 - Polars Internals:** ✅ CREATED (deep format, 6 notes)
- **07 - Candle Advanced:** ✅ CREATED (deep format, 6 notes)
- **Projects:** ✅ IMPROVED (6 notes, ~1190 avg lines)
- **Extra:** ✅ IMPROVED (7 notes, ~940 avg lines)

### 3. Note Improvement Priority — ALL DONE ✅

| Course | Current Avg Lines | Target | Status |
|--------|-------------------|--------|--------|
| Go 01 - Fundamentals | ~600 | 400-600 | ✅ Done |
| Go 02 - Cloud Native | ~500 | 400-600 | ✅ Done |
| Go 03 - Microservices | ~500 | 400-600 | ✅ Done |
| Go 04 - DevSecOps | ~520 | 400-600 | ✅ Done |
| Go 05 - Local AI | ~550 | 400-600 | ✅ Done |
| Go 06 - ML Backend | ~530 | 400-600 | ✅ Done |
| Rust 01 - Fundamentals | ~510 | 400-600 | ✅ Done |
| Rust 02 - Advanced Rust | ~770 | 400-600 | ✅ Done |
| Rust 03 - Data Engineering | ~640 | 400-600 | ✅ Done |
| Rust 04 - Rust for ML/AI | ~760 | 400-600 | ✅ Done |
| Rust 05 - WebAssembly | ~790 | 400-600 | ✅ Done |
| Rust 06 - Agentic AI | ~730 | 400-600 | ✅ Done |
| Go projects/extra | ~770 | 300-500 | ✅ Done |
| Rust projects/extra | ~1065 | 300-500 | ✅ Done |
| Go 07 deep courses | ~380 | 400-600 | ✅ Done |
| Rust 07 deep courses | ~330 | 400-600 | ✅ Done |

### 4. Deep Content Expansion — Go (✅ COMPLETE)
**Course: Gorgonia — Deep Learning in Go** (6 notes) ✅ CREATED
**Course: LocalAI — Local LLM Server** (6 notes) ✅ CREATED
**Course: Wails — Desktop Apps with Go** (6 notes) ✅ CREATED

### 5. Deep Content Expansion — Rust (✅ COMPLETE)
**Course: Polars Internals and Advanced** (6 notes) ✅ CREATED
**Course: Candle Advanced Patterns** (6 notes) ✅ CREATED

### 6. Portfolio & Skills Analysis (✅ COMPLETE)
- Portfolio gap analysis completed → 22 skills missing from Learning vault
- Two skill maps created: `Skills Universe - Portfolio Aligned.md` + `Skills Tree - The AI-ML Engineer Growth Map.md`
- Folder renamed: `ML and IA Engineering` → `SW-ML-AI Engineering`

### 7. NEW — Learning Vault Gap Fill 🚧 PENDING (22 skills, 6 additions)

Priority: 🔴 HIGH — These are skills identified as missing from the Learning vault that are needed for job market competitiveness.

#### 7A. Expand `M09 - Extra` (+8 notes)
Skills: JAX/Flax, Synthetic Data Generation, Speculative Decoding, Agent Swarms, Browser Agents, TinyML/Edge ML, Federated Learning, Azure ML

#### 7B. New course: `M10 - Reinforcement Learning for AI Engineers` (+6 notes)
Skills: RL fundamentals, PPO, DQN, RLHF, reward modeling, alignment theory, RL case study
Critical because: RLHF is how ChatGPT, Claude, Gemini are aligned.

#### 7C. New course: `M10 - Graph Neural Networks` (+4 notes)
Skills: GCN, GAT, GraphSAGE, geometric DL case study
Applications: molecular discovery, recommendation systems, knowledge graphs.

#### 7D. New course: `M10 - Distributed ML Infrastructure` (+7 notes)
Skills: Apache Kafka, Ray/Ray Serve, NVIDIA Triton, Airflow/Prefect, dbt, Dagster
Aggregates all missing distributed infrastructure skills.

#### 7E. New course: `M10 - ML Platform Engineering` (+5 notes)
Skills: Kubeflow, Weights & Biases, Seldon Core, Great Expectations, BentoML
The evolution beyond MLOps — internal tooling for ML teams.

#### 7F. Expand `Extra Skills/` (+3 notes)
Skills: ML Interview Preparation, Technical Sales/Solutions Engineering, ML Platform Engineering overview
Direct job market preparation.

| Addition | Location | Notes | Priority |
|----------|----------|:-----:|:--------:|
| Expand M09 Extra | SW-ML-AI Engineering/M09 | 8 | 🔴 High |
| RL for AI Engineers | SW-ML-AI Engineering/M10 | 6 | 🔴 High |
| Graph Neural Networks | SW-ML-AI Engineering/M10 | 4 | 🟡 Medium |
| Distributed ML Infra | SW-ML-AI Engineering/M10 | 7 | 🔴 High |
| ML Platform Engineering | SW-ML-AI Engineering/M10 | 5 | 🟡 Medium |
| Expand Extra Skills | Extra Skills/ | 3 | 🟡 Medium |
| **TOTAL** | | **33** | |

### 8. Portfolio Web Redesign (PLANNED)
Update `leito2.github.io` to professional ML/AI Engineer portfolio with 76 skills (up from 34).

---

## Deep Content Reference (Go & Rust for ML)

This section provides reference material for expanding Go and Rust tracks with deeper ML content. Use these when creating new notes or expanding existing ones.

---

### 🐹 Go Deep Content: Gorgonia

**What is Gorgonia?**
Gorgonia is a library for machine learning in Go, similar to TensorFlow/PyTorch but native to Go. It provides:
- Tensor operations (N-D arrays)
- Automatic differentiation (autodiff)
- Neural network building blocks (layers, activations, losses)
- GPU support via CUDA

**Key Concepts to Cover:**
```
Gorgonia Architecture:
├── gorgonia.org/tensor        # NDArray operations (like NumPy)
├── gorgonia.org/gorgonia      # Computational graphs + autodiff
├── gorgonia.org/genfuncs      # Generic functions for tensors
└── gorgonia.org/vm            # Virtual machine for graph execution
```

**API Comparison:**
| Operation | NumPy | PyTorch | Gorgonia |
|-----------|-------|---------|----------|
| Matrix multiply | `np.dot(A, B)` | `torch.matmul(A, B)` | `gorgonia.MatMul(A, B)` |
| ReLU | `np.maximum(0, x)` | `torch.relu(x)` | `gorgonia.Rectify(x)` |
| Softmax | `scipy.special.softmax(x)` | `torch.softmax(x, dim)` | `gorgonia.SoftMax(x)` |
| Backward | N/A (manual) | `loss.backward()` | `gorgonia.Grad(loss)` |

**When to Use Gorgonia:**
- Go-only environments (no Python allowed)
- Embedded systems where Go is the primary language
- Building custom ML frameworks
- Prototyping before porting to PyTorch

**Real Cases:**
- CockroachDB: uses gonum for query optimization
- Various fintech companies: Go-first ML pipelines

---

### 🐹 Go Deep Content: LocalAI

**What is LocalAI?**
LocalAI is an open-source, drop-in replacement for OpenAI API that runs locally. Written in Go by Mudler (Ettore Di Giacinto). It enables:
- Running LLMs, image generation, audio transcription, and embedding models locally
- OpenAI-compatible API (use existing OpenAI SDKs with local models)
- Multiple backends: llama.cpp, whisper.cpp, stable-diffusion.cpp, piper TTS
- Hardware acceleration: CUDA, Metal, Vulkan, CPU

**Architecture:**
```
LocalAI Architecture:
├── API Layer (Go)             # OpenAI-compatible REST API
├── Backend Manager (Go)       # Manages model backends
├── Backends (C/C++/Go):
│   ├── llama.cpp              # LLM inference (LLama, Mistral, etc.)
│   ├── whisper.cpp            # Audio transcription
│   ├── stable-diffusion.cpp   # Image generation
│   ├── piper                  # Text-to-speech
│   └── sentencetransformers   # Embeddings
├── Model Gallery              # One-click model download
└── Protobuf gRPC             # Communication with backends
```

**Key APIs:**
| Endpoint | Function | Backends |
|----------|----------|----------|
| `/v1/chat/completions` | Chat with LLM | llama.cpp |
| `/v1/completions` | Text completion | llama.cpp |
| `/v1/embeddings` | Generate embeddings | sentence-transformers |
| `/v1/images/generations` | Generate images | stable-diffusion |
| `/v1/audio/transcriptions` | Transcribe audio | whisper.cpp |

**Configuration (YAML):**
```yaml
name: llama-3-8b
backend: llama
parameters:
  model: llama-3-8b-Q4_K_M.gguf
  temperature: 0.7
  max_tokens: 2048
context_size: 8192
f16: true
threads: 8
gpu_layers: 35  # Offload to GPU
```

**Real Cases:**
- Enterprises deploying AI without sending data to cloud
- Hobbyists running AI on Raspberry Pi, old laptops
- Developers building local-first AI applications

---

### 🐹 Go Deep Content: ML Ecosystem

**gonum (Numerical Computing):**
```go
import (
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/gonum/optimize"
)

// Matrix operations
A := mat.NewDense(3, 3, []float64{1,2,3,4,5,6,7,8,9})
B := mat.NewDense(3, 3, nil)
B.Mul(A, A.T())  // Matrix multiply

// Statistics
data := []float64{1, 2, 3, 4, 5}
mean := stat.Mean(data, nil)
stddev := stat.StdDev(data, nil)
```

**gotch (PyTorch Bindings):**
```go
import "github.com/sugarme/gotch"

// Create tensor
ts := gotch.NewTensor([][]float32{{1, 2}, {3, 4}})

// Neural network
model := nn.NewLinear(vs, 784, 10, nil)
output := model.Forward(&input)
loss := output.CrossEntropyForLogits(&target)
```

**hfgo (HuggingFace for Go):**
```go
import "github.com/rhasspy-ai/hfgo"

// Download model
model := hfmodel.New("bert-base-uncased", nil)
tokenizer := model.Tokenizer()
tokens := tokenizer.Encode("Hello world")
```

---

### 🦀 Rust Deep Content: Polars Internals

**Polars Architecture:**
```
Polars Execution Model:
├── DataFrame (Eager)          # Immediate execution
├── LazyFrame (Lazy)           # Query planning + optimization
│   ├── Logical Plan           # Abstract query tree
│   ├── Optimizer              # Predicate pushdown, projection pushdown, common subexpression elimination
│   └── Physical Plan          # Optimized execution plan
├── ChunkedArray               # Columnar storage (Apache Arrow)
├── Series                     # 1D typed array
└── DataType                   # Int8-64, Float32/64, Utf8, Boolean, Date, Datetime, List, Struct
```

**Advanced Features to Cover:**

1. **Lazy Evaluation Optimization:**
```rust
// Polars optimizes BEFORE execution
let result = LazyCsvReader::new("data.csv")?
    .filter(col("age").gt(lit(18)))           // Predicate pushdown: filter at read time
    .select([col("name"), col("salary")])      // Projection pushdown: only read needed columns
    .groupby([col("department")])
    .agg([col("salary").mean()])
    .collect()?;                                // Optimization happens here
```

2. **Memory Mapping for Large Files:**
```rust
// Memory-map a Parquet file (zero-copy read)
let df = LazyParquetReader::new("huge_file.parquet")?
    .with_memory_map(true)    // Memory-map instead of loading
    .collect()?;
```

3. **Streaming for Out-of-Core Processing:**
```rust
// Process files larger than RAM
let df = LazyCsvReader::new("100GB_file.csv")?
    .with_streaming(true)     // Process in chunks
    .collect()?;
```

**Performance Formulas:**
```
Polars_Speedup ≈ 10-100× vs Pandas (single-threaded)
Polars_Memory ≈ 0.1× Pandas (Arrow format)
Parallel_Speedup ≈ min(Cores, Chunks) for embarrassingly parallel ops
```

---

### 🦀 Rust Deep Content: Candle Advanced

**Candle Architecture:**
```
Candle ML Framework:
├── candle-core                 # Tensor operations, autodiff
│   ├── CpuDevice              # CPU backend
│   ├── CudaDevice             # NVIDIA GPU (via cudarc)
│   ├── MetalDevice            # Apple Silicon GPU
│   └── DType                  # F16, F32, F64, BF16
├── candle-nn                   # Neural network layers
│   ├── Linear, Conv2d, LSTM
│   ├── BatchNorm, LayerNorm
│   └── Activations (ReLU, GELU, SiLU)
├── candle-transformers         # Pre-built model architectures
│   ├── Llama, Mistral, Phi
│   ├── Stable Diffusion
│   ├── Whisper
│   └── BERT
└── candle-wasm                 # WebAssembly support
```

**Custom Model Example:**
```rust
use candle_core::{Tensor, DType, Device, Result};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

struct MLP {
    l1: Linear,
    l2: Linear,
}

impl MLP {
    fn new(vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: Linear::new(vs.get((128, 784), "l1.weight")?, Some(vs.get(128, "l1.bias")?)),
            l2: Linear::new(vs.get((10, 128), "l2.weight")?, Some(vs.get(10, "l2.bias")?)),
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.l1)?.gelu()?.apply(&self.l2)
    }
}
```

**GPU Acceleration:**
```rust
// Candle automatically uses GPU if available
let device = Device::cuda_if_available(0)?;
let tensor = Tensor::randn(0f32, 1f32, (1024, 1024), &device)?;
```

---

### 🦀 Rust Deep Content: PyO3 Advanced Patterns

**Complex Type Conversion:**
```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// Rust struct ↔ Python dict
#[pyclass]
#[derive(Clone)]
struct MLConfig {
    #[pyo3(get, set)]
    learning_rate: f64,
    #[pyo3(get, set)]
    batch_size: usize,
    #[pyo3(get, set)]
    model_name: String,
}

#[pymethods]
impl MLConfig {
    #[new]
    fn new(lr: f64, bs: usize, name: String) -> Self {
        Self { learning_rate: lr, batch_size: bs, model_name: name }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("learning_rate", self.learning_rate)?;
        dict.set_item("batch_size", self.batch_size)?;
        dict.set_item("model_name", &self.model_name)?;
        Ok(dict.into())
    }
}
```

**Async PyO3 with Tokio:**
```rust
use pyo3::prelude::*;
use tokio::runtime::Runtime;

#[pyfunction]
fn async_predict(py: Python<'_>, input: Vec<f64>) -> PyResult<PyObject> {
    let rt = Runtime::new()?;
    py.allow_threads(|| {
        rt.block_on(async {
            // Async Rust inference
            let result = model.predict_async(input).await?;
            Ok(result)
        })
    })
}
```

---

### 🦀 Rust Deep Content: Vector Search Algorithms

**HNSW (Hierarchical Navigable Small World):**
```
HNSW Index Structure:
├── Layer 0 (dense)           # All vectors connected
├── Layer 1 (sparse)          # Subset of vectors
├── Layer 2 (sparser)         # Fewer connections
└── Layer M (top)             # 1-2 vectors

Search Algorithm:
1. Start at top layer entry point
2. Greedy search: move to neighbor closest to query
3. When no improvement, drop to next layer
4. Repeat until layer 0
5. Return k nearest neighbors

Parameters:
- M: max connections per node (16-64)
- efConstruction: build-time search width (100-500)
- efSearch: query-time search width (10-500)
```

**Formulas:**
```
Recall@k = |Relevant_in_top_k| / |Total_Relevant|
Precision@k = |Relevant_in_top_k| / k
MRR = (1/|Q|) × Σ(1/rank_i)  # Mean Reciprocal Rank
HNSW_Query_Time = O(log(N))   # N = total vectors
```

---

### 🔗 Official Links for Deep Content

| Resource | URL | Use For |
|----------|-----|---------|
| Gorgonia | https://gorgonia.org/ | Go deep learning |
| LocalAI | https://localai.io/ | Local LLM server |
| gonum | https://gonum.org/ | Go numerical computing |
| gotch | https://github.com/sugarme/gotch | Go PyTorch bindings |
| Polars Book | https://pola.rs/polars/book/ | Rust DataFrames |
| Candle Docs | https://huggingface.co/docs/candle | Rust ML framework |
| PyO3 Guide | https://pyo3.rs/ | Rust-Python bindings |
| Qdrant Docs | https://qdrant.tech/documentation/ | Vector database |

---

## Git Commands

```powershell
# Check git status
git -C "C:\Users\Leito\Documents\Learning" status

# Stage, commit, push
git -C "C:\Users\Leito\Documents\Learning" add -A
git -C "C:\Users\Leito\Documents\Learning" commit -m "message"
git -C "C:\Users\Leito\Documents\Learning" push origin master

# Verify course files (USE ALWAYS after subagents)
Get-ChildItem -Recurse -File "C:\Users\Leito\Documents\Learning\PATH\TO\COURSE" | ForEach-Object { $_.FullName }

# Full structure view
Get-ChildItem -Recurse "C:\Users\Leito\Documents\Learning\Software Engineering\Go Engineering"
```

## Repository
- **GitHub:** https://github.com/Leito2/Learning
- **Branch:** master
- **Local path:** C:\Users\Leito\Documents\Learning

---

> **Tip for next session:** Start by verifying the filesystem state of any in-progress work, then proceed with the next batch of subagents following the max-2-parallel, max-7-notes rule.
