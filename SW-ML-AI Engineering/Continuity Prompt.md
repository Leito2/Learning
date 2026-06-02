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

## 1. The Problem and Why This Solution Exists
(Deep historical context. Competing approaches and their failures.
Architecture diagrams from OFFICIAL sources or Wikimedia, NOT ASCII.
No artificial paragraph limit — let the theory breathe.)

## 2. Conceptual Deep Dive
(Algorithms, data flow, mathematics. LaTeX for equations.
Code integrated into explanation, not isolated.
ONE Mermaid diagram ONLY IF it adds genuine architectural insight.)

## 3. Production Reality
(Hardware requirements, tradeoffs, known failure modes.
Real case: <Company X uses this for Y>.
Comparison table ONLY when comparing multiple approaches.)

## 4. Code in Practice
(Minimal, focused, runnable code. Not a "compression code"
that tries to cover everything — just the key pattern.)

---

## 🎯 Key Takeaways
(5-7 bullet points)

## References
(Papers, docs, vault [[links]])
```

### Visual Guidelines

| Rule | Detail |
|------|--------|
| **ASCII art** | **BANNED.** No `┌─│└├┘┤┬┴┼` diagrams. |
| **Real images** | Prefer Wikimedia Commons URLs: `![Alt](https://upload.wikimedia.org/...)` |
| **Official diagrams** | Use arxiv figures, GitHub docs, NVIDIA/Apple docs for project-specific architecture |
| **Mermaid** | Maximum 1-2 per note. Only for architecture/data flow that genuinely needs visualization. |
| **LaTeX** | Use aggressively for mathematical concepts. |
| **Tables** | Only when comparing approaches or listing tradeoffs — not as filler. |

### Line Targets

| Note Type | Lines |
|-----------|:-----:|
| Course welcome (00) | 60-90 |
| Core concept note | 300-450 |
| Capstone note | 350-500 |
| Quick-reference | 150-250 |
| Tool comparison | 150-300 |

### Key Rules
- **Theory BEFORE code** in every section. Conceptual and functional theory is the HIGHEST priority.
- Each note covers **ONE concept deeply** — do not cram multiple topics into a single note.
- **English** for all new content; **Spanish** only for existing M00-M08 modules.
- Code blocks MUST have language tag: ` ```python `, ` ```go `, ` ```rust `, ` ```sql `, ` ```bash `, ` ```yaml `, ` ```json `.
- Tables MUST use aligned columns.
- Mermaid MUST use ` ```mermaid ` wrapper.
- Use `[[...]]` for internal Obsidian links.
- File names: `## - Descriptive Name.md`. **Never use `/`** (Windows restriction).
- One H1 per note only.
- **THIS DEEP FORMAT IS THE DEFAULT STANDARD.** All notes must follow it. All subagents must be instructed to follow it.

---

## ⚡ Cutting-Edge Technology Preference

All new courses MUST prioritize the most modern, production-grade technologies.

### Preferred Over Legacy

| Prefer | Over | Because |
|--------|------|---------|
| **FP8 (E4M3/E5M2)** | INT8/INT4 quantization | Native H100/B200 support, hybrid precision |
| **Multi-Head Latent Attention (MLA)** | GQA/MQA | 90% KV cache reduction via compressed latent vectors |
| **Eagle / MTP speculative decoding** | Basic draft-model speculation | Feature-level speculation, no separate model needed |
| **Disaggregated prefill/decode** | Monolithic serving | 2x throughput via hardware specialization |
| **ExecuTorch / ONNX Runtime** | Raw PyTorch inference | NPU-ready, INT4/FP4 mapped to silicon |
| **Self-Refinement / Self-Consistency** | Static greedy decoding | Dynamic compute allocation per token |
| **SGLang RadixAttention** | vLLM PagedAttention (structured workloads) | Prefix sharing across program branches |

### Research Divisions to Cover

1. **Architecture-Level Optimizations**: MLA, MoE inference routing, SSM-based alternatives (Mamba-2), diffusion LLMs, mixture of depths.
2. **Runtime/System-Level Optimizations**: Speculative decoding 2.0, disaggregated serving, KV cache eviction, hybrid precision, prompt compression.

---

## 📐 Course Design Patterns — From Python Básico and Intermedio

> These courses are the canonical reference for what great vault notes look like. All future courses MUST follow these patterns.

### 1. One Concept Per Note (Granular Division)

| What the Python courses do | What the old format did |
|----------------------------|-------------------------|
| Each built-in type gets its own note (strings, lists, dicts, tuples, sets) | 5 different techniques crammed into one "Advanced Inference Optimization" note |
| OOP split across 3 notes (classes, inheritance, encapsulamiento) | One note per module with 5 mechanically repeated subsections |
| The capstone/project is a **separate note** (note 12) | "Documented Project" forced into every note |

**Rule**: If a topic has 3+ distinct sub-concepts, split it into separate notes. No exceptions.

### 2. Code That Converses (Inline Annotations)

Every code block must tell a mini-story. Use these patterns:

**a) `¡Sorpresa!` moments** — non-obvious behavior that creates an "aha" moment:
```python
print(agregar_medicion(22.5))  # [22.5]
print(agregar_medicion(23.1))  # [22.5, 23.1] ¡Sorpresa!
# El valor por defecto es el MISMO objeto lista en cada invocación.
```

**b) ❌/✅ antipattern pattern** — show the wrong way, then the right way:
```python
# ❌ Anidamiento profundo (arrow anti-pattern)
if usuario:
    if usuario.activo:
        if usuario.rol == "admin": ...

# ✅ Mejor: guard clauses u operadores lógicos
if usuario and usuario.activo and usuario.rol == "admin": ...
```

**c) Inline explanations** — each line gets a comment explaining what happened and why:
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)  # [1, 2, 3, 4] — 'a' y 'b' referencian el mismo objeto
```

### 3. Inline Warnings and Tips (NOT Separate Sections)

```markdown
⚠️ **Advertencia:** Insertar (`insert`) o eliminar (`pop(0)`) al principio de una lista es O(n)...
💡 **Tip:** Si encuentras `for i in range(len(lista))`, probablemente necesites `enumerate()`...
```

Warnings must appear **immediately after** the code block they refer to. Do NOT isolate them in a separate "Common Pitfalls" section.

### 4. "Caso real" Mini-Stories (NOT Generic Tables)

The old format used: `| ML Use Case | This Concept | Impact |` — cold, generic, filler.

The Python courses use narrative mini-stories:
```markdown
Caso real: En un pipeline de ML, usamos `isinstance(datos, (list, tuple))` para
aceptar secuencias sin importar su tipo concreto, permitiendo polimorfismo.
```

Every concept MUST have at least one "Caso real" that anchors it to a concrete ML/Backend scenario.

### 5. Código de Compresión (End-of-Note Runnable Summary)

Every note MUST end with a runnable script that exercises all key concepts in ~20-40 lines:

```python
# 📦 Código de compresión: Variables y Tipos de Datos
# Cubre: tipos built-in, id/type/isinstance, casting, mutabilidad, hashability

# 1. Variables como referencias
a = [1, 2, 3]
b = a
print(f"Mismo objeto? {a is b}")  # True

# 2. Introspección
x = 256
print(f"type={type(x).__name__}, id={id(x)}")

# 3. Hashability
print(f"hash((1, 'a')) = {hash((1, 'a'))}")
try:
    hash([1, 2])
except TypeError as e:
    print(f"Error esperado: {e}")
```

This is NOT a "Documented Project". It is a concise, executable reference card.

### 6. Direct Opening Hook

Every note opens with **1-2 sentences** connecting the topic to ML/AI/Backend reality:

```markdown
# 📦 01 - Variables y Tipos de Datos

Python es un lenguaje de tipado dinámico, pero esto no significa que carezca de
tipos. Comprender qué ocurre en memoria cuando escribes `x = 5` es fundamental
para depurar errores de aliasing en pipelines de datos y para optimizar el
consumo de memoria en servidores Backend.
```

No generic "What this concept is" paragraphs. Hook the reader immediately.

### 7. Theory and Code Coexist

The old format separated theory and code into distinct sections (`X.1 Theoretical Foundation` → `X.3 Syntax`).

The Python courses interleave them naturally:
```markdown
## 2. Tipos Built-in Fundamentales

| Tipo | Descripción | Mutable | Ejemplo |
|------|-------------|---------|---------|
| `int` | Enteros de precisión arbitraria | No | `42` |

💡 **Tip:** `bool` es subclase de `int` en CPython. `True == 1`...

```python
x = 10
print(type(x))          # <class 'int'>
print(id(x))            # 140735... (dirección en memoria)
print(isinstance(x, (int, float)))  # True
```

Caso real: En un pipeline de ML, usamos `isinstance(datos, (list, tuple))`...
```

Theory, table, tip, code, and case study appear in the SAME section. The section heading describes the concept, not the format of its content.

### 8. Organic Section Flow (No Rigid Subsections)

| Old format (bad) | Python format (good) |
|------------------|----------------------|
| `## Module X: Concept Name` followed by `X.1` through `X.7` mechanically | `## 1. Concepto A`, `## 2. Concepto B`, `## 3. Concepto C` — headings describe concepts, not template slots |
| Every module gets the same 7 subsections | Each note decides its own structure based on what the concept needs |

**Rule**: Section headings must describe the concept being covered (e.g., `## 3. Mutabilidad vs Inmutabilidad`), not the format slot (`X.2 Mental Model`).

### 9. What We ELIMINATED from the Old Format

| Eliminated | Why |
|------------|-----|
| `X.1-X.7` rigid subsections | Mechanical repetition destroys engagement |
| ASCII art with box-drawing characters | Banned; use real images or Mermaid |
| Mandatory "Documented Project" in every note | Project belongs in the capstone note only |
| Mandatory "Knowledge Check ❓" per module | Verification is the compression code, not quizzes |
| Mandatory `| ML Use Case | This Concept | Impact |` tables | Replaced by "Caso real" mini-stories |
| "Compression Code" trying to cover ALL topics | End-of-note code exercises only THIS note's concepts |
| 400-600 line targets | 300-450 is enough when each note is focused |

### 10. Adaptive Depth: Flexibility by Course Type

> **The 9 patterns above are the UNIVERSAL baseline.** Every note, regardless of topic, applies them. What varies is the *depth profile* — how much theory, images, LaTeX, and code each course type needs.

#### The Golden Rule

> **No course should be LESS engaging than Python Básico, but some courses MUST be MORE theoretical, MORE visual, and MORE mathematical than Python Básico.** The topic dictates the depth, not the template.

#### Flexibility Matrix

| Course Type | Vault Examples | Theory | Images | LaTeX | Code | Line Target |
|-------------|---------------|:------:|:------:|:-----:|:----:|:-----------:|
| **Language / Framework** | Python, Go, Rust, SQL, Docker, Markdown | Medium | Low | Low | **HIGH** | 250-400 |
| **ML Theory & Algorithms** | DL with PyTorch, CV, NLP, GNN, RL, TF | **HIGH** | Medium-High | **HIGH** | Medium-High | 350-500 |
| **Production / Infrastructure** | ColBERT, SGLang, Inference Opt., MLOps, Cloud, K8s | **HIGH** | **HIGH** | **HIGH** | Low-Medium | 350-500 |
| **Agents / Protocols** | MCP, A2A, LangGraph, CrewAI, Multi-Agent | High | Medium | Low-Medium | Medium-High | 300-450 |
| **Transversal / Business** | Leadership, Communication, Product Strategy | Low-Medium | Low | None | None | 150-300 |

#### Concrete Guidelines Per Type

**Language / Framework** (Python, Go, Rust, SQL, Docker):
- Code **IS** the primary teaching tool. Theory exists to explain WHY the code works.
- 60% code, 40% theory. Minimal images (logos ok). Minimal LaTeX.
- Example: `SW-ML-AI Engineering/03 - Advanced Python/01 - Python Basico/`

**ML Theory & Algorithms** (DL, CV, NLP, GNN, RL):
- Theory and math are **equal partners** with code. LaTeX for loss functions, gradients, attention.
- 50% theory + LaTeX, 50% code. Images for architecture (CNN diagrams, attention maps).
- Example: `SW-ML-AI Engineering/05 - Deep Learning y CV/03 - DL con PyTorch/`

**Production / Infrastructure** (ColBERT, SGLang, Inference, MLOps):
- Theory is the **protagonist**. Code demonstrates, not replaces.
- 70% theory + LaTeX + images, 30% code. Images are ESSENTIAL: GPU memory hierarchy, KV cache structure, quantization error distributions, hardware diagrams.
- LaTeX for compression ratios, throughput math, latency calculations.
- Example: The new `06/17 - ColBERT, SGLang and Next-Gen Inference/`

**Agents / Protocols** (MCP, LangGraph, CrewAI):
- Conceptual architecture + protocol design. Diagrams for agent flows.
- 50% theory, 50% code. Images for protocol sequences, agent graphs.

**Transversal / Business**:
- Pure conceptual writing. No code. Minimal images.

#### When to Use Each Tool (Priority by Course Type)

| Tool | Language | ML Theory | Production | Agents | Transversal |
|------|:--------:|:---------:|:----------:|:------:|:-----------:|
| `❌/✅` pattern | 🔴 Always | 🔴 Always | 🟡 Often | 🔴 Always | 🟢 Rarely |
| `¡Sorpresa!` moments | 🔴 Always | 🟡 Often | 🔴 Always | 🟡 Often | 🟢 Rarely |
| **LaTeX equations** | 🟢 Rarely | 🔴 Always | 🔴 Always | 🟡 Often | 🟢 Rarely |
| **Real images** (Wikimedia) | 🟡 Often | 🔴 Always | 🔴 **Essential** | 🟡 Often | 🟢 Rarely |
| **Official docs images** (NVIDIA, arxiv) | 🟢 Rarely | 🟡 Often | 🔴 **Essential** | 🟢 Rarely | 🟢 Never |
| **Mermaid diagrams** | 🟡 Often | 🟡 Often | 🔴 Always | 🔴 Always | 🟡 Often |
| **Comparison tables** | 🟡 Often | 🔴 Always | 🔴 Always | 🔴 Always | 🟡 Often |
| **Caso real: Company X** | 🟡 Often | 🔴 Always | 🔴 Always | 🔴 Always | 🟡 Often |
| **Código de compresión** | 🔴 Always | 🔴 Always | 🔴 Always | 🔴 Always | 🟢 Rarely |
| **Long inline code blocks** | 🔴 Always | 🟡 Often | 🟡 Often | 🟡 Often | 🟢 Never |

#### Specific Example: How 06/17 Differs from Python Básico

| Aspect | Python Básico (01) | Next-Gen Inference (06/17) |
|--------|-------------------|--------------------------|
| **Primary tool** | Code blocks with inline annotations | Theory + LaTeX + architecture images |
| **Opening hook** | "Por qué esto importa para ML/Backend" | "If you serve 70B models, FP8 halves your GPU bill" |
| **Theory sections** | 1-2 short paragraphs before code | 3-5 deep paragraphs with LaTeX, hardware specifics |
| **Code role** | Teaches syntax and patterns | Demonstrates the concept in a single runnable block |
| **Images per note** | 0-1 (logos, Mermaid) | 2-4 (hardware diagrams, precision layouts, architecture figures) |
| **LaTeX per note** | None | 3-10 equations (MaxSim, attention scores, compression ratios) |
| **Compression Code** | 30-line exercises of all syntax | 15-20 line demonstration of the key algorithm |
| **Warnings** | Language gotchas | Hardware constraints, accuracy tradeoffs, version requirements |

#### The Safety Check: Ask Yourself Before Writing

Before writing any note, answer these 3 questions:

1. **"Can a reader explain WHY this works after reading?"** → If no, add more theoretical foundation.
2. **"Can a reader spot the WRONG way to do this?"** → If no, add a ❌/✅ pair.
3. **"Can a reader run ONE script and see the concept working?"** → If no, add a Código de Compresión.

These 3 questions ensure every note hits the minimum quality bar, regardless of course type.

---

## Project Context

We are building an **Obsidian vault** inside `/home/white/Learning`, a **Git repo** connected to `https://github.com/Leito2/Learning.git`. The vault contains compressed markdown courses for an **AI/ML Engineer** learning path (~600 notes).

**Current Goal:** Maintain and deepen the vault as a job-ready knowledge base. The user is seeking their first ML/AI Engineer role.

**Gap-fill initiative:** ✅ COMPLETE — 22 original gaps filled (50 notes across 12 courses).

**Current total: ~603 notes** — all redistributed into logical numbered modules (00-16).

**Format update initiative:** ✅ COMPLETE — All 4 courses with full old-format (X.1-X.7 template) rewritten to Deep Format: 05/09 TF (7 notes), 06/16 HuggingFace (10 notes), 10/32 System Design (6 notes), 10/33 Vector DBs (12 notes). Plus 14 Rust Engineering: 01 Fundamentals (6), 07 Candle (5), 07 Polars (6) full rewrites. Remaining 70+ notes cleaned of minor old elements (Documented Project, Knowledge Check, ASCII art). Only 03 Advanced Python remains as the canonical new-format reference; all other courses ~600+ notes now consistently follow the new format.

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
├── 05 - Deep Learning y CV/            (44 notes: 18 Spanish + 26 English)
│   ├── 03 - DL con PyTorch         (7)
│   ├── 04 - CV Avanzada            (6)
│   ├── 05 - Multimodal AI          (5)
│   ├── 06 - CV Pipeline            (1 EN)
│   ├── 07 - Reinforcement Learning (7 EN)
│   ├── 08 - Graph Neural Networks  (5 EN)
│   ├── 09 - Deep Learning with TF  (7 EN)
│   └── 10 - JAX Deep Dive          (6 EN)
│
├── 06 - Large Language Models/         (70 notes: 30 Spanish + 40 English)
│   ├── 06 - Fundamentos de LLMs   (7)
│   ├── 07 - Fine-Tuning            (6)
│   ├── 08 - Gen de Texto           (6)
│   ├── 09 - LLMs en Produccion     (6)
│   ├── 10 - Arq Avanzadas/MoE      (5)
│   ├── 11 - Fine-Tuning LLMs       (5 EN)
│   ├── 12 - Production RAG         (6 EN)
│   ├── 13 - vLLM and Advanced RAG  (7 EN)
│   ├── 14 - Unsloth                (5 EN)
│   ├── 15 - LLM Security           (5 EN)
│   ├── 16 - HuggingFace Transf Deep Dive (10 EN)
│   ├── 17 - ColBERT, SGLang and Next-Gen Inference (11 EN)
│   ├── 18 - TensorRT-LLM           (3 EN)
│   └── 19 - LLM Gateway Patterns and LiteLLM (7 EN)
│
├── 07 - AI Agents/                     (34 notes: 22 Spanish + 12 English)
│   ├── 11 - Fundamentos Agentes    (6)
│   ├── 12 - Frameworks             (6)
│   ├── 13 - Multi-Agente           (5)
│   ├── 14 - Agentes Autonomos      (5)
│   ├── 15 - MCP and Agentic Prot   (6 EN)
│   └── 16 - OpenShell and Agent Sandboxes (6 EN)
│
├── 08 - NLP Avanzado/                  (17 notes, Spanish)
│
├── 09 - MLOps/                         (59 notes: 25 Spanish + 34 English)
│   ├── 18 - Experiment Tracking    (7)
│   ├── 19 - Feature Engineering    (6)
│   ├── 20 - Deployment             (6)
│   ├── 21 - Monitoreo              (6)
│   ├── 22 - End-to-End ML          (5 EN)
│   ├── 23 - Advanced MLOps         (1 EN)
│   ├── 24 - Weights and Biases     (5 EN)
│   ├── 25 - Tooling Comparison     (1 EN)
│   ├── 26 - ML Platform Eng        (6 EN)
│   ├── 27 - Feast                  (5 EN)
│   ├── 28 - Testing in ML          (4 EN)
│   ├── 29 - CI-CD for ML           (4 EN)
│   ├── 30 - TorchServe             (4 EN)
│   ├── 31 - Evidently AI and Phoenix (4 EN)
│   ├── 32 - KServe and Knative     (3 EN)
│   └── 33 - Temporal for ML Pipelines (3 EN)
│
├── 10 - Cloud, Infra/                  (60 notes: 20 Spanish + 40 English)
│   ├── 22 - Cloud Computing        (6)
│   ├── 23 - Infrastructure as Code (10 EN)
│   ├── 24 - Backend para ML        (6)
│   ├── 25 - BD y Message Queues    (6)
│   ├── 26 - Databricks for ML      (5 EN)
│   ├── 27 - Apache Spark           (6 EN)
│   ├── 28 - BigQuery               (3 EN)
│   ├── 29 - Distributed ML Infra   (7 EN)
│   ├── 30 - WebSockets             (5 EN)
│   ├── 31 - FastAPI for ML         (6 EN)
│   ├── 32 - System Design for ML   (6 EN)
│   └── 33 - Vector Databases and Semantic Search (12 EN)
│   └── 34 - DuckDB            (4 EN)
│   └── 35 - Vector Quantization and Approximate Nearest Neighbors (6 EN)
│   └── 36 - PostgreSQL for AI-ML Workloads (6 EN)
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
├── 16 - Harness Engineering/           (10 notes, English)
│   ├── 00 - Welcome to Harness Engineering and SDD
│   ├── 01 - The Context Crisis: Why AI Development Fails
│   ├── 02 - The Three Pillars: Context, Harness, SDD
│   ├── 03 - Harness Engineering: Architecture of Control
│   ├── 04 - Specification-Driven Development: The Formal Method
│   ├── 05 - File Architecture: The Structural Foundation
│   ├── 06 - Multi-Agent Orchestration and Capstone
│   ├── 07 - Complete Harness Taxonomy
│   ├── 08 - Verification and Quality Gates
│   └── 09 - Tools, Provider Abstraction, and Memory
│
├── Extra/                              # Bun Runtime + cross-cutting topics
└── projects/                           (15+ guides)
```

---

## Language Policy

- **Modules 00-02 (Markdown, SQL, Docker), 03 (Advanced Python), 04-12:** Spanish. **Do NOT modify** unless requested.
- **New courses integrated into 05-12 (English-labeled sub-modules), Go Engineering (13), Rust Engineering (14), Transversal Skills (15), Harness Engineering (16), and projects:** **English**.
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
| **SGLang** | ✅ (06/17) | RadixAttention for structured output. Deep course: ColBERT, SGLang, Next-Gen Inference (FP8/MLA/Edge). |
| **TensorRT-LLM** | ✅ (06/18) | NVIDIA's max-throughput engine. Model compilation, builder optimization, Triton integration, multi-GPU, benchmarking. Compare vLLM/SGLang. |
| **Unsloth** | ✅ (06/14) | Fine-tuning 2-5x faster, 80% less memory. Deep course covering QLoRA, SFT, DPO, deployment. |

### Advanced RAG (partially created)
| Tech | Status | Why |
|------|:------:|-----|
| **Hybrid Search (BM25 + Dense)** | ✅ (06/13) | Redis + vector search. Already used in LLM Gateway project. |
| **Reranking (Cohere, bge-reranker)** | ✅ (06/13) | Second-stage precision. Critical for production RAG accuracy. |
| **GraphRAG** | ✅ (06/13) | Microsoft. Multi-hop reasoning over knowledge graphs. |
| **RAGAS / DeepEval** | ✅ (06/13) | RAG quality evaluation. Bridges to LLM Evaluation Suite project. |
| **ColBERT (late interaction)** | ✅ (06/17) | Token-level retrieval. State-of-the-art for passage search. Deep course with PLAID indexing, hybrid RAG capstone. |
| **Late Chunking (Jina)** | ✅ (06/13/06) | Context-aware chunking. 1 note added to Advanced RAG course: late embed → segment vs chunk → embed. |

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
| **Knative / KServe** | ✅ (09/32) | Serverless model serving on K8s. InferenceService CRD, canary, scale-to-zero, Knative Eventing, event-driven ML. |
| **Temporal** | ✅ (09/33) | Durable execution for ML pipelines. Workflows, activities, human-in-the-loop approval, crash-survivable training orchestration. |

### Real-Time ML & Streaming
| Tech | Status | Why |
|------|:------:|-----|
| **WebSockets for ML Serving** | ✅ (10/30) | Deep course: protocol internals, real-time inference, scaling, ML Gateway. Complements Go/Fiber/Sudoku Together. |
| **Redis Pub/Sub + WebSocket scaling** | ✅ (10/30) | Horizontal scaling of WS connections. Natural extension of LLM Gateway. |
| **Server-Sent Events (SSE)** | ✅ (Go Local AI) | Token streaming. Covered but could deepen. |

### Data & Feature Engineering
| Tech | Status | Why |
|------|:------:|-----|
| **DuckDB** | ✅ (10/34) | OLAP in-process. Fast analytics in Python/Go without Spark. Deep course covering SQL, Python/Polars integration, RAG preprocessing, ML pipelines. |
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
| ~~6~~ | ~~ColBERT, SGLang and Next-Gen Inference~~ | ~~11~~ | ✅ REWRITTEN (06/17). 5 cutting-edge vectors: Inference-Time Scaling, Speculative Decoding 2.0, MLA + KV eviction, FP8 hybrid precision, Disaggregated serving + Edge inference. Plus ColBERT + SGLang deep dives. Capstone: hybrid RAG.
| ~~9~~ | ~~SDD: AI-Assisted Project Architecture~~ | ~~11~~ | ✅ REWRITTEN AS **Harness Engineering** (16/00-09, 10 notes, 4,132 lines). Deep theory rewrite from 6 video sources: Gentle Framework, agent loop, provider abstraction, SDD formal method, 20-harness taxonomy, verification gates, tool/memory integration. Removed 424 lines of Python compression code, zero Spanish.
| 10 | **Vector Databases and Semantic Search** | 12 | ✅ CREATED (10/33). pgvector, Qdrant, Milvus deep courses + comparison + capstone with Go and Python.
| 11 | **HuggingFace Transformers Deep Dive** | 10 | ✅ CREATED (06/16). from_pretrained, tokenizers, Trainer, generation, vision/audio/multimodal, optimum, Diffusers (2 notes), capstone.
| 12 | **Deep Learning with TensorFlow** | 7 | ✅ CREATED (05/09). tf.keras architectures, tf.data/TFRecord, distributed training (TPU/GPU), TensorBoard/callbacks/tuning, SavedModel/TF Serving/TFLite, CV capstone with EfficientNet.

### Remaining High-Priority Gaps (4 courses)

| # | Course | Est. Notes | Justification |
|:--:|--------|:-----:|---------------|
| 13 | **JAX Deep Dive** | 6 | ✅ CREATED (05/10). jit/vmap/grad/XLA fundamentals, NumPy→JAX functional paradigm, autodiff (grad/vjp/jvp/Hessians), Flax/Linen, Optax training loops. Fills "Present but shallow" gap. |
| 14 | **TorchServe** | 4 | ✅ CREATED (09/30). Architecture, MAR files, model archiver, custom handlers, multi-model endpoints, Docker/K8s deployment, performance tuning, monitoring. Bridges PyTorch training to production. |
| 15 | **Evidently AI / Phoenix** | 4 | ✅ CREATED (09/31). Data drift (KS, JS, Wasserstein, PSI), concept drift, Evidently Reports/Test Suites/CI-CD, Phoenix LLM observability, spans/traces, embedding drift (UMAP), RAG evaluation. Connects to LLM Evaluation Suite project. |
| 16 | **DuckDB** | 4 | ✅ CREATED (10/34). In-process OLAP, SQL analytics, Python/pandas/Polars/Parquet integration, RAG preprocessing, ML feature engineering, Go bindings for ML Gateway. |
| 17 | **LLM Gateway Patterns and LiteLLM** | 7 | ✅ CREATED (06/19). Multi-provider abstraction, LiteLLM unified interface, Router with cost/latency/quality routing, fallback chains, observability + cost tracking, self-hosted proxy with SSO/spend limits, capstone multi-provider RAG gateway. |
| 18 | **Vector Quantization and ANN** | 6 | ✅ CREATED (10/35). PQ/OPQ deep dive, anisotropic VQ + ScaNN, BQ + RaBitQ (2024 frontier), production FAISS engineering (index factories, sharding, GPU), capstone billion-vector search. |
| 19 | **PostgreSQL for AI-ML Workloads** | 6 | ✅ CREATED (10/36). pgvector production tuning (halfvec/bit/iterative scan), pgvector vs Qdrant/Milvus cost equation, pgvectorscale + DiskANN, LISTEN/NOTIFY + pg_stat_statements + CDC, capstone feature store. |

---

> **⚠️ MANDATORY for next session:**
> 1. Read the "CRITICAL: CONTEXT MANAGEMENT" section.
> 2. ALWAYS use subagents (`task` tool) for bulk creation — max 2 parallel, max 7 notes each.
> 3. Never write 5+ full course notes in the main thread.
> 4. Verify filesystem state after every subagent batch.
> 5. **Reorganized:** 00=Markdown, 01=SQL, 02=Docker, 03=Advanced Python, 04=Engineering Fundamentals, 05-12=ML core, 13=Go, 14=Rust, 15=Transversal, 16=Harness Engineering, Extra/ at end.
> 6. Next priority: **ALL GAPS COMPLETE.** 🎉 **Zero 🚨 remaining.** Entire High-Value Tech scan is ✅.
> 7. **Course 06/17 REWRITTEN:** 11 notes, 4,735 lines. 5 next-gen vectors.
> 8. **JAX Deep Dive:** 6 notes, 2,363 lines (05/10).
> 9. **TorchServe:** 4 notes, 1,707 lines (09/30).
> 10. **Evidently/Phoenix:** 4 notes, 1,100 lines (09/31).
> 11. **DuckDB:** 4 notes, 1,342 lines (10/34).
> 11b. **LLM Gateway Patterns (06/19):** 7 notes, 2,848 lines. LiteLLM as Python multi-provider standard. Cross-links: 13/06/06 (Go gateway), 06/17/10 (ColBERT capstone), 10/30 (WebSockets).
> 11c. **Vector Quantization (10/35):** 6 notes, 2,500 lines. Deepens 10/33/02 (PQ/OPQ/DiskANN/ScaNN intro). Adds RaBitQ, BQ, SQ, FAISS engineering.
> 11d. **PostgreSQL for AI/ML (10/36):** 6 notes, 1,972 lines. English counterpart to 10/25/01 (Spanish general). Deepens 10/33/03-04 (pgvector basics). Adds pgvectorscale/DiskANN, halfvec/bit, CDC.
> 12. **IaC REWRITTEN:** 10 notes, 4,478 lines (10/23).
> 13. **TensorRT-LLM:** 3 notes (06/18)  |  **KServe+Knative:** 3 notes (09/32)  |  **Temporal:** 3 notes (09/33)  |  **Late Chunking:** 1 note (06/13/06).
> 14. **Format Update — Full Rewrites:** 05/09 TF (7→2,188 lines), 06/16 HF (10→3,370), 10/32 System Design (6→1,913), 10/33 Vector DBs (12→3,834), 14/01 Rust Fundamentals (6→1,716), 14/07 Candle (5→1,585), 14/07 Polars (6→1,684). **Minor cleanups:** 04 Eng Fund (23 notes), 05/04 CV (6), 09/18 Exp Track (7), 10/25 BD (6), 14 Rust (45 notes), 15 Transversal (3) — removed Documented Project, Knowledge Check, ASCII art.
> 15. Portfolio URL: https://white-portfolio-ia-ml-engineer.netlify.app/
> 16. **OpenShell and Agent Sandboxes (07/16):** 6 notes, 2,301 lines, English. NVIDIA OpenShell = safe runtime for autonomous agents. 4-layer defense (FS/Network/Process/Inference), Rust 88.9%, gateway+sandbox+policy_engine+privacy_router. Agents integrated: Claude Code, OpenCode, Codex, Copilot, Deep Agents (LangChain), Hermes, OpenClaw, Ollama, Pi. Cross-links: 07/15 MCP, 06/15 LLM Security, 06/17 ColBERT/SGLang, 16 Harness Engineering, 14 Rust, 09/32 KServe, 13/06 LLM Edge Gateway.

---

## 📋 Pendientes y Próximas Acciones

> Tracking de tareas recurrentes. Las ✅ están completadas, las 🟡 están abiertas para decisión del usuario, las 🔵 son sugerencias para sesiones futuras.

### ✅ Completadas en la sesión actual

| # | Tarea | Detalle | Archivos creados |
|---|-------|---------|------------------|
| 1 | Crear 3 cursos nuevos 2026 | LLM Gateway Patterns, Vector Quantization, PostgreSQL for AI-ML | 19 notes, 7,320 lines |
| 2 | Banner SVG para Markdown | Welcome banner con sintaxis coloreada, logo M, badges | `00 - Curso Markdown/markdown-course-banner.svg` (5.4 KB) |
| 3 | Banner SVG para SQL/PostgreSQL | Welcome banner con elefante estilizado, query SQL | `01 - Curso SQL con PostgreSQL/sql-postgres-course-banner.svg` (5.7 KB) |
| 4 | Banner SVG para Docker | Welcome banner con 3 containers apilados, FROM/CMD | `02 - Docker Profesional/docker-course-banner.svg` (6.4 KB) |
| 5 | Actualizar Master Index | 3 entradas añadidas a tree map + course tables | `00 - Indice Maestro de Cursos.md` |
| 6 | Actualizar Continuity Prompt | Vault structure + course list + mandatory list + este tracking | Este archivo |
| 7 | **Crear curso OpenShell and Agent Sandboxes (07/16)** | 6 notes, 2,301 lines. Welcome + Agent Security Crisis + Architecture + YAML Policies + Agent Integrations + Production Deployment & Capstone. Cross-links a MCP/A2A, LLM Security, Harness Engineering, KServe, Rust, LLM Edge Gateway. | `07 - AI Agents y Agentic Systems/16 - OpenShell and Agent Sandboxes/` (6 files) |
| 8 | **Actualizar Master Index con OpenShell** | Tree map + course table | `00 - Indice Maestro de Cursos.md` |

### 🟡 Pendientes (esperando decisión del usuario)

| # | Tarea | Contexto | Bloqueante |
|---|-------|----------|------------|
| P1 | **Banners para los otros cursos principales** | Usuario preguntó si los quería para Python, Go, Rust, FastAPI, etc. tras crear los 3 primeros | Esperando `sí` o `no` |
| P2 | **Banners para los nuevos cursos 2026 (06/19, 10/35, 10/36)** | Los 3 cursos recién creados también tienen Welcome notes sin banner | Esperando `sí` o `no` |
| P3 | **Commit de los cambios** | 19 notes nuevos + 3 SVGs + 2 index updates sin commitear | Esperando `commit -m "..."` |
| P4 | **Decisión sobre 06/19 capstone (659 líneas)** | Subagent reportó 159 líneas sobre el target. Contenido justificado (7 módulos de código) pero excede guideline | Esperando `aceptar` o `recortar` |

### 🔵 Sugerencias para futuras sesiones

| # | Idea | Justificación |
|---|------|---------------|
| S1 | **Banners para los 3 cursos nuevos 06/19, 10/35, 10/36** | Mantener consistencia visual con SQL/Docker/Markdown |
| S2 | **Crear templates de banner SVG** | Para cursos futuros: 3 paletas (cyan/azul, verde, naranja) reutilizables |
| S3 | **Auditoría visual del vault** | Revisar 17 módulos, identificar cuáles merecen banner (welcome notes) |
| S4 | **Próximos cursos candidatos** (ver tier 2/3 del scan original) | OpenTelemetry, LangSmith, DSPy, Mamba-2, WebGPU |
| S5 | **Traducir 06/19 a español selectivamente** | Si audiencia en Colombia lee inglés, mantener EN; si no, traducir |

### 📐 Plantilla reusable de banner SVG (paletas)

Si en sesión futura se piden más banners, usar este patrón:

```
Template: 1200×360 SVG self-contained
- Gradiente azul oscuro + glow radial
- Tag "CURSO COMPRIMIDO" → título grande → underline gradiente
- Tagline + subtítulo monoespaciado
- Bloque de código coloreado (sintaxis del tema)
- 4 badges (X notas · Idioma · Característica 1 · Característica 2)
- Ícono temático (lado derecho)
- Barra gradiente inferior (4px)

Paletas disponibles:
- Cyan/Indigo: #38bdf8 → #818cf8 (Markdown, Genérico)
- Azul Postgres: #336791 → #7dd3fc (SQL, Data)
- Azul Docker: #0db7ed → #1d4ed8 (DevOps, Infra)
- Verde: #34d399 → #10b981 (ML, AI)
- Naranja: #fb923c → #f97316 (Frameworks, Tools)
- Púrpura: #a78bfa → #7c3aed (Agents, Protocolos)
- Rojo: #f87171 → #dc2626 (Security, Performance)
```

### 📦 Convenciones de archivos para banners

| Regla | Detalle |
|-------|---------|
| **Ubicación** | Mismo directorio que el Welcome note (sibling) |
| **Naming** | `<course-slug>-course-banner.svg` (kebab-case, sufijo `-banner`) |
| **Embed syntax** | `![Banner del Curso X](<slug>-course-banner.svg)` (relative path) |
| **Tamaño objetivo** | 4-7 KB (self-contained, sin assets externos) |
| **Self-contained** | Solo system fonts (ui-sans-serif, ui-monospace) — sin Google Fonts, sin imágenes linkeadas |
| **Mermaid en vez de SVG** | Solo si el contenido es arquitectura/flujo (no bienvenida estática) |
