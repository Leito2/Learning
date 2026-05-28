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
├── 06 - Large Language Models/         (66+ notes: 30 Spanish + 36 English)
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
│   └── 17 - ColBERT, SGLang and Next-Gen Inference (11 EN)
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
| **SGLang** | ✅ (06/17) | RadixAttention for structured output. Deep course: ColBERT, SGLang, Next-Gen Inference (FP8/MLA/Edge). |
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
| ~~6~~ | ~~ColBERT, SGLang and Next-Gen Inference~~ | ~~11~~ | ✅ REWRITTEN (06/17). 5 cutting-edge vectors: Inference-Time Scaling, Speculative Decoding 2.0, MLA + KV eviction, FP8 hybrid precision, Disaggregated serving + Edge inference. Plus ColBERT + SGLang deep dives. Capstone: hybrid RAG.
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
> 6. Next priority: **Complete rewrite of 06/17** (11 notes total: restructure 00-04 + 06, write 05-09 new notes with cutting-edge vectors) OR **JAX Deep Dive** (5-6 notes) OR **TorchServe** (3-4 notes).
> 7. **Course 06/17 REWRITE IN PROGRESS:** ColBERT, SGLang and Next-Gen Inference — expanding from 7 to 11 notes. New format: no ASCII art, real images, LaTeX, organic structure. Cutting-edge vectors: Inference-Time Scaling, Speculative Decoding 2.0 (Eagle/MTP), MLA + H2O/StreamingLLM, FP8 (E4M3/E5M2) + Transformer Engines, Disaggregated serving + Edge (ExecuTorch/NPU).
> 8. Portfolio URL: https://white-portfolio-ia-ml-engineer.netlify.app/
