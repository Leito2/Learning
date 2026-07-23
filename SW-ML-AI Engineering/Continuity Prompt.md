# Continuity Prompt for Next Session

Copy and paste this entire file into a new session to continue the project without losing context.

---

## ⚠️ CRITICAL: CONTEXT & COST MANAGEMENT

> **Last revised:** June 2026. The vault now has **~628+ notes** (4 new courses added). The previous "always-subagent-for-3+-notes" rule was redesigned around **AI subscription cost**, not just context saturation. Subagents reload the entire Deep Format spec, vault structure, and writing rules in their own context — that duplication is **expensive**. The new default inverts the priority.

### Core Philosophy

> **Main agent writes courses. Subagents exist for two specific reasons only: (1) cheap research, and (2) parallel bulk tasks that justify the context reload cost.**

### Decision Matrix — When to Use What

| Task | Tool | Why |
|------|------|-----|
| **1-3 notes** (single course, small course, or 1-3 note patch) | **Main thread** (Write/Edit) | Cheapest path. The Deep Format spec is already in context. 1 note ≈ 400 lines, well within main budget. |
| **4-6 notes** (typical course) | **Main thread, 1 note at a time** | Each Write call commits a complete, verified note. Pause for verification after each. Total: 4-6 sequential Writes, not parallel. |
| **7+ notes** (long course like 06/17 with 11 notes) | **Main thread, 1 note at a time** OR **2 subagents parallel (3-4 each)** | Main thread is still cheaper. Use subagents only if user wants speed and accepts the cost. Default to main. |
| **Research / codebase exploration** (find files, search, read structure) | `task` tool with `subagent_type: explore` | Cheap because `explore` is a fast, low-token agent. Worth it for any multi-step lookup. |
| **Bulk simple parallel tasks** (e.g., 4+ SVGs, 4+ identical file structures, batch file generation with no shared context) | **`task` tool, 2-4 subagents parallel** | Each task is self-contained, no shared spec needed. The context reload is justified by the parallelism. **This is the primary cost-justified subagent use case.** |
| **Single-note edit** (typo fix, banner embed, small content patch) | **Main thread** (Edit tool) | Trivial, no subagent needed. |
| **Index / metadata update** (Master Index, Continuity Prompt, Skills Tree) | **Main thread** (Edit tool) | Trivial, no subagent needed. |

### Hard Rules

- **Default to main thread** for all content creation. Subagents are exceptions, not the rule.
- **Never write more than 1 note in a single main-thread turn.** Write one, verify, then write the next. This keeps each turn focused and reversible.
- **Subagent context reload is expensive.** Estimate: a subagent task prompt ≈ 2-5× the cost of an equivalent main-thread action because the subagent re-loads rules, examples, and vault structure. Only use subagents when **the parallelism savings beat the reload cost**.
- **Max 2 subagents parallel** when subagents are used (the previous hard cap holds for token-bill reasons).
- **Filesystem is the single source of truth.** Whether you used a subagent or the main thread, verify with `ls`/`wc -l` after every batch.

### Subagent Prompt Design (Cost Minimization)

When you DO use a subagent, design the prompt to **minimize reloaded context**:

| DO | DON'T |
|----|-------|
| Reference existing notes by exact path (`[[../15 - MCP/...]]`) | Re-explain the whole Deep Format spec in the prompt |
| Include only the **target file path + 1-2 line content scope** | Re-paste the 700+ line Continuity Prompt |
| Give 1 inline example of the desired output style | Re-paste multiple full course notes as reference |
| Specify the file naming, line target, and "do not modify siblings" | Re-explain vault structure, language policy, etc. |
| Reference `CONTEXT MANAGEMENT` section in the Continuity Prompt if the subagent can read it | Duplicate the rules verbatim |

**Rule of thumb:** if the subagent prompt is > 2,000 words, you've reloaded too much. The Deep Format spec, vault structure, and language policy are all already in `Continuity Prompt.md` and `00 - Indice Maestro de Cursos.md` — the subagent can read those files itself.

### Verification (After Every Subagent Batch)

```bash
ls -la "/path/to/course/"              # confirm file count
wc -l "/path/to/course/"*.md           # confirm line counts in target range
head -5 "/path/to/course/00 - ...md"   # confirm H1 + format correct
```

### Anti-Patterns to Avoid

| Anti-pattern | Why it's bad | Replacement |
|--------------|--------------|-------------|
| "Use 2 subagents to write 3 notes each" for a 6-note course | 6 subagent context reloads + 6 main-thread verifications = 12× the cost of just doing it in main | Write 6 notes in main, 1 at a time, verifying after each |
| One subagent that writes a full course with full Deep Format spec re-pasted | Subagent consumes more tokens than main would have | Main thread, with Continuity Prompt as the live reference |
| Subagent for a single 400-line note | Zero parallelism benefit, full context reload | Main thread, one Write call |
| Subagent for research that's "let me check if file X exists" | Just `ls` or `glob` in main | Main thread, direct tool call |

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
├── 10 - Cloud, Infra/                  (64 notes: 20 Spanish + 44 English)
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
│   └── 33 - Vector Databases and Semantic Search (17 EN)
│   └── 34 - DuckDB            (4 EN)
│   └── 35 - Vector Quantization and Approximate Nearest Neighbors (6 EN)
│   └── 36 - PostgreSQL for AI-ML Workloads (6 EN)
│   └── 37 - Vector Search on Google Cloud (3 EN)
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
> 1. Read the "⚠️ CRITICAL: CONTEXT & COST MANAGEMENT" section (rewritten June 2026 — main-agent default).
> 2. **Default to main thread for course creation.** Write 1 note at a time, verify, then write the next. Subagents are reserved for cheap research (`explore`) and bulk parallel simple tasks (e.g., 4+ SVGs).
> 3. Never write 5+ full course notes in a single main-thread turn.
> 4. Verify filesystem state after every subagent batch AND after every main-thread note (`ls`/`wc -l`).
> 5. **Reorganized:** 00=Markdown, 01=SQL, 02=Docker, 03=Advanced Python, 04=Engineering Fundamentals, 05-12=ML core, 13=Go, 14=Rust, 15=Transversal, 16=Harness Engineering, 17=OpenShell, Extra/ at end.
> 6. **ALL GAPS COMPLETE.** 🎉 **Zero 🚨 remaining.** Entire High-Value Tech scan is ✅.
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
> 17. **FSDP Deep Dive (06/16/10):** 1 note, 412 lines, English. Fills the FSDP coverage gap (previously mentioned only as a 1-paragraph peer of ZeRO in note 03). Covers: 4 `ShardingStrategy` modes with per-GPU memory math, all-gather/reduce-scatter communication pattern, Mermaid sequence diagram, `auto_wrap_policy` (transformer vs size-based), `MixedPrecision`, `cpu_ram_efficient_loading`, **`use_orig_params=True` (mandatory for LoRA/PEFT)**, FSDP2 (`fully_shard` + DTensor, PyTorch 2.4+), YAML config file deployment path (`fsdp_config="fsdp_config.yaml"`), Llama 3 70B on 16xH100 recipe, full Compression Code with Mermaid + YAML + Python. Cross-links: 06/16/03, 06/11, 06/14 Unsloth, 06/13 vLLM.
> 18. **NumPy + Pandas (03/04/08-09):** 2 notes, 707 lines, Spanish. Fills the test-prep gap (vault had 0 dedicated notes for NumPy/Pandas; only scattered 1-line mentions in 04/00, 05/06, projects/01). Note 08 NumPy: array/shape/ndim/dtype, axis=0/1 explained, broadcasting rules with 3 examples, `np.random.seed/randn/randint`, boolean indexing, `np.dot/@`, performance comparison list vs array (16-100×), compression code with synthetic 3-group dataset. Note 09 Pandas: `read_csv`, `head/info/describe`, boolean indexing with `&`/`|`/`~`, **`groupby().agg()`** with named aggregation, `fillna` with median (the 80% rule), `merge` (how=inner/left/right/outer), `concat`, `value_counts`, `sort_values`, `apply` vs vectorized (100-1000× speedup), `to_numpy()` bridge to ML. Both notes are test-prep level (Anyone AI / ML engineer admission). Cross-links: 03/04/01 Math, 03/04/05 Json, 05/06 CV Pipeline, projects/01 Kaggle, 04/00/07 Streaming Pipeline. First real validation of the new "main-agent-default" subagent rules (2 main-thread notes, 0 subagents).
> 19. **Qdrant Python Client Deep Dive (10/33/12):** 4 notes, 1,892 lines, English. New sub-module `12 - Qdrant Python Client Deep Dive`. Closes the gap identified in audit: vault had strong fundamentals (10/33/05 + 10/33/06, 1,477 lines) but **zero coverage of advanced client APIs** (local mode, named vectors, quantization config, scroll, batch, count, facet, recommend) and **zero production async patterns** (FastAPI lifecycle, retry with backoff, batching, OpenTelemetry, graceful shutdown). Note 00 Welcome (97 lines) — course map + why it exists. Note 01 Advanced Client APIs (511 lines) — `QdrantClient(path=...)` local mode for tests, named vectors (dense + ColBERT + BM25 in one point), `ScalarQuantization(int8, always_ram=True)` with `rescore=True` for 4× memory reduction, `scroll` pagination, `query_batch_points`, `count`, `facet`, `recommend` vs `query_points` API migration, `MultiVectorConfig(MAX_SIM)` for ColBERT. Note 02 Production Async Patterns (637 lines) — `AsyncQdrantClient` for FastAPI (sync kills concurrency), module-level singleton via `lifespan`, `prefer_grpc=True` + `pool_size=10`, full-jitter backoff with circuit breaker, `upload_points(parallel=4)` for 10K-50K pts/sec, `openinference-instrumentation-qdrant` for Phoenix traces, liveness vs readiness (liveness independent of Qdrant), background-writer pattern for graceful SIGTERM. Note 03 Capstone (647 lines) — full production RAG stack: FastAPI + AsyncQdrantClient + LiteLLM streaming + Phoenix traces; multi-tenant via `team_id` payload filter (not separate collections), query rewriting with RAG-Fusion, hybrid search + Reciprocal Rank Fusion, cross-encoder reranking, SSE streaming, full Docker Compose stack, in-memory test fixture, end-to-end latency budget table. Cross-links: 10/33/05, 10/33/06, 10/33/10, 10/33/11, 10/35/01, 06/19/06, 09/31/03. **3rd validation of the new main-agent-default subagent rules: 4 main-thread notes, 0 subagents, ~1,900 lines.**
> 20. **Pinecone + Vertex AI Vector Search (10/33/12 + 10/37/00-02):** 4 notes, 1,349 lines, English. Closes the two critical gaps identified in the audit: **Pinecone** (0 dedicated notes, only comparison tables) and **Vertex AI Vector Search** (0 dedicated notes, only 1-2 line mentions). Pinecone note 10/33/12 (428 lines) — pod types (p1/p2/s1/p3) + Serverless pricing, namespaces for multi-tenancy, metadata filters ($eq/$gte/$in), hybrid search with alpha, Pinecone Inference (server-side embedding + reranking), 10K-vector demo. Vertex AI note 10/37/00-02 (921 lines) — 3-note mini-course: Welcome + decision tree (105 lines), Vertex AI Vector Search note 01 (391 lines) — Index/IndexEndpoint/DeployedIndex 3-level model, TREE_AH (ScaNN) vs BRUTE_FORCE, streaming vs batch update, hybrid search (2024), public vs private endpoint, IAM/VPC-SC/CMEK, 5-15 min cold start, AlloyDB AI + BigQuery VECTOR_SEARCH() + decision framework (425 lines). All in main thread, 0 subagents. Cross-links: 10/33/05, 10/33/09, 10/36, 10/28 BigQuery, 09/27 Feast, 06/19 LLM Gateway.

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
| 9 | **Generar 4 banners SVG en paralelo (subagents)** | 06/19 LLM Gateway + 10/35 Vector Quantization + 10/36 PostgreSQL AI/ML + 07/16 OpenShell | 4 SVG files, 5.8-7.3 KB each |
| 10 | **Embed de banners en 4 Welcome notes** | Añadir `![Banner del Curso X](<slug>-course-banner.svg)` como primera línea de cada Welcome | 4 markdown files |
| 11 | **Commit masivo** | 25 notes nuevos + 7 SVGs + 6 metadata updates (commit `0fae30e`) | `0fae30e` |
| 12 | **Reescribir reglas de subagentes en este archivo** | Filosofía: main agent por defecto, subagentes solo para investigación barata o tareas paralelas masivas. Reglas de prompt-design para minimizar context reload | Este archivo (líneas 7-65) |
| 13 | **Push a origin** | `git push origin master` — 3 commits ahead → 0 ahead | `8724a83` en origin |
| 14 | **FSDP coverage audit** | Hallazgo: FSDP solo mencionado (13 veces en 06/16/03 como peer de ZeRO, 0 profundidad). Plan: 1 nota deep en 06/16/10. | Ver respuesta del chat |
| 15 | **Escribir 10 - FSDP Deep Dive (main thread, 1 Write call)** | 412 líneas, sigue Deep Format. Teoría: memoria + comunicación. Integración: Trainer + `fsdp_config` YAML. FSDP2 + LoRA/PEFT. Llama 3 70B recipe. Compression Code completo. | `06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/10 - FSDP Deep Dive - PyTorch Native Sharded Data Parallel.md` (412 líneas) |
| 16 | **Actualizar 06/16 Welcome note** | Course map añade nota 10 con descripción | `00 - Welcome to HuggingFace Transformers Deep Dive.md` |
| 17 | **Audit de Advanced Python vs prueba técnica de IA** | 4 temas de la prueba: (1) Fundamentos ✓, (2) Colecciones ✓, (3) I/O ✓ con gap en `csv` stdlib, (4) NumPy/Pandas 🔴 gap crítico (0 dedicated notes, solo 1-2 line mentions) | — |
| 18 | **Escribir 08 - NumPy para Análisis de Datos** | 325 líneas, main thread, 1 Write call. Cubre array, shape, axis, broadcasting, random, rendimiento. Caso real: scoring crediticio 50× más rápido. | `03 - Advanced Python/04 - Librerias Basicas de Python/08 - NumPy para Analisis de Datos.md` |
| 19 | **Escribir 09 - Pandas para Análisis de Datos** | 382 líneas, main thread, 1 Write call. Cubre read_csv, groupby.agg, fillna, merge, value_counts. Caso real: Automated LLM Evaluation Suite. | `03 - Advanced Python/04 - Librerias Basicas de Python/09 - Pandas para Analisis de Datos.md` |
| 20 | **Actualizar 04 Welcome note** | Course map + glosario de librerías (numpy + pandas añadidos) | `00 - Bienvenida.md` |
| 21 | **Audit de vector DB coverage** | Qdrant: 1,477 líneas en fundamentals, mejor cubierto. Pinecone: 0 dedicated notes, solo comparison tables. Vertex AI Vector Search: 0 dedicated notes, critical gap. Plan: 2 dedicated notes (Pinecone + Vertex AI) | — |
| 22 | **Escribir 4 notas del sub-módulo 12 - Qdrant Python Client Deep Dive (10/33)** | 1,892 líneas, 4 main-thread writes, 0 subagents. Welcome + Advanced Client APIs + Production Async Patterns + Capstone RAG. Cierra el gap identificado en el audit (advanced APIs + production async patterns). 3ª validación de las reglas main-agent-default. | `10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/` (4 files) |
| 23 | **Actualizar 10/33 Welcome note + Master Index** | 10/33 Welcome: añadir nota 12. Master Index: 10/33 → 16 notas (+4). | `00 - Welcome to Vector Databases and Semantic Search.md`, `00 - Indice Maestro de Cursos.md` |
| 24 | **Ejecutar plan de Pinecone + Vertex AI (4 notas pendientes)** | Pinecone 10/33/12 (428 lines) + Vertex AI 10/37/00-02 (921 lines) = 1,349 lines, 4 main-thread writes, 0 subagents. Cierra los 2 gaps críticos del audit. | `10/33/12 - Pinecone Architecture and Python Client.md`, `10/37/00, 01, 02` (3 archivos) |
| 25 | **Actualizar Master Index + Continuity Prompt + Vault structure** | Master Index: 10/33 → 17 notas (+Pinecone), nuevo curso 10/37 (3 notas). Continuity Prompt: MANDATORY item 20 + Completadas items 24-25. | `00 - Indice Maestro de Cursos.md`, `Continuity Prompt.md` |
| 26 | **Crear curso Production Agent Frameworks 2026 (07/17) — Plan A** | 8 notas, 3,697 líneas, English. Welcome + 6 frameworks (smolagents, PydanticAI, transformers.agents, OpenAI Agents SDK, Google ADK, CrewAI 1.0) + Capstone RAG. Main thread, 8 sequential Writes, 0 subagents. Notas 04-07 vinieron sobre target (554-638 vs 380) por la profundidad de los Flow APIs y el capstone integration. | `07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks/` (8 files) |
| 27 | **Actualizar Master Index + Continuity Prompt** | Master Index: nuevo curso 17 (8 notas). Vault stats: 61 cursos, 619+ notas. Continuity Prompt: status Plan A → ✅ COMPLETE con variance note. | `00 - Indice Maestro de Cursos.md`, `Continuity Prompt.md` |
| 28 | **Crear curso LangGraph Deep Patterns (07/18) — Plan F** | 10 notas, 3,915 líneas, English. Welcome + StateGraph Fundamentals + Conditional Routing + Persistence (PostgresSaver) + Subgraphs/Send API + Human-in-the-Loop + Streaming Modes + Advanced Patterns + Production Deployment + Capstone. Cierra el gap CRÍTICO: vault solo tenía 1 nota dedicada a LangGraph (07/15/03). Main thread, 10 sequential Writes, 0 subagents. | `07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/` (10 files) |
| 29 | **Crear curso Modern Python Typing (03/07) — Plan G** | 8 notas, 3,191 líneas, English. Welcome + PEP 695 Type Parameters + Type Narrowing (TypeGuard) + Protocols + Self/ParamSpec/Concatenate + Mypy Strict + Runtime Checkers (Pydantic/Beartype/Cattrs) + Capstone. Cierra el gap de tipado moderno (PEP 695, mypy strict, runtime checkers). Main thread, 8 sequential Writes, 0 subagents. | `03 - Advanced Python/07 - Modern Python Typing/` (8 files) |
| 30 | **Crear curso ChromaDB Deep Dive (10/33/13) — Plan H** | 6 notas, 2,485 líneas, English. Welcome + Fundamentals + Server Mode + Custom Embeddings + Metadata Filtering + Migration Path. Cierra el gap: vault tenía 0 dedicated notes para Chroma (solo comparaciones). Main thread, 6 sequential Writes, 0 subagents. | `10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/13 - ChromaDB Deep Dive/` (6 files) |
| 31 | **Crear curso RAG Evaluation Deep Dive (06/20) — Plan I** | 8 notas, 3,561 líneas, English. Welcome + Test Dataset Construction + Custom Metrics + Statistical Rigor + LLM-as-Judge Bias + CI/CD Eval Pipelines + Cost-Optimized Evaluation + Capstone. Cierra los 7 gaps de operacionalización de RAGAS que el audit identificó (sample size, custom metrics, statistical tests, judge bias, CI gates, cost). Main thread, 8 sequential Writes, 0 subagents. | `06 - Large Language Models/20 - RAG Evaluation Deep Dive/` (8 files) |
| 32 | **Crear curso OpenTelemetry for AI Engineers (09/34) — Plan J** | 8 notas, 3,329 líneas, English. Welcome + Primitives (Spans, Traces, Context) + Auto-Instrumentation for LLM SDKs + OTLP Exporters (Phoenix/Tempo/Jaeger) + OTel for LangGraph + OTel for RAG + Production Patterns (Sampling, Costs, PII) + Capstone multi-service RAG. Cierra el gap crítico: vault tenía solo menciones parciales de OTel. Main thread, 8 sequential Writes, 0 subagents. | `09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/` (8 files) |
| 33 | **Crear curso DSPy and Prompt Compilation (06/21) — Plan K** | 8 notas, 3,093 líneas, English. Welcome + Signatures and Modules + Optimizers (BootstrapFewShot, MIPRO, COPRO) + DSPy for RAG + DSPy + LangGraph Integration + DSPy Assertions + Production DSPy + Capstone Compiled RAG. Cierra el gap: vault tenía 0 coverage de DSPy (mencionado solo como "niche for later"). Main thread, 8 sequential Writes, 0 subagents. | `06 - Large Language Models/21 - DSPy and Prompt Compilation/` (8 files) |
| 34 | **Crear curso LangSmith Deep Dive (09/35) — Plan L** | 8 notas, 2,571 líneas, English. Welcome + Core Primitives (Traces, Runs, Projects, Datasets) + Auto-Instrumentation for LLM SDKs + Datasets and Evaluations + Online Evaluators with LLM-as-Judge + Annotation Queues and Human Feedback + Production Patterns (Sampling, Costs, PII) + Capstone Production RAG. Cierra el gap: vault tenía LangSmith mencionado en places pero sin curso dedicado. Main thread, 8 sequential Writes, 0 subagents. | `09 - MLOps y Produccion/35 - LangSmith Deep Dive/` (8 files) |
| 35 | **Crear curso WebGPU and On-Device ML (15/04) — Plan M** | 6 notas, 2,123 líneas, English. Welcome + WebGPU Fundamentals + ONNX Runtime Web + WebLLM (full LLMs in browser) + Transformers.js (HuggingFace in browser) + Production Patterns (Privacy, Offline, Limits). Cierra el gap de "on-device ML" — única categoría que el vault no tenía. Main thread, 6 sequential Writes, 0 subagents. | `15 - Transversal Skills/04 - WebGPU and On-Device ML/` (6 files) |
| 36 | **Crear curso Instructor and Structured Generation (06/22) — Tier-1 #1** | 6 notas, 2,691 líneas, English. Welcome (5 paradigms decision framework) + Instructor (Pydantic-native, retries, partial=True streaming, multi-modal, async batch) + Outlines (FSA constrained decoding, regex/JSON Schema/CFG, vLLM/server-side enforcement) + Guidance (token-level control flow, `gen/select/regex/json`, token caching 50-80%) + LMQL (SQL-like DSL, `where` + `assert`, `lmql serve`) + Capstone (FastAPI + Instructor + LiteLLM + Phoenix streaming + MOCK_JSON tests + Docker Compose = fifth portfolio project). Cierra 4 gaps en structured outputs (Instructor 0 dedicated, Outlines 0 dedicated, Guidance 0 dedicated, LMQL 0 dedicated, frameworks that use them under the hood como smolagents/PydanticAI/OpenAI Agents cubiertos por separado). Main thread, 6 sequential Writes, 0 subagents. Cross-links: 03/06 Pydantic (ancla), 06/12 RAG, 06/13 vLLM, 06/17 SGLang, 06/19 LLM Gateway, 06/20 RAG Eval, 06/21 DSPy, 07/17 Production Agent Frameworks, 07/18 LangGraph, 09/31 Phoenix, 09/34 OTel, 10/31 FastAPI, 10/39 Auth. **Variance note:** 2,691 líneas vs 2,150 target (+25%). Notas 01, 02 y 05 sobre target por la profundidad de los antipatterns (5+ por nota), las case studies (Replit bug-bot, fintech KYC, ETH legal extraction, StayBot/LLM Eval Suite) y el capstone integration (FastAPI + LiteLLM + Phoenix + SSE + MOCK_JSON + Docker Compose). | `06 - Large Language Models/22 - Instructor and Structured Generation/` (6 files) |
| 37 | **Crear curso LangFuse Deep Dive (09/36) — Tier-1 #2** | 6 notas, 3,007 líneas, English. Welcome (4-tool observability landscape, cost economics) + Fundamentals (Project/Trace/Span/Observation model, self-hosted vs cloud, Postgres+ClickHouse+MinIO, $285/mo vs $399/mo) + SDK Auto-Instrumentation (OpenAI/Anthropic/Cohere wrappers, LiteLLM callbacks, LangChain/LlamaIndex/Instructor integration, sampling) + Datasets/Evaluations/Prompts (versioned datasets, LLM-as-Judge via langfuse.evaluate, compare_experiments with Welch t-test, prompt registry with labels, A/B testing, Git commit linking) + Online Evaluators (10% sampling, user feedback, drift detection, alerting thresholds, auto-rollback) + Capstone (Docker Compose: LangFuse + Worker + Postgres + ClickHouse + Redis + MinIO + Qdrant + FastAPI RAG service + Instructor + LiteLLM + online evaluators + drift cron + CI evaluation runner). Cierra el gap crítico de LangFuse (0 dedicated notes, antes solo mencionado en comparativas). Complementa 09/34 OTel (protocol) + 09/35 LangSmith (SaaS counterpart) + 09/31 Phoenix (drift on inputs). Main thread, 6 sequential Writes, 0 subagents. Cross-links: 06/19 LLM Gateway (multi-provider cost attribution), 06/20 RAG Eval (offline statistical rigor), 06/22 Instructor (structured output tracing), 06/21 DSPy (compiled optimization), 07/18 LangGraph (callback integration), 09/31 Phoenix (drift orthogonality), 09/34 OTel (protocol layer), 09/35 LangSmith (SaaS counterpart), 10/22 Cloud Computing (K8s deployment), 10/31 FastAPI (service patterns), 02 Docker (Compose patterns). **Variance note:** 3,007 líneas vs 2,200 target (+37%). Notas 01 (cost economics table), 02 (per-SDK wrappers), 03 (CI runner + statistical tests), 05 (full Docker Compose + FastAPI + drift cron + CI integration) sobre target por la profundidad operativa. | `09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/` (6 files) |
| 38 | **Crear curso Semantic Kernel + AutoGen Deep Dive (07/19) — Tier-1 #3** | 6 notas, 3,207 líneas, English. Welcome (5 major agent frameworks, Microsoft stack positioning) + SK Fundamentals (Kernel/services/plugins/functions, @kernel_function, multi-language parity Python/.NET/JS/Java, FunctionCallingPlanner) + SK Process Framework + Memory (SteppedProcess vs CyclicProcess, KernelProcess runtime with checkpointing Temporal-style, Kernel Memory vector+episodic, Azure AI Search integration) + AutoGen Fundamentals (RoutedAgent/ConversableAgent/AssistantAgent/UserProxyAgent v0.5 hierarchy, RoundRobinGroupChat vs SelectorGroupChat, termination conditions, Docker code executor, reflection/debate patterns) + AutoGen Advanced (FunctionTool registration, MCP integration, RAG via Qdrant/Azure AI Search/LangChain, error recovery with retries/fallbacks, swarm patterns with DiGraphBuilder, cost limits via TokenUsageTermination, deployment FastAPI + LangFuse) + Capstone (multi-framework composition: SK plugin orchestration + AutoGen researcher-critic debate + LangGraph approval subgraph; FastAPI service; Azure Container Apps deployment via Bicep; LangFuse + Phoenix unified observability; 50-item golden dataset + CI eval; 6th portfolio project). Cierra el gap crítico de Microsoft agentic stack (vault tenía LangGraph + LangChain + LlamaIndex + smolagents + PydanticAI pero 0 coverage de Semantic Kernel / AutoGen). Main thread, 6 sequential Writes, 0 subagents. Cross-links: 06/15 LLM Security, 06/19 LLM Gateway, 06/22 Instructor, 07/11 Fundamentos Agentes, 07/15 MCP, 07/17 Production Agent Frameworks, 07/18 LangGraph Deep Patterns, 09/31 Phoenix, 09/34 OTel, 09/36 LangFuse, 10/22 Cloud Computing, 10/31 FastAPI, 10/33 Vector Databases, 13/06 Go ML Backend, 16 Harness Engineering. **Variance note:** 3,207 líneas vs 2,400 target (+34%). Notas 02 (KernelProcess runtime, Process Framework), 03 (v0.5 actor model + termination), 04 (MCP + RAG + swarm + cost control), 05 (multi-framework composition + Azure Bicep) sobre target por la profundidad operativa. | `07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive/` (6 files) |
| 39 | **Crear curso Serverless LLM Platforms (06/23) — Tier-1 #4** | 6 notas, 2,861 líneas, English. Welcome (4 main platforms: Modal/Replicate/Together/Fireworks; cost economics crossover at ~100M tokens/day) + Modal (`@app.function()`, 7 GPU tiers T4-B200, Volumes for persistent storage, Secrets via `modal secret create`, web endpoints + streaming, async via `function.map()`, cron + queues) + Replicate (community models via `replicate.run()` + `replicate.stream()`, Cog packaging with `cog.yaml` + `predict.py`, per-second GPU billing) + Together AI + Fireworks (OpenAI-compatible APIs, model catalog Llama 3.3 / Mixtral / DeepSeek-V3 / Qwen 2.5 / 405B, function calling + JSON mode + streaming, Fireworks sub-400ms TTFT vs Together $0.88/M, Together serverless fine-tuning $2-5 per LoRA run) + Cost Optimization (cold-start cost analysis 7× more expensive than warm; keep_warm + scaledown_window strategies; prefix caching 80%+ savings; semantic caching via Redis Stack vector search 60-80% hit rate; hybrid architecture saves 40-60%; LiteLLM usage-based-routing-v2; per-tenant budget enforcement) + Capstone (FastAPI + LiteLLM cost-aware router across Modal/Replicate/Together/Fireworks/vLLM, Redis semantic cache, per-tenant budget enforcement, Modal custom LoRA app, Replicate Cog package, LangFuse cost attribution + Prometheus metrics, 7th portfolio project). Cierra el gap crítico: serverless LLM platforms tenían 0 dedicated notes (vault tenía vLLM/SGLang self-hosted y OpenAI/Anthropic SaaS pero faltaba Modal/Replicate/Together/Fireworks). Main thread, 6 sequential Writes, 0 subagents. Cross-links: 06/13 vLLM, 06/17 SGLang, 06/19 LLM Gateway, 06/20 RAG Eval, 06/22 Instructor, 09/36 LangFuse, 10/22 Cloud Computing, 10/30 WebSockets, 10/31 FastAPI, 02 Docker, 13/06 Go ML Backend. **Variance note:** 2,861 líneas vs 2,500 target (+14%). Notas 01 (Modal volumes + secrets + cron), 03 (Together/Fireworks comparison + function calling), 04 (cost math), 05 (full hybrid stack + budget) sobre target por la profundidad operativa. | `06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/` (6 files) |
| 40 | **Crear curso Production RAG: Haystack + txtai (06/24) — Tier-1 #5** | 6 notas, 3,161 líneas, English. Welcome (4 frameworks comparison: LangChain/LlamaIndex/Haystack/txtai, deepset Cloud positioning, decision tree) + Haystack Fundamentals 2.x (`@component` DSL typed inputs/outputs, InMemory/Qdrant/Milvus/Pinecone/Elasticsearch document stores, 8+ retrievers and rankers, basic RAG pipeline) + Haystack Advanced (hybrid BM25+dense with cross-encoder reranking, RRF weights tuning [0.6, 0.4], agentic RAG with `Agent` and `@tool` registration, cross-framework integration with LangChain and LlamaIndex retrievers, offline evaluation via AnswerExactMatchEvaluator/ContextRelevanceEvaluator/FaithfulnessEvaluator with RAGAS-style metrics, streaming + FastAPI integration, deepset Cloud + Kubernetes deployment, OpenTelemetry + LangFuse observability) + txtai Fundamentals (5-line semantic search, persistent indexes via `save()` / `load()`, Qdrant/Faiss/Milvus backends, RAG pipeline with OpenAI/Anthropic/HF, knowledge graphs for multi-hop retrieval, `Workflow` DSL, built-in FastAPI server via `python -m txtai.server`, decision matrix vs Haystack) + Production RAG Patterns (decision tree for framework selection, cross-framework integration patterns: Haystack+LangChain/LlamaIndex/txtai adapters + custom component wrappers, cost optimization across frameworks: Redis semantic cache, prefix caching on Together, selective reranking, observability via LangFuse + Phoenix, multi-framework architecture: LlamaIndex for ingestion + Haystack for retrieval + txtai for multi-hop + LangGraph for agents) + Capstone (LlamaIndex multi-modal ingestion + txtai multi-hop graph retrieval + Haystack hybrid + cross-encoder reranking + LiteLLM multi-provider LLM + Instructor Pydantic structured RAGResponse with citations + LangFuse per-tenant cost attribution + RAGAS golden dataset CI evaluation; FastAPI service with `/ingest` `/query` endpoints; Docker Compose stack with Qdrant + LangFuse + Phoenix; 8th portfolio project). Cierra el gap crítico: Haystack y txtai tenían 0 dedicated notes (vault tenía LangChain + LlamaIndex covered pero faltaban los 2 frameworks enterprise + multi-hop). Main thread, 6 sequential Writes, 0 subagents. Cross-links: 06/12 Production RAG, 06/13 vLLM, 06/17 SGLang, 06/19 LLM Gateway, 06/20 RAG Eval, 06/22 Instructor, 06/23 Serverless LLM Platforms, 07/18 LangGraph Deep Patterns, 09/20 RAG Eval, 09/31 Phoenix, 09/36 LangFuse, 10/31 FastAPI, 10/33 Vector Databases. **Variance note:** 3,161 líneas vs 2,500 target (+26%). Notas 02 (hybrid retrieval tuning + agentic + cross-framework), 03 (txtai graph + workflows + multi-hop), 04 (decision tree + cross-framework integration patterns), 05 (full multi-framework capstone con multimodal ingestion + Haystack hybrid + txtai graph + LiteLLM + Instructor + LangFuse) sobre target por la profundidad operativa. | `06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/` (6 files) |
| 41 | **Crear curso Production Incident Response for AI Systems (09/39) — Tier-2 #2** | 6 notas, 2,919 líneas, English. Welcome (¿por qué los incidentes AI son diferentes? health signals vs quality signals, los 7 failure modes únicos) + AI Incident Taxonomy (7 categorías: hallucination cascades / cost explosions / latency cliffs / quality drift / prompt injection / data leakage / cascading agent failures; cada una con signature, root causes, blast radius, detection signal, first response; real cases: $4M retailer refund, $47K agent loop, $200K legal model upgrade) + Detection (3 core dashboards: latency, cost, quality; Prometheus multi-window burn rate alerts 5min + 1h; LangFuse quality score sampling + drift detection via z-score > 3; per-tenant cost attribution; PagerDuty + Slack routing; leading vs lagging indicators; alert fatigue reduction; on-call rotation pattern) + Triage (OODA loop aplicada: Observe → Orient → Decide → Act; 5-minute diagnostic checklist; rollback procedures en 4 niveles: prompt → model → deploy → kill switch; status communication en 4 niveles: status page, Slack, email, postmortem kickoff; incident note estructurado; cuándo pagear al team) + Resolution Patterns and Resilience Engineering (circuit breakers con 3 estados CLOSED/OPEN/HALF_OPEN + recovery; graceful degradation con quality tiers high/good/medium/low; per-tenant token bucket rate limiting con tiers free/pro/enterprise; dead-letter queues con exponential backoff retries; shadow traffic validation pre-rollback; auto-mitigation: "auto-mitigate then page"; composición defense-in-depth de los 6 patterns) + Capstone (ResilientLLMService unificando los 6 patterns + IncidentSimulator con 3 scenarios: cost explosion / provider outage / prompt injection + auto-mitigation monitor asyncio task + on-call runbook con escalation tree + blameless postmortem template con principios (sistemas, no personas; good intent; systemic improvements) + capstone run script con verification steps; 9th portfolio project: senior engineer incident response skill). Cierra el gap crítico: production incident response tenía 0 dedicated coverage (vault tenía 09/34 OTel + 09/36 LangFuse + 09/31 Phoenix pero el pattern operacional completo de detect-triage-resolve-postmortem era missing). Main thread, 6 sequential Writes, 0 subagents. Cross-links: 06/15 LLM Security, 06/19 LLM Gateway, 06/22 Instructor, 09/21 Monitoreo, 09/28 Testing ML, 09/31 Phoenix, 09/34 OTel, 09/36 LangFuse, 10/22 Cloud Computing, 10/31 FastAPI, 13/03 Microservices with Go (circuit breakers), 16 Harness Engineering. **Variance note:** 2,919 líneas vs 2,500 target (+17%). Notas 02 (dashboards + multi-window alerts + z-score anomaly detection), 04 (6 resilience patterns cada uno con case study), 05 (full capstone con simulator + runbook + postmortem template) sobre target por la profundidad operacional. | `09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/` (6 files) |

### 🟡 Pendientes (esperando decisión del usuario)

| # | Tarea | Contexto | Bloqueante |
|---|-------|----------|------------|
| P5 | **Banners para los cursos grandes principales** (Python Básico/Intermedio, Go Fundamentals, Rust Fundamentals, FastAPI, etc.) | Tras crear los 7 banners existentes, los cursos "anchor" del vault aún no tienen banner | Esperando `sí` o `no` (P1 resuelto a 4 banners, no a los grandes) |
| P6 | **Plan A — Production Agent Frameworks 2026 (07/17)** | ✅ COMPLETE (Julio 2026). 8 notas, 3,697 líneas. 6 frameworks + capstone RAG. | ✅ Done |
| P7 | **Plan B — ML System Design Interviews (06/33)** | ✅ COMPLETE (Julio 2026). 11 notas, 2,948 líneas. 8 problemas canónicos + CLEAR framework + Capstone. | ✅ Done |
| P8 | **Plan C — Pydantic Deep Dive (03/06)** | ✅ COMPLETE (Julio 2026). 10 notas, 1,580 líneas. Core Pydantic v2 + SQLModel + Settings + Performance. | ✅ Done |
| P9 | **Plan D — Python Production Flow for MLOps (04/03)** | ✅ COMPLETE (Julio 2026). 7 notas, 1,064 líneas. Profiling + Ray + Async + Packaging + Capstone. Cierra 4 gaps críticos. | ✅ Done |
| P10 | **Plan E — K8s for ML/LLM (04/04)** | ✅ COMPLETE (Julio 2026). 7 notas, 1,191 líneas. GPU scheduling + KServe + Autoscaling + Cluster Ops + Capstone. Unifica contenido disperso en 5 cursos. | ✅ Done |
| P11 | **Next: ???** | Esperando decisión del usuario. | ⏳ Esperando |

### 🟡 Pendientes Nuevos — Plan B (ML System Design Interviews)

**Status:** ✅ COMPLETE (July 2026)
**Code:** `06 - Large Language Models / 33 - ML System Design Interviews`
**Language:** English
**Methodology:** Main thread, 1 note at a time, 11 sequential Write calls. **0 subagents**.

| # | Note | Target | Actual | Status |
|---|------|:------:|:------:|:------:|
| 00 | Welcome + Interview Format Decoder | 80 | 92 | ✅ |
| 01 | The CLEAR Framework | 350 | 240 | ✅ |
| 02 | Problem 1: Airbnb Search Ranking | 420 | 268 | ✅ |
| 03 | Problem 2: DoorDash Dispatch | 420 | 268 | ✅ |
| 04 | Problem 3: Twitter/X Timeline | 420 | 284 | ✅ |
| 05 | Problem 4: Uber ETA Prediction | 420 | 298 | ✅ |
| 06 | Problem 5: Netflix Recommendations | 420 | 282 | ✅ |
| 07 | Problem 6: YouTube Recommendations | 420 | 287 | ✅ |
| 08 | Problem 7: TikTok For You Page | 420 | 306 | ✅ |
| 09 | Problem 8: Spotify Discover Weekly | 420 | 307 | ✅ |
| 10 | Capstone: Design Your Own | 500 | 316 | ✅ |
| **Total** | | **~4,400** | **2,948** | ✅ |

**Per-problem template (Deep Format):** Problem Statement → Clarifying Questions → Back-of-Envelope → Architecture (Mermaid) → ML Component → System Component → Tradeoffs → Production Reality → Compression Code.
**Mandatory cross-links:** 06/32 System Design, 09/21 Monitoreo, 09/27 Feast, 10/30 WebSockets, 10/33 Vector DBs, 10/23 IaC.

**Variance note:** Total 2,948 lines vs 4,400 target (-33%). Las notas son más densas que las de Plan A; cada nota cubre los 9 secciones del template de manera compacta. La cobertura de los 8 problemas canónicos + capstone es completa. Para expandir en sesiones futuras, cada nota tiene notas inline `💡 Tip:` que marcan puntos de expansión.

**Per-problem template (Deep Format):** Problem Statement → Clarifying Questions → Back-of-Envelope → Architecture (Mermaid) → ML Component → System Component → Tradeoffs → Production Reality → Compression Code.
**Mandatory cross-links:** 06/32 System Design, 09/21 Monitoreo, 09/27 Feast, 10/30 WebSockets, 10/33 Vector DBs, 10/23 IaC.
**Why 8 problems:** Cited en libros de referencia (Acing the System Design Interview, ML System Design interviews en YouTube), cubren familias distintas (search, dispatch, ranking, prediction, recs, feed), dan amplitud técnica (CF, deep learning, gradient boosting, multi-modal).

### 🟡 Pendientes Nuevos — Plan C (Pydantic Deep Dive)

**Status:** ✅ COMPLETE (July 2026)
**Code:** `03 - Advanced Python / 06 - Pydantic Deep Dive`
**Language:** English
**Methodology:** Main thread, 1 note at a time, 10 sequential Write calls. **0 subagents**.

| # | Note | Target | Actual | Status |
|---|------|:------:|:------:|:------:|
| 00 | Welcome — Why Pydantic v2 | 80 | 50 | ✅ |
| 01 | BaseModel, Field and Type System | 120 | 184 | ✅ |
| 02 | Validators Deep Dive | 120 | 218 | ✅ |
| 03 | model_config Systematic Tour | 100 | 142 | ✅ |
| 04 | Serialization Mastery | 100 | 155 | ✅ |
| 05 | Advanced Model Patterns | 120 | 161 | ✅ |
| 06 | Pydantic Settings | 100 | 164 | ✅ |
| 07 | SQLModel Bridge | 100 | 146 | ✅ |
| 08 | JSON Schema and OpenAPI | 80 | 157 | ✅ |
| 09 | Performance & Production + Capstone | 180 | 203 | ✅ |
| **Total** | | **~1,100** | **1,580** | ✅ |

**Gaps closed:** `@computed_field`, `WrapValidator`/`PlainValidator`, Pydantic+SQLModel, `create_model`, generics, serialization deep-dive, strict mode, `model_config` systematic, `ValidationError` structure, pydantic-settings patterns, JSON Schema customization, `cached_property`, custom types, v1→v2 migration.
**Pre-requisites:** Python 3.10+ typing (03/01-03), FastAPI for ML (10/31) recommended but not required.

**Variance note:** Total 1,580 lines vs 1,100 target (+44%). Las notas combinan 20 topics de Pydantic v2 que antes tenían cobertura cero. Cada nota incluye código ejecutable y tablas de referencia. Para expandir: nota 02 (Validators) y 09 (Production) tienen los mayores potenciales de expansión.

### 🟡 Pendientes Nuevos — Plan D (Python Production Flow for MLOps)

**Status:** ✅ COMPLETE (July 2026)
**Code:** `04 - Engineering Fundamentals / 03 - Python Production Flow for MLOps`
**Language:** English
**Methodology:** Main thread, 1 note at a time, 7 sequential Write calls. **0 subagents**.

| # | Note | Target | Actual | Status |
|---|------|:------:|:------:|:------:|
| 00 | Welcome — The 4 Gaps in Python ML Engineering | 50 | 36 | ✅ |
| 01 | Python Profiling: cProfile, py-spy, memory, flame graphs, GPU | 180 | 151 | ✅ |
| 02 | Ray: Core, Serve, Train, Data, Cluster | 180 | 189 | ✅ |
| 03 | Async Patterns: batch inference, SSE streaming, graceful shutdown | 150 | 171 | ✅ |
| 04 | Python Packaging: poetry dep groups, monorepo, packaging models | 150 | 167 | ✅ |
| 05 | Production Pipeline: profiling → async → packaging → Ray | 150 | 155 | ✅ |
| 06 | Capstone: Profile → Optimize → Package → Deploy on Ray | 200 | 195 | ✅ |
| **Total** | | **~1,060** | **1,064** | ✅ |

**Gaps closed:** Python profiling (🔴 → ✅ zero to full coverage), Ray for ML (🟡 → ✅ 1 note to 1 module), Async for MLOps (🟡 → ✅ fragmented to unified), Packaging for ML (🟡 → ✅ scattered to systematic).
**Pre-requisites:** Python Avanzado para ML (04/00), FastAPI for ML (10/31), Docker (02).

**Variance note:** Total 1,064 lines vs 1,060 target (on target). Las notas cierran 4 gaps críticos que antes tenían cobertura cero o mínima. Nota 02 (Ray) y 06 (Capstone) son las más largas (189-195 líneas) por la amplitud de Ray APIs y el ciclo completo de optimización. Para expandir: nota 02 (Ray Data + Ray Train) tiene mayor potencial de expansión.

### 🟡 Pendientes Nuevos — Plan E (Kubernetes for ML and LLM)

**Status:** ✅ COMPLETE (July 2026)
**Code:** `04 - Engineering Fundamentals / 04 - Kubernetes for ML and LLM`
**Language:** English
**Methodology:** Main thread, 1 note at a time, 7 sequential Write calls. **0 subagents**.

| # | Note | Target | Actual | Status |
|---|------|:------:|:------:|:------:|
| 00 | Welcome — Why K8s for ML | 60 | 35 | ✅ |
| 01 | K8s Fundamentals for ML Engineers | 150 | 177 | ✅ |
| 02 | GPU Scheduling: NVIDIA Operator, MIG, sharing | 150 | 160 | ✅ |
| 03 | KServe: Model Serving on K8s | 180 | 197 | ✅ |
| 04 | Autoscaling: HPA, VPA, KEDA for Inference | 150 | 184 | ✅ |
| 05 | Cluster Ops: Node pools, taints, tolerations | 120 | 193 | ✅ |
| 06 | Capstone: Multi-Model Inference Platform | 200 | 245 | ✅ |
| **Total** | | **~1,010** | **1,191** | ✅ |

**Gaps closed:** K8s for ML (disperso en 5 cursos → unificado). GPU scheduling (🔴 scattered → ✅ systematic). KServe deployment patterns. Autoscaling strategies (HPA + KEDA + Knative). Cluster operations for GPU workloads.
**Pre-requisites:** Docker (02), Python Production Flow for MLOps (04/03).

**Variance note:** Total 1,191 lines vs 1,010 target (+18%). Nota 06 (Capstone) es la más larga (245 líneas) por la integración completa: KServe + KEDA + GPU pool + canary + monitoring. Notas 04 (Autoscaling) y 05 (Cluster Ops) también sobre target por la cobertura de KEDA + Knative y las guías de producción. Para expandir: nota 06 (modelos adicionales en el capstone, e.g. vLLM + KServe) tiene mayor potencial.

### 🟡 Pendientes Nuevos — Plan A (Production Agent Frameworks 2026)

**Status:** ✅ COMPLETE (June 2026) — see entries 26-27  
**Code:** `07 - AI Agents y Agentic Systems / 17 - Production Agent Frameworks`  
**Language:** English  
**Methodology:** Main thread, 1 note at a time, 8 sequential Write calls. **0 subagents**.

| # | Note | Target | Actual | Status |
|---|------|:------:|:------:|:------:|
| 00 | Welcome + Agent Framework Landscape 2026 | 90 | 90 | ✅ |
**Code:** `07 - AI Agents y Agentic Systems / 17 - Production Agent Frameworks`  
**Language:** English  
**Methodology:** Main thread, 1 note at a time, 8 sequential Write calls. **0 subagents** (justified: Deep Format spec is already in main context; subagent context reload is ~2-5× costlier per the June 2026 subagent rules).

| # | Note | Target lines | Actual | Status |
|---|------|:------------:|:------:|:------:|
| 00 | Welcome + Agent Framework Landscape 2026 | 90 | 90 | ✅ |
| 01 | smolagents (HuggingFace) | 380 | 420 | ✅ |
| 02 | PydanticAI | 380 | 428 | ✅ |
| 03 | transformers.agents (HuggingFace) | 350 | 400 | ✅ |
| 04 | OpenAI Agents SDK | 380 | 554 | ✅ |
| 05 | Google ADK (Agent Development Kit) | 380 | 554 | ✅ |
| 06 | CrewAI 1.0 (production) | 380 | 613 | ✅ |
| 07 | Capstone: Multi-Framework RAG Agent | 480 | 638 | ✅ |
| **Total** | | **~3,200** | **3,697** | ✅ |

**Mandatory cross-links per note:** 07/15 (MCP), 07/16 (OpenShell), 06/19 (LLM Gateway / LiteLLM), 16 (Harness Engineering), 06/12 (Production RAG).

**Framework decision rationale:**
- ✅ Included: smolagents, PydanticAI, transformers.agents, OpenAI Agents SDK, Google ADK, CrewAI 1.0
- ❌ Excluded (already in 07/12-07/14): LangGraph, LangChain, LlamaIndex, AutoGen, CrewAI legacy
- ❌ Excluded (niche for later): DSPy

**Capstone integration:** Uses smolagents (note 01) as the runtime, LiteLLM (06/19) as the model provider, Phoenix (09/31) for observability, OpenShell (07/16) for sandbox. Connects to Plan C (Second Brain RAG) as the real-world demo.

**Variance note:** Notes 04-07 came in over the 380-line target (554, 554, 613, 638) because the OpenAI Agents SDK, Google ADK, CrewAI 1.0 Flow API, and the capstone integration all warranted deeper treatment than originally scoped. The Deep Format spec is more important than the line target — better to over-deliver on completeness than to artificially truncate.

### 🔵 Sugerencias para futuras sesiones

| # | Idea | Justificación |
|---|------|---------------|
| S1 | **Banners para los cursos grandes principales** (Python Básico/Intermedio, Go Fundamentals, Rust Fundamentals, FastAPI, etc.) | Mantener consistencia visual en los cursos "anchor" del vault |
| S2 | **Crear templates de banner SVG** | Para cursos futuros: 3 paletas (cyan/azul, verde, naranja) reutilizables |
| S3 | **Auditoría visual del vault** | Revisar 17 módulos, identificar cuáles merecen banner (welcome notes) |
| S4 | **Próximos cursos candidatos** (ver tier 2/3 del scan original) | OpenTelemetry, LangSmith, DSPy, Mamba-2, WebGPU |
| S5 | **Traducir 06/19 a español selectivamente** | Si audiencia en Colombia lee inglés, mantener EN; si no, traducir |
| S6 | **Probar el nuevo flujo main-agent-first** | La próxima vez que crees un curso nuevo (1-6 notas), escribe 1 nota a la vez en main thread y mide el coste. Las reglas reescritas de subagentes asumen que main thread es más barato — vale la pena validarlo. |

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
