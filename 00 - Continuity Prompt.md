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
├── ML and IA Engineering/              # Main vault for ML courses
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
├── Go Engineering/                     # ✅ COMPLETE (49 notes - English)
│   ├── 00 - Welcome to Go Engineering.md
│   ├── 01 - Go Fundamentals/           # 7 notes
│   ├── 02 - Go for Cloud Native/       # 7 notes
│   ├── 03 - Microservices with Go/     # 7 notes
│   ├── 04 - DevSecOps and CLI Tools/   # 7 notes
│   ├── 05 - Local AI with Go/          # 7 notes
│   ├── 06 - Go for ML Backend/         # 7 notes
│   ├── projects/                       # 6 notes
│   └── extra/                          # 7 notes
│
└── Rust Engineering/                   # IN PROGRESS (~48 notes planned - English)
    ├── 00 - Welcome to Rust Engineering.md
    ├── 01 - Rust Fundamentals/         # 7 notes
    ├── 02 - Advanced Rust/             # 7 notes
    ├── 03 - Rust for Data Engineering/ # 7 notes
    ├── 04 - Rust for ML and AI/        # 7 notes
    ├── 05 - WebAssembly and Edge AI/   # 7 notes
    ├── 06 - Agentic AI with Rust/      # 7 notes
    ├── projects/                       # 6 notes
    └── extra/                          # 7 notes
```

**Current total: ~385 notes created**

---

## Language Policy

- **Existing modules (M00-M08, Advanced Python, Docker, Markdown, SQL):** Written in Spanish. **DO NOT modify** unless explicitly requested.
- **All new content (SE extra/projects, ML extra/projects, Transversal, Go Engineering, future Rust Engineering):** Written entirely in **English**.
- **Welcome.md and indices:** May contain both Spanish and English sections to bridge old and new content.

---

## Note Style and Format (New English Content)

Each course note follows this exact structure:

### For Full Course Notes (01-06 within each course folder)

```markdown
# <Title with emoji>

## Introduction
2-3 paragraphs explaining relevance to ML/AI engineering. Use [[...]] internal links.

## 1. <Section Name>
Deep conceptual explanations with:
- Bullet points for key ideas
- "Real case: <specific ML/AI company example>" at least once
- ⚠️ Warnings about common mistakes
- 💡 Tips and mnemonic rules

## 2. <Section Name>
Same depth. Include comparison tables where useful.

## 3. <Section Name>
At least 2-3 images/diagrams per note:
- Mermaid diagrams (flowcharts, decision trees, timelines)
- Wikimedia Commons image URLs: `![description](https://upload.wikimedia.org/wikipedia/commons/thumb/...)`

## 4. <Section Name>
Python/Go code blocks when applicable.

## 5. <Section Name> (if needed)

---

## 📦 Compression Code
A complete script summarizing the entire topic.

## 🎯 Documented Project

### Description
2-3 sentences about a real-world project.

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
- <Paper or library reference>
- <Tool or framework reference>
```

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

### 1. Go Engineering (✅ COMPLETE)
All 49 notes inside `Software Engineering/Go Engineering/` are complete and pushed to GitHub.

### 2. Rust Engineering (IN PROGRESS — Priority 1)
Complete ~48-55 notes inside `Software Engineering/Rust Engineering/`. Expands the existing `SE/extra/05 - Rust for ML Infra.md` into a full track:

| Course | Notes | Status |
|--------|-------|--------|
| 01 - Rust Fundamentals | 7 | Pending |
| 02 - Advanced Rust | 7 | Pending |
| 03 - Rust for Data Engineering | 7 | Pending |
| 04 - Rust for ML and AI | 7 | Pending |
| 05 - WebAssembly and Edge AI | 7 | Pending |
| 06 - Agentic AI with Rust | 7 | Pending |
| projects/ | 6 | Pending |
| extra/ | 7 | Pending |

**Rules for Rust content:**
- All in English.
- Same depth as M00-M08 (formulas, real cases, mermaid, tables, compression code, documented project).
- Cover: Ownership, Borrowing, Lifetimes, Traits, Async, Unsafe, Polars, Arrow, uv, Ruff, PyO3, Candle, ONNX, WASM, MCP, Goose.
- Official links required: Rust Book, Cargo, Polars, Candle, PyO3, wasm-bindgen, Goose, Ruff, uv.

### 3. Portfolio Web Redesign (PLANNED)
Update `leito2.github.io` from basic course index to a professional ML/AI Engineer portfolio showcasing:
- GitHub repos with real projects
- Kaggle profile
- HuggingFace models
- Blog posts / technical articles

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
