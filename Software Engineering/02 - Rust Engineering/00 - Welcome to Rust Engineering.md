# 🦀 Welcome to Rust Engineering

Welcome to the **Rust Engineering** track — a comprehensive, English-language learning path designed to transform you into a systems-level engineer with expertise in **ML infrastructure**, **data engineering**, **WebAssembly**, and **autonomous AI agents**.

Rust has shifted from being a "niche" language to a cornerstone of modern tech infrastructure. For ML/AI Engineers, Rust offers the unique combination of C++-level performance with memory safety at compile time — no garbage collector, no segfaults, no data races. This makes Rust the ideal language for building inference servers, data pipelines, vector databases, and autonomous agents that must run reliably for weeks without memory leaks.

---

## 🎯 Why Rust for ML/AI Engineering?

| Advantage | Explanation |
|-----------|-------------|
| **Performance** | Compiles to native code. C++ speed without C++ pain. |
| **Memory Safety** | Ownership model eliminates segfaults and data races at compile time. |
| **Zero-Cost Abstractions** | High-level code compiles to optimal machine code. |
| **Concurrency** | Fearless concurrency — the compiler prevents data races. |
| **ML Ecosystem** | Candle (HuggingFace), PyO3 (Python bindings), ONNX Runtime Rust. |
| **Data Engineering** | Polars, Apache Arrow, Qdrant — all written in Rust. |
| **WebAssembly** | Rust is the #1 language for WASM, enabling browser and edge ML. |
| **Job Market** | Growing demand for Rust in infrastructure, fintech, and ML systems. |

---

## 📚 Course Structure

### Core Courses

| # | Course | Focus | Notes |
|---|--------|-------|-------|
| 01 | [[02 - Rust Engineering/01 - Rust Fundamentals/00 - Welcome\|👋 Rust Fundamentals]] | Ownership, traits, cargo, error handling, patterns | 7 notes |
| 02 | [[02 - Rust Engineering/02 - Advanced Rust/00 - Welcome\|🚀 Advanced Rust]] | Tokio, async, unsafe, FFI, macros, profiling | 7 notes |
| 03 | [[02 - Rust Engineering/03 - Rust for Data Engineering/00 - Welcome\|📊 Rust for Data Engineering]] | Polars, Arrow, ETL, uv, Ruff, CLIs | 7 notes |
| 04 | [[02 - Rust Engineering/04 - Rust for ML and AI/00 - Welcome\|🤖 Rust for ML and AI]] | PyO3, Candle, ONNX, inference servers, vector DBs | 7 notes |
| 05 | [[02 - Rust Engineering/05 - WebAssembly and Edge AI/00 - Welcome\|🌐 WebAssembly and Edge AI]] | WASM, browser ML, WASI, serverless, edge deployment | 7 notes |
| 06 | [[02 - Rust Engineering/06 - Agentic AI with Rust/00 - Welcome\|🧠 Agentic AI with Rust]] | MCP, Goose, autonomous agents, orchestration, security | 7 notes |

### Deep Content Expansion

| # | Course | Focus | Notes |
|---|--------|-------|-------|
| 07 | [[02 - Rust Engineering/07 - Polars Internals and Advanced/00 - Welcome to Polars Internals\|📊 Polars Internals and Advanced]] | Lazy evaluation, memory mapping, streaming, Arrow, window functions | 6 notes |
| 08 | [[02 - Rust Engineering/07 - Candle Advanced Patterns/00 - Welcome to Candle Advanced Patterns\|🕯️ Candle Advanced Patterns]] | Custom models, GPU, transformers, WebAssembly, production tuning | 6 notes |

### Practical Tracks

| Track | Description |
|-------|-------------|
| [[00 - Rust Project Planning Guide\|📋 projects/]] | Step-by-step project guides to build your portfolio |
| [[00 - Welcome to Rust Extra\|👋 extra/]] | Advanced topics: memory model, async internals, crypto, embedded, kernel, blockchain |

---

## 🏗️ Recommended Learning Path

```mermaid
graph LR
    A[01 - Fundamentals] --> B[02 - Advanced]
    B --> C[03 - Data Engineering]
    C --> D[04 - ML and AI]
    D --> E[05 - WASM and Edge]
    E --> F[06 - Agentic AI]
    F --> G[Projects]
    G --> H[Extra]
    style D fill:#FF6B6B,stroke:#333,stroke-width:2px
```

**Phase 1 (Weeks 1-2):** Complete Fundamentals + Advanced Rust
**Phase 2 (Weeks 3-4):** Data Engineering + ML and AI (CORE for ML roles)
**Phase 3 (Weeks 5-6):** WASM and Edge + Agentic AI
**Phase 4 (Weeks 7-8):** Build 2-3 portfolio projects from the `projects/` folder
**Phase 5 (Ongoing):** Dive into `extra/` topics as needed

---

## 🎓 Capstone Project

By the end of this track, you will build a **production-grade AI agent system in Rust** that:

1. Implements the Model Context Protocol (MCP) for tool integration
2. Uses Candle or ONNX Runtime for model inference
3. Stores and retrieves embeddings via Qdrant
4. Executes OS commands in a secure sandbox
5. Deploys as a CLI tool and/or WASM module
6. Includes comprehensive benchmarks and security hardening

This project will be the **centerpiece of your portfolio** when applying for ML Engineer and Systems Engineer roles.

---

## 🔗 Related Tracks

- [[../../ML and IA Engineering/Welcome\|🧠 ML and IA Engineering]] — Deep learning, LLMs, MLOps (theory in Python)
- [[../../Software Engineering/Go Engineering/00 - Welcome to Go Engineering\|🐹 Go Engineering]] — Cloud, microservices, local AI with Ollama
- [[../extra/00 - Welcome to Software Engineering Extra\|⚡ Software Engineering Extra]] — FastAPI, System Design, Testing
- [[../../Extra Skills/00 - Welcome to Transversal Skills\|🌉 Transversal Skills]] — Technical English, Communication, Leadership

---

## 📖 Official Resources

| Resource | URL | Why It Matters |
|----------|-----|----------------|
| The Rust Book | https://doc.rust-lang.org/book/ | The definitive Rust learning resource |
| Rust by Example | https://doc.rust-lang.org/rust-by-example/ | Hands-on Rust patterns |
| Rust Reference | https://doc.rust-lang.org/reference/ | Complete language reference |
| Cargo Book | https://doc.rust-lang.org/cargo/ | Package management and build system |
| Polars | https://pola.rs/ | DataFrames at the speed of light |
| Candle | https://github.com/huggingface/candle | HuggingFace's ML framework in Rust |
| PyO3 | https://pyo3.rs/ | Bind Python and Rust together |
| Qdrant | https://qdrant.tech/ | Vector database in Rust |
| WebAssembly | https://webassembly.org/ | WASM specification and resources |

---

> **💡 Tip:** Rust has a steeper learning curve than Go, but the payoff is greater for systems-level work. The borrow checker will frustrate you for weeks, then it will become your superpower.

> **⚠️ Warning:** Do not fight the borrow checker with excessive `clone()` or `unsafe`. Invest time in understanding ownership semantics before writing production Rust; otherwise, you will produce slow, idiomatically poor code that is painful to maintain.
