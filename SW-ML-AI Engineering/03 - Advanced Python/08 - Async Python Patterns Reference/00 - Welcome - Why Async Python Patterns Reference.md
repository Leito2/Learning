# 🏷️ Welcome — Async Python Patterns Reference

> **The reference you reach for when async gets subtle. Event loop internals, the library ecosystem, decision frameworks, subinterpreters, and advanced patterns you only meet after 10K lines of async code.**

## 🎯 Learning Objectives
- Understand the event loop at the selector-and-callback level (uvloop, asyncio, Windows)
- Navigate the Python async library ecosystem: HTTP clients, DB drivers, queues, files
- Apply the 20 anti-patterns that show up in async code review
- Use Python 3.12+ subinterpreters for CPU-bound + async hybrid work
- Build a small async library that exposes structured concurrency correctly
- Decide between `asyncio`, `trio`, `anyio`, and `uvloop` for your project
- Recognize when async is the wrong tool (use threads or processes instead)

## Introduction

By 2026, asynchronous Python is the dominant pattern for I/O-bound workloads: web APIs, message queues, database drivers, LLM inference, vector databases. The previous courses in the vault have taught you how to use async:

- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/01 - ASGI Architecture and Async Python for ML|FastAPI note 01]] — 920 lines of foundations
- [[../../10 - Cloud, Infra y Backend/38 - SQLAlchemy 2.0 Async + Alembic for FastAPI|SQLAlchemy Async course]] — 7 notes
- [[../../10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI|Background Workers course]] — 5 notes
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]

But there's still a **reference gap**. When you're at 2 AM debugging a deadlock, you don't need another tutorial — you need a **lookup table**:

- "What library should I use for X?"
- "Which event loop policy for Windows?"
- "What's the antipattern that matches my stack trace?"
- "Should I use subinterpreters here?"

This course is that reference. Six notes:

1. **Event loop internals** — the actual mechanics, not the mental model
2. **Library ecosystem** — the decision framework
3. **Anti-patterns reference** — 20 patterns with stack-trace signatures
4. **Subinterpreters** — Python 3.12+ for CPU-bound + async hybrid work
5. **Library ecosystem in depth** — generators, queues, locks, channels
6. **Capstone** — build a small async library

![Async Python Patterns Reference](https://python.langchain.com/svg/python-async.svg)

---

## 1. The 2026 Async Python Ecosystem

| Library | Use case | Performance | Async-first |
|---------|----------|-------------|-------------|
| **asyncio** (stdlib) | Standard event loop | Baseline | ✅ |
| **uvloop** (drop-in) | Production event loop | 2-4× faster | ✅ |
| **Trio** | Structured concurrency | Comparable | ✅ (alt) |
| **AnyIO** | Cross-runtime (asyncio/trio) | Comparable | ✅ |
| **httpx** | HTTP client/server | Comparable | ✅ |
| **aiohttp** | HTTP client/server | Slower than httpx | ✅ |
| **asyncpg** | PostgreSQL | Fastest Python PG driver | ✅ |
| **psycopg 3** | PostgreSQL | Comparable | ✅ |
| **SQLAlchemy 2.0 async** | ORM | Comparable | ✅ |
| **Redis (asyncio client)** | Cache/queue | Fast | ✅ |
| **aio-pika** | RabbitMQ | Standard | ✅ |
| **confluent-kafka-python** | Kafka | Fast | Partial |
| **aiokafka** | Kafka | Comparable | ✅ |
| **Pydantic v2 async** | Validation | Fast | ✅ |
| **Instructor async** | LLM | Standard | ✅ |
| **ARQ** | Redis-based jobs | Fast | ✅ |
| **Dramatiq** | Multi-broker jobs | Standard | ❌ (uses threads) |

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why Async Patterns Reference | This overview |
| 01 | Event Loop Internals | uvloop, selector, Windows policy, GIL interplay |
| 02 | Library Ecosystem — Decision Framework | HTTP, DB, queues, jobs, validation |
| 03 | Anti-Patterns Reference Card | 20 patterns with stack-trace signatures |
| 04 | Subinterpreters and Python 3.12+ Async | CPU-bound + async hybrid work |
| 05 | Capstone — Build an Async Library | Hands-on async library design |

---

## 3. Prerequisites

You should already be comfortable with:

- **async/await fundamentals** — coroutines, event loop, `asyncio.gather`
- **FastAPI or async web framework** — [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML|10/31]] course or equivalent
- **Async testing** — `pytest-asyncio` basics
- **Production deployment** — Docker, K8s, observability

If you have not yet read [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]], do so before Note 01 of this course.

---

## 4. Cross-Module Connections

This course complements the existing async coverage:

| Vault Module | Connection |
|--------------|-----------|
| [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/01 - ASGI Architecture and Async Python for ML\|FastAPI note 01]] | Deep dive on FastAPI async specifics |
| [[../../10 - Cloud, Infra y Backend/38 - SQLAlchemy 2.0 Async + Alembic for FastAPI\|SQLAlchemy Async]] | Async DB drivers |
| [[../../10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI\|Background Workers]] | ARQ/Celery async patterns |
| [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing\|Note 11 — Advanced Async Patterns]] | Cancellation, debugging, testing |
| [[../../03 - Advanced Python/03 - Python Avanzado/06 - Concurrencia - Threading y Asyncio\|Python Avanzado Concurrencia]] | Spanish fundamentals |
| [[../../09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering\|Incident Response Note 04]] | Circuit breaker, async patterns |
| [[../../14 - Rust Engineering/02 - Advanced Rust/01 - Concurrency with Tokio and Async-Await\|Rust Tokio]] | Cross-language comparison |
| [[../../16 - Harness Engineering/03 - Harness Engineering - Architecture of Control\|Harness Engineering]] | Concurrency in agent systems |

---

## 5. What You Will Build

By Note 05, you will have:

- A mental model of **why** uvloop is 2-4× faster than asyncio (not just benchmarks)
- A **library selection framework** for every async use case
- A **20-pattern antipattern reference card** with stack-trace signatures
- A **subinterpreter example** that does CPU-bound + async hybrid work
- A small **async library** you build in the capstone (a structured task runner)
- The senior engineer skill: knowing when async is the wrong tool

---

## 6. When Async Is the Wrong Tool

Before diving in, recognize when async is NOT the right choice:

| Scenario | Why async is wrong | Right tool |
|----------|-------------------|-----------|
| **CPU-bound work** | Async doesn't help; GIL still serializes CPU | `multiprocessing` or subinterpreters (Note 04) |
| **Synchronous library** | Wrapping sync in async loses concurrency | Run in thread pool (`asyncio.to_thread`) |
| **Simple scripts** | Async adds complexity without benefit | Plain synchronous code |
| **Libraries without async support** | Wrapping blocks the event loop | `asyncio.to_thread` |
| **Web scraping with Selenium** | Selenium is sync; async wrappers are slow | Use sync + multiprocessing |

Async is the right choice when:

- **I/O-bound work** dominates (HTTP calls, DB queries, file I/O)
- **Concurrency** is the bottleneck
- You have an **async-compatible library ecosystem** (FastAPI, asyncpg, httpx)

---

## 7. The Cutting Edge in 2026

Three frontiers are emerging:

1. **Subinterpreters (Python 3.12+)** — true parallelism for CPU-bound code with async-compatible I/O. `asyncio.Runner` (3.12) provides explicit event loop control.
2. **Trio and AnyIO** — structured concurrency alternatives to asyncio. Some teams prefer Trio's cleaner exception handling.
3. **uvloop 2.x** — even more performance, with `uvloop.run()` as the production event loop.

These map directly onto the user's portfolio: the **LLM Edge Gateway** can use uvloop for 2× throughput; the **Automated LLM Evaluation Suite** can use subinterpreters for CPU-bound metrics; the **Multi-Agent Research System** benefits from structured concurrency.

---

⚠️ The async Python ecosystem evolves fast. New library versions ship monthly. The **patterns** in this course are stable; the **specific library APIs and benchmarks** will need updating. Always cross-check against [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html) and the relevant library docs before deploying.