# 🏷️ Welcome — TypeScript for AI Engineers

![Banner del Curso TypeScript for AI Engineers](<typescript-ai-course-banner.svg>)

## 🎯 Learning Objectives
- Translate Python AI patterns to TypeScript with type safety
- Build Next.js frontends with the Vercel AI SDK (`useChat`, streaming)
- Deploy LLMs to Cloudflare Workers for edge inference
- Use LangChain.js + OpenAI SDK for type-safe LLM calls
- Build AI-native backends with Hono / Bun / tRPC
- Apply TypeScript patterns to your portfolio projects (StayBot UI, MARS interface)

## Introduction

The 2026 AI engineering stack is no longer Python-only. Every AI product has three layers:

| Layer | Language | Why |
|-------|----------|-----|
| **Frontend** | TypeScript (React/Next.js) | UI for AI products |
| **Edge / API** | TypeScript (Cloudflare Workers, Hono) | Sub-50ms latency |
| **Core AI** | Python (your stack) | ML frameworks, GPU serving |

You already own the Core AI layer. This course teaches the other two.

The TypeScript AI ecosystem has matured in 2024-2026:

- **Vercel AI SDK** (`ai` package) — streaming UI primitives for Next.js
- **LangChain.js** — 1:1 with Python LangChain
- **OpenAI Node SDK** — official client with full type coverage
- **Cloudflare Workers AI** — edge inference, <50ms latency
- **Workers + Durable Objects** — stateful edge AI
- **Bun** — fast runtime, native TypeScript
- **tRPC + Zod** — end-to-end type safety from Python backend → TypeScript frontend

This note gets you to "TypeScript-fluent for AI" in 6 lessons. Master this and you can build, deploy, and maintain full-stack AI products.

---

## 1. Why TypeScript for AI Engineers (2026)

| Use case | Stack |
|---------|-------|
| **Chatbot UI** | Next.js + Vercel AI SDK |
| **Streaming responses** | Server-Sent Events (SSE) + React Server Components |
| **Edge inference** | Cloudflare Workers + Workers AI |
| **AI Agent UI** | Next.js + CopilotKit / Vercel AI Chat |
| **Mobile (iOS/Android)** | React Native + Expo |
| **Desktop apps** | Tauri + Tauri AI plugin |
| **Browser extensions** | WXT + Chrome AI APIs (Gemini Nano) |

The combo Python backend + TypeScript frontend is the **dominant pattern** for AI products in 2026. Even pure-Python AI companies (OpenAI, Anthropic, Mistral) have TypeScript SDKs as their second-priority client.

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why TypeScript for AI Engineers | This overview |
| 01 | TypeScript Fundamentals for Python Developers | Syntax, types, async/await, modules |
| 02 | Next.js + Vercel AI SDK | Streaming chat UI, RSC, AI primitives |
| 03 | Cloudflare Workers for Edge AI | Workers AI, Durable Objects, edge inference |
| 04 | LangChain.js + AI SDK Integration | LLM orchestration, agents, type-safe wrappers |
| 05 | Capstone — Building a TypeScript AI App | End-to-end project |

---

## 3. Cross-Module Connections

| Vault Module | Connection |
|--------------|-----------|
| [[02 - Docker Profesional\|Docker]] | Container deployment |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | Edge compute |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI]] | Python↔TS via HTTP |
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway]] | LLM Gateway can be TS or Python |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor]] | Python uses Pydantic; TS uses Zod |
| [[Extra/Bun Runtime\|Bun Runtime]] | Bun = TS runtime alternative to Node |
| [[07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive\|SK + AutoGen]] | SK has TS SDK |

---

## 4. What You Will Build

By Note 05, you will have:

- A Next.js chatbot UI with streaming responses
- A Cloudflare Worker that proxies LLM calls at the edge
- A LangChain.js agent with type-safe tool definitions
- A complete full-stack AI app (Next.js + Hono backend + Workers)

This is the **fifteenth portfolio project**: full-stack AI engineering.

---

## 5. Prerequisites

You should already be comfortable with:

- **Python async** — `asyncio`, `await`, `TaskGroup` from [[03 - Advanced Python/08 - Async Python Patterns Reference|03/08 Async Python Patterns Reference]]
- **HTTP and REST APIs** — from [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|10/31 FastAPI]]
- **LLM fundamentals** — from [[06 - Large Language Models/22 - Instructor and Structured Generation|06/22 Instructor]]
- **JavaScript basics** — variables, functions, async/await (the language is similar enough that AI engineers learn fast)

---

## 6. The Cutting Edge in 2026

Three frontiers:

1. **Vercel AI SDK 5** — agents built into `useChat`; multi-modal streaming.
2. **Cloudflare Workers AI** — native LLM inference at edge; <50ms globally.
3. **Bun + Anthropic SDK** — 3× faster than Node; production-ready.

These map directly onto your portfolio: the **LLM Edge Gateway** could have a TS frontend; the **StayBot** UI is a Next.js app; the **Multi-Agent Research System** can be visualized with React Flow + tRPC.

---

⚠️ The TypeScript AI ecosystem moves fast. Vercel AI SDK shipped 4 major versions in 2025; LangChain.js ships monthly. The **patterns** in this course are stable; the **specific library APIs** will need updating. Always cross-check against [sdk.vercel.ai/docs](https://sdk.vercel.ai/docs) and [js.langchain.com](https://js.langchain.com) before deploying.