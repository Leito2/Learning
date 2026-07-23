# 🏷️ Welcome — Instructor and Structured Generation

## 🎯 Learning Objectives
- Identify the five paradigms for forcing structure from LLMs and know when each fails
- Choose between Instructor, Outlines, Guidance, and LMQL based on the workload profile
- Reuse your Pydantic v2 skills (from [[03 - Advanced Python/06 - Pydantic Deep Dive]]) to build validated LLM pipelines
- Ship a production FastAPI service that streams structured outputs with partial validation, retries on validation error, and Phoenix traces
- Recognize which agent frameworks in [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks]] depend on structured outputs under the hood

## Introduction

For three years the LLM community treated structured outputs as a parsing problem. You asked the model for JSON, parsed the response with `json.loads()`, wrapped it in a `try/except`, and hoped. The model occasionally hallucinated fields, returned text outside the braces, produced invalid JSON, or simply ignored the schema. By 2024 the failure rate of naive `json.loads(response)` was 5-15% in production — catastrophic when the downstream pipeline assumes a typed object, and silent when the parser is tolerant.

Three converging waves made structured outputs a first-class engineering discipline rather than a prompt-engineering hack. First, OpenAI shipped **JSON mode** in late 2023 and **strict function calling** in mid-2024 — the model itself constrains its sampling to valid JSON via token masking. Second, the **Instructor** library (jina-ai, then independent) operationalized the Pydantic-native pattern: declare a `BaseModel`, pass it as `response_model=`, and let the library handle retries on `ValidationError` automatically. Third, a category of libraries — **Outlines**, **Guidance**, and **LMQL** — gave developers the ability to **constrain generation at the token level**, with regex, JSON Schema, and context-free grammars, so the model cannot generate invalid tokens in the first place.

This course is the missing piece in the vault's LLM engineering track. You have already studied LLM serving with vLLM and SGLang in [[06 - Large Language Models/13 - vLLM and Advanced RAG]] and [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference]], and you have production RAG and LLM evaluation covered. What you have not had is the **structured-output substrate** that ties it all together: the LLM-as-Judge evaluators in [[06 - Large Language Models/20 - RAG Evaluation Deep Dive]] depend on it; the tool-use agents in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]] depend on it; the LLM Gateway capstone in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]] could not exist without it.

By the end of these six notes you will have written a production structured-extraction service that fans out across multiple providers, validates with Pydantic, retries on schema failure, streams partial objects, and emits OpenTelemetry-compatible traces. That service is the **fifth portfolio project** in your AI/ML Engineer toolkit.

![Instructor: pydantic-native structured outputs](https://python.use-instructor.com/_/og-image.png)

---

## 1. The Five Paradigms — From Prompting to Token Constraints

Any system that forces an LLM to produce structure falls into one of five paradigms, ordered by how much of the work happens inside the model versus outside:

| # | Paradigm | Where constraints live | Failure rate | Cost overhead |
|---|----------|------------------------|:------------:|:-------------:|
| 1 | **Prompt + parse** | In the prompt only | 5-15% | 1× |
| 2 | **JSON mode** | Provider-side token mask | < 1% | 1-2% |
| 3 | **Function calling / tool use** | Provider-side schema | < 1% | 1-3% |
| 4 | **Constrained decoding library (Outlines, Guidance)** | Library + model logits | 0% (impossible to fail) | 5-20% |
| 5 | **Grammar-guided generation** | Custom grammar interpreter | 0% | 5-30% |

💡 The historical progression is paradigm 1 → 2/3 → 4/5. The cutting edge in 2025-2026 is **paradigm 4 with provider integration**: libraries that sit on top of OpenAI's JSON mode, vLLM's guided decoding, or llama.cpp's grammar grammars, and orchestrate retries, validation, and streaming on top.

Every production LLM application in your portfolio falls somewhere on this ladder. The LLM-as-Judge in [[06 - Large Language Models/20 - RAG Evaluation Deep Dive]] could use paradigm 3 (function calling). The RAG extraction pipeline in [[06 - Large Language Models/12 - Production RAG]] benefits from paradigm 4 (Outlines). The agent tool schemas in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]] are essentially paradigm 3 expressed in LangChain's `Tool` schema. Choosing the right paradigm per use case is a senior-engineer-level decision.

---

## 2. The 2026 Library Landscape

| Library | Paradigm | License | Stars (2026) | Pydantic native | Multi-modal | Open source self-hostable | Used internally by |
|---------|:--------:|:-------:|:------------:|:---------------:|:-----------:|:------------------------:|-------------------|
| **Instructor** | 3 (function calling), retries via Pydantic | MIT | 9.5k+ | ✅ | ✅ (vision, audio, PDF) | ✅ | — |
| **Outlines** | 4 (regex, JSON Schema, CFG) | Apache 2.0 | 13k+ | ✅ | Limited | ✅ | vLLM, TGI |
| **Guidance** | 4 (token-level control + generation) | MIT | 22k+ | ❌ (uses its own types) | Limited | ✅ | Microsoft Research |
| **LMQL** | 5 (declarative DSL with constraints) | Apache 2.0 | 2.5k+ | Partial | ❌ | ✅ | — |
| **Native function calling** | 3 | Provider API | — | ❌ (OpenAI uses JSON Schema only) | ✅ | ❌ | LangChain, every agent framework |
| **SGLang `regex` / `grammar`** | 4 | Apache 2.0 | — | ✅ | ✅ | ✅ | — |
| **vLLM guided decoding** | 4 | Apache 2.0 | — | ✅ | ✅ | ✅ | Outlines, Instructor via backend |
| **llama.cpp grammars** | 5 | MIT | — | ✅ | Limited | ✅ | llama.cpp, Ollama, LM Studio |

💡 **Tip:** The four libraries covered in this course (Instructor, Outlines, Guidance, LMQL) are not competitors — they solve different problems. Instructor is the orchestration layer (retries, validation, multi-provider). Outlines is the highest-throughput constraint engine (works inside vLLM, no per-call cost). Guidance is the expressivity layer (interleave control + generation, ideal for complex prompts). LMQL is the declarative DSL (cleanest semantics, hardest to debug).

---

## 3. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — The Structured Output Crisis | This overview |
| 01 | Instructor — Pydantic-Native Structured Outputs | `response_model=`, retries on ValidationError, streaming `partial=True`, multi-modal |
| 02 | Outlines — Constrained Decoding at the Token Level | Regex, JSON Schema, CFG, integration with vLLM and Transformers |
| 03 | Guidance — Token-Level Control and Prompt Programming | `gen()`, `select()`, `regex()`, caching, interleaved control+generation |
| 04 | LMQL — A Query Language for LLMs | Declarative DSL, `[model]` blocks, types and assertions, `lmql serve` |
| 05 | Capstone — Production Structured Extraction Service | FastAPI + Instructor + LiteLLM + Phoenix streaming + async batch |

---

## 4. Prerequisites

You should already be comfortable with:

- **Pydantic v2** — `BaseModel`, `Field`, validators, custom types from [[03 - Advanced Python/06 - Pydantic Deep Dive]]. Instructor reuses every Pydantic pattern you already know.
- **Async Python** — `asyncio`, `async/await`, `httpx`. The capstone service is async end-to-end.
- **FastAPI basics** — Dependency injection, lifespan, streaming responses from [[10 - Cloud, Infra y Backend/31 - FastAPI for ML]].
- **LLM fundamentals** — chat completion, system/user/assistant messages, temperature, function calling from [[06 - Large Language Models/06 - Fundamentos de LLMs]] and [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]].
- **OpenTelemetry basics** — spans, traces, attributes from [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers]]. We instrument structured outputs in the capstone.

💡 If you have not used Instructor before, install it first: `pip install instructor`. For the capstone you also need `litellm`, `fastapi`, `openinference-instrumentation-instructor`, and `opentelemetry-instrumentation-fastapi`.

---

## 5. Cross-Module Connections

This course is the substrate for much of what is already in the vault:

| Vault Module | Connection to This Course |
|--------------|---------------------------|
| [[03 - Advanced Python/06 - Pydantic Deep Dive\|Pydantic Deep Dive]] | Instructor and Outlines both reuse Pydantic v2 `BaseModel` as the schema source |
| [[06 - Large Language Models/12 - Production RAG\|Production RAG]] | Structured extraction from chunks (entities, citations, summaries) is paradigm 4 territory |
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns]] | LiteLLM is the multi-provider transport; Instructor wraps it transparently |
| [[06 - Large Language Models/20 - RAG Evaluation Deep Dive\|RAG Evaluation Deep Dive]] | LLM-as-Judge evaluators are structured outputs — this course shows the underlying mechanism |
| [[06 - Large Language Models/21 - DSPy and Prompt Compilation\|DSPy and Prompt Compilation]] | DSPy signatures compile down to structured-output libraries (Instructor, Outlines) for typed predictors |
| [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks\|Production Agent Frameworks]] | smolagents, PydanticAI, OpenAI Agents SDK all rely on structured tool calls under the hood |
| [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns\|LangGraph Deep Patterns]] | LangGraph state reducers validate typed state with Pydantic — same patterns as Instructor |
| [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix\|Evidently AI and Phoenix]] | Phoenix spans capture structured outputs as JSON attributes — full lineage from prompt to schema to consumer |
| [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers\|OpenTelemetry]] | The capstone emits OTEL spans for every validation, retry, and streaming token |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | The capstone is a FastAPI service with `EventSourceResponse` for partial streaming |

---

## 6. What You Will Build

By Note 05, you will have constructed a **production-shaped structured extraction service** with:

- A FastAPI app exposing `POST /extract`, `POST /extract/batch`, `POST /extract/stream`
- Instructor as the validation orchestrator and Pydantic v2 as the schema source
- LiteLLM as the multi-provider transport (OpenAI, Anthropic, Groq, self-hosted vLLM)
- Up to 4 automatic retries on `ValidationError` with the error message fed back to the model
- Server-Sent Events streaming partial objects with `partial=True`
- OpenTelemetry traces through `openinference-instrumentation-instructor` and `opentelemetry-instrumentation-fastapi`
- Phoenix UI for inspecting every span: prompt, response, retries, validation errors, cost
- A test suite with `pytest` + `pytest-asyncio` using `instructor.Mode.MOCK_JSON` for deterministic tests
- A `docker-compose.yml` that boots the whole stack with a single `docker compose up`

This capstone is the **fifth portfolio project**. It is small enough to understand end-to-end and faithful enough to deploy.

---

## 7. Reading Order for Vault Natives

If you have already finished the LLM serving and RAG tracks, the recommended order is:

1. Note 00 (this file) — overview and the five-paradigm decision framework
2. Note 01 (Instructor) — most directly applicable to your existing Pydantic skills
3. Note 02 (Outlines) — for when throughput matters and you control the inference backend
4. Note 05 (Capstone) — production synthesis
5. Notes 03 and 04 (Guidance and LMQL) — read these last; they are expressive but specialized

💡 If you are a research engineer focused on prompt-programming rather than API wiring, **read 03 and 04 first, then 01 and 02**. Guidance and LMQL give you control that the other libraries do not.

---

## 8. The Cutting Edge in 2026

Three frontiers are emerging as of mid-2026:

1. **Provider-native guided decoding convergence.** OpenAI's strict function calling, vLLM's guided decoding backend, and llama.cpp's GBNF grammars are converging on a common JSON Schema → token-mask primitive. Instructor 1.x and Outlines 0.1+ can drop down to provider-side grammar natively.
2. **Streaming structured outputs as the default.** Naive pipelines parse the full response. Streaming with `partial=True` (Instructor) or `Guidance` chunks gives users partial validation within ~200ms instead of waiting for the full completion — critical for voice agents and real-time UIs.
3. **Multi-modal structured outputs.** Images, audio, PDFs as schema fields. Instructor 1.x supports `Image`, `Audio`, `PDF` types natively. The capstone demonstrates PDF extraction (a common production use case).

These frontiers map directly onto the user's portfolio: the **LLM Edge Gateway** already routes by schema type; the **Automated LLM Evaluation Suite** uses LLM-as-Judge (paradigm 3); the **Multi-Agent Research System** streams agent state (paradigm 4 prerequisite); the **StayBot** parses Airbnb listings into structured fields (capstone use case).

⚠️ The structured-output ecosystem evolves fast. Library APIs change every 2-3 months, new providers add JSON mode quarterly, and vLLM ships guided-decoding improvements with every minor release. The **patterns** in this course are stable; the **library APIs and prices** in the code examples will need updating. Always cross-check against the official docs ([python.use-instructor.com](https://python.use-instructor.com), [outlines-dev.github.io](https://outlines-dev.github.io/outlines), [guidance.readthedocs.io](https://guidance.readthedocs.io), [lmql.ai](https://lmql.ai)) before deploying.
