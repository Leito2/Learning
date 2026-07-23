# 🏷️ Welcome — LangFuse: Open-Source LLM Observability

## 🎯 Learning Objectives
- Identify the four-tool LLM observability landscape and pick the right tool per scenario
- Self-host LangFuse on Docker Compose and Kubernetes for full data sovereignty
- Instrument OpenAI, Anthropic, Cohere, LangChain, LlamaIndex, and Instructor with three lines of decorator code
- Manage versioned datasets, run reproducible LLM-as-Judge evaluations, and track prompt A/B tests
- Wire online evaluators on 10% sampled production traffic for live drift detection
- Build a self-hosted multi-provider RAG stack with full prompt-to-cost lineage

## Introduction

For three years the LLM observability market has been shaped by a single constraint: **where does the trace data live**. SaaS tools like LangSmith store it on someone else's servers — convenient, fast to onboard, but a non-starter for HIPAA, GDPR, and SOC2-strict industries. Open-source alternatives like LangFuse and Phoenix store it on your own infrastructure — slower to set up, but the only viable answer for production regulated workloads.

LangFuse occupies a specific niche in that landscape. It is open-source (MIT-licensed core, EE for SSO/SCIM), self-hostable (Docker, Kubernetes), and brings more than tracing to the table. Where LangSmith is essentially "made-up-of-LangChain", LangFuse treats the LLM call as one observation among many in a typed schema with first-class **datasets**, **prompts**, **evaluations**, and **online evaluators**. A team using LangFuse can manage prompts in version control, run reproducible evaluations against historic datasets, and wire live sampling for drift detection — all without leaving the platform.

By 2026 LangFuse has surpassed 13,000 GitHub stars, raised a $5M seed round in 2024, integrated with every major LLM SDK (OpenAI, Anthropic, Cohere, HuggingFace, LangChain, LlamaIndex, Instructor, DSPy, smolagents, PydanticAI), and become the default LLM observability tool for cost-conscious startups and regulated industries. It also has a managed cloud offering for teams that want the UX without the ops overhead.

This course is the missing piece in the vault's observability track. You already have:

- **OpenTelemetry for AI Engineers** ([[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers]]) — the protocol-level foundation for any observability tool.
- **Evidently AI and Phoenix** ([[09 - MLOps y Produccion/31 - Evidently AI and Phoenix]]) — drift detection + Phoenix spans for LLM traces.
- **LangSmith Deep Dive** ([[09 - MLOps y Produccion/35 - LangSmith Deep Dive]]) — the SaaS gold standard.

What you do not yet have is an **open-source, self-hostable** alternative with first-class dataset and prompt management. That is exactly what this module teaches.

By the end of these six notes you will have built a self-hosted LangFuse stack that captures traces from every LLM call in a multi-provider RAG service, runs reproducible evaluations on historic datasets, manages versioned prompts with A/B testing, and emits online evaluators that flag degraded quality in real time.

![LangFuse UI: traces view](https://langfuse.com/_next/image?url=%2Fstatic%2Fimages%2Flanding%2Fhero-l.png&w=1920&q=75)

---

## 1. The Observability Stack in 2026

| Tool | License | Hosting | Traces | Datasets | Prompts | Eval | Online | Cost at 10M traces/mo |
|------|---------|---------|:------:|:--------:|:-------:|:----:|:------:|------------------------|
| **LangSmith** | Proprietary | SaaS only | ✅ | ✅ | ✅ | ✅ | ✅ | ~$3000/mo (Team plan, 5 users) |
| **LangFuse Cloud** | MIT + EE | SaaS | ✅ | ✅ | ✅ | ✅ | ✅ | $0 OSS, $59/mo Pro, $399/mo Enterprise |
| **LangFuse Self-hosted** | MIT | Your infra | ✅ | ✅ | ✅ | ✅ | ✅ | $0 license + $200-500/mo infra |
| **Phoenix** | Apache 2.0 | Self-host or cloud | ✅ | ❌ | ❌ | ✅ | ❌ | $0 license + $50-200/mo infra |
| **Helicone** | MIT + EE | Cloud-first | ✅ | ❌ | ❌ | ✅ | ✅ | Free up to 100K requests; $20-200/mo |
| **OpenLLMetry** | Apache 2.0 | Client-only | ✅ (collects) | ❌ | ❌ | ❌ | ❌ | $0 license + your backend |
| **Portkey** | MIT + EE | Cloud-first | ✅ | ❌ | ❌ | ❌ | ✅ | $0 OSS, $49/mo Pro |

💡 **Tip:** The right tool depends on three factors: **regulatory constraints** (dictate self-hosting), **team size** (cloud is cheaper for < 5 engineers), and **prompt iteration velocity** (teams changing prompts daily need first-class prompt management, not just tracing).

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why Open-Source LLM Observability | This overview |
| 01 | LangFuse Fundamentals — Architecture and Core Primitives | Self-hosted vs cloud, Project/Trace/Span/Observation, storage tiers |
| 02 | LLM SDK Auto-Instrumentation | OpenAI, Anthropic, Cohere, LangChain, LlamaIndex, Instructor |
| 03 | Datasets, Evaluations and Prompt Management | Dataset versioning, LLM-as-Judge, prompt A/B tests |
| 04 | Online Evaluators and Production Patterns | Sampling, user feedback, drift alerting |
| 05 | Capstone — Self-Hosted LangFuse for Multi-Provider RAG | Docker Compose + FastAPI + Instructor + LiteLLM + online evals |

---

## 3. Prerequisites

You should already be comfortable with:

- **OpenTelemetry concepts** — traces, spans, attributes from [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers]].
- **FastAPI service patterns** — async, lifespan, dependency injection from [[10 - Cloud, Infra y Backend/31 - FastAPI for ML]].
- **Multi-provider LLM calls** — LiteLLM or instructor patterns from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]] and [[06 - Large Language Models/22 - Instructor and Structured Generation]].
- **Docker and Docker Compose** — multi-service stacks from [[02 - Docker Profesional]].
- **Basic Kubernetes** — Deployments, Services, Ingress from [[10 - Cloud, Infra y Backend/22 - Cloud Computing]].

💡 If you have not yet read [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers]], skim it before Note 02 — LangFuse is an OTEL-compatible backend, not a replacement.

---

## 4. Cross-Module Connections

This course is part of the observability spine across modules 06, 09, and 10:

| Vault Module | Connection to This Course |
|--------------|---------------------------|
| [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix\|Evidently AI and Phoenix]] | Phoenix for drift detection + LangFuse for traces + evals |
| [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers\|OpenTelemetry for AI Engineers]] | LangFuse accepts OTLP — common span protocol |
| [[09 - MLOps y Produccion/35 - LangSmith Deep Dive\|LangSmith Deep Dive]] | SaaS counterpart; comparison of features and cost |
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns]] | LiteLLM callbacks integrate with LangFuse for cost tracking |
| [[06 - Large Language Models/20 - RAG Evaluation Deep Dive\|RAG Evaluation]] | LangFuse evaluation runs replace RAGAS CI steps |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor and Structured Generation]] | `Instructor` traces via `InstructorInstrumentor` arrive in LangFuse |
| [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns\|LangGraph Deep Patterns]] | LangChain/LangGraph integration captures agent state |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | FastAPI service in capstone uses `observe()` decorator |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | Kubernetes manifests for self-hosting |
| [[02 - Docker Profesional\|Docker Profesional]] | Docker Compose stack in capstone |

---

## 5. What You Will Build

By Note 05, you will have a **self-hosted LangFuse stack** running locally on Docker Compose that:

- Captures traces from a multi-provider RAG service (OpenAI + Anthropic + Groq + vLLM via LiteLLM)
- Stores datasets of historic questions and ground-truth answers
- Runs scheduled LLM-as-Judge evaluations against every new prompt version
- Versions prompts with Git-like checkout semantics
- Wires an online evaluator on 10% of production traffic for live drift detection
- Emits OpenTelemetry-compatible spans to the same LangFuse UI used by Phoenix from [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix]]
- Costs less than $50/month to operate on a small VM (1 vCPU, 4 GB RAM)
- Is portable to Kubernetes via `Helm` and `langfuse/langfuse` chart

This is the **observability spine of the production LLM stack**. Every portfolio project — LLM Edge Gateway, Automated LLM Evaluation Suite, Multi-Agent Research System, StayBot, and the capstone from [[06 - Large Language Models/22 - Instructor and Structured Generation]] — can be retrofitted to LangFuse with a few lines of integration code.

---

## 6. Reading Order for Vault Natives

If you already have the observability foundation:

1. Note 00 (this file) — landscape and decision framework
2. Note 01 (Fundamentals) — required before any further note
3. Note 02 (SDK Auto-Instrumentation) — most directly applicable
4. Notes 03-04 (Datasets + Online Evaluators) — for evaluation-driven teams
5. Note 05 (Capstone) — production synthesis

If you are switching from LangSmith:

1. Note 00 → Note 01 → Note 02 → Note 05 (skip 03-04 if you don't need them yet)
2. Then add Notes 03-04 as you adopt prompt management and online evaluators

---

## 7. The Cutting Edge in 2026

Three frontiers are emerging:

1. **Multi-modal traces.** Beyond text completions: image generation, audio TTS, video, embedding dimensions. LangFuse 2.x ships a generic `Observation` schema that captures any payload — useful for the multimodal capstone from [[15 - Transversal Skills/04 - WebGPU and On-Device ML]].
2. **Prompt-as-code integration.** Linking prompt registry to source control via commit SHA so the trace shows exactly which code version produced the prompt. The capstone shows the GitHub Actions pattern.
3. **Online evaluators at the edge.** Distributed scoring with `langfuse-edge` for sub-100ms feedback loops. Critical for voice agents and real-time UIs.

These frontiers map directly onto the user's portfolio: the **LLM Edge Gateway** ([Go/Fiber/Redis portfolio project]) benefits from cost attribution per team; the **Automated LLM Evaluation Suite** ([Python/asyncio portfolio project]) integrates as a LangFuse evaluation runner; the **Multi-Agent Research System** ([LangGraph portfolio project]) uses LangChain auto-instrumentation; the **StayBot** ([LangGraph/CrewAI/FastAPI/Supabase portfolio project]) uses user feedback collection for prompt improvement.

---

⚠️ The LangFuse ecosystem evolves fast. The SDK packages change monthly (e.g. `langfuse` v3 → v3.x with new decorators), the self-hosted Helm chart evolves quarterly, and the Python decorator API has had three breaking changes since v2.0. The **patterns** in this course are stable; the **SDK imports and version pins** in the code examples will need updating. Always cross-check against the official docs at [langfuse.com/docs](https://langfuse.com/docs) before deploying.
