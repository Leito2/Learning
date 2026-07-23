# 🏷️ Welcome — Serverless LLM Platforms: Modal, Replicate, Together, and Fireworks

## 🎯 Learning Objectives
- Distinguish the four leading serverless GPU/LLM platforms and pick the right one per workload
- Deploy a Modal app with GPU selection, persistent volumes, and web endpoints in < 50 lines
- Package a custom model with Replicate's Cog and serve it via the Replicate API
- Use Together AI and Fireworks as drop-in OpenAI replacements for production-grade inference
- Design hybrid serverless + self-hosted architectures that optimize for cost, latency, and burst
- Build a FastAPI gateway that fans out across providers with cost attribution per tenant

## Introduction

For five years the LLM infrastructure narrative has been dominated by one question: **do you own the GPUs or rent them?** Self-hosting gives you full control, predictable costs at scale, and access to cutting-edge hardware (B200, H200). Renting through OpenAI, Anthropic, Google, etc. gives you zero ops and a per-token price. But a third option has matured into a first-class category by 2026: **serverless GPU platforms** that let you run inference workloads without owning hardware, paying only for the GPU-seconds you consume.

The four platforms covered in this course represent the spectrum of serverless LLM infrastructure. **Modal** is the Python-native platform built around the `@app.function()` decorator — write Python code, declare GPU type, deploy with one CLI command. **Replicate** is the cog-powered marketplace of open-source models — package any model, run it on demand, pay per second. **Together AI** is the production-grade LLM API for open-weight models — Llama, Mixtral, DeepSeek, Qwen with optimized kernels and OpenAI-compatible endpoints. **Fireworks** is the multi-model inference leader — fast speculative decoding, function calling, JSON mode at low latency.

Each platform solves a different problem. Modal is for **custom code** that needs GPUs (fine-tuning, custom inference, batch jobs). Replicate is for **packaged models** (run community models, share yours). Together/Fireworks are for **production LLM serving** (drop-in OpenAI replacement with better cost-per-quality for open-weight models).

By the end of these six notes you will have built a hybrid multi-provider stack that uses Modal for custom GPU jobs, Replicate for one-off inference, Together/Fireworks for production traffic, and self-hosted vLLM for steady state — with FastAPI + LiteLLM routing and LangFuse cost attribution per tenant.

![Modal architecture diagram](https://modal-public.s3.amazonaws.com/static/og-image.png)

---

## 1. The Serverless Landscape in 2026

| Platform | GPU options | Cold start | Pricing model | Strength |
|----------|-------------|:----------:|---------------|----------|
| **Modal** | T4, L4, A10G, A100, H100, H200, B200 | 5-15s | $/GPU-second | Pythonic dev, custom code, async |
| **Replicate** | T4, A40, A100, H100 | 15-60s | $/second | Cog packaging, model marketplace |
| **Together AI** | A100, H100 | ~5s | $/1M tokens | Open-weight models, optimized kernels |
| **Fireworks** | A10G, A100, H100 | ~3s | $/1M tokens | Fastest inference, function calling |
| **Anyscale Endpoints** | A100 | ~5s | $/1M tokens | Ray-based, low-latency |
| **OpenAI** | H100 (private) | ~500ms | $/1M tokens | Frontier closed models |
| **Self-hosted vLLM** | Whatever you buy | <1s (warm) | Hardware amortization | Full control, predictable cost at scale |

The cost calculus for a typical 70B inference workload ($0.50-$1.00 per million output tokens):

| Traffic (M tok/day) | Self-hosted vLLM | Modal A100 | Together AI | Fireworks |
|--------------------:|----------------:|-----------:|------------:|----------:|
| 1 | $50/mo (hardware) | ~$30/mo | ~$20/mo | ~$15/mo |
| 10 | $50/mo (hardware) | ~$300/mo | ~$200/mo | ~$150/mo |
| 100 | $300/mo (hardware) | ~$3,000/mo | ~$1,800/mo | ~$1,400/mo |
| 1000 | $2,000/mo (multi-GPU) | ~$30,000/mo | ~$15,000/mo | ~$12,000/mo |

The crossover where self-hosting beats serverless is **~100M tokens/day** — below that, serverless wins on cost and ops; above that, self-hosted wins on per-token economics. Most production LLM applications sit below the crossover.

💡 **Tip:** The right pattern is **hybrid** — serverless for burst + custom jobs, self-hosted for steady state. A typical production stack: 70% self-hosted (predictable traffic) + 30% serverless (burst + new workloads). The capstone (Note 05) shows this in code.

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Serverless LLM Platforms Landscape | This overview |
| 01 | Modal — Python-Native Serverless GPU | `@app.function()`, GPU selection, volumes, web endpoints, secrets |
| 02 | Replicate — Cog-Powered Inference | Cog packaging, `replicate.run()`, custom models |
| 03 | Together AI and Fireworks — Production-Grade LLM APIs | OpenAI-compatible inference, function calling, JSON mode |
| 04 | Serverless Cost Optimization and Patterns | Cold-start, caching, hybrid architecture, multi-provider failover |
| 05 | Capstone — Production Multi-Provider Serverless Stack | FastAPI + LiteLLM + Modal + Replicate + Together + Fireworks + LangFuse |

---

## 3. Prerequisites

You should already be comfortable with:

- **Python async/await** — all platforms use async patterns.
- **Docker basics** — Modal, Replicate, and Fireworks build container images.
- **LLM serving fundamentals** — vLLM, SGLang from [[06 - Large Language Models/13 - vLLM and Advanced RAG]] and [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference]].
- **LiteLLM routing** — covered in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]].
- **FastAPI service patterns** — covered in [[10 - Cloud, Infra y Backend/31 - FastAPI for ML]].

💡 If you have not yet read [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]], skim it before Note 03 — LiteLLM is the standard routing layer for the capstone.

---

## 4. Cross-Module Connections

This course is part of the LLM serving spine across modules 06, 09, 10:

| Vault Module | Connection to This Course |
|--------------|---------------------------|
| [[06 - Large Language Models/13 - vLLM and Advanced RAG\|vLLM and Advanced RAG]] | Self-hosted comparison; the alternative to serverless |
| [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference\|ColBERT, SGLang and Next-Gen Inference]] | Frontier inference patterns; what's available serverless |
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns]] | LiteLLM routes across all four platforms |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor and Structured Generation]] | Structured outputs work seamlessly on all four |
| [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability\|LangFuse Deep Dive]] | Cost attribution per tenant via metadata |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | Deployment, GPU pricing |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | The capstone FastAPI service |
| [[10 - Cloud, Infra y Backend/30 - WebSockets and Real-Time ML\|WebSockets for ML]] | Streaming responses via SSE |
| [[02 - Docker Profesional\|Docker Profesional]] | Container fundamentals |
| [[13 - Go Engineering/06 - Go for ML Backend\|Go ML Backend]] | The LLM Edge Gateway project integration |

---

## 5. What You Will Build

By Note 05, you will have constructed a **production multi-provider serverless stack** with:

- A Modal app that hosts a custom fine-tuned Llama 3 model for domain-specific inference
- A Replicate deployment of Llama 3.1 70B with LoRA adapters for rapid A/B testing
- A Together AI endpoint serving Llama 3.3 70B as the production default
- A Fireworks endpoint with speculative decoding for the low-latency path
- A FastAPI service that exposes `/chat` and routes across all four via LiteLLM
- LangFuse cost attribution per `team_id` so each tenant pays their own GPU bill
- Auto-scaling: serverless handles burst, a self-hosted vLLM pod handles steady state
- A CI pipeline that runs offline evals against all four providers and compares latency, cost, and quality

This is the **seventh portfolio project**: a multi-provider serverless inference layer that any cost-conscious ML team would recognize. It demonstrates mastery of cost engineering (the highest-paid discipline in production ML in 2026) without requiring 6-figure GPU hardware investments.

---

## 6. Reading Order for Vault Natives

If you already have vLLM/LiteLLM mastery:

1. Note 00 (this file) — landscape and decision framework
2. Note 01 (Modal) — Pythonic GPU platform
3. Note 03 (Together/Fireworks) — drop-in OpenAI replacement
4. Note 02 (Replicate) — model packaging
5. Note 04 (Cost Optimization) — patterns and pitfalls
6. Note 05 (Capstone) — synthesis

If you are migrating from OpenAI-only:

1. Note 00 → Note 03 → Note 04 → Note 05 (focus on the OpenAI-compatible APIs first)
2. Then Notes 01 + 02 to add custom GPU jobs

---

## 7. The Cutting Edge in 2026

Three frontiers are emerging:

1. **Serverless fine-tuning.** Modal launched `@app.function(gpu="H100")` with full fine-tuning support in late 2025. Replicate added `cog train` for custom LoRA jobs. Together AI ships managed LoRA at $0.50/M tokens.
2. **Edge serverless.** Cloudflare Workers AI, Vercel AI SDK, Modal `@web` endpoints with low-latency inference for the first token. Streaming from cold-warm transitions in <500ms.
3. **Multi-region routing.** Fireworks and Together AI both offer regional endpoints (US, EU, Asia). Latency drops 30-50% for users in the same region.

These frontiers map directly onto the user's portfolio: the **LLM Edge Gateway** (Go/Fiber/Redis) can fall back to Modal for burst, the **Automated LLM Evaluation Suite** runs offline evals against all four providers, the **Multi-Agent Research System** uses Modal for custom GPU jobs, the **StayBot** routes per-region via Together AI.

---

⚠️ The serverless ecosystem evolves fast. Modal ships new features monthly, Replicate's Cog tool evolves quarterly, Together/Fireworks pricing adjusts 2-4× per year. The **patterns** in this course are stable; the **prices and CLI commands** in the code examples will need updating. Always cross-check the official docs ([modal.com/docs](https://modal.com/docs), [replicate.com/docs](https://replicate.com/docs), [docs.together.ai](https://docs.together.ai), [docs.fireworks.ai](https://docs.fireworks.ai)) before deploying.
