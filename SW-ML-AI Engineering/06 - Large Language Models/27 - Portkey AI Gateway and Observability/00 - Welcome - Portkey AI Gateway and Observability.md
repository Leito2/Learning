# 🏷️ Welcome — Portkey AI Gateway and Observability

![Banner del Curso Portkey AI Gateway and Observability](<portkey-course-banner.svg>)

## 🎯 Learning Objectives
- Compare Portkey to LiteLLM (covered in 06/19) — when to use which
- Set up Portkey as an LLM gateway with multi-provider routing
- Use Portkey's observability features for production debugging
- Implement cost tracking and budget enforcement per tenant
- Configure fallbacks, load balancing, and conditional routing
- Apply PII redaction and audit logging for compliance
- Build a complete production deployment with Portkey

## Introduction

You already have a working LLM gateway via **LiteLLM** (covered in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|06/19]]). **Portkey** is the next-generation alternative with more built-in features.

| Feature | LiteLLM | Portkey |
|---------|---------|---------|
| Multi-provider routing | ✅ | ✅ |
| Cost tracking | Manual via callbacks | **Built-in per request** |
| Caching | Manual | **Built-in (semantic + exact)** |
| Fallbacks | ✅ | ✅ + **conditional** |
| Load balancing | Custom | **Built-in (weighted)** |
| Observability | Via LangFuse | **Native dashboards** |
| Rate limiting | Custom | **Built-in per tenant** |
| PII redaction | Custom | **Built-in** |
| Audit trail | Custom | **Built-in** |
| Webhooks | ❌ | ✅ |
| Self-host | ✅ | ✅ (open-source) |
| SaaS option | ❌ | ✅ (with hosted UI) |

The killer Portkey features:
1. **Native observability** — built-in dashboards; no LangFuse needed
2. **PII redaction** — automatic before sending to LLM
3. **Audit logs** — every request logged for compliance
4. **Conditional routing** — different models for different scenarios

This note teaches when to choose Portkey over LiteLLM, and how to deploy it in production.

---

## 1. Portkey vs LiteLLM — When to Use Each

**Choose LiteLLM when:**
- You already use LangFuse (covered in 09/36)
- You want pure open-source, no SaaS option
- You have a custom observability stack
- You want maximum control over routing logic

**Choose Portkey when:**
- You want LLM gateway + observability + compliance in one
- You need PII redaction out of the box
- You want audit logs for regulated industries (healthcare, finance)
- You want a managed UI for non-engineers
- You need conditional routing (different models per scenario)

For your portfolio: **LiteLLM + LangFuse** (existing) for the LLM Edge Gateway; consider **Portkey** if adding healthcare/finance compliance.

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Portkey AI Gateway and Observability | This overview |
| 01 | Portkey Core — Gateway Fundamentals | Setup, basic routing, providers |
| 02 | Observability and Cost Tracking | Dashboards, per-tenant cost, alerts |
| 03 | Fallbacks, Load Balancing, Conditional Routing | Reliability patterns |
| 04 | PII Redaction and Compliance | GDPR, HIPAA, audit logs |
| 05 | Capstone — Production Portkey Stack | End-to-end deployment |

---

## 3. Prerequisites

You should already be comfortable with:

- **LiteLLM** — multi-provider routing from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|06/19]]
- **LLM providers** — OpenAI, Anthropic, Together, Fireworks from [[06 - Large Language Models/23 - Serverless LLM Platforms|06/23]]
- **Observability** — LangFuse from [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|09/36]]
- **FastAPI or Next.js** — for the gateway proxy

---

## 4. Cross-Module Connections

| Vault Module | Connection |
|--------------|-----------|
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns]] | LiteLLM alternative |
| [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability\|LangFuse Deep Dive]] | Observability alternative |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor]] | Structured outputs |
| [[06 - Large Language Models/23 - Serverless LLM Platforms\|Serverless LLM]] | Multi-provider cost |
| [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/02 - Cost Visibility - Per-Tenant Attribution, Chargeback, and Showback\|Cost Visibility]] | Per-tenant tracking |
| [[06 - Large Language Models/25 - AI Compliance and Governance\|AI Compliance]] | Compliance + audit |

---

## 5. What You Will Build

By Note 05, you will have:

- A Portkey gateway routing across OpenAI, Anthropic, Together AI
- Per-tenant cost attribution and budget alerts
- Fallbacks for reliability (OpenAI → Anthropic → Together)
- PII redaction for compliance
- Audit logs for SOC 2 / GDPR
- Complete Docker Compose deployment

This is the **sixteenth portfolio project**: enterprise-grade LLM gateway.

---

## 6. The Cutting Edge in 2026

Three frontiers:

1. **Portkey v3** — added conditional routing by request body, PII redaction v2, AI guardrails
2. **Portkey AI Gateway open-source** — full self-host option (replaced the previous SaaS-only model)
3. **Portkey Analytics** — hosted dashboard with prompt A/B testing built-in

These map directly onto your portfolio: the **LLM Edge Gateway** could be migrated to Portkey for compliance reasons; the **Automated LLM Evaluation Suite** benefits from Portkey's prompt A/B testing; the **Multi-Agent Research System** uses Portkey's conditional routing.

---

⚠️ The Portkey API and pricing change frequently. The **patterns and architecture** in this course are stable; the **specific API endpoints and pricing** will need updating. Always cross-check against [portkey.ai/docs](https://portkey.ai/docs) before deploying.