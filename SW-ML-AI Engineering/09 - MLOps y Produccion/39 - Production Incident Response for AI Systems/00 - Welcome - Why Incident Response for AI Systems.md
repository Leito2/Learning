# 🏷️ Welcome — Production Incident Response for AI Systems

## 🎯 Learning Objectives
- Distinguish AI-specific failure modes from traditional software incidents and design responses for each
- Set up detection signals (latency, cost, quality) with Prometheus, LangFuse, and Phoenix
- Diagnose AI incidents using the OODA loop and LangFuse traces in < 5 minutes
- Implement resilience patterns: circuit breakers, graceful degradation, rate limiting, dead-letter queues
- Run a post-incident review that produces blameless, actionable findings
- Operate a 24-hour on-call rotation for a production LLM service

## Introduction

In 2024 a Fortune 500 retailer deployed a customer-support chatbot. Three weeks later, the bot started hallucinating product specifications with 18% of responses. The team didn't notice for two days because their dashboards only tracked **latency** and **error rate** — both of which were healthy. By the time the customer trust team flagged the issue, the bot had sent 40,000 incorrect answers. The postmortem cost $4M in refunds, brand damage, and engineering re-deployment. **The incident was 100% preventable with AI-aware observability.**

This scenario is not unique. AI systems fail in ways that traditional software does not:

- **Hallucination cascades** — one agent's bad output contaminates the next agent's input
- **Cost explosions** — a runaway agent loops 10,000 times; your OpenAI bill arrives 4× higher than projected
- **Latency cliffs** — the LLM provider has an outage; latency jumps from 500ms to 30s; queue piles up; service degrades
- **Quality drift** — the model's outputs gradually degrade as users hit edge cases the eval suite missed
- **Prompt injection** — a malicious user crafts an input that bypasses the system prompt; the agent exfiltrates data
- **Feedback loops** — the model trains on user feedback; bad feedback teaches it the wrong behavior

Traditional SRE training prepares engineers for CPU spikes, memory leaks, and network partitions. It does not prepare them for **quality degradation that doesn't surface as an error**, **cost patterns that take weeks to detect**, or **adversarial inputs designed to exploit the LLM**. This course is the missing piece.

By the end of these six notes you will have built a complete incident response playbook for an AI service: detection (alerts), diagnosis (LangFuse + Phoenix tracing), mitigation (resilience patterns), resolution (rollback procedures), and learning (blameless postmortems). The capstone is a **production on-call simulation** where you face synthetic incidents and resolve them in real time.

![Incident response flow](https://example.com/incident-response.png)

---

## 1. Why AI Incidents Are Different

| Traditional software | AI system |
|---------------------|-----------|
| Errors are binary: 500 OK or exception | Quality is a spectrum: 0-1 confidence |
| Latency degrades under load, recovers automatically | Latency can degrade without load (provider outage) |
| Failures are reproducible (same input → same error) | Failures are stochastic (same input → different output) |
| Cost is fixed per request | Cost varies with tokens generated; runaway possible |
| Users immediately report failures | Quality degradation often goes unreported for days |
| Stack traces pinpoint the bug | Bug might be in the model's training data, not in code |
| Security: SQL injection, XSS | Security: prompt injection, data exfiltration, jailbreak |
| Recovery: redeploy previous version | Recovery: rollback prompt, rollback model, disable feature |

The implication: **traditional monitoring alone is insufficient**. AI systems need quality signals alongside health signals, and they need incident playbooks that account for the unique failure modes.

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why Incident Response for AI Systems | This overview |
| 01 | AI Incident Taxonomy — What Can Break | The 7 categories of AI incidents |
| 02 | Detection — Alerts, Metrics, and Anomaly Detection | Prometheus, LangFuse, multi-window alerts |
| 03 | Triage — Diagnosing AI Incidents | OODA loop, traces, 5-minute checklist |
| 04 | Resolution Patterns and Resilience Engineering | Circuit breakers, graceful degradation, rollback |
| 05 | Capstone — Production Incident Response Simulation | On-call runbook, synthetic alerts, postmortem |

---

## 3. Prerequisites

You should already be comfortable with:

- **Observability basics** — Prometheus, OpenTelemetry from [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers]] and [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability]]
- **Production LLM serving** — FastAPI, LiteLLM from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]] and [[10 - Cloud, Infra y Backend/31 - FastAPI for ML]]
- **Python async/await** — for the resilience patterns
- **Basic system administration** — Linux, Docker, logs

💡 If you have not yet read [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]], skim it before Note 02 — LangFuse spans are the primary detection signal.

---

## 4. Cross-Module Connections

This course draws on every operational pattern in the vault:

| Vault Module | Connection to This Course |
|--------------|---------------------------|
| [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix\|Evidently and Phoenix]] | Drift detection signals |
| [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers\|OpenTelemetry]] | Span propagation across services |
| [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability\|LangFuse]] | Quality score traces |
| [[09 - MLOps y Produccion/28 - Testing in ML Systems\|Testing in ML]] | Pre-production checks |
| [[09 - MLOps y Produccion/29 - CI-CD for ML\|CI/CD]] | Rollback pipelines |
| [[09 - MLOps y Produccion/20 - RAG Evaluation Deep Dive\|RAG Eval]] | Quality metrics for RAG pipelines |
| [[09 - MLOps y Produccion/26 - ML Platform Engineering\|ML Platform Engineering]] | On-call rotation design |
| [[06 - Large Language Models/15 - LLM Security and Guardrails\|LLM Security]] | Prompt injection detection |
| [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM\|LLM Gateway]] | Multi-provider failover |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor]] | Validation as incident containment |
| [[06 - Large Language Models/23 - Serverless LLM Platforms\|Serverless LLM]] | Cold-start incident patterns |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | Multi-region failover |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | Service-level incident patterns |
| [[13 - Go Engineering/03 - Microservices with Go\|Microservices]] | Circuit breaker patterns |

---

## 5. What You Will Build

By Note 05, you will have:

- A complete **on-call runbook** for a multi-provider LLM service
- A Prometheus alerting configuration with multi-window burn rate alerts
- A LangFuse quality scoring pipeline with automatic drift detection
- A circuit breaker pattern implementation for LLM calls
- A **graceful degradation** service that falls back to cached responses when primary fails
- A **postmortem template** with blameless framing
- A **synthetic incident simulation** that triggers cost explosions and quality degradation
- A **resolved timing report** showing how the team would respond in real time

This is the **ninth portfolio project**: the senior engineer skill that distinguishes incident responders from feature developers.

---

## 6. Reading Order for Vault Natives

If you already have production observability experience:

1. Note 00 (this file) — AI-specific failures
2. Note 01 (Taxonomy) — what can break
3. Note 02 (Detection) — alerts and metrics
4. Note 04 (Resolution) — patterns to apply before incidents happen
5. Note 03 (Triage) — the OODA loop when incidents happen
6. Note 05 (Capstone) — runbook synthesis

If you're new to on-call:

1. Note 00 → Note 01 → Note 03 → Note 02 → Note 04 → Note 05

---

## 7. The Cutting Edge in 2026

Three frontiers are emerging:

1. **AI-aware SLOs.** Traditional SLOs (99.9% availability, p99 latency) miss quality degradation. The new pattern: combined health + quality SLOs (e.g., "99% of responses < 2s AND > 95% of responses have quality score > 0.8").
2. **Auto-remediation for LLM pipelines.** Roll back the prompt automatically when quality scores drop below threshold. Roll back the model when the provider's health check fails. The 2026 pattern is "auto-mitigate, then page" — the system resolves common incidents without waking anyone up.
3. **Adversarial incident detection.** Prompt injection is now detected via gradient-based and embedding-based detectors (covered in [[06 - Large Language Models/15 - LLM Security and Guardrails]]). The on-call engineer gets paged when the detector triggers, with the malicious input and the agent's response captured.

These frontiers map directly onto the user's portfolio: the **LLM Edge Gateway** needs circuit breakers and graceful degradation; the **Automated LLM Evaluation Suite** is the detection signal; the **Multi-Agent Research System** is the test environment for incident simulation; the **StayBot** is the production service for runbook validation.

---

⚠️ The incident response field is moving fast. New failure modes emerge each quarter as AI systems grow more complex. The **patterns** in this course are stable; the **specific alert thresholds and tooling** will need updating. Always cross-check against current operational runbooks (Google SRE Book, Datadog Incident Response, PagerDuty State of Digital Operations) before deploying.
