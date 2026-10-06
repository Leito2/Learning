# 📈 Welcome — Grafana and Latency Engineering for ML Systems

"It's fast" is not a measurement, and "the dashboard looks fine" is not an SLO. Real-time ML systems are judged by their **tails** — the slowest 5% or 1% of decisions — and by whether anyone notices when those tails degrade. This course turns metrics into evidence: how to instrument ML services so percentiles are correct, how to build dashboards that answer questions instead of decorating screens, how to alert on what users feel, and how to run load tests whose numbers can go on a CV.

## 🎯 Learning Objectives
- Design **Prometheus metrics** for ML services: histograms vs summaries, bucket choice, label cardinality
- Build **Grafana dashboards as code** with provisioning, variables, and panels that answer specific questions
- Define **SLIs/SLOs** and alert with **multi-window burn rates** instead of static thresholds
- Measure latency correctly: percentiles, HdrHistogram, **coordinated omission**, open-loop **load testing**
- Build **LLM observability** dashboards: tokens, cost, TTFT, cache hit rate — joining OpenTelemetry, Langfuse, and Prometheus
- Produce benchmark results that are **reproducible and honest** (hardware, versions, repetitions)

## Introduction

Observability for ML has two audiences. **Operators** need to know, in seconds, whether the system is healthy and where it hurts. **Decision-makers** — including interviewers reading your README — need to trust the numbers you report about throughput and latency. Both depend on the same foundations: metrics with the right types and buckets, dashboards built around questions, alerts tied to user-visible objectives, and measurement methodology that doesn't fool itself.

The vault already covers **tracing** in depth ([[../34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers|OpenTelemetry for AI Engineers]]), LLM-specific tracing tools ([[../36 - LangFuse - Open-Source LLM Observability/00 - Welcome - Why Open-Source LLM Observability|Langfuse]]), and a first Prometheus/Grafana overview for infrastructure ([[../../10 - Cloud, Infra y Backend/23 - Infrastructure as Code/07 - Infrastructure Observability - Prometheus, Grafana and Distributed Tracing|Infrastructure Observability]]). This course focuses on **metrics, dashboards, SLOs, and latency methodology** for ML services, using the three portfolio projects as running examples: P1's p95 ≤ 80 ms fraud pipeline, P2's routing coverage and surges, and P3's freshness and TTFT.

![Prometheus architecture](https://prometheus.io/assets/docs/architecture.svg)

*Figure: Prometheus scrapes targets, stores time series, evaluates rules, and sends alerts; Grafana queries it for dashboards. Source: prometheus.io.*

---

## Course Map

| #  | Note                                              | Core question                                                           |
|----|---------------------------------------------------|-------------------------------------------------------------------------|
| 01 | Prometheus Metrics Design for ML                  | Which metric types, buckets, and labels give correct, cheap percentiles? |
| 02 | Grafana Dashboards as Code                        | How do I build dashboards that answer questions and live in Git?         |
| 03 | SLOs, Alerting and Burn Rates                     | How do I alert on what users feel, without alert fatigue?                |
| 04 | Latency Measurement and Load Testing Methodology  | How do I produce latency numbers I can defend?                           |
| 05 | LLM Observability Dashboards                      | How do I see tokens, cost, TTFT, and cache behavior across services?     |

## Prerequisites
- Latency budgets ([[../43 - Real-time Feature Serving with Redis/04 - Latency Budgets for Online Serving|Latency Budgets for Online Serving]])
- Docker Compose and basic PromQL familiarity (helpful, not required)
- OpenTelemetry primitives ([[../34 - OpenTelemetry for AI Engineers/01 - OTel Primitives - Spans Traces and Context Propagation|OTel Primitives]])

⚠️ **Version note:** Prometheus 3.x (2024+) added native histograms as a stable option and UTF-8 metric names; Grafana's provisioning formats and alerting UI evolve across major versions. The concepts here are stable — check syntax against the versions you pin.

## References
- Beyer et al., *Site Reliability Engineering* (O'Reilly, 2016) and *The Site Reliability Workbook* (2018) — SLOs and burn-rate alerting
- Prometheus documentation — metric types, histograms, best practices · Grafana documentation — provisioning, dashboards, alerting
- Gil Tene, *How NOT to Measure Latency*
