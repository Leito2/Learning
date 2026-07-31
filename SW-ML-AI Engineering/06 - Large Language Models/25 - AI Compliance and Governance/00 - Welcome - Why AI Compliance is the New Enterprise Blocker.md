# 🏷️ Welcome — AI Compliance and Governance

## 🎯 Learning Objectives
- Identify the regulatory landscape in 2026: EU AI Act, NIST AI RMF, sector-specific frameworks
- Classify AI systems under the EU AI Act risk tiers (unacceptable / high / limited / minimal)
- Generate model cards and datasheets that satisfy compliance requirements
- Detect and mitigate bias using AI Fairness 360 and demographic-parity metrics
- Run red-teaming campaigns with Garak and PyRIT to harden against adversarial attacks
- Build a complete compliance pipeline for a regulated-industry deployment
- Distinguish compliance from security, governance, and observability as adjacent disciplines

## Introduction

In 2024 the EU AI Act became the first comprehensive regulation on artificial intelligence. By 2026 every Fortune 500 AI deployment is in scope. Banks, insurers, hospitals, defense contractors, and any company whose AI makes decisions about humans must comply with risk classifications, transparency requirements, bias audits, model documentation, and post-market monitoring. The penalties for non-compliance reach **7% of global revenue**.

Yet most AI engineers were never trained for this. University curricula cover algorithms, not governance. Bootcamps cover LLMs, not EU regulations. The result: when a hiring manager asks "what's your plan for EU AI Act Article 6?", most engineers cannot answer.

This course is the missing piece. It teaches you the **four pillars of AI compliance**:

1. **EU AI Act risk classification** — where your system falls and what obligations apply
2. **Model cards and datasheets** — the documentation regulators require
3. **Bias and fairness** — the metrics, the toolkit, and the LLM-specific patterns
4. **Red-teaming and adversarial testing** — how to harden your system before deploy

By the end of these six notes you will have built a complete compliance pipeline that produces audit-ready reports. The capstone is the **eleventh portfolio project**: the senior engineer skill of shipping AI that meets regulatory requirements.

![EU AI Act risk pyramid](https://example.com/eu-ai-act-pyramid.png)

---

## 1. The Regulatory Landscape in 2026

| Framework | Jurisdiction | Effective | Penalty |
|-----------|-------------|-----------|---------|
| **EU AI Act** | EU + global reach | 2024-2026 phased | Up to 7% global revenue |
| **NIST AI RMF** | US voluntary | 2023 | None (regulatory) |
| **Colorado AI Act** | US Colorado | 2026 | Up to $20K per violation |
| **NYC Local Law 144** | US NYC | 2023 | Up to $1.5K per violation |
| **China Generative AI Measures** | China | 2023 | Service suspension + fines |
| **UK AI White Paper** | UK | 2023 | Pro-innovation principles |
| **ISO 42001** | Global | 2023 | Certifiable |
| **SOC 2 / ISO 27001** | Global | Existing | Audit failure |

The EU AI Act is the most consequential because of its extraterritorial reach: any AI system that affects EU residents falls in scope, regardless of where the system is built.

---

## 2. Why This Course Now

Three forces are converging:

1. **Regulatory enforcement is real.** The EU AI Act's high-risk provisions entered enforcement in 2026. Companies deploying LLMs for hiring, credit, healthcare, or biometric identification face immediate fines.
2. **Insurance requirements.** Cyber-insurance policies increasingly require AI risk assessments. Companies without model cards and bias audits are uninsurable.
3. **Customer demand.** Enterprise procurement (banks, governments, healthcare systems) requires compliance artifacts before signing contracts. Model cards and bias reports are now table-stakes.

The engineers who understand compliance get hired first. The engineers who ignore compliance get caught when their company receives a regulatory notice.

---

## 3. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why AI Compliance is the New Enterprise Blocker | This overview |
| 01 | EU AI Act 2024 — Risk Classification and Compliance | Risk tiers, obligations, penalties, Article 6 |
| 02 | Model Cards and Datasheets for Datasets | Mitchell framework, Gebru framework, templates, automation |
| 03 | Bias and Fairness — AI Fairness 360 and Demographic Parity | Bias types, metrics, AIF360 toolkit, LLM-specific bias |
| 04 | Red-Teaming and Adversarial Testing | Prompt injection, jailbreaking, Garak, PyRIT, adversarial pipeline |
| 05 | Capstone — Compliance Pipeline for a Regulated Industry | EU AI Act + model card + bias + red-team for a bank |

---

## 4. Prerequisites

You should already be comfortable with:

- **Python production patterns** — async, testing, observability from [[03 - Advanced Python/08 - Async Python Patterns Reference|03/08 Async Python Patterns Reference]]
- **LLM engineering fundamentals** — function calling, structured outputs from [[06 - Large Language Models/22 - Instructor and Structured Generation|06/22 Instructor]]
- **ML evaluation basics** — accuracy, RAGAS, LLM-as-judge from [[06 - Large Language Models/20 - RAG Evaluation Deep Dive|06/20 RAG Evaluation]]
- **Production deployment** — Docker, K8s, CI/CD from [[09 - MLOps y Produccion/22 - End-to-End ML Project|09/22 E2E ML Project]]

💡 If you have not yet read [[06 - Large Language Models/15 - LLM Security and Guardrails|06/15 LLM Security]], skim Note 04 (Red-Teaming) before this course — the concepts build on each other.

---

## 5. Cross-Module Connections

This course draws on every regulatory-adjacent module in the vault:

| Vault Module | Connection |
|--------------|-----------|
| [[06 - Large Language Models/15 - LLM Security and Guardrails\|LLM Security]] | Adversarial testing foundation |
| [[06 - Large Language Models/20 - RAG Evaluation Deep Dive\|RAG Evaluation]] | Bias evaluation patterns |
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor]] | Structured compliance outputs |
| [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix\|Evidently AI and Phoenix]] | Bias drift detection |
| [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability\|LangFuse]] | Audit trail for production usage |
| [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers\|OpenTelemetry]] | Compliance telemetry |
| [[12 - Producto, Negocio y Open Source/30 - Producto y Estrategia de IA/04 - Legal y Compliance en IA\|Legal y Compliance en IA]] | Spanish-language legal primer |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | Compliance deployment (BAA, SOC 2) |

---

## 6. What You Will Build

By Note 05, you will have:

- A **risk classification tool** that maps any AI system to an EU AI Act tier
- A **model card generator** that produces Mitchell-framework-compliant documentation
- A **bias evaluation pipeline** using AI Fairness 360 with LLM-specific metrics
- A **red-team campaign runner** using Garak with custom probes
- A **compliance report generator** that produces audit-ready PDFs
- A **Capstone for a regulated industry** — a bank deploying a credit-scoring LLM

This is the **eleventh portfolio project**: the senior engineer skill of shipping AI that meets regulatory requirements.

---

## 7. The Cutting Edge in 2026

Three frontiers are emerging:

1. **AI Bill of Rights (US)** — voluntary framework becoming procurement-required. Some federal agencies now require it for vendor contracts.
2. **ISO 42001 certification** — auditable AI management systems. Companies can get certified, similar to ISO 27001 for security.
3. **Automated compliance pipelines** — tools that continuously monitor AI systems for compliance violations, similar to how SIEM tools monitor security.

These map directly onto the user's portfolio: the **LLM Edge Gateway** needs EU AI Act risk classification per system; the **Automated LLM Evaluation Suite** is the natural home for bias testing; the **Multi-Agent Research System** needs model card documentation; the **StayBot** (which makes decisions about Airbnb listings) falls in the "limited risk" category of EU AI Act.

---

⚠️ The regulatory landscape evolves fast. The **EU AI Act** is rolling out provisions through 2026-2028. The **NIST AI RMF** updates annually. Sector-specific frameworks (FDA for medical, ECOA for credit, EEOC for hiring) add layers. The **frameworks and patterns** in this course are stable; the **specific article numbers and dates** will need updating. Always cross-check against official sources (eur-lex.europa.eu for EU AI Act, nist.gov for NIST AI RMF) before deploying.