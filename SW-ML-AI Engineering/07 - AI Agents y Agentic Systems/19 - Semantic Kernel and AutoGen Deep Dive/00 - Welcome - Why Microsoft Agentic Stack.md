# 🏷️ Welcome — Semantic Kernel and AutoGen: The Microsoft Agentic Stack

## 🎯 Learning Objectives
- Distinguish the five major agent frameworks and identify where Microsoft fits
- Use Semantic Kernel for plugin-based function orchestration in Python, .NET, and TypeScript
- Run AutoGen multi-agent conversations with GroupChat for enterprise research workflows
- Combine Semantic Kernel's plugin architecture with AutoGen's multi-agent debate patterns
- Integrate the Microsoft stack with LangGraph (cross-framework subgraph delegation)
- Deploy hybrid agent systems to Azure with full observability through LangFuse and Phoenix

## Introduction

By 2026 the agent framework market has crystallized into five dominant platforms. **LangGraph** (covered in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]]) and **LangChain** own the open-source Python ecosystem. **LlamaIndex** dominates data-heavy RAG agents. **smolagents** and **PydanticAI** are the lightweight ergonomic layer. And then there is **Microsoft**, which ships two complementary frameworks that no serious enterprise ignores: **Semantic Kernel** (production-grade function orchestration) and **AutoGen** (multi-agent conversation research).

The two frameworks solve different problems. **Semantic Kernel** is a **plugin orchestration layer**: it models agent capabilities as typed functions, plugins, and memories; the planner invokes them based on LLM reasoning. It is the Microsoft answer to LangChain's `Tool` abstraction — but with first-class .NET, TypeScript, and Java support, and with native Azure integration that makes it the default for enterprises running on Microsoft infrastructure. **AutoGen** is a **multi-agent conversation framework**: it models agents as conversational partners in a group chat, with researcher-critic structures, human-in-the-loop moderation, and code execution as a first-class action. It is the Microsoft answer to CrewAI's role-based orchestration — but with stronger research provenance and academic backing.

Together, the two frameworks are the spine of enterprise agentic AI. Microsoft's own Copilot stack, the Azure AI Agent Service, and the AutoGen Studio IDE all build on these primitives. By mastering them you add to your portfolio a cluster of skills that immediately signals "enterprise-ready" to recruiters and hiring managers — the same way Go for ML Backend ([[13 - Go Engineering/06 - Go for ML Backend]]) signals "infrastructure-ready".

This course is the missing piece in the vault's agentic track. You already have LangGraph, LangChain, LlamaIndex, smolagents, PydanticAI, OpenAI Agents SDK, Google ADK, CrewAI 1.0 (all covered in [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks]]), and LangGraph Deep Patterns ([[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]]). What you do not yet have is the **Microsoft-native path** through Azure, the multi-language story (.NET, TypeScript, Python), or the AutoGen multi-agent conversation research patterns.

By the end of these six notes you will have built a hybrid agent that uses Semantic Kernel to orchestrate enterprise plugin calls (calendar, CRM, database) and AutoGen for a multi-agent researcher-critic loop that validates the SK agent's answer before returning it to the user.

![Semantic Kernel architecture diagram](https://learn.microsoft.com/en-us/semantic-kernel/media/semantic-kernel.png)

---

## 1. The Five Major Agent Frameworks in 2026

| Framework | Language(s) | Sweet spot | Strength | Weakness |
|-----------|------------|------------|----------|----------|
| **LangGraph** | Python | Cyclic graphs, state machines | Explicit control flow | Python-only |
| **LangChain** | Python + JS | Composition via chains | Ubiquitous, broad integrations | Abstraction overhead |
| **LlamaIndex** | Python | RAG agents | Data ingest + retrieval | Less flexible for tool use |
| **Semantic Kernel** | Python, .NET, JS, Java | Enterprise plugins, function orchestration | Microsoft stack, Azure native | Less research-grade than AutoGen |
| **AutoGen** | Python, .NET | Multi-agent conversation research | Group chat, human-in-the-loop, code execution | Steeper learning curve, no production hardening out of the box |

💡 **Tip:** The right framework depends on (1) **language constraints** (.NET shop → SK), (2) **agent topology** (multi-agent debate → AutoGen, cyclic state → LangGraph, plugin orchestration → SK), and (3) **deployment surface** (Azure → SK, GCP → LangGraph, on-prem → AutoGen). The capstone combines SK + AutoGen + LangGraph subgraphs to show they compose.

---

## 2. Why Microsoft, Why Now

Three reasons:

1. **Enterprise adoption.** Microsoft Copilot Studio, Azure AI Agent Service, and the Microsoft 365 Copilot stack all build on Semantic Kernel. If you work at a Fortune 500 with Microsoft contracts, you will be deploying agents with SK.
2. **Multi-language parity.** SK is the only major framework with first-class Python, .NET/C#, Java, and TypeScript support. The same agent definition compiles to Python for prototyping, .NET 8 for production, TypeScript for the browser extension, and Java for the legacy system.
3. **Research-grade multi-agent.** AutoGen v0.4 (2024) and AG2 (2025 successor) brought actor-model concurrency, type-safe messages, and modern Python packaging. Group chats, nested chats, swarm patterns — these are best-in-class for research papers.

For the AI/ML Engineer profile (LangGraph + FastAPI + Redis + LangChain workflows), adding SK + AutoGen is the **enterprise-readiness upgrade**. Portfolio projects become shippable to mid-market and enterprise customers; interview answers cover .NET backends, Azure deployments, and multi-language agent systems.

![AutoGen GroupChat architecture](https://microsoft.github.io/autogen/0.2/docs/_static/img/chatgroup_main.png)

---

## 3. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — Why Microsoft Agentic Stack | This overview |
| 01 | Semantic Kernel Fundamentals — Kernel, Plugins, Functions | Native and semantic functions, plugin registry, function calling |
| 02 | Semantic Kernel Process Framework and Memory | Stepped vs cyclic processes, Kernel Memory, planners |
| 03 | AutoGen Fundamentals — Conversable Agents and GroupChat | Agent types, multi-agent conversations, code execution |
| 04 | AutoGen Advanced — RAG, Tools and Production Patterns | Function registration, RAG integration, termination, error recovery |
| 05 | Capstone — Multi-Framework Enterprise Agent | SK plugin orchestration + AutoGen debate + LangGraph subgraph integration |

---

## 4. Prerequisites

You should already be comfortable with:

- **Python async/await** — both frameworks use asyncio extensively.
- **Function calling / structured outputs** — SK and AutoGen both model tools as typed functions (covered in [[06 - Large Language Models/22 - Instructor and Structured Generation]]).
- **Multi-agent patterns** — group chat, reflection, debate (covered in [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks]] and [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]]).
- **Vector databases and RAG** — for the AutoGen RAG integration in Note 04 (covered in [[06 - Large Language Models/12 - Production RAG]] and [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search]]).
- **Azure basics** — for the capstone deployment (covered in [[10 - Cloud, Infra y Backend/22 - Cloud Computing]]).
- **.NET fundamentals (optional)** — for the multi-language story in Note 01.

💡 If you have not yet read [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]], skim it before Note 03 — AutoGen's GroupChat pattern is conceptually similar to LangGraph's supervisor pattern.

---

## 5. Cross-Module Connections

This course is part of the agentic AI spine across modules 06, 07, 09:

| Vault Module | Connection to This Course |
|--------------|---------------------------|
| [[06 - Large Language Models/22 - Instructor and Structured Generation\|Instructor and Structured Generation]] | SK plugin functions use the same Pydantic-style validation pattern |
| [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks\|Production Agent Frameworks]] | SK and AutoGen are two of the six frameworks covered; complements LangGraph + LangChain + LlamaIndex + smolagents + PydanticAI |
| [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns\|LangGraph Deep Patterns]] | The capstone combines SK + AutoGen + LangGraph subgraphs |
| [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix\|Evidently AI and Phoenix]] | SK functions emit OpenTelemetry spans; AutoGen emits Phoenix traces |
| [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability\|LangFuse Deep Dive]] | SK and AutoGen both integrate with LangFuse for trace/eval pipelines |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | The capstone deploys to Azure App Service + Azure OpenAI |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | The capstone exposes the hybrid agent as a FastAPI endpoint |
| [[13 - Go Engineering/06 - Go for ML Backend\|Go for ML Backend]] | The .NET interop story pairs naturally with Go microservices |
| [[16 - Harness Engineering\|Harness Engineering]] | The capstone follows the file architecture pattern |

---

## 6. What You Will Build

By Note 05, you will have constructed a **hybrid enterprise agent** with:

- A Semantic Kernel orchestrator that routes user requests to typed plugins (calendar, CRM, document search)
- An AutoGen GroupChat with researcher + critic + summarizer agents that validates the SK output
- A FastAPI service exposing the hybrid agent at `/agent/chat`
- LangGraph subgraphs for long-running workflows (e.g., multi-step approval flows)
- LangFuse traces for every LLM call, Phoenix spans for the AutoGen agent graph
- Azure deployment via the Container Apps CLI with Azure OpenAI as the model provider
- A CI pipeline that runs an offline eval against a 50-item golden dataset

This is the **sixth portfolio project**: a multi-framework enterprise agent that no single-framework candidate could build. It demonstrates that you can compose the right framework for the right sub-task — the canonical signal of a senior engineer.

---

## 7. Reading Order for Vault Natives

If you already have LangGraph + LangChain mastery:

1. Note 00 (this file) — landscape and decision framework
2. Note 01 (SK Fundamentals) — required foundation
3. Note 03 (AutoGen Fundamentals) — required foundation for the second half
4. Note 02 (SK Process + Memory) — read after 01 if you need advanced SK patterns
5. Note 04 (AutoGen Advanced) — read after 03 if you need production patterns
6. Note 05 (Capstone) — synthesis

If you are migrating from LangChain to SK (or .NET shop):

1. Note 00 → Note 01 → Note 02 → Note 05 (the SK half first)
2. Then Notes 03 + 04 to add AutoGen for multi-agent debate

---

## 8. The Cutting Edge in 2026

Three frontiers are emerging:

1. **AG2 and the AutoGen 0.4 → 0.5 migration.** The original AutoGen team split into AG2 (fork) and continued maintenance; the two projects now track each other. Semantic Kernel 1.x added `AutoGenProcessAdapter` for inter-framework collaboration.
2. **Multi-language agents via Semantic Kernel's gRPC bridge.** A .NET SK agent and a Python AutoGen agent can collaborate via `kernel.add_service(AutoGenService())` (preview). The capstone demonstrates this with a Python service calling into a .NET legacy CRM.
3. **Process Framework state recovery.** The new `KernelProcess` runtime supports durable execution across crashes (Temporal-style). Critical for production agents with hour-long workflows.

These frontiers map directly onto the user's portfolio: the **LLM Edge Gateway** can delegate to a SK kernel in .NET for legacy system calls; the **Automated LLM Evaluation Suite** runs AutoGen GroupChat as one evaluator among many; the **Multi-Agent Research System** (LangGraph) can dispatch to AutoGen for debate; the **StayBot** benefits from SK's calendar/CRM plugins.

---

⚠️ The Microsoft agentic stack evolves fast. Semantic Kernel ships minor releases monthly (`1.x → 1.x+1`), AutoGen ships major versions yearly (`0.2 → 0.4 → 0.5 in 2024-2026`), and AG2 sometimes outpaces upstream. The **patterns** in this course are stable; the **library APIs and version pins** in the code examples will need updating. Always cross-check against the official docs ([learn.microsoft.com/semantic-kernel](https://learn.microsoft.com/en-us/semantic-kernel/), [microsoft.github.io/autogen](https://microsoft.github.io/autogen/)) before deploying.
