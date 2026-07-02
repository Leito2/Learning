# 🏷️ Welcome to Production Agent Frameworks

## 🎯 Learning Objectives

- Map the **2026 production agent framework landscape** and pick the right tool per workload instead of defending one as universally best
- Master six production-grade frameworks — **smolagents, PydanticAI, transformers.agents, OpenAI Agents SDK, Google ADK, CrewAI 1.0** — with hands-on code in each
- Identify which frameworks are **type-safe** (PydanticAI), which are **minimal** (smolagents), which are **provider-native** (OpenAI Agents SDK, Google ADK), and which are **multi-agent orchestrators** (CrewAI 1.0)
- Compose frameworks via [[../15 - MCP and Agentic Protocols/00 - Welcome to MCP and Agentic Protocols.md|MCP]] and the [[../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM.md|LLM Gateway]] so an agent written in one framework can call tools written in another
- Build a **multi-framework RAG agent** capstone that combines smolagents (orchestration) with LiteLLM (model routing) and Phoenix (observability) inside an [[../16 - OpenShell and Agent Sandboxes/00 - Welcome to OpenShell and Agent Sandboxes.md|OpenShell sandbox]]

---

## Why Production Agent Frameworks Matter in 2026

By 2026 the agentic AI stack has fragmented in the same way the JavaScript framework stack fragmented in 2018. There is no single dominant framework, and that is not a bug — it is a feature. Different workloads demand different abstractions: a one-shot research agent has no use for CrewAI's role-playing ceremony, and a multi-step production pipeline crashes if it depends on smolagents' permissive tool execution. The skill that distinguishes a senior AI engineer in 2026 is **framework literacy** — the ability to read the docs of a new framework, run the canonical example, and decide in an afternoon whether it fits the production constraint at hand.

This course sits between the protocol layer ([[../15 - MCP and Agentic Protocols/00 - Welcome to MCP and Agentic Protocols.md|MCP]] and [[../15 - MCP and Agentic Protocols/02 - A2A Agent-to-Agent Protocol.md|A2A]]) and the runtime layer ([[../16 - OpenShell and Agent Sandboxes/00 - Welcome to OpenShell and Agent Sandboxes.md|OpenShell]]). Protocols are the universal interface; runtimes are the trust boundary; frameworks are the developer ergonomics that decide whether you ship the agent in a week or a month. You already know the frameworks of the previous generation — LangGraph, LangChain, LlamaIndex, AutoGen, and CrewAI 0.x — from [[../12 - Frameworks y Orquestacion/00 - Bienvenida|Frameworks y Orquestación]] and [[../13 - Sistemas Multi-Agente/00 - Bienvenida|Sistemas Multi-Agente]]. This course covers the next wave: frameworks designed in 2025-2026 for production deployment, multi-provider routing, and type-safe integration with the rest of the Python ML stack.

For your portfolio, the integration is direct. The **Multi-Agent Research System** you built with LangGraph cyclic agents can be re-expressed in smolagents to compare the developer ergonomics on a workload you already understand. The **LLM Edge Gateway** in Go already routes 5+ providers via LiteLLM semantics; PydanticAI in Python gives you the same provider-agnostic layer for any agent you build from scratch. The **StayBot** Airbnb agent in CrewAI can be ported to CrewAI 1.0 to gain the new `Flow` API and the structured-output guarantees. The **Automated LLM Evaluation Suite** can swap its ad-hoc eval harness for OpenAI Agents SDK's tracing + guardrails. The capstone of this course is a multi-framework RAG agent that uses smolagents as the runtime, LiteLLM as the model router, and Phoenix as the observability backend — three frameworks composing cleanly because each one stays inside its lane.

---

## Course Map

| #  | Note | Focus | Framework family |
|----|------|-------|------------------|
| 00 | [[00 - Welcome to Production Agent Frameworks\|Welcome + Landscape]] | Why six frameworks, decision matrix, what each framework is and is not | (this note) |
| 01 | [[01 - smolagents - Minimal Code Agents from HuggingFace\|smolagents]] | `CodeAgent`, `ToolCallingAgent`, custom `@tool`, E2B sandbox integration | Minimal / code-first |
| 02 | [[02 - PydanticAI - Type-Safe Agents\|PydanticAI]] | Pydantic models as agent contracts, `Agent[DepT, OutT]`, dependency injection, FastAPI | Type-safe / Pythonic |
| 03 | [[03 - transformers-agents - The HuggingFace Agent Loop\|transformers.agents]] | `HfApiEngine`, `react.json` agent, custom tool decorator, multi-modal agents | HF-native / research |
| 04 | [[04 - OpenAI Agents SDK - Provider-Native Production Agents\|OpenAI Agents SDK]] | `Runner`, `handoffs`, `guardrails`, `sessions`, tracing | Provider-native / OpenAI |
| 05 | [[05 - Google ADK - Sequential, Parallel and Loop Agents\|Google ADK]] | `LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, Vertex AI deploy | Provider-native / Google |
| 06 | [[06 - CrewAI 1.0 - Production Multi-Agent Flows\|CrewAI 1.0]] | `Flow` API, YAML crew config, structured outputs, observability hooks | Multi-agent / roles |
| 07 | [[07 - Capstone - Multi-Framework RAG Agent\|Capstone]] | smolagents + LiteLLM + Qdrant + Phoenix + OpenShell: one production RAG agent | Composition |

---

## Prerequisites

| Concept | Expected Knowledge | Review If Needed |
|---------|-------------------|------------------|
| Async Python | `async def`, `await`, `asyncio.gather`, context managers | [[../../03 - Advanced Python/02 - Python Intermedio\|Python Intermedio]] |
| Pydantic | `BaseModel`, `Field`, `model_validator`, type hints | [[../../03 - Advanced Python/03 - Python Avanzado\|Python Avanzado]] |
| Function calling | LLM tool schemas, JSON mode, parallel calls | [[../11 - Fundamentos de Agentes AI/02 - Tool Use y Function Calling\|Tool Use y Function Calling]] |
| ReAct loop | Reason → Act → Observe cycle, plan/execute | [[../11 - Fundamentos de Agentes AI/04 - Planning y Razonamiento\|Planning y Razonamiento]] |
| LiteLLM multi-provider | `completion(model="...")`, fallbacks, routing | [[../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM\|LLM Gateway Patterns and LiteLLM]] |
| MCP and A2A | Tool discovery, agent cards | [[../15 - MCP and Agentic Protocols/00 - Welcome to MCP and Agentic Protocols\|MCP and Agentic Protocols]] |
| RAG | Hybrid search, reranking, citations | [[../../06 - Large Language Models/12 - Production RAG/00 - Welcome to Production RAG\|Production RAG]] |

> 💡 **Tip**: If you built the Multi-Agent Research System, you already know ReAct. This course moves sideways: instead of one framework deeply, you learn six frameworks shallowly — enough to read the docs of a seventh and decide in 30 minutes whether it fits.

---

## Project Outcome

By the end of this course you will have a `production-agents/` repository with one example per framework, a side-by-side comparison harness that runs the same task (e.g., "summarize this paper") across all six frameworks and reports latency, cost, and quality, and a capstone multi-framework RAG agent that demonstrates framework composition via MCP. The capstone is the project you ship to your portfolio; the per-framework examples are the interview prep that lets you answer "have you used PydanticAI?" with code, not slides.

---

## 📦 Compression Code

```python
# COURSE_MAP: Production Agent Frameworks
# 8 notes, ~3,200 lines, English
# Six frameworks studied: smolagents, PydanticAI, transformers.agents, OpenAI Agents SDK, Google ADK, CrewAI 1.0
# Decision axis: type-safety (PydanticAI) vs minimal (smolagents) vs provider-native (OpenAI, Google) vs multi-agent (CrewAI)
# Composition layer: MCP for cross-framework tool discovery, LiteLLM for cross-provider model routing
# Capstone: smolagents + LiteLLM + Qdrant + Phoenix + OpenShell — one production RAG agent
# Cross-links: 07/15 MCP, 07/16 OpenShell, 06/19 LiteLLM, 16 Harness Engineering, 06/12 Production RAG
```

## 🎯 Key Takeaways

- **There is no best agent framework in 2026** — the right answer depends on type safety, provider, team Python fluency, and observability requirements
- **smolagents** is for minimal code-executing agents; **PydanticAI** is for type-safe, contract-driven agents; **transformers.agents** is the HF-native research path
- **OpenAI Agents SDK** and **Google ADK** are provider-native — best when you commit to a single provider and want first-class tracing
- **CrewAI 1.0** keeps the role-and-task model but adds `Flow` for production state machines
- **Composition is via MCP and LiteLLM** — pick the framework that fits the workload, share tools and models through the protocol layer, observe through Phoenix

## References

- smolagents: https://github.com/huggingface/smolagents
- PydanticAI: https://ai.pydantic.dev/
- transformers.agents: https://huggingface.co/docs/transformers/en/agents
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- Google ADK: https://google.github.io/adk-docs/
- CrewAI 1.0: https://docs.crewai.com/
- Model Context Protocol: https://modelcontextprotocol.io
- LiteLLM: https://docs.litellm.ai/
- Phoenix (Arize): https://docs.arize.com/phoenix
