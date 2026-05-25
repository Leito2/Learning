# 🏷️ Welcome to MCP and Agentic Protocols

## 🎯 Learning Objectives

- Understand why **protocol standardization** is the missing layer in production agent systems
- Master **Model Context Protocol (MCP)** — dynamic tool discovery, transport layers, and custom server implementation
- Implement **Agent-to-Agent (A2A)** communication for multi-agent orchestration with agent cards and task lifecycles
- Integrate **LangGraph agents with MCP servers** for runtime tool discovery instead of hardcoded function calls
- Build **Computer Use and browser agents** that control real UIs via screenshot-to-action loops
- Design **agent evaluation and observability** pipelines with LangSmith, Braintrust, and OpenTelemetry

---

## Why Protocols Matter for Agent Interoperability

In 2025, the agentic AI landscape has a fragmentation problem. Every framework — LangGraph, CrewAI, AutoGen, OpenAI Agents SDK — implements its own tool-calling protocol. Every LLM provider — Anthropic, OpenAI, Google — defines its own function-calling schema. The result: agents built in one ecosystem cannot discover or invoke tools from another. An agent you build in LangGraph cannot dynamically call a tool defined in an MCP server. A research agent cannot delegate a subtask to a verification agent unless both speak the same protocol.

This course addresses the **interoperability layer** that sits below frameworks and above individual LLM APIs. If you already understand agent architectures from [[../../03 - AI Agents y Agentic Systems/11 - Fundamentos de Agentes AI/02 - Tool Use y Function Calling.md|Tool Use y Function Calling]] and have built multi-agent systems with [[../../03 - AI Agents y Agentic Systems/12 - Frameworks y Orquestacion/03 - CrewAI y AutoGen.md|CrewAI y AutoGen]], you know the pain of tool integration. MCP solves this by providing a universal interface that any LLM can use to discover and invoke any tool. A2A solves the equivalent problem for agent-to-agent delegation.

For your portfolio projects — the **Multi-Agent Research System** with LangGraph cyclic agents and the **StayBot Airbnb Agent** with CrewAI orchestration — MCP would let you replace hardcoded tool bindings with a dynamic registry. Your Research agent could discover new Tavily-like search tools at runtime without code changes. Your StayBot booking agent could delegate calendar availability checks to a specialized A2A subagent. The evaluation layer from your **Automated LLM Evaluation Suite** could hook into the same protocol to audit every tool call. This course teaches you to build the protocol infrastructure that makes all of this possible.

Production agent deployments need the same rigor as production ML systems. [[../../05 - MLOps y Produccion/20 - Deployment y Serving/00 - Bienvenida.md|Deployment patterns]] like canary releases and [[../../05 - MLOps y Produccion/21 - Monitoreo y Mantenimiento/00 - Bienvenida.md|monitoring]] apply equally to agent pipelines. This course extends MLOps practices into the agent domain.

---

## Course Map

| # | Note | Focus | Portfolio Connection |
|---|------|-------|---------------------|
| 01 | [[01 - Model Context Protocol Deep Dive\|MCP Deep Dive]] | Client-server architecture, tool servers, resources, prompts | Research agent tool registry |
| 02 | [[02 - A2A Agent-to-Agent Protocol\|A2A Protocol]] | Agent cards, task lifecycle, streaming | StayBot subagent delegation |
| 03 | [[03 - LangGraph + MCP Integration\|LangGraph + MCP]] | Dynamic tool nodes, runtime discovery, error handling | Multi-Agent Research System |
| 04 | [[04 - Computer Use and Browser Agents\|Computer Use & Browser Agents]] | Screenshot-to-action loop, sandboxing, guardrails | Web automation use cases |
| 05 | [[05 - Agent Evaluation and Observability\|Agent Observability]] | LangSmith, Braintrust, OpenTelemetry, metrics | LLM Evaluation Suite extension |

Each note is **self-contained** and includes complete runnable code, ASCII mental models, Mermaid diagrams, and documented project walkthroughs.

---

## Prerequisites

| Concept | Expected Knowledge | Review If Needed |
|---------|-------------------|-----------------|
| LangGraph fundamentals | StateGraph, nodes, edges, conditional routing | [[../../03 - AI Agents y Agentic Systems/12 - Frameworks y Orquestacion/01 - LangChain en Profundidad.md\|LangChain en Profundidad]] |
| Function calling / tool use | LLM tool schemas, JSON mode, parallel calls | [[../../03 - AI Agents y Agentic Systems/11 - Fundamentos de Agentes AI/02 - Tool Use y Function Calling.md\|Tool Use y Function Calling]] |
| Python async | `async def`, `await`, `asyncio.gather`, context managers | FastAPI documentation |
| REST / gRPC concepts | HTTP methods, status codes, streaming, protobuf basics | General backend knowledge |
| Agent architecture | ReAct loop, planning, memory systems | [[../../03 - AI Agents y Agentic Systems/11 - Fundamentos de Agentes AI/04 - Planning y Razonamiento.md\|Planning y Razonamiento]] |

> 💡 **Tip**: If you built the Multi-Agent Research System, you already understand agent orchestration. This course moves one layer down — into the protocols that make orchestration interoperable across frameworks.

---

## Project Outcome

By the end of this course you will have a deployable `agent-protocols/` repository with MCP servers, A2A agents, LangGraph-MCP integration, browser automation, and observability dashboards — all production-ready and interview-ready.

---

## 📦 Compression Code

```python
# COURSE_MAP: MCP and Agentic Protocols
# Notes: [00_Welcome, 01_MCP_DeepDive, 02_A2A_Protocol, 03_LangGraph_MCP, 04_BrowserAgent, 05_Observability]
# Key protocols: MCP (Anthropic, 2024), A2A (Google, 2025)
# Frameworks: LangGraph, BrowserUse, LangSmith, OpenTelemetry
```

## 🎯 Key Takeaways

- **MCP** standardizes LLM ↔ tool communication; **A2A** standardizes agent ↔ agent communication
- Protocols are the interoperability layer — they sit below frameworks and above LLM APIs
- Every agent you build can benefit from dynamic tool discovery (MCP) and standardized delegation (A2A)
- Evaluation and observability are first-class concerns for production agents, not afterthoughts

## References

- Anthropic MCP Specification: https://modelcontextprotocol.io
- Google A2A Protocol: https://github.com/google/A2A
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Browser-Use Framework: https://github.com/browser-use/browser-use
- LangSmith: https://smith.langchain.com
