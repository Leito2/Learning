# 🕸️ Welcome to LangGraph Deep Patterns

LangGraph is the framework your portfolio already runs on — your **Multi-Agent Research System** and **StayBot** Airbnb agent are both `StateGraph` cyclic agents — but until now the vault has only one dedicated note ([[../15 - MCP and Agentic Protocols/03 - LangGraph + MCP Integration.md|LangGraph + MCP Integration]]), and that note treats LangGraph as a vehicle for MCP tool discovery, not as the orchestration primitive it actually is. This course fixes that gap.

By the end of these ten notes you will have rebuilt the Multi-Agent Research System as a production-grade LangGraph application: subgraph-isolated agents, Postgres-backed persistence, human-in-the-loop approval gates, real-time token streaming, and a deployable FastAPI surface. Every concept ties back to the **state graph** — the mental model where agents are nodes, control flow is edges, and durable memory is the checkpointer.

## 🎯 Learning Objectives

- Master the `StateGraph` primitive: `TypedDict` state, `Annotated` reducers, node/edge contracts, and the compile/invoke lifecycle.
- Design conditional routing, dynamic edges, and parallel fan-out via `Send` API without spaghetti `if/else` chains.
- Persist conversations across restarts with `MemorySaver`, `PostgresSaver`, and `thread_id`-based resumption.
- Compose agents into subgraphs that hide their internal state while exposing a clean parent contract.
- Implement production human-in-the-loop patterns with `interrupt()`, `Command`, and approval/reject flows.
- Stream tokens, node updates, and custom events with `stream_mode="values|messages|custom|updates"`.
- Wire retries, fallbacks, and configurable recursion limits into agents that survive bad LLM output.
- Deploy via **LangGraph Studio**, **LangGraph CLI** (`langgraph dev`, `langgraph up`), and a FastAPI surface that respects the same `thread_id` semantics.

## Course Map

| # | Note | Core Concept | Portfolio Hook |
|:-:|------|--------------|----------------|
| 00 | [[00 - Welcome to LangGraph Deep Patterns\|You are here]] | Why LangGraph still wins for stateful agents | Course map |
| 01 | [[01 - StateGraph Fundamentals\|StateGraph Fundamentals]] | `TypedDict` state, reducers, node/edge contract | Rebuild any agent from scratch |
| 02 | [[02 - Conditional Routing and Dynamic Edges\|Conditional Routing]] | `add_conditional_edges`, path map, runtime branching | Triage router in agents |
| 03 | [[03 - Persistence, Checkpointers and thread_id\|Persistence]] | `MemorySaver`, `PostgresSaver`, `thread_id`, time travel | Multi-day conversation resume |
| 04 | [[04 - Subgraphs and Send API\|Subgraphs]] | Subgraph isolation, `Send` API, parallel fan-out | Multi-tenant research dispatch |
| 05 | [[05 - Human-in-the-Loop with interrupt() and Command\|Human-in-the-Loop]] | `interrupt()`, `Command(resume=...)`, approval gates | Tool-call confirmation in StayBot |
| 06 | [[06 - Streaming Modes - values, messages, updates, custom\|Streaming]] | `stream_mode`, `astream_events`, custom writers | Real-time token UI |
| 07 | [[07 - Advanced Patterns - branches, retries, fallback\|Advanced Patterns]] | Configurable recursion limits, retry policies, fallback nodes | Surviving bad LLM output |
| 08 | [[08 - Production Deployment - Studio, CLI, FastAPI\|Production Deployment]] | `langgraph.json`, `langgraph dev/up`, FastAPI integration | Demo on portfolio |
| 09 | [[09 - Capstone - Rebuilding the Multi-Agent Research System\|Capstone]] | Subgraphs + persistence + HITL + streaming | Portfolio showcase |

## Why LangGraph still wins in 2026

The 2024-2025 agent framework wave (LangChain, LlamaIndex, CrewAI, AutoGen) all assumed the agent is **stateless**: every call is a fresh `agent.invoke()`, every multi-agent conversation is a manually-wired orchestration. That assumption breaks the moment your agent needs to:

1. **Remember across turns.** A user adds a flight on Monday, asks about it on Friday. Without persistence, the agent hallucinates.
2. **Pause for approval.** "I want to book this hotel — but only if the user confirms." Without `interrupt()`, you write your own break/continue logic outside the framework.
3. **Stream tokens mid-node.** The UI wants token-by-token output, but the agent also wants to emit "Searching Tavily..." status events. Without a streaming primitive, you bolt on callbacks.
4. **Branch on runtime context.** A triage agent looks at the user's intent, hands off to a billing or technical agent, and may need to escalate. Without conditional edges, you write nested `if/else`.
5. **Recover from crashes.** The LLM returns malformed JSON mid-execution. Without retry nodes and configurable limits, the workflow explodes.

LangGraph's answer is the **state graph**: every interaction is a transition in a typed, inspectable, resumable, streamable directed graph. The framework that 18 months ago felt over-engineered now feels inevitable — and the alternatives (raw LLM loops, CrewAI's role-based paradigm, smolagents' code-as-action) all surrender one or more of those five properties.

> 💡 **Tip:** The most common LangGraph mistake in 2026 is **treating it as a glorified `if/else`**. The framework earns its keep when you lean on state reducers, persistence, subgraphs, and `interrupt()` — not when you wrap a single LLM call.

## Prerequisites

- **Python 3.11+** for `typing.Self`, `ParamSpec`, and the modern typing syntax.
- **Advanced Python (03/03)**: type hints, async, decorators. The [[../../../03 - Advanced Python/03 - Python Avanzado/05 - Type Hints y Anotaciones.md|Type Hints note]] is the entry point.
- **Pydantic Deep Dive (03/06)**: most state schemas use `BaseModel`, `Annotated`, and `Field` for validation.
- **Production RAG (06/12)** and **ColBERT/SGLang (06/17)**: the capstone uses RAG primitives.
- **MCP and Agentic Protocols (07/15)**: the existing [[../15 - MCP and Agentic Protocols/03 - LangGraph + MCP Integration.md|LangGraph + MCP]] note shows dynamic tool discovery.
- **FastAPI for ML (10/31)**: the production deployment note wraps LangGraph in FastAPI.

## How to read this course

1. **Notes 01-03** are the foundation. Skim them if you already know LangGraph basics.
2. **Notes 04-06** are the differentiating patterns (subgraphs, HITL, streaming). These are the notes that pay back the time investment.
3. **Notes 07-08** are production hardening. Read them when you are about to deploy.
4. **Note 09** is the capstone — a 400-500 line rebuild of the portfolio Multi-Agent Research System that ties every prior note together.

## 📦 Compression Code

```python
# 📦 Welcome - The LangGraph mental model in one file
# A complete, runnable research agent in 60 lines: state + nodes + edges + persistence.

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 1. State = TypedDict + Annotated reducers
class ResearchState(TypedDict):
    query: str
    findings: Annotated[list[str], lambda old, new: old + new]  # append-only
    draft: str

# 2. Nodes = pure(state) -> partial_state
def search(state: ResearchState) -> dict:
    return {"findings": [f"Result for {state['query']}"]}

def synthesize(state: ResearchState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"Synthesize: {state['findings']}"
    return {"draft": llm.invoke(prompt).content}

# 3. Graph = nodes + edges + compile
graph = StateGraph(ResearchState)
graph.add_node("search", search)
graph.add_node("synthesize", synthesize)
graph.add_edge(START, "search")
graph.add_edge("search", "synthesize")
graph.add_edge("synthesize", END)

# 4. Compile with a checkpointer
app = graph.compile(checkpointer=MemorySaver())

# 5. Invoke with thread_id for persistence
config = {"configurable": {"thread_id": "user-42"}}
result = app.invoke({"query": "LangGraph reducers"}, config)
print(result["draft"][:80])
# "LangGraph reducers control how state updates merge into the existing graph state..."
```

> 💡 **Tip:** If you can read this 60-line file and predict what `app.invoke(...)` returns, you are ready for note 01. If not, start there.

## References

- [[../15 - MCP and Agentic Protocols/03 - LangGraph + MCP Integration.md|LangGraph + MCP Integration]] — the only LangGraph note before this course.
- [[../17 - Production Agent Frameworks/00 - Welcome to Production Agent Frameworks.md|Production Agent Frameworks]] — smolagents, PydanticAI, CrewAI 1.0 alternatives compared.
- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- LangGraph Studio: https://studio.langchain.com/