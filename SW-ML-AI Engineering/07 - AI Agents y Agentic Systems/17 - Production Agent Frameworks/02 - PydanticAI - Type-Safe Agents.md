# 🔒 PydanticAI — Type-Safe Agents Built on Pydantic

## 🎯 Learning Objectives

- Understand **why PydanticAI exists** as the type-safe answer to the "string-typed tool arguments" problem that plagues every other agent framework
- Master the **`Agent[DepT, OutT]` generic type signature** that makes dependencies and outputs compile-time-checked
- Build **structured-output agents** with Pydantic `BaseModel` as the response contract, validated by the framework
- Use **`RunContext` for dependency injection** so tools can access databases, API clients, and request-scoped state without globals
- Integrate with [[../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM.md|LiteLLM]] for multi-provider routing and [[../../06 - Cloud, Infra y Backend/31 - FastAPI for ML/00 - Welcome|FastAPI]] for production endpoints
- Compose PydanticAI agents with [[../15 - MCP and Agentic Protocols/00 - Welcome to MCP and Agentic Protocols.md|MCP servers]] and stream structured responses with `agent.iter()`

---

## Introduction

PydanticAI is the agent framework built by the team that built Pydantic — the validation library that became the de-facto standard for typed data in the Python ecosystem. Released in late 2024 and now at v0.5+ in 2026, PydanticAI's pitch is simple: **the same `BaseModel` you use to validate a FastAPI request body should validate an agent's tool arguments and final response**. Every other framework of the previous generation treated tool arguments and agent outputs as strings or `dict`s, with validation deferred to your own code (or, more often, not done at all). PydanticAI makes the validation a property of the type system, and the type system is Python's `typing` module.

The practical consequence is that you write an agent the way you write a FastAPI endpoint: declare the input model, declare the output model, write a function that takes both, and let the framework do the validation. If the LLM produces a tool argument of the wrong type, the framework catches it before the tool is called. If the LLM produces a final response that does not match the output model, the framework retries with a structured error message. For your portfolio projects — particularly the **LLM Edge Gateway** in Go and the **StayBot** Airbnb agent — this is the framework that lets you ship "the LLM cannot produce malformed output" as a guarantee, not a hope.

The cost is that PydanticAI does not have smolagents' code-as-action expressiveness. PydanticAI is **tool-calling only** (no Python generation), which is a deliberate design choice: the framework is for production backends where the input/output contracts matter more than the agent's freedom. If you need arbitrary code execution, use smolagents (note 01) and accept the sandboxing burden. If you need type-safe production endpoints, use PydanticAI and accept the tool-calling expressiveness ceiling.

For the **Automated LLM Evaluation Suite** in your portfolio, PydanticAI is a natural fit: the eval results are a `BaseModel` (`EvalResult(score: float, reasoning: str, flags: list[str])`), the judge prompt is a `BaseModel`-typed input, and the LLM-as-judge is a `PydanticAI.Agent` that returns the structured `EvalResult`. The same model can be used to evaluate a different model (the "judge" is decoupled from the "candidate"), and the structured output means your eval dashboard does not need to parse free-form text.

---

## 1. The Problem and Why This Solution Exists

### 1.1 The string-typed tool argument problem

Every agent framework before PydanticAI had the same problem: tool arguments are typed at the Python function signature, but the LLM never sees the Python signature. The LLM sees the JSON schema that the framework generates from the signature, and the LLM may produce JSON that **does not match the schema**. The framework then has to validate, re-prompt, and retry — or worse, pass the bad data to the tool, where the tool either crashes or silently produces wrong results.

```python
# FRAGILE: LangChain tool with weak typing
from langchain_core.tools import tool

@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight from origin to destination on date."""
    # date is supposed to be ISO 8601, but the LLM might pass "next Tuesday"
    # The framework does not enforce the format
    return call_booking_api(origin, destination, date)
```

PydanticAI solves this by **making the tool definition a Pydantic model**, not a function. The model is the schema, the model is the validator, and the framework re-prompts the LLM with the validation error when the LLM produces data that does not validate. The result is that a tool that says `date: datetime` will never receive "next Tuesday" — the LLM will see the validation error and re-prompt with the correct format.

### 1.2 The untyped-output problem

The same problem exists for the agent's final response. Every other framework returns the LLM's raw text (or, at best, a `dict` parsed from the text). The consumer of the agent's output — a FastAPI endpoint, a database, a UI — has to validate the response itself. PydanticAI flips this: the agent's output type is declared at construction time (`Agent[DepT, OutT]`), and the framework validates the LLM's final response against the `OutT` model before returning. If the response does not validate, the framework retries with a structured error message.

```python
# TYPED: PydanticAI agent with structured output
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class FlightBooking(BaseModel):
    origin: str = Field(min_length=3, max_length=3, description="IATA code")
    destination: str = Field(min_length=3, max_length=3, description="IATA code")
    date: datetime
    passengers: int = Field(ge=1, le=9)

agent = Agent(
    model="openai:gpt-4o",
    output_type=FlightBooking,
    system_prompt="You are a flight booking assistant. Extract the user's request.",
)

result = agent.run_sync("Book me a flight from Medellín to Madrid on March 15 for 2 people")
# result.output is a FlightBooking instance, not a string
print(result.output.origin, result.output.date)
```

The `result.output` is a `FlightBooking` instance. No parsing, no `try/except` around `json.loads()`, no defensive code in the consumer. The validation happened at the framework boundary.

### 1.3 The dependencies-without-globals problem

Production agents need access to request-scoped state: a database connection, an authenticated user, a request ID for logging, a rate-limit budget. Other frameworks solve this with module-level globals, class attributes, or framework-specific context objects. PydanticAI solves it with **dependency injection**: the agent is parameterized over a `DepT` type, and every tool receives a `RunContext[DepT]` that exposes the dependencies.

```python
@dataclass
class Deps:
    db: Database
    user_id: str
    request_id: str

class SupportAgent:
    agent = Agent(
        model="openai:gpt-4o",
        deps_type=Deps,
        output_type=SupportResponse,
    )

    @agent.tool
    async def get_user_tickets(ctx: RunContext[Deps]) -> list[Ticket]:
        return await ctx.deps.db.fetch_tickets(ctx.deps.user_id)
```

The agent is a class, the dependencies are a dataclass, and the framework passes them through automatically. Tests can instantiate the agent with mock dependencies; production instantiates with real ones. No globals, no monkey-patching, no `setUp`/`tearDown` choreography.

---

## 2. Conceptual Deep Dive

### 2.1 The `Agent[DepT, OutT]` type signature

PydanticAI's central design choice is the generic type signature. An `Agent` is parameterized over two types:

- `DepT`: the type of the dependencies object. If you do not use dependencies, this defaults to `None`.
- `OutT`: the type of the agent's final response. If you do not declare one, the agent returns `str`.

```python
Agent[None, str]                # simple text agent, no dependencies
Agent[Deps, SupportResponse]     # typed output, typed dependencies
Agent[None, FlightBooking]       # typed output, no dependencies
```

The framework uses these types at construction time to set up the tool validation, the output validation, and the dependency injection. The types are not just documentation — they are the contract. If a tool's argument type is `int` and the LLM produces `"3"`, the framework converts it (Pydantic's coercion) and passes `3` to the tool. If the LLM produces `"three"`, the framework re-prompts with the validation error.

### 2.2 The `RunContext` and dependency injection

Every tool function receives a `RunContext[DepT]` as its first argument. The context exposes:

- `ctx.deps`: the `DepT` instance for the current run.
- `ctx.usage`: a `Usage` object tracking tokens, requests, and tool calls.
- `ctx.messages`: the full message history (mutable; tools can inspect or modify).
- `ctx.tool_name`: the name of the currently executing tool (for tools that call other tools).

The context is request-scoped: a new `RunContext` is created for every `agent.run()` / `agent.run_sync()` call. This means tools can safely modify the context (e.g., increment a counter, log a debug message) without affecting concurrent runs.

### 2.3 The model interface

PydanticAI uses a thin wrapper around the underlying chat model. The wrapper exposes a `Model` interface with a `request(messages, settings)` method. The framework ships adapters for OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere, and HuggingFace. All adapters accept the same `AgentSettings` and produce the same `ModelResponse`. The model is selected at construction time via the `model` argument, which accepts a string in the format `provider:model_name` (e.g., `openai:gpt-4o`, `anthropic:claude-sonnet-4.5`, `gemini:gemini-2.5-pro`).

For multi-provider routing, PydanticAI has a built-in `FallbackModel` that tries models in order. For semantic caching and cost tracking, the integration with [[../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM.md|LiteLLM]] is via the `LiteLLMModel` wrapper (same pattern as smolagents). For your **LLM Edge Gateway** in Go, point PydanticAI at the gateway's OpenAI-compatible endpoint and the gateway's caching/rate-limiting kicks in transparently.

### 2.4 Streaming with `agent.iter()`

For real-time UI updates, PydanticAI provides `agent.iter()` which yields a stream of events:

- `ModelRequestNode`: the agent is about to call the model.
- `ModelResponseNode`: the model has responded (with partial content if streaming).
- `ToolCallNode`: the model has decided to call a tool.
- `ToolResultNode`: the tool has returned its result.
- `EndNode`: the agent has finished, output is in `result.output`.

The stream-based API is the right pattern for [[../../06 - Cloud, Infra y Backend/30 - WebSockets and Real-Time ML/00 - Welcome to WebSockets and Real-Time ML|WebSocket-based]] real-time ML serving: the FastAPI handler iterates over the events, sends each chunk to the WebSocket, and the UI updates incrementally. The same code works for `text/event-stream` HTTP responses (SSE), making PydanticAI a natural fit for both.

### 2.5 Validation retries and `validation_retries`

When the LLM produces a tool argument that does not validate, PydanticAI re-prompts the model with the validation error message. The number of retries is controlled by `validation_retries` (default 3). When the LLM produces a final response that does not match `OutT`, the same retry logic applies. After `validation_retries` exhausted failures, the framework raises a `ValidationError` that you can catch and handle (e.g., return a fallback response, escalate to a human).

The validation retry mechanism is what makes the "the LLM cannot produce malformed output" guarantee work. The framework does not promise that the LLM *will* produce the right output; it promises that the LLM *will be re-prompted until it does, or the framework will fail loudly*. The latter is the more important guarantee for production: silent failure is the worst kind of failure.

### 2.6 MCP integration

PydanticAI ships an `MCPClient` adapter that wraps an MCP server's tool list as PydanticAI tools. The integration is 5 lines:

```python
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent

server = MCPServerStdio(command="python", args=["my_mcp_server.py"])
agent = Agent(model="openai:gpt-4o", mcp_servers=[server], output_type=MyOutput)
async with agent:
    result = await agent.run("Use the MCP tools to answer this.")
```

The composition with [[../15 - MCP and Agentic Protocols/00 - Welcome to MCP and Agentic Protocols.md|MCP]] is identical to smolagents: the tools can come from any MCP server (Python, TypeScript, Go, Rust) and the agent uses them with full type validation. The capstone of this course (note 07) demonstrates a multi-framework RAG agent where the retrieval tool comes from an MCP server, the agent is a PydanticAI `Agent`, and the model is routed through LiteLLM.

---

## 3. Production Reality

### 3.1 Type safety is the differentiator

PydanticAI's production case is the company that ships "the LLM cannot produce malformed output" as a guarantee. The pattern: a PydanticAI agent is wrapped in a FastAPI endpoint, the agent's `output_type` is a `BaseModel`, and the FastAPI response model is the same `BaseModel`. The agent runs, validates its own output, and the FastAPI endpoint returns a response that is guaranteed to be valid against the model. If the validation retries are exhausted, the endpoint returns a 500 with a structured error. The caller never has to validate; the schema is the contract.

This is the pattern that makes the **LLM Edge Gateway** portfolio project composable: a Go backend that calls a Python PydanticAI agent for structured extraction, parses the response as a `BaseModel`, and passes it downstream. The type safety is the property of the agent, not the Go backend.

### 3.2 Latency profile

A PydanticAI agent run has the same latency profile as any other tool-calling agent: 1 LLM turn per tool call, plus 1 LLM turn for the final response. A 5-tool run with GPT-4o is typically 3-6 seconds. A 5-tool run with Claude Sonnet 4.5 is 4-8 seconds. Validation retries add 1-2 seconds per retry. The framework overhead is negligible (<5ms per run).

For sub-100ms responses, PydanticAI is the wrong tool — use a pre-scripted workflow or a smaller model with a tighter prompt. For multi-second responses with structured output, PydanticAI is the right tool.

### 3.3 Cost profile

PydanticAI's validation retries are the main cost driver. A 5-tool run that succeeds on the first try costs the same as a 5-tool run in any other framework. A 5-tool run that needs 2 validation retries costs 7 LLM turns. The `Usage` object on `RunContext` tracks the total cost, and the framework's `ModelSettings` accepts a `max_tokens` and `temperature` for cost control.

For cost-sensitive workloads, the [[../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM.md|LiteLLM]] integration gives you Redis semantic caching for free: repeated queries (semantically similar, threshold 0.95) return the cached response in <10ms with zero LLM cost. Combined with PydanticAI's structured output, you get the "best of both" — type safety at the framework boundary, cost efficiency at the model boundary.

### 3.4 Production case — a structured extraction service

A common 2026 production pattern is **structured extraction**: a PydanticAI agent that takes a free-form input (an email, a support ticket, a research paper) and returns a structured `BaseModel` (priority, category, entities, action items). The pattern is used by:

- Customer support triage: input is a ticket, output is `{priority, category, suggested_agent, sla_hours}`.
- Research digest: input is a paper, output is `{title, authors, claims, methodology, datasets}`.
- Email routing: input is an email, output is `{to, priority, tags, suggested_reply}`.

The same agent can be reused for many extraction tasks by changing the `output_type` and the system prompt. The validation guarantees mean the consumer code does not need to defensive-code against malformed output.

### 3.5 Failure modes

| Failure mode | Symptom | Fix |
|--------------|---------|-----|
| LLM cannot produce valid output | `ValidationError` after retries | Increase `validation_retries`; provide better system prompt with examples |
| Tool receives wrong type | LLM re-prompted, eventually succeeds | Improve tool docstring; add `Field(description=...)` to the Pydantic model |
| Streaming output breaks UI | Partial JSON in the response | Use `output_type=BaseModel` and stream structured deltas, not raw text |
| MCP server disconnects | Tool call fails with connection error | Wrap in retry with backoff; check server health before `agent.run` |
| Dependencies not injected | `ctx.deps` is `None` | Always pass `deps=...` to `agent.run()` |

### 3.6 Comparison: PydanticAI vs the other five frameworks

| Framework | Type safety | Best for | Worst for |
|-----------|:-----------:|----------|-----------|
| **PydanticAI** | ✅ Compile-time | Production backends, structured extraction | Code-as-action, rapid prototyping |
| **smolagents** | ⚠️ String-typed | Composable multi-step workflows | Production backends with strict contracts |
| **transformers.agents** | ⚠️ String-typed | Research, HF model ecosystem | Production backends |
| **OpenAI Agents SDK** | ⚠️ Schema-typed | OpenAI-only stacks | Multi-provider, non-OpenAI models |
| **Google ADK** | ⚠️ Schema-typed | GCP deployments | Non-Google stacks |
| **CrewAI 1.0** | ⚠️ String-typed | Multi-agent role-playing | Structured single-agent workflows |

---

## 4. Code in Practice

### 4.1 Minimal example: typed output

```python
# 🔒 MINIMAL: PydanticAI with typed output
# Install: pip install pydantic-ai

from pydantic import BaseModel, Field
from pydantic_ai import Agent
import os

class CityInfo(BaseModel):
    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Approximate population")
    timezone: str = Field(description="IANA timezone, e.g. 'America/Bogota'")

agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=CityInfo,
    system_prompt="You are a geography expert. Answer the user's question about a city.",
)

result = agent.run_sync("Tell me about Medellín, Colombia.")
print(type(result.output))        # <class 'CityInfo'>
print(result.output.name)         # "Medellín"
print(result.output.country)      # "Colombia"
print(result.output.population)   # 2500000
print(result.output.timezone)     # "America/Bogota"
```

### 4.2 Tools with typed arguments and dependency injection

```python
# TOOLS: PydanticAI tools with typed arguments and deps
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    db: object  # would be a real Database client in production
    user_id: str

class SupportResponse(BaseModel):
    found: bool
    answer: str | None
    related_articles: list[str] = Field(default_factory=list)

agent = Agent(
    model="anthropic:claude-sonnet-4.5",
    deps_type=Deps,
    output_type=SupportResponse,
    system_prompt="You are a customer support agent. Use the tools to find answers.",
)

@agent.tool
async def search_docs(ctx: RunContext[Deps], query: str, max_results: int = 3) -> list[dict]:
    """Search the support documentation.

    Args:
        query: Search query, e.g. "refund policy"
        max_results: Max docs to return, between 1 and 10
    """
    # ctx.deps.db is the real database connection
    return await ctx.deps.db.search(query, limit=max_results)

# Run with explicit dependencies
deps = Deps(db=fake_db, user_id="user_42")
result = agent.run_sync("What is the refund policy?", deps=deps)
print(result.output.found)            # True
print(result.output.answer)          # "Refunds are available within 30 days..."
print(result.output.related_articles)  # ["refund-policy", "shipping-info", "returns"]
```

### 4.3 Streaming structured output via FastAPI + WebSocket

```python
# STREAMING: PydanticAI + FastAPI WebSocket for real-time structured output
from fastapi import FastAPI, WebSocket
from pydantic_ai import Agent

app = FastAPI()
agent = Agent(model="openai:gpt-4o", output_type=CityInfo)

@app.websocket("/ws/city")
async def city_info(ws: WebSocket):
    await ws.accept()
    query = await ws.receive_text()
    async with agent.iter(query) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                # Stream the model's reasoning as it generates
                async for chunk in agent.stream_text(node):
                    await ws.send_json({"event": "token", "data": chunk})
            elif Agent.is_end_node(node):
                # Send the final validated output
                await ws.send_json({"event": "final", "data": run.result.output.model_dump()})
    await ws.close()
```

### 4.4 Multi-provider via LiteLLM fallback

```python
# MULTI-PROVIDER: PydanticAI + LiteLLM fallback chain
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai import Agent

# Try Claude first, fall back to GPT-4o if Claude is overloaded
model = FallbackModel(
    AnthropicModel("claude-sonnet-4.5"),
    OpenAIModel("gpt-4o"),
)

agent = Agent(model=model, output_type=CityInfo)

# If Claude returns 529 (overloaded) or 5xx, the framework retries with GPT-4o.
# The output_type is validated against the same CityInfo model regardless of provider.
result = agent.run_sync("Tell me about Bogotá.")
```

### 4.5 Common pitfalls

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| Forgetting `Field(description=...)` on `BaseModel` fields | LLM produces wrong values | Add explicit descriptions; descriptions are part of the schema |
| Mutable default in `BaseModel` | `ValueError` at import | Use `Field(default_factory=list)` |
| `output_type` is a `dict[str, Any]` | No validation, no autocomplete | Use a typed `BaseModel`; the framework's main value is type safety |
| Streaming partial JSON | UI shows broken JSON | Use `output_type=BaseModel` and stream structured deltas |
| Tool returns a `BaseModel` instance directly | LLM cannot read it | Tools must return JSON-serializable types (`dict`, `str`, `list`) |
| `ctx.deps` is `None` | Tool crashes with AttributeError | Always pass `deps=...` to `agent.run()` / `agent.run_sync()` |

> 💡 **Tip**: Use `agent.run_sync()` in tests and `await agent.run()` in production async code. `run_sync` is a blocking wrapper that creates its own event loop; it will deadlock if called from inside an existing event loop.

---

## 📦 Compression Code

```python
# NOTE: 02 - PydanticAI
# Repo: github.com/pydantic/pydantic-ai (MIT, 6k+ stars, v0.5+)
# Core abstraction: Agent[DepT, OutT] — generic over dependencies and output type
# Tool validation: Pydantic BaseModel schemas at the framework boundary, re-prompt on failure
# Output validation: Output_type BaseModel, validated before return, re-prompt on failure
# Dependency injection: RunContext[DepT] passed to every tool, request-scoped
# Streaming: agent.iter() yields ModelRequestNode, ModelResponseNode, ToolCallNode, EndNode
# Models: openai, anthropic, gemini, groq, mistral, cohere, huggingface, plus LiteLLM and Fallback
# MCP: MCPServerStdio and MCPServerHTTP — wrap any MCP server's tools as PydanticAI tools
# FastAPI integration: run_sync in tests, await run in async, iter() for streaming
# Cross-cuts: same MCP servers as smolagents, same LiteLLM, different execution model

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

class CityInfo(BaseModel):
    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Approximate population")

@dataclass
class Deps:
    user_id: str

agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=Deps,
    output_type=CityInfo,
    system_prompt="You are a geography expert.",
)

@agent.tool
async def get_local_time(ctx: RunContext[Deps], city: str) -> str:
    """Get the current local time for a city.

    Args:
        city: City name
    """
    return f"Local time in {city}: 14:30"  # mock

result = agent.run_sync("Tell me about Medellín.", deps=Deps(user_id="u1"))
print(result.output.name)        # "Medellín"
print(result.output.population)  # 2500000
```

## 🎯 Key Takeaways

- **PydanticAI makes the LLM-to-Python boundary type-safe** — the same `BaseModel` validates tool arguments and final responses
- **The `Agent[DepT, OutT]` signature** is the framework's contract: dependencies and outputs are types, not strings
- **Validation retries** are the production safety net — the LLM is re-prompted until it produces valid output, or the framework fails loudly
- **Dependency injection via `RunContext`** replaces globals with a request-scoped deps object — testable, mockable, type-checked
- **Streaming with `agent.iter()`** integrates naturally with FastAPI WebSocket and SSE for real-time structured output

## References

- PydanticAI documentation: https://ai.pydantic.dev/
- PydanticAI GitHub: https://github.com/pydantic/pydantic-ai
- PydanticAI examples: https://ai.pydantic.dev/examples/
- Pydantic v2 documentation: https://docs.pydantic.dev/
- FastAPI integration guide: https://ai.pydantic.dev/agents/#running-agents
- MCP for PydanticAI: https://ai.pydantic.dev/mcp/
- LiteLLM model adapter: https://ai.pydantic.dev/models/#litellm
- Streaming API: https://ai.pydantic.dev/agents/#streaming
