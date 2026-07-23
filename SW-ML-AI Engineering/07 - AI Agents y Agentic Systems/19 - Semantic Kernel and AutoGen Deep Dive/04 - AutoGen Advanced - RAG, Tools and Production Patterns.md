# 🎯 04 - AutoGen Advanced — RAG, Tools and Production Patterns

> **Wire AutoGen agents to production systems. Tool registration, RAG integration, error recovery, swarm patterns, and deployment patterns.**

## 🎯 Learning Objectives
- Register tools and functions that AutoGen agents can invoke automatically
- Build RAG-augmented agents with vector databases (Qdrant, pgvector, Azure AI Search)
- Implement error recovery with retries, fallbacks, and structured exception handling
- Configure swarm patterns for handoff-based agent collaboration
- Add structured outputs (Pydantic) for tool returns
- Deploy AutoGen to production with tracing, rate limiting, and cost controls

## Introduction

Note 03 covered the foundation: agent types, GroupChat, termination conditions, and code execution. But production agents need more than multi-agent conversation — they need **tools, retrieval, error handling, and deployment patterns**. This note covers the patterns that turn AutoGen from a research framework into a production-grade agent system.

The tool integration story is the first gap. AutoGen v0.5 supports **function-calling tools** natively: register a Python function as a tool, the agent calls it via the LLM's function-calling API. This is the same pattern as Semantic Kernel's `@kernel_function` (Note 01) and Instructor/Outlines (covered in [[06 - Large Language Models/22 - Instructor and Structured Generation]]) — but expressed in AutoGen's actor model.

The RAG integration story is the second. An agent with retrieval can ground its answers in private data; this is the most common production use case. AutoGen v0.5 supports retrieval via `Memory` extensions: vector databases, keyword indexes, and episodic stores.

Error recovery is the third gap. Production agents fail for many reasons: API rate limits, tool exceptions, agent loops, hallucinated tool arguments. The patterns in this note — retry policies, fallback tools, structured exception handling — turn AutoGen from a research tool into a hardened production system.

By the end of this note you will have built an AutoGen agent with 5+ tools, RAG retrieval, retry policies, and deployment-ready error recovery — the patterns that show up in real production deployments.

![AutoGen tools and RAG architecture](https://learn.microsoft.com/en-us/semantic-kernel/media/memories.png)

---

## 1. Tool Registration

### 1.1 Function-calling tools

The simplest tool integration: register a Python function:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

def search_documents(query: str, k: int = 5) -> list[dict]:
    """Search internal documents for relevant context.
    
    Args:
        query: The search query.
        k: Number of results to return.
    
    Returns:
        List of {title, text, score} dicts.
    """
    return vector_store.search(query, k=k)

search_tool = FunctionTool(
    search_documents,
    name="search_documents",
    description="Search internal documents for relevant context using semantic similarity.",
)

researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[search_tool],
    system_message="You are a research analyst. Use the search_documents tool to gather information before answering.",
)
```

The agent's LLM automatically calls `search_documents` based on the conversation. The function signature is parsed for argument types; the docstring for the description.

For **async** tools:

```python
import aiohttp

async def fetch_url(url: str) -> str:
    """Fetch a URL and return the text content."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

fetch_tool = FunctionTool(fetch_url, name="fetch_url", description="Fetch a URL.")
```

AutoGen runs async tools automatically. No special configuration needed.

### 1.2 MCP tools

AutoGen v0.5 supports the **Model Context Protocol** (MCP, covered in [[07 - AI Agents y Agentic Systems/15 - MCP and Agentic Protocols]]) for standardized tool servers:

```python
from autogen_ext.tools.mcp import McpTool

mcp_tool = McpTool.from_server_command(
    server_command="npx -y @modelcontextprotocol/server-filesystem /data",
    tool_name="read_file",
    description="Read a file from the local filesystem",
)

researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[mcp_tool, search_tool],
    system_message="Use the read_file tool for filesystem access, search_documents for knowledge base.",
)
```

This makes AutoGen tools **portable** across frameworks: an MCP server built for one framework works in AutoGen, LangGraph, Semantic Kernel, etc.

### 1.3 Pydantic-typed tool returns

Use Pydantic models for structured tool returns:

```python
from pydantic import BaseModel
from typing import Literal

class SearchResult(BaseModel):
    title: str
    snippet: str
    source: str
    relevance: float

class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_found: int

def search_typed(query: str, k: int = 5) -> SearchResponse:
    """Search internal documents with typed results."""
    results = vector_store.search(query, k=k)
    return SearchResponse(
        results=[SearchResult(title=r.title, snippet=r.text, source=r.source, relevance=r.score) for r in results],
        total_found=len(results),
    )

typed_tool = FunctionTool(search_typed, name="search_documents", description="...")
```

The agent sees a typed response and can reason about its structure. The Pydantic model integrates with the structured-output libraries from [[06 - Large Language Models/22 - Instructor and Structured Generation]].

---

## 2. RAG Integration

The most common production pattern: AutoGen agent + vector database.

### 2.1 With Qdrant

```python
from qdrant_client import QdrantClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

client = QdrantClient(url="http://localhost:6333", collection_name="documents")

def search_qdrant(query: str, k: int = 5) -> list[dict]:
    """Search the Qdrant vector store for documents matching the query.
    
    Args:
        query: The query string.
        k: Number of results.
    
    Returns:
        List of {text, score, payload} dicts.
    """
    # Embed the query
    from openai import OpenAI
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    query_embedding = openai.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
    
    # Search Qdrant
    hits = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=k,
    )
    
    return [{"text": h.payload["text"], "score": h.score, "payload": h.payload} for h in hits]

qdrant_tool = FunctionTool(search_qdrant, name="search_documents", description="...")

researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[qdrant_tool],
    system_message="Use search_documents to retrieve relevant context before answering.",
)
```

The agent calls `search_documents` whenever it needs context. The Qdrant search returns semantic hits. The agent reasons about which results are relevant.

### 2.2 With Azure AI Search

For Azure-native deployments (covered in [[10 - Cloud, Infra y Backend/22 - Cloud Computing]]):

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="rag_documents",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
)

def search_azure(query: str, k: int = 5) -> str:
    """Search Azure AI Search with hybrid vector + keyword.
    
    Args:
        query: The query string.
        k: Number of results.
    """
    results = search_client.search(
        search_text=query,
        top=k,
        search_mode="any",
        query_type="semantic",
        semantic_configuration_name="default",
    )
    return "\n\n".join([
        f"[{r['@search.score']:.2f}] {r['content']}"
        for r in results
    ])

azure_search_tool = FunctionTool(search_azure, name="search_azure", description="...")
```

Azure AI Search supports **hybrid search**: vector + keyword + semantic reranking, all in one query.

### 2.3 With LangChain retriever (interop)

For LangChain-based stacks, AutoGen can wrap any LangChain retriever:

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

lc_qdrant = Qdrant(client=client, collection_name="documents", embeddings=OpenAIEmbeddings())
retriever = lc_qdrant.as_retriever(search_kwargs={"k": 5})

def search_via_langchain(query: str) -> str:
    """Search documents via the LangChain retriever."""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

lc_tool = FunctionTool(search_via_langchain, name="search_documents", description="...")
```

This pattern lets you reuse existing LangChain infrastructure (loaders, splitters, retrievers) within AutoGen agents.

---

## 3. Error Recovery and Resilience

Production agents fail. The patterns in this section turn a research demo into a hardened system.

### 3.1 Retry policy on LLM calls

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

# Wrap the model client with retry logic
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,  # AutoGen built-in retries
)
```

AutoGen retries transient errors (rate limits, timeouts) by default. For custom retry policies:

```python
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RateLimitError),
)
async def call_llm_with_retry(client, messages):
    return await client.create(messages=messages)
```

### 3.2 Tool fallback

When a tool fails, fallback to a default value or a simpler tool:

```python
def search_documents_with_fallback(query: str, k: int = 5) -> list[dict]:
    """Search with fallback to keyword-only if vector search fails."""
    try:
        return semantic_search(query, k)
    except VectorSearchError as e:
        logger.warning(f"Vector search failed: {e}; falling back to keyword")
        return keyword_search(query, k)

search_tool = FunctionTool(search_documents_with_fallback, name="search_documents", description="...")
```

Or use AutoGen's tool middleware:

```python
from autogen_core.tools import ToolException

def fallback_middleware(tool_call, next):
    """Catch tool exceptions and return a fallback response."""
    try:
        return next(tool_call)
    except ToolException as e:
        logger.warning(f"Tool {tool_call.name} failed: {e}; returning empty result")
        return {"error": str(e), "fallback": True}

researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[search_tool],
    tool_middleware=[fallback_middleware],
)
```

### 3.3 Termination on unrecoverable errors

If a tool consistently fails, terminate the conversation:

```python
from autogen_agentchat.conditions import FunctionCallTermination

# Terminate when a tool returns an error more than 3 times
class ErrorBudgetTermination:
    def __init__(self, max_errors: int = 3):
        self.error_count = 0
    
    async def check(self, messages):
        last_msg = messages[-1]
        if "error" in str(last_msg.content).lower():
            self.error_count += 1
            if self.error_count >= 3:
                return True
        return False

team = RoundRobinGroupChat(
    participants=[researcher, critic],
    termination_condition=ErrorBudgetTermination(max_errors=3),
)
```

This prevents the agent from looping on a broken tool. After 3 errors, the conversation ends and the orchestrator gets a "tool failure" result.

### 3.4 Structured exception responses

Tools should return structured error responses, not raise exceptions:

```python
class ToolResponse(BaseModel):
    success: bool
    data: any | None = None
    error: str | None = None

def safe_search(query: str) -> ToolResponse:
    try:
        results = vector_store.search(query)
        return ToolResponse(success=True, data=results)
    except Exception as e:
        return ToolResponse(success=False, error=str(e))
```

The agent sees the structured response and can reason about it: "the search failed, let me try a different query" vs an exception bubbling up.

---

## 4. Swarm Patterns — Handoff-Based Collaboration

For workflows without a central orchestrator, use **swarm patterns** where agents hand off to each other based on conditions:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm

# Define agents with handoff conditions
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    system_message="Research topics. If you find enough information, hand off to the writer.",
    handoffs=[WriterAgent],  # hand off to writer when done
)

writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="Write the final answer. If the writer needs more research, hand off back to the researcher.",
    handoffs=[ResearcherAgent],
)

team = Swarm([researcher, writer])
```

The swarm decides who to hand off to based on the conversation state. There's no central `GroupChatManager`; the agents themselves route. This is the **swarm pattern** from LangGraph Swarm but with AutoGen's actor model.

For more complex swarms, use the `DiGraphBuilder` API:

```python
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

builder = DiGraphBuilder()
researcher_node = builder.add_node(researcher).add_edge(...).add_edge(...)
writer_node = builder.add_node(writer).add_edge(...).add_edge(...)
critic_node = builder.add_node(critic).add_edge(...).add_edge(...)
graph = builder.build()

team = GraphFlow(graph)
```

This is the canonical **graph orchestration** pattern: explicit nodes and edges, conditional branching, cyclic re-entry. Close to LangGraph's mental model but expressed in AutoGen's actor primitives.

---

## 5. Cost Control

### 5.1 Token tracking

```python
from autogen_core.models import ModelUsage

total_tokens = {"prompt": 0, "completion": 0}

async def count_tokens(result, ctx):
    """Callback that runs after each model call."""
    if hasattr(result, "usage"):
        total_tokens["prompt"] += result.usage.prompt_tokens
        total_tokens["completion"] += result.usage.completion_tokens
    print(f"Total tokens: {total_tokens}")

team = RoundRobinGroupChat(
    participants=[researcher, critic],
    termination_condition=MaxMessageTermination(max_messages=10),
)
await team.run(task="...", output_callbacks=[count_tokens])
```

### 5.2 Cost limits

```python
from autogen_agentchat.conditions import TokenUsageTermination

team = RoundRobinGroupChat(
    participants=[researcher, critic],
    termination_condition=TokenUsageTermination(max_tokens=50_000),  # kill at $0.10 (gpt-4o-mini)
)

# Or use a custom condition
class CostTermination:
    def __init__(self, max_dollars: float):
        self.max_dollars = max_dollars
        self.spent = 0.0
        self.token_costs = {
            "gpt-4o": (0.005, 0.015),  # $/1K tokens (input, output)
            "gpt-4o-mini": (0.00015, 0.0006),
            "claude-3-5-sonnet": (0.003, 0.015),
        }
    
    async def check(self, messages):
        # Update spent based on the latest usage
        last_msg = messages[-1]
        if hasattr(last_msg, "usage") and last_msg.usage:
            model = last_msg.metadata.get("model", "gpt-4o-mini")
            input_cost, output_cost = self.token_costs.get(model, (0.001, 0.002))
            self.spent += (last_msg.usage.prompt_tokens / 1000) * input_cost
            self.spent += (last_msg.usage.completion_tokens / 1000) * output_cost
        return self.spent >= self.max_dollars
```

This is the **production-cost guardrail**. No agent run can spend more than $X. Critical for multi-tenant deployments where a runaway agent would otherwise rack up thousands of dollars in LLM calls.

---

## 6. Rate Limiting

For high-throughput deployments, rate-limit at the team level:

```python
from autogen_ext.rate_limiters import TokenBucketRateLimiter

rate_limiter = TokenBucketRateLimiter(
    max_tokens=100_000,  # 100K tokens per minute
    refill_rate=100_000 / 60,  # refill per second
)

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    rate_limiter=rate_limiter,
)
```

For per-tenant rate limits, instantiate one rate limiter per tenant and one team per tenant.

---

## 7. Deployment to Production

### 7.1 FastAPI wrapper

Wrap the team in a FastAPI endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    task: str
    thread_id: str | None = None

class QueryResponse(BaseModel):
    result: str
    thread_id: str
    tokens_used: int

@app.post("/agent/run", response_model=QueryResponse)
async def run_agent(req: QueryRequest) -> QueryResponse:
    result = await team.run(task=req.task)
    return QueryResponse(
        result=result.messages[-1].content,
        thread_id=req.thread_id or "default",
        tokens_used=total_tokens["prompt"] + total_tokens["completion"],
    )
```

### 7.2 Streaming response

```python
from fastapi.responses import StreamingResponse

@app.post("/agent/stream")
async def stream_agent(req: QueryRequest):
    async def event_stream():
        async for message in team.run_stream(task=req.task):
            yield f"data: {message.content}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 7.3 LangFuse integration

Every agent run traces to LangFuse (covered in [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability]]):

```python
from langfuse import Langfuse, observe

langfuse = Langfuse()

@observe()
async def run_agent_traced(task: str) -> str:
    result = await team.run(task=task)
    return result.messages[-1].content
```

The `@observe` decorator wraps the entire agent run; all LLM calls become child spans in LangFuse.

---

## 8. Antipatterns

### 8.1 Antipattern 1: Tools that raise raw exceptions

```python
# ❌ Uncaught exception bubbles up and crashes the agent
def search(query: str):
    return vector_store.search(query)  # raises if DB down

# ✅ Catch exceptions, return structured error
def search(query: str) -> dict:
    try:
        results = vector_store.search(query)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 8.2 Antipattern 2: Long-running code without timeouts

```python
# ❌ No timeout: hangs forever on long queries
def fetch_url(url: str) -> str:
    return requests.get(url).text  # no timeout

# ✅ Always set timeouts
def fetch_url(url: str) -> str:
    return requests.get(url, timeout=10).text
```

### 8.3 Antipattern 3: No cost tracking

```python
# ❌ Agent loops, $5000 bill
team = RoundRobinGroupChat(participants=[researcher, critic], ...)

# ✅ Always set cost/token guardrails
team = RoundRobinGroupChat(
    participants=[researcher, critic],
    termination_condition=TokenUsageTermination(max_tokens=100_000) | CostTermination(max_dollars=1.0),
)
```

### 8.4 Antipattern 4: Sharing tools across agents without permissions

```python
# ❌ The user-facing agent can read filesystem / execute shell
user_facing = AssistantAgent(name="user", tools=[read_file, execute_shell], ...)

# ✅ Separate agents for different trust levels
researcher = AssistantAgent(name="researcher", tools=[search_documents], ...)  # safe
admin = AssistantAgent(name="admin", tools=[read_file, execute_shell], ...)  # restricted
```

### 8.5 Antipattern 5: Forgetting to register Pydantic return types

```python
# ❌ Untyped returns force the agent to guess
def search(query: str) -> list[dict]: return ...

# ✅ Typed returns let the agent reason about structure
def search(query: str) -> SearchResponse: return SearchResponse(...)
```

---

## 🎯 Key Takeaways

- Tools are registered as `FunctionTool`; Async tools, MCP tools, and Pydantic-typed tools all supported.
- RAG integration via Qdrant, Azure AI Search, or wrapping a LangChain retriever.
- Error recovery: retry policies, tool fallback, structured exceptions, termination on error budget.
- Swarm patterns: handoff-based collaboration without central orchestrator; `DiGraphBuilder` for explicit graphs.
- Cost control via `TokenUsageTermination`, custom cost terminators, and rate limiters.
- Deployment: FastAPI wrapper with streaming; LangFuse for observability.
- Avoid raw exceptions, no timeouts, no cost tracking, shared trust levels, untyped returns.

## References

- AutoGen Tools — [microsoft.github.io/autogen/dev/user-guide/core-user-guide/framework/tools.html](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/framework/tools.html)
- AutoGen RAG — [microsoft.github.io/autogen/dev/user-guide/core-user-guide/framework/rag.html](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/framework/rag.html)
- MCP integration — [modelcontextprotocol.io](https://modelcontextprotocol.io/)
- [[06 - Large Language Models/12 - Production RAG|Production RAG]] — RAG fundamentals
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — Pydantic validation for tool returns
- [[07 - AI Agents y Agentic Systems/15 - MCP and Agentic Protocols|MCP and Agentic Protocols]] — MCP server tooling
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — graph orchestration alternative
- [[07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive/03 - AutoGen Fundamentals - Conversable Agents and GroupChat|Note 03 — AutoGen Fundamentals]]
- [[07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive/05 - Capstone - Multi-Framework Enterprise Agent|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — observability for AutoGen traces
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — service deployment patterns
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — Azure deployment
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant, pgvector, Milvus, ChromaDB, Pinecone