# 🎯 02 - Semantic Kernel Process Framework and Memory

> **Beyond single-function calls: stateful multi-step workflows. Kernel Memory for RAG. Plan and execute a 5-step research pipeline with the Process Framework.**

## 🎯 Learning Objectives
- Build stateful multi-step workflows with the Process Framework (Stepped vs Cyclic)
- Persist process state across crashes with the new KernelProcess runtime (Temporal-style)
- Implement Kernel Memory for vector + episodic retrieval
- Choose between FunctionCallingPlanner, SequentialPlanner, and StepwisePlanner
- Integrate Azure AI Search as a managed vector backend
- Compose plugins with memory and process for a production RAG agent

## Introduction

Note 01 covered the foundation: a Kernel registers services and plugins; the planner invokes functions based on LLM reasoning. But real agents are not single-function calls. They are **multi-step workflows** with state, branching, retries, and long-running operations. The **Process Framework** is SK's answer to LangGraph's cyclic state machines — it lets you model a workflow as a directed graph of steps, each holding typed state, with conditional transitions and persistence.

The **Kernel Memory** abstraction handles retrieval. It treats vector search, keyword search, and episodic memory as composable memory sources; the kernel queries them in parallel and merges the results. This is the SK equivalent of LangChain's `RetrievalQA` and LlamaIndex's `QueryEngine` — but with first-class Azure AI Search integration.

Together, Process Framework + Kernel Memory give you the building blocks for production agents: long-running workflows with persisted state, retrieval-augmented context, and Azure-native deployment. The pattern maps directly to LangGraph subgraphs from [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]] — you could implement the same workflow in either; SK is the Microsoft-native choice.

```mermaid
stateDiagram-v2
    [*] --> Intake
    Intake --> Retrieve: step 1
    Retrieve --> Plan: step 2
    Plan --> Execute: step 3
    Execute --> Validate: step 4
    Validate --> Execute: retry on fail (cyclic)
    Validate --> [*]: success
    Validate --> HumanReview: ambiguous (stepped)
    HumanReview --> [*]: approved
```

![Kernel Memory architecture](https://learn.microsoft.com/en-us/semantic-kernel/media/memories.png)

---

## 1. The Process Framework — Stateful Workflows

### 1.1 SteppedProcess vs CyclicProcess

Two process patterns:

- **SteppedProcess** — linear sequence of steps with conditional branching. Each step has typed inputs and outputs. Use for predictable workflows (form intake → review → approval → notify).
- **CyclicProcess** — steps can re-enter previous steps based on conditions. Use for iterative refinement (plan → execute → validate → re-plan if validation fails).

A stepped process for customer onboarding:

```python
from semantic_kernel.processes import ProcessBuilder, KernelProcessStep
from semantic_kernel.kernel import Kernel
from pydantic import BaseModel


class IntakeState(BaseModel):
    customer_id: str
    name: str
    email: str


class IntakeStep(KernelProcessStep):
    @kernel_function(name="collect", description="Collect customer intake info")
    async def collect(self, ctx, input_data: dict) -> IntakeState:
        # Real implementation would call external API
        return IntakeState(
            customer_id=input_data["customer_id"],
            name=input_data["name"],
            email=input_data["email"],
        )


class ReviewStep(KernelProcessStep):
    @kernel_function(name="review", description="Review intake for completeness")
    async def review(self, ctx, intake: IntakeState) -> dict:
        valid = bool(intake.email and intake.name)
        return {"valid": valid, "intake": intake}


class NotifyStep(KernelProcessStep):
    @kernel_function(name="notify", description="Send welcome email")
    async def notify(self, ctx, intake: IntakeState) -> dict:
        return {"status": "notified", "customer_id": intake.customer_id}


# Build the process
builder = ProcessBuilder("CustomerOnboarding")
intake_step = builder.add_step(IntakeStep)
review_step = builder.add_step(ReviewStep)
notify_step = builder.add_step(NotifyStep)

builder.on_input_event("start").send_event_to(intake_step, function_name="collect")
intake_step.on_function_result("collect").send_event_to(review_step, parameter_name="intake")
review_step.on_function_result("review").send_event_to(
    notify_step,
    condition=lambda result: result["valid"],
    parameter_name="intake",
)

process = builder.build()
```

The builder API is fluent: chain `on_input_event().send_event_to()` to wire steps. Each step's `on_function_result()` callback fires when its function returns, optionally filtered by condition.

For a **cyclic process** with retry:

```python
class PlanStep(KernelProcessStep):
    @kernel_function(name="plan", description="Generate execution plan")
    async def plan(self, ctx, input_data: dict) -> dict:
        # Generate plan via LLM
        plan = await generate_plan(input_data["goal"])
        return {"plan": plan}


class ExecuteStep(KernelProcessStep):
    @kernel_function(name="execute", description="Execute plan step")
    async def execute(self, ctx, plan: dict) -> dict:
        # Execute via LLM
        result = await execute_step(plan["plan"])
        return {"result": result}


class ValidateStep(KernelProcessStep):
    @kernel_function(name="validate", description="Validate execution result")
    async def validate(self, ctx, execution: dict) -> dict:
        # LLM-as-judge
        valid = await validate_result(execution["result"])
        return {"valid": valid, "result": execution["result"]}


# Build cyclic process
builder = ProcessBuilder("ResearchPipeline")
plan_step = builder.add_step(PlanStep)
exec_step = builder.add_step(ExecuteStep)
val_step = builder.add_step(ValidateStep)

builder.on_input_event("start").send_event_to(plan_step, function_name="plan")
plan_step.on_function_result("plan").send_event_to(exec_step, parameter_name="plan")
exec_step.on_function_result("execute").send_event_to(val_step, parameter_name="execution")

# Cyclic: on validation failure, retry plan
val_step.on_function_result("validate").send_event_to(
    plan_step,
    function_name="plan",
    condition=lambda result: not result["valid"],
    parameter_name=lambda result: {"goal": result["result"].get("original_goal")},
)

# On success, end
val_step.on_function_result("validate").send_event_to(
    "end",
    condition=lambda result: result["valid"],
    parameter_name=lambda result: result["result"],
)
```

The same builder API handles cycles via re-entrant transitions. This is SK's answer to LangGraph's cyclic state machines.

### 1.2 State persistence with KernelProcess runtime (2026)

The `KernelProcess` runtime added in late 2025 supports **durable execution**: process state is checkpointed to a store (Postgres, Redis, Azure Cosmos DB) on every step transition. If the process crashes mid-execution, it resumes from the last checkpoint.

```python
from semantic_kernel.processes.runtime import KernelProcessRuntime
from semantic_kernel.processes.storage import PostgresCheckpointStore

runtime = KernelProcessRuntime(
    checkpoint_store=PostgresCheckpointStore(
        connection_string="postgresql://user:pass@localhost:5432/sk_processes"
    )
)

# Start a process (returns immediately; the process runs in the background)
process_handle = await runtime.start_process(
    process=process,
    initial_event={"goal": "Research the impact of AI on software engineering jobs"},
    thread_id="research_001",
)

# Resume after crash
await runtime.resume_process(thread_id="research_001")
```

This is Temporal-style durable execution, but built into the kernel. For agents that run for minutes to hours, this is the missing piece that prevents losing work on transient failures.

---

## 2. Kernel Memory — Vector + Episodic

### 2.1 The memory abstraction

`KernelMemory` is a unified interface for retrieval. It treats vector search, keyword search, and structured queries as composable sources.

```python
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

memory = SemanticTextMemory(
    storage=VolatileMemoryStore(),
    embeddings_generator=OpenAITextEmbedding(
        ai_model_id="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)

# Save
await memory.save_information(
    collection="documents",
    id="doc_1",
    text="Semantic Kernel is Microsoft's enterprise SDK for AI agents.",
    description="SK overview",
)

# Search
results = await memory.search("documents", "What is Semantic Kernel?", limit=3)
for result in results:
    print(f"[{result.relevance:.2f}] {result.text}")
```

The default `VolatileMemoryStore` keeps everything in memory — fine for development. For production, swap in a persistent backend:

| Backend | Persistence | Performance | When to use |
|---------|:-----------:|:-----------:|-------------|
| `VolatileMemoryStore` | ❌ | ⚡ Fastest | Dev, tests |
| `PostgresMemoryStore` | ✅ | 🟢 Fast | Single-region production |
| `AzureAISearchMemoryStore` | ✅ | 🟢 Fast | Multi-region, Azure-native |
| `QdrantMemoryStore` | ✅ | 🟢 Fast | Self-hosted, low-latency |
| `PineconeMemoryStore` | ✅ | 🟢 Fast | Serverless, multi-cloud |
| `RedisMemoryStore` | ✅ | ⚡ Fastest | Existing Redis infra |

For Azure-native deployments, `AzureAISearchMemoryStore` is the canonical choice. It supports hybrid search (vector + keyword + semantic reranking) and integrates with Azure's identity, RBAC, and Private Link.

### 2.2 Multiple memory sources

Compose vector + keyword search for hybrid retrieval:

```python
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore

# Vector-backed memory
vector_memory = SemanticTextMemory(
    storage=AzureCognitiveSearchMemoryStore(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        api_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name="documents",
    ),
    embeddings_generator=OpenAITextEmbedding(
        ai_model_id="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)

# Use in a plugin
class DocumentSearchPlugin:
    def __init__(self, memory: SemanticTextMemory):
        self._memory = memory
    
    @kernel_function(name="search", description="Search internal documents for relevant context")
    async def search(self, query: Annotated[str, "Search query"]) -> str:
        results = await self._memory.search("documents", query, limit=5)
        return "\n\n".join([r.text for r in results])

kernel.add_plugin(DocumentSearchPlugin(vector_memory), plugin_name="Documents")
```

The plugin's `search` function returns retrieved context. The planner invokes it based on the user's request. This is the **canonical RAG pattern** in SK — `Documents` plugin + chat service + function calling.

### 2.3 Episodic memory

For agents that need to remember past conversations, use episodic memory:

```python
from semantic_kernel.memory import EpisodicMemory

memory = EpisodicMemory(
    storage=RedisMemoryStore(redis_url="redis://localhost:6379"),
)

# Save conversation turn
await memory.add_turn(
    session_id="user_123_session_abc",
    user_id="user_123",
    role="user",
    content="What's the weather in Paris?",
)

# Later, retrieve the conversation history
history = await memory.get_history("user_123_session_abc", limit=10)
```

Episodic memory is indexed by session ID and supports time-range queries. Use it for chat history, tool call records, and any other sequential data.

---

## 3. Planners — Beyond FunctionCallingPlanner

### 3.1 FunctionCallingPlanner (recommended)

The default. Uses the LLM's native function calling to select functions. Fast, reliable, supported by all major providers.

```python
from semantic_kernel.planners import FunctionCallingPlanner

planner = FunctionCallingPlanner()
result = await planner.create_plan(goal="...", kernel=kernel)
```

### 3.2 StepwisePlanner

Decomposes a complex goal into multiple steps, executing them one by one with intermediate reasoning:

```python
from semantic_kernel.planners import StepwisePlanner

planner = StepwisePlanner(max_steps=10)
result = await planner.execute(goal="Research X, then write a report on Y", kernel=kernel)
```

Useful for research-style workflows where the LLM needs to reason about intermediate state. Slower than `FunctionCallingPlanner` but more flexible.

### 3.3 HandlebarsPlanner

Generates a Handlebars-templated program from the goal, then executes it. Best for complex multi-step workflows with conditional logic:

```python
from semantic_kernel.planners import HandlebarsPlanner

planner = HandlebarsPlanner()
result = await planner.execute(goal="...", kernel=kernel)
```

Deprecated in v1.x — use `FunctionCallingPlanner` or migrate to the Process Framework for complex workflows.

💡 **Tip:** In 2026 production, the right pattern is **`FunctionCallingPlanner` for single-task agents + the Process Framework for multi-step workflows**. The other planners are research tools, not production standards.

---

## 4. Azure AI Search Integration

Azure AI Search is Microsoft's managed vector + keyword search service. It integrates natively with Semantic Kernel:

```python
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)

memory = SemanticTextMemory(
    storage=AzureCognitiveSearchMemoryStore(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        api_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name="rag_documents",
    ),
    embeddings_generator=AzureTextEmbedding(
        deployment_name="text-embedding-3-small",
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    ),
)

# Hybrid search
results = await memory.search(
    collection="rag_documents",
    query="What is the impact of AI on healthcare?",
    limit=5,
    min_relevance=0.7,
)
```

Azure AI Search supports:
- **Vector search** (HNSW index)
- **Keyword search** (BM25)
- **Semantic reranking** (LLM-based)
- **Hybrid queries** (vector + keyword combined)
- **Faceting and filtering** (metadata-based filters)

For production RAG on Azure, **Azure AI Search is the canonical vector backend**. It scales to billions of vectors, supports multi-region replication, and integrates with Azure RBAC and Private Link.

---

## 5. A Production RAG Agent — Putting It Together

```python
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from semantic_kernel.functions import kernel_function
from semantic_kernel.planners import FunctionCallingPlanner
from typing import Annotated


class CalendarPlugin:
    @kernel_function(name="get_events", description="Get upcoming calendar events for a user.")
    def get_events(self, user_id: Annotated[str, "User ID"]) -> list[dict]:
        return [{"title": "Q3 planning", "date": "2026-07-24"}]


class DocumentSearchPlugin:
    def __init__(self, memory: SemanticTextMemory):
        self._memory = memory
    
    @kernel_function(name="search", description="Search internal documents for relevant context.")
    async def search(self, query: Annotated[str, "Search query"], k: Annotated[int, "Top-k results"] = 5) -> str:
        results = await self._memory.search("rag_documents", query, limit=k)
        return "\n\n---\n\n".join([r.text for r in results])


async def main():
    # 1. Build the kernel
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            deployment_name="gpt-4o",
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    )
    
    # 2. Configure memory
    memory = SemanticTextMemory(
        storage=AzureCognitiveSearchMemoryStore(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            api_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name="rag_documents",
        ),
        embeddings_generator=AzureTextEmbedding(
            deployment_name="text-embedding-3-small",
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
    )
    
    # 3. Register plugins
    kernel.add_plugin(CalendarPlugin(), plugin_name="Calendar")
    kernel.add_plugin(DocumentSearchPlugin(memory), plugin_name="Documents")
    
    # 4. Plan and execute
    planner = FunctionCallingPlanner()
    plan = await planner.create_plan(
        goal="Find Q3 planning documents and schedule a 30-minute meeting tomorrow at 2 PM to discuss them with the team.",
        kernel=kernel,
    )
    result = await plan.invoke(kernel=kernel)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

The kernel:
1. Routes the request through the planner
2. The LLM sees the registered functions: `Calendar.get_events`, `Documents.search`, plus the implicit date/time functions
3. Calls `Documents.search` first to retrieve Q3 planning docs
4. Then `Calendar.get_events` to find a free slot
5. Calls the implicit `schedule_meeting` function with the team's email addresses
6. Returns the final answer

This is a complete RAG agent in 40 lines of Python.

---

## 6. Cross-Framework Collaboration — Calling AutoGen from SK

In 2026, SK added `AutoGenProcessAdapter` for inter-framework collaboration. A SK kernel can dispatch a sub-task to an AutoGen GroupChat:

```python
from semantic_kernel.connectors.ai.auto_gen import AutoGenProcessAdapter

# Configure the AutoGen service
kernel.add_service(
    AutoGenProcessAdapter(
        endpoint="http://localhost:8001",  # AutoGen service URL
        timeout_seconds=60,
    )
)

# Use it as a plugin
class AutoGenDebatePlugin:
    def __init__(self, kernel):
        self._kernel = kernel
    
    @kernel_function(name="research_debate", description="Run a multi-agent debate on the topic.")
    async def research_debate(
        self,
        topic: Annotated[str, "Research topic"],
        rounds: Annotated[int, "Number of debate rounds"] = 3,
    ) -> str:
        from autogen import GroupChat, GroupChatManager
        
        # Configure AutoGen agents
        researcher = AssistantAgent("researcher", llm_config={...})
        critic = AssistantAgent("critic", llm_config={...})
        groupchat = GroupChat(agents=[researcher, critic], messages=[], max_round=rounds)
        manager = GroupChatManager(groupchat=groupchat, llm_config={...})
        
        # Run the debate
        result = await researcher.a_initiate_chat(manager, message=topic)
        return result.summary

kernel.add_plugin(AutoGenDebatePlugin(kernel), plugin_name="Debate")
```

This is the **cross-framework pattern**: SK for plugin orchestration, AutoGen for multi-agent debate. The capstone (Note 05) builds a full production system on this pattern.

---

## 7. Antipatterns

### 7.1 Antipattern 1: Using `StepwisePlanner` for every multi-step workflow

```python
# ❌ StepwisePlanner is slow and verbose for predictable workflows
planner = StepwisePlanner()
result = await planner.execute(goal="Run 5 sequential form steps", kernel=kernel)

# ✅ Use the Process Framework for predictable multi-step workflows
process = build_form_process()  # from section 1.1
result = await runtime.start_process(process, initial_event=...)
```

### 7.2 Antipattern 2: Using `VolatileMemoryStore` in production

```python
# ❌ State lost on restart
storage = VolatileMemoryStore()

# ✅ Postgres or Azure AI Search for production
storage = PostgresMemoryStore(connection_string="postgresql://...")
```

### 7.3 Antipattern 3: Mixing Pydantic and dict in plugin returns

```python
# ❌ Inconsistent return types confuse the LLM
@kernel_function(name="get_user")
def get_user(self, id: str) -> dict: return {...}  # dict

@kernel_function(name="get_order")
def get_order(self, id: str) -> Order: return Order(...)  # Pydantic

# ✅ Pick one. Pydantic v2 with .model_dump() is cleaner.
@kernel_function(name="get_user")
def get_user(self, id: str) -> User: return User(id=id, name="Alice")
# The planner handles .model_dump() automatically when registered with response_model
```

### 7.4 Antipattern 4: Not registering embeddings generator

```python
# ❌ Memory search returns empty because no embeddings generator
memory = SemanticTextMemory(storage=PostgresMemoryStore(...))  # no embeddings!

# ✅ Always pair storage with embeddings generator
memory = SemanticTextMemory(
    storage=PostgresMemoryStore(...),
    embeddings_generator=OpenAITextEmbedding(...),
)
```

### 7.5 Antipattern 5: Forgetting to handle step failures in cyclic processes

```python
# ❌ Infinite loop: validation never succeeds
val_step.on_function_result("validate").send_event_to(
    plan_step,
    function_name="plan",
    condition=lambda result: not result["valid"],
)

# ✅ Add a max-retry counter and human-review fallback
state.retry_count += 1
if state.retry_count > 3:
    val_step.on_function_result("validate").send_event_to("human_review")
else:
    val_step.on_function_result("validate").send_event_to(plan_step, ...)
```

---

## 🎯 Key Takeaways

- The Process Framework models multi-step workflows as directed graphs of typed steps with state transitions.
- SteppedProcess for linear workflows; CyclicProcess for iterative refinement.
- `KernelProcess` runtime (2025) adds durable execution with checkpointing — Temporal-style for SK.
- Kernel Memory: vector + keyword + episodic, swappable storage backends.
- Azure AI Search is the canonical Azure-native vector backend.
- `FunctionCallingPlanner` for single-task agents; Process Framework for multi-step workflows.
- Cross-framework collaboration via `AutoGenProcessAdapter` — SK + AutoGen + LangGraph all compose.
- Avoid stepwise planners for predictable flows, in-memory storage in production, mixed return types, missing embeddings, and unbounded retries.

## References

- Semantic Kernel Process Framework — [learn.microsoft.com/en-us/semantic-kernel/concepts/processes](https://learn.microsoft.com/en-us/semantic-kernel/concepts/processes)
- Kernel Memory — [learn.microsoft.com/en-us/semantic-kernel/concepts/memories](https://learn.microsoft.com/en-us/semantic-kernel/concepts/memories)
- Azure AI Search integration — [learn.microsoft.com/en-us/azure/search/search-what-is-azure-search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search)
- [[07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive/01 - Semantic Kernel Fundamentals - Kernel, Plugins, Functions|Note 01 — SK Fundamentals]]
- [[07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive/03 - AutoGen Fundamentals - Conversable Agents and GroupChat|Note 03 — AutoGen Fundamentals]]
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — cyclic state machine alternative
- [[06 - Large Language Models/12 - Production RAG|Production RAG]] — RAG fundamentals
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — vector DB landscape
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — Azure deployment patterns
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — Pydantic validation pattern
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — observability for SK traces
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — Phoenix spans for SK