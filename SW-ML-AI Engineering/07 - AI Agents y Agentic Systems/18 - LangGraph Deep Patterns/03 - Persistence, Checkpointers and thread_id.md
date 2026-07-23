# 💾 Persistence, Checkpointers and `thread_id`

Without persistence, every `app.invoke(...)` is a fresh start — the agent has no memory beyond the current call. With persistence, the same `thread_id` resumes the conversation across restarts, deploys, and days. The Multi-Agent Research System became production-ready the moment it adopted a checkpointer: a user can ask a question on Monday, leave, return on Friday, and pick up exactly where the synthesis was paused. This note is the bridge between "LangGraph as a fancy `if/else`" and "LangGraph as the runtime for a stateful AI product".

The persistence layer is small: pick a checkpointer (`MemorySaver` for testing, `PostgresSaver` for production), supply a `thread_id` in `config["configurable"]`, and the framework handles the rest. The hard parts — what to persist, how to handle sensitive state, how to enforce thread isolation, how to time-travel for debugging — are what this note covers.

## 🎯 Learning Objectives

- Wire `MemorySaver` and `PostgresSaver` checkpointers into a compiled graph.
- Use `thread_id` to scope conversations and resume across `invoke()` calls.
- Read and interpret `StateSnapshot` — the inspectable state of a thread.
- Time-travel debug with `get_state_history()` and `update_state()`.
- Decide which fields go in `state` vs `config["configurable"]` vs runtime context (Prometheus / OTel spans).
- Set up `PostgresSaver` for production with the right schema and connection pooling.
- Build user/tenant isolation patterns via `thread_id` prefixes.

## 1. The Problem: Stateless Invokes Are a Demo, Not a Product

```python
# Without persistence — every call is independent
result1 = app.invoke({"query": "Add Denver to the trip"})
result2 = app.invoke({"query": "Add flights"})
# result2 has no idea the trip exists.
```

This is what every LangGraph tutorial shows, and it's why most agents in the wild feel like toys. Real products need:

- **Multi-turn conversations**: "Yes, that hotel" must reference the prior hotel the user was shown.
- **Long-running workflows**: a research agent may take 10 minutes; an HTTP timeout can't lose the work.
- **Crash recovery**: if the FastAPI worker dies mid-execution, the conversation resumes from the last checkpoint.
- **Human approval gates**: see [[05 - Human-in-the-Loop with interrupt() and Command|note 05]] — persistence is the prerequisite.
- **Audit trail**: "what did the agent do at 14:32 last Tuesday?" requires every step stored.

All five are the same requirement: **serialize the graph state after every node, and rehydrate it on the next call**.

## 2. Checkpointers: MemorySaver vs PostgresSaver

A **checkpointer** is a storage adapter that LangGraph uses to save and restore graph state. Two built-in options cover 95% of real use cases.

### `MemorySaver` — In-Process, for Tests

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}
app.invoke({"query": "Hello"}, config)
app.invoke({"query": "Are you still there?"}, config)  # resumes thread-1
```

`MemorySaver` keeps state in a Python dict, scoped by `(thread_id, checkpoint_id)`. It dies with the process — restart, and the state is gone. Use it for unit tests and local dev only.

| Aspect | MemorySaver |
|--------|-------------|
| Storage | In-memory dict |
| Thread scope | Single process |
| Persistence | Lost on restart |
| Performance | O(1) read/write |
| Best for | Unit tests, examples, dev |

### `PostgresSaver` — Production Standard

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Connection string from env / secrets
DB_URI = "postgresql://user:pass@host:5432/langgraph"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # First-time setup — creates the required tables
    checkpointer.setup()

    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-42-session-1"}}
    app.invoke({"query": "Hello"}, config)
```

`PostgresSaver` serializes each checkpoint as a JSON row in dedicated tables (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`). The schema is auto-created by `setup()`. Queries use the indexed `(thread_id, checkpoint_ns, checkpoint_id)` triple to find the latest state for a thread in O(log n).

| Aspect | PostgresSaver |
|--------|---------------|
| Storage | Postgres tables (JSON) |
| Thread scope | Cross-process, cross-host |
| Persistence | Survives restarts |
| Performance | ~1-5ms per checkpoint write |
| Best for | Production, multi-worker FastAPI, audit trail |

> ⚠️ **Advertencia:** For high-throughput agents (>100 concurrent threads), configure Postgres with adequate connection pool size. Each `invoke` writes multiple rows per checkpoint, so a connection-starved pool will throttle the agent. Pool size = `(expected concurrent threads / workers per host) + 5` headroom.

> 💡 **Tip:** Always run `checkpointer.setup()` before the first `compile()` in a fresh database. The setup creates 3-4 tables and several indexes — without it, the first `invoke` will fail with `relation "checkpoints" does not exist`.

## 3. The `thread_id` Contract

Every persistent graph needs a `thread_id` in `config["configurable"]`. The framework uses it as the **conversation key**:

```python
config = {"configurable": {"thread_id": "user-42-session-1"}}

# Same thread_id — same conversation
app.invoke({"query": "Hello"}, config)
app.invoke({"query": "Who are you?"}, config)

# Different thread_id — independent conversation
config2 = {"configurable": {"thread_id": "user-42-session-2"}}
app.invoke({"query": "Hello, new session"}, config2)
```

Two `invoke()` calls with the same `thread_id` **resume** the prior state. Two calls with different `thread_id`s start fresh. There is no implicit cross-thread leakage.

### Naming Conventions

The `thread_id` is a free-form string. Common patterns:

| Pattern | Example | Use case |
|---------|---------|----------|
| `user-{id}-session-{n}` | `user-42-session-1` | Single-user apps |
| `{tenant_id}:{user_id}:{session_id}` | `acme:42:abc-def` | Multi-tenant SaaS |
| `{user_id}:{thread_topic}` | `42:trip-planning` | Persistent user conversations |
| UUIDv4 | `7f8a1c93-...` | Anonymous threads |

> 💡 **Tip:** Prefix `thread_id` with tenant ID for B2B SaaS. This lets you `DELETE FROM checkpoints WHERE thread_id LIKE 'acme:%'` for tenant offboarding — no graph code changes.

### What if I forget the thread_id?

```python
app.invoke({"query": "X"}, config={})  # No thread_id
# Raises ValueError: Checkpointer requires thread_id to be configured.
```

The checkpointer **requires** `thread_id`. This is by design — a checkpoint without a scope is a memory leak waiting to happen.

## 4. StateSnapshot and State History

Once a thread has checkpoints, you can inspect them:

```python
snapshot = app.get_state(config)
print(snapshot.values)      # Current state dict
print(snapshot.next)        # Tuple of next node names to run
print(snapshot.config)      # RunnableConfig that produced this snapshot
print(snapshot.metadata)    # source, step, writes
```

`get_state(config)` returns the **latest** snapshot. `get_state_history(config)` returns the **full history**:

```python
history = list(app.get_state_history(config))
for i, snap in enumerate(history):
    print(f"[{i}] step={snap.metadata.get('step')}, next={snap.next}, values_keys={list(snap.values.keys())}")
```

```python
# [
#   [0] step=0, next=('search',), values_keys=['query'],
#   [1] step=1, next=('synthesize',), values_keys=['query', 'findings'],
#   [2] step=2, next=(), values_keys=['query', 'findings', 'draft'],
# ]
```

**Caso real — Production debugging:** A Multi-Agent Research System user reports "the agent loops forever on Tavily errors." You pull their `thread_id` history, see the validator returning `is_valid=False` 12 times, identify the upstream bug (Tavily rate limit response was parsed as a hit), and fix the prompt. Without persistence this requires reproducing the user's exact input and praying.

## 5. Time Travel: Rewind and Replay

`update_state` lets you edit a prior snapshot and continue from there. Combined with `get_state_history`, this is a **time-travel debugger**:

```python
# 1. Get the bad snapshot
history = list(app.get_state_history(config))
bad_snap = history[5]  # pick the step where things went wrong

# 2. Branch from it with a correction
new_config = app.update_state(
    bad_snap.config,
    values={"is_valid": True, "findings": ["corrected"]},
)
# new_config is a RunnableConfig with a new checkpoint_id

# 3. Resume from the corrected state
result = app.invoke(None, new_config)
```

The graph continues from the corrected state, treating the changes as if a node had returned them.

> ⚠️ **Advertencia:** `update_state` rewinds one thread. It does NOT erase history — prior checkpoints are immutable. This is correct (audit trail) but confusing for first-time users who expect it to "undo". Use `get_state_history` to read, `update_state` to branch.

## 6. What Goes in State vs Configurable vs External

A frequent architectural question: where does X live?

| Data | Where | Why |
|------|-------|-----|
| User query | `state["query"]` | Workflow-changing |
| Intermediate results | `state["findings"]` | Workflow-changing |
| Final answer | `state["draft"]` | Workflow-changing |
| User ID | `config["configurable"]` | Request-scoped, not workflow-changing |
| Tenant ID | `config["configurable"]` | Request-scoped, multi-tenancy |
| Thread ID | `config["configurable"]` | Persistence scope |
| Model name | `config["configurable"]` | Per-request override |
| LLM API key | `os.environ` or secrets manager | Never in state — leaks to checkpoint |
| Prometheus span ID | OTel context, not state | Cross-cutting; survives via trace context |
| Audit log | External logging, not state | High-write; don't pollute checkpoints |

**Rule of thumb:** if it changes the workflow outcome, it goes in state. If it scopes a request, it goes in `configurable`. If it's a cross-cutting concern (auth, secrets, traces), it stays out of both.

> ⚠️ **Advertencia:** **Never** put secrets (API keys, passwords, tokens) in state. The state is serialized to Postgres on every checkpoint — a leaked Postgres backup leaks all secrets. Use `os.environ` or a secrets manager (HashiCorp Vault, AWS Secrets Manager).

## 7. Production Configuration

### Postgres Connection Pool

```python
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True},
)

with PostgresSaver.from_conn_string(DB_URI, pool=pool) as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

The pool is shared across graph instances — important for multi-worker FastAPI deploys where each worker creates its own compiled graph but reuses one Postgres pool.

### Async Checkpointer

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
    await app.ainvoke({"query": "x"}, config)
```

Required for `await app.ainvoke()` — sync and async checkpointers are separate types.

### TTL and Cleanup

Checkpoints are immutable, so they accumulate. Strategies:

```sql
-- Daily cleanup of threads older than 90 days
DELETE FROM checkpoints
WHERE thread_id IN (
    SELECT thread_id FROM checkpoint_metadata
    WHERE created_at < NOW() - INTERVAL '90 days'
);
```

Or in Python, surface a `DELETE /threads/{thread_id}` endpoint that the user can hit to forget a conversation (a privacy right under GDPR).

## 8. Production Reality

**Caso real — StayBot's conversation resumption:** The CrewAI → LangGraph migration in [[../17 - Production Agent Frameworks/06 - CrewAI 1.0 - Production Multi-Agent Flows.md|CrewAI 1.0 note 06]] included the addition of `PostgresSaver`. The `thread_id` is `user_id + ":" + property_id` — the same user querying the same property gets the same `thread_id`, and the agent remembers the search criteria, the negotiable price, and the last-seen calendar slots. Without persistence, every CrewAI turn was a fresh start.

**Caso real — Multi-Agent Research System long-running:** A research task that takes 5-10 minutes (multiple Tavily calls, fact-audit, synthesis) was crashing the original implementation when the FastAPI worker timed out. Adding `PostgresSaver` and running the agent as a background task with a websocket for status updates moved the timeout off the critical path and the agent recovered from intermittent Postgres blips via re-invocation.

## 📦 Compression Code

```python
# 📦 Compression: persistence, thread_id, time travel in one file
# Requires: pip install langgraph-checkpoint-postgres psycopg[binary]
# For MemorySaver: pip install langgraph

from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # swap for PostgresSaver

class State(TypedDict):
    query: str
    log: Annotated[list[str], add]

def step_a(state: State) -> dict:
    return {"log": [f"A({state['query']})"]}

def step_b(state: State) -> dict:
    return {"log": ["B"]}

graph = StateGraph(State)
graph.add_node("a", step_a); graph.add_node("b", step_b)
graph.add_edge(START, "a"); graph.add_edge("a", "b"); graph.add_edge("b", END)

# MemorySaver for test, PostgresSaver for production
app = graph.compile(checkpointer=MemorySaver())

# Persistent thread
config = {"configurable": {"thread_id": "thread-1"}}
print(app.invoke({"query": "first"},  config)["log"])  # ['A(first)', 'B']
print(app.invoke({"query": "again"}, config)["log"])  # ['A(first)', 'B', 'A(again)', 'B']

# Inspect current state
snap = app.get_state(config)
print(snap.next, snap.values)  # () {'query': 'again', 'log': [...]}

# History (chronological from oldest)
for s in app.get_state_history(config):
    print(s.metadata.get("step"), s.values["log"])

# Branch / time travel
hist = list(app.get_state_history(config))
branch = app.update_state(hist[0].config, values={"query": "branched"})
print(app.invoke(None, branch)["log"])  # 'A(branched)', 'B'
```

## 🎯 Key Takeaways

1. **`MemorySaver` for testing, `PostgresSaver` for production.** Same API; the only difference is who owns the storage.
2. **`thread_id` is required** and scopes the conversation. Prefix with tenant ID for multi-tenant SaaS.
3. **`get_state` returns the latest snapshot; `get_state_history` returns the full trace.** Combined with `update_state`, they form a time-travel debugger.
4. **Secrets never go in state** — the checkpointer serializes to durable storage, which may be backed up or replicated.
5. **State vs `config["configurable"]`: workflow-changing vs request-scoped.** Mixing them is a common architectural mistake.
6. **Postgres pool sizing matters** at >100 concurrent threads. Use `psycopg_pool.ConnectionPool` and share across workers.
7. **Checkpoints are immutable and accumulate.** Plan a cleanup policy (TTL, user-driven `DELETE /threads`) for GDPR and storage cost.

## References

- [[01 - StateGraph Fundamentals - Nodes Edges State and Reducers|StateGraph Fundamentals]] — the underlying state primitive.
- [[02 - Conditional Routing and Dynamic Edges|Conditional Routing]] — `thread_id` flows through `config["configurable"]` into conditional edges.
- [[05 - Human-in-the-Loop with interrupt() and Command|Human-in-the-Loop]] — persistence is the prerequisite for `interrupt()`.
- [[08 - Production Deployment - Studio, CLI, FastAPI|Production Deployment]] — multi-worker FastAPI patterns.
- LangGraph Persistence: https://langchain-ai.github.io/langgraph/concepts/persistence/