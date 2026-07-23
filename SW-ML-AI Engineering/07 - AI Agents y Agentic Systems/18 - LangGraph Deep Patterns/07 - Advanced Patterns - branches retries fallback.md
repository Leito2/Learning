# 🛠️ Advanced Patterns — Branches, Retries, Fallback and Recursion Limits

The framework gets you 80% of the way. Production gets you the last 20%: "what happens when the LLM returns malformed JSON", "what happens when the workflow would loop forever", "what happens when a node takes 30 seconds and the user is impatient". LangGraph exposes these as **declarative configuration**: `RetryPolicy`, `CachePolicy`, `recursion_limit`, per-node `timeout`, and `defer` for delayed side effects. Most courses skip this material; production engineers need it on day one.

This note covers the four production hardening primitives — retry policies, fallback nodes, recursion limits, and node-level timeouts — plus the map-reduce and conditional-merge patterns that turn a flat graph into a robust one. By the end you will be able to read any production LangGraph deployment and predict its behavior on bad input, slow LLMs, infinite loops, and transient outages.

## 🎯 Learning Objectives

- Configure `RetryPolicy` for automatic node retry on transient failures.
- Implement fallback nodes that catch uncaught exceptions from upstream.
- Set `recursion_limit` to bound cycles and prevent infinite loops.
- Add per-node `timeout` to surface hanging LLM calls.
- Use `CachePolicy` for deterministic nodes (skips redundant work).
- Distinguish `_write_channel` and `_return` patterns for advanced state mutation.
- Catch errors in tool-execution nodes without crashing the parent graph.
- Map-reduce with `Send` and state reducers for large-list aggregation.

## 1. The Problem: Real-World Failure Modes

A research agent that demos correctly will, in production, encounter:

| Failure | Symptom | Default behavior |
|---------|---------|-------------------|
| Tavily 5xx | `httpx.HTTPStatusError` | Crashes node, graph halts |
| LLM malformed JSON | `ValidationError` on parse | Crashes node |
| Slow LLM (15s instead of 4s) | No timeout | Worker held |
| Cyclic validation looping | Validator rejects always | Infinite loop, worker holds |
| Deterministic classifier | Same input every time | Re-runs every call |
| Multi-step map (100 queries) | Naive parallelism | Overload downstream API |

None of these are addressed by the basic graph construction. All require the patterns in this note.

## 2. `RetryPolicy` — Automatic Node Retry

```python
from langgraph.types import RetryPolicy

graph.add_node(
    "llm_call",
    llm_call_fn,
    retry_policy=RetryPolicy(
        max_attempts=5,            # total attempts including initial
        initial_interval=0.5,      # seconds before first retry
        backoff_factor=2.0,        # exponential backoff multiplier
        max_interval=10.0,         # cap for backoff interval
        jitter=True,               # add random jitter to avoid thundering herd
        retry_on=lambda e: isinstance(e, (ConnectionError, TimeoutError)),
    ),
)
```

`RetryPolicy` is per-node. The framework retries the node automatically; if all attempts fail, the graph halts with the last exception.

### Selector: Retry on Specific Exceptions

```python
def only_retry_transient(exception: BaseException) -> bool:
    """Don't retry ValidationError (LLM bug) — retry Timeout (LLM slow)."""
    return isinstance(exception, (ConnectionError, TimeoutError, httpx.HTTPStatusError))

graph.add_node(
    "tavily_search",
    tavily_fn,
    retry_policy=RetryPolicy(max_attempts=4, retry_on=only_retry_transient),
)
```

> ⚠️ **Advertencia:** Don't retry on `ValueError` or schema-validation errors — those are code bugs, not transient failures. Retrying just delays the failure.

### Default Retry Policy

If `retry_policy` is omitted, nodes have **no retry**. The graph stops at the first error. For LLM-facing nodes, always set a `RetryPolicy` to handle transient provider outages.

## 3. Fallback Nodes — Catch Uncaught Exceptions

```python
def risky_llm_call(state: State) -> dict:
    response = llm.invoke(state["query"])  # may raise on provider outage
    return {"answer": parse(response)}

def safe_fallback(state: State) -> dict:
    """Runs only if risky_llm_call raises after retries."""
    return {
        "answer": "We're having trouble reaching the LLM. Please try again.",
        "fallback_used": True,
    }

graph.add_node("llm", risky_llm_call)
graph.add_node("fallback", safe_fallback)
graph.add_edge("llm", "fallback")  # always runs after llm
```

But the edge above runs after `llm` regardless of success. To run **only on failure**, wrap with a path function:

```python
def should_fallback(state: State) -> str:
    if state.get("error"):
        return "fallback"
    return "END"

graph.add_conditional_edges("llm", should_fallback, path_map={"fallback": "fallback", "END": END})
```

Better: combine retry + fallback. The retry handles transient failures; the fallback handles permanent ones.

### Try-Except Inside Nodes

```python
def llm_call_with_inner_fallback(state: State) -> dict:
    try:
        response = llm.invoke(state["query"])
        return {"answer": parse(response)}
    except ValidationError:
        # LLM returned malformed output — fall back to a deterministic template
        return {"answer": template_response(state["query"])}
    except (TimeoutError, ConnectionError):
        # Transient — let retry handle
        raise
```

Use **inner try/except** for recoverable errors (LLM JSON parse → template). Use **outer retry policy** for transient errors (provider 5xx). Use **outer fallback node** for permanent errors (provider outage, config missing).

> 💡 **Tip:** A common 3-tier pattern: (1) inner try/except for malformed output, (2) `RetryPolicy` for transient errors, (3) `fallback` node for permanent failures. This catches 95% of real-world LLM agent failures.

## 4. Configurable Recursion Limits

```python
app = graph.compile(
    recursion_limit=25,  # default 25; max 1000
)
```

`recursion_limit` bounds the **number of super-steps** the graph can execute. When exceeded, the graph raises `GraphRecursionError`. This prevents infinite loops in cyclic validation, agent self-correction, and self-play scenarios.

### Adjust Per Call

```python
# Override the compiled limit for one invoke
app.invoke(input, config={"recursion_limit": 100})
```

### Pattern: Cyclic Validator with Bound

```python
def should_continue(state: ValidatorState) -> str:
    if state["is_valid"] or state["attempts"] >= 5:
        return "synthesize"
    return "validate"  # self-loop

graph.add_conditional_edges("validate", should_continue)
```

Even with `recursion_limit=25`, this cycle exits after 5 iterations. The path function's bound check is the **logic**; the recursion limit is the **safety net**.

> ⚠️ **Advertencia:** A node returning a state identical to its input causes an infinite loop. Always include a counter or terminator condition in cyclic paths.

## 5. Per-Node Timeouts

```python
import asyncio

async def slow_llm_call(state: State) -> dict:
    try:
        result = await asyncio.wait_for(llm.ainvoke(state["query"]), timeout=10.0)
        return {"answer": result.content}
    except asyncio.TimeoutError:
        return {"answer": "LLM timeout — using cached response.", "cache_hit": True}

graph.add_node("llm", slow_llm_call)
```

For most LLM SDKs, `asyncio.wait_for` wraps the call cleanly. For sync nodes, use `concurrent.futures` with timeout.

> 💡 **Tip:** Set per-node timeouts to **2× the median latency**. If the LLM has p50 = 3s, set timeout = 6s. Setting it at p99 = 12s catches hung workers without aborting slow-but-valid calls.

## 6. `CachePolicy` — Skip Redundant Work

```python
from langgraph.types import CachePolicy

def deterministic_classifier(state: State) -> dict:
    """Input: query string. Output: intent label. Deterministic."""
    intent = classify(state["query"])  # pure function
    return {"intent": intent}

graph.add_node(
    "classify",
    deterministic_classifier,
    cache_policy=CachePolicy(ttl=300),  # cache result for 5 minutes
)
```

The framework caches the node's output keyed by the input state fields. Subsequent calls with the same input skip the node entirely.

### When to Cache

| Pattern | Cacheable? | TTL |
|---------|------------|-----|
| LLM classifier on user query | ✅ (high hit rate in production) | 5-30 min |
| Tavily search results | ✅ (TTL = freshness window) | 1 hour |
| Embedding generation | ✅ (deterministic for same text) | 1 hour+ |
| LLM synthesis on long context | ❌ (chat history varies) | 0 |
| Tool calls with side effects | ❌ (must execute) | 0 |
| Subgraph with persistence | ❌ (state-coupled) | 0 |

### ❌ Don't Cache Side Effects

```python
# ❌ Caching a write to Postgres — disaster
def db_write(state: State) -> dict:
    db.execute("INSERT INTO ...", state["data"])
    return {"status": "written"}

graph.add_node("write", db_write, cache_policy=CachePolicy(ttl=60))
# Second call with same state skips the write — silent data loss!
```

## 7. Map-Reduce Patterns

Large-list aggregation is the bread-and-butter use case for `Send` ([[04 - Subgraphs and Send API|note 04]]) + reducers.

### Single-Round Map-Reduce

```python
class AggregateState(TypedDict):
    queries: list[str]
    results: Annotated[list[dict], add]  # append reducer

def dispatch(state: AggregateState) -> list[Send]:
    return [Send("process_one", {"q": q}) for q in state["queries"]]

graph.add_node("process_one", process_fn)
graph.add_conditional_edges(START, dispatch, ["process_one"])
# merge node optional — reducer already aggregates
```

### Two-Round Map-Reduce (parse then process)

```python
class ParseState(TypedDict):
    raw: list[dict]
    parsed: Annotated[list[dict], add]
    enriched: dict

def parse_dispatch(state):
    return [Send("parse_one", {"item": item}) for item in state["raw"]]

def enrich_dispatch(state):
    return [Send("enrich_one", {"item": item}) for item in state["parsed"]]

graph.add_node("parse_one", parse_fn)
graph.add_node("enrich_one", enrich_fn)
graph.add_node("merge", lambda s: {"enriched": {"count": len(s["parsed"])}})

graph.add_conditional_edges(START, parse_dispatch, ["parse_one"])
graph.add_edge("parse_one", "enrich_one")  # all parse finishes, then dispatch enrich
graph.add_conditional_edges("enrich_one", enrich_dispatch, ["enrich_one"])  # buggy — see below
```

> ⚠️ **Advertencia:** `Send` cannot be chained in one node — `parse_one → enrich_one` is one super-step; all parse must finish before enrich dispatches. Combine with an explicit "barrier" node if you need two-stage parallelism.

```python
graph.add_conditional_edges("parse_one", lambda _: "barrier")  # single path
graph.add_node("barrier", lambda s: {})  # does nothing
graph.add_conditional_edges("barrier", enrich_dispatch, ["enrich_one"])
```

## 8. `defer` — Side Effects After Main Path

```python
def write_to_db(state: State) -> dict:
    """Side effect — runs only if the main path succeeds."""
    db.execute("INSERT ...", state["final_answer"])
    return {}  # no state change

graph.add_node("write_db", write_db_fn)
graph.add_edge("main_path", "write_db")
```

The `defer` style: nodes that emit side effects should run **after** the main path reaches `END` or a terminal state, with the framework providing `defer=True` to ensure they only run on success.

## 9. ❌/✅ Antipatterns

### ❌ Catching and ignoring all errors

```python
# ❌ Silent failures
def risky_node(state):
    try:
        return {"data": llm.invoke(state["query"])}
    except Exception:
        return {}  # ❌ swallowed exception, no telemetry
```

### ✅ Surface failures with structured error state

```python
def risky_node(state):
    try:
        return {"data": llm.invoke(state["query"])}
    except ValidationError as e:
        return {"error": str(e), "error_node": "risky_node"}
    except (TimeoutError, ConnectionError) as e:
        return {"error": "transient", "retry": True}
```

### ❌ Infinite recursion with no terminator

```python
def loop_node(state):
    return {"counter": state["counter"] + 1}  # never breaks

graph.add_conditional_edges("loop", lambda s: "loop", path_map={"loop": "loop"})
# GraphRecursionError after recursion_limit super-steps
```

### ✅ Always include a counter/terminator

```python
def loop_node(state):
    return {"counter": state["counter"] + 1}

def should_loop(state):
    return "loop" if state["counter"] < 5 else "synthesize"

graph.add_conditional_edges("loop", should_loop, path_map={"loop": "loop", "synthesize": "synthesize"})
```

### ❌ Cache on side-effecting nodes

```python
graph.add_node("send_email", send_email_fn, cache_policy=CachePolicy(ttl=60))
# Two calls with same input — second one silently skipped. Email lost!
```

### ✅ No cache on side-effecting nodes

```python
graph.add_node("send_email", send_email_fn)  # no cache policy
```

## 10. Production Reality

**Caso real — Multi-Agent Research System retry:** The Tavily search node has a `RetryPolicy(max_attempts=4, retry_on=lambda e: isinstance(e, (httpx.HTTPStatusError, ConnectionError)))` with `backoff_factor=2` and `jitter=True`. On the 5xx storms that hit during peak traffic, the agent retries 3× with backoff (0.5s, 1s, 2s, 4s) and recovers transparently. Without retry, every Tavily blip crashed the workflow.

**Caso real — StayBot timeout:** The Airbnb availability node uses `asyncio.wait_for(..., timeout=5.0)` and a fallback that returns "We couldn't reach Airbnb — please try again." Per-node timeout at 2× median caught three worker hangs in production where the Airbnb API returned a 200 with an empty body and the LLM hung parsing it.

## 📦 Compression Code

```python
# 📦 Compression: retry, fallback, recursion limit, cache, map-reduce in 90 lines
# Covers: per-node retry policy, fallback path, cache policy, Send map-reduce, error surface

import asyncio
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy, interrupt, Command
from langgraph.constants import Send

class State(TypedDict):
    queries: list[str]
    results: Annotated[list[str], add]
    fallback_used: bool

# --- Map step: parallel query processing with cache ---
def process_one(query: str) -> dict:
    """Deterministic per-query processing — cacheable."""
    return {"results": [f"result-{query}"]}

process_subgraph = (
    StateGraph({"q": str})
    .add_node("p", lambda s: process_one(s["q"]))
    .add_edge(START, "p").add_edge("p", END)
    .compile()
)

# --- Reduce step + fallback ---
def reduce(state: State) -> dict:
    if not state["results"]:
        return {"fallback_used": True, "results": ["default result"]}
    return {}

def dispatch(state: State):
    return [Send("process_subgraph", {"q": q}) for q in state["queries"]]

# --- Top-level graph with retry + cache ---
graph = StateGraph(State)
graph.add_node("process_subgraph", process_subgraph,
               cache_policy=CachePolicy(ttl=120))  # cache for 2 minutes
graph.add_node("reduce", reduce,
               retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.2))
graph.add_conditional_edges(START, dispatch, ["process_subgraph"])
graph.add_edge("process_subgraph", "reduce")
graph.add_edge("reduce", END)

app = graph.compile(recursion_limit=20)

print(app.invoke({"queries": ["a", "b", "c"], "results": [], "fallback_used": False}))
# {'queries': ['a','b','c'], 'results': ['result-a','result-b','result-c'], 'fallback_used': False}
```

## 🎯 Key Takeaways

1. **`RetryPolicy` handles transient failures** (provider 5xx, timeouts). Use `retry_on` to scope which exceptions to retry.
2. **Three-tier error handling:** inner try/except for recoverable errors, `RetryPolicy` for transient, fallback nodes for permanent.
3. **`recursion_limit` is the safety net** for cyclic loops. The path function's counter is the logic; the limit is the guard.
4. **Per-node timeouts at `2× median latency`** catch hung workers without aborting slow-but-valid LLM calls.
5. **Cache only deterministic nodes** (classifiers, embeddings, fresh searches). Never cache side-effecting nodes.
6. **`Send` enables map-reduce**, but barrier nodes are needed for multi-stage parallel pipelines.
7. **Surface errors as state**, not silent no-ops. `state["error"]` + `state["error_node"]` is more debuggable than `return {}`.

## References

- [[01 - StateGraph Fundamentals - Nodes Edges State and Reducers|StateGraph Fundamentals]] — reducers enable map-reduce aggregation.
- [[04 - Subgraphs and Send API|Subgraphs]] — `Send` is the parallel-dispatch primitive.
- [[05 - Human-in-the-Loop with interrupt() and Command|Human-in-the-Loop]] — combine retry with interrupt for human escalation on persistent failure.
- [[08 - Production Deployment - Studio, CLI, FastAPI|Production Deployment]] — observability for retry/fallback events.
- LangGraph Retry/Cache: https://langchain-ai.github.io/langgraph/concepts/functional_api/