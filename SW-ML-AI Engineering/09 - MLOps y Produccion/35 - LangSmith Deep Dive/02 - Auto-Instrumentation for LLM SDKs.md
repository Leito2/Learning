# 🔌 Auto-Instrumentation for LLM SDKs

LangSmith's killer feature: **every LangChain, LangGraph, OpenAI, Anthropic, and Cohere call is auto-traced** with one environment variable. No SDK changes, no decorator work, no manual span creation. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` and **the rest is automatic** — every LLM call, every chain run, every tool call lands in LangSmith with inputs, outputs, latency, and token counts.

This note covers the auto-instrumentation patterns for the major LLM SDKs: LangChain (native), LangGraph (native), OpenAI (one-line patch), Anthropic (one-line patch), Cohere, and custom Python code (`@traceable`). By the end you can instrument any LLM application in under 5 minutes.

## 🎯 Learning Objectives

- Enable LangSmith auto-instrumentation with environment variables.
- Instrument **LangChain** components (native).
- Instrument **LangGraph** state machines (native).
- Patch **OpenAI, Anthropic, Cohere** SDKs (one line each).
- Wrap custom Python functions with `@traceable`.
- Pass **metadata and tags** through the run tree.
- Avoid the four most common auto-instrumentation pitfalls.

## 1. The One-Line Setup

```bash
# .env or os.environ
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-app
LANGSMITH_ENDPOINT=https://api.smith.langchain.com  # default
```

That's it. Any LLM call from a supported SDK now lands in LangSmith.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is LangSmith?")
# → Run "ChatOpenAI" created in LangSmith with input, output, latency, tokens
```

## 2. LangChain — Native Integration

LangChain is the framework LangSmith was built for. **Every component is auto-traced:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# All of these are auto-traced
prompt = ChatPromptTemplate.from_template("What is {topic}?")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# The chain runs as a single trace with sub-runs for each step
chain = prompt | llm | parser
result = chain.invoke({"topic": "LangSmith"})
```

The trace tree shows:

```
chain (latency=1.2s)
├── ChatPromptTemplate (latency=2ms)
├── ChatOpenAI (latency=1.1s, tokens=120)
└── StrOutputParser (latency=5ms)
```

## 3. LangGraph — Native Integration

LangGraph's `StateGraph` is auto-traced with **each node as a run**:

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    findings: list[str]
    answer: str

def research(state: State) -> dict:
    return {"findings": tavily_search(state["query"])}

def synthesize(state: State) -> dict:
    return {"answer": llm.invoke(state["query"])}

graph = StateGraph(State)
graph.add_node("research", research)
graph.add_node("synthesize", synthesize)
graph.add_edge(START, "research")
graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()

# Auto-traced: each node is a run
result = app.invoke({"query": "What is LangGraph?"})
```

Trace tree:

```
LangGraph.invoke
├── research_node (latency=2.4s)
│   └── openai_chat (latency=2.3s, tokens=450)
└── synthesize_node (latency=1.8s)
    └── openai_chat (latency=1.7s, tokens=300)
```

## 4. OpenAI — Patch with One Line

For non-LangChain code that uses OpenAI directly:

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"

# Patch the OpenAI SDK
from langsmith.wrappers import wrap_openai
from openai import OpenAI

client = wrap_openai(OpenAI())  # wrapped client

# All calls now traced
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is X?"}],
)
```

Or globally:

```python
from langsmith import patch

patch()  # patches all LLM SDKs globally

from openai import OpenAI
client = OpenAI()  # auto-traced
```

## 5. Anthropic — Same Pattern

```python
from langsmith.wrappers import wrap_anthropic
from anthropic import Anthropic

client = wrap_anthropic(Anthropic())

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 6. Cohere — Same Pattern

```python
from langsmith.wrappers import wrap_cohere
import cohere

client = wrap_cohere(cohere.Client())
response = client.chat(message="Hello", model="command-r-plus")
```

## 7. Custom Python Code — `@traceable`

For business logic that doesn't call an LLM directly:

```python
from langsmith import traceable

@traceable
def preprocess_query(query: str) -> str:
    return query.strip().lower()

@traceable(name="custom_search")
def search(query: str) -> list[str]:
    results = vector_db.search(query, top_k=10)
    return [r.content for r in results]

@traceable
def main_pipeline(query: str) -> str:
    preprocessed = preprocess_query(query)
    results = search(preprocessed)
    return llm.invoke(f"Context: {results}\n\nQuestion: {query}")
```

## 8. Passing Metadata and Tags

```python
@traceable(
    metadata={"user_id": "u-42", "tenant_id": "acme", "version": "1.2.3"},
    tags=["rag", "gpt-4o-mini", "user-tier:premium"],
    name="rag_query",  # custom name in UI
)
def rag_query(query: str) -> str:
    return llm.invoke(query)
```

**Metadata is structured (filterable). Tags are free-form labels.** Both are queryable in the LangSmith UI and via the SDK.

## 9. Tracing Async Code

```python
import asyncio
from langsmith import traceable

@traceable
async def async_pipeline(query: str) -> str:
    response = await async_llm.ainvoke(query)
    return response
```

LangSmith handles async correctly — the run spans the entire await.

## 10. The Trace Hierarchy Pattern

```python
@traceable(name="user_request")
def handle_request(user_query: str, user_id: str):
    """Outer: business-logic boundary."""
    # Set metadata on the current run
    from langsmith import get_current_run_tree
    run = get_current_run_tree()
    run.add_metadata({"user_id": user_id, "query_length": len(user_query)})

    # Nested calls are sub-runs
    cleaned = clean_query(user_query)
    results = vector_search(cleaned)
    answer = generate_answer(cleaned, results)
    return answer
```

## 11. ❌/✅ Antipatterns

### ❌ Instrumenting after SDK import

```python
# ⚠️ Patches don't apply to already-imported modules
from openai import OpenAI  # imported
patch()  # too late
client = OpenAI()
```

### ✅ Instrument BEFORE imports

```python
# ✅ Patches applied first
from langsmith import patch
patch()  # patches openai, anthropic, etc.

from openai import OpenAI
client = OpenAI()  # now instrumented
```

### ❌ Trace every internal function

```python
# ⚠️ Trace explosion — too many runs
@traceable
def step1(x): return x + 1
@traceable
def step2(x): return x * 2
@traceable
def step3(x): return x ** 2
```

### ✅ Trace business-logic boundaries

```python
# ✅ One trace per meaningful operation
@traceable(name="compute_features")
def pipeline(x):
    return step3(step2(step1(x)))
```

### ❌ No metadata, no tags

```python
# ⚠️ Can't filter or analyze
@traceable
def my_step(query):
    return process(query)
```

### ✅ Metadata + tags

```python
@traceable(
    metadata={"user_id": user_id, "model": "gpt-4o-mini"},
    tags=["rag", "user-tier:premium"],
)
def my_step(query):
    return process(query)
```

### ❌ Capturing secrets in inputs

```python
# ⚠️ API key persisted to LangSmith
@traceable
def call_api(api_key, query):
    return requests.post(..., headers={"Authorization": api_key})
```

### ✅ Strip secrets before tracing

```python
@traceable
def call_api(api_key, query):
    # api_key passed as arg but not in inputs
    return requests.post(..., headers={"Authorization": api_key})
# LangSmith only sees `query` in inputs; api_key is filtered
```

## 12. Production Reality

**Caso real — Production RAG Project:** LangSmith is auto-instrumented via environment variables in the FastAPI service. Every LLM call, every LangGraph node, every tool invocation lands in LangSmith with token counts and latency. **No SDK changes were needed** — just the 3 env vars.

**Caso real — Multi-Agent Research System:** LangGraph's auto-tracing produces a 30+ run trace per user request. The support team uses LangSmith to debug customer issues — finding the slow or failing node in minutes.

## 📦 Compression Code

```python
# 📦 Compression: Auto-instrumentation in 20 lines

import os
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-app"

# Patch ALL SDKs globally
from langsmith import patch
patch()  # one line — OpenAI, Anthropic, Cohere, etc.

# Now use any SDK
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini")

# Custom functions
from langsmith import traceable

@traceable(metadata={"user_id": "u-42"})
def my_step(query):
    return llm.invoke(query)
```

## 🎯 Key Takeaways

1. **Three env vars** enable auto-instrumentation: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`.
2. **LangChain / LangGraph** are natively traced — no extra work.
3. **`wrap_openai`, `wrap_anthropic`, `wrap_cohere`** for non-LangChain SDKs.
4. **`patch()` for global patching** — covers everything in one call.
5. **`@traceable`** for custom Python functions.
6. **Metadata + tags** at every trace for filterability.
7. **Instrument BEFORE imports** — patches don't apply to already-imported modules.

## References

- [[00 - Welcome to LangSmith|Welcome]] — course map.
- [[01 - LangSmith Core|Core primitives]] — traces, runs, projects.
- [[03 - Datasets and Evaluations|Datasets]] — versioned test sets.
- [[../../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/00 - Welcome to LangGraph Deep Patterns.md|LangGraph]] — native integration.
- LangSmith auto-instrumentation: https://docs.smith.langchain.com/observability/how_to_guides/annotate_code