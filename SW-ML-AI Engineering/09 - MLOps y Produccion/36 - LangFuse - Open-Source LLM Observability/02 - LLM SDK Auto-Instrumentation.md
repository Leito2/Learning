# 🎯 02 - LLM SDK Auto-Instrumentation with LangFuse

> **Wire LangFuse into every major LLM SDK with three lines. Native traces, no wrapper code.**

## 🎯 Learning Objectives
- Instrument OpenAI, Azure OpenAI, Anthropic, Cohere, Mistral, and Google Gemini in 30 seconds
- Capture LangChain, LlamaIndex, Instructor, and DSPy calls with one environment variable
- Use the `@observe()` decorator to enrich framework-agnostic code with custom spans
- Distinguish between SDK-level instrumentation (wrapper) and framework-level instrumentation (callback)
- Calibrate sampling rate for cost-effective production observability
- Avoid the trace duplication antipattern that doubles your observability bill

## Introduction

LangFuse distinguishes two integration layers: **SDK-level wrappers** that intercept specific provider calls (OpenAI, Anthropic, etc.) and **framework-level callbacks** that hook into LangChain/LlamaIndex's existing telemetry pipeline. SDK wrappers give you cleaner data (token counts, model parameters, cost) without any code change. Framework callbacks ride existing infrastructure — useful when you want framework-native observability alongside LangFuse's interface.

The two layers compose cleanly. A production codebase typically uses SDK wrappers for raw provider calls (e.g. `openai.chat.completions.create`) and framework callbacks for higher-level abstractions (e.g. `langchain.agents.AgentExecutor`). LangFuse's UI unifies both — every trace looks the same regardless of source.

![LangFuse SDK architecture](https://langfuse.com/_next/image?url=%2Fstatic%2Fimages%2Fdocs%2Fpython-sdk.png&w=1200&q=75)

The cleanest approach for most teams in 2026 is **SDK wrappers everywhere + the `@observe()` decorator for custom spans**. This pattern works whether you call OpenAI directly, route through LiteLLM, or use agent frameworks — every LLM call becomes a LangFuse Generation observation without explicit instrumentation.

---

## 1. SDK Wrappers — Three Lines Per Provider

### 1.1 OpenAI

```python
import openai
from langfuse.openai import openai as langfuse_openai  # wrapper

# Set env vars:
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com  (or self-hosted URL)

# Replace the import — that's it
client = langfuse_openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    # LangFuse captures: model, messages, response, usage, cost, latency
)
```

The `langfuse.openai.openai` module is a drop-in replacement for `openai`. Every method on the OpenAI client is wrapped: `chat.completions`, `completions`, `embeddings`, `images.generate`, `audio.speech`, `audio.transcriptions`, `responses` (OpenAI's new unified endpoint). Each call becomes a Generation observation with full metadata.

For async:

```python
from langfuse.openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.chat.completions.create(...)
```

**Cost calculation** is automatic for 100+ models in the LangFuse model registry. The library ships with a JSON file mapping model name → input/output $/1M tokens, updated weekly.

### 1.2 Azure OpenAI

```python
from langfuse.openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

response = client.chat.completions.create(
    model="gpt-4o",  # deployment name
    messages=[{"role": "user", "content": "Hello"}],
)
```

Same wrapper, same auto-instrumentation. Azure deployments are matched to the model registry via the `model` parameter.

### 1.3 Anthropic

```python
from langfuse.anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude"}],
)
```

Captures Anthropic-specific features: prompt caching, citations, extended thinking (Claude 3.7+).

### 1.4 Cohere

```python
from langfuse.cohere import cohere as langfuse_cohere

co = langfuse_cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
response = co.chat(
    model="command-r-plus",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 1.5 Google Gemini (via Vertex AI or AI Studio)

```python
# AI Studio
import google.generativeai as genai
from langfuse import Langfuse
from langfuse.decorators import langfuse_context

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

# Wrap with LangFuse manually
@observe(as_type="generation")
def generate(prompt: str) -> str:
    langfuse_context.update_current_observation(
        model="gemini-1.5-pro",
        input=[{"role": "user", "content": prompt}],
    )
    response = model.generate_content(prompt)
    langfuse_context.update_current_observation(
        output=response.text,
        usage={"input_tokens": response.usage_metadata.prompt_token_count,
               "output_tokens": response.usage_metadata.candidates_token_count},
    )
    return response.text
```

The Google Gemini SDK has no LangFuse wrapper yet — use the manual `@observe()` pattern. Token counts are tracked via `usage_metadata`.

💡 **Tip:** For Gemini and Mistral, the manual `@observe()` pattern is the canonical way. For OpenAI/Anthropic/Cohere, use the wrapper for auto-capture.

### 1.6 Mistral

```python
from langfuse.openai import OpenAI  # Mistral supports OpenAI-compatible API

client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)

response = client.chat.completions.create(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Mistral's OpenAI-compatible API works with the LangFuse OpenAI wrapper. Same auto-capture.

---

## 2. LiteLLM Integration

If you use LiteLLM as your multi-provider transport (per [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]]), enable the LangFuse callback:

```python
import litellm
from litellm import completion

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# Or set in env:
# LITELLM_SUCCESS_CALLBACK=langfuse
# LITELLM_FAILURE_CALLBACK=langfuse

response = completion(
    model="gpt-4o-mini",  # or "anthropic/claude-3-5-sonnet", "groq/llama-3.3-70b"
    messages=[{"role": "user", "content": "Hello"}],
    metadata={
        "trace_user_id": "user_123",
        "trace_session_id": "session_abc",
        "trace_tags": ["production", "rag"],
    },
)
```

LiteLLM's callback forwards every completion to LangFuse with the same data model as the SDK wrappers. This is the **canonical production pattern**: one transport, every provider traced.

For cost tracking per team, add `metadata={"trace_user_id": team_id}` — the trace is attributed to that team for cost dashboards.

---

## 3. Framework Integrations

### 3.1 LangChain

```python
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

# Initialize LangChain handler
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

llm = ChatOpenAI(model="gpt-4o-mini")
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

chain = prompt | llm

# Pass handler via config
response = chain.invoke(
    {"question": "What is the capital of France?"},
    config={"callbacks": [langfuse_handler]},
)
```

Every LangChain component (LLM, retriever, agent) becomes a LangFuse observation. For chains with multiple steps, each step is a Span; each LLM call is a Generation. The handler captures tokens, model, latency, and tool use automatically.

For more details on LangGraph patterns, see [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]] and [[09 - MLOps y Produccion/35 - LangSmith Deep Dive/03 - Datasets and Evaluations|Datasets and Evaluations in LangSmith]] — the same callback pattern works for both.

### 3.2 LlamaIndex

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LangfuseCallbackHandler

langfuse_handler = LangfuseCallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

Settings.callback_manager = CallbackManager([langfuse_handler])

# All LlamaIndex operations are now traced
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")
```

LlamaIndex operations (loading, splitting, embedding, retrieval, synthesis) all become LangFuse observations. Useful for debugging RAG pipelines — see the LlamaIndex pattern from [[06 - Large Language Models/12 - Production RAG]].

### 3.3 Instructor (the structured-output library)

The capstone of [[06 - Large Language Models/22 - Instructor and Structured Generation]] uses Instructor with Pydantic. The LangFuse trace captures the LLM call (under the hood) plus the validation retries:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from langfuse.openai import openai as langfuse_openai

class Person(BaseModel):
    name: str
    age: int

client = instructor.from_openai(langfuse_openai.OpenAI())  # wrapped client → LangFuse traces

person = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Maria is 28 and works on LLMs."}],
    response_model=Person,
    max_retries=4,  # each retry creates a separate LangFuse span
)

# LangFuse captures: parent Generation + 4 child Generations (retries) + final Pydantic-validated output
```

For Instructor on a custom @observe wrapper:

```python
from langfuse import observe

@observe(as_type="span")
def extract_person(text: str) -> Person:
    return client.chat.completions.create(  # traced as Generation via wrapper
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        response_model=Person,
    )
```

### 3.4 DSPy and LangGraph

For DSPy (covered in [[06 - Large Language Models/21 - DSPy and Prompt Compilation]]), wrap the DSPy module call in `@observe()`:

```python
import dspy
from langfuse import observe

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

@observe()
def run_dspy_module(question: str) -> str:
    predictor = dspy.ChainOfThought("question -> answer")
    result = predictor(question=question)
    return result.answer
```

For LangGraph agents (covered in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]]), use the LangChain CallbackHandler from section 3.1 — LangGraph is built on LangChain.

---

## 4. The `@observe()` Decorator for Custom Spans

For non-LLM operations (retrieval, reranking, vector search, business logic), use the decorator to capture custom spans:

```python
from langfuse import observe, langfuse_context

@observe(name="retrieve_documents", as_type="span")
def retrieve_documents(query: str, k: int = 5) -> list[str]:
    results = vector_store.search(query, k=k)
    
    # Enrich with metadata
    langfuse_context.update_current_observation(
        metadata={"k": k, "result_count": len(results)},
    )
    
    return [r.text for r in results]

@observe(name="rerank", as_type="span")
def rerank(query: str, documents: list[str]) -> list[str]:
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    scores = cross_encoder.predict([[query, doc] for doc in documents])
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    langfuse_context.update_current_observation(
        metadata={"input_count": len(documents), "top_score": ranked[0][1]},
    )
    return [doc for doc, _ in ranked[:5]]
```

The decorator handles:
- Span start (entry into function)
- Span end with timing (exit)
- Input capture (function args)
- Output capture (return value)
- Exception capture (raises → trace flagged)
- Metadata updates via `langfuse_context`

This is the **bridge** between framework instrumentation and custom code. A RAG trace might have:
- 1 Generation (LLM call, auto-captured by wrapper)
- 2 Spans (retrieve_documents, rerank, manually captured by decorator)
- 1 Score (user feedback, manually attached)
- All unified into one Trace

---

## 5. Sampling — Cost Control for High RPS

For high-throughput pipelines (1K+ requests/sec), tracing every request becomes expensive ($0.001 per trace × 1M traces/day = $1000/day just for tracing). LangFuse supports sampling:

```python
import os
os.environ["LANGFUSE_SAMPLE_RATE"] = "0.1"  # trace 10% of requests
```

The sample rate is **deterministic by session_id** — if a session is sampled, every call in that session is traced. This gives 100% coverage of important user sessions while cutting cost.

For random sampling:

```python
from langfuse import Langfuse
langfuse = Langfuse(sample_rate=0.05)  # 5% of traces
```

💡 **Tip:** Use **session-based sampling for user-driven apps** (every user interaction is fully traced; only some users are sampled) and **rate-based sampling for batch jobs** (every Nth batch is traced). Avoid random sampling — it produces incoherent traces.

---

## 6. The @observe Pattern: Synchronous, Async, and Streaming

### 6.1 Synchronous and async

```python
@observe()
def sync_fn(): ...

@observe()
async def async_fn(): ...
```

Both work out of the box.

### 6.2 Streaming

For streaming LLM responses, LangFuse accumulates tokens and reports the full Generation:

```python
@observe(as_type="generation")
def stream_openai(prompt: str):
    langfuse_context.update_current_observation(model="gpt-4o-mini")
    
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    
    full_response = "".join(chunks)
    langfuse_context.update_current_observation(
        output=full_response,
        usage={"input_tokens": 100, "output_tokens": len(full_response.split()) // 2},
    )
    return full_response
```

The trace shows the complete generation latency (including stream completion) and the full final output. For time-to-first-token (TTFT), use `langfuse_context.update_current_span(metadata={"ttft_ms": ...})` from the first chunk callback.

---

## 7. Antipatterns

### 7.1 Antipattern 1: Double-tracing with both wrapper AND callback

```python
# ❌ Every call creates TWO traces — bill doubles
from langfuse.openai import openai as langfuse_openai
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

client = langfuse_openai.OpenAI()  # traces via wrapper
chat = ChatOpenAI(openai_client=client)
chat.invoke("...", config={"callbacks": [CallbackHandler()]})  # ALSO traces via callback
```

The LangChain callback captures the LLM call; the wrapped OpenAI client captures it again. Each call becomes a duplicate trace. Use one OR the other per call:

```python
# ✅ Option A: just wrapper (skip LangChain callback)
client = langfuse_openai.OpenAI()
chat = ChatOpenAI(openai_client=client)  # wrapper traces, no callback needed

# ✅ Option B: just callback (skip wrapper)
client = openai.OpenAI()  # NOT langfuse_openai
chat = ChatOpenAI(openai_client=client)
chat.invoke("...", config={"callbacks": [CallbackHandler()]})  # callback traces

# ❌ NOT both
```

### 7.2 Antipattern 2: Tracing at 100% in high-RPS production

```python
# ❌ 1M traces/day = $300+/day
# ✅ Sample at 5-10% in production; 100% in dev/staging
os.environ["LANGFUSE_SAMPLE_RATE"] = "0.1"  # production
os.environ["LANGFUSE_SAMPLE_RATE"] = "1.0"   # dev/staging
```

### 7.3 Antipattern 3: Forgetting to attach the handler to async chains

```python
# ❌ Bug: async chain without handler attachment
import asyncio
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({"a": step_a, "b": step_b})
result = await chain.ainvoke(input)  # LangChain subprocesses don't inherit handler automatically

# ✅ Correct: pass handler via config or use the langchain integration explicitly
result = await chain.ainvoke(input, config={"callbacks": [handler]})
```

### 7.4 Antipattern 4: Capturing raw embeddings as trace inputs

```python
# ❌ Huge blobs: 1536-dim embeddings per request as trace input
trace = langfuse.trace(input={"query": "What is ML?", "embedding": [0.0123, ...]})  # 6KB per trace

# ✅ Hash or omit the embedding
import hashlib
trace = langfuse.trace(input={
    "query": "What is ML?",
    "embedding_hash": hashlib.sha256(str(embedding).encode()).hexdigest()[:12],
})
```

Embeddings are not useful in trace UI; only their hash identifies a cache hit.

### 7.5 Antipattern 5: Configuring wrapper at runtime instead of import time

```python
# ❌ Race condition: wrapper initialized before env vars are set
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "..."
from langfuse.openai import openai as langfuse_openai  # wrapper may have already loaded config

# ✅ Set env vars BEFORE importing the wrapper
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
from langfuse.openai import openai as langfuse_openai  # NOW the wrapper reads the config
```

---

## 🎯 Key Takeaways

- `langfuse.openai.openai` is a drop-in replacement for the OpenAI client; same pattern for Anthropic and Cohere.
- LiteLLM callbacks (`litellm.success_callback = ["langfuse"]`) auto-trace every provider via one transport.
- LangChain, LlamaIndex, DSPy integrations use the framework callback pattern.
- `@observe(as_type="span")` adds custom spans for retrieval, reranking, business logic.
- Session-based sampling (`sample_rate=0.1`) cuts observability cost without losing user-coherence.
- Streaming Generations accumulate tokens and report full output on completion.
- Avoid double-tracing with wrapper + callback, 100% sampling in production, missing async handler attachment, raw embedding capture, and post-import env var setting.

## References

- LangFuse SDK docs — [langfuse.com/docs/sdk/python](https://langfuse.com/docs/sdk/python)
- LangFuse integrations — [langfuse.com/docs/integrations](https://langfuse.com/docs/integrations)
- LangChain integration — [langfuse.com/docs/integrations/langchain](https://langfuse.com/docs/integrations/langchain)
- LlamaIndex integration — [langfuse.com/docs/integrations/llamaindex](https://langfuse.com/docs/integrations/llamaindex)
- LiteLLM integration — [langfuse.com/docs/integrations/litellm](https://langfuse.com/docs/integrations/litellm)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider transport
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured output library
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — agent observability
- [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers|OpenTelemetry for AI Engineers]] — protocol-level observability
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/01 - LangFuse Fundamentals - Architecture and Core Primitives|Note 01 — Fundamentals]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/05 - Capstone - Self-Hosted LangFuse for Multi-Provider RAG|Note 05 — Capstone]]
