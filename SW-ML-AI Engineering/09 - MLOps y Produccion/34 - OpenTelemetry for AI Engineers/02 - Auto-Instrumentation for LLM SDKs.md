# 🔌 Auto-Instrumentation for LLM SDKs

Writing manual OTel spans for every OpenAI, Anthropic, Cohere, and HuggingFace call is tedious and error-prone. **Auto-instrumentation** solves this: a one-line `instrument()` call patches the SDK at import time so every API call automatically emits OTel spans with the right semantic attributes. By the end of this note you will have instrumented OpenAI, Anthropic, Cohere, LiteLLM, and HuggingFace SDKs, plus the supporting libraries (httpx, requests, asyncio, FastAPI, LangChain).

The OpenTelemetry Python ecosystem ships auto-instrumentation packages for nearly every LLM-related library:

| Library | Package | Spans emitted |
|---------|---------|---------------|
| OpenAI | `opentelemetry-instrumentation-openai` | `openai.chat.completions.create`, `openai.embeddings.create` |
| Anthropic | `opentelemetry-instrumentation-anthropic` | `anthropic.messages.create` |
| Cohere | `opentelemetry-instrumentation-cohere` | `cohere.chat`, `cohere.embed` |
| LiteLLM | `opentelemetry-instrumentation-litellm` | Every LiteLLM completion |
| LangChain | `opentelemetry-instrumentation-langchain` | Every chain run, LLM call, tool use |
| LlamaIndex | `opentelemetry-instrumentation-llama-index` | Query engine, retriever |
| HTTPX | `opentelemetry-instrumentation-httpx` | HTTP request spans |
| FastAPI | `opentelemetry-instrumentation-fastapi` | Request/response spans |
| asyncio | `opentelemetry-instrumentation-asyncio` | Context propagation across awaits |

This is the 80/20: 80% of your traces come from auto-instrumentation with 20% of the effort.

## 🎯 Learning Objectives

- Instrument OpenAI, Anthropic, Cohere, and LiteLLM SDKs with one line.
- Capture `gen_ai.*` semantic attributes (model, tokens, latency).
- Instrument supporting libraries: HTTPX, FastAPI, asyncio.
- Tune auto-instrumentation: suppress spans, set capture modes, filter sensitive data.
- Build a unified tracer setup function reused across services.
- Avoid the four most common auto-instrumentation pitfalls.

## 1. The Standard Pattern

```python
# tracer_setup.py — drop into every service
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

def setup_tracing(service_name: str, otlp_endpoint: str = "http://localhost:6006/v1/traces"):
    """One-call OTel setup. Returns a TracerProvider for advanced control."""
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.environ.get("ENV", "dev"),
    })

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider
```

Every service calls `setup_tracing(...)` once at startup. Then auto-instrumentation patches the SDKs.

## 2. OpenAI Auto-Instrumentation

```bash
pip install opentelemetry-instrumentation-openai
```

```python
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

# At app startup — patches the openai module
OpenAIInstrumentor().instrument()

# Now every OpenAI call is traced
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
# A span "openai.chat" was emitted with attributes:
# - gen_ai.system = "openai"
# - gen_ai.request.model = "gpt-4o-mini"
# - gen_ai.usage.input_tokens = 1
# - gen_ai.usage.output_tokens = 5
# - gen_ai.response.model = "gpt-4o-mini-2024-07-18"
```

### Capture Modes

```python
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

# Capture only the metadata (default — privacy-safe)
OpenAIInstrumentor().instrument()

# Capture message content (PII risk!)
OpenAIInstrumentor().instrument(
    capture_prompts=True,        # captures full prompts
    capture_completions=True,    # captures full completions
)

# Async streaming
from openai import AsyncOpenAI
client = AsyncOpenAI()
# Async + streaming are both instrumented automatically
```

> ⚠️ **Advertencia:** `capture_prompts=True` stores the full prompt in the span. **This persists to your backend** (Phoenix, Tempo, Datadog) — production PII risk. **Default to False** unless you have explicit consent and PII redaction in place.

## 3. Anthropic Auto-Instrumentation

```bash
pip install opentelemetry-instrumentation-anthropic
```

```python
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
AnthropicInstrumentor().instrument()

from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
# Span "anthropic.messages" emitted with:
# - gen_ai.system = "anthropic"
# - gen_ai.request.model = "claude-3-5-sonnet-20241022"
# - gen_ai.usage.input_tokens = 5
# - gen_ai.usage.output_tokens = 50
```

## 4. Cohere Auto-Instrumentation

```bash
pip install opentelemetry-instrumentation-cohere
```

```python
from opentelemetry.instrumentation.cohere import CohereInstrumentor
CohereInstrumentor().instrument()

import cohere
client = cohere.Client()
response = client.chat(message="Hello", model="command-r-plus")
# Span "cohere.chat" emitted
```

## 5. LiteLLM Auto-Instrumentation

LiteLLM proxies 100+ LLM providers. One instrumentation covers all of them:

```bash
pip install opentelemetry-instrumentation-litellm litellm
```

```python
from opentelemetry.instrumentation.litellm import LitellmInstrumentor
LitellmInstrumentor().instrument()

import litellm
response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
# Span emitted regardless of underlying provider (OpenAI, Anthropic, etc.)
```

**This is the recommended path for multi-provider apps.** One instrument call covers OpenAI, Anthropic, Cohere, HuggingFace, Replicate, Together, etc.

## 6. LangChain / LlamaIndex

```bash
pip install opentelemetry-instrumentation-langchain opentelemetry-instrumentation-llama-index
```

```python
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
LangchainInstrumentor().instrument()

# Every chain.run() / llm.invoke() is traced
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke("Hello")
# Multiple spans: langchain.llm, langchain.chain.run, etc.
```

## 7. Supporting Libraries

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

def instrument_all():
    """Instrument every supported library."""
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    AsyncioInstrumentor().instrument()
    ThreadingInstrumentor().instrument()

# Call once at app startup
instrument_all()
```

After this call, every HTTP request, every async task, every thread spawn has automatic span propagation.

## 8. Selective Suppression

Sometimes you don't want every span. The `suppress_instrumentation` context manager disables auto-instrumentation in a scope:

```python
from opentelemetry.instrumentation import suppress_instrumentation

# Don't trace the health check
with suppress_instrumentation():
    response = client.health_check()

# Don't trace background polling
with suppress_instrumentation():
    while True:
        poll_external_service()
        time.sleep(60)
```

Use this for **noisy** internal calls (health checks, polling) that would clutter traces.

## 9. The Unified Setup Function

```python
# otel_setup.py — drop into every service

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# LLM SDK instrumentors
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.instrumentation.litellm import LitellmInstrumentor

# Supporting instrumentors
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor

def setup_telemetry(
    service_name: str,
    otlp_endpoint: str = None,
    capture_content: bool = False,
    console_export: bool = False,
):
    """Initialize OTel SDK + instrument all LLM SDKs and supporting libs."""
    otlp_endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:6006/v1/traces",
    )

    # 1. Resource: who is producing traces
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: os.environ.get("SERVICE_VERSION", "1.0.0"),
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.environ.get("ENV", "dev"),
    })

    # 2. Tracer provider
    provider = TracerProvider(resource=resource)

    # 3. Exporter(s)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    # 4. LLM SDK auto-instrumentation
    OpenAIInstrumentor().instrument(capture_prompts=capture_content, capture_completions=capture_content)
    AnthropicInstrumentor().instrument(capture_prompts=capture_content, capture_completions=capture_content)
    CohereInstrumentor().instrument(capture_prompts=capture_content, capture_completions=capture_content)
    LitellmInstrumentor().instrument()

    # 5. Supporting libraries
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    AsyncioInstrumentor().instrument()

    return provider

# Usage in any service:
# setup_telemetry("chat-service", capture_content=False)
```

## 10. The Tracer for Custom Spans

After auto-instrumentation, use the tracer for **business-level spans** that the SDK doesn't know about:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def my_business_function(user_id: str):
    with tracer.start_as_current_span("validate_user_request") as span:
        span.set_attribute("user.id", user_id)
        # ... validation logic ...

        with tracer.start_as_current_span("fetch_user_preferences"):
            # ... nested span ...
            pass
```

The custom span is a **child** of the auto-instrumented OpenAI span if you call OpenAI inside, so the full tree renders.

## 11. ❌/✅ Antipatterns

### ❌ Instrument only the OpenAI SDK

```python
# ⚠️ Misses Qdrant, embeddings, custom logic
OpenAIInstrumentor().instrument()
```

### ✅ Instrument everything

```python
OpenAIInstrumentor().instrument()
HTTPXClientInstrumentor().instrument()  # Qdrant, Pinecone, etc. all use httpx
AsyncioInstrumentor().instrument()
```

### ❌ `capture_prompts=True` in production

```python
# ⚠️ PII in trace storage
OpenAIInstrumentor().instrument(capture_prompts=True)
```

### ✅ Capture content only in dev

```python
import os
OpenAIInstrumentor().instrument(
    capture_prompts=os.environ.get("ENV") == "dev",
    capture_completions=os.environ.get("ENV") == "dev",
)
```

### ❌ Instrument AFTER imports

```python
# ⚠️ Patches don't take effect on already-imported modules
from openai import OpenAI  # imported
OpenAIInstrumentor().instrument()  # too late
client = OpenAI()  # uses non-instrumented OpenAI
```

### ✅ Instrument BEFORE imports

```python
# ✅ Patches applied before any SDK is imported
from otel_setup import setup_telemetry
setup_telemetry("my-service")  # patches openai, anthropic, etc.

# Now imports are instrumented
from openai import OpenAI
client = OpenAI()
```

### ❌ Different exporter per service

```python
# ⚠️ Spans don't correlate across services
# Service A: Phoenix
# Service B: Tempo
# Service C: Datadog
```

### ✅ Same OTLP collector for all

```python
# ✅ One collector fans out to multiple backends
all_services → OTLP Collector → Phoenix + Tempo + Datadog
```

## 12. Production Reality

**Caso real — Production RAG Project:** `setup_telemetry("chat-service", capture_content=False)` is called once at app startup. Every LLM call, every vector search (via httpx → qdrant), every FastAPI request emits OTel spans to Phoenix. The `capture_content=False` flag is critical — prompts contain user data and never leave the application.

**Caso real — Multi-Agent Research System:** LitellmInstrumentor + LangchainInstrumentor covers the entire agent stack in one call. Phoenix dashboard shows the full trace tree (research → fact-audit → synthesis) with token counts, latency, and cost per LLM call. The Phoenix dashboard replaces 3 separate log searches.

## 📦 Compression Code

```python
# 📦 Compression: One-call OTel setup in 30 lines

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup(service_name: str = "ai-service"):
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint=os.environ.get("OTEL_ENDPOINT", "http://localhost:6006/v1/traces"),
    )))
    trace.set_tracer_provider(provider)

    # Auto-instrument all LLM SDKs
    OpenAIInstrumentor().instrument()
    AnthropicInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()  # Covers Qdrant, Pinecone, etc.

# Call BEFORE imports
# setup("my-service")
# from openai import OpenAI  # now instrumented
```

## 🎯 Key Takeaways

1. **Auto-instrumentation is one line per SDK** — `OpenAIInstrumentor().instrument()` patches the SDK.
2. **Capture content only in dev** — `capture_prompts=False` in production prevents PII leakage.
3. **Instrument BEFORE imports** — patches don't take effect on already-imported modules.
4. **LiteLLM is the multi-provider path** — one instrument call covers 100+ providers.
5. **Supporting libraries matter** — httpx, FastAPI, asyncio instrument the non-LLM spans.
6. **Same OTLP collector across services** — fan-out to multiple backends is the standard pattern.
7. **The unified `setup_telemetry` function** — drop into every service for consistent coverage.

## References

- [[00 - Welcome to OpenTelemetry for AI Engineers|Welcome]] — course map.
- [[01 - OTel Primitives|Spans, traces, context]] — what gets emitted.
- [[03 - OTLP Exporters|Exporters]] — where the spans go.
- OTel Python contrib: https://github.com/open-telemetry/opentelemetry-python-contrib
- OTel registry: https://opentelemetry.io/ecosystem/registry/