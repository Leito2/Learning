# 🎓 Capstone — Multi-Service RAG with OpenTelemetry

This capstone wires every prior note into one deployable artifact: a **multi-service RAG system** with full OpenTelemetry coverage, vendor-neutral OTLP export, sampling, PII redaction, and a Phoenix dashboard for the team. It is the integration test for the entire course: traces flow from the FastAPI chat service, through the LangGraph agent, into the embedding service, the Qdrant vector store, and the LLM providers — **one trace_id across all five services**, queryable in Phoenix (or any OTel backend).

The capstone ships as a Docker Compose stack: an OTel Collector at the center, three application services (chat, embedding, agent) sending OTLP to it, and Phoenix + Tempo as backends. **Switching backends is a Collector config change, not a code change.** Production observability that survives vendor decisions.

By the end of this note you will have a working reference deployment with full trace coverage. Total: ~300 lines of Python + 100 lines of docker-compose + 150 lines of Collector config. **Run it locally, query the trace, then deploy to production.**

## 🎯 Learning Objectives

- Deploy a complete multi-service RAG system with full OTel instrumentation.
- Configure the OTel Collector for production: fan-out, sampling, PII redaction.
- Trace requests across 5 services with one trace_id.
- Apply cost attribution and latency budgets.
- Build Phoenix dashboards for cross-service debugging.
- Migrate from Phoenix to Tempo (or vice versa) without code changes.

## 1. The Architecture

```mermaid
flowchart LR
    User[User request] --> Chat[chat-service<br/>FastAPI + LangGraph]
    Chat -->|OTLP| Coll[OTel Collector]
    Chat -->|HTTP| Emb[embedding-service]
    Chat -->|gRPC| Qdrant[(Qdrant)]
    Emb -->|OTLP| Coll
    Qdrant -.->|OTLP via instrumentation| Coll
    Chat -->|HTTPS| OpenAI[OpenAI API]
    Chat -->|HTTPS| Anthropic[Anthropic API]
    OpenAI -.->|OTLP| Coll
    Coll -->|OTLP| Phoenix[Phoenix]
    Coll -->|OTLP| Tempo[Tempo]
    Coll -->|filter| PII[PII Redaction]
```

Five services, one trace, multiple backends.

## 2. The Docker Compose Stack

```yaml
# docker-compose.yml
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.110.0
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol/config.yaml:ro
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    depends_on:
      - phoenix

  phoenix:
    image: arizephoenix/phoenix:latest
    environment:
      PHOENIX_SQL_DATABASE_URL: postgresql://postgres:pass@postgres:5432/phoenix
    ports:
      - "6006:6006"
    depends_on:
      - postgres

  tempo:
    image: grafana/tempo:latest
    command: ["--config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml
    ports:
      - "3200:3200"

  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: phoenix

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"

  embedding-service:
    build: ./services/embedding
    environment:
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4317
      OTEL_SERVICE_NAME: embedding-service
    ports:
      - "8001:8001"
    depends_on:
      - otel-collector

  chat-service:
    build: ./services/chat
    environment:
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4317
      OTEL_SERVICE_NAME: chat-service
      QDRANT_HOST: qdrant
      EMBEDDING_SERVICE_URL: http://embedding-service:8001
    ports:
      - "8000:8000"
    depends_on:
      - embedding-service
      - otel-collector

  agent-service:
    build: ./services/agent
    environment:
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4317
      OTEL_SERVICE_NAME: agent-service
    ports:
      - "8002:8002"
    depends_on:
      - chat-service
      - otel-collector
```

## 3. The OTel Collector Config

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  # Limit memory usage
  memory_limiter:
    check_interval: 1s
    limit_percentage: 80
    spike_limit_percentage: 20

  # Tail-based sampling: keep errors, slow, expensive; sample the rest
  tail_sampling:
    decision_wait: 30s
    num_traces: 50000
    expected_new_traces_per_sec: 100
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}

      - name: high-latency
        type: latency
        latency: {threshold_ms: 5000}

      - name: expensive-tokens
        type: numeric_attribute
        numeric_attribute:
          key: gen_ai.usage.total_tokens
          min_value: 5000

      - name: low-confidence
        type: string_attribute
        string_attribute:
          key: rag.confidence
          values: [low, very_low]

      - name: success-sample
        type: probabilistic
        probabilistic: {sampling_percentage: 5}

  # PII redaction
  attributes/remove_pii:
    actions:
      - key: user.email
        action: delete
      - key: user.credit_card
        action: delete
      - key: gen_ai.prompt
        action: delete
      - key: gen_ai.completion
        action: delete
      - key: request.headers.authorization
        action: delete
      - key: user_id
        action: hash

  # Batch for efficiency
  batch:
    timeout: 5s
    send_batch_size: 1024

exporters:
  otlp/phoenix:
    endpoint: phoenix:4317
    tls: {insecure: true}

  otlp/tempo:
    endpoint: tempo:4317
    tls: {insecure: true}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, tail_sampling, attributes/remove_pii, batch]
      exporters: [otlp/phoenix, otlp/tempo]

  metrics:
    receivers: [otlp]
    processors: [memory_limiter, batch]
    exporters: [otlp/phoenix]
```

## 4. The Shared `otel_setup.py`

```python
# services/_shared/otel_setup.py
import os
from opentelemetry import trace, baggage
from opentelemetry.sdk.trace import TracerProvider, BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor


def setup_telemetry(service_name: str, sampling_rate: float = 0.1):
    """Standard OTel setup for all services in the stack."""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.environ.get("SERVICE_VERSION", "1.0.0"),
        "deployment.environment": os.environ.get("ENV", "production"),
        "cost.center": os.environ.get("COST_CENTER", "ai-platform"),
    })

    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(sampling_rate)),
    )

    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=os.environ.get(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://otel-collector:4317",
                ),
            ),
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )
    )
    trace.set_tracer_provider(provider)

    # Auto-instrument LLM SDKs (capture_content=False in production)
    is_dev = os.environ.get("ENV") == "dev"
    OpenAIInstrumentor().instrument(
        capture_prompts=is_dev,
        capture_completions=is_dev,
    )
    AnthropicInstrumentor().instrument(
        capture_prompts=is_dev,
        capture_completions=is_dev,
    )

    # Supporting libraries
    HTTPXClientInstrumentor().instrument()  # Qdrant, OpenAI HTTP, etc.
    AsyncioInstrumentor().instrument()

    return provider


# Helper for thread_id propagation
def attach_thread_context(thread_id: str, user_id: str = ""):
    from opentelemetry.context import attach
    ctx = baggage.set_baggage("thread_id", thread_id)
    if user_id:
        ctx = baggage.set_baggage("user_id", user_id, context=ctx)
    return attach(ctx)
```

## 5. The Chat Service

```python
# services/chat/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header
from pydantic import BaseModel
import httpx

# OTel MUST be set up before any framework imports
from otel_setup import setup_telemetry, attach_thread_context
setup_telemetry("chat-service")
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor().instrument()

from opentelemetry import trace, baggage
from opentelemetry.context import attach, detach

tracer = trace.get_tracer(__name__)


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Chat Service", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_user_id: str = Header(default="")):
    token = attach_thread_context(req.thread_id, x_user_id)
    try:
        with tracer.start_as_current_span("chat.handler") as span:
            span.set_attribute("rag.user_query", req.query[:200])
            span.set_attribute("user.id", x_user_id)

            # Stage 1: Embedding (cross-service call)
            with tracer.start_as_current_span("embedding.call") as span:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://embedding-service:8001/embed",
                        json={"text": req.query},
                        headers={"traceparent": inject_traceparent()},  # propagate context
                    )
                    embedding = response.json()["embedding"]
                    span.set_attribute("embedding.dimensions", len(embedding))

            # Stage 2: Vector search
            with tracer.start_as_current_span("qdrant.search") as span:
                span.set_attribute("db.system", "qdrant")
                span.set_attribute("db.collection", "research")
                # qdrant-client emits its own spans via instrumentation
                from qdrant_client import QdrantClient
                client = QdrantClient(host="qdrant", port=6333)
                results = client.search(
                    collection_name="research",
                    query_vector=embedding,
                    limit=10,
                )
                span.set_attribute("db.qdrant.results_count", len(results))

            # Stage 3: Generation
            with tracer.start_as_current_span("openai.chat") as span:
                from openai import OpenAI
                client = OpenAI()
                prompt = build_prompt(req.query, results)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content
                span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
                span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)

            return ChatResponse(answer=answer, sources=[r.id for r in results])
    finally:
        from opentelemetry.context import detach
        detach(token)


def inject_traceparent() -> str:
    """Inject current trace context into HTTP headers."""
    from opentelemetry.propagate import inject
    headers = {}
    inject(headers)
    return headers.get("traceparent", "")


def build_prompt(query: str, results) -> str:
    context = "\n".join([r.payload.get("text", "") for r in results])
    return f"Context:\n{context}\n\nQuestion: {query}"
```

## 6. The Embedding Service

```python
# services/embedding/main.py
from otel_setup import setup_telemetry
setup_telemetry("embedding-service")

from fastapi import FastAPI
from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel

app = FastAPI()
tracer = trace.get_tracer(__name__)
client = OpenAI()


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    with tracer.start_as_current_span("embedding.generate") as span:
        span.set_attribute("text.length", len(req.text))
        span.set_attribute("gen_ai.request.model", "text-embedding-3-small")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=req.text,
        )
        embedding = response.data[0].embedding
        span.set_attribute("embedding.dimensions", len(embedding))
        return EmbedResponse(embedding=embedding)
```

## 7. Running the Stack

```bash
# 1. Build images
docker compose build

# 2. Start the stack
docker compose up -d

# 3. Send a request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Id: u-42" \
  -d '{"query": "What is OpenTelemetry?", "thread_id": "demo-1"}'

# 4. View traces in Phoenix
open http://localhost:6006
# Filter by thread_id=demmo-1 → see the full trace tree

# 5. View traces in Tempo (Grafana)
open http://localhost:3200  # Tempo API
# Configure Grafana datasource at http://localhost:3000
```

## 8. Phoenix Dashboard Configuration

Create a Phoenix dashboard that shows the cross-service view:

```python
# dashboard.py
import phoenix as px

project = px.Client().get_project(name="chat-service")

# Save the project's dashboard config
dashboard_config = {
    "metrics": [
        {"name": "request_count", "type": "count", "groupby": ["service.name"]},
        {"name": "latency_p99", "type": "percentile", "percentile": 99, "groupby": ["http.route"]},
        {"name": "cost_usd_total", "type": "sum", "field": "cost.usd", "groupby": ["gen_ai.request.model"]},
        {"name": "tokens_total", "type": "sum", "field": "gen_ai.usage.total_tokens", "groupby": ["service.name"]},
        {"name": "error_rate", "type": "rate", "field": "exception", "groupby": ["service.name"]},
    ],
}
```

Or use the Phoenix UI directly: **Dashboards → New → Add Panel → Query from spans**.

## 9. Migrating Backends

To migrate from Phoenix to Tempo:

```yaml
# otel-collector-config.yaml
exporters:
  otlp/tempo:
    endpoint: tempo-new:4317
  # otlp/phoenix:  # disabled
  #   endpoint: phoenix:4317
service:
  pipelines:
    traces:
      exporters: [otlp/tempo]  # only Tempo now
```

**App code unchanged.** The Collector routes the same OTLP to a different backend.

To add Datadog alongside Phoenix:

```yaml
exporters:
  otlp/phoenix:
    endpoint: phoenix:4317
  datadog:
    api:
      key: ${DD_API_KEY}
service:
  pipelines:
    traces:
      exporters: [otlp/phoenix, datadog]
```

Two backends simultaneously. The team uses Phoenix for AI-specific queries; Datadog for general APM.

## 10. Production Reality

**Caso real — Production RAG Project:** The stack runs in production with 5 services. Phoenix shows the trace tree for every user request; the team's debugging time for "why is this slow?" dropped from 30 minutes (log archaeology) to 2 minutes (Phoenix tree). The OTel Collector fans out to Phoenix (for AI queries) and Tempo (for general HTTP traces); the team added Datadog APM by changing Collector config, no app code changes.

**Caso real — Multi-Agent Research System:** With 8 services in the agent stack, the trace tree per request has 30+ spans. Tail-based sampling keeps all errors and slow traces; the 5% sample of fast traces gives aggregate metrics. Cost attribution via `cost.usd` attributes enables per-team budget dashboards in Grafana.

## 📦 Compression Code

```yaml
# docker-compose.yml (compressed)
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.110.0
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes: [./otel-collector-config.yaml:/etc/otelcol/config.yaml:ro]
    ports: ["4317:4317", "4318:4318"]

  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["6006:6006"]

  tempo:
    image: grafana/tempo:latest
    ports: ["3200:3200"]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]

  chat-service:
    build: ./services/chat
    environment:
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4317
      OTEL_SERVICE_NAME: chat-service
    ports: ["8000:8000"]
```

```python
# otel_setup.py — drop into every service (compressed)
def setup_telemetry(service_name: str, sampling_rate: float = 0.1):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(sampling_rate)))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument(capture_prompts=False, capture_completions=False)
    HTTPXClientInstrumentor().instrument()
    FastAPIInstrumentor().instrument()
```

## 🎯 Key Takeaways

1. **One OTLP collector for the whole stack** — services send to `collector:4317`, Collector fans out.
2. **W3C Trace Context propagation** — every HTTP call injects `traceparent`; downstream services extract it.
3. **Same `otel_setup.py` for every service** — sampling rate, resource labels, instrumentation.
4. **PII redaction at the Collector** — final defense, never rely on app-level filtering.
5. **Tail sampling for production** — keep errors and slow traces; sample 5% of fast successes.
6. **Cost attribution via span attributes** — `cost.usd`, `cost.model`, `cost.input_tokens`.
7. **Switching backends is config** — Phoenix → Tempo, +Datadog, all without app code changes.

## References

- [[00 - Welcome to OpenTelemetry for AI Engineers|Welcome]] — course map.
- [[01 - OTel Primitives|Spans, traces, context]] — the foundation.
- [[02 - Auto-Instrumentation|LLM SDKs]] — instrumented automatically.
- [[03 - OTLP Exporters|Exporters]] — Phoenix + Tempo + Datadog.
- [[04 - OTel for LangGraph|Agent tracing]] — LangGraph with OTel.
- [[05 - OTel for RAG Pipelines|RAG tracing]] — the 5-stage tree.
- [[06 - Production Patterns|Sampling, costs, PII]] — the production discipline.
- Docker Compose example: https://opentelemetry.io/docs/collector/deployment/
- Phoenix: https://docs.arize.com/phoenix