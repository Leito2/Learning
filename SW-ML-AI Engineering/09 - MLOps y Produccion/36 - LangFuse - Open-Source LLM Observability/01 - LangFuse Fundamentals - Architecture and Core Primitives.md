# 🎯 01 - LangFuse Fundamentals — Architecture and Core Primitives

> **The data model that every other note builds on. Project, Trace, Span, Observation. Self-hosted vs cloud. Storage tiers. Auth.**

## 🎯 Learning Objectives
- Distinguish Trace / Span / Observation / Generation / Event in the LangFuse data model
- Deploy LangFuse locally on Docker Compose (Postgres + ClickHouse + Redis + MinIO)
- Decide between self-hosted and LangFuse Cloud based on cost, regulation, and team size
- Wire API key authentication for a service and JWT for an SSO-enabled org
- Choose storage backends (Postgres for metadata, ClickHouse for trace events, S3/MinIO for blobs)
- Apply the `@observe()` decorator and the context-manager `langfuse_context` patterns

## Introduction

LangFuse's data model is a deliberately small set of primitives that compose cleanly. A **Project** owns every other entity — one per application or per team in a multi-tenant deployment. A **Trace** represents one logical user-facing request (e.g. one RAG query, one agent run). A **Span** is a unit of work within a trace — a "retrieve documents" span, a "rerank" span, a "generate answer" span. An **Observation** is the actual LLM or generic call — its input, output, model, tokens, latency, cost. Three subtypes exist: **Generation** for LLM calls, **Embedding** for vector calls, and **Agent** for autonomous loop spans.

The whole stack runs on three storage layers: **Postgres** for metadata (projects, datasets, prompts, users, scores), **ClickHouse** (or Postgres, depending on tier) for trace events, and an object store (S3, MinIO, GCS, Azure Blob) for large blobs (logs of full payloads, attachments). This separation mirrors the design used by every production observability tool — Splunk, Datadog, Honeycomb — and is what enables LangFuse to handle billions of trace events without bloating metadata storage.

```mermaid
graph TB
    subgraph "LangFuse Application (Node + Next.js)"
        UI[Web UI :3000]
        API[Public API :3000]
        WORKER[Worker :3030]
    end
    
    subgraph Storage
        PG[(Postgres<br/>metadata)]
        CH[(ClickHouse<br/>events)]
        S3[(S3 / MinIO<br/>blobs)]
        REDIS[(Redis<br/>cache + queue)]
    end
    
    subgraph "SDKs and Apps"
        SDK[LangFuse SDK]
        OTLP[OTLP Receiver]
    end
    
    SDK -->|ingest| API
    OTLP -->|OTLP HTTP| API
    API --> PG
    API --> CH
    API --> S3
    WORKER --> PG
    WORKER --> CH
    REDIS -.->|queue| WORKER
```

The architecture supports three deployment modes: **SaaS Cloud** (hosted by LangFuse), **Self-hosted OSS** (Docker Compose, Kubernetes, single-VM), and **Enterprise** (multi-region, SSO, audit logs). For the AI/ML Engineer profile, **self-hosted OSS on a single VM** is the sweet spot — full data sovereignty at ~$50/month infrastructure cost.

![LangFuse architecture diagram](https://langfuse.com/_next/image?url=%2Fstatic%2Fimages%2Fdocs%2Farchitecture.png&w=1920&q=75)

---

## 1. The Data Model

### 1.1 Project, Trace, Span, Observation

```
Project (1)
├── Trace (N)
│   ├── Observation (N)  ← Generation | Span | Event | Embedding | Agent
│   └── Score (N)        ← human feedback, LLM-as-judge, custom metrics
├── Dataset (N)
│   ├── DatasetItem (N)
│   └── DatasetRun (N)
├── Prompt (N)
│   └── PromptVersion (N)
└── Score (N)             ← run-level config + templates
```

| Primitive | Purpose | Mutability |
|-----------|---------|------------|
| **Project** | Top-level container. Holds API keys, members, settings | Editable |
| **Trace** | One logical request from user input to system output | Immutable once committed |
| **Span** | A unit of work. Used for non-LLM operations (retrieval, reranking) | Immutable |
| **Generation** | An LLM call. Captures model, prompt, completion, tokens, cost | Immutable |
| **Embedding** | A vector call. Captures model, input, dimensions | Immutable |
| **Event** | A timestamped event (cache hit, retry, validation error) | Immutable |
| **Agent** | An autonomous-loop wrapper for multi-step agents | Immutable |
| **Score** | Numeric/string evaluation result attached to a trace | Editable (add/delete) |
| **Dataset** | Versioned collection of inputs + expected outputs | Editable |
| **Prompt** | Versioned prompt template with variables | Editable (each version immutable) |

### 1.2 A concrete example

Consider a RAG query: "What is the capital of France?"

```python
from langfuse import Langfuse, observe
from openai import OpenAI

langfuse = Langfuse()  # env vars: LANGFUSE_PUBLIC_KEY, SECRET_KEY, HOST
openai = OpenAI()

@observe(name="rag_query")
def rag_query(question: str) -> str:
    # 1. Span — document retrieval
    docs = retrieve_documents(question)
    
    # 2. Generation — answer synthesis
    answer = synthesize_answer(question, docs)
    return answer

@observe(name="retrieve_documents", as_type="span")
def retrieve_documents(question: str) -> list[str]:
    return [doc.text for doc in vector_store.search(question, k=5)]

@observe(name="synthesize_answer", as_type="generation")
def synthesize_answer(question: str, docs: list[str]) -> str:
    context = "\n".join(docs)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using this context:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
```

This produces **one Trace** with **two Spans** (retrieve_documents, rag_query itself as parent span) and **one Generation** (synthesize_answer). The Generation captures model, prompt, completion, token counts, cost. The Spans capture retrieval latency and result size. The Trace has the question and final answer.

In the UI:
- Trace list shows the question + total cost + total latency
- Trace detail shows the call tree with input/output per span
- Scores can be attached to the trace (user thumbs up, LLM-as-judge score)

This is the **canonical observability pattern**: every request leaves a structured trace with the full lineage from input to output through every intermediate step.

---

## 2. Self-Hosted Deployment

### 2.1 Docker Compose (single VM)

The fastest path to a self-hosted LangFuse is the official Docker Compose. It boots four services on one machine:

```yaml
# docker-compose.yml
version: "3.9"

services:
  langfuse-web:
    image: langfuse/langfuse:main
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
      - NEXTAUTH_SECRET=mysecret  # change me
      - NEXTAUTH_URL=http://localhost:3000
      - TELEMETRY_DISABLED=true
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minio
      - MINIO_ROOT_PASSWORD=miniosecret  # change me
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

```bash
docker compose up -d
# Wait 30s for migrations
open http://localhost:3000
```

First-time setup creates an admin user, organization, and project. API keys appear under Project Settings → API Keys — copy the `pk-lf-...` (public) and `sk-lf-...` (secret) keys.

⚠️ **Watch out:** The default `NEXTAUTH_SECRET` is a placeholder. Set a strong 32+ char random string in production. For SSO, swap `NEXTAUTH_SECRET` and add `LANGFUSE_ALLOWED_ORGANANIZATION_DOMAINS=yourcompany.com`.

### 2.2 Kubernetes (production scale)

For multi-tenant production, the LangFuse Helm chart handles the same services plus Ingress, HPA, secrets:

```bash
helm repo add langfuse https://langfuse.github.io/langfuse-helm
helm repo update

helm install langfuse langfuse/langfuse \
    --namespace langfuse --create-namespace \
    --set langfuse.nextauth.secret=$(openssl rand -base64 32) \
    --set postgresql.auth.password=$(openssl rand -base64 24) \
    --set redis.auth.password=$(openssl rand -base64 24) \
    --set clickhouse.auth.password=$(openssl rand -base64 24)
```

The chart installs:
- `langfuse-web` (Deployment, 2+ replicas, HPA on CPU)
- `langfuse-worker` (Deployment, 1+ replica, runs eval/processing jobs)
- `postgresql` (StatefulSet, 20Gi PVC)
- `clickhouse` (StatefulSet, 50Gi PVC) — enabled via `clickhouse.enabled=true`
- `redis` (StatefulSet or external)
- `minio` (StatefulSet — or use external S3/GCS)

For Cloud-managed Postgres (RDS, Cloud SQL), set `postgresql.enabled=false` and override `externalPostgresql.host` etc. Same for ClickHouse (LangFuse supports ClickHouse Cloud, AWS-managed, Altinity Cloud).

### 2.3 Cost economics at 10M traces/month

| Component | Self-hosted cost | Cloud equivalent |
|-----------|-----------------:|-----------------:|
| Postgres (4 vCPU, 16GB RAM) | ~$80/mo (Hetzner, OVH) | ~$200/mo (RDS db.r6g.xlarge) |
| ClickHouse (8 vCPU, 32GB RAM) | ~$120/mo | ~$400/mo (ClickHouse Cloud) |
| Redis (1 vCPU, 2GB RAM) | ~$15/mo | ~$30/mo (ElastiCache cache.t4g.small) |
| MinIO/S3 (500GB storage) | ~$10/mo (Hetzner Storage Box) | ~$12/mo (S3 Standard) |
| LangFuse app (2 × 2 vCPU, 4GB) | ~$60/mo | ~$120/mo (ECS Fargate) |
| **Total** | **~$285/mo** | **~$760/mo** |
| **LangFuse Cloud Pro** | — | **$399/mo (unlimited traces, 5 users)** |

For 10M traces/month and a 5-person team, self-hosting is **$285/mo vs $399/mo Cloud Pro** — close. For larger teams (10+ users), Cloud Pro becomes $59/user/mo additional. Self-hosting wins on data sovereignty and customization; Cloud wins on ops overhead and shared security audits.

💡 **Tip:** Pick self-hosting if (1) regulatory constraints require it (HIPAA, GDPR-DPA, FedRAMP), (2) you need custom plugins or PII redaction, or (3) your team has 1+ DevOps FTE. Pick Cloud if you have < 5 engineers and need to ship in < 1 day.

---

## 3. Authentication and Authorization

### 3.1 API keys (service-to-service)

Generated per project under Settings → API Keys. Two key types:

- **Public key** (`pk-lf-...`) — safe to embed in client browsers; can only ingest trace data.
- **Secret key** (`sk-lf-...`) — server-side only; can manage prompts, datasets, scores.

```python
import os
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),  # pk-lf-...
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),  # sk-lf-...
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)
```

### 3.2 JWT and SSO (Enterprise tier)

For org-wide auth, LangFuse supports:
- **Google OAuth** — set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `LANGFUSE_ALLOWED_GOOGLE_DOMAINS`
- **GitHub OAuth** — set `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `LANGFUSE_ALLOWED_GITHUB_LOGINS`
- **Generic OIDC** — `AUTH_<PROVIDER>_CLIENT_ID`, etc.
- **Email/Password** with managed SMTP — useful for self-hosted single-org deployments
- **SCIM provisioning** (Enterprise) — automate user lifecycle

### 3.3 The `organizationId` and `projectId` for multi-tenant

A self-hosted LangFuse can host multiple projects per organization. Each project has independent API keys. For SaaS apps serving multiple customers:

```python
# Per-tenant LangFuse instance
def make_langfuse(tenant_id: str) -> Langfuse:
    return Langfuse(
        public_key=f"pk-lf-{tenant_id}-...",  # pseudocode; real keys are project-scoped
        secret_key=f"sk-lf-{tenant_id}-...",
        host=os.getenv("LANGFUSE_HOST"),
    )
```

In production, use **one LangFuse project per tenant** to isolate trace data and enforce per-tenant RBAC. The traces UI can be filtered to show only the current tenant's data.

---

## 4. The `@observe()` Decorator

The single most useful LangFuse primitive for application code:

```python
from langfuse import observe, langfuse_context

@observe()
def my_llm_call(prompt: str) -> str:
    # LangFuse automatically captures:
    # - function input → trace input
    # - function output → trace output
    # - latency → span duration
    # - nested LLM calls → child observations
    return call_openai(prompt)

@observe(as_type="generation", name="openai_call")
def call_openai(prompt: str) -> str:
    # Marked as Generation observation
    # Can attach metadata:
    langfuse_context.update_current_observation(
        model="gpt-4o-mini",
        model_parameters={"temperature": 0.7, "max_tokens": 500},
        metadata={"user_id": "user_123"},
    )
    response = openai.chat.completions.create(...)
    langfuse_context.update_current_observation(
        usage={"input_tokens": 100, "output_tokens": 50},
        cost=0.00015,
    )
    return response.choices[0].message.content
```

`@observe()` works recursively: nested `@observe()` calls become nested spans. The decorator handles the span lifecycle (start on entry, finish on exit, capture exceptions). The `langfuse_context.update_current_observation()` lets you enrich an observation with metadata, usage, cost, and scores.

### 4.1 Async support

```python
@observe(as_type="generation")
async def call_openai_async(prompt: str) -> str:
    response = await openai_async.chat.completions.create(...)
    langfuse_context.update_current_observation(
        usage={"input_tokens": 100, "output_tokens": 50},
        cost=0.00015,
    )
    return response.choices[0].message.content
```

Async functions work out of the box — the decorator handles the asyncio task lifecycle.

### 4.2 Capture-and-flush for batch jobs

For scripts that process thousands of records, `@observe()` accumulates observations in memory and flushes to the API in batches:

```python
import langfuse
langfuse.flush()  # force flush
```

Add `langfuse.flush()` at the end of any batch script to ensure all traces are persisted before the script exits. Without it, in-process observations can be lost.

---

## 5. The Trace API and Manual Span Management

For low-level control, use the trace API:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Manually start a trace
trace = langfuse.trace(
    name="manual_rag_query",
    input={"question": "What is the capital of France?"},
    metadata={"user_id": "user_123", "session_id": "abc-def"},
    tags=["production", "rag"],
)

# Span
span = trace.span(
    name="retrieve_documents",
    input={"query": "What is the capital of France?", "k": 5},
)
results = vector_store.search("What is the capital of France?", k=5)
span.end(output={"documents": results})

# Generation
generation = trace.generation(
    name="synthesize_answer",
    model="gpt-4o-mini",
    model_parameters={"temperature": 0.7},
    input=[{"role": "user", "content": f"Context: {results}\n\nAnswer the question."}],
)
response = openai.chat.completions.create(...)
generation.end(
    output=response.choices[0].message.content,
    usage={
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    },
    metadata={"provider": "openai", "request_id": response.id},
)

# Score attached to the trace
trace.score(name="user_feedback", value=1, comment="Helpful answer")

# Update trace with final output
trace.update(output={"answer": response.choices[0].message.content})
```

This pattern is the right choice when:
- The traced function is **not a Python function** (e.g. a Flask/FastAPI route handler).
- You need **manual control** over span boundaries (e.g. the span starts at request receipt and ends at response send, spanning middleware).
- The trace should attach to **business logic events** rather than function calls.

⚠️ **Watch out:** Always call `.end()` on manually-managed spans. Unclosed spans become orphaned observations that clutter the UI. Use the `@observe()` decorator for tight scope; use the trace API only for cross-cutting lifecycle.

---

## 6. Storage Tiers and Tradeoffs

### 6.1 Postgres-only (default)

For < 1M traces/month, the simple Postgres-only setup is sufficient:

```
langfuse-web → Postgres (metadata + events)
```

This is what `docker compose up` on a single VM gives you. Performance is acceptable up to ~10K traces/day with proper indexing.

### 6.2 Postgres + ClickHouse (scale-out)

For > 1M traces/month, LangFuse supports ClickHouse as a separate event store:

```
langfuse-web → Postgres (metadata)
              → ClickHouse (trace events)
```

ClickHouse's columnar storage handles the high-cardinality trace data efficiently. Aggregation queries (e.g. "p95 latency by model per day") are 10-100× faster than Postgres. ClickHouse Cloud or self-hosted ClickHouse both work.

### 6.3 S3/MinIO for blobs

Full payloads (multi-MB prompts/completions, image inputs, audio outputs) are stored in S3-compatible object storage. The trace event references the S3 path. For self-hosted, MinIO provides an S3-compatible API.

| Storage | Postgres | ClickHouse | S3/MinIO |
|---------|:--------:|:----------:|:--------:|
| **Self-hosted default** | ✅ | ❌ | ✅ MinIO |
| **Self-hosted scale** | ✅ | ✅ | ✅ MinIO or S3 |
| **Cloud-managed** | RDS | ClickHouse Cloud | S3 / GCS / Azure Blob |

💡 **Tip:** Use S3-compatible storage (MinIO, SeaweedFS) even at small scale. Saves the Postgres footprint, simplifies backups (snapshots of metadata only), and keeps the door open for object lifecycle policies.

---

## 7. Antipatterns

### 7.1 Antipattern 1: Logging full PII in trace inputs

```python
# ❌ Compliance bug: SSN in trace input persists forever
@observe()
def credit_check(ssn: str) -> bool:
    return llm_classify(f"Should we approve SSN {ssn}?")

# ✅ Correct: hash or omit PII before tracing
import hashlib
@observe()
def credit_check(ssn: str) -> bool:
    ssn_hash = hashlib.sha256(ssn.encode()).hexdigest()[:12]
    return llm_classify(f"Should we approve SSN hash {ssn_hash}?")
```

For full PII redaction, use [[06 - Large Language Models/15 - LLM Security and Guardrails]] before tracing, or configure LangFuse's `mask` patterns.

### 7.2 Antipattern 2: Forgetting to flush in batch jobs

```python
# ❌ Bug: end of script → process exit → in-process spans lost
for doc in docs:
    extract(doc)

# ✅ Correct: explicit flush at end
for doc in docs:
    extract(doc)
langfuse.flush()  # wait for queued observations to be sent
```

### 7.3 Antipattern 3: Tracing every function recursively

```python
# ❌ Cost explosion: every helper function becomes a span
@observe()
@observe()
@observe()
@observe()
def depth_4():
    return helper()
```

Each `@observe()` adds an HTTP call to the LangFuse API. At 10K requests/sec, that is 40K HTTP calls/sec. Limit `@observe()` to one per logical unit (one LLM call, one retrieval, one tool use).

### 7.4 Antipattern 4: Not using namespaces/tags

```python
# ❌ Indistinguishable traces
@observe(name="llm_call")
def my_call(prompt: str) -> str: ...

# ✅ Correct: tag for filtering
@observe(name="llm_call", tags=["production", "english", "gpt-4o-mini"])
def my_call(prompt: str) -> str: ...
trace.update(tags=["...")
```

Without tags, the UI is unfilterable. With tags, you can filter to "production, english, gpt-4o-mini" and analyze just that subset.

### 7.5 Antipattern 5: Using LangFuse Cloud for HIPAA data

```python
# ❌ Regulatory: PHI in a SaaS without BAA
trace = langfuse.trace(name="phi_query", input={"patient_id": "..."})

# ✅ Use self-hosted LangFuse or enable Enterprise BAA from LangFuse Cloud
```

Self-hosted is the only universally compliant option. LangFuse Cloud offers a BAA on Enterprise tier; for HIPAA-strict industries, self-host on your own infrastructure.

---

## 🎯 Key Takeaways

- LangFuse's data model has five primary entities: Project, Trace, Span/Observation, Dataset, Prompt — plus Scores.
- Self-hosting on Docker Compose costs ~$285/mo for 10M traces/month; Cloud Pro is $399/mo for the first 5 users.
- `@observe()` decorator captures function calls, async included; the trace API gives manual control over span lifecycle.
- Storage tiers: Postgres-only for small scale, +ClickHouse for >1M traces/month, +MinIO/S3 for blob storage.
- API keys: public (`pk-lf-...`) for client-side ingestion, secret (`sk-lf-...`) for full management.
- Use SSO/OIDC for org-wide auth; one project per tenant for multi-tenant SaaS apps.
- Avoid logging PII in trace inputs, forgetting `langfuse.flush()` in batch jobs, recursive `@observe()`, and using SaaS for regulated data.

## References

- LangFuse docs — [langfuse.com/docs](https://langfuse.com/docs)
- Self-hosting guide — [langfuse.com/docs/deployment/self-host](https://langfuse.com/docs/deployment/self-host)
- Helm chart — [github.com/langfuse/langfuse-helm](https://github.com/langfuse/langfuse-helm)
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — Phoenix for LLM traces + drift
- [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers|OpenTelemetry for AI Engineers]] — protocol layer
- [[09 - MLOps y Produccion/35 - LangSmith Deep Dive|LangSmith Deep Dive]] — SaaS counterpart
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — Kubernetes deployment
- [[02 - Docker Profesional|Docker Profesional]] — Docker Compose patterns
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — service integration patterns
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/02 - LLM SDK Auto-Instrumentation|Note 02 — SDK Auto-Instrumentation]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/05 - Capstone - Self-Hosted LangFuse for Multi-Provider RAG|Note 05 — Capstone]]
