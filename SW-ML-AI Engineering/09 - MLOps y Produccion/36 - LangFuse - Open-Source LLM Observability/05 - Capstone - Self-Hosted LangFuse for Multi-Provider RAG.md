# 🎯 05 - Capstone — Self-Hosted LangFuse for Multi-Provider RAG

> **The capstone. A complete self-hosted LangFuse stack integrated with a FastAPI + Instructor + LiteLLM multi-provider RAG service. Every trace, prompt version, dataset, and online evaluator in one Docker Compose.**

## 🎯 Learning Objectives
- Stand up a complete LangFuse stack (Web + Worker + Postgres + ClickHouse + Redis + MinIO) on Docker Compose
- Wire a FastAPI RAG service to LangFuse via SDK wrappers and decorators
- Implement multi-provider routing through LiteLLM with LangFuse cost attribution
- Bootstrap an evaluation dataset from production traces
- Wire online evaluators on 10% sampled traffic with alerting
- Build a CI pipeline that runs offline evaluations on every prompt change

## Introduction

The capstone ties together every concept from the previous four notes into a single self-hosted, production-shaped observability stack. The architecture is deliberately realistic: a FastAPI RAG service running on port 8080, LangFuse on port 3000, MinIO on 9000, Postgres on 5432 — all reachable from a developer's laptop with one `docker compose up` command.

```mermaid
graph TB
    subgraph "External Traffic"
        CLIENT[Frontend / curl / SDK]
    end
    
    subgraph "Self-Hosted Stack (Docker Compose)"
        subgraph "LangFuse (:3000)"
            WEB[LangFuse Web]
            WORKER[LangFuse Worker]
        end
        
        subgraph "Storage"
            PG[(Postgres :5432)]
            CH[(ClickHouse :8123)]
            REDIS[(Redis :6379)]
            MINIO[(MinIO :9000)]
        end
        
        subgraph "RAG Service (:8080)"
            FA[FastAPI]
            INST[instructor + LiteLLM]
            EVAL[Online Evaluator]
        end
        
        subgraph "LLM Providers"
            OA[OpenAI API]
            ANT[Anthropic API]
            GR[Groq API]
        end
    end
    
    subgraph "User Feedback"
        FB[Thumbs up/down]
    end
    
    CLIENT -->|HTTP| FA
    FA -->|trace ingestion| WEB
    FA -->|LLM call| INST
    INST -->|HTTP| OA
    INST -->|HTTP| ANT
    INST -->|HTTP| GR
    EVAL -->|LLM-as-Judge 10%| OA
    CLIENT -->|feedback| FB
    FB -->|score| WEB
    WEB --> PG
    WEB --> CH
    WEB --> MINIO
    WORKER --> PG
    WORKER --> CH
    WEB -.->|cache| REDIS
```

The flow:
1. Frontend calls FastAPI `/query` with a question
2. FastAPI retrieves context, calls LLM via Instructor + LiteLLM
3. Every LLM call auto-traces to LangFuse (via SDK wrappers)
4. Online evaluator scores 10% of responses via LLM-as-Judge
5. User feedback (thumbs up/down) attaches to traces
6. Offline evaluation runs on a fixed dataset via CI
7. Prompt registry stores versioned prompts with rollback

This is the **observability spine of a production RAG service**. Every line of every project — LLM Edge Gateway, Automated LLM Evaluation Suite, Multi-Agent Research System, StayBot, and the capstone from [[06 - Large Language Models/22 - Instructor and Structured Generation]] — can be retrofitted to this pattern.

![Self-hosted LangFuse UI](https://langfuse.com/_next/image?url=%2Fstatic%2Fimages%2Flanding%2Fself-host-light.png&w=1920&q=75)

---

## 1. Project Layout

```
langfuse-rag-stack/
├── docker-compose.yml              # LangFuse + storage + RAG service
├── .env.example                    # API keys, secrets
├── langfuse/
│   ├── seed-prompts.py            # Initial prompt registry setup
│   ├── seed-datasets.py            # Initial evaluation dataset
│   ├── seed-evaluators.py          # LLM-as-judge evaluator config
│   └── alerting/
│       └── drift-detector.py       # Hourly cron: moving average + alert
├── app/
│   ├── main.py                     # FastAPI app + lifespan
│   ├── retriever.py                # Vector search with Qdrant
│   ├── generator.py                # Instructor + LiteLLM orchestrator
│   ├── observability.py            # @observe decorators + SDK wrappers
│   └── feedback.py                 # User feedback endpoint
├── tests/
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_e2e.py                 # End-to-end RAG with mock LLM
├── eval/
│   ├── run_offline_eval.py         # CI evaluation runner
│   └── compare_prompts.py          # Statistical comparison
├── pyproject.toml
└── README.md
```

The project follows the structure recommended in [[16 - Harness Engineering/05 - File Architecture|The Harness Engineering File Architecture]] — one concern per file, observability as a first-class module, evaluation as a separate `eval/` package.

---

## 2. The Docker Compose Stack (`docker-compose.yml`)

```yaml
version: "3.9"

services:
  # ========== RAG Service ==========
  rag-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
      - LANGFUSE_HOST=http://langfuse-web:3000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      langfuse-web:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    restart: unless-stopped

  # ========== LangFuse ==========
  langfuse-web:
    image: langfuse/langfuse:main
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
      - NEXTAUTH_URL=http://localhost:3000
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
      - TELEMETRY_DISABLED=true
      - LANGFUSE_INIT_ORG_ID=acme
      - LANGFUSE_INIT_ORG_NAME=Acme Corp
      - LANGFUSE_INIT_PROJECT_ID=rag_prod
      - LANGFUSE_INIT_PROJECT_NAME=RAG Production
      - LANGFUSE_INIT_USER_EMAIL=admin@acme.com
      - LANGFUSE_INIT_USER_NAME=Admin
      - LANGFUSE_INIT_USER_PASSWORD=${ADMIN_PASSWORD}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/public/health"]
      interval: 10s
      timeout: 3s
      retries: 5

  langfuse-worker:
    image: langfuse/langfuse:main
    command: ["worker"]
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy

  # ========== Storage ==========
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

  clickhouse:
    image: clickhouse/clickhouse-server:24-alpine
    environment:
      - CLICKHOUSE_DB=langfuse
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8123/ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minio
      - MINIO_ROOT_PASSWORD=${MINIO_PASSWORD}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ========== Vector DB ==========
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  postgres_data:
  clickhouse_data:
  minio_data:
  qdrant_data:
```

```bash
# One command to boot the entire stack
cp .env.example .env  # add your API keys
docker compose up -d

# Wait 30s for migrations
open http://localhost:3000  # LangFuse UI
open http://localhost:8080/docs  # RAG service docs
```

The stack is fully self-contained — no external dependencies beyond the LLM APIs. All data stays on your infrastructure.

---

## 3. The RAG Service (`app/main.py`)

```python
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retriever import Retriever
from .generator import Generator
from .feedback import router as feedback_router
from .observability import setup_observability


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup: create retriever, generator, observe handles."""
    app.state.retriever = Retriever(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection="documents",
    )
    app.state.generator = Generator(
        default_model="gpt-4o-mini",
        fallback_models=["claude-3-5-sonnet-20241022", "groq/llama-3.3-70b-versatile"],
    )
    yield
    # No teardown needed


app = FastAPI(title="Production RAG with LangFuse", lifespan=lifespan)
setup_observability(app)
app.include_router(feedback_router)


class QueryRequest(BaseModel):
    question: str
    model: str = "gpt-4o-mini"
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    trace_id: str  # for client-side feedback


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """RAG query with full LangFuse trace lineage."""
    try:
        result = await app.state.generator.generate(
            retriever=app.state.retriever,
            question=req.question,
            model=req.model,
            top_k=req.top_k,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
```

The `setup_observability(app)` call (in `observability.py`) wires the LangFuse SDK into FastAPI. Every request becomes a trace; every LLM call becomes a Generation.

---

## 4. The Generator — Instructor + LiteLLM with LangFuse (`app/generator.py`)

```python
import instructor
from litellm import acompletion
from openai import AsyncOpenAI
from langfuse.openai import openai as langfuse_openai
from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel
import os

from .retriever import Retriever


class Citation(BaseModel):
    source: str
    text: str
    score: float


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float


class Generator:
    def __init__(self, default_model: str, fallback_models: list[str]):
        self.default_model = default_model
        self.fallback_models = fallback_models
        
        # Use the LangFuse-wrapped OpenAI client for auto-tracing
        self.openai_client = langfuse_openai.AsyncOpenAI()
        self.instructor_client = instructor.from_openai(self.openai_client)
    
    @observe(name="rag_query")
    async def generate(
        self,
        retriever: Retriever,
        question: str,
        model: str,
        top_k: int,
    ) -> dict:
        # 1. Retrieve context (traced as Span)
        context = await retriever.retrieve(question, top_k=top_k)
        
        # 2. Generate answer (traced as Generation via wrapper)
        response: RAGResponse = await self.instructor_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            response_model=RAGResponse,
            max_retries=3,
        )
        
        # 3. Add trace metadata
        langfuse_context.update_current_observation(
            metadata={"question_length": len(question), "context_length": len(context)},
        )
        langfuse_context.score_current_observation(
            name="answer_confidence",
            value=response.confidence,
        )
        
        # 4. Return with trace_id for client-side feedback
        trace_id = langfuse_context.get_current_trace_id()
        return {
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "confidence": response.confidence,
            "trace_id": trace_id,
        }
    
    def _get_system_prompt(self) -> str:
        """Fetch the active prompt from the registry."""
        from langfuse import Langfuse
        langfuse = Langfuse()
        prompt = langfuse.get_prompt("rag_system_prompt")
        return prompt.compile()
```

The Generator uses:
- `langfuse.openai.openai.AsyncOpenAI()` — drop-in replacement for OpenAI client with auto-tracing
- `instructor.from_openai(...)` — wraps for Pydantic validation per [[06 - Large Language Models/22 - Instructor and Structured Generation]]
- `@observe(name="rag_query")` — root span for the entire RAG flow
- `langfuse_context.update_current_observation(...)` — enrich with metadata
- `langfuse_context.score_current_observation(...)` — attach self-reported confidence

The trace lineage in LangFuse shows: `rag_query → retrieve_documents (Span) → generate_answer (Generation) → LLM call → Pydantic validation`.

---

## 5. Online Evaluator Configuration (`langfuse/seed-evaluators.py`)

```python
"""Seed online evaluators in LangFuse. Run once after first deployment."""
from langfuse import Langfuse
import os

langfuse = Langfuse()

# LLM-as-judge evaluator: 10% sampling
langfuse.create_evaluator(
    name="answer_quality",
    type="llm_as_judge",
    model="gpt-4o-mini",
    sampling_rate=0.1,
    prompt_template="""You are evaluating the quality of an answer to a user question.

Question: {{question}}
Model answer: {{output}}

Rate on a 0-1 scale:
- 1.0: perfect, correct, concise, helpful
- 0.8: correct but slightly verbose or missing a minor detail
- 0.6: mostly correct but missing an important detail
- 0.4: partially correct, factual errors
- 0.2: irrelevant or off-topic
- 0.0: completely wrong or hallucinated

Output JSON only: {"score": <number>, "reasoning": "<one sentence>"}""",
    config={"temperature": 0.3, "max_tokens": 200},
    metadata={"owner": "team-rag", "version": "1.0.0"},
)

# Code evaluator: answer length (always run, very cheap)
langfuse.create_evaluator(
    name="length_check",
    type="code",
    code="""
def evaluate(trace):
    output = trace.output.get('answer', '') if isinstance(trace.output, dict) else str(trace.output)
    return {
        'score': min(len(output) / 500, 1.0),
        'reasoning': f'Length: {len(output)} chars (target: 500).'
    }
""",
    sampling_rate=1.0,
)
```

Run once after first deployment:
```bash
python langfuse/seed-evaluators.py
```

The evaluators are now configured and will run automatically on every new trace matching the sampling rate.

---

## 6. Drift Detection Cron (`langfuse/alerting/drift-detector.py`)

```python
"""Hourly drift detector. Run via cron or scheduled task."""
import os
from datetime import datetime, timedelta
from langfuse import Langfuse
import requests

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

langfuse = Langfuse()


def fetch_scores(name: str, hours: int) -> list[float]:
    """Fetch all scores for a metric in the last N hours."""
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    scores = langfuse.fetch_scores(
        name=name,
        from_timestamp=start.isoformat(),
        to_timestamp=now.isoformat(),
    )
    return [s.value for s in scores.data if s.value is not None]


def send_slack(message: str):
    if not SLACK_WEBHOOK_URL:
        print(f"[no slack configured] {message}")
        return
    requests.post(SLACK_WEBHOOK_URL, json={"text": message})


def check_drift(metric_name: str, threshold_warn: float = 0.05, threshold_critical: float = 0.10):
    """Detect drift on a single metric. Returns (current, baseline, delta)."""
    current_scores = fetch_scores(metric_name, hours=1)
    baseline_scores = fetch_scores(metric_name, hours=24*7)  # weekly baseline
    
    if not current_scores or not baseline_scores:
        return None
    
    current = sum(current_scores) / len(current_scores)
    baseline = sum(baseline_scores) / len(baseline_scores)
    delta = (current - baseline) / baseline if baseline > 0 else 0
    
    if delta < -threshold_critical:
        send_slack(f"🚨 CRITICAL drift on {metric_name}: {current:.2f} vs baseline {baseline:.2f} ({delta:+.1%})")
    elif delta < -threshold_warn:
        send_slack(f"⚠️  WARN drift on {metric_name}: {current:.2f} vs baseline {baseline:.2f} ({delta:+.1%})")
    else:
        print(f"✅ {metric_name}: {current:.2f} vs baseline {baseline:.2f} ({delta:+.1%})")
    
    return (current, baseline, delta)


if __name__ == "__main__":
    for metric in ["answer_quality", "answer_relevance", "user_feedback"]:
        check_drift(metric)
```

```bash
# Run hourly via cron
0 * * * * cd /app && python langfuse/alerting/drift-detector.py >> /var/log/drift.log 2>&1
```

The drift detector fetches scores from LangFuse, computes the moving average, and alerts Slack on degradation. In production, replace Slack with PagerDuty for critical metrics.

---

## 7. Offline Evaluation Runner (`eval/run_offline_eval.py`)

```python
"""Run offline evaluation against a dataset. Called from CI on every prompt change."""
import os
import sys
import json
from pathlib import Path
from langfuse import Langfuse
from langfuse.evaluation import evaluate
import openai

# Add parent dir for imports
sys.path.append(str(Path(__file__).parent.parent))
from app.generator import Generator

langfuse = Langfuse()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def target(item):
    """Run the RAG pipeline on a dataset item."""
    # Use the prompt from the registry (currently in 'production' label)
    prompt = langfuse.get_prompt("rag_system_prompt")
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt.compile()},
            {"role": "user", "content": item.input["question"]},
        ],
    )
    return response.choices[0].message.content


def judge(item, output):
    """LLM-as-judge evaluator."""
    prompt = f"""Question: {item.input['question']}
Expected: {item.expected_output.get('answer', 'N/A')}
Model: {output}

Rate 0-1 on correctness, relevance, conciseness. Output JSON only."""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)
    return [
        {"name": "correctness", "value": scores["correctness"]},
        {"name": "relevance", "value": scores["relevance"]},
        {"name": "conciseness", "value": scores["conciseness"]},
    ]


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "ci_baseline"
    
    experiment = evaluate(
        dataset_name="rag_eval_v1",
        target=target,
        evaluators=[judge],
        experiment_name=experiment_name,
        metadata={"git_commit": os.getenv("GITHUB_SHA", "local")},
    )
    
    print(f"Experiment: {experiment_name}")
    print(f"  Correctness: {experiment.scores['correctness'].mean():.2f}")
    print(f"  Relevance:   {experiment.scores['relevance'].mean():.2f}")
    print(f"  Conciseness: {experiment.scores['conciseness'].mean():.2f}")
    
    # Fail CI if quality regressed
    if experiment.scores["correctness"].mean() < 0.7:
        sys.exit(1)
```

CI integration in `.github/workflows/eval.yml`:

```yaml
name: Eval on Prompt Change
on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'langfuse/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run offline evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
        run: |
          python eval/run_offline_eval.py "pr_${GITHUB_PR_NUMBER}"
```

Every prompt PR triggers an evaluation against the held-out dataset. If the new prompt regresses on `correctness < 0.7`, the CI fails and the PR cannot merge.

---

## 8. Bootstrap Dataset from Production Traces

```python
"""Run weekly to bootstrap a new evaluation dataset from high-quality production traces."""
from langfuse import Langfuse

langfuse = Langfuse()


def bootstrap_dataset(threshold: float = 0.85, limit: int = 500):
    """Pull highly-rated production traces and create a new dataset."""
    traces = langfuse.fetch_traces(
        tags=["production"],
        from_timestamp=(datetime.utcnow() - timedelta(days=7)).isoformat(),
        limit=limit * 3,  # over-fetch; we filter
        filters=[{"name": "user_feedback", "operator": ">=", "value": threshold}],
    )
    
    dataset_name = f"weekly_curated_{datetime.utcnow().strftime('%Y_%m_%d')}"
    dataset = langfuse.create_dataset(
        name=dataset_name,
        description=f"Auto-curated from production traces with user_feedback >= {threshold}",
    )
    
    for trace in traces.data[:limit]:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=trace.input,
            expected_output=trace.output,
            metadata={"trace_id": trace.id},
        )
    
    print(f"Created dataset '{dataset_name}' with {len(traces.data[:limit])} items")


if __name__ == "__main__":
    bootstrap_dataset()
```

Run weekly:
```bash
0 0 * * 0 cd /app && python langfuse/seed-datasets.py
```

The dataset grows with the production app — it represents what users actually ask, not synthetic test cases.

---

## 9. The Operational Workflow

The full lifecycle:

```
Day 0 (Initial Setup):
1. docker compose up
2. python langfuse/seed-prompts.py     # upload v1 of rag_system_prompt
3. python langfuse/seed-evaluators.py  # configure answer_quality, length_check
4. python langfuse/seed-datasets.py    # create rag_eval_v1 with 50 hand-curated items

Weekly:
5. python langfuse/seed-datasets.py    # bootstrap weekly_curated_YYYY_MM_DD
6. python eval/run_offline_eval.py "weekly_v1_$(date +%Y%m%d)"

Per PR:
7. Update prompt in prompts/rag_system_prompt_v2.txt
8. python langfuse/seed-prompts.py     # upload v2 to staging label
9. CI: python eval/run_offline_eval.py "pr_${PR}"
10. compare v1 vs v2 in LangFuse UI; promote v2 if significant

Hourly (cron):
11. python langfuse/alerting/drift-detector.py  # check drift on metrics

On Alert:
12. Investigate in LangFuse UI
13. If regression: rollback via langfuse.update_prompt_labels
14. If persistent: deeper investigation (data quality, model drift)
```

This loop is the **production observability substrate**. Every decision — promote a prompt, rollback a prompt, fix a bug, ship a new feature — is data-backed.

---

## 10. Production Deployment Checklist

Before shipping to production:

- [ ] Self-hosted stack on Kubernetes (not Docker Compose)
- [ ] Postgres + ClickHouse backups automated (daily snapshots, 30-day retention)
- [ ] MinIO lifecycle policies (move old blobs to cold storage after 90 days)
- [ ] SSO enabled via OIDC (Google, GitHub, or generic)
- [ ] SCIM provisioning (Enterprise tier)
- [ ] API keys rotated quarterly
- [ ] Drift detector runs every 5 minutes (not hourly)
- [ ] Slack/PagerDuty integration tested with synthetic alert
- [ ] RBAC: read-only role for stakeholders; admin role for engineers
- [ ] Audit logs exported to SIEM (Splunk, Datadog)
- [ ] GDPR compliance: PII redaction on trace inputs (mask function)
- [ ] HIPAA compliance: BAA with LangFuse Cloud OR fully self-hosted
- [ ] Disaster recovery: backup LangFuse stack to a different region

---

## 🎯 Key Takeaways

- Self-hosted LangFuse on Docker Compose: LangFuse Web + Worker + Postgres + ClickHouse + Redis + MinIO.
- FastAPI RAG service uses `langfuse.openai` SDK wrapper for auto-tracing and `instructor.from_openai` for structured outputs.
- Multi-provider routing via LiteLLM with cost attribution per tenant via `metadata={"trace_user_id": ...}`.
- Online evaluators: LLM-as-judge on 10% sample + code evaluator on 100% (length check).
- User feedback (thumbs up/down) attaches via `langfuse.score()`; high-rated traces bootstrap new evaluation datasets.
- Drift detector: hourly cron computes 24h moving average; alerts Slack on -5% / -10% / -20% drops.
- Offline evaluation runs in CI on every prompt PR; comparison via `compare_experiments()`.
- Operational loop: seed prompts/datasets/evaluators → bootstrap weekly → eval per PR → drift per hour → promote on significance.

## References

- LangFuse Self-hosting — [langfuse.com/docs/deployment/self-host](https://langfuse.com/docs/deployment/self-host)
- LangFuse Helm chart — [github.com/langfuse/langfuse-helm](https://github.com/langfuse/langfuse-helm)
- LangFuse Evaluations — [langfuse.com/docs/evaluation](https://langfuse.com/docs/evaluation)
- LangFuse Prompt Management — [langfuse.com/docs/prompts](https://langfuse.com/docs/prompts)
- Docker Compose reference — [github.com/langfuse/langfuse/blob/main/docker-compose.yml](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml)
- [[02 - Docker Profesional|Docker Profesional]] — multi-service stack patterns
- [[03 - Advanced Python/06 - Pydantic Deep Dive|Pydantic Deep Dive]] — schemas for RAGResponse, Citation
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider transport
- [[06 - Large Language Models/20 - RAG Evaluation Deep Dive|RAG Evaluation Deep Dive]] — LLM-as-judge rigor
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured output for RAGResponse
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — agent observability
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — drift detection on input features
- [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers|OpenTelemetry for AI Engineers]] — protocol foundation
- [[09 - MLOps y Produccion/35 - LangSmith Deep Dive|LangSmith Deep Dive]] — SaaS counterpart
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/01 - LangFuse Fundamentals - Architecture and Core Primitives|Note 01 — Fundamentals]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/02 - LLM SDK Auto-Instrumentation|Note 02 — SDK Auto-Instrumentation]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/03 - Datasets, Evaluations and Prompt Management|Note 03 — Datasets, Evaluations and Prompt Management]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/04 - Online Evaluators and Production Patterns|Note 04 — Online Evaluators]]
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — Kubernetes production deployment
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — service patterns
- [[16 - Harness Engineering/05 - File Architecture|File Architecture]] — project structure