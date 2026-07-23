# 🏭 Production Patterns — Sampling, Costs, and PII

LangSmith at production scale emits **thousands of runs per second** with **tokens, prompts, and completions flowing through**. Without discipline, three things break: (1) cost explodes as you store every prompt and completion, (2) latency from LangSmith's auto-tracing affects production response times, and (3) PII leaks into trace storage, creating GDPR/HIPAA violations. This note teaches the production patterns that keep LangSmith observability **affordable, fast, and safe** at scale.

Three production disciplines: **sampling** (don't keep every trace; keep the ones that matter), **cost attribution** (know what each service costs), and **PII redaction** (strip sensitive data before it reaches LangSmith storage). By the end you can operate LangSmith at production scale without bankrupting the team or violating compliance.

## 🎯 Learning Objectives

- Apply **deterministic sampling** to control storage costs.
- Configure **trace-level filters** to drop noisy traces.
- Implement **PII redaction** before trace export.
- Set up **cost attribution** via metadata and tags.
- Configure **production SLAs** for trace completeness.
- Avoid the four most common production LangSmith pitfalls.

## 1. The Scale Problem

A typical LLM service emits:

| Service | Runs/req | QPS | Runs/sec |
|---------|----------|-----|----------|
| Chat API | 8 | 100 | 800 |
| Embedding | 2 | 50 | 100 |
| LangGraph agent | 12 | 20 | 240 |
| **Total** | | | **1,140** |

At 1,140 runs/sec = **98M runs/day = 3B runs/month**. At LangSmith's pricing ($0.50/1K runs + storage), that's **$1,500/month in ingest + storage**. **Doubling** with retries and longer context.

The answer: **sample**. Don't keep every run; keep the ones that matter.

## 2. Sampling Strategies

### Project-Level Sampling

LangSmith has two project types:
- **Free**: keeps 5,000 traces/month, then **stops accepting**.
- **Paid**: unlimited traces, pay per trace.

For paid projects, **per-trace sampling** keeps the trace count manageable:

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"  # auto-trace
os.environ["LANGSMITH_SAMPLING_RATE"] = "0.10"  # 10% sampling
```

Or programmatically via the SDK:

```python
from langsmith import Client

client = Client()

# Per-project sampling
client.update_project_sampling_rate(
    project_name="production-chatbot",
    sampling_rate=0.10,
)
```

### Per-User Sampling

```python
from langsmith import traceable

@traceable
def my_step(query: str, user_id: str):
    # Sample premium users more, free users less
    sampling_rate = 1.0 if user_id.startswith("premium-") else 0.05
    # LangSmith respects per-run sampling based on trace metadata
    return process(query)
```

### Sampling by Trace Type

```python
@traceable(metadata={"trace_type": "critical"})
def critical_step(query: str):
    # Critical traces: 100% sampled
    return process(query)

@traceable(metadata={"trace_type": "routine"})
def routine_step(query: str):
    # Routine traces: 1% sampled
    return process(query)
```

LangSmith's sampler respects metadata filters — set up rules in the dashboard.

## 3. Trace Filtering: Drop the Noise

Some traces are noise — health checks, polling, cache hits. Drop them:

```python
from langsmith import traceable

# Health check — don't trace
@traceable(run_type="chain", tags=["health-check"])  # tags it for filtering
def health_check():
    return {"status": "ok"}

# Configure project to drop health-check traces
client.update_project_filters(
    project_name="production-chatbot",
    drop_filters=['in(tags, ["health-check", "polling", "cache-hit"])'],
)
```

**Common noise to drop**:
- Health checks and liveness probes.
- Cache-hit traces (no LLM call happened).
- Polling jobs (every 60 seconds).
- Background batch jobs (not user-facing).

## 4. PII Redaction

Three layers of PII defense:

### Layer 1: SDK-Level Filtering

```python
from langsmith import traceable

@traceable
def process(query: str, user_id: str):
    # Trace only non-PII fields
    return llm.invoke(query)

# LangSmith's `inputs` and `outputs` are auto-captured.
# `user_id` is in inputs — but if it contains PII, it leaks.

@traceable(
    metadata={"user_id_hash": hash(user_id)},  # hash instead of raw
    tags=["rag"],
)
def process_safe(query: str, user_id: str):
    return llm.invoke(query)
```

### Layer 2: Manual Filtering with `@traceable` Decorator

```python
from langsmith import traceable

def redact_pii(text: str) -> str:
    """Strip emails, SSNs, credit cards."""
    import re
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    text = re.sub(r'\d{3}-?\d{2}-?\d{4}', '[SSN]', text)
    text = re.sub(r'\d{4}-?\d{4}-?\d{4}-?\d{4}', '[CC]', text)
    return text


@traceable
def process(query: str, user_id: str):
    # Redact before tracing
    safe_query = redact_pii(query)
    return llm.invoke(safe_query)
```

### Layer 3: Project-Level PII Filter

```python
client.update_project_filters(
    project_name="production-chatbot",
    redact_filters=[
        # Drop traces where inputs contain emails
        'contains(inputs, "@")',
        # Mask specific fields
        'mask(inputs.email, "[EMAIL]")',
        'mask(outputs.user_credit_card, "[CC]")',
    ],
)
```

**Apply all three layers** — defense in depth. Even with SDK-level filtering, the project-level filter catches misconfigurations.

## 5. Cost Attribution

```python
@traceable(
    metadata={
        "service": "chat-api",
        "cost.center": "ai-platform",
        "cost.team": "search",
        "user_id_hash": hash(user_id),  # no PII
        "user_tier": user_tier,  # free, premium, enterprise
    },
    tags=["rag", "gpt-4o-mini"],
)
def rag_query(query: str, user_id: str, user_tier: str):
    response = openai_client.chat.completions.create(...)
    # Token counts auto-captured
    return {"answer": response.choices[0].message.content}
```

LangSmith's dashboard groups cost by tag and metadata:
- **Total cost per service**: sum of token costs grouped by `service`.
- **Cost per user tier**: grouped by `user_tier`.
- **Cost per model**: grouped by tag `gpt-4o-mini`.

### Cost Calculation

```python
@traceable(metadata={"model": "gpt-4o-mini"})
def call_llm(prompt: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # Calculate cost and attach to trace
    cost_usd = (
        response.usage.prompt_tokens * 0.00015 / 1000 +
        response.usage.completion_tokens * 0.0006 / 1000
    )

    # Set as run metadata
    from langsmith import get_current_run_tree
    run = get_current_run_tree()
    run.add_metadata({"cost.usd": cost_usd})

    return response.choices[0].message.content
```

## 6. Production SLAs

| SLA | Target | How to measure |
|-----|--------|----------------|
| **Trace completeness** | 100% of in-process traces exported | LangSmith dashboard: total runs |
| **Sampling rate** | 5-10% | LangSmith project settings |
| **Storage cost per run** | <$0.001 | Token cost attributes |
| **PII leakage** | 0 events/month | Audit on project filters |
| **Eval alert latency** | <1 hour | Alert Slack channel |

### Alert Rules

Configure alerts in the LangSmith dashboard:

```python
# Alert: faithfulness drop
client.create_alert(
    name="faithfulness-drop",
    project_name="production-chatbot",
    metric="feedback_score",
    metric_key="faithfulness",
    condition="rolling_avg_3d < 0.85",
    notification="slack://#prod-alerts",
)
```

## 7. ❌/✅ Antipatterns

### ❌ 100% sampling at full token retention

```python
# ⚠️ Every prompt + completion stored
os.environ["LANGSMITH_TRACING"] = "true"
# No sampling — all 100% kept
# Inputs/outputs auto-captured, including PII
```

### ✅ Sampled with PII redaction

```python
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_SAMPLING_RATE"] = "0.10"
# Apply PII filters at project level
```

### ❌ No drop filters for noise

```python
# ⚠️ Health checks flood LangSmith
@app.get("/health")
def health():
    return {"status": "ok"}  # traced by default
```

### ✅ Drop filters for known noise

```python
# ✅ Filter out health-check traces
@traceable(tags=["health-check"])
def health():
    return {"status": "ok"}
client.update_project_filters(drop_filters=['in(tags, ["health-check"])'])
```

### ❌ PII in run inputs

```python
# ⚠️ User email persisted to LangSmith
@traceable
def process(user):
    return llm.invoke(f"Email {user.email} ...")
```

### ✅ Hash PII before tracing

```python
@traceable(metadata={"user_id_hash": hash(user.email)})
def process(user):
    return llm.invoke(f"User {user.id} ...")
```

### ❌ No cost tracking

```python
# ⚠️ Don't know which service costs the most
response = llm.invoke(...)  # cost invisible
```

### ✅ Per-run cost attributes

```python
run.add_metadata({"cost.usd": cost_usd, "cost.model": "gpt-4o-mini"})
```

## 8. Production Reality

**Caso real — Production RAG Project:** Started with 100% sampling and no PII filter. Cost: $1,800/month + 1 incident where user emails were exposed. **Fixed:** (1) 10% sampling with deterministic selection, (2) project-level PII filters (drop traces with emails in inputs), (3) per-team cost attribution. **New cost: $200/month, 0 PII incidents.**

**Caso real — Multi-Agent Research System:** Per-agent sampling rates (research 20%, audit 5%, synthesis 10%). Total $400/month for 3 agents. Each agent has its own PII filter and alert rules.

## 📦 Compression Code

```python
# 📦 Compression: Production LangSmith in 20 lines

import os

# 1. Sampling
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_SAMPLING_RATE"] = "0.10"

# 2. PII filters (project-level)
from langsmith import Client
client = Client()
client.update_project_filters(
    project_name="production-chatbot",
    redact_filters=[
        'mask(inputs.email, "[EMAIL]")',
        'mask(outputs.credit_card, "[CC]")',
    ],
)

# 3. Cost attribution per run
from langsmith import get_current_run_tree
def call_llm(prompt):
    response = openai_client.chat.completions.create(...)
    cost = response.usage.prompt_tokens * 0.00015 / 1000
    get_current_run_tree().add_metadata({"cost.usd": cost})
    return response
```

## 🎯 Key Takeaways

1. **Sample 5-10%** — full 100% is $1,500+/month; sampling cuts cost 10×.
2. **Deterministic sampling** by `run_id` hash — same trace consistently evaluated.
3. **Drop noise** — health checks, cache hits, polling.
4. **PII redaction at 3 layers** — SDK decorator + project filters + manual scrub.
5. **Cost attribution per run** — `cost.usd`, `cost.model`, `cost.team`.
6. **Production SLAs + alerts** — faithfulness drop, sampling drift, PII leak.
7. **LangSmith + OTel** — use LangSmith for LLM-specific features, OTel for general traces.

## References

- [[00 - Welcome to LangSmith|Welcome]] — course map.
- [[01 - LangSmith Core|Core primitives]] — runs, traces.
- [[02 - Auto-Instrumentation|Auto-Instrumentation]] — what gets traced.
- [[04 - Online Evaluators|Online Evals]] — sampling for evals.
- [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/06 - Production Patterns - Sampling Costs and PII Redaction.md|OTel production patterns]] — the standard alternative.
- LangSmith sampling: https://docs.smith.langchain.com/observability/how_to_guides/sample_traces