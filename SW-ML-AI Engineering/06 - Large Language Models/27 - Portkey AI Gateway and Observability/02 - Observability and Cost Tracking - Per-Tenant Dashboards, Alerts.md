# 🎯 02 - Portkey Observability and Cost Tracking — Per-Tenant Dashboards, Alerts

> **Built-in observability that replaces LangFuse for many teams. Per-tenant cost tracking, latency dashboards, quality scores, and budget alerts via Portkey's native UI.**

## 🎯 Learning Objectives
- Use Portkey's native observability dashboards
- Track per-tenant cost automatically via metadata
- Set up budget alerts via Slack/PagerDuty
- Capture and analyze quality scores (LLM-as-judge)
- Use Portkey's request logs for incident investigation
- Compare Portkey observability to LangFuse for migration decisions

## Introduction

Portkey's **native observability** is its killer feature for many teams. No separate LangFuse setup needed; every request is automatically logged with:

- Request + response payloads
- Latency, tokens, cost
- Provider + model used
- Custom metadata (tenant_id, feature, user_id)
- Quality scores (if attached)
- Errors and retries

The UI provides:
- Real-time request stream
- Per-tenant dashboards
- Cost analytics
- Latency histograms
- Error rates
- Quality score trends

This note covers the observability features and how to migrate from LangFuse (when you want to consolidate).

---

## 1. The Native Observability Stack

When you make a Portkey call:

```python
from portkey_ai import Portkey

client = Portkey(api_key="pk-***", Authorization="vk-***")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    config={
        "metadata": {
            "tenant_id": "alice",
            "feature": "chat",
            "user_id": "user-123",
            "session_id": "sess-456",
        },
    },
)
```

Portkey automatically captures:
- The full request (messages, model, params)
- The response (content, usage)
- Latency (total, time-to-first-token)
- Cost (per-request, per-model)
- Provider + model used
- Errors and stack traces
- Custom metadata

All visible in the dashboard immediately.

---

## 2. The Dashboard

The Portkey dashboard (`https://app.portkey.ai`) shows:

| Section | What it shows |
|---------|---------------|
| **Logs** | Real-time stream of every request |
| **Analytics** | Aggregated cost, latency, error rate by tenant/model |
| **Dashboards** | Custom charts (filter by metadata) |
| **Spend** | Per-tenant spend; budget alerts |
| **Users** | Per-user activity |
| **Feedback** | Quality scores (if submitted) |

### 2.1 Real-time Logs

Every request appears in the Logs tab within seconds:

```json
{
    "request_id": "req-abc-123",
    "timestamp": "2026-07-23T12:34:56Z",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "tenant_id": "alice",
    "feature": "chat",
    "user_id": "user-123",
    "request": {"messages": [{"role": "user", "content": "Hello"}]},
    "response": {"content": "Hi there!", "usage": {"input_tokens": 8, "output_tokens": 4}},
    "latency_ms": 234,
    "cost_usd": 0.000015,
    "status": "success"
}
```

Click on any request to see the full trace: messages, response, latency breakdown, retries.

---

## 3. Per-Tenant Cost Tracking

The metadata you set becomes a filter dimension in the dashboard.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    config={
        "metadata": {
            "tenant_id": "alice",
            "feature": "chat",
            "model_tier": "free",  # for usage-based pricing
        },
    },
)
```

In the dashboard:
- Filter by `metadata.tenant_id = "alice"` → see all of Alice's usage
- Group by `metadata.feature` → see cost per feature (chat vs. RAG vs. embeddings)
- Group by `metadata.model_tier` → see free-tier vs. pro-tier usage

A custom chart:
- X-axis: time
- Y-axis: cost
- Group by: `metadata.tenant_id`

Shows the cost per tenant over time.

---

## 4. Budget Alerts via Webhooks

Configure webhooks in the Portkey dashboard:
- **Budget threshold reached** (e.g., 80% of monthly budget)
- **Budget exceeded** (100% of monthly budget)
- **Anomalous spend** (e.g., 5× baseline)

When triggered, Portkey sends an HTTP POST to your webhook URL:

```python
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/portkey/webhook")
async def portkey_webhook(req: Request):
    payload = await req.json()
    
    event = payload.get("event")
    tenant_id = payload.get("data", {}).get("metadata", {}).get("tenant_id")
    
    if event == "budget.threshold_reached":
        # Notify tenant
        await slack_notify(
            f"⚠️ Tenant `{tenant_id}` reached 80% of monthly budget",
        )
    
    elif event == "budget.exceeded":
        # Critical alert
        await pagerduty_alert(
            severity="critical",
            summary=f"Tenant `{tenant_id}` exceeded monthly budget",
        )
        
        # Optionally block further requests
        await portkey_update_virtual_key(tenant_id, status="blocked")
    
    return {"status": "ok"}
```

The webhook delivers within 30 seconds of the threshold event.

---

## 5. Quality Scores (LLM-as-Judge)

Add a quality score to each trace:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain Kubernetes"}],
    config={
        "metadata": {"tenant_id": "alice"},
    },
)

# Submit quality score (async, doesn't block the response)
client.feedback.create(
    trace_id=response.id,
    key="accuracy",
    score=0.92,  # 0-1
    metadata={"evaluator": "gpt-4o-mini-judge"},
)
```

Or attach multiple scores:

```python
client.feedback.create(
    trace_id=response.id,
    key="accuracy",
    score=0.92,
)
client.feedback.create(
    trace_id=response.id,
    key="relevance",
    score=0.88,
)
client.feedback.create(
    trace_id=trace_id,
    key="user_rating",
    score=1,  # 1 for thumbs up, 0 for thumbs down
    metadata={"user_id": "user-123"},
)
```

The dashboard shows quality trends per model, per tenant.

---

## 6. Anomaly Detection

Portkey has built-in anomaly detection:

| Anomaly | Trigger |
|---------|---------|
| **Cost spike** | Per-tenant spend > 5× 7-day baseline |
| **Latency spike** | p95 latency > 3× baseline |
| **Error rate spike** | Error rate > 10% over 1 hour |
| **Model switch** | Routing config changed without code deploy |
| **Provider outage** | Provider 5xx rate > 50% |

Anomalies generate alerts in the dashboard and (optionally) webhooks.

---

## 7. LangFuse Migration Decision

**When to migrate from LangFuse to Portkey**:

| Use case | Recommendation |
|---------|---------------|
| LangChain-based with deep LangSmith integration | Stay with LangSmith |
| Need LangFuse dataset management | Stay with LangFuse |
| Want LLM gateway + observability in one | **Migrate to Portkey** |
| Need PII redaction out of the box | **Migrate to Portkey** |
| Need audit logs for compliance | **Migrate to Portkey** |
| Already happy with LangFuse + LiteLLM | Stay with LangFuse |

For most teams: **Portkey + LiteLLM migration is straightforward**. Portkey's OpenAI-compatible API works with any existing client.

---

## 8. The Production Observability Stack

For the **LLM Edge Gateway** portfolio project:

```python
# Before (LangFuse + LiteLLM)
from langfuse import observe

@observe()
async def chat(request):
    response = litellm.completion(...)
    return response

# After (Portkey)
from portkey_ai import Portkey

async def chat(request):
    response = portkey_client.chat.completions.create(
        ...,
        config={
            "metadata": {"tenant_id": request.tenant_id, "feature": "chat"},
        },
    )
    return response
```

Remove the LangFuse decorator. Portkey captures everything automatically.

---

## 9. Antipatterns

### 9.1 Antipattern 1: No metadata tags

```python
# ❌ No metadata; can't filter by tenant
response = client.chat.completions.create(model="gpt-4o-mini", messages=...)

# ✅ Always tag with metadata
config = {
    "metadata": {
        "tenant_id": "alice",
        "feature": "chat",
        "user_id": "user-123",
    },
}
```

### 9.2 Antipattern 2: Manually calculating costs

```python
# ❌ Manual cost calculation
input_tokens = response.usage.prompt_tokens
cost = input_tokens / 1_000_000 * 0.15  # error-prone

# ✅ Portkey tracks cost automatically
# View in dashboard: $X spent today, breakdown per model/tenant
```

### 9.3 Antipattern 3: No quality scores

```python
# ❌ No quality tracking
# Don't know if the model is drifting

# ✅ Attach LLM-as-judge scores
client.feedback.create(trace_id=..., key="accuracy", score=0.92)
```

### 9.4 Antipattern 4: Ignoring budget alerts

```python
# ❌ Disable budget alerts
# → surprise $50K bill

# ✅ Always enable budget alerts
# → early warning before overrun
```

### 9.5 Antipattern 5: Not aggregating by tenant

```python
# ❌ Total cost only
total_cost = sum(all_costs)

# ✅ Cost by tenant
cost_by_tenant = group_by(all_costs, "tenant_id")
# Then chargeback to the right customer
```

---

## 🎯 Key Takeaways

- Portkey's native observability replaces LangFuse for many teams.
- Metadata tags become filter dimensions; always tag with tenant_id + feature + user_id.
- Budget alerts via webhooks; integrate with Slack/PagerDuty.
- Quality scores attach LLM-as-judge results to traces.
- Anomaly detection catches cost spikes, latency spikes, error rates.
- Migration from LangFuse is straightforward (drop-in API).
- Avoid no metadata, manual cost calc, no quality scores, ignoring alerts, no aggregation.

## References

- Portkey Observability — [portkey.ai/docs/product/observability](https://portkey.ai/docs/product/observability)
- Portkey Feedback — [portkey.ai/docs/product/observability/feedback](https://portkey.ai/docs/product/observability/feedback)
- Portkey Webhooks — [portkey.ai/docs/product/webhooks](https://portkey.ai/docs/product/webhooks)
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/01 - Portkey Core - Gateway Fundamentals|Note 01 — Portkey Core]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — observability alternative
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/02 - Cost Visibility - Per-Tenant Attribution, Chargeback, and Showback|Cost Visibility]]