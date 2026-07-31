# 🎯 02 - Cost Visibility — Per-Tenant Attribution, Chargeback, and Showback

> **Who pays for what? The question every CFO asks. The answer requires per-tenant cost attribution via LangFuse metadata + Prometheus + aggregation. Chargeback reports and dashboards.**

## 🎯 Learning Objectives
- Attribute every LLM call to a tenant + use case + model
- Implement per-tenant cost dashboards in Grafana
- Build chargeback reports that map costs to customers
- Set up cost anomaly detection before bills arrive
- Distinguish chargeback (actual billing) from showback (visibility only)
- Apply to portfolio projects: LLM Gateway + Eval Suite + StayBot

## Introduction

The hardest question in production LLM is: **"How much did tenant X cost us last month?"** Without per-tenant attribution, you can't answer. With it, you can:

- **Charge customers** for their actual LLM usage (usage-based pricing)
- **Allocate costs** to internal teams (engineering, product, marketing)
- **Detect anomalies** when a tenant's cost spikes (often a sign of abuse or bug)
- **Forecast** future costs based on historical per-tenant patterns

This note covers the architecture: LangFuse metadata → Prometheus counters → Grafana dashboards → per-tenant reports.

![Cost attribution pipeline](https://example.com/cost-attribution.png)

---

## 1. The Attribution Problem

Why is it hard?

1. **Tokens are abstract** — a token isn't a "thing" you bill for
2. **Multiple calls per request** — agent systems make 5+ LLM calls per user request
3. **Async pipelines** — the cost happens in a worker, the user is in the API
4. **Multiple models** — different providers, different costs per token
5. **Shared caches** — cache hits reduce cost but are tied to specific tenants

The solution: tag every LLM call with metadata that identifies the tenant, feature, and use case. Then aggregate.

---

## 2. The Metadata Schema

Tag every LLM call with:

```python
from langfuse import observe, langfuse_context


@observe()
async def chat_with_tenant(
    tenant_id: str,
    feature: str,  # "chat", "summarize", "embed", "rag"
    messages: list[dict],
) -> dict:
    """Chat completion with full cost attribution."""
    
    # Tag every trace with tenant_id + feature + cost metadata
    langfuse_context.update_current_observation(
        metadata={
            "tenant_id": tenant_id,
            "feature": feature,
            "user_id": context.get("user_id"),
            "session_id": context.get("session_id"),
        },
    )
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    
    # Compute and tag cost
    usage = response.usage
    cost = calculate_cost_openai(usage, "gpt-4o-mini")
    
    langfuse_context.score_current_observation(
        name="cost_usd",
        value=cost,
        metadata={
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "model": "gpt-4o-mini",
        },
    )
    
    return {
        "content": response.choices[0].message.content,
        "cost_usd": cost,
    }
```

Every LLM call now has:
- `tenant_id`: who initiated the call
- `feature`: which product feature
- `user_id`: which user (for billing to customer)
- `cost_usd`: actual cost of this call
- `input_tokens`, `output_tokens`: token breakdown

---

## 3. Prometheus Metrics for Cost

Export per-tenant cost to Prometheus:

```python
from prometheus_client import Counter, Histogram
from langfuse import observe, langfuse_context


# Prometheus metrics
tenant_cost_total = Counter(
    "llm_cost_usd_per_tenant_total",
    "Total LLM cost per tenant",
    ["tenant_id", "feature", "model"],
)
tenant_tokens_total = Counter(
    "llm_tokens_per_tenant_total",
    "Total tokens per tenant",
    ["tenant_id", "feature", "model", "token_type"],  # token_type: input/output
)
tenant_request_count = Counter(
    "llm_requests_per_tenant_total",
    "Total LLM requests per tenant",
    ["tenant_id", "feature", "model"],
)


@observe()
async def chat_with_tenant(tenant_id: str, feature: str, messages: list[dict]):
    """Chat with full cost attribution."""
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    
    usage = response.usage
    cost = calculate_cost_openai(usage, "gpt-4o-mini")
    
    # Update Prometheus metrics
    tenant_cost_total.labels(
        tenant_id=tenant_id,
        feature=feature,
        model="gpt-4o-mini",
    ).inc(cost)
    
    tenant_tokens_total.labels(
        tenant_id=tenant_id,
        feature=feature,
        model="gpt-4o-mini",
        token_type="input",
    ).inc(usage.prompt_tokens)
    
    tenant_tokens_total.labels(
        tenant_id=tenant_id,
        feature=feature,
        model="gpt-4o-mini",
        token_type="output",
    ).inc(usage.completion_tokens)
    
    tenant_request_count.labels(
        tenant_id=tenant_id,
        feature=feature,
        model="gpt-4o-mini",
    ).inc()
    
    # Also export to LangFuse for trace correlation
    langfuse_context.score_current_observation(
        name="cost_usd",
        value=cost,
    )
    
    return response
```

---

## 4. Grafana Dashboards

The per-tenant dashboard:

```json
{
    "title": "LLM Cost Attribution by Tenant",
    "panels": [
        {
            "title": "Top 10 tenants by cost (last 24h)",
            "type": "table",
            "targets": [
                {
                    "expr": "topk(10, sum by (tenant_id) (rate(llm_cost_usd_per_tenant_total[24h])))",
                    "format": "table",
                    "instant": true,
                }
            ],
        },
        {
            "title": "Cost per tenant per feature (last 24h)",
            "type": "barchart",
            "targets": [
                {
                    "expr": "sum by (tenant_id, feature) (rate(llm_cost_usd_per_tenant_total[24h]))",
                    "legend_format": "{{tenant_id}} - {{feature}}",
                }
            ],
        },
        {
            "title": "Cost per tenant per model (last 7d, stacked)",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "sum by (model) (rate(llm_cost_usd_per_tenant_total[7d]))",
                    "legend_format": "{{model}}",
                }
            ],
        },
        {
            "title": "Tokens per tenant (last 24h)",
            "type": "table",
            "targets": [
                {
                    "expr": "sum by (tenant_id, token_type) (rate(llm_tokens_per_tenant_total[24h]))",
                    "format": "table",
                }
            ],
        },
        {
            "title": "Cost anomaly: top 5 most expensive single requests (last 24h)",
            "type": "table",
            "targets": [
                {
                    "expr": "topk(5, max_over_time(llm_cost_usd_per_request[24h]))",
                    "format": "table",
                }
            ],
        },
    ],
}
```

---

## 5. Cost Anomaly Detection

Detect cost spikes before they hit the bill:

```python
import asyncio
from datetime import datetime, timedelta
from prometheus_client import Counter
from river.drift import ADWIN


# Prometheus alert
cost_anomaly_alerts = Counter(
    "llm_cost_anomaly_alerts_total",
    "Cost anomaly alerts",
    ["tenant_id", "severity"],
)


class CostAnomalyDetector:
    """Detect per-tenant cost anomalies in real time."""
    
    def __init__(self, z_threshold: float = 3.0):
        self.tenant_baselines: dict[str, dict] = {}
        self.z_threshold = z_threshold
    
    async def check_tenant(self, tenant_id: str, current_cost: float):
        """Check if tenant's costs are anomalous."""
        
        # Get baseline (last 7 days, excluding today)
        baseline = await self.get_tenant_baseline(tenant_id, days=7)
        
        if not baseline:
            return {"anomaly": False}
        
        z_score = (current_cost - baseline["mean"]) / baseline["std"]
        
        if z_score > self.z_threshold:
            severity = "high" if z_score > 5 else "medium"
            cost_anomaly_alerts.labels(
                tenant_id=tenant_id,
                severity=severity,
            ).inc()
            
            await self.alert_team(tenant_id, current_cost, baseline, z_score)
            
            return {
                "anomaly": True,
                "severity": severity,
                "z_score": z_score,
                "current": current_cost,
                "baseline": baseline["mean"],
            }
        
        return {"anomaly": False, "z_score": z_score}
    
    async def get_tenant_baseline(self, tenant_id: str, days: int) -> dict:
        """Get historical baseline for tenant."""
        import numpy as np
        
        # Pull last 7 days of cost from Prometheus
        costs = await query_prometheus(
            f'sum by (hour) (rate(llm_cost_usd_per_tenant_total{{tenant_id="{tenant_id}"}}[{days}d))'
        )
        
        if len(costs) < 10:
            return None  # not enough data
        
        return {
            "mean": np.mean(costs),
            "std": np.std(costs),
        }
    
    async def alert_team(self, tenant_id, current, baseline, z_score):
        """Send cost anomaly alert."""
        async with httpx.AsyncClient() as client:
            await client.post(
                os.getenv("SLACK_WEBHOOK_URL"),
                json={
                    "text": (
                        f"💰 Cost anomaly: tenant `{tenant_id}`\n"
                        f"Current: ${current:.2f}\n"
                        f"Baseline: ${baseline['mean']:.2f} ± ${baseline['std']:.2f}\n"
                        f"Z-score: {z_score:.1f}σ"
                    )
                }
            )
```

---

## 6. Chargeback vs Showback

| Concept | Definition | Use case |
|---------|-----------|----------|
| **Showback** | Visibility only; team sees their cost but isn't billed | Internal teams, R&D, pre-launch |
| **Chargeback** | Actual billing; cost is allocated to a budget | Customer-facing products, multi-tenant SaaS |

### 6.1 Showback report (internal)

```python
def generate_showback_report(period: str = "monthly") -> dict:
    """Generate per-team cost showback report."""
    
    teams = ["engineering", "marketing", "product", "customer-success"]
    report = {}
    
    for team in teams:
        metrics = query_prometheus(
            f'sum by (feature) (rate(llm_cost_usd_per_tenant_total{{team="{team}"}}[{period}d))'
        )
        report[team] = metrics
    
    return report
```

### 6.2 Chargeback report (customer-facing)

```python
def generate_chargeback_report(
    customer_id: str,
    period_start: datetime,
    period_end: datetime,
) -> dict:
    """Generate per-customer chargeback invoice."""
    
    # Pull per-feature, per-model usage from Prometheus
    usage = query_prometheus(
        f'sum by (feature, model) (rate(llm_cost_usd_per_tenant_total{{customer_id="{customer_id}"}}[{period_start}:{period_end}]))'
    )
    
    # Apply markup and minimum fees
    itemized = []
    for row in usage:
        cost = float(row["value"])
        # 30% markup, $0 minimum per feature
        billed_cost = max(cost * 1.30, 0.0)
        itemized.append({
            "feature": row["feature"],
            "model": row["model"],
            "raw_cost": cost,
            "billed_cost": billed_cost,
        })
    
    total_billed = sum(item["billed_cost"] for item in itemized)
    
    return {
        "customer_id": customer_id,
        "period": f"{period_start.date()} to {period_end.date()}",
        "itemized": itemized,
        "total_raw_cost": sum(item["raw_cost"] for item in itemized),
        "total_billed": total_billed,
        "currency": "USD",
    }
```

---

## 7. Real-World Case — SaaS Chargeback

A SaaS company with 100 customers, each on a usage-based pricing model:

| Item | Without FinOps | With FinOps |
|------|---------------|-------------|
| Cost visibility | Quarterly manual reports | Per-tenant dashboards real-time |
| Anomaly detection | CFO notices surprise invoice | Auto-alert at 5σ |
| Customer questions | "Why is my bill $X?" → days to investigate | Click dashboard, see exact breakdown |
| Internal cost tracking | Spreadsheet | Real-time per-feature |
| Cost optimization | Ad-hoc | Systematic model selection + caching |

The same SaaS with 100 customers can answer "what did tenant X cost us in Q3?" in seconds vs days.

---

## 8. The LLM Gateway + Chargeback Integration

Add per-tenant cost attribution to your LLM Gateway (covered in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|06/19]]):

```python
# In the gateway layer
@app.middleware("http")
async def track_tenant_cost(request: Request, call_next):
    """Track tenant cost on every request."""
    
    tenant_id = request.headers.get("X-Tenant-ID")
    
    # Process the request
    response = await call_next(request)
    
    # Extract cost from response headers (set by the LLM client)
    cost_usd = response.headers.get("X-LLM-Cost-USD")
    prompt_tokens = int(response.headers.get("X-LLM-Prompt-Tokens", 0))
    completion_tokens = int(response.headers.get("X-LLM-Completion-Tokens", 0))
    
    # Update Prometheus
    if tenant_id and cost_usd:
        tenant_cost_total.labels(
            tenant_id=tenant_id,
            feature=request.url.path,
            model=response.headers.get("X-LLM-Model", "unknown"),
        ).inc(float(cost_usd))
        
        # Check budget
        await check_tenant_budget(tenant_id, tenant_config)
    
    return response
```

This pattern works for any LLM gateway (LiteLLM, OpenLLMetry, custom).

---

## 9. Antipatterns

### 9.1 Antipattern 1: Cost visibility only at the CFO level

```python
# ❌ CFO gets a monthly bill; engineers see nothing
# No one knows costs until the invoice arrives

# ✅ Per-tenant dashboards visible to engineering, product, customer success
# Real-time visibility for the people who can act on it
```

### 9.2 Antipattern 2: Chargeback without cost optimization

```python
# ❌ Track costs but don't optimize
# Charge customers; don't reduce their cost; they churn

# ✅ Showback + optimization recommendations
# Dashboards show: "Tenant X uses GPT-4o for tasks that GPT-4o-mini could handle"
```

### 9.3 Antipattern 3: Per-tenant cost without per-feature cost

```python
# ❌ "Tenant X cost $1000 last month"
# But which feature? Which model? Which use case?

# ✅ Per-tenant + per-feature + per-model breakdown
# "Tenant X spent $500 on chat (GPT-4o), $300 on summarization (gpt-4o-mini), $200 on embeddings"
```

### 9.4 Antipattern 4: Cost alerts at the wrong threshold

```python
# ❌ Alert on every 1% cost increase
# -> Alert fatigue; team ignores

# ✅ Alert at 3σ (statistically meaningful) or 5x baseline
# -> Only meaningful anomalies trigger alerts
```

### 9.5 Antipattern 5: Cost attribution that decays

```python
# ❌ Tag every call with tenant_id, but Prometheus retention is 7 days
# Cost attribution for last month is impossible

# ✅ Long-term storage (S3, BigQuery) for cost history
# Prometheus for real-time; long-term storage for historical reporting
```

---

## 🎯 Key Takeaways

- Per-tenant cost attribution is mandatory for any multi-tenant LLM service.
- Tag every LLM call with `tenant_id`, `feature`, `model`, and `cost_usd`.
- Use LangFuse metadata + Prometheus counters for real-time attribution.
- Build per-tenant dashboards in Grafana; engineers + product + customer success see them.
- Cost anomaly detection with z-score > 3 prevents surprise bills.
- Distinguish showback (visibility) from chargeback (billing).
- Long-term cost storage (S3, BigQuery) for historical reporting.
- Avoid CFO-only visibility, chargeback without optimization, missing per-feature, wrong thresholds, decaying attribution.

## References

- LangFuse Metadata — [langfuse.com/docs/observability/features/metadata](https://langfuse.com/docs/observability/features/metadata)
- Prometheus Labels — [prometheus.io/docs/prometheus/latest/configuration/recording_rules](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)
- CloudZero LLM Cost — [cloudzero.com/blog/llm-cost](https://www.cloudzero.com/blog/llm-cost/)
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]]
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/01 - LLM Cost Fundamentals|Note 01 — Cost Fundamentals]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/03 - Cost Optimization Patterns|Note 03 — Cost Optimization]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/04 - Forecasting and Budget Management|Note 04 — Forecasting]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/05 - Capstone - FinOps Pipeline|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems|Incident Response]] — cost anomaly runbook