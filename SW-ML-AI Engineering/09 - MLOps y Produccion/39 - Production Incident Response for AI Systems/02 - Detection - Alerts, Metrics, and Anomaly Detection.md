# 🎯 02 - Detection — Alerts, Metrics, and Anomaly Detection for AI Systems

> **The first line of defense. Prometheus multi-window burn rate alerts, LangFuse quality scores, and anomaly detection on cost and latency. Built before the first incident.**

## 🎯 Learning Objectives
- Build the three core dashboards: latency, cost, quality
- Configure Prometheus multi-window burn rate alerts for AI services
- Set up LangFuse quality score sampling and drift detection
- Wire PagerDuty and Slack alerting with appropriate severity routing
- Implement anomaly detection on cost-per-tenant spikes
- Reduce alert fatigue with proper severity tiers and on-call rotation
- Distinguish leading indicators (early warnings) from lagging indicators (already-broken)

## Introduction

The best incident is the one you detect and resolve before any user notices. The worst incident is the one you discover via a customer complaint three days after it started. Detection quality — the time between incident start and human awareness — is the single most important operational metric for AI services.

Traditional monitoring (Prometheus + Grafana) was built for **health signals**: CPU, memory, latency, error rate. AI services need all of these PLUS **quality signals**: LLM-as-judge scores, RAGAS faithfulness, user feedback distributions. Without quality signals, hallucination cascades and quality drift go undetected for days or weeks.

This note covers the complete detection stack for an AI service:

1. **Latency dashboards** — Prometheus metrics for p50, p95, p99 latency
2. **Cost dashboards** — LangFuse + Prometheus for cost-per-tenant, cost-per-model
3. **Quality dashboards** — LangFuse + RAGAS for LLM-as-judge scores, faithfulness
4. **Anomaly detection** — alerts on quality drift, cost spikes, latency cliffs
5. **Multi-window burn rate alerts** — SRE best practice for catching issues early without alert fatigue
6. **PagerDuty / Slack routing** — severity-based escalation

The patterns here draw from [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] (covered previously) and [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers|OpenTelemetry]] (the span protocol layer). Mastering detection is the prerequisite for triage (Note 03) and resolution (Note 04).

![Detection dashboard mockup](https://example.com/dashboard.png)

---

## 1. The Three Core Dashboards

### 1.1 Latency Dashboard

The standard Prometheus metrics for LLM latency:

```yaml
# llm-latency-recording-rules.yml
groups:
  - name: llm_latency
    rules:
      - record: llm:request_duration_seconds:p50
        expr: histogram_quantile(0.50, sum by (le, model) (rate(llm_request_duration_seconds_bucket[5m])))
      - record: llm:request_duration_seconds:p95
        expr: histogram_quantile(0.95, sum by (le, model) (rate(llm_request_duration_seconds_bucket[5m])))
      - record: llm:request_duration_seconds:p99
        expr: histogram_quantile(0.99, sum by (le, model) (rate(llm_request_duration_seconds_bucket[5m])))
```

Grafana panel query:
```promql
llm:request_duration_seconds:p99{model="gpt-4o-mini"}
```

### 1.2 Cost Dashboard

Per-tenant cost tracking via LangFuse metadata + Prometheus:

```python
# In the LLM service code
from langfuse import observe, langfuse_context
import time

@observe()
async def chat_completion(tenant_id: str, messages: list[dict]) -> dict:
    langfuse_context.update_current_observation(metadata={"tenant_id": tenant_id})
    response = await openai_client.chat.completions.create(...)
    
    # Compute cost
    cost = calculate_cost(response.usage, response.model)
    langfuse_context.score_current_observation(name="cost_usd", value=cost)
    
    # Emit to Prometheus
    cost_counter.labels(tenant_id=tenant_id, model=response.model).inc(cost)
    return response
```

Prometheus queries:

```promql
# Total cost last 24 hours
sum(increase(llm_cost_usd_total[24h]))

# Cost by tenant
sum by (tenant_id) (increase(llm_cost_usd_total[24h]))

# Cost anomaly (> 5x baseline)
sum(increase(llm_cost_usd_total[1h])) > 5 * avg_over_time(sum(increase(llm_cost_usd_total[1h]))[7d])
```

### 1.3 Quality Dashboard

LangFuse quality score traces (covered in [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability]]):

```python
# Sampled LLM-as-judge scoring (10% sampling rate)
@observe(as_type="evaluator")
def evaluate_response(trace_id: str, response: str, expected: str) -> dict:
    judge_prompt = f"""Rate this response 0-1 on correctness.
    
    Expected: {expected}
    Actual: {response}
    
    Output JSON only: {{"score": <number>, "reasoning": "<one sentence>"}}
    """
    judge_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )
    scores = json.loads(judge_response.choices[0].message.content)
    return [{"name": "correctness", "value": scores["score"]}]
```

Grafana panel queries (LangFuse exposes Prometheus-compatible metrics):

```promql
# Mean quality score over 1 hour
avg_over_time(langfuse_quality_score_sum[1h]) / avg_over_time(langfuse_quality_score_count[1h])

# Quality score < 0.7
langfuse_quality_score < 0.7

# Quality drift (current vs baseline)
avg_over_time(langfuse_quality_score_sum[1h]) / avg_over_time(langfuse_quality_score_sum[1h] offset 7d) < 0.9
```

---

## 2. Multi-Window Burn Rate Alerts

The SRE best practice for catching issues early without alert fatigue. Two windows: a short window (5 min) and a long window (1 hour). Alert when **both** fire:

```yaml
# llm-alerts.yml
groups:
  - name: llm_sev1
    rules:
      # SEV-1: 14.4x error rate in 5 min AND 6x error rate in 1 hour
      - alert: LLMHighErrorRate
        expr: |
          (
            sum(rate(llm_requests_total{status="error"}[5m]))
            / sum(rate(llm_requests_total[5m]))
          ) > 0.10
          and
          (
            sum(rate(llm_requests_total{status="error"}[1h]))
            / sum(rate(llm_requests_total[1h]))
          ) > 0.05
        for: 2m
        labels:
          severity: sev1
          team: ai-platform
        annotations:
          summary: "LLM error rate spike"
          description: "Error rate is {{ $value | humanizePercentage }} over 5 min AND 1 hour windows"
          runbook: "https://wiki.company.com/runbooks/llm-error-rate"

  - name: llm_sev2
    rules:
      # SEV-2: cost anomaly (5x baseline in 1 hour)
      - alert: LLMCostAnomaly
        expr: |
          sum(increase(llm_cost_usd_total[1h])) > 5 * avg_over_time(sum(increase(llm_cost_usd_total[1h]))[7d])
          and
          sum(increase(llm_cost_usd_total[3h])) > 3 * avg_over_time(sum(increase(llm_cost_usd_total[3h]))[7d])
        for: 10m
        labels:
          severity: sev2
          team: ai-platform
        annotations:
          summary: "LLM cost anomaly"
          description: "Cost is 5x baseline over 1 hour"
          runbook: "https://wiki.company.com/runbooks/llm-cost-anomaly"

      # SEV-2: quality score drop (10% below baseline)
      - alert: LLMQualityDrop
        expr: |
          avg(langfuse_quality_score{metric="correctness"}[1h]) < 0.9 * avg_over_time(langfuse_quality_score{metric="correctness"}[7d])
          and
          avg(langfuse_quality_score{metric="correctness"}[3h]) < 0.95 * avg_over_time(langfuse_quality_score{metric="correctness"}[7d])
        for: 30m
        labels:
          severity: sev2
          team: ai-platform
        annotations:
          summary: "LLM quality drop"
          description: "Quality score dropped below 90% of baseline"
          runbook: "https://wiki.company.com/runbooks/llm-quality-drop"

  - name: llm_sev3
    rules:
      # SEV-3: p99 latency 2x baseline for 10+ min
      - alert: LLMLatencySpike
        expr: |
          llm:request_duration_seconds:p99 > 2 * avg_over_time(llm:request_duration_seconds:p99[7d])
        for: 10m
        labels:
          severity: sev3
          team: ai-platform
        annotations:
          summary: "LLM p99 latency spike"
          description: "p99 latency is 2x baseline"
          runbook: "https://wiki.company.com/runbooks/llm-latency-spike"
```

The **two-window pattern** prevents two alert failure modes:
- **Single-window short**: only catches sudden spikes, not slow drifts
- **Single-window long**: only catches persistent issues, has slow response time

By requiring both windows to fire, the alert fires only on **real** sustained issues — no false positives.

---

## 3. Anomaly Detection on Cost

For cost anomalies that aren't captured by simple thresholds:

```python
# cost-anomaly-detector.py
import numpy as np
from typing import Annotated
from langfuse import Langfuse


class CostAnomalyDetector:
    """Detect cost anomalies using rolling statistics + z-score."""
    
    def __init__(self, langfuse: Langfuse, z_threshold: float = 3.0):
        self.langfuse = langfuse
        self.z_threshold = z_threshold
        self.baseline_window_days = 7
    
    def is_anomaly(self, tenant_id: str, current_cost: float) -> tuple[bool, float]:
        """Returns (is_anomaly, z_score)."""
        # Pull last 7 days of cost data for this tenant
        history = self.langfuse.fetch_scores(
            name="cost_usd",
            from_timestamp=datetime.utcnow() - timedelta(days=self.baseline_window_days),
            filter={"metadata.tenant_id": tenant_id},
        )
        historical_costs = [s.value for s in history.data]
        
        if len(historical_costs) < 100:
            return False, 0.0  # not enough data
        
        mean = np.mean(historical_costs)
        std = np.std(historical_costs)
        z_score = (current_cost - mean) / std if std > 0 else 0
        
        return z_score > self.z_threshold, z_score
```

This is the **statistical pattern**: z-score > 3 = 99.7% confidence the cost is anomalous.

---

## 4. LangFuse Quality Score Sampling and Drift Detection

```python
# Sample 10% of production traffic for LLM-as-judge evaluation
SAMPLE_RATE = 0.1

@observe()
async def evaluate_quality_async(trace_id: str, tenant_id: str) -> None:
    if random.random() > SAMPLE_RATE:
        return  # not sampled
    
    # Pull the trace and its response
    trace = langfuse.get_trace(trace_id)
    response = trace.output.get("answer", "")
    question = trace.input.get("question", "")
    
    # LLM-as-judge
    score = await llm_judge(question, response)
    
    # Attach to trace
    langfuse.score(
        trace_id=trace_id,
        name="correctness",
        value=score,
        metadata={"tenant_id": tenant_id, "judge_model": "gpt-4o-mini"},
    )
```

For drift detection (Note 04 covers rollback; this is just detection):

```python
# Daily drift detection job
async def detect_quality_drift() -> list[dict]:
    """Compare last 24h vs previous 7d for each metric."""
    drift_alerts = []
    
    metrics = ["correctness", "relevance", "faithfulness", "user_feedback"]
    for metric in metrics:
        # Last 24 hours
        recent = langfuse.fetch_scores(name=metric, hours_back=24)
        recent_avg = sum(s.value for s in recent) / len(recent) if recent else 0
        
        # Previous 7 days
        historical = langfuse.fetch_scores(name=metric, hours_back=168, exclude_last=24)
        historical_avg = sum(s.value for s in historical) / len(historical) if historical else 0
        
        # Detect drift
        if historical_avg > 0:
            drift_ratio = recent_avg / historical_avg
            if drift_ratio < 0.9:  # >10% drop
                drift_alerts.append({
                    "metric": metric,
                    "recent_avg": recent_avg,
                    "historical_avg": historical_avg,
                    "drift_ratio": drift_ratio,
                })
    
    return drift_alerts
```

This pattern catches **slow quality drift** that single-window alerts miss.

---

## 5. Alert Routing with PagerDuty

For severity-based escalation:

```yaml
# alertmanager.yml
route:
  receiver: 'ai-platform-default'
  group_by: ['alertname', 'severity']
  routes:
    - match:
        severity: sev1
      receiver: 'pagerduty-oncall-primary'
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 1h
    
    - match:
        severity: sev2
      receiver: 'pagerduty-oncall-secondary'
      group_wait: 5m
      group_interval: 30m
      repeat_interval: 6h
    
    - match:
        severity: sev3
      receiver: 'slack-ai-platform'
      group_wait: 30m
      group_interval: 6h
```

Severity tiers:
- **SEV-1**: pages primary on-call immediately
- **SEV-2**: pages secondary on-call after 5 min
- **SEV-3**: Slack notification only

---

## 6. Leading vs Lagging Indicators

The best alerting systems distinguish:

| Indicator type | Definition | Examples |
|----------------|-----------|----------|
| **Leading** | Early warnings; problem is forming | Quality drift trend, cost growth rate, p95 latency creep |
| **Lagging** | Problem already in motion | Error rate spike, timeout rate, current cost explosion |

Most teams only monitor lagging indicators. The best teams monitor **both**:

```yaml
# Lagging indicators (page immediately)
- alert: LLMCostExplosion
  expr: sum(increase(llm_cost_usd_total[10m])) > 10 * avg_over_time(sum(increase(llm_cost_usd_total[10m]))[7d])
  for: 2m
  severity: sev1

# Leading indicators (warn early)
- alert: LLMCostGrowingRapidly
  expr: |
    sum(increase(llm_cost_usd_total[1h]))
    > 2 * avg_over_time(sum(increase(llm_cost_usd_total[1h]))[7d])
    and
    sum(increase(llm_cost_usd_total[3h]))
    > 1.5 * avg_over_time(sum(increase(llm_cost_usd_total[3h]))[7d])
  for: 30m
  severity: sev3
```

Leading indicators **warn before the problem**; lagging indicators **page when the problem is happening**.

---

## 7. On-Call Rotation

For a 5-person team, a standard weekly rotation:

```python
# Standard on-call rotation
oncall_rotation = {
    "week_1": "alice",
    "week_2": "bob",
    "week_3": "carol",
    "week_4": "dave",
    "week_5": "eve",
}
# Each person gets 1 week on primary; 1 week on secondary (escalation)

# Schedule managed in PagerDuty:
# - Primary: pages immediately for SEV-1
# - Secondary: pages after 5 min if primary doesn't ack
# - Manager: pages after 15 min if neither acks
```

The rotation must be:
- **Time-bound**: max 1 week on call, then rotate
- **Documented**: who is on call is queryable
- **Compensated**: on-call pay is standard
- **Followed by recovery**: post-shift day off if escalated

---

## 8. Alert Fatigue Reduction

The biggest threat to a good alerting system is **alert fatigue** — too many alerts, the team ignores them.

Strategies:

| Strategy | Implementation |
|----------|----------------|
| **Tier severity** | SEV-1 pages; SEV-3 Slack only |
| **Multi-window** | Both windows must fire (avoids transient spikes) |
| **Group related alerts** | 10 cost anomalies for the same tenant → 1 alert |
| **Auto-resolve** | If condition clears in 5 min, auto-resolve the alert |
| **Suppress during deploys** | Don't alert on metrics during known deploys |
| **Runbook linkage** | Every alert links to a runbook |
| **Self-service remediation** | Auto-mitigate simple cases (Note 04 covers this) |

After 6 months of operation, **review your alert dashboard**:
- Which alerts fired without requiring action? (Remove or downgrade)
- Which incidents were missed by alerts? (Add new alerts)
- What was the median time-to-ack? (Adjust thresholds)

---

## 9. Case Studies

### 9.1 Case real 1: Hallucination cascade caught by quality alerts

A customer-support bot's quality score dropped from 92% to 78% in 4 hours. The LangFuse quality alert (configured in this note) paged the on-call engineer at 2 AM. Investigation revealed: a Prometheus deployment had changed the model's `temperature` parameter from 0.3 to 0.9, causing hallucinations. Rollback restored quality in 10 minutes. **Total cost of incident: 30 minutes of engineer time.**

### 9.2 Case real 2: Cost explosion detected via per-tenant alert

A SaaS customer's CI/CD pipeline had a bug that caused the AI service to be called 100x more than expected for one user. The per-tenant cost alert (configured here) paged within 15 minutes; the on-call engineer identified the runaway pattern, disabled the feature for that user, and the team fixed the CI/CD bug. **Total cost: $200 instead of $20,000.**

### 9.3 Case real 3: Latency cliff caught by multi-window alert

A new LLM provider (Fireworks) had a regional issue; latency spiked from 500ms to 8s. The multi-window burn rate alert paged within 5 minutes; the LiteLLM router fell over to the secondary provider automatically. **Total downtime: 30 seconds.**

---

## 10. Antipatterns

### 10.1 Antipattern 1: Page-happy alerts

```yaml
# ❌ Every minor latency fluctuation triggers a page
- alert: LLMAnyLatency
  expr: llm:request_duration_seconds:p99 > 1s
  for: 30s
  severity: sev1

# ✅ Tune thresholds to fire only on real issues
- alert: LLMHighLatency
  expr: |
    llm:request_duration_seconds:p99 > 2 * avg_over_time(llm:request_duration_seconds:p99[7d])
  for: 10m
  severity: sev2
```

### 10.2 Antipattern 2: Only monitoring latency

```python
# ❌ Misses hallucination cascades, quality drift, cost explosions
prometheus_alerts = ["llm_latency_p99 > 5s"]

# ✅ Comprehensive signal coverage
prometheus_alerts = [
    "llm_latency_p99 > 2 * baseline",
    "llm_cost_daily > budget",
    "llm_quality_score < 0.7",
    "llm_error_rate > 5%",
    "llm_pii_in_logs > 0",
]
```

### 10.3 Antipattern 3: No alert grouping

```yaml
# ❌ 10 alerts when 1 user has a cost spike
- alert: User1CostSpike
- alert: User2CostSpike
... (10 more)

# ✅ Group by tenant
- alert: TenantCostAnomaly
  expr: sum by (tenant_id) (increase(llm_cost_usd_total[1h])) > 10 * avg_over_time(sum by (tenant_id) (increase(llm_cost_usd_total[1h]))[7d])
```

### 10.4 Antipattern 4: Forgetting to set up the alert before launch

```python
# ❌ Production launched without alerts
# First incident: hours of confusion

# ✅ Always: alerts + runbook before production launch
# 1. Set up Prometheus alerts (Note 02)
# 2. Set up LangFuse quality scoring
# 3. Write runbooks for each alert
# 4. Test alerts in staging
# 5. On-call team trained
# 6. THEN launch
```

### 10.5 Antipattern 5: Ignoring the leading indicators

```yaml
# ❌ Only page when the system is down
- alert: LLMErrorRate
  expr: llm_error_rate > 10%
  severity: sev1

# ✅ Page early on drift trends; catch problems before they happen
- alert: LLMQualityDriftTrend
  expr: |
    avg_over_time(langfuse_quality_score[1h]) < 0.95 * avg_over_time(langfuse_quality_score[7d])
  severity: sev3
  # Catches the drift BEFORE quality collapses
```

---

## 🎯 Key Takeaways

- Three core dashboards: latency, cost, quality. Quality is the most important for AI-specific incidents.
- Multi-window burn rate alerts catch real issues without alert fatigue (5 min + 1 hour windows).
- LangFuse quality scores (LLM-as-judge) detect hallucination cascades and quality drift.
- Per-tenant cost attribution via metadata + Prometheus enables cost anomaly detection.
- Severity tiers: SEV-1 pages immediately; SEV-3 Slack only.
- Distinguish leading indicators (drift trends) from lagging indicators (current issues).
- On-call rotation: 1 week max, documented, compensated, follow-up recovery.
- Reduce alert fatigue via tier severity, grouping, auto-resolve, suppression during deploys.
- Avoid page-happy alerts, only-latency monitoring, no grouping, no pre-launch alerts, ignoring leading indicators.

## References

- Google SRE Book — Multi-Window Burn Rate Alerts — [sre.google/workbook/alerting-on-slos](https://sre.google/workbook/alerting-on-slos/)
- Prometheus Alerting Best Practices — [prometheus.io/docs/practices/alerting](https://prometheus.io/docs/practices/alerting/)
- PagerDuty Incident Response — [response.pagerduty.com](https://response.pagerduty.com/)
- [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers|OpenTelemetry for AI Engineers]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]]
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — drift detection
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/01 - AI Incident Taxonomy - What Can Break|Note 01 — Taxonomy]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/03 - Triage - Diagnosing AI Incidents|Note 03 — Triage]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering|Note 04 — Resolution Patterns]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/05 - Capstone - Production Incident Response Simulation|Note 05 — Capstone]]
- [[06 - Large Language Models/15 - LLM Security and Guardrails|LLM Security]] — prompt injection detection
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — failover patterns
- [[13 - Go Engineering/03 - Microservices with Go|Microservices]] — circuit breakers