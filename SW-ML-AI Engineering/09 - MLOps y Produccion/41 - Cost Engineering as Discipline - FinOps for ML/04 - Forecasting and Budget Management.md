# 🎯 04 - Forecasting and Budget Management — Predict Costs, Alert Before Bills

> **Forecast the future. The discipline that prevents $50K surprise bills. Time series, budget alerts, approval workflows, and capacity planning.**

## 🎯 Learning Objectives
- Build time-series cost forecasts using Prophet and statistical methods
- Set up budget alerts that catch overruns before they hit the bill
- Implement approval workflows for high-cost operations
- Plan reserved capacity for predictable workloads
- Calibrate forecasts with confidence intervals
- Apply to portfolio projects: budget control + approval gates

## Introduction

The best cost optimization is the one that prevents waste. Three patterns:

1. **Forecasting**: predict next month's cost with statistical confidence
2. **Budget alerts**: page the team when burn rate exceeds plan
3. **Approval workflows**: require human sign-off for high-cost operations

Together, these prevent the surprise $50K bill that no one saw coming.

![Forecasting and budgeting](https://example.com/forecasting.png)

---

## 1. Cost Forecasting with Prophet

Use Facebook Prophet (or statsmodels SARIMA) for time-series forecasting:

```python
import pandas as pd
from prophet import Prophet
from prometheus_api_client import PrometheusConnect


def forecast_costs(days_ahead: int = 30) -> dict:
    """Forecast LLM costs for the next N days."""
    
    # Pull historical cost from Prometheus
    prom = PrometheusConnect(url="http://prometheus:9090")
    data = prom.get_metric_range_data(
        "sum(rate(llm_cost_usd_total[1d]))",
        start_time=(datetime.utcnow() - timedelta(days=180)).isoformat(),
    )
    
    df = pd.DataFrame(data)
    df.columns = ["ds", "y"]  # Prophet format
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Fit Prophet model
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.add_country_holidays(country_name="US")
    model.fit(df)
    
    # Forecast
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    
    # Return summary
    monthly_forecast = forecast.tail(days_ahead)["yhat"].sum()
    monthly_lower = forecast.tail(days_ahead)["yhat_lower"].sum()
    monthly_upper = forecast.tail(days_ahead)["yhat_upper"].sum()
    
    return {
        "forecast_period_days": days_ahead,
        "expected_cost": monthly_forecast,
        "lower_bound_95ci": monthly_lower,
        "upper_bound_95ci": monthly_upper,
    }
```

### 1.1 Per-tenant forecasting

For per-tenant cost forecasts:

```python
def forecast_tenant_costs(tenant_id: str, days_ahead: int = 30) -> dict:
    """Forecast costs for a specific tenant."""
    
    data = prom.get_metric_range_data(
        f'sum by (day) (rate(llm_cost_usd_per_tenant_total{{tenant_id="{tenant_id}"}}[1d]))',
        start_time=(datetime.utcnow() - timedelta(days=90)).isoformat(),
    )
    
    df = pd.DataFrame(data)
    df.columns = ["ds", "y"]
    
    model = Prophet(weekly_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    
    next_month = forecast.tail(days_ahead)["yhat"].sum()
    
    return {
        "tenant_id": tenant_id,
        "expected_monthly_cost": next_month,
        "trend": "increasing" if forecast["trend"].iloc[-1] > forecast["trend"].iloc[0] else "decreasing",
    }
```

---

## 2. Budget Alerts

Detect overruns before they hit the bill:

```yaml
# prometheus-alerts.yml
groups:
  - name: cost_alerts
    rules:
      # Warning: 80% of monthly budget consumed
      - alert: LLMMonthlyBudgetWarning
        expr: |
          sum(increase(llm_cost_usd_total[30d])) > 0.8 * 25000
        for: 5m
        labels:
          severity: warning
          team: ai-platform
        annotations:
          summary: "LLM budget 80% consumed"
          runbook: "https://wiki.company.com/runbooks/cost-budget"

      # Critical: 100% of monthly budget consumed
      - alert: LLMMonthlyBudgetExceeded
        expr: |
          sum(increase(llm_cost_usd_total[30d])) > 25000
        for: 1m
        labels:
          severity: critical
          team: ai-platform
        annotations:
          summary: "LLM monthly budget exceeded"
          runbook: "https://wiki.company.com/runbooks/cost-budget"
          action: "Block non-critical LLM calls; investigate"

      # Per-tenant: 5x baseline
      - alert: TenantCostSpike
        expr: |
          sum(increase(llm_cost_usd_per_tenant_total[1h])) > 5 * avg_over_time(sum(increase(llm_cost_usd_per_tenant_total[1h]))[7d])
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Tenant cost spike: 5x baseline"

      # Forecast: next month > budget
      - alert: LLMForecastOverrun
        expr: |
          predict_linear(llm_cost_usd_total[30d], 30*86400) > 30000
        for: 1h
        labels:
          severity: warning
```

---

## 3. Per-Tenant Budget Enforcement

```python
import asyncio
import redis
from datetime import datetime, timedelta


class TenantBudgetManager:
    """Enforce per-tenant monthly budgets."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_budget(self, tenant_id: str, additional_cost_usd: float) -> dict:
        """Check if a tenant is within budget; return decision."""
        
        # Get tenant config
        config = await self.get_tenant_config(tenant_id)
        monthly_budget = config["monthly_budget_usd"]
        
        # Get current spend (this month)
        current_spend = await self.get_current_spend(tenant_id)
        
        projected = current_spend + additional_cost_usd
        
        if projected > monthly_budget:
            # Over budget — block
            return {
                "allow": False,
                "reason": "over_budget",
                "current_spend": current_spend,
                "monthly_budget": monthly_budget,
                "projected": projected,
            }
        
        if projected > 0.9 * monthly_budget:
            # Approaching budget — warn but allow
            return {
                "allow": True,
                "reason": "approaching_budget",
                "warning": True,
                "current_spend": current_spend,
                "monthly_budget": monthly_budget,
                "projected": projected,
            }
        
        return {
            "allow": True,
            "current_spend": current_spend,
            "monthly_budget": monthly_budget,
            "projected": projected,
        }
    
    async def get_tenant_config(self, tenant_id: str) -> dict:
        """Get tenant budget configuration."""
        config = self.redis.hgetall(f"tenant:{tenant_id}:config")
        return {
            "monthly_budget_usd": float(config.get(b"monthly_budget_usd", 1000)),
            "tier": config.get(b"tier", b"standard").decode(),
        }
    
    async def get_current_spend(self, tenant_id: str) -> float:
        """Get this month's spend for tenant."""
        month_key = datetime.utcnow().strftime("%Y-%m")
        spend = self.redis.hget(f"tenant:{tenant_id}:spend", month_key)
        return float(spend or 0)
    
    async def record_cost(self, tenant_id: str, cost_usd: float):
        """Record cost to monthly spend."""
        month_key = datetime.utcnow().strftime("%Y-%m")
        self.redis.hincrbyfloat(f"tenant:{tenant_id}:spend", month_key, cost_usd)
        self.redis.expire(f"tenant:{tenant_id}:spend", 86400 * 90)  # 90-day retention
```

---

## 4. Approval Workflows for High-Cost Operations

For operations above a threshold, require human approval:

```python
class ApprovalWorkflow:
    """Approval workflow for high-cost LLM operations."""
    
    HIGH_COST_THRESHOLD_USD = 50.0
    
    def __init__(self):
        self.pending_approvals: dict[str, dict] = {}
    
    async def request_approval(self, operation: str, estimated_cost: float, requester: str) -> str:
        """Request approval for high-cost operation."""
        
        if estimated_cost < self.HIGH_COST_THRESHOLD_USD:
            # Auto-approve low-cost operations
            return "auto_approved"
        
        # Require human approval
        approval_id = f"approval-{uuid.uuid4().hex[:8]}"
        
        self.pending_approvals[approval_id] = {
            "operation": operation,
            "estimated_cost_usd": estimated_cost,
            "requester": requester,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # Send Slack notification
        await self.notify_approvers(approval_id, operation, estimated_cost)
        
        return approval_id
    
    async def notify_approvers(self, approval_id: str, operation: str, cost: float):
        """Notify approvers in Slack."""
        async with httpx.AsyncClient() as client:
            await client.post(
                os.getenv("SLACK_WEBHOOK_URL"),
                json={
                    "text": (
                        f"💰 Approval needed\n"
                        f"Operation: `{operation}`\n"
                        f"Estimated cost: ${cost:.2f}\n"
                        f"Approve: `/approve {approval_id}`\n"
                        f"Deny: `/deny {approval_id}`"
                    )
                }
            )
    
    async def approve(self, approval_id: str, approver: str) -> bool:
        """Approve a pending request."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id]["status"] = "approved"
            self.pending_approvals[approval_id]["approver"] = approver
            return True
        return False
```

In the LLM Gateway:

```python
approval_workflow = ApprovalWorkflow()


@app.middleware("http")
async def approval_check(request: Request, call_next):
    """Check approval for high-cost operations."""
    
    # Estimate cost (rough)
    estimated_cost = estimate_request_cost(request)
    
    if estimated_cost > approval_workflow.HIGH_COST_THRESHOLD_USD:
        approval_id = await approval_workflow.request_approval(
            operation=f"LLM call to {request.url.path}",
            estimated_cost=estimated_cost,
            requester=request.headers.get("X-User-ID", "unknown"),
        )
        
        if approval_id == "auto_approved":
            pass  # proceed
        elif not await wait_for_approval(approval_id, timeout=300):
            return JSONResponse({"error": "Approval timeout"}, status_code=408)
    
    return await call_next(request)
```

---

## 5. Cost-Aware Deployment Gates

Block deploys that increase cost beyond threshold:

```yaml
# .github/workflows/cost-gate.yml
on:
  pull_request:
    paths:
      - 'models/**'

jobs:
  cost-gate:
    runs-on: ubuntu-latest
    steps:
      - name: Estimate cost impact
        run: |
          # Estimate tokens for new model
          ESTIMATED_INCREASE=$(python -c "
          from app.cost_calc import estimate_cost_increase
          print(estimate_cost_increase('${{ github.event.pull_request.base.sha }}', '${{ github.sha }}'))
          ")
          echo "Cost increase: $ESTIMATED_INCREASE"
          
          # Check against threshold
          if (( $(echo "$ESTIMATED_INCREASE > 500" | bc -l) )); then
            echo "::error::Cost increase too high: \$$ESTIMATED_INCREASE/month"
            exit 1
          fi
```

---

## 6. Reserved Capacity Planning

For predictable workloads:

```python
def recommend_reserved_capacity(monthly_traffic_tokens: int, baseline_cost_per_million: float) -> dict:
    """Recommend reserved capacity vs on-demand."""
    
    # AWS Bedrock pricing (example)
    on_demand_cost = monthly_traffic_tokens / 1_000_000 * baseline_cost_per_million
    
    # 1-year commit discount
    one_year_discount = 0.30  # 30% off
    one_year_cost = on_demand_cost * (1 - one_year_discount)
    
    # 3-year commit (better discount but lock-in)
    three_year_discount = 0.50  # 50% off
    three_year_cost = on_demand_cost * (1 - three_year_discount)
    
    # Break-even
    one_year_savings = on_demand_cost - one_year_cost
    three_year_savings = on_demand_cost - three_year_cost
    
    return {
        "on_demand_monthly": on_demand_cost,
        "one_year_monthly": one_year_cost,
        "one_year_savings": one_year_savings,
        "three_year_monthly": three_year_cost,
        "three_year_savings": three_year_savings,
        "recommendation": "one_year" if on_demand_cost > 5000 else "on_demand_only",
    }
```

---

## 7. The Cost Dashboard (Executive View)

```python
def generate_executive_cost_report() -> dict:
    """Monthly cost report for the CFO."""
    
    current_month_cost = get_current_month_cost()
    forecast = forecast_costs(days_ahead=30)
    budget = get_monthly_budget()
    
    return {
        "month_to_date": current_month_cost,
        "forecast": forecast["expected_cost"],
        "budget": budget,
        "variance": forecast["expected_cost"] - budget,
        "variance_pct": (forecast["expected_cost"] - budget) / budget,
        "top_tenants": get_top_tenants_by_cost(limit=10),
        "recommendations": generate_optimization_recommendations(),
    }
```

```python
def generate_optimization_recommendations() -> list[str]:
    """Generate cost optimization recommendations."""
    
    recommendations = []
    
    # Check model tiering opportunity
    if get_gpt4o_usage_pct() > 0.30:
        recommendations.append(
            "30% of traffic uses GPT-4o. Consider tiering: 80% could be on GPT-4o-mini (40× cheaper)"
        )
    
    # Check cache hit rate
    cache_hit_rate = get_cache_hit_rate()
    if cache_hit_rate < 0.40:
        recommendations.append(
            f"Cache hit rate is {cache_hit_rate:.1%}. Implement semantic cache; target 50%+"
        )
    
    # Check provider mix
    expensive_provider_pct = get_provider_cost_pct("openai")
    if expensive_provider_pct > 0.60:
        recommendations.append(
            f"{expensive_provider_pct:.0%} of spend is on OpenAI. Consider Together AI / Fireworks for cost savings"
        )
    
    return recommendations
```

---

## 8. Antipatterns

### 8.1 Antipattern 1: No forecasting

```python
# ❌ No idea what next month costs
# CFO asks: "What's next month's bill?"
# Answer: "I don't know"

# ✅ Forecast + alerts
forecast = forecast_costs(days_ahead=30)
# Always know the answer
```

### 8.2 Antipattern 2: Alerts that fire too late

```python
# ❌ Alert when bill exceeds budget (already spent too much)
- alert: BillExceedsBudget
  expr: sum(increase(llm_cost_usd_total[30d])) > 30000

# ✅ Alert on projected overruns (predictive)
- alert: ProjectedOverrun
  expr: predict_linear(llm_cost_usd_total[30d], 30*86400) > 30000
```

### 8.3 Antipattern 3: Per-tenant budgets without enforcement

```python
# ❌ Budget set in config, never enforced
tenant.budget = 1000  # USD/month
# But nothing stops tenant from spending $10,000

# ✅ Enforce at the gateway
if projected > tenant.budget:
    raise HTTPException(429, "Tenant budget exceeded")
```

### 8.4 Antipattern 4: Approval workflow without timeout

```python
# ❌ Wait forever for approval
await wait_for_approval(approval_id)  # never times out

# ✅ Timeout after 5 minutes, default deny
try:
    await asyncio.wait_for(wait_for_approval(approval_id), timeout=300)
except asyncio.TimeoutError:
    return HTTPException(408, "Approval timeout")
```

### 8.5 Antipattern 5: Forecasting without confidence intervals

```python
# ❌ Single number, no uncertainty
forecast = "Next month: $5,000"

# ✅ Forecast with confidence intervals
forecast = {
    "expected": 5000,
    "lower_95ci": 4200,
    "upper_95ci": 5800,
    "confidence": "high",  # based on historical variance
}
```

---

## 🎯 Key Takeaways

- Forecast with Prophet + confidence intervals; never give single-point estimates.
- Budget alerts: 80% warning, 100% critical; forecast overruns detected early.
- Per-tenant budgets enforced at the gateway; not just config.
- Approval workflows for high-cost operations; default deny on timeout.
- Reserved capacity for predictable workloads (>10K tokens/month).
- Cost-aware deployment gates prevent cost regression in CI.
- Executive reports with recommendations; not just numbers.
- Avoid no forecasting, late alerts, unenforced budgets, no-timeout approval, single-point forecasts.

## References

- Prophet — [facebook.github.io/prophet](https://facebook.github.io/prophet/)
- AWS Bedrock Reserved Capacity — [aws.amazon.com/bedrock/pricing](https://aws.amazon.com/bedrock/pricing)
- Azure OpenAI Provisioned — [learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput)
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/01 - LLM Cost Fundamentals|Note 01 — Cost Fundamentals]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/02 - Cost Visibility|Note 02 — Cost Visibility]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/03 - Cost Optimization Patterns|Note 03 — Optimization]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/05 - Capstone - FinOps Pipeline|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems|Incident Response]] — cost alerts in runbooks