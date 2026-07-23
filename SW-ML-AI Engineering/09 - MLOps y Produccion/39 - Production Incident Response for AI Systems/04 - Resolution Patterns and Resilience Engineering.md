# 🎯 04 - Resolution Patterns and Resilience Engineering for AI Systems

> **The patterns that prevent incidents from becoming outages. Circuit breakers, graceful degradation, rate limiting, dead-letter queues, shadow traffic. Build these before the first incident, not during.**

## 🎯 Learning Objectives
- Implement circuit breakers for LLM calls with auto-recovery
- Build graceful degradation with cached responses and fallback models
- Configure rate limiting per tenant with token bucket algorithms
- Wire dead-letter queues for failed requests with retry policies
- Use shadow traffic for safe rollback validation
- Build auto-mitigation: the system resolves common incidents without waking anyone
- Distinguish resilience patterns for steady state vs incident state

## Introduction

The best incident response is **the incident that doesn't happen**. Resilience patterns — circuit breakers, graceful degradation, rate limiting — are designed into the system so that common failure modes (provider outage, runaway agent, cost explosion) are auto-contained before they cascade into customer-facing outages.

This note covers the six resilience patterns that every production AI service should have:

1. **Circuit breakers** — auto-fail when a provider is unhealthy
2. **Graceful degradation** — fall back to cached or simpler responses
3. **Rate limiting** — per-tenant token bucket with backpressure
4. **Dead-letter queues** — async retry for transient failures
5. **Shadow traffic** — validate rollbacks before activating
6. **Auto-mitigation** — auto-resolve common incidents

Each pattern is presented with:
- **What it solves** — the failure mode
- **How it works** — the implementation
- **When to apply** — the trigger conditions
- **Anti-patterns** — what to avoid

By Note 05, you'll have the capstone that combines all six patterns into a single resilient service.

![Resilience patterns](https://example.com/resilience-patterns.png)

---

## 1. Circuit Breakers for LLM Calls

### 1.1 The problem

Without a circuit breaker, an LLM provider outage causes:
- Every request takes 30+ seconds
- Connection pool exhausts (max connections reached)
- Queue piles up
- All users see degraded service
- Recovery is slow even when the provider returns

### 1.2 The pattern

The circuit breaker has three states:

```
   CLOSED (normal)
        ↓ error rate > threshold
   OPEN (fail fast, return cached)
        ↓ after timeout
   HALF-OPEN (try one request)
        ↓ success → CLOSED
        ↓ failure → OPEN
```

```python
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for LLM calls."""
    
    failure_threshold: int = 5  # consecutive failures to open
    recovery_timeout: float = 30.0  # seconds to wait before half-open
    success_threshold: int = 2  # consecutive successes to close
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    
    def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Call the function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # reset on success
    
    def _on_failure(self):
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


class CircuitOpenError(Exception):
    pass
```

### 1.3 LLM-specific circuit breaker

```python
# Per-provider circuit breakers
provider_breakers = {
    "openai": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "anthropic": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "together": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
}


async def call_with_breaker(provider: str, request):
    """Call an LLM with circuit breaker protection."""
    try:
        return await provider_breakers[provider].call(
            litellm.acompletion, model=f"{provider}/...", messages=request.messages
        )
    except CircuitOpenError:
        # Fail fast; fall back to cached response or alternative provider
        return await fallback_response(request)
```

The circuit breaker **fails fast** (returns in <1ms) instead of waiting 30 seconds for the provider. The fallback path then takes over.

---

## 2. Graceful Degradation with Cached Responses

### 2.1 The pattern

When the primary service fails, return the best available alternative:

```python
async def graceful_response(query: str, tenant_id: str) -> dict:
    """Return the best available response with graceful degradation."""
    
    # 1. Try primary LLM (best quality)
    try:
        response = await primary_llm(query)
        return {"answer": response, "source": "primary", "quality": "high"}
    except (CircuitOpenError, ProviderOutage) as e:
        logger.warning(f"Primary LLM failed: {e}")
    
    # 2. Try fallback LLM (slightly lower quality)
    try:
        response = await fallback_llm(query)
        return {"answer": response, "source": "fallback", "quality": "good"}
    except Exception as e:
        logger.warning(f"Fallback LLM failed: {e}")
    
    # 3. Try semantic cache (previous similar query)
    cached = semantic_cache.get(query, tenant_id)
    if cached:
        return {"answer": cached, "source": "cache", "quality": "medium"}
    
    # 4. Return graceful degradation message
    return {
        "answer": "I'm temporarily unable to answer your question. Please try again in a few minutes.",
        "source": "degraded",
        "quality": "low",
    }
```

### 2.2 Quality tiers

The pattern exposes **quality tiers** to the user:
- **High**: primary LLM
- **Good**: fallback LLM
- **Medium**: semantic cache
- **Low**: generic apology

This is **honest degradation** — the user knows when they're getting a degraded response. They can decide to retry later.

---

## 3. Rate Limiting with Token Buckets

### 3.1 The problem

Without rate limits, one runaway user can exhaust the budget. With flat rate limits, you over-throttle normal users.

### 3.2 Token bucket algorithm

The standard approach: each tenant has a bucket of tokens that refills at a fixed rate:

```python
import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Per-tenant token bucket for rate limiting."""
    
    capacity: int  # max tokens
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    
    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.time()
    
    def try_consume(self, n: int) -> bool:
        """Try to consume n tokens; returns True if allowed."""
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


# Per-tenant buckets
tenant_buckets: dict[str, TokenBucket] = {}


def get_bucket(tenant_id: str) -> TokenBucket:
    if tenant_id not in tenant_buckets:
        tenant_buckets[tenant_id] = TokenBucket(
            capacity=1000,  # burst capacity
            refill_rate=1000 / 60,  # 1000 tokens per minute sustained
        )
    return tenant_buckets[tenant_id]


async def rate_limited_call(tenant_id: str, func, *args, **kwargs):
    """Call func only if rate limit allows."""
    bucket = get_bucket(tenant_id)
    if not bucket.try_consume(1):
        raise RateLimitedError(f"Tenant {tenant_id} rate limited")
    return await func(*args, **kwargs)


class RateLimitedError(Exception):
    pass
```

### 3.3 Tiered limits

Different tenants should have different rate limits:

```python
TIER_LIMITS = {
    "free": {"capacity": 100, "refill_rate": 100/3600},  # 100/hour
    "pro": {"capacity": 1000, "refill_rate": 1000/60},   # 1000/minute
    "enterprise": {"capacity": 10000, "refill_rate": 10000/60},  # 10K/minute
}


def get_bucket_for_tenant(tenant_id: str, tier: str) -> TokenBucket:
    config = TIER_LIMITS[tier]
    return TokenBucket(capacity=config["capacity"], refill_rate=config["refill_rate"])
```

This pattern is **cost-aware rate limiting**: higher tiers pay more, get higher limits.

---

## 4. Dead-Letter Queues for Failed Requests

### 4.1 The problem

Some requests fail transiently (network blip, provider throttling). Without a retry mechanism, the user sees an error. With synchronous retries, latency spikes. With synchronous retries without limits, you DDoS the failing provider.

### 4.2 The pattern: DLQ with async retry

```python
import json
import redis
from datetime import datetime, timedelta


class DeadLetterQueue:
    """DLQ for failed requests with exponential backoff retry."""
    
    def __init__(self, redis_client: redis.Redis, max_retries: int = 5):
        self.redis = redis_client
        self.max_retries = max_retries
    
    def enqueue(self, request: dict, error: str):
        """Add a failed request to the DLQ."""
        self.redis.lpush("dlq:requests", json.dumps({
            "request": request,
            "error": error,
            "enqueued_at": datetime.utcnow().isoformat(),
            "retries": 0,
        }))
    
    def process(self, func):
        """Worker that processes the DLQ with exponential backoff."""
        while True:
            item_json = self.redis.brpop("dlq:requests", timeout=5)
            if item_json is None:
                continue
            
            item = json.loads(item_json[1])
            retry_count = item["retries"]
            
            if retry_count >= self.max_retries:
                # Max retries exceeded; alert team
                self._alert_team(item)
                continue
            
            try:
                func(item["request"])
                # Success; item removed
            except Exception as e:
                # Retry with exponential backoff
                backoff = 2 ** retry_count  # 1s, 2s, 4s, 8s, 16s
                item["retries"] += 1
                item["last_error"] = str(e)
                
                # Re-enqueue with delay
                asyncio.get_event_loop().call_later(
                    backoff,
                    self.redis.lpush,
                    "dlq:requests",
                    json.dumps(item),
                )
```

### 4.3 Failure modes the DLQ handles

- **Provider throttling** (429): retry in 30s
- **Network blip**: retry in 1s
- **Transient 500**: retry in 2s
- **Schema validation failure**: don't retry; alert team
- **Prompt injection attempt**: don't retry; alert security

The DLQ separates **retriable failures** from **non-retriable failures**. The user gets a quick response; the system retries in the background.

---

## 5. Shadow Traffic for Safe Rollback

### 5.1 The problem

You want to roll back a model version. But how do you know the rollback won't introduce a new problem?

### 5.2 The pattern: shadow traffic

Send a copy of production traffic to the candidate version; compare outputs without serving them:

```python
async def shadow_traffic_test(production_call, candidate_call, traffic_fraction: float = 0.1):
    """Test a candidate version against production traffic without serving it."""
    import random
    
    # Production call (always served)
    production_response = await production_call()
    
    # Shadow call (10% of traffic, not served)
    if random.random() < traffic_fraction:
        try:
            candidate_response = await candidate_call()
            
            # Compare
            similarity = compute_similarity(production_response, candidate_response)
            
            # Log to LangFuse for analysis
            langfuse.score(
                name="shadow_similarity",
                value=similarity,
                metadata={"candidate": "v2", "production": "v1"},
            )
        except Exception as e:
            logger.warning(f"Shadow traffic failed: {e}")
    
    return production_response


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts."""
    # Use embedding cosine similarity or token overlap
    from sklearn.metrics.pairwise import cosine_similarity
    embeddings = embed([text_a, text_b])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

After 1 hour of shadow traffic, you have 36,000 comparisons. If the candidate is similar to production (>0.9 cosine similarity on average), it's safe to promote.

---

## 6. Auto-Mitigation

### 6.1 The pattern

Some incidents are **so common** that the system should auto-resolve them:

| Incident | Auto-mitigation |
|----------|-----------------|
| Cost explosion (per-tenant > 10x) | Auto rate-limit that tenant to 1 RPS |
| Provider latency spike (p99 > 5s) | Auto-failover via LiteLLM Router |
| Provider 5xx rate > 5% | Auto-open circuit breaker for 30s |
| Prompt injection pattern detected | Auto-block user for 1 hour |
| Quality score < 0.5 for > 30 min | Auto-rollback prompt to previous version |
| Error rate > 50% | Auto-disable feature flag (kill switch) |

### 6.2 The implementation

```python
# auto-mitigation.py
from prometheus_client import Gauge

auto_mitigations_triggered = Gauge(
    "auto_mitigations_triggered_total",
    "Number of auto-mitigations triggered",
    ["type"],
)


async def auto_mitigate_on_cost_spike(tenant_id: str, current_cost: float, baseline: float):
    """Auto-rate-limit a tenant when cost spikes >10x baseline."""
    if current_cost > 10 * baseline:
        # Apply 1 RPS rate limit
        apply_tenant_rate_limit(tenant_id, max_rps=1)
        auto_mitigations_triggered.labels(type="cost_spike").inc()
        
        # Page the team
        await page_team(
            severity="sev2",
            message=f"Auto-rate-limited {tenant_id}: cost spike to ${current_cost:.2f}",
        )


async def auto_mitigate_on_prompt_injection(tenant_id: str, attack_pattern: str):
    """Auto-block a tenant that attempts prompt injection."""
    block_tenant_for_duration(tenant_id, hours=1)
    auto_mitigations_triggered.labels(type="prompt_injection").inc()
    
    await page_team(
        severity="sev2",
        message=f"Blocked {tenant_id}: prompt injection attempt",
    )
```

### 6.3 The principle: "auto-mitigate, then page"

Modern incident response philosophy: **the system should resolve simple incidents without waking anyone**. The on-call engineer is paged to investigate and learn, not to do the basic mitigation.

---

## 7. Building Resilience Patterns Together

The patterns compose:

```
User request
    ↓
Rate limiter (per tenant)
    ↓
Cache check (semantic cache)
    ↓ if miss
LLM call
    ↓
Circuit breaker (per provider)
    ↓ if open
Graceful degradation (fallback provider → cache → generic)
    ↓
DLQ retry (for transient failures)
    ↓
Shadow traffic (parallel call for safety)
    ↓
Response
```

Each pattern protects against a different failure mode:

- **Rate limiter**: runaway agent, DDoS
- **Cache**: provider outage, transient failures
- **Circuit breaker**: provider outage
- **Graceful degradation**: total system failure
- **DLQ**: transient provider errors
- **Shadow traffic**: rollback safety

The composition is **defense in depth** — no single pattern fails the system.

---

## 8. Case Studies

### 8.1 Case real 1: Circuit breaker saved a 4-hour outage

A provider (Anthropic) had a 4-hour outage during peak traffic. Without circuit breakers, every request would have waited 30+ seconds, exhausting the connection pool. With circuit breakers, requests failed fast (in <1ms) and fell back to GPT-4o via LiteLLM. **Total customer impact: 30 seconds of degraded service.**

### 8.2 Case real 2: Rate limiter saved $50K

A SaaS customer had a runaway CI/CD pipeline calling the LLM service 100x more than expected. Without rate limits, the cost would have been $50K in 24 hours. The per-tenant rate limiter caught the spike in 15 minutes and limited the customer to 1 RPS. **Total cost: $200.**

### 8.3 Case real 3: Auto-mitigation saved a 3 AM page

A quality drift alert fired at 3 AM. The auto-mitigation system detected the drift, identified the recent prompt deploy as the cause, rolled back to the previous version, and restored quality in 8 minutes. **The on-call engineer was never paged.**

---

## 9. Antipatterns

### 9.1 Antipattern 1: Synchronous retries on failure

```python
# ❌ Synchronous retry waits; user sees slow error
for attempt in range(5):
    try:
        return await call_llm()
    except Exception:
        time.sleep(2 ** attempt)
return {"error": "LLM unavailable"}

# ✅ Use DLQ; user gets fast response, system retries in background
try:
    return await call_llm()
except RetriableError:
    dlq.enqueue(request)
    return {"status": "queued, retrying"}
```

### 9.2 Antipattern 2: No circuit breaker

```python
# ❌ Without circuit breaker: every request waits 30s during outage
async def chat(request):
    return await litellm.acompletion(...)  # waits 30s when provider is down

# ✅ With circuit breaker: requests fail fast
async def chat(request):
    try:
        return await breaker.call(litellm.acompletion, ...)
    except CircuitOpenError:
        return await fallback(request)
```

### 9.3 Antipattern 3: Global rate limits

```python
# ❌ One global rate limit; normal users get throttled when one user has a problem
RATE_LIMIT = 1000  # all users combined

# ✅ Per-tenant rate limits with tiers
buckets = {tenant_id: TokenBucket(capacity=tier_capacity[tier], refill_rate=tier_refill[tier])}
```

### 9.4 Antipattern 4: No graceful degradation

```python
# ❌ Returns 500 when LLM fails
async def chat(request):
    try:
        return await litellm.acompletion(...)
    except Exception:
        raise HTTPException(500)  # user sees error

# ✅ Returns the best available response
async def chat(request):
    try:
        return await litellm.acompletion(...)
    except Exception:
        return await graceful_response(request)  # cache → fallback → apology
```

### 9.5 Antipattern 5: No shadow traffic before rollback

```python
# ❌ Promote new model to 100% traffic immediately
client = OpenAI(model="gpt-4o-turbo")  # new model

# ✅ Shadow test first; promote only if similar to production
await shadow_traffic_test(
    production_call=lambda: openai_v1(),
    candidate_call=lambda: openai_v2(),
    traffic_fraction=0.1,
)
# After 1 hour, check similarity scores; promote if > 0.9
```

---

## 🎯 Key Takeaways

- Circuit breakers fail fast when providers are unhealthy; recover via half-open state.
- Graceful degradation returns the best available response: primary → fallback → cache → apology.
- Token bucket rate limiting per tenant with tiered limits (free/pro/enterprise).
- Dead-letter queues retry transient failures async; user gets fast response.
- Shadow traffic validates rollbacks before serving; ensures safety.
- Auto-mitigation handles common incidents without waking the on-call.
- Compose the patterns as defense in depth: rate limit → cache → circuit breaker → DLQ → shadow.
- Avoid sync retries, no circuit breakers, global rate limits, no graceful degradation, no shadow traffic.

## References

- Release It! by Michael Nygard — Circuit breakers, stability patterns
- Google SRE Book — Cascading Failures — [sre.google/sre-book/addressing-cascading-failures](https://sre.google/sre-book/addressing-cascading-failures/)
- AWS Builders Library — [aws.amazon.com/builders-library](https://aws.amazon.com/builders-library/)
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/01 - AI Incident Taxonomy - What Can Break|Note 01 — Taxonomy]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/02 - Detection - Alerts, Metrics, and Anomaly Detection|Note 02 — Detection]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/03 - Triage - Diagnosing AI Incidents|Note 03 — Triage]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/05 - Capstone - Production Incident Response Simulation|Note 05 — Capstone]]
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider failover
- [[13 - Go Engineering/03 - Microservices with Go|Microservices]] — circuit breaker patterns