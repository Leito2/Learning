# 🎯 03 - Fallbacks, Load Balancing, and Conditional Routing

> **Make your LLM service resilient. The reliability patterns that prevent provider outages from becoming customer-facing incidents.**

## 🎯 Learning Objectives
- Configure automatic fallbacks across providers
- Implement weighted load balancing for cost optimization
- Use conditional routing for different request types
- Set up retries with exponential backoff
- Handle rate limits gracefully
- Build a circuit breaker around the LLM gateway

## Introduction

**Reliability** is the difference between a service that goes down for 6 hours and one that survives a provider outage with a 30-second blip. The patterns:

| Pattern | Purpose |
|---------|---------|
| **Fallbacks** | If provider A fails, try B |
| **Load balancing** | Distribute load across providers |
| **Conditional routing** | Different models for different scenarios |
| **Retries with backoff** | Handle transient errors |
| **Circuit breaker** | Stop calling a failing provider |
| **Rate limit handling** | Don't blow through quotas |

This note covers each pattern with Portkey config.

---

## 1. The Fallback Pattern

When the primary provider fails (500, 503, rate limit), automatically try the next one.

```python
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",
    Authorization="vk-***",
    config={
        "provider": "openai",  # primary
        "fallbacks": [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
            {"provider": "together-ai", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        ],
        "retry": {
            "attempts": 3,
            "on_status_codes": [429, 500, 502, 503, 504],
        },
    },
)

response = client.chat.completions.create(
    model="gpt-4o",  # try primary
    messages=[{"role": "user", "content": "Hello"}],
)
```

What happens:
1. Try OpenAI GPT-4o
2. If 500/503/504 → retry once with backoff
3. If still failing → try OpenAI GPT-4o-mini
4. If failing → try Anthropic Claude 3.5 Haiku
5. If failing → try Together AI Llama 3 70B
6. If all fail → return error

The user gets a response even when a provider is down.

---

## 2. The Load Balancing Pattern

Distribute load across providers to:
- Distribute risk (one provider outage doesn't kill you)
- Optimize cost (use cheaper providers for non-critical)
- Reduce rate limit hits

```python
config = {
    "strategy": {
        "mode": "loadbalance",
        "on": [
            {"provider": "openai", "model": "gpt-4o-mini", "weight": 0.5},
            {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "weight": 0.3},
            {"provider": "together-ai", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "weight": 0.2},
        ],
    },
}
```

50% of traffic → OpenAI, 30% → Anthropic, 20% → Together AI.

---

## 3. The Conditional Routing Pattern

Different request types → different models.

```python
def get_config(user_request) -> dict:
    """Pick routing config based on request type."""
    
    if user_request.get("task_type") == "code":
        # Code tasks: Claude is best
        return {
            "provider": "anthropic",
            "override_params": {"model": "claude-3-5-sonnet-20241022"},
        }
    elif user_request.get("task_type") == "creative":
        # Creative: GPT-4o is best
        return {
            "provider": "openai",
            "override_params": {"model": "gpt-4o"},
        }
    elif user_request.get("priority") == "low":
        # Background tasks: cheap Llama
        return {
            "provider": "together-ai",
            "override_params": {"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
        }
    else:
        # Default
        return {
            "provider": "openai",
            "override_params": {"model": "gpt-4o-mini"},
        }


# Usage
config = get_config(user_request)
response = client.chat.completions.create(
    model="gpt-4o-mini",  # overridden by config
    messages=[{"role": "user", "content": user_request["prompt"]}],
    config=config,
)
```

The application decides routing based on business logic.

---

## 4. The Retry Pattern

```python
config = {
    "retry": {
        "attempts": 3,
        "on_status_codes": [429, 500, 502, 503, 504],
        "backoff": "exponential",  # 1s, 2s, 4s, ...
    },
}
```

The retry policy:
- Attempts: 3 (initial + 2 retries)
- On these status codes: 429 (rate limit), 500/502/503/504 (server errors)
- Backoff: exponential, starting at 1s

For different retry policies per error:

```python
config = {
    "retry": {
        "attempts": 3,
        "on_status_codes": [429, 500],
        "retry_conditions": [
            {
                "on_status_codes": [429],
                "max_retries": 5,  # rate limits: more aggressive retries
                "backoff": "exponential",
            },
            {
                "on_status_codes": [500, 502, 503],
                "max_retries": 2,  # server errors: fewer retries
                "backoff": "exponential",
            },
        ],
    },
}
```

---

## 5. The Circuit Breaker Pattern

Stop calling a failing provider for a while to give it time to recover.

```python
config = {
    "circuit_breaker": {
        "failure_threshold": 0.5,  # 50% failure rate
        "window_size": 60,  # over 60 seconds
        "cooldown_period": 300,  # 5 minutes before retrying
    },
}
```

When a provider's failure rate exceeds 50% over 60 seconds, Portkey stops calling it for 5 minutes.

This protects against:
- Cascading failures (one slow provider causes the gateway to slow)
- Wasting retries on a known-bad provider
- Cost amplification (retries cost money)

---

## 6. Rate Limit Handling

```python
config = {
    "rate_limit": {
        "provider": "openai",
        "requests_per_minute": 500,  # tier 1 limit
        "tokens_per_minute": 200000,
    },
    "on_rate_limit": {
        "action": "fallback",  # try another provider
        "wait_time": 0,  # immediate fallback
    },
}
```

When OpenAI returns 429 (rate limit), Portkey immediately tries the fallback (e.g., Together AI).

For more sophisticated rate limit handling:

```python
config = {
    "rate_limit": {
        "provider": "openai",
        "requests_per_minute": 500,
        "tokens_per_minute": 200000,
        "strategy": "queue",  # queue requests until rate limit resets
        "max_queue_size": 100,
        "max_queue_wait_seconds": 30,
    },
}
```

Queue requests up to 30 seconds; if rate limit doesn't reset, return 429.

---

## 7. Combined Pattern

For a production LLM service:

```python
config = {
    # Primary provider
    "provider": "openai",
    "override_params": {"model": "gpt-4o-mini"},
    
    # Fallbacks (in order)
    "fallbacks": [
        {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
        {"provider": "together-ai", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    ],
    
    # Retries
    "retry": {
        "attempts": 3,
        "on_status_codes": [429, 500, 502, 503, 504],
        "backoff": "exponential",
    },
    
    # Circuit breaker
    "circuit_breaker": {
        "failure_threshold": 0.5,
        "window_size": 60,
        "cooldown_period": 300,
    },
    
    # Caching
    "cache": {
        "mode": "semantic",
        "max_age": 3600,
    },
    
    # Observability
    "metadata": {
        "tenant_id": "alice",
        "feature": "chat",
    },
}
```

This config:
- Tries OpenAI first
- Falls back to Anthropic if OpenAI fails
- Falls back to Together AI if Anthropic fails
- Retries with exponential backoff on 429/5xx
- Circuit-breaks any provider with >50% failure rate
- Caches semantic-similar queries for 1 hour
- Tags every trace for observability

---

## 8. The Portfolio Pattern — LLM Edge Gateway

Your existing **LLM Edge Gateway** (covered in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|06/19]]) uses LiteLLM. Migration to Portkey adds reliability features:

```python
# Before (LiteLLM)
import litellm

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[...],
)

# After (Portkey with fallbacks + retries)
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",
    config={
        "provider": "openai",
        "override_params": {"model": "gpt-4o-mini"},
        "fallbacks": [
            {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
            {"provider": "together-ai", "model": "Llama-3.3-70B-Instruct-Turbo"},
        ],
        "retry": {"attempts": 3, "on_status_codes": [429, 500, 502, 503, 504]},
        "circuit_breaker": {"failure_threshold": 0.5, "window_size": 60, "cooldown_period": 300},
    },
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
)
```

Same code, but now with reliability features.

---

## 9. Antipatterns

### 9.1 Antipattern 1: No fallbacks

```python
# ❌ Single provider; outage = full outage
config = {"provider": "openai"}

# ✅ Always have a fallback chain
config = {
    "provider": "openai",
    "fallbacks": [
        {"provider": "anthropic"},
        {"provider": "together-ai"},
    ],
}
```

### 9.2 Antipattern 2: Aggressive retries on bad errors

```python
# ❌ Retry on 4xx client errors (no point)
config = {"retry": {"attempts": 10, "on_status_codes": [400, 401, 403, 404]}}

# ✅ Only retry on transient errors
config = {"retry": {"attempts": 3, "on_status_codes": [429, 500, 502, 503, 504]}}
```

### 9.3 Antipattern 3: No circuit breaker

```python
# ❌ Continue calling a failing provider
# Wastes time + cost on retries

# ✅ Circuit breaker stops calling a bad provider
config = {
    "circuit_breaker": {
        "failure_threshold": 0.5,
        "cooldown_period": 300,
    },
}
```

### 9.4 Antipattern 4: Load balancing without quality control

```python
# ❌ Random load balancing — high-quality traffic goes to cheap provider
config = {"strategy": {"mode": "loadbalance", "on": [...]}}

# ✅ Use quality-aware routing
config = {
    "strategy": {
        "mode": "loadbalance",
        "on": [...],
        "quality_threshold": 0.85,  # require minimum quality
    },
}
```

### 9.5 Antipattern 5: Conditional routing without fallbacks

```python
# ❌ Code → Claude (no fallback)
config = {
    "strategy": {
        "mode": "conditional",
        "conditions": [{"query": {"contains": "code"}, "then": "anthropic"}],
    },
}
# Claude is down → 500 error to user

# ✅ Conditional + fallback
config = {
    "strategy": {
        "mode": "conditional",
        "conditions": [{"query": {"contains": "code"}, "then": "anthropic"}],
    },
    "fallbacks": [
        {"provider": "openai"},  # if Anthropic down, try OpenAI
    ],
}
```

---

## 🎯 Key Takeaways

- Fallbacks: if primary fails, try next provider; resilience to outages.
- Load balancing: distribute risk + cost across providers.
- Conditional routing: code → Claude, creative → GPT-4o, background → Llama.
- Retries: only on transient errors (429, 5xx); exponential backoff.
- Circuit breaker: stop calling a failing provider for cooldown period.
- Rate limit handling: queue requests or immediate fallback.
- Combine patterns: fallbacks + retries + circuit breaker + caching.
- Avoid no fallbacks, aggressive retries on bad errors, no circuit breaker, quality-blind load balancing, conditional without fallback.

## References

- Portkey Reliability — [portkey.ai/docs/product/reliability](https://portkey.ai/docs/product/reliability)
- Portkey Configurations — [portkey.ai/docs/product/ai-gateway-stream](https://portkey.ai/docs/product/ai-gateway-stream)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — LiteLLM comparison
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/01 - Portkey Core - Gateway Fundamentals|Note 01 — Portkey Core]]
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/04 - PII Redaction and Compliance|Note 04 — PII Redaction]]
- [[06 - Large Language Models/23 - Serverless LLM Platforms|Serverless LLM Platforms]] — multi-provider cost