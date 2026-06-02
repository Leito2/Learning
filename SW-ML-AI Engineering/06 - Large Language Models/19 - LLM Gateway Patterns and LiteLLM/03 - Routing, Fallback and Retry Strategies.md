# 🏷️ Routing, Fallback and Retry Strategies

## 🎯 Learning Objectives
- Master `litellm.Router` for declarative multi-model, multi-provider routing
- Implement weight-based, cost-based, and latency-based routing
- Build fallback chains that survive provider outages and context-window overflow
- Configure cooldown logic to avoid flapping between degraded providers
- Reason about retry semantics and exponential backoff
- Compare routing strategies on a fair axis system

## Introduction

The library-level interface in [[02 - LiteLLM Core - Unified Multi-Provider Interface]] is the foundation. Once you are routing traffic through a single `completion()` call, the next question is *which* model, *which* provider, *in what order*, with *what retry policy*. The library-level API treats every request as a single point-in-time decision; the `Router` class turns that into a continuous decision over a population of deployments, each with its own health, latency, and cost profile.

This note covers the production routing surface: weight-based splits for A/B tests and gradual rollouts, cost-based routing for the same model class across providers, latency-based routing with periodic health checks, fallback chains that survive both provider outages and context-window overflow, cooldown logic that prevents flapping, and the retry semantics that turn a 30-second outage from a customer-visible event into a transient log line. By the end, you should be able to read a Router config and predict the cost, latency, and reliability of the system under any failure scenario.

---

## 1. The `Router` Class: Why It Exists

`litellm.completion()` is a one-shot decision: model in, response out. The `Router` is a *population* of deployments with a *policy* on which to pick for each call. The use cases that demand it:

- **Cost optimization**: The same model class (e.g., Llama 3.3 70B) is available on Groq, Together, and Fireworks at different prices. The Router picks the cheapest healthy one.
- **Latency optimization**: Periodically ping each provider's `/v1/models` endpoint, bias traffic to the lowest p50.
- **Reliability**: If the primary provider errors, fall back to a secondary; if that errors, a tertiary.
- **A/B tests**: 10% of traffic to a new model to measure quality vs the control.
- **Per-tenant routing**: Premium users get GPT-4o; free-tier users get Llama 3.3 8B on Groq.

The Router is configured declaratively through a `model_list` and a set of `routing_strategy`, `fallbacks`, `context_window_fallbacks`, and retry settings. The library handles the health-checking, retry, and fallback transparently — your code calls `router.acompletion()` exactly the same way it would call `litellm.acompletion()`.

---

## 2. Model Lists and Weight-Based Routing

The simplest Router configuration routes a single model name to multiple deployments, weighted by traffic share:

```python
from litellm import Router

model_list = [
    {
        "model_name": "gpt-4o-prod",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": "sk-...",
            "rpm": 10_000,  # rate limit per minute
            "tpm": 5_000_000,  # tokens per minute
        },
    },
    {
        "model_name": "gpt-4o-prod",
        "litellm_params": {
            "model": "azure/gpt-4o",
            "api_key": "...",
            "api_base": "https://mycompany.openai.azure.com",
            "rpm": 10_000,
            "tpm": 5_000_000,
        },
    },
]

router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",  # or "usage-based-routing-v2"
)

# Calls go to gpt-4o or Azure gpt-4o with simple round-robin
response = await router.acompletion(
    model="gpt-4o-prod",
    messages=[{"role": "user", "content": "Hello"}],
)
```

`simple-shuffle` rotates through deployments in a weighted round-robin. `usage-based-routing-v2` tracks each deployment's RPM/TPM utilization and biases toward the least-loaded one — this is what you want for elastic load with rate limits.

### 2.1 Weighted Routing for A/B Tests

A common pattern: 95% of traffic to the control model, 5% to a candidate, to measure quality in production. Weights are integers; the Router normalizes them.

```python
model_list = [
    {"model_name": "chat-prod", "litellm_params": {"model": "gpt-4o"}, "weight": 95},
    {"model_name": "chat-prod", "litellm_params": {"model": "claude-3-5-sonnet-20241022"}, "weight": 5},
]
```

The Router picks a deployment at random, weighted. Over 10K requests, roughly 950 go to GPT-4o and 50 to Claude. The weights can be adjusted dynamically by re-uploading the config (we cover this in [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]]).

---

## 3. Cost-Based Routing

When the same logical model is available from multiple providers at different prices, cost-based routing routes to the cheapest healthy one. LiteLLM does not have a built-in `routing_strategy="cheapest"`, but the pattern is straightforward to express with weights and a periodic price re-fetch:

```python
# Each deployment's "weight" is the inverse of its per-token cost
# Cheaper deployments get higher weights → more traffic
model_list = [
    {
        "model_name": "llama-3.3-70b-prod",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile",
            "api_key": os.environ["GROQ_API_KEY"],
        },
        "weight": 100,  # $0.59/$0.79 per MTok
    },
    {
        "model_name": "llama-3.3-70b-prod",
        "litellm_params": {
            "model": "fireworks/llama-3.3-70b-instruct",
            "api_key": os.environ["FIREWORKS_API_KEY"],
        },
        "weight": 90,  # $0.65/$0.85 per MTok
    },
    {
        "model_name": "llama-3.3-70b-prod",
        "litellm_params": {
            "model": "together_ai/llama-3.3-70b-instruct",
            "api_key": os.environ["TOGETHER_API_KEY"],
        },
        "weight": 80,  # $0.72/$0.88 per MTok
    },
]
```

For more sophisticated cost-routing, combine the Router with a periodic job that fetches the latest per-model prices from a price aggregator (e.g., LiteLLM's own cost dictionary, or a custom service) and rewrites the `weight` fields. The pattern generalizes: **weights are an arbitrary decision signal**, so any metric — quality score from offline evaluation, latency percentile, error rate — can drive routing.

---

## 4. Latency-Based Routing

Latency-based routing keeps a sliding window of recent response times per deployment and biases traffic toward the lowest p50. The simplest implementation is a health-check loop that pings each provider's `/v1/models` endpoint and records the round-trip time, then uses that as the weight:

```python
import asyncio
import time
import litellm

class LatencyAwareRouter:
    def __init__(self, deployments):
        self.deployments = deployments  # list of model strings
        self.latencies = {d: [] for d in deployments}
        self.lock = asyncio.Lock()

    async def health_check_loop(self):
        while True:
            for d in self.deployments:
                start = time.time()
                try:
                    await litellm.acompletion(
                        model=d,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                    )
                    latency = time.time() - start
                    async with self.lock:
                        self.latencies[d].append(latency)
                        if len(self.latencies[d]) > 50:
                            self.latencies[d].pop(0)
                except Exception:
                    async with self.lock:
                        self.latencies[d] = [9.99]  # mark unhealthy
            await asyncio.sleep(30)  # health-check every 30s

    def p50(self, deployment):
        samples = sorted(self.latencies[deployment])
        return samples[len(samples) // 2] if samples else 9.99

    def pick(self):
        return min(self.deployments, key=self.p50)
```

In production, this is what services like **OpenRouter** and **Portkey** do internally; the open-source path requires you to wire the health-check loop yourself, but it is ~30 lines of code. The Router class itself does not maintain this state, which is why many teams combine the library with a custom observability layer (covered in [[04 - Observability, Cost Tracking and Rate Limiting]]).

---

## 5. Fallback Chains

A fallback chain is an ordered list of alternative deployments tried when the primary fails. The triggering failures are configurable; the most common is any non-2xx, but you can be more selective.

### 5.1 Basic Fallback

```python
fallbacks = [
    {
        "gpt-4o-prod": [
            {"model": "claude-3-5-sonnet-20241022"},
            {"model": "gemini/gemini-2.0-flash"},
            {"model": "groq/llama-3.3-70b-versatile"},
        ],
    }
]

router = Router(model_list=model_list, fallbacks=fallbacks)
```

If `gpt-4o-prod` returns a 5xx, times out, or hits a rate limit, the Router tries Claude, then Gemini, then Llama. The application sees a single `acompletion()` call; the failover is invisible.

### 5.2 Context-Window Fallback

A subtler case: the primary model has a 128K context window, the request has 150K tokens, the call fails with `ContextWindowExceededError`. A context-window fallback chain tries models with progressively larger windows:

```python
context_window_fallbacks = [
    {
        "gpt-4o-prod": [  # 128K
            {"model": "claude-3-5-sonnet-20241022"},  # 200K
            {"model": "gemini/gemini-2.5-pro"},  # 1M–2M
        ],
    }
]

router = Router(
    model_list=model_list,
    fallbacks=fallbacks,
    context_window_fallbacks=context_window_fallbacks,
)
```

The error specificity matters: a generic 400 with a malformed prompt should *not* trigger a fallback (it will fail on every model). The Router only triggers context-window fallback on `ContextWindowExceededError` exceptions, which are provider-specific in name but normalized by LiteLLM.

### 5.3 Content-Policy Fallback

The third common pattern: a content filter refuses a prompt on the primary provider but accepts it on a secondary (different training, different safety stack). This requires a custom exception trigger:

```python
content_policy_fallbacks = [
    {
        "gpt-4o-prod": [
            {"model": "claude-3-5-sonnet-20241022"},
        ],
    }
]
```

This is rare in production — usually the right fix is to rewrite the prompt — but it is a real lever in some customer-support and creative-writing applications.

---

## 6. Cooldown Logic

A naive fallback chain will flap: a provider returns one 5xx, traffic is rerouted, the provider recovers, traffic is restored, the provider has another 5xx, traffic is rerouted again. The user experience is intermittent latency. Cooldown solves this by quarantining a failing deployment for a configurable duration after consecutive failures.

```python
router = Router(
    model_list=model_list,
    fallbacks=fallbacks,
    cooldown_time=60,            # seconds to quarantine
    num_retries=2,               # retries within a single deployment
    timeout=30,
    retry_policy=RetryPolicy(
        BadRequestErrorRetries=0,        # don't retry bad prompts
        TimeoutErrorRetries=2,
        RateLimitErrorRetries=3,
        InternalServerErrorRetries=3,
    ),
)
```

The combination is what matters: `num_retries=2` retries transient errors within a single deployment, then `fallbacks` activates if those retries are exhausted, and `cooldown_time=60` prevents the failing deployment from being picked again for the next minute. After cooldown expires, the deployment is eligible again and a health-check (or first real call) determines its current state.

A subtler benefit: the cooldown state is **per-worker-process**. For a horizontally-scaled service, the cooldown propagates only as fast as the slowest worker retries. The proxy server (covered in [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]]) centralizes this state in Redis, which is the right answer at scale.

---

## 7. Retry Semantics

A subtle but important distinction: retries and fallbacks are different things.

- **Retry**: re-send the *same* request to the *same* provider after a transient failure. Idempotent for non-streaming calls.
- **Fallback**: switch to a *different* provider or model after retries are exhausted.

The retry layer handles the noise of the network; the fallback layer handles the structural failure of a provider. Conflating them leads to either over-retrying (amplifying outages) or under-retrying (turning 1-second blips into user-visible errors).

```python
from litellm.router import RetryPolicy

router = Router(
    model_list=model_list,
    num_retries=3,                          # global default
    retry_policy=RetryPolicy(
        BadRequestErrorRetries=0,           # 400 is not transient
        AuthenticationErrorRetries=0,       # 401 is not transient
        RateLimitErrorRetries=3,            # 429 — back off
        TimeoutErrorRetries=3,              # 408/timeout
        InternalServerErrorRetries=3,       # 500
        ServiceUnavailableErrorRetries=3,   # 503
    ),
    retry_after=5,                          # base delay for backoff
)
```

Exponential backoff is the default: each retry waits `retry_after * 2^attempt` seconds, capped at 60s. The library also respects the `Retry-After` HTTP header that providers send with 429 responses, which is the polite way to back off (most providers send 1–30 seconds; OpenAI has been known to send 60).

⚠️ **Streaming and retries don't mix**. If a streaming call is interrupted mid-stream, retrying replays the *prompt* but cannot resume the *stream*. The common pattern is to fall back to a fresh non-streaming call for retries, then resume streaming from the fallback. This is one of the rough edges in LiteLLM as of mid-2026.

---

## 8. Code in Practice: A Production Router Config

A 5-strategy config that combines weight-based A/B, cost-based open-weights routing, latency-based failover, context-window fallback, and a tight retry budget:

```python
import os
import litellm
from litellm import Router
from litellm.router import RetryPolicy

model_list = [
    # === TIER 1: Premium users (5% of traffic) — GPT-4o only ===
    {
        "model_name": "chat-premium",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": os.environ["OPENAI_API_KEY"],
            "rpm": 5_000,
            "tpm": 2_000_000,
        },
    },
    # === TIER 2: Standard users — Claude 3.5 with Gemini fallback ===
    {
        "model_name": "chat-standard",
        "litellm_params": {
            "model": "claude-3-5-sonnet-20241022",
            "api_key": os.environ["ANTHROPIC_API_KEY"],
            "rpm": 10_000,
            "tpm": 5_000_000,
        },
    },
    # === TIER 3: Budget tier — open-weights on Groq, cost-routed ===
    {
        "model_name": "chat-budget",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile",
            "api_key": os.environ["GROQ_API_KEY"],
            "rpm": 30_000,
            "tpm": 10_000_000,
        },
    },
    {
        "model_name": "chat-budget",
        "litellm_params": {
            "model": "fireworks/llama-3.3-70b-instruct",
            "api_key": os.environ["FIREWORKS_API_KEY"],
            "rpm": 30_000,
            "tpm": 10_000_000,
        },
    },
    # === CONTEXT-WINDOW FALLBACK TARGETS ===
    {
        "model_name": "long-context",
        "litellm_params": {
            "model": "gemini/gemini-2.5-pro",
            "api_key": os.environ["GOOGLE_API_KEY"],
        },
    },
]

fallbacks = [
    {"chat-premium": [{"model": "chat-standard"}]},
    {"chat-standard": [
        {"model": "chat-budget"},
        {"model": "claude-3-5-sonnet-20241022"},  # direct provider
    ]},
    {"chat-budget": [{"model": "fireworks/llama-3.3-70b-instruct"}]},
]

context_window_fallbacks = [
    {"chat-premium": [{"model": "long-context"}]},
    {"chat-standard": [{"model": "long-context"}]},
    {"chat-budget": [{"model": "long-context"}]},
]

router = Router(
    model_list=model_list,
    fallbacks=fallbacks,
    context_window_fallbacks=context_window_fallbacks,
    routing_strategy="usage-based-routing-v2",
    num_retries=2,
    cooldown_time=60,
    timeout=30,
    retry_policy=RetryPolicy(
        BadRequestErrorRetries=0,
        RateLimitErrorRetries=2,
        TimeoutErrorRetries=2,
        InternalServerErrorRetries=2,
    ),
)

# === USAGE ===
async def route_request(user_tier: str, messages: list):
    model_name = {
        "premium": "chat-premium",
        "standard": "chat-standard",
        "budget": "chat-budget",
    }[user_tier]
    return await router.acompletion(model=model_name, messages=messages)
```

The behavior under failure scenarios:

- Premium user, GPT-4o returns 5xx → falls back to chat-standard (Claude)
- Standard user, Claude rate-limits → falls back to chat-budget (Llama)
- Budget user, Groq errors → falls back to Fireworks (same model, different provider)
- Any tier, context overflow → falls back to Gemini 2.5 Pro (1M+ context)
- Health-check shows Groq is slowest for 60s → traffic shifts to Fireworks via the usage-based router

---

## 9. Real Case: Cursor IDE's Multi-Model Routing

Cursor (the AI code editor) is one of the most public examples of multi-model routing in production. Their backend is a layered routing system: GPT-4o for general chat, Claude 3.5 Sonnet for code understanding (Cursor publicly recommends it for large refactors), and a self-hosted mix of Llama and DeepSeek for cheaper classifications and embeddings. Per-user tier is enforced at the gateway: Pro users can choose their model; Business users default to Claude; Free users default to GPT-4o-mini with strict rate limits.

The interesting detail is that they expose model choice to the *user* for Pro. A custom build of `litellm.Router` (or an equivalent) maps user-facing model names to deployments, applies per-tier rate limits, and falls back transparently when the chosen model is degraded. The architecture is open enough that several community blog posts have reverse-engineered the routing table.

---

## 10. Comparison: Routing Strategies

| Strategy | Best For | Implementation Cost | Failure Mode |
|----------|----------|---------------------|--------------|
| **Weighted round-robin** | A/B tests, gradual rollouts | Trivial | Ignores health, ignores cost |
| **Usage-based (RPM/TPM)** | Rate-limited providers, elastic load | Trivial | Slow to react to sudden capacity loss |
| **Cost-based (weighted by price)** | Open-weights across providers | Trivial + price feed | Quality variance across providers |
| **Latency-based (health-checked)** | Latency-sensitive interactive chat | Custom health-check loop | Periodic ping cost; noisy at low traffic |
| **Quality-based (offline eval)** | Stable production with periodic reweighting | Periodic eval job | Slow to detect quality regression |
| **User-tier-based** | B2B SaaS with multiple SKUs | Trivial | Ignores provider health per tier |

A useful rule: **start with weighted round-robin + tight retry policy + fallback chain**. Add latency-based routing only when p95 latency becomes a problem. Add cost-based routing only when monthly spend exceeds $20K. The complexity has a maintenance cost; do not pay it before the benefit is measurable.

---

## 11. Connection to the Rest of the Vault

The routing surface is identical across language implementations. The Go gateway in [[13 - Go Engineering/06 - Go for ML Backend/06 - Building a Production ML Gateway]] implements the same weight/cost/latency/fallback matrix in idiomatic Go, and the Bun gateway in [[Extra/Bun Runtime/06 - Bun for ML and Data Engineering]] does the same in TypeScript. The self-hosted vLLM endpoints from [[06 - Large Language Models/13 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving\|vLLM Serving]] are first-class deployments in this config — they are simply `openai/<model>` with an `api_base` pointed at the local server. The SGLang runtime from [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference/03 - SGLang - Structured Generation and RadixAttention\|SGLang]] exposes the same OpenAI-compatible surface, so it is a drop-in alternative deployment for the same model.

---

## 🎯 Key Takeaways

- `litellm.Router` is the production routing surface: it manages a population of deployments with policy on which to pick per call
- Weight-based routing covers A/B tests and gradual rollouts; `usage-based-routing-v2` adds rate-limit awareness
- Fallback chains trigger on configurable exceptions; `context_window_fallbacks` is a separate chain for oversized prompts
- Cooldown logic prevents flapping by quarantining failing deployments for a configurable window
- Retries and fallbacks are different layers: retries handle transient noise, fallbacks handle structural failure
- The combination of `num_retries=2` + `cooldown_time=60` + a 3-step fallback chain is a battle-tested default
- Latency-based routing requires a custom health-check loop; it is worth adding only when p95 latency becomes a bottleneck
- The Go and TypeScript implementations in the vault cover the same matrix — LiteLLM is the Python realization of the pattern

## References

- LiteLLM Router documentation, [docs.litellm.ai/docs/routing](https://docs.litellm.ai/docs/routing)
- LiteLLM Routing Strategies, [docs.litellm.ai/docs/routing#routing-strategies](https://docs.litellm.ai/docs/routing#routing-strategies)
- Retry Policy reference, [docs.litellm.ai/docs/routing#retry-policy](https://docs.litellm.ai/docs/routing#retry-policy)
- Cursor engineering blog, [cursor.com/blog](https://cursor.com/blog)
- Vault cross-links: [[02 - LiteLLM Core - Unified Multi-Provider Interface]], [[04 - Observability, Cost Tracking and Rate Limiting]], [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]], [[13 - Go Engineering/06 - Go for ML Backend/06 - Building a Production ML Gateway]], [[Extra/Bun Runtime/06 - Bun for ML and Data Engineering]]
