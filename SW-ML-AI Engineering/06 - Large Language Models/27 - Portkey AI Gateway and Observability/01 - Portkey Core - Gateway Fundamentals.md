# 🎯 01 - Portkey Core — Gateway Fundamentals

> **The LLM gateway built for production. Multi-provider routing, configurable providers, virtual keys, and a REST API that's a drop-in for OpenAI clients.**

## 🎯 Learning Objectives
- Set up Portkey (cloud or self-hosted) in production
- Configure multiple LLM providers with virtual keys
- Use the OpenAI-compatible API as a drop-in replacement
- Apply the Config system for per-request routing
- Use the Gateway API directly (curl, fetch)
- Understand the auth model (virtual keys + provider keys)

## Introduction

**Portkey AI Gateway** is an LLM gateway that sits between your application and the LLM providers. It exposes an OpenAI-compatible API, so any client that works with OpenAI works with Portkey unchanged.

The architecture:

```
Your App ──► Portkey Gateway ──► OpenAI
                              ──► Anthropic
                              ──► Together AI
                              ──► ... 200+ providers
```

Key features:
- **OpenAI-compatible API** — drop-in replacement
- **Virtual keys** — per-tenant, per-team, per-environment
- **Config system** — dynamic routing per request
- **Observability** — built-in (no LangFuse needed)
- **Reliability** — fallbacks, retries, timeouts
- **Compliance** — PII redaction, audit logs

This note covers the basics: setup, virtual keys, the Config system.

---

## 1. Setup

### 1.1 Cloud (fastest)

```bash
# Sign up at portkey.ai
# Create an account, get the API key

# Install the SDK
pip install portkey-ai
```

```python
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",  # Portkey API key
    provider="openai",  # Default provider
    Authorization="sk-***",  # OpenAI API key (virtual or real)
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 1.2 Self-hosted (data sovereignty)

```bash
git clone https://github.com/Portkey-AI/gateway.git
cd gateway

# Configure
cp .env.example .env
# Edit .env with your provider keys

# Run with Docker
docker compose up -d
```

The self-hosted gateway runs on `http://localhost:8787` and is OpenAI-compatible.

---

## 2. Virtual Keys

A virtual key is a Portkey-managed credential that:

- Wraps a real provider API key
- Can be scoped to specific providers, models, budgets
- Can be revoked without touching the provider

```python
from portkey_ai import Portkey

# Create a virtual key for a specific tenant
tenant_key = Portkey(
    api_key="pk-***",
    provider="openai",
    Authorization="vk-tenant-alice-***",  # virtual key for tenant Alice
)
```

Virtual keys are created in the Portkey dashboard:

```bash
curl -X POST https://api.portkey.ai/v1/virtual-keys \
    -H "Authorization: pk-***" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "tenant-alice",
        "provider": "openai",
        "budget": 1000,  # monthly budget in USD
        "budget_duration": "monthly",
        "metadata": {
            "tenant_id": "alice",
            "tier": "pro"
        }
    }'
```

The virtual key is bound to:
- A specific provider (OpenAI only)
- A monthly budget ($1000)
- Metadata (tenant_id, tier)

The application uses the virtual key. Portkey enforces the budget.

---

## 3. The Config System

The **Config** is a JSON object that controls routing per request:

```python
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",
    Authorization="vk-***",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    config={
        # Provider routing
        "provider": "openai",
        "override_params": {"model": "gpt-4o-mini"},
        
        # Caching
        "cache": {
            "mode": "semantic",
            "max_age": 3600,  # cache for 1 hour
        },
        
        # Retries
        "retry": {
            "attempts": 3,
            "on_status_codes": [429, 500, 502, 503]
        },
        
        # Fallbacks
        "fallbacks": [
            {"provider": "openai"},
            {"provider": "anthropic"},
            {"provider": "together-ai"}
        ],
        
        # Metadata for observability
        "metadata": {
            "tenant_id": "alice",
            "feature": "chat",
            "user_id": "user-123"
        }
    }
)
```

This single request:
1. Tries OpenAI first
2. If OpenAI fails (429/500/etc.), tries Anthropic
3. If Anthropic fails, tries Together AI
4. Caches the response semantically for 1 hour
5. Tags the trace with metadata for observability

---

## 4. Multi-Provider Routing

### 4.1 Load balancing

```python
config = {
    "strategy": {
        "mode": "loadbalance",
        "on": ["openai", "anthropic", "together-ai"],
        "weights": {
            "openai": 0.5,
            "anthropic": 0.3,
            "together-ai": 0.2,
        },
    },
}
```

50% of traffic goes to OpenAI, 30% to Anthropic, 20% to Together AI.

### 4.2 Conditional routing

```python
config = {
    "strategy": {
        "mode": "conditional",
        "conditions": [
            {
                "query": {"contains": "code"},
                "then": "anthropic",  # Claude is great at code
            },
            {
                "query": {"contains": "creative"},
                "then": "openai",  # GPT-4o is creative
            },
            {
                "default": True,
                "then": "together-ai",  # Llama for cost
            }
        ],
    },
}
```

Different queries route to different providers based on content.

### 4.3 Cost-based routing

```python
config = {
    "strategy": {
        "mode": "cost-optimized",
        "on": ["openai", "anthropic", "together-ai"],
        "cost-optimization": {
            "max-cost-per-1k-tokens": 0.005,  # pick cheapest that meets this
        },
    },
}
```

Automatically pick the cheapest provider that meets your quality threshold.

---

## 5. The OpenAI-Compatible API

Portkey exposes the standard OpenAI endpoints:

```python
import openai

# Use the standard OpenAI SDK, just point it at Portkey
client = openai.OpenAI(
    api_key="vk-***",  # virtual key
    base_url="https://api.portkey.ai/v1",  # Portkey endpoint
)

# Use it like normal OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user": "content": "Hello"}],
)
```

This works with ANY OpenAI-compatible client: Vercel AI SDK, LangChain, LlamaIndex, etc.

---

## 6. The Gateway REST API

For non-Python clients (e.g., curl):

```bash
curl -X POST https://api.portkey.ai/v1/chat/completions \
    -H "Authorization: vk-***" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "config": {
            "provider": "openai",
            "metadata": {"tenant_id": "alice"}
        }
    }'
```

---

## 7. Real-World Example — Drop-in LiteLLM Replacement

Your existing **LLM Edge Gateway** (covered in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|06/19]]) uses LiteLLM. Migration to Portkey is trivial:

```python
# Before (LiteLLM)
import litellm

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"tenant_id": "alice", "feature": "chat"},
)

# After (Portkey — drop-in)
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",
    provider="openai",
    Authorization="vk-***",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    config={
        "metadata": {"tenant_id": "alice", "feature": "chat"},
        "cache": {"mode": "semantic", "max_age": 3600},
        "retry": {"attempts": 3, "on_status_codes": [429, 500]},
        "fallbacks": [
            {"provider": "openai"},
            {"provider": "anthropic"},
            {"provider": "together-ai"},
        ],
    },
)
```

Portkey adds caching, retries, and fallbacks without code changes.

---

## 8. Provider Configuration

Configure multiple providers once:

```python
client = Portkey(
    api_key="pk-***",
    config={
        "providers": {
            "openai": {
                "Authorization": "sk-openai-***",
            },
            "anthropic": {
                "x-api-key": "sk-ant-***",
                "anthropic-version": "2023-06-01",
            },
            "together-ai": {
                "Authorization": "Bearer tog-***",
            },
        },
    },
)
```

The client now has access to all three providers; the routing config decides which to use per request.

---

## 9. Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Same as OpenAI streaming.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Using provider keys directly (no Portkey)

```python
# ❌ No central observability
import openai
client = openai.OpenAI(api_key="sk-***")
# If you have a bug, you don't know who hit it

# ✅ Use Portkey for all LLM calls
from portkey_ai import Portkey
client = Portkey(api_key="pk-***", Authorization="vk-***")
# Every call traced; tenant_id visible in dashboard
```

### 10.2 Antipattern 2: No fallbacks

```python
# ❌ Single provider — outage = full outage
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

### 10.3 Antipattern 3: Sending PII to providers

```python
# ❌ Sending PII
config = {
    "metadata": {"user_email": "alice@example.com"},  # PII
}

# ✅ Use PII redaction (covered in Note 04)
config = {
    "metadata": {"user_id": "user-123"},  # anonymized
    "redact_pii": True,
}
```

### 10.4 Antipattern 4: No cache for repeated queries

```python
# ❌ Same query = same cost
response = client.chat.completions.create(model="gpt-4o-mini", messages=...)

# ✅ Cache repeated queries
config = {
    "cache": {"mode": "semantic", "max_age": 3600},
}
```

### 10.5 Antipattern 5: Hardcoded model names

```python
# ❌ Hardcoded; change requires code deploy
response = client.chat.completions.create(model="gpt-4o-mini", ...)

# ✅ Configurable via virtual key or env
model = process.env.LLM_MODEL || "gpt-4o-mini"
response = client.chat.completions.create(model=model, ...)
```

---

## 🎯 Key Takeaways

- Portkey is the OpenAI-compatible LLM gateway with native observability.
- Virtual keys enforce per-tenant budgets and provider scoping.
- The Config system enables per-request routing, fallbacks, caching, retries.
- Use Portkey for compliance (PII redaction, audit logs) and reliability (fallbacks).
- Self-host for data sovereignty; cloud for hosted UI.
- Avoid direct provider keys, no fallbacks, sending PII, no caching, hardcoded models.

## References

- Portkey docs — [portkey.ai/docs](https://portkey.ai/docs)
- Portkey GitHub — [github.com/Portkey-AI/gateway](https://github.com/Portkey-AI/gateway)
- OpenAI SDK — [github.com/openai/openai-python](https://github.com/openai/openai-python)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — LiteLLM comparison
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — observability alternative
- [[06 - Large Language Models/23 - Serverless LLM Platforms|Serverless LLM Platforms]] — multi-provider
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor]] — structured outputs
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/02 - Observability and Cost Tracking|Note 02 — Observability]]