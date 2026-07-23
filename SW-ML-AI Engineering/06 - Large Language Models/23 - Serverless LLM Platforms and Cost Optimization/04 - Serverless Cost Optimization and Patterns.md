# 🎯 04 - Serverless Cost Optimization and Patterns

> **Cold-start, caching, hybrid architectures, multi-provider failover. The patterns that determine whether your LLM system is cost-efficient or burns cash.**

## 🎯 Learning Objectives
- Quantify cold-start overhead and design warm-container strategies
- Implement semantic and prefix caching to avoid repeated LLM calls
- Design hybrid architectures: serverless for burst, self-hosted for steady state
- Configure multi-provider failover with cost-aware routing
- Monitor per-request cost with LangFuse and trigger budget alerts
- Understand the math: when serverless beats self-hosted (and vice versa)

## Introduction

The cheapest way to run an LLM is **not to run it**. The second cheapest is to run it efficiently. Notes 01-03 covered the platforms and their APIs; this note covers the **patterns** that turn raw capability into cost-efficient production systems.

The cost dimensions matter at different scales:
- **Cold-start** dominates low-traffic workloads (1K requests/day): 30s setup per request = 50% of total GPU time
- **Routing** dominates medium-traffic (1M requests/day): choosing the right provider per request saves 30-50%
- **Caching** dominates high-traffic with repetition (10M requests/day with 80% duplicates): 5x reduction
- **Hybrid architecture** dominates enterprise scale (100M+ tokens/day): serverless for burst, self-hosted for steady state

This note covers each dimension with patterns you can copy, plus the math that tells you when each applies.

![Cost optimization patterns](https://example.com/cost-optimization.png)

---

## 1. Cold-Start Optimization

Cold starts are the killer of low-traffic serverless deployments. A cold Llama 3 70B inference on Modal can take 30-60 seconds; on Replicate it's 15-60 seconds; on Together Serverless it's 5-8 seconds.

### 1.1 The cold-start cost

For 1K requests/day on Modal Llama 3 70B:
- 30s setup + 5s inference per cold call = 35s × $0.000972 = $0.034
- Setup is 86% of total cost

For warm (cache hit):
- 0s setup + 5s inference = 5s × $0.000972 = $0.005

Cold inference is **7× more expensive** than warm. The fix: **keep warm containers**.

### 1.2 keep_warm parameter

```python
# Modal
@app.function(
    gpu="A100-80GB",
    keep_warm=3,  # always keep 3 containers warm
    memory=8192,
)
def generate(prompt: str) -> str:
    # ... inference ...
```

Cost of `keep_warm=3` on A100-80GB: 3 × $3.50/hr × 24 hr/day × 30 days = $7,560/month. Ouch.

Optimize with **time-based scaling**:

```python
@app.function(
    gpu="A100",
    scaledown_window=300,  # keep warm for 5 minutes after last call
    container_idle_timeout=900,  # shutdown after 15 min idle
    keep_warm=0,  # don't pre-warm
)
def generate(prompt: str) -> str:
    ...
```

This avoids the cost of always-on containers while reducing cold-start latency for warm requests.

### 1.3 Replicate keep_warm on private models

```python
# cog.yaml
resources:
  gpu: "A100"
  keep_warm: 1  # on private models only
```

Cost: ~$0.005/container/minute on A100 = $216/month per warm container.

### 1.4 Together Serverless + Dedicated Endpoints

Together offers two modes:
- **Serverless** (default): cold starts in 5-8s, pay per token
- **Dedicated**: warm containers, pay by GPU-hour, no per-token cost

```python
# Switch from serverless to dedicated for steady traffic
client = OpenAI(
    base_url=f"https://api.together.xyz/{dedicated_endpoint_id}/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
)
```

The breakpoint where Dedicated beats Serverless is **>200M tokens/day** for a single model. Below that, stay on Serverless.

---

## 2. Caching Strategies

### 2.1 Prefix caching — for repeated system prompts

Most production LLM calls share a long system prompt (legal instructions, company policies, etc.). Anthropic and Together both support **prefix caching**: if multiple requests share a prefix, the prefix is cached and not re-processed.

```python
# Cache hit: only the new user message is processed
SYSTEM_PROMPT = """[5000 tokens of legal instructions]"""  # cached

response = client_together.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},  # cached on first call
        {"role": "user", "content": "User question 1"},  # new
    ],
)
# Subsequent calls with the same SYSTEM_PROMPT save 80%+ on input token cost

response2 = client_together.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},  # cache hit
        {"role": "user", "content": "User question 2"},  # new
    ],
)
```

Together, Fireworks, Anthropic, and OpenAI all support prefix caching with subtle differences. Together's cache TTL is 5-10 minutes; Anthropic's is ~5 minutes; Fireworks is per-request.

### 2.2 Semantic caching — for repeated user queries

The most impactful caching strategy: if a user's query is semantically similar to a previous one, return the cached response.

```python
from redis import Redis
from openai import OpenAI

redis_client = Redis(...)
openai_client = OpenAI(api_key=..., base_url="https://api.together.xyz/v1")

def semantically_cached_chat(user_query: str) -> str:
    """Cache LLM responses by semantic similarity."""
    query_embedding = embed(user_query)
    
    # Search Redis vector store for similar past queries
    similar = redis_client.ft("llm_cache").search(
        f"*=>[KNN 5 @embedding $vec AS score]",
        query_params={"vec": query_embedding.numpy().tobytes()},
        sort_by="score",
        limit=1,
    )
    
    if similar.docs and (1 - similar.docs[0].score) < 0.05:  # 95% similar
        return json.loads(similar.docs[0].content)["response"]  # cache hit
    
    # Cache miss: call LLM
    response = openai_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": user_query}],
    )
    
    # Store in cache
    response_text = response.choices[0].message.content
    redis_client.ft("llm_cache").add_document(
        document_id=hash(query_embedding.tobytes()),
        content={"query": user_query, "response": response_text},
        embedding=query_embedding.numpy().tobytes(),
        ttl=86400,  # 24 hour TTL
    )
    
    return response_text
```

For a customer support workload with 100K queries/day where 60% are near-duplicates ("How do I reset my password?", "I forgot my password", "Reset password steps", etc. — all the same intent), semantic caching saves **60% of LLM cost**.

### 2.3 Response caching with TTL

For fully deterministic responses (translation, classification, simple extraction), cache by exact request hash:

```python
import hashlib
import json

def cached_extraction(text: str) -> dict:
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    result = openai_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": f"Extract: {text}"}],
        response_format={"type": "json_object"},
    )
    response = json.loads(result.choices[0].message.content)
    
    redis_client.setex(cache_key, 3600, json.dumps(response))  # 1 hour TTL
    return response
```

This is the **cheapest cache** — works for any deterministic call. Set TTL based on the freshness requirement.

---

## 3. Hybrid Architecture — Serverless Burst + Self-Hosted Steady State

The canonical production pattern: **mix serverless and self-hosted** based on traffic profile.

```
                       ┌─ Self-hosted vLLM (predictable 70% of traffic)
Production traffic ────┤
                       └─ Serverless (burst 30% of traffic)
```

For 100M tokens/day:
- **70M tokens/day** on self-hosted vLLM: ~3-5 GPU-hours/day × $3.50/hr = $10-17/day
- **30M tokens/day** on serverless burst: $30/M × 30 = $900/day
- **Total**: ~$920/day

Compare to **all serverless**:
- 100M tokens/day × $0.88/M = $88/day = $2,640/month

Compare to **all self-hosted**:
- Need GPUs to handle peak traffic: 24/7 × $3.50/hr × 8 GPUs = $840/day = $25k/month

The hybrid is **40-60% cheaper** than all-serverless at peak and **5-10× cheaper** than all-self-hosted.

### 3.1 LiteLLM Router with hybrid routing

```python
router = litellm.Router(
    model_list=[
        # Self-hosted primary
        {
            "model_name": "production-llm",
            "litellm_params": {
                "model": "openai/meta-llama/Llama-3.3-70B-Instruct",  # via vLLM OpenAI-compatible
                "api_base": "http://vllm.internal:8000/v1",
                "api_key": "EMPTY",
            },
        },
        # Serverless fallback (Together or Fireworks)
        {
            "model_name": "production-llm",
            "litellm_params": {
                "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "api_key": os.getenv("TOGETHER_API_KEY"),
            },
        },
    ],
    routing_strategy="simple-shuffle",  # or "usage-based-routing-v2"
    # When vLLM returns 429 or 503, fall back to Together
    fallbacks=[{"production-llm": ["production-llm"]}],
    num_retries=2,
)
```

When self-hosted vLLM is at capacity (queue backed up), LiteLLM routes to Together. The user request succeeds without affecting latency meaningfully.

### 3.2 Autoscale patterns

For K8s, use **KEDA** to autoscale based on the queue depth or request rate:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-autoscaler
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 1
  maxReplicaCount: 8
  triggers:
    - type: prometheus
      metadata:
        query: rate(vllm_queue_depth[1m]) > 100
```

When the queue depth exceeds 100, KEDA scales vLLM pods up to the max. When load drops, scale down.

---

## 4. Multi-Provider Failover with Cost Awareness

For production multi-cloud resilience, route across providers with cost attributes:

```python
router = litellm.Router(
    model_list=[
        # Cheap + fast
        {
            "model_name": "production-llm",
            "litellm_params": {"model": "fireworks_ai/llama-v3p3-70b", "api_key": ...},
            "model_info": {"cost_per_token": 0.0009},  # $/M output tokens
            "rpm": 1000,  # rate limit
        },
        # Reliable
        {
            "model_name": "production-llm",
            "litellm_params": {"model": "together_ai/llama-3.3-70b", "api_key": ...},
            "model_info": {"cost_per_token": 0.00088},
            "rpm": 500,
        },
        # Last resort
        {
            "model_name": "production-llm",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": ...},
            "model_info": {"cost_per_token": 0.015},
            "rpm": 10000,
        },
    ],
    routing_strategy="usage-based-routing-v2",  # cost-aware
)
```

Cost-aware routing avoids the expensive GPT-4o unless the cheaper providers fail.

---

## 5. Cost Monitoring with LangFuse

Track per-tenant cost with LangFuse (covered in [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability]]):

```python
from langfuse import observe
import openai

@observe()
def tenant_chat(tenant_id: str, messages: list[dict]):
    return openai_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
    )
```

Attach `tenant_id` as a metadata or via `langfuse_context.update_current_observation(metadata={"tenant_id": tenant_id})`. The LangFuse dashboard shows per-tenant cost; alerts trigger when a tenant exceeds budget.

```python
@observe()
def tenant_chat(tenant_id: str, messages: list[dict]):
    langfuse_context.update_current_observation(
        metadata={
            "tenant_id": tenant_id,
            "budget_usd": 100.0,  # monthly budget
        },
    )
    response = openai_client.chat.completions.create(...)
    
    # Cost attribution
    cost = calculate_cost(response.usage, model="llama-3.3-70b")
    langfuse_context.score_current_observation(
        name="cost_usd",
        value=cost,
    )
    
    return response
```

LangFuse UI shows:
- Per-tenant cost over time
- Top tenants by cost
- Alerts when a tenant exceeds budget

---

## 6. The Math — When to Use What

For a workload of N requests/day at average M output tokens each:

| Pattern | Daily cost | When it wins |
|---------|-----------|--------------|
| **Self-hosted vLLM** | (GPU-hours × $3.50) ~ $20-100/day | N > 5M (steady state) |
| **Serverless (Together/Fireworks)** | N × M × $0.88/M = N × 0.00000088M | N < 5M, burst |
| **Hybrid** | $50-200/day depending on burst | Enterprise production |
| **OpenAI/Anthropic** | N × M × $15/M | Frontier quality required |

The exact numbers shift with model size and provider pricing. As of 2026:

- **Below 1M requests/day at 200 output tokens**: serverless wins on cost and ops
- **1M-100M requests/day**: hybrid wins (70% self-hosted steady + 30% serverless burst)
- **100M+ requests/day**: self-hosted wins, but expect to operate Kubernetes

For the user's portfolio projects:
- **LLM Edge Gateway**: serverless (low traffic, edge-deployed)
- **Automated LLM Evaluation Suite**: serverless (batch on-demand)
- **Multi-Agent Research System**: hybrid (self-hosted for repeated queries, serverless for one-off)
- **StayBot**: serverless (low volume, latency-sensitive)

---

## 7. Antipatterns

### 7.1 Antipattern 1: Always-on serverless for low traffic

```python
# ❌ 1000 requests/day with keep_warm=5
@app.function(gpu="A100", keep_warm=5)
def generate(prompt: str) -> str:
    ...

# Cost: 5 × $3.50/hr × 24h = $420/day just to keep warm
# But: traffic is 1000/day, well below capacity

# ✅ Use Serverless (Together/Fireworks) for low traffic
client = OpenAI(api_key=..., base_url="https://api.together.xyz/v1")
```

### 7.2 Antipattern 2: No caching at all

```python
# ❌ Every chat call is a fresh LLM call
response = openai_client.chat.completions.create(...)

# ✅ Add semantic + prefix caching
# Save 60-80% on real workloads
```

### 7.3 Antipattern 3: Using the most expensive model by default

```python
# ❌ Default to GPT-4o for everything
response = openai_client_gpt4o.chat.completions.create(...)

# ✅ Tier models by complexity
response = (
    openai_client_gpt4o.chat.completions.create(...)  # only complex queries
    if is_complex(query)
    else openai_client_llama_8b.chat.completions.create(...)  # simple queries
)
```

### 7.4 Antipattern 4: Not setting budget alerts

```python
# ❌ No budget tracking → surprise $10k bill
for _ in range(1_000_000):
    call_llm()

# ✅ Set budget alerts via LangFuse or Prometheus
```

### 7.5 Antipattern 5: Re-downloading models on every cold start

```python
# ❌ Modal Cog with no volumes
@app.function(gpu="A100")
def generate(prompt: str) -> str:
    model = load_from_hub("meta-llama/Meta-Llama-3-8B")  # 30s + bandwidth

# ✅ Cache with Volumes
@app.function(gpu="A100", volumes={"/models": modal.Volume.from_name("llama")})
def generate(prompt: str) -> str:
    model = load_from_volume("/models/llama-3-8b")  # 5s
```

---

## 🎯 Key Takeaways

- Cold-start is 7× more expensive than warm; minimize with `keep_warm` or `scaledown_window`.
- Prefix caching saves 80%+ on shared system prompts; semantic caching saves 60%+ on repeated user queries.
- Hybrid architecture (serverless burst + self-hosted steady) saves 40-60% vs all-serverless at peak.
- Cost-aware routing in LiteLLM routes by $/token + RPM limits.
- LangFuse metadata enables per-tenant cost tracking and budget alerts.
- The math: < 1M req/day → serverless wins; > 100M req/day → self-hosted wins; in between → hybrid wins.
- Avoid always-on serverless for low traffic, no caching, expensive model default, no budget alerts, model re-download.

## References

- Modal scaling — [modal.com/docs/guide/scaling](https://modal.com/docs/guide/scaling)
- Replicate keep_warm — [replicate.com/docs/topics/deployments/keep-warm](https://replicate.com/docs/topics/deployments/keep-warm)
- Together caching — [docs.together.ai/docs/prompt-caching](https://docs.together.ai/docs/prompt-caching)
- LiteLLM Router — [docs.litellm.ai/docs/routing](https://docs.litellm.ai/docs/routing)
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM and Advanced RAG]] — self-hosted comparison
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider routing
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured outputs
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — cost attribution
- [[10 - Cloud, Infra y Backend/30 - WebSockets and Real-Time ML|WebSockets for ML]] — streaming patterns
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — K8s autoscaling