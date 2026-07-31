# 🎯 03 - Cost Optimization Patterns — Model Selection, Caching, and Cost-Aware Routing

> **The patterns that turn $15K/month into $500/month for the same workload. Model selection, semantic caching, prefix caching, batching, and cost-aware routing.**

## 🎯 Learning Objectives
- Apply the 8 cost optimization patterns to reduce LLM spend by 60-90%
- Choose the right model tier for each task (right-sized models)
- Implement semantic caching with Redis + embeddings
- Use prefix caching (Anthropic, Together, Fireworks)
- Apply batching and concurrency patterns for higher throughput
- Build cost-aware routing that picks cheapest provider per request
- Forecast the savings before applying each pattern

## Introduction

Most LLM cost reductions come from **the same 8 patterns**, applied systematically. The companies that achieve 60-90% savings typically apply 4-5 of these patterns in combination.

The 8 patterns:

1. **Model selection** — use smaller models for simpler tasks
2. **Prompt optimization** — fewer tokens = lower cost
3. **Semantic caching** — skip the LLM call when possible
4. **Prefix caching** — reuse cached prefixes across calls
5. **Request batching** — pack multiple requests per LLM call
6. **Cost-aware routing** — pick cheapest provider per request
7. **Quantization** — self-host smaller models
8. **Reserved capacity** — committed-use discounts

This note covers each pattern with a cost calculator that shows the savings.

![Cost optimization patterns](https://example.com/cost-optimization.png)

---

## 1. Model Selection — Right-Sized Models

The highest-impact pattern. Different tasks need different model tiers:

| Task | Frontier model | Mid-tier | Minimal cost |
|------|---------------|----------|--------------|
| **Critical reasoning** | GPT-4o / Claude 3.5 Sonnet | GPT-4o-mini | — |
| **Standard chat** | GPT-4o | **GPT-4o-mini** (40× cheaper) | — |
| **Summarization** | GPT-4o | **GPT-4o-mini** | Llama 3 8B (Together) |
| **Classification** | GPT-4o | **GPT-4o-mini** | Llama 3 8B |
| **Extraction** | GPT-4o | **GPT-4o-mini** | Llama 3 8B |
| **Embeddings** | OpenAI text-embedding-3 | OpenAI text-embedding-3-small | Together embeddings |

### 1.1 Routing by task complexity

```python
import litellm


def smart_route(task_complexity: str, prompt: str) -> str:
    """Route by task complexity to right-sized model."""
    
    if task_complexity == "simple":
        # Use smallest viable model
        return litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
    elif task_complexity == "medium":
        return litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
    elif task_complexity == "complex":
        return litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
    elif task_complexity == "frontier":
        return litellm.completion(
            model="gpt-5",  # or Claude Opus
            messages=[{"role": "user", "content": prompt}],
        )


# Classifier that determines complexity
async def classify_complexity(prompt: str) -> str:
    """Classify task complexity using a small model."""
    # Use the smallest model for classification
    response = await smart_route("simple", f"Classify this task as simple, medium, complex, or frontier: {prompt[:200]}")
    return response.choices[0].message.content.strip().lower()
```

The classifier adds cost but saves more. For complex tasks, use frontier; for simple, use mini.

### 1.2 Real case: 70% cost reduction

A team migrating from GPT-4o to a tiered approach:
- 80% of queries: GPT-4o-mini (40× cheaper)
- 15% of queries: GPT-4o (1×)
- 5% of queries: GPT-5 (10×)

Total cost: `0.80 × (1/40) + 0.15 × 1 + 0.05 × 10 = 0.67` of original

**~33% of original cost** (the original GPT-4o cost would be 1.0). 70% reduction.

---

## 2. Prompt Optimization — Fewer Tokens

Long prompts = higher cost. Optimize:

### 2.1 Token counting and optimization

```python
import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a given model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def optimize_prompt(prompt: str) -> str:
    """Remove unnecessary tokens."""
    # Remove redundant whitespace
    optimized = " ".join(prompt.split())
    
    # Remove verbose phrases
    replacements = {
        "Could you please ": "",
        "I would like you to ": "",
        "Please note that ": "",
        "It is important to ": "",
    }
    for verbose, concise in replacements.items():
        optimized = optimized.replace(verbose, concise)
    
    return optimized


def trim_examples(examples: list[str], max_examples: int = 3) -> list[str]:
    """Keep only the most informative examples."""
    # Sort by length (assume longer = more informative)
    examples.sort(key=len, reverse=True)
    return examples[:max_examples]
```

### 2.2 Real-world optimizations

| Optimization | Tokens saved | Cost saved |
|--------------|-------------|------------|
| Remove "Could you please" | 3 | $0.00003 / req |
| Replace 500-token examples with 100-token | 400 | $0.004 / req |
| Compress verbose system prompt | 200 | $0.002 / req |
| Use abbreviations in repeated context | 50 | $0.0005 / req |
| **Total for 1M req** | 500K | $5,000 |

---

## 3. Semantic Caching

When the user asks a similar question to a recent one, skip the LLM call.

```python
import hashlib
import json
import redis
import numpy as np


class SemanticCache:
    """Cache LLM responses by semantic similarity."""
    
    def __init__(self, redis_client: redis.Redis, threshold: float = 0.95):
        self.redis = redis_client
        self.threshold = threshold
    
    def get(self, query: str) -> str | None:
        """Check if similar query exists in cache."""
        
        # Embed query
        query_embedding = embed(query)
        
        # Search for similar cached queries
        similar = self.redis.ft("llm_cache").search(
            f"*=>[KNN 1 @embedding $vec AS score]",
            query_params={"vec": query_embedding.numpy().tobytes()},
            sort_by="score",
            limit=1,
        )
        
        if similar.docs and (1 - similar.docs[0].score) >= self.threshold:
            return json.loads(similar.docs[0].content)["response"]
        
        return None
    
    def set(self, query: str, response: str):
        """Cache response for query."""
        query_embedding = embed(query)
        doc_id = hashlib.sha256(query_embedding.numpy().tobytes()).hexdigest()[:32]
        
        self.redis.ft("llm_cache").add_document(
            document_id=doc_id,
            content=json.dumps({"query": query, "response": response}),
            embedding=query_embedding.numpy().tobytes(),
            ttl=86400,  # 24 hours
        )
```

### 3.1 When to use semantic caching

| Use case | Cache hit rate | Cost reduction |
|----------|---------------|----------------|
| FAQ bot | 60-80% | 60-80% |
| Customer support (repetitive) | 40-60% | 40-60% |
| Creative writing | 5-10% | 5-10% |
| Code generation | 10-20% | 10-20% |
| Personalized recommendations | <5% | <5% |

Real case: customer support bot with 60% cache hit rate → 60% cost reduction.

### 3.2 Invalidation

```python
def invalidate_cache(pattern: str = None):
    """Invalidate cache entries."""
    if pattern is None:
        # Clear all
        self.redis.flushdb()
    else:
        # Clear by pattern (e.g., "tenant_123")
        keys = self.redis.keys(f"doc:{pattern}:*")
        self.redis.delete(*keys)
```

When a model is updated or knowledge changes, invalidate the cache.

---

## 4. Prefix Caching

Most LLM calls share a long system prompt. Prefix caching reuses the cached computation:

| Provider | Cache discount |
|----------|---------------|
| **Anthropic** | 90% off cached input ($0.30 vs $3.00) |
| **OpenAI** | 50% off auto-cached (>1024 tokens) |
| **Together AI** | Free (up to context size) |
| **Fireworks** | Free |

```python
# Anthropic with prompt caching
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=[
        {
            "type": "text",
            "text": long_system_prompt,
            "cache_control": {"type": "ephemeral"}  # Cache for 5 minutes
        }
    ],
    messages=[{"role": "user", "content": user_query}],
)
```

Real case: 1000-token system prompt × 1M requests:
- Without cache: $3,000
- With Anthropic cache: $300 (90% discount)
- **Savings: $2,700/month**

---

## 5. Request Batching

Multiple requests in one LLM call = amortized overhead:

```python
async def batch_requests(requests: list[dict]) -> list[dict]:
    """Process multiple requests in one LLM call."""
    
    # Build batched prompt
    prompt = "\n\n".join([
        f"Request {i}: {r['prompt']}"
        for i, r in enumerate(requests)
    ])
    
    # Single LLM call processes all
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt + "\n\nRespond for each request, one per line."}],
    )
    
    # Parse response (one line per request)
    lines = response.choices[0].message.content.split("\n")
    return [lines[i] if i < len(lines) else "" for i in range(len(requests))]
```

Batching is most useful for **bulk operations** (e.g., batch-classify 1000 emails). For real-time user-facing requests, batching adds latency.

---

## 6. Cost-Aware Routing

Pick the cheapest provider per request:

```python
import litellm


router = litellm.Router(
    model_list=[
        {
            "model_name": "cheap",
            "litellm_params": {"model": "gpt-4o-mini"},
            "model_info": {"cost_per_million": 0.15},
        },
        {
            "model_name": "balanced",
            "litellm_params": {"model": "gpt-4o"},
            "model_info": {"cost_per_million": 2.50},
        },
        {
            "model_name": "free-tier",
            "litellm_params": {"model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"},
            "model_info": {"cost_per_million": 0.88},
        },
    ],
    routing_strategy="usage-based-routing-v2",  # routes to cheapest available
    num_retries=2,
)
```

### 6.1 Routing by quality requirements

```python
def route_by_quality_requirement(
    prompt: str,
    quality_requirement: str,  # "low", "medium", "high"
) -> str:
    """Route by quality requirement."""
    
    if quality_requirement == "low":
        return "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"  # 6× cheaper
    elif quality_requirement == "medium":
        return "gpt-4o-mini"  # 17× cheaper than GPT-4o
    elif quality_requirement == "high":
        return "gpt-4o"  # standard
```

### 6.2 Cost-aware routing with quality fallback

```python
def smart_route(prompt: str, max_cost_usd: float) -> str:
    """Find cheapest model that meets cost constraint."""
    
    models = [
        ("gpt-5", 0.030),  # $0.030 per 1K tokens (estimate)
        ("gpt-4o", 0.010),
        ("gpt-4o-mini", 0.0006),
        ("together-llama-3-70b", 0.00088),
    ]
    
    # Estimate tokens
    tokens = count_tokens(prompt)
    
    # Find cheapest that fits
    for model, cost_per_k in models:
        cost = tokens / 1000 * cost_per_k
        if cost <= max_cost_usd:
            return model
    
    # Fallback to cheapest available
    return models[-1][0]
```

---

## 7. Quantization and Self-Hosting

For very high-volume workloads, self-hosting quantized models:

| Model size | VRAM | Quantization | Self-host cost (H100/hr) |
|-----------|------|--------------|--------------------------|
| Llama 3 8B | 16GB | FP16 | $0.0006 / 1K tokens |
| Llama 3 8B | 8GB | INT8 | $0.0003 / 1K tokens |
| Llama 3 8B | 4GB | INT4 | $0.00015 / 1K tokens |
| Llama 3 70B | 140GB | FP16 | $0.0007 / 1K tokens |

At >10M tokens/day, self-hosted vLLM with quantized models is cheaper than Together AI.

---

## 8. Reserved Capacity

For predictable workloads:

| Provider | Reserved discount |
|----------|-------------------|
| **AWS Bedrock** | 30-50% for 1-year commit |
| **Together AI** | 10-20% for monthly commit |
| **Azure OpenAI** | 30-40% for provisioned throughput |

For a $10K/month workload, reserved can save $3-5K/month.

---

## 9. The Combined Optimization — Real Case

For a 1M requests/month workload (500 input + 1000 output):

| Optimization | Monthly cost | Savings |
|--------------|--------------|---------|
| Baseline (GPT-4o, no cache) | $12,500 | $0 |
| + Model tiering (80% mini, 15% gpt-4o, 5% gpt-5) | $5,000 | $7,500 |
| + Semantic cache (40% hit rate) | $3,000 | $2,000 |
| + Prefix cache (90% Anthropic) | $2,400 | $600 |
| + Cost routing (prefer Together) | $1,800 | $600 |
| **Final** | **$1,800** | **$10,700 (85%)** |

Single optimization: 40-60% reduction. **Combined: 80-90% reduction** with the same quality.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Optimizing only one pattern

```python
# ❌ Just caching, no model tiering
cache_hit_rate = 0.40  # 40% reduction
# Misses: 60% of calls still on GPT-4o; should be on GPT-4o-mini

# ✅ Apply multiple patterns
optimizations = {
    "model_tiering": "80% on gpt-4o-mini",
    "semantic_cache": "40% cache hit rate",
    "prefix_cache": "90% Anthropic cache hit",
    "cost_routing": "Together AI for non-critical",
}
```

### 10.2 Antipattern 2: Caching with stale data

```python
# ❌ Cache never invalidated; serves old responses
cache.set(query, response)

# ✅ Invalidate on model updates or knowledge changes
def on_model_update(new_version: str):
    cache.invalidate_cache()  # clear all

def on_knowledge_update(domain: str):
    cache.invalidate_cache(pattern=domain)  # clear domain
```

### 10.3 Antipattern 3: Aggressive cache threshold

```python
# ❌ Threshold 0.85 → too aggressive; serves wrong answers
cache = SemanticCache(threshold=0.85)

# ✅ Threshold 0.95 → strict semantic match
cache = SemanticCache(threshold=0.95)
```

### 10.4 Antipattern 4: Caching sensitive data without encryption

```python
# ❌ PII in cache, unencrypted
cache.set(query, response_with_pii)

# ✅ Redact PII before caching, or use encryption
sanitized = redact_pii(response)
cache.set(query, sanitized)
```

### 10.5 Antipattern 5: Optimizing without measuring

```python
# ❌ "I think we should switch to gpt-4o-mini"
# No data; might hurt quality

# ✅ Measure first; optimize with quality guardrails
# Run A/B: GPT-4o vs gpt-4o-mini; measure cost AND quality
# Switch only if quality remains acceptable
```

---

## 🎯 Key Takeaways

- The 8 patterns: model selection, prompt optimization, semantic cache, prefix cache, batching, cost-aware routing, quantization, reserved.
- Single optimization: 40-60% reduction. Combined: 80-90%.
- Tier by task complexity (frontier for critical, mini for standard).
- Cache by semantic similarity (Redis + embeddings); invalidate on updates.
- Prefix cache with Anthropic / Together / Fireworks for shared system prompts.
- Cost-aware routing with LiteLLM Router; pick cheapest viable provider.
- Quantization for >10M tokens/day.
- Measure quality before/after; never optimize blindly.
- Avoid single-pattern optimization, stale cache, aggressive thresholds, PII in cache, blind optimization.

## References

- LiteLLM Router — [docs.litellm.ai/docs/routing](https://docs.litellm.ai/docs/routing)
- Anthropic Prompt Caching — [docs.anthropic.com/en/docs/build-with-claude/prompt-caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- vLLM Quantization — [docs.vllm.ai/en/latest/quantization](https://docs.vllm.ai/en/latest/quantization/)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — cost-aware routing
- [[06 - Large Language Models/23 - Serverless LLM Platforms/04 - Serverless Cost Optimization and Patterns|Serverless Cost Optimization]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/01 - LLM Cost Fundamentals|Note 01 — Cost Fundamentals]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/02 - Cost Visibility - Per-Tenant Attribution, Chargeback, and Showback|Note 02 — Cost Visibility]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/04 - Forecasting and Budget Management|Note 04 — Forecasting]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/05 - Capstone - FinOps Pipeline|Note 05 — Capstone]]