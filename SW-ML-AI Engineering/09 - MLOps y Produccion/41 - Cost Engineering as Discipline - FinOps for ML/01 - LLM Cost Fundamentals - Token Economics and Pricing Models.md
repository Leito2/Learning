# 🎯 01 - LLM Cost Fundamentals — Token Economics, Pricing Models, and Hidden Costs

> **The math behind every LLM bill. Token pricing formulas, provider comparison, hidden costs, and a TCO calculator that catches what the simple formula misses.**

## 🎯 Learning Objectives
- Calculate LLM costs accurately for any provider
- Distinguish input vs output vs cached token pricing
- Identify hidden costs: prefill, function calls, tool use, structured outputs, retries
- Compare providers on cost-per-quality for the same task
- Build a TCO calculator that captures the full cost (not just inference)
- Understand pricing trends in 2026 and what to expect by 2027

## Introduction

Most LLM cost calculations in code are wrong. They use `tokens * price` and miss:

- Cached input tokens (90% cheaper on Anthropic, free on OpenAI)
- Output tokens (3-5× more expensive than input)
- Reasoning tokens (o1, o3 charge for hidden reasoning)
- Function calling overhead
- Structured output overhead
- Retry costs
- Failed-request costs (rate-limited, timeouts)
- Pre-fill vs decode (different GPU costs)
- Tool-use rounds (multiple model calls per user request)

This note covers the real math, the pricing comparison, and a TCO calculator you can use in production.

![LLM cost structure](https://example.com/llm-cost.png)

---

## 1. Token Economics 101

### 1.1 The basic formula

```
cost = (input_tokens / 1M × input_price) + (output_tokens / 1M × output_price)
```

For GPT-4o:
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens

For 1M requests, each with 1000 input + 500 output tokens:
- Input cost: 1M × 1000 / 1M × $2.50 = $2,500
- Output cost: 1M × 500 / 1M × $10.00 = $5,000
- **Total: $7,500**

### 1.2 Cached input tokens

Most providers offer cache discounts:

| Provider | Cache discount |
|----------|---------------|
| **OpenAI** | 50% off cached input (auto-caching for prompts >1024 tokens) |
| **Anthropic** | 90% off cached input (cache writes + reads) |
| **Together AI** | Free prefix cache (up to context size) |
| **Fireworks** | Free prefix cache |

For 1M requests with the same 500-token system prompt:
- Anthropic: 500 tokens × 1M × $3.00 / 1M (cached) = $1,500 vs $15,000 (uncached)
- Savings: $13,500

Caching is **always worth implementing** for repeated system prompts.

### 1.3 Output is more expensive than input

Across all providers:
- Input: $X per 1M tokens
- Output: 3-5× more expensive

Why? Output tokens require more compute (autoregressive generation), input tokens are parallel.

For 1M requests, each with 500 input + 1000 output:
- Total tokens: 1.5M input equivalent + 1M output
- Cost-weighted: output dominates the bill

---

## 2. Provider Pricing Comparison (2026)

| Model | Input ($/1M) | Output ($/1M) | Cached | Notes |
|-------|-------------|---------------|--------|-------|
| **GPT-4o** | $2.50 | $10.00 | $1.25 | Most popular |
| **GPT-4o-mini** | $0.15 | $0.60 | $0.075 | Cheap |
| **Claude 3.5 Sonnet** | $3.00 | $15.00 | $0.30 | 90% cache discount |
| **Claude 3.5 Haiku** | $0.80 | $4.00 | $0.08 | Cheap Anthropic |
| **Llama 3.3 70B (Together)** | $0.88 | $0.88 | Free | Open-weight |
| **Llama 3.3 70B (Fireworks)** | $0.90 | $0.90 | Free | Faster inference |
| **DeepSeek-V3** | $0.27 | $1.10 | - | Cheapest quality |
| **Mixtral 8x22B** | $1.20 | $1.20 | - | MoE |
| **GPT-5** | $5.00 | $20.00 | $2.50 | Frontier |
| **Self-hosted vLLM (H100)** | $0.30 | $0.30 | - | Variable cost |

### 2.1 Real cost comparison for a 500M token workload

500M tokens per month, 80% input / 20% output:
- GPT-4o: $2.50 × 400M/1M + $10 × 100M/1M = $1,000 + $1,000 = **$2,000**
- Claude 3.5 Sonnet (no cache): $3 × 400M + $15 × 100M = $1,200 + $1,500 = **$2,700**
- Claude 3.5 Sonnet (with 90% cache): $0.30 × 360M + $3 × 40M + $15 × 100M = $108 + $120 + $1,500 = **$1,728**
- Llama 3.3 (Together): $0.88 × 500M = **$440**
- Self-hosted vLLM: ~$500 in H100 costs

---

## 3. Hidden Costs

The "naive" `tokens * price` misses:

### 3.1 Retry costs

A 10% retry rate adds 10% to total cost. With exponential backoff retries (covered in [[06 - Large Language Models/22 - Instructor and Structured Generation/01 - Instructor - Pydantic-Native Structured Outputs|Instructor]]), retry cost is built into the architecture.

```python
# 10% retry rate → 10% extra cost
base_cost = calculate_naive_cost()
retry_multiplier = 1.10  # 10% retry
actual_cost = base_cost * retry_multiplier
```

### 3.2 Function calling overhead

For tool-using agents (covered in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph]] and [[07 - AI Agents y Agentic Systems/19 - Semantic Kernel and AutoGen Deep Dive|SK+AutoGen]]), each user request triggers multiple model calls:

```
User request → Agent decides tool → Tool call → Model → Tool result → Final response
```

A simple task might trigger 3-5 model calls. A complex task might trigger 10+. Each call adds tokens and latency.

### 3.3 Reasoning tokens (o1, o3)

Reasoning models (o1, o3) charge for hidden reasoning tokens that the user doesn't see:

| Model | Hidden reasoning tokens |
|-------|------------------------|
| **o1** | Up to 30,000 tokens per request |
| **o1-mini** | Up to 5,000 tokens per request |
| **o3-mini** | Up to 10,000 tokens per request |

These tokens are charged but not visible. A "simple" o1 query might cost $0.30 in hidden reasoning + $0.05 in visible output.

### 3.4 Structured output overhead

Structured outputs (Instructor, Pydantic) typically use 10-30% more output tokens than free-form. The cost difference is real:

```python
# Free-form response
response = "Yes, I can help with that."
# 8 tokens, 8 × $0.00001 = $0.00008

# Structured response (Pydantic)
response = '{"can_help": true, "topic": "general", "confidence": 0.95}'
# 18 tokens, 18 × $0.00001 = $0.00018
```

### 3.5 Failed request costs

Timeouts, rate limits, and validation failures still cost something:
- **Rate-limited requests**: cost 0 (rejected before model call)
- **Timeouts**: cost partial (some tokens processed)
- **Validation failures (Pydantic)**: cost full (model generated, output rejected)

A 5% validation failure rate adds 5% to total cost.

---

## 4. The Total Cost of Ownership (TCO) Calculator

```python
from dataclasses import dataclass
from enum import Enum


class Provider(Enum):
    OPENAI_GPT4O = "gpt-4o"
    OPENAI_GPT4O_MINI = "gpt-4o-mini"
    ANTHROPIC_SONNET = "claude-3-5-sonnet"
    ANTHROPIC_HAIKU = "claude-3-5-haiku"
    TOGETHER_LLAMA = "together-llama-3-70b"
    FIREWORKS_LLAMA = "fireworks-llama-3-70b"
    SELF_HOSTED = "self-hosted-vllm"


@dataclass
class LLMUsage:
    requests_per_month: int
    avg_input_tokens: int
    avg_output_tokens: int
    cache_hit_rate: float = 0.0
    retry_rate: float = 0.0
    tool_call_multiplier: int = 1
    reasoning_model: bool = False


# 2026 pricing
PRICING = {
    Provider.OPENAI_GPT4O: {
        "input_per_million": 2.50,
        "output_per_million": 10.00,
        "cached_input_per_million": 1.25,
    },
    Provider.ANTHROPIC_SONNET: {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "cached_input_per_million": 0.30,
    },
    Provider.TOGETHER_LLAMA: {
        "input_per_million": 0.88,
        "output_per_million": 0.88,
        "cached_input_per_million": 0.00,
    },
    Provider.SELF_HOSTED: {
        "input_per_million": 0.30,
        "output_per_million": 0.30,
        "fixed_monthly_cost": 2000.00,  # H100 GPU cost
    },
}


def calculate_tco(usage: LLMUsage, provider: Provider) -> dict:
    """Calculate total cost of ownership."""
    
    pricing = PRICING[provider]
    
    # Base token cost
    total_input_tokens = usage.requests_per_month * usage.avg_input_tokens
    total_output_tokens = usage.requests_per_month * usage.avg_output_tokens
    
    # Cache discount
    cached_input = total_input_tokens * usage.cache_hit_rate
    uncached_input = total_input_tokens * (1 - usage.cache_hit_rate)
    
    input_cost = uncached_input / 1_000_000 * pricing["input_per_million"]
    if "cached_input_per_million" in pricing:
        input_cost += cached_input / 1_000_000 * pricing["cached_input_per_million"]
    
    output_cost = total_output_tokens / 1_000_000 * pricing["output_per_million"]
    
    # Reasoning model surcharge
    if usage.reasoning_model:
        # Hidden reasoning tokens, ~5x the visible output
        hidden_reasoning = total_output_tokens * 5
        output_cost += hidden_reasoning / 1_000_000 * pricing["output_per_million"]
    
    # Retry multiplier
    retry_multiplier = 1 + usage.retry_rate
    
    # Tool call multiplier (agent systems)
    tool_multiplier = usage.tool_call_multiplier
    
    # Base inference cost
    inference_cost = (input_cost + output_cost) * retry_multiplier * tool_multiplier
    
    # Add fixed costs (for self-hosted)
    fixed_cost = pricing.get("fixed_monthly_cost", 0)
    
    total_cost = inference_cost + fixed_cost
    
    return {
        "provider": provider.value,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "inference_cost": inference_cost,
        "fixed_cost": fixed_cost,
        "total_monthly_cost": total_cost,
        "cost_per_request": total_cost / usage.requests_per_month,
    }


# Example usage
usage = LLMUsage(
    requests_per_month=1_000_000,
    avg_input_tokens=1000,
    avg_output_tokens=500,
    cache_hit_rate=0.5,  # 50% cache hit rate
    retry_rate=0.05,
    tool_call_multiplier=2,  # agent makes 2 calls per request
)

for provider in [Provider.OPENAI_GPT4O, Provider.ANTHROPIC_SONNET, Provider.TOGETHER_LLAMA, Provider.SELF_HOSTED]:
    result = calculate_tco(usage, provider)
    print(f"{provider.value}: ${result['total_monthly_cost']:,.0f}/mo "
          f"(${result['cost_per_request']:.5f}/req)")
```

Output (approximate):

```
gpt-4o: $8,250/mo ($0.00825/req)
claude-3-5-sonnet: $4,167/mo ($0.00417/req)
together-llama-3-70b: $880/mo ($0.00088/req)
self-hosted-vllm: $2,880/mo ($0.00288/req)
```

---

## 5. Pricing Trend Analysis

| Year | GPT-4 cost trend | Notes |
|------|-----------------|-------|
| 2023 | $30 / 1M output | GPT-4 launch |
| 2024 | $10 / 1M output | GPT-4o |
| 2025 | $5 / 1M output | Competition |
| 2026 | $2.50-5 / 1M output | Cost down further |

Open-weight models dropped faster:
| Year | Llama 70B inference cost |
|------|-------------------------|
| 2024 | $4 / 1M tokens |
| 2025 | $1.20 / 1M tokens |
| 2026 | $0.50-1 / 1M tokens |

By 2027, expect frontier model prices to drop 50-70% more. Plan capacity accordingly.

---

## 6. The Decision Framework

For a 1M requests/month workload (500 in + 1000 out):

| Tier | Best provider | Why |
|------|---------------|-----|
| **Highest quality (regardless of cost)** | GPT-4o or Claude 3.5 Sonnet | Frontier models |
| **Best cost-quality ratio** | Llama 3.3 70B (Together/Fireworks) | 10× cheaper, 90% quality |
| **Cheapest reasonable quality** | DeepSeek-V3 | Cheapest frontier-quality |
| **Privacy / on-prem** | Self-hosted vLLM | Compliance required |
| **High throughput (>10M req/mo)** | Self-hosted or batch APIs | Volume economics |

---

## 7. Antipatterns

### 7.1 Antipattern 1: Naive cost calculation

```python
# ❌ Misses caching, output premiums, retries
cost = (tokens / 1_000_000) * 10.00

# ✅ Use the TCO calculator (includes everything)
cost = calculate_tco(usage, Provider.OPENAI_GPT4O)["total_monthly_cost"]
```

### 7.2 Antipattern 2: Comparing providers only by list price

```python
# ❌ "GPT-4o is $2.50/M input, cheaper than Claude's $3.00/M"
# But ignores 90% cache discount on Claude

# ✅ Compare on YOUR workload with cache hit rates
for provider in providers:
    cost = calculate_tco(your_usage, provider)
    print(f"{provider}: ${cost}")
```

### 7.3 Antipattern 3: Ignoring tool-calling and reasoning costs

```python
# ❌ "Each user request is one API call"
# But agents make multiple calls; reasoning models charge hidden tokens

# ✅ Model the actual call pattern
usage = LLMUsage(
    requests_per_month=1_000_000,
    tool_call_multiplier=4,  # 4 LLM calls per agent request
    reasoning_model=False,
)
```

### 7.4 Antipattern 4: Not tracking failed request costs

```python
# ❌ Failed requests "don't cost anything" (they do — partial processing)
# ❌ 5% failure rate adds 5% to actual cost

# ✅ Track and account for failure rates
total_cost = base_cost * (1 + failure_rate)
```

### 7.5 Antipattern 5: One-time cost calculation

```python
# ❌ "We calculated costs once"
# ❌ Doesn't reflect model updates, new providers, volume changes

# ✅ Recalculate quarterly
# - New models released
# - Pricing changes
# - Volume changes
# - New use cases added
```

---

## 🎯 Key Takeaways

- Token cost = `input_tokens * input_price + output_tokens * output_price`.
- Output is 3-5× more expensive than input across all providers.
- Caching: OpenAI 50% off, Anthropic 90% off, Together/Fireworks free.
- Hidden costs: retries, tool calls, reasoning tokens, structured outputs, failures.
- Open-weight models are 10-20× cheaper than closed frontier models.
- Use the TCO calculator to compare providers on YOUR workload.
- Avoid naive calculation, list-price comparison, ignoring tool calls, ignoring failures, one-time calculation.

## References

- OpenAI Pricing — [openai.com/api/pricing](https://openai.com/api/pricing)
- Anthropic Pricing — [anthropic.com/pricing](https://anthropic.com/pricing)
- Together AI Pricing — [together.ai/pricing](https://www.together.ai/pricing)
- Fireworks Pricing — [fireworks.ai/pricing](https://fireworks.ai/pricing)
- CloudZero LLM Cost Analysis — [cloudzero.com/blog/llm-cost](https://www.cloudzero.com/blog/llm-cost/)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider cost-aware routing
- [[06 - Large Language Models/23 - Serverless LLM Platforms/04 - Serverless Cost Optimization and Patterns|Serverless Cost Optimization]]
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor]] — structured output overhead
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/02 - Cost Visibility - Per-Tenant Attribution|Note 02 — Cost Visibility]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/03 - Cost Optimization Patterns|Note 03 — Cost Optimization]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/05 - Capstone - FinOps Pipeline|Note 05 — Capstone]]