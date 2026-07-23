# 🎯 03 - Together AI and Fireworks — Production-Grade LLM APIs

> **The two leading OpenAI-compatible providers for open-weight LLMs. Lower cost, faster inference, full function calling and structured output support.**

## 🎯 Learning Objectives
- Use Together AI and Fireworks as drop-in OpenAI replacements
- Choose between providers based on latency, cost, and model availability
- Implement function calling and JSON mode on both providers
- Use Together's serverless fine-tuning and Fireworks' model replication
- Stream responses with SSE for low-latency UIs
- Compare with OpenAI, Anthropic, and self-hosted vLLM for the same workload

## Introduction

Together AI and Fireworks are the two leading **OpenAI-compatible** LLM API providers specializing in open-weight models. Where Modal and Replicate are general-purpose GPU platforms, Together and Fireworks are purpose-built for production LLM serving — they offer optimized inference kernels, model replication, function calling, JSON mode, fine-tuning, and 99.9% uptime SLAs.

**Together AI** launched in 2022 as a research-focused provider of open-weight inference. By 2026 it serves Llama 3.3, Mixtral, DeepSeek-V3, Qwen 2.5, and 50+ other open-weight models. Pricing is competitive — Llama 3.1 70B at $0.88/M output tokens vs OpenAI's GPT-4o at $15/M. The platform also offers Together Serverless Endpoints (cold-start optimized) and Together Dedicated Endpoints (always-on reserved capacity).

**Fireworks AI** launched in 2022 with a focus on **low-latency inference** using proprietary kernels (FireAttention, speculative decoding). By 2026 it serves Llama 3.3, Mixtral, DeepSeek, Qwen, and many more. The killer feature is **sub-second time-to-first-token** for chat completions — typically 200-400ms faster than Together for the same model. Pricing is similar to Together.

The two providers are **not interchangeable** for all workloads. Fireworks wins on latency-sensitive applications (voice agents, real-time chat). Together wins on price-per-token at scale and serverless fine-tuning. Most production deployments use **both**: Fireworks for the latency-critical path, Together as a fallback or for fine-tuning.

For the AI/ML Engineer profile, these are the **canonical open-weight APIs** that any production deployment should evaluate. Where OpenAI offers frontier closed models, Together/Fireworks offer open-weight models at 5-20× lower cost with comparable quality for many tasks.

![Together AI logo](https://together.ai/logo.png)

---

## 1. The OpenAI-Compatibility Pattern

Both providers expose OpenAI-compatible endpoints. The same client works:

```python
from openai import OpenAI

# Together AI client
client_together = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# Fireworks client
client_fireworks = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1",
)

# Use the same .chat.completions API as OpenAI
response = client_together.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

This means any code that uses OpenAI's Python SDK works with Together and Fireworks by swapping the `base_url` and `api_key`. Tools, function calling, JSON mode, streaming all transfer over.

### 1.1 LiteLLM integration

```python
import litellm

# Together via LiteLLM
response = litellm.completion(
    model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello"}],
    api_key=os.getenv("TOGETHER_API_KEY"),
)

# Fireworks via LiteLLM
response = litellm.completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    api_key=os.getenv("FIREWORKS_API_KEY"),
)
```

LiteLLM recognizes both as providers and handles the routing, retries, cost tracking automatically. This is the canonical integration pattern from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]].

---

## 2. Streaming with SSE

Both providers support streaming with the same OpenAI streaming protocol:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Write a 500-word essay on transformers"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Time-to-first-token (TTFT) benchmarks for Llama 3 70B (output 200 tokens):

| Provider | TTFT (warm) | Total latency | $/M output tokens |
|----------|:-----------:|:-------------:|------------------:|
| **Fireworks** | 200-400ms | 2-4s | $0.90 |
| **Together Serverless** | 500-800ms | 4-7s | $0.88 |
| **Together Dedicated** | 200-400ms | 2-4s | $1.20 (flat rate) |
| **OpenAI GPT-4o** | 300-500ms | 2-3s | $15.00 |
| **Anthropic Claude 3.5 Sonnet** | 400-700ms | 3-5s | $15.00 |
| **Self-hosted vLLM on H100** | 100-300ms | 2-5s | $0.30-0.50 |

For latency-sensitive applications, **Fireworks is the fastest open-weight provider**. For batch workloads where throughput matters more than latency, Together Serverless wins on cost.

---

## 3. Function Calling and JSON Mode

Both providers support OpenAI-compatible function calling:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
# get_weather {"location": "Paris", "unit": "celsius"}
```

For JSON mode:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Extract person info: Maria is 28 and works on LLMs."}],
    response_format={"type": "json_object"},
)
```

Combined with Instructor (covered in [[06 - Large Language Models/22 - Instructor and Structured Generation]]):

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1",
))

person = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{"role": "user", "content": "Maria is 28 and works on LLMs."}],
    response_model=Person,
)
print(person.name, person.age)  # Maria 28 — typed Pydantic
```

This is the **canonical production pattern**: Instructor + Fireworks/Together as the provider. The retries on validation error work transparently.

---

## 4. Model Selection — The Catalog

### 4.1 Together AI

| Model | Size | Input $/M | Output $/M | Best for |
|-------|-----:|----------:|-----------:|----------|
| Llama 3.3 70B Instruct Turbo | 70B | $0.88 | $0.88 | General chat |
| Llama 3.1 405B Instruct Turbo | 405B | $3.50 | $3.50 | Frontier open-weight |
| Mixtral 8x22B Instruct | 141B (MoE) | $1.20 | $1.20 | Reasoning |
| DeepSeek-V3 | 671B (MoE) | $1.10 | $1.10 | Reasoning + code |
| Qwen 2.5 72B Instruct | 72B | $0.88 | $0.88 | Multilingual |
| CodeLlama 70B Instruct | 70B | $0.88 | $0.88 | Code generation |
| Llama 3.2 3B Instruct | 3B | $0.06 | $0.06 | Fast + cheap |
| Llama 3.2 1B Instruct | 1B | $0.06 | $0.06 | Edge |

### 4.2 Fireworks

| Model | Size | Input $/M | Output $/M | Best for |
|-------|-----:|----------:|-----------:|----------|
| Llama 3.3 70B Instruct | 70B | $0.90 | $0.90 | General chat |
| Llama 3.1 405B Instruct | 405B | $3.50 | $3.50 | Frontier |
| Mixtral 8x22B | 141B (MoE) | $1.20 | $1.20 | Reasoning |
| DeepSeek-V3 | 671B (MoE) | $1.10 | $1.10 | Reasoning |
| Qwen 2.5 72B | 72B | $0.90 | $0.90 | Multilingual |
| CodeLlama 70B | 70B | $0.90 | $0.90 | Code |
| Yi-Large | 34B | $0.80 | $0.80 | Multilingual |

Both providers cover the same core models with similar pricing. The differentiation is **latency and infrastructure**, not model catalog.

---

## 5. Together Fine-Tuning — Serverless LoRA

Together AI's killer feature is **serverless fine-tuning** of open-weight models:

```python
import together
from together import Together

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Upload training file
file_response = client.files.upload(
    file="training_data.jsonl",
    purpose="fine-tune",
)

# Start fine-tuning job
ft_response = client.fine_tuning.create(
    training_file=file_response.id,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    n_epochs=3,
    n_checkpoints=1,
    batch_size=4,
    learning_rate=5e-5,
    lora=True,  # LoRA fine-tuning
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)

# Wait for completion (~30 min for 8B on 10K examples)
fine_tune_id = ft_response.id
status = client.fine_tuning.retrieve(id=fine_tune_id)
print(status.status)  # "completed" when done

# Use the fine-tuned model
response = client.chat.completions.create(
    model=f"your-username/{ft_response.output_model_name}",
    messages=[{"role": "user", "content": "..."}],
)
```

Cost: ~$2-5 per fine-tune run for Llama 3.1 8B on 10K examples. Compare to Modal's $2.50/hour for an A100.

---

## 6. Fireworks Model Replication and Caching

Fireworks uses proprietary kernels (FireAttention, speculative decoding) for low-latency inference:

```python
# Same API as Together — but with sub-second TTFT
response = client_fireworks.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    # Optional: configure inference parameters
    extra_body={
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
)
```

Fireworks' speculative decoding uses a small draft model to predict tokens and the large model to verify them. On Llama 3 70B this gives 2-3× throughput vs naive decoding.

For ultra-low latency, use Fireworks' **edge deployments**:

```python
# Deploy to a regional endpoint for sub-100ms latency
response = client_fireworks.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[...],
    extra_headers={"X-Fireworks-Region": "us-east-1"},
)
```

Regional endpoints are deployed across 8+ regions globally; users connect to the nearest one.

---

## 7. The LiteLLM Router — Production Multi-Provider Pattern

For production, use LiteLLM to route across providers:

```python
import litellm

router = litellm.Router(
    model_list=[
        {
            "model_name": "production-llm",
            "litellm_params": {
                "model": "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
                "api_key": os.getenv("FIREWORKS_API_KEY"),
            },
        },
        {
            "model_name": "production-llm",
            "litellm_params": {
                "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "api_key": os.getenv("TOGETHER_API_KEY"),
            },
        },
        {
            "model_name": "production-llm",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ],
    # 70% to Fireworks (latency), 30% to Together (cost), fallback to OpenAI
    routing_strategy="usage-based",
    num_retries=3,
)

# Send a request — LiteLLM picks the provider
response = router.completion(
    model="production-llm",
    messages=[{"role": "user", "content": "Hello"}],
)
```

This is the **canonical production routing pattern**: Fireworks as primary for latency, Together as cost-optimized fallback, OpenAI as last-resort fallback. The same code from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]].

---

## 8. Case Studies

### 8.1 Case real 1: Voice agent with Fireworks

A voice agent company runs Llama 3.1 70B on Fireworks for sub-300ms TTFT:

```python
async def generate_response_stream(prompt: str):
    stream = client_fireworks.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,  # short responses for voice
        temperature=0.7,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

TTFT of 250-350ms means the agent starts speaking within 350ms of the user finishing — critical for natural conversation flow.

### 8.2 Case real 2: Batch summarization with Together

A document processing service runs 100K summaries per day on Together:

```python
async def summarize_batch(documents: list[str]) -> list[str]:
    tasks = []
    for doc in documents:
        tasks.append(replicate.async_run(...))  # also works with Together
    return await asyncio.gather(*tasks)

# Or with the OpenAI client:
async def summarize_batch_openai(docs):
    client = OpenAI(api_key=..., base_url="https://api.together.xyz/v1")
    tasks = [
        client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": f"Summarize: {doc}"}],
            max_tokens=200,
        )
        for doc in docs
    ]
    return await asyncio.gather(*tasks)
```

Cost: 100K × 200 tokens × $0.88/M = $17.60/day. Compare to OpenAI GPT-4o at $15/M × 200 × 100K = $300/day.

---

## 9. Antipatterns

### 9.1 Antipattern 1: Not setting the API key per provider

```python
# ❌ Bug: using OpenAI's key for Together calls
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # wrong key!
    base_url="https://api.together.xyz/v1",
)

# ✅ Correct: each provider's own key
client_together = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url="...")
```

### 9.2 Antipattern 2: Assuming exact model names

```python
# ❌ Provider-specific model names that won't translate
response = client_together.chat.completions.create(model="llama-3-70b", ...)  # might not exist

# ✅ Use full canonical names
response = client_together.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ...
)
```

### 9.3 Antipattern 3: Falling for "lowest price" without checking quality

```python
# ❌ $0.06/M tokens on a 1B model might not match GPT-4o quality
response = client_together.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    ...
)
# Quality gap of 30-50% vs 70B; cost savings rarely compensate

# ✅ Run offline evals before swapping providers (covered in [[06 - Large Language Models/20 - RAG Evaluation Deep Dive]])
```

### 9.4 Antipattern 4: Ignoring rate limits

```python
# ❌ Sending 1000 concurrent requests without backoff
results = await asyncio.gather(*(client_together.chat.completions.create(...) for _ in range(1000)))

# ✅ Use LiteLLM Router with built-in rate limiting and retries
router = litellm.Router(model_list=[...], num_retries=3, timeout=30)
```

### 9.5 Antipattern 5: Hard-coding model strings

```python
# ❌ Hard-coded model string in 50 places
def chat(prompt):
    return client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", ...)

# ✅ Centralize model selection
MODELS = {
    "primary": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "fast": "meta-llama/Llama-3.2-3B-Instruct",
    "frontier": "meta-llama/Llama-3.1-405B-Instruct-Turbo",
}

def chat(prompt, tier="primary"):
    return client.chat.completions.create(model=MODELS[tier], ...)
```

---

## 🎯 Key Takeaways

- Together AI and Fireworks are the leading OpenAI-compatible providers for open-weight models.
- Fireworks wins on latency (sub-400ms TTFT); Together wins on cost ($0.88 vs $0.90/M tokens for Llama 3 70B).
- Both support function calling, JSON mode, and streaming via the OpenAI SDK.
- Together has serverless fine-tuning at $2-5 per LoRA run; Fireworks has regional endpoints for sub-100ms latency.
- Production routing via LiteLLM: Fireworks primary, Together fallback, OpenAI last-resort.
- Avoid wrong API keys, wrong model names, price-only decisions without quality checks, no rate limiting, hard-coded model strings.

## References

- Together AI docs — [docs.together.ai](https://docs.together.ai)
- Fireworks AI docs — [docs.fireworks.ai](https://docs.fireworks.ai)
- Together pricing — [together.ai/pricing](https://www.together.ai/pricing)
- Fireworks pricing — [fireworks.ai/pricing](https://fireworks.ai/pricing)
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM and Advanced RAG]] — self-hosted comparison
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider routing
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured outputs
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/01 - Modal - Python-Native Serverless GPU|Note 01 — Modal]] — for custom GPU code
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/02 - Replicate - Cog-Powered Inference and the Model Marketplace|Note 02 — Replicate]] — for packaged models
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/04 - Serverless Cost Optimization and Patterns|Note 04 — Cost Optimization]]
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/05 - Capstone - Production Multi-Provider Serverless Stack|Note 05 — Capstone]]
- [[06 - Large Language Models/20 - RAG Evaluation Deep Dive|RAG Evaluation Deep Dive]] — provider quality comparison