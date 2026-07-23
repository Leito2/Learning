# 🎯 02 - Replicate — Cog-Powered Inference and the Model Marketplace

> **Package any model. Run on demand. Pay per second. The simplest path from "I have a model" to "I have an API."**

## 🎯 Learning Objectives
- Use the Replicate API to run open-source models with a few lines of Python
- Package a custom model with Cog and serve it via Replicate
- Stream inference responses with `replicate.stream()` for low-latency UIs
- Choose between community models, official models, and your own Cog-deployed models
- Calculate cost economics at different traffic profiles
- Integrate Replicate with LangFuse, LiteLLM, and FastAPI

## Introduction

Replicate is the **model marketplace** for ML practitioners. Where Modal gives you Python code to run on GPUs, Replicate gives you a hosted API for thousands of pre-packaged models — text generation, image generation, speech synthesis, embeddings, video, audio, vision. The platform runs on top of AWS, GCP, and Azure GPU pools and bills per-second for the exact model you invoke.

Two consumption modes. **Run community models** with a single line: `replicate.run("stability-ai/sdxl", input={...})`. Or **package your own model** with [Cog](https://github.com/replicate/cog) — a containerization tool that wraps your model in a Docker image with a standardized API. The same model runs locally for testing and on Replicate's cloud for production.

The killer features in 2026 are the **model marketplace** (1000+ community models, many fine-tuned for specific tasks like Llama 3 for legal / medical / coding), the **second-level pricing** (you pay only for the GPU-seconds consumed, not for idle), and the **stream-first API** (`replicate.stream()` returns generators like LangChain `BaseChatModel.stream`).

For the AI/ML Engineer profile, Replicate is the **fastest path to deploy** a custom model without managing infrastructure. The Cog packaging is 20 minutes of work; the deployment is one command. Compare to Kubernetes + GPU node pools + custom serving code: 2-4 weeks.

![Replicate architecture](https://replicate.com/static/og-image.png)

---

## 1. The Replicate API — Three Lines Per Model

### 1.1 Run a community model

```python
import replicate

# Any text generation model
output = replicate.run(
    "meta/meta-llama-3-70b-instruct",
    input={
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.7,
    },
)
print(output)  # "Paris"
```

That's it. The `replicate.run()` call returns the model's output as a Python object (string, list, URL, file path, etc.).

### 1.2 Streaming output

```python
import replicate

for event in replicate.stream(
    "meta/meta-llama-3-70b-instruct",
    input={
        "prompt": "Write a 500-word essay on transformer attention.",
        "max_tokens": 500,
    },
):
    print(event, end="")
```

`replicate.stream()` returns a generator yielding token-by-token output. First token arrives in 1-2 seconds (depending on model size and cold start).

### 1.3 Async support

```python
import asyncio
import replicate

async def main():
    output = await replicate.async_run(
        "stability-ai/sdxl",
        input={"prompt": "A futuristic city at sunset"},
    )
    print(output[0])  # URL to generated image

asyncio.run(main())
```

For high-throughput pipelines, use `async_run()` with `asyncio.gather()`.

---

## 2. The Model Marketplace — 1000+ Models

The marketplace organizes models by task:

| Category | Top models | Cost per run |
|----------|-----------|--------------|
| **Text generation** | Llama 3.1 70B, Mixtral 8x22B, Claude 3.5 | $0.001-0.05 |
| **Image generation** | Stable Diffusion XL, Flux, SD3 | $0.005-0.05 |
| **Image-to-image** | InstructPix2Pix, ControlNet | $0.01-0.05 |
| **Speech** | Whisper, Bark, Tortoise TTS | $0.001-0.02 |
| **Embeddings** | BGE, GTE, E5 | $0.0001-0.001 |
| **Vision** | LLaVA, CogVLM, Florence-2 | $0.005-0.03 |
| **Upscaling** | Real-ESRGAN, SwinIR | $0.005-0.02 |
| **Video** | Stable Video Diffusion, AnimateDiff | $0.05-0.50 |

For Llama 3 70B specifically:

```python
output = replicate.run(
    "meta/meta-llama-3-70b-instruct",
    input={"prompt": "...", "max_tokens": 500},
)
# Cost: ~$0.05 per 500 tokens = $0.10 per million output tokens
```

Compare to Fireworks Llama 3 70B at ~$0.90/M tokens and Together AI at ~$0.88/M tokens. Replicate is **competitive** for moderate-traffic workloads.

---

## 3. Cog — Package Custom Models

Cog is the containerization tool that makes Replicate different from a pure marketplace. With Cog you define a model in `cog.yaml` + `predict.py`, and Replicate handles the rest.

### 3.1 Project structure

```
my-model/
├── cog.yaml         # container config
├── predict.py       # inference logic
└── weights/         # model weights (gitignored)
```

### 3.2 cog.yaml

```yaml
image: "r8.im/cog-base:cuda11.8-python3.11"
predict: "predict.py:Predictor"
train: "predict.py:train"  # optional

resources:
  gpu: "A100"
  memory: "16Gi"

env:
  - "MODEL_PATH=/weights"
```

### 3.3 predict.py

```python
from cog import BasePredictor, Input, Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Predictor(BasePredictor):
    def setup(self):
        """Load model once at startup. Runs inside the container."""
        self.tokenizer = AutoTokenizer.from_pretrained("/weights/llama-3-8b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "/weights/llama-3-8b",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    
    def predict(
        self,
        prompt: str = Input(description="Text prompt"),
        max_tokens: int = Input(default=100, ge=1, le=1000),
        temperature: float = Input(default=0.7, ge=0, le=2),
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 3.4 Build and deploy

```bash
# Install Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Login to Replicate
cog login

# Build the container (locally for testing)
cog build -t my-model

# Run locally for testing
docker run -gpus all my-model python -m predict

# Push to Replicate
cog push r8.im/your-username/my-model

# Run via API
output = replicate.run(
    "your-username/my-model",
    input={"prompt": "Hello, world"},
)
```

The full Cog workflow takes ~20 minutes for a simple model.

### 3.5 Stream from Cog

```python
from typing import Iterator

class Predictor(BasePredictor):
    def setup(self):
        # ... load model ...
        pass
    
    def predict(self, prompt: str, max_tokens: int = 100) -> Iterator[str]:
        """Yield tokens as they're generated."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        for token_id in generate_token_by_token(self.model, inputs, max_tokens):
            yield self.tokenizer.decode(token_id, skip_special_tokens=True)
```

Replicate streams `Iterator[str]` outputs to the client as Server-Sent Events.

---

## 4. Pricing Economics

Replicate bills per **GPU-second**, not per token. The cost depends on:

1. Model's GPU tier (T4, A40, A100, H100)
2. Cold start vs warm start (cold is slower)
3. Inference duration (input length + output length + setup time)

For Llama 3 70B on A100:

| Scenario | Setup time | Inference time | Total GPU-seconds | Cost |
|----------|:----------:|:--------------:|:-----------------:|-----:|
| Cold start, 100 tokens | 30s | 5s | 35s | $0.027 |
| Warm start, 100 tokens | 0s | 5s | 5s | $0.004 |
| Cold start, 500 tokens | 30s | 25s | 55s | $0.042 |
| Warm start, 500 tokens | 0s | 25s | 25s | $0.019 |

**Warm inference is 7-10× cheaper than cold**. The `keep_warm` setting on private models keeps N containers warm at $0.005/container/minute.

For high-traffic production (1M inferences/day), consider:
- `keep_warm=3` ($216/month for 3 warm A100s on private models)
- Direct provider (Together/Fireworks) at $0.88/M tokens = $880/day at 100M tokens/day

The crossover where private Cog deployment beats direct API is again **~100M tokens/day**. Below that, community models on Replicate win; above that, self-hosted private deployment or direct provider API wins.

---

## 5. Integration Patterns

### 5.1 LiteLLM routing through Replicate

```python
import litellm

# LiteLLM has Replicate as a provider
response = litellm.completion(
    model="replicate/meta/meta-llama-3-70b-instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    api_key=os.getenv("REPLICATE_API_TOKEN"),
)
print(response.choices[0].message.content)
```

This makes Replicate a first-class provider in the LiteLLM router. You can route to Replicate as a fallback when other providers fail.

### 5.2 LangFuse tracing

```python
from langfuse import observe
import replicate

@observe()
def run_replicate(prompt: str) -> str:
    return replicate.run(
        "meta/meta-llama-3-70b-instruct",
        input={"prompt": prompt, "max_tokens": 200},
    )[0]
```

The Replicate call becomes a LangFuse span with model, tokens, latency, and cost.

### 5.3 FastAPI streaming endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import replicate

app = FastAPI()

@app.post("/generate/stream")
async def stream(prompt: str):
    async def generate():
        for event in replicate.stream(
            "meta/meta-llama-3-70b-instruct",
            input={"prompt": prompt},
        ):
            yield f"data: {event}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

Standard FastAPI SSE pattern, same as the LangGraph streaming notes from [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]].

---

## 6. Case Studies

### 6.1 Case real 1: Rapid LoRA experimentation

A research team experiments with 50 LoRA variants per week. Each variant is a Cog-pushed private model:

```bash
cog push r8.im/team/lora-variant-001
cog push r8.im/team/lora-variant-002
# ... 48 more
```

Then they run evals against each via the same API:

```python
for variant in range(50):
    output = replicate.run(
        f"team/lora-variant-{variant:03d}",
        input=test_input,
    )
    # ... evaluate ...
```

The same API endpoint for all 50 variants — no per-variant infrastructure. Cost: ~$0.05 per eval × 50 variants × 100 test cases = $250 per week.

### 6.2 Case real 2: Stable Diffusion for marketing

A marketing agency uses Replicate for batch image generation:

```python
import asyncio
import replicate

prompts = ["Modern logo design for startup X", "Social media post about AI", ...]  # 100 prompts

async def generate_all():
    tasks = [
        replicate.async_run(
            "stability-ai/sdxl",
            input={"prompt": p, "num_outputs": 4},
        )
        for p in prompts
    ]
    results = await asyncio.gather(*tasks)
    return results

# 100 images × 4 variants = 400 images
# Cost: $0.025 × 400 = $10
```

Compared to a $2,000/month Midjourney subscription for 100 images/week, $10 for the batch is dramatically cheaper.

---

## 7. Antipatterns

### 7.1 Antipattern 1: Using Cog without testing locally

```bash
# ❌ Push and pray
cog push r8.im/team/model
# Cost: $0.05 per failed test run × 100 failures

# ✅ Always test locally first
cog build -t my-model
docker run -gpus all my-model python -m predict
cog push r8.im/team/model  # only after local works
```

### 7.2 Antipattern 2: Re-creating the model on every prediction call

```python
# ❌ Bug: setup() loads the model, but load happens inside predict() instead
class Predictor(BasePredictor):
    def predict(self, prompt):
        # Load model on every call (slow!)
        model = AutoModelForCausalLM.from_pretrained(...)
        return model.generate(...)

# ✅ Correct: setup() runs once, predict() uses pre-loaded model
class Predictor(BasePredictor):
    def setup(self):
        self.model = AutoModelForCausalLM.from_pretrained(...)
    
    def predict(self, prompt):
        return self.model.generate(...)
```

### 7.3 Antipattern 3: Heavy models on small GPUs

```yaml
# ❌ Llama 3 70B on T4 (won't fit in 16GB)
image: cog-base
resources:
  gpu: "T4"  # 16GB VRAM — model needs 140GB

# ✅ Match GPU to model size
resources:
  gpu: "A100-80GB"  # 80GB VRAM — still won't fit 70B in FP16 but works with FP8/INT4 quantization
```

### 7.4 Antipattern 4: Synchronous batch inference

```python
# ❌ 100 prompts serially = 50 minutes total
results = [replicate.run(model, input={"prompt": p}) for p in prompts]

# ✅ Async gather = 5x faster
results = await asyncio.gather(*(replicate.async_run(model, input={"prompt": p}) for p in prompts))
```

### 7.5 Antipattern 5: Forgetting input validation

```python
# ❌ Predictor accepts any input, including exploits
def predict(self, prompt: str) -> str:
    return self.model.generate(prompt)

# ✅ Validate inputs in predict()
def predict(self, prompt: str = Input(description="...", max_length=10000)) -> str:
    if len(prompt) > 10000:
        raise ValueError("Prompt too long")
    return self.model.generate(prompt)
```

Replicate passes inputs as HTTP query params — long prompts inflate cost and latency dramatically.

---

## 🎯 Key Takeaways

- Replicate is the model marketplace + per-second GPU billing; community models in 1 line, custom models with Cog in 20 minutes.
- `replicate.run()` for one-off inference; `replicate.stream()` for low-latency token-by-token UIs.
- Cog packages any model in a Docker container with a standardized API.
- Pricing is GPU-seconds-based: warm Llama 3 70B is $0.019/500 tokens; cold is $0.042.
- Integration via LiteLLM, LangFuse, and FastAPI SSE — no special SDK needed.
- Avoid pushing without local testing, loading models inside predict(), wrong GPU size, sync batching, missing input validation.

## References

- Replicate docs — [replicate.com/docs](https://replicate.com/docs)
- Cog docs — [github.com/replicate/cog](https://github.com/replicate/cog)
- Replicate Python client — [github.com/replicate/replicate-python](https://github.com/replicate/replicate-python)
- Replicate pricing — [replicate.com/pricing](https://replicate.com/pricing)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider routing
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured outputs on Replicate
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/01 - Modal - Python-Native Serverless GPU|Note 01 — Modal]] — alternative serverless approach
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/03 - Together AI and Fireworks - Production-Grade LLM APIs|Note 03 — Together/Fireworks]] — alternative production APIs
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/04 - Serverless Cost Optimization and Patterns|Note 04 — Cost Optimization]]
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/05 - Capstone - Production Multi-Provider Serverless Stack|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — observability
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — service deployment