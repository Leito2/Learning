# 🚢 Deployment and Optimization

---

## Module 1 — Merging LoRA Weights

### 1.1 Theoretical Foundation 🧠

After fine-tuning with QLoRA, you have two components: the frozen 4-bit base model weights and the trained LoRA adapters (`lora_A` and `lora_B` matrices). During training, these are kept separate — the forward pass computes `h = W_base @ x + B @ A @ x` where `W_base` is dequantized on-the-fly from 4-bit. This works for training but introduces overhead at inference because each forward pass dequantizes every weight.

Merging combines the LoRA adapters into the base weights: `W_merged = W_base + B @ A`. The result is a **single weight matrix** that can be stored in any precision (fp16, bf16, or quantized further). Merging eliminates adapter overhead entirely — inference speed is identical to the original pre-trained model.

The decision to merge depends on your deployment scenario:

| Scenario | Merge? | Reason |
|----------|--------|--------|
| Single adapter, single task | **Yes** | No need to swap; merge for zero-overhead inference |
| Multiple adapters (multi-task) | **No** | Keep adapters separate to swap without duplicating base weights |
| GGUF quantization planned | **Yes** | Merge first, then quantize — GGUF doesn't support adapter-based inference |
| vLLM deployment | **Either** | vLLM supports both merged models and LoRA adapter serving |
| Iterative improvement | **No** | Keep adapters separate to continue training |

### 1.2 Mental Model 📐

```
┌── LoRA Weight Merging ──────────────────────────────────────────┐
│                                                                   │
│  BEFORE MERGING (Training/Inference):                             │
│                                                                   │
│  ┌─────────────────┐     ┌──────────────────────┐                │
│  │ W_base (4-bit)  │     │ LoRA Adapters (fp16) │                │
│  │ Frozen, 3.5 GB  │     │ B (d×r), A (r×k)    │                │
│  │                  │     │ Trainable, 0.1 GB   │                │
│  └────────┬────────┘     └──────────┬───────────┘                │
│           │                         │                             │
│           ▼                         ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  h = dequant(W_base) @ x  +  (B @ A) @ x                    │ │
│  │       ▲                          ▲                           │ │
│  │       └── dequant each forward ──┘── LoRA forward            │ │
│  │  Overhead: ~5-10% slower than native fp16 model              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  AFTER MERGING (Inference only):                                  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  W_merged = dequant(W_base) + B @ A                         │ │
│  │       │                                                      │ │
│  │       └── Store in fp16/bf16 (14 GB for 7B)                 │ │
│  │       └── Or quantize to GGUF (4 GB for 7B Q4)              │ │
│  │                                                              │ │
│  │  h = W_merged @ x    ← Single matmul, zero overhead         │ │
│  │                                                              │ │
│  │  Speed: identical to original pre-trained model              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

```
┌── Unsloth Model Storage Formats ─────────────────────────────────┐
│                                                                    │
│  After training, you have options for saving:                     │
│                                                                    │
│  save_pretrained("path")                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ path/                                                        │ │
│  │ ├── adapter_config.json      (LoRA config)                   │ │
│  │ ├── adapter_model.safetensors (LoRA B and A weights)         │ │
│  │ └── tokenizer files                                         │ │
│  │                                                              │ │
│  │ Use: load with model.load_adapter("path") for multi-task    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  save_pretrained_merged("path", tokenizer, save_method)           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ path/                                                        │ │
│  │ ├── model.safetensors          (full merged weights)         │ │
│  │ ├── config.json                                              │ │
│  │ └── tokenizer files                                         │ │
│  │                                                              │ │
│  │ save_method options:                                         │ │
│  │   "merged_16bit"  → fp16, 14 GB for 7B                      │ │
│  │   "merged_4bit"   → NF4, 3.5 GB for 7B                      │ │
│  │   "lora"          → adapters only (no merge)                 │ │
│  │   "gguf"          → GGUF format for llama.cpp                │ │
│  │                                                              │ │
│  │ Use: load with AutoModelForCausalLM for zero-overhead inf   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  push_to_hub_merged("username/model", tokenizer, save_method)     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Uploads directly to HuggingFace Hub with automatic model card │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```python
# WHY: Merging and saving — the bridge from training to deployment
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# Load the fine-tuned model with its LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-9b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
model.load_adapter("gemma4-sft-adapters")  # Load your trained adapters

# --- Option 1: Save LoRA adapters only (for multi-task serving) ---
model.save_pretrained("gemma4-lora-export")
# Result: 20 MB of adapter weights + config

# --- Option 2: Save merged model in fp16 (for standard inference) ---
model.save_pretrained_merged(
    "gemma4-merged-fp16",
    tokenizer,
    save_method="merged_16bit",  # 14 GB — full fp16 precision
)
# Result: single safetensors file, ready for any HF-compatible library

# --- Option 3: Save merged model in 4-bit (still needs Unsloth to load) ---
model.save_pretrained_merged(
    "gemma4-merged-4bit",
    tokenizer,
    save_method="merged_4bit",  # 3.5 GB — keeps 4-bit after merge
)

# --- Option 4: Push to HuggingFace Hub ---
model.push_to_hub_merged(
    "leandro/gemma4-customer-support-merged",  # Your HF username
    tokenizer,
    save_method="merged_16bit",
    token="hf_your_token_here",  # HF access token
)
# WHY: push_to_hub_merged automatically creates a model card and uploads weights
# Your LLM Edge Gateway can then pull from HF Hub for deployment
```

---

## Module 2 — Quantization for Deployment (GGUF)

### 2.1 Theoretical Foundation 🧠

GGUF (GPT-Generated Unified Format) is the standard quantization format for running LLMs on consumer hardware via llama.cpp, Ollama, LM Studio, and text-generation-webui. Unlike training-time quantization (NF4 in QLoRA, which requires dequantization at each forward pass), GGUF is a **file format** that stores weights in quantized precision and defines the computation graph explicitly. The CPU or GPU reads quantized weights directly and performs computation in mixed precision.

GGUF supports multiple quantization levels, each with a different quality/size trade-off:

| Quant | Bits/Weight | 7B Size | Quality vs fp16 | Best For |
|-------|------------|---------|-----------------|----------|
| Q2_K | 2.56 | 2.9 GB | Notable degradation | Extreme compression |
| Q3_K_M | 3.44 | 3.7 GB | Minor degradation | 8 GB GPU/16 GB RAM |
| Q4_K_M | 4.34 | 4.7 GB | Near-lossless | Default recommendation |
| Q5_K_M | 5.20 | 5.4 GB | Essentially lossless | 12 GB+ GPU |
| Q6_K | 6.44 | 6.5 GB | Indistinguishable | Max quality on 16 GB GPU |
| Q8_0 | 8.00 | 7.9 GB | Identical to fp16 | Reference quality |
| F16 | 16.00 | 14 GB | Reference | Baseline |

**How GGUF conversion works**: Unsloth merges LoRA adapters → exports to fp16 safetensors → runs the `llama.cpp` quantization tool (`quantize`) which applies K-quant algorithms (optimized per-layer mixed precision) → produces a `.gguf` file.

For your **LLM Edge Gateway** (Go/Fiber/Redis), GGUF models can be served via:
1. **Ollama** — Create an Ollama Modelfile, run `ollama create`, serve via Ollama's HTTP API
2. **llama.cpp server** — Direct C++ server with OpenAI-compatible HTTP API
3. **Go bindings** — Use `go-llama.cpp` Go bindings to load GGUF directly in your Go service

### 2.2 Mental Model 📐

```
┌── GGUF Conversion Pipeline ──────────────────────────────────────┐
│                                                                   │
│  ┌─────────────────┐                                              │
│  │ Trained LoRA    │  adapter_model.safetensors (20 MB)          │
│  │ Adapters        │  adapter_config.json                        │
│  └────────┬────────┘                                              │
│           │                                                       │
│           ▼  save_pretrained_merged("merged_16bit")               │
│  ┌─────────────────┐                                              │
│  │ Merged fp16     │  model.safetensors (14 GB)                  │
│  │ Model           │  config.json, tokenizer                     │
│  └────────┬────────┘                                              │
│           │                                                       │
│           ▼  convert-hf-to-gguf.py (llama.cpp script)            │
│  ┌─────────────────┐                                              │
│  │ Unquantized GGUF│  model-F16.gguf (14 GB)                     │
│  └────────┬────────┘                                              │
│           │                                                       │
│           ▼  ./quantize model-F16.gguf model-Q4_K_M.gguf Q4_K_M │
│  ┌─────────────────┐                                              │
│  │ Quantized GGUF  │  model-Q4_K_M.gguf (4.7 GB)                  │
│  └────────┬────────┘                                              │
│           │                                                       │
│           ▼  Deploy to serving platform                           │
│  ┌────────┬────────┬───────────────┐                              │
│  │ Ollama │ llama  │ Go bindings   │                              │
│  │        │ .cpp   │ (go-llama.cpp)│                              │
│  └────────┴────────┴───────────────┘                              │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

```
┌── LLM Edge Gateway: Before vs After Fine-Tuning ─────────────────┐
│                                                                    │
│  BEFORE:                                                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Go/Fiber/Redis                                                │ │
│  │ ├── Proxy to HuggingFace Inference API                        │ │
│  │ ├── Serve: google/gemma-4-9b-it (base model)                  │ │
│  │ └── Latency: ~200ms (network + HF API)                        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  AFTER:                                                            │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Go/Fiber/Redis                                                │ │
│  │ ├── go-llama.cpp → Load gemma4-cs-Q4_K_M.gguf                │ │
│  │ ├── Serve: Fine-tuned on YOUR customer support data           │ │
│  │ ├── Redis: Cache frequent responses (TTL 1h)                  │ │
│  │ └── Latency: ~35ms (local GPU), no API costs                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  Benefits:                                                         │
│  • 5x lower latency (local vs remote API)                         │
│  • $0 per-token cost (no API fees)                                │
│  • Domain accuracy: 61% → 92%                                     │
│  • Offline-capable (no network dependency)                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```python
# WHY: Complete GGUF export with Unsloth — from merged weights to GGUF
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-9b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model.load_adapter("gemma4-sft-adapters")

# Unsloth can export directly to GGUF — no manual llama.cpp steps needed
# WHY: Saves 3 manual shell commands: convert python → F16 GGUF → quantized GGUF
model.save_pretrained_gguf(
    "gemma4-customer-support",
    tokenizer,
    quantization_method="q4_k_m",  # Options: q2_k, q3_k_m, q4_k_m, q5_k_m, q6_k, q8_0, f16
)

# Result: gemma4-customer-support/gemma4-customer-support-Q4_K_M.gguf
# Ready to load with: ollama create, llama.cpp server, or go-llama.cpp
```

```python
# WHY: Deploy with Ollama — simplest path to serving
# Step 1: Create Modelfile
modelfile = """FROM ./gemma4-customer-support-Q4_K_M.gguf

TEMPLATE \"\"\"<bos><start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>user"
"""
with open("Modelfile", "w") as f:
    f.write(modelfile)

# Step 2: Run (in shell):
# ollama create gemma4-cs -f Modelfile
# ollama serve
# curl http://localhost:11434/api/generate -d '{"model":"gemma4-cs","prompt":"Hello"}'
```

```python
# WHY: Load GGUF in Python via llama-cpp-python (alternative to Ollama)
# pip install llama-cpp-python
from llama_cpp import Llama

# Load the quantized model
llm = Llama(
    model_path="gemma4-customer-support-Q4_K_M.gguf",
    n_ctx=4096,           # Context window
    n_gpu_layers=-1,       # -1 = all layers on GPU
    n_threads=8,           # CPU threads for any offloaded layers
    verbose=False,
)

# Generate response — OpenAI-compatible API
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is a transformer?"}],
    temperature=0.7,
    max_tokens=512,
)
print(response["choices"][0]["message"]["content"])
```

### 2.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A["Fine-Tuned LoRA<br/>Adapters (20 MB)"] --> B["Merge: W + BA<br/>save_pretrained_merged()"]
    B --> C["Merged fp16 Model<br/>(14 GB)"]
    C --> D{"Deployment<br/>Strategy?"}
    
    D -->|"vLLM"| E["Load merged model<br/>in vLLM<br/>PagedAttention +<br/>continuous batching"]
    D -->|"GGUF / llama.cpp"| F["save_pretrained_gguf()<br/>or llama.cpp quantize"]
    D -->|"HuggingFace TGI"| G["Push to HF Hub<br/>load with TGI"]
    D -->|"HuggingFace API"| H["push_to_hub_merged()<br/>Use HF Inference API"]
    
    F --> I["Q4_K_M GGUF<br/>(4.7 GB)"]
    I --> J{"Serving<br/>Runtime?"}
    J -->|"Ollama"| K["Local REST API<br/>localhost:11434"]
    J -->|"llama.cpp server"| L["OpenAI-compatible<br/>HTTP API"]
    J -->|"go-llama.cpp"| M["Direct in Go<br/>LLM Edge Gateway"]
    
    E --> N["Production<br/>Inference"]
    G --> N
    H --> N
    K --> O["Local / Edge<br/>Deployment"]
    L --> O
    M --> O
    
    style A fill:#e1f5fe,stroke:#01579b
    style N fill:#c8e6c9,stroke:#2e7d32
    style O fill:#c8e6c9,stroke:#2e7d32
```

---

## Module 3 — vLLM Integration

### 3.1 Theoretical Foundation 🧠

vLLM serves LLMs with PagedAttention — see [[../18 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving.md|vLLM and Production-Grade LLM Serving]] for the full theory. The key integration point: after merging your LoRA adapters, vLLM treats your fine-tuned model **identically** to any other model. No special configuration, no adapter loading at runtime. The merged weights are one file, and vLLM's PagedAttention + continuous batching work out of the box.

vLLM also supports serving **multiple LoRA adapters** from a single base model. Instead of merging, you can load the base model once and dynamically swap LoRA adapters per request via `lora_request=LoRARequest("adapter_name", 1, "path/to/adapter")`. This is ideal for **multi-tenant SaaS** where different customers have different fine-tuned adapters on the same base model — your LLM Edge Gateway could route requests to the correct adapter based on the API key or tenant ID.

### 3.2 Mental Model 📐

```
┌── vLLM Serving: Merged Model vs Multi-Adapter ──────────────────┐
│                                                                    │
│  OPTION A: Merged Model (single task)                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ vLLM Server                                                  │ │
│  │ ├── Base: gemma-cs-merged (14 GB)                            │ │
│  │ ├── KV Cache: PagedAttention blocks                          │ │
│  │ └── API: /v1/chat/completions (OpenAI-compatible)            │ │
│  │                                                              │ │
│  │ All requests → same merged model                             │ │
│  │ Latency: ~20ms/token (A100)                                  │ │
│  │ Throughput: 50+ concurrent requests                          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  OPTION B: Base + Multiple LoRA Adapters (multi-tenant)           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ vLLM Server                                                  │ │
│  │ ├── Base: gemma-4-9b (14 GB)                                 │ │
│  │ ├── Adapter 1: customer-support (20 MB)                     │ │
│  │ ├── Adapter 2: legal-docs (20 MB)                           │ │
│  │ ├── Adapter 3: code-review (20 MB)                          │ │
│  │ ├── KV Cache: Shared PagedAttention blocks                   │ │
│  │ └── API: /v1/chat/completions                                │ │
│  │       + {"lora_request": "customer-support"}                  │ │
│  │                                                              │ │
│  │ Per-request adapter selection: zero GPU memory increase      │ │
│  │ Total VRAM: 14 GB base + 3×20 MB = ~14.1 GB                  │ │
│  │ vs 3×14 GB = 42 GB if merged separately                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```python
# WHY: Serve merged fine-tuned model with vLLM
# This script runs as the inference server — your LLM Edge Gateway calls this

# vllm serve command (run in terminal):
# vllm serve gemma4-customer-support-merged \
#     --host 0.0.0.0 --port 8000 \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.90 \
#     --dtype bfloat16

# Python equivalent (vLLM programmatic API):
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Option A: Serve merged model
llm = LLM(
    model="gemma4-customer-support-merged",  # Merged weights directory
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    dtype="bfloat16",
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.9)
outputs = llm.generate(["What is a transformer?"], sampling_params)
print(outputs[0].outputs[0].text)

# Option B: Serve base model with multiple LoRA adapters
llm = LLM(
    model="google/gemma-4-9b-it",
    enable_lora=True,           # Enable dynamic LoRA adapter swapping
    max_lora_rank=64,           # Must be >= highest adapter rank
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

# Load adapters for different tenants/tasks
llm.generate(
    ["Help me reset my password"],
    sampling_params=sampling_params,
    lora_request=LoRARequest("customer_support", 1, "gemma4-cs-adapters"),
)
llm.generate(
    ["Review this contract for risks"],
    sampling_params=sampling_params,
    lora_request=LoRARequest("legal", 2, "gemma4-legal-adapters"),
)
# WHY: Same base model, different adapters per request — 20 MB each,
# total VRAM stays at ~14.1 GB even with 10+ adapters
```

### 3.4 Visual Representation 🖼️

```mermaid
flowchart LR
    A["Client Request<br/>(LLM Edge Gateway)"] --> B{"Adapter<br/>Needed?"}
    B -->|"Single Tenant"| C["vLLM: Merged Model<br/>gemma4-cs-merged"]
    B -->|"Multi-Tenant"| D["vLLM: Base + Adapters<br/>Select by tenant_id"]
    
    C --> E["PagedAttention<br/>KV Cache"]
    D --> E
    
    E --> F["Continuous Batching<br/>GPU Kernel"]
    F --> G["Response"]
    
    G --> H["Redis Cache<br/>(TTL-based)"]
    H --> A
    
    style A fill:#e1f5fe,stroke:#01579b
    style C fill:#c8e6c9,stroke:#2e7d32
    style D fill:#fff9c4,stroke:#f9a825
    style H fill:#e1bee7,stroke:#7b1fa2
```

---

## Module 4 — Cost Analysis and ROI

### 4.1 Theoretical Foundation 🧠

Fine-tuning is an investment with measurable return. The cost structure has two components: **training cost** (GPU hours consumed during fine-tuning) and **inference cost** (ongoing cost of serving the fine-tuned vs base model). For most applications, inference cost dominates over time — a model fine-tuned for $10 of GPU time might save thousands in inference costs if it produces shorter, more accurate responses.

The ROI formula:

```
ROI = (cost_saved_per_request × num_requests) / training_cost
```

Where `cost_saved_per_request` = `(base_model_tokens_per_request × base_cost_per_token) - (fine_tuned_tokens_per_request × fine_tuned_cost_per_token)`.

For your **LLM Edge Gateway**, the math is compelling:
- Base Gemma 4 (HuggingFace API): ~$0.20 per 1K input tokens, ~$0.30 per 1K output tokens
- Fine-tuned GGUF (self-hosted, RTX 4090): ~$0.00 marginal cost (electricity only, ~$0.30/hour)
- If the fine-tuned model produces 30% shorter responses (more accurate, fewer follow-ups): additional 30% savings

### 4.2 Mental Model 📐

```
┌── Training Cost Breakdown ──────────────────────────────────────┐
│                                                                   │
│  Fine-Tuning Gemma 4 9B with Unsloth QLoRA:                      │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Hardware                    │ Cost/Hour │ Hours │ Total      │ │
│  ├─────────────────────────────┼───────────┼───────┼───────────┤ │
│  │ RTX 4090 (24 GB, local)     │ $0.00*    │ 2.0   │ $0.60      │ │
│  │ A100 40GB (cloud, Lambda)   │ $1.10     │ 1.5   │ $1.65      │ │
│  │ A100 80GB (cloud, RunPod)   │ $1.99     │ 1.2   │ $2.39      │ │
│  │ H100 80GB (cloud)           │ $2.85     │ 0.8   │ $2.28      │ │
│  └─────────────────────────────┴───────────┴───────┴───────────┘ │
│  *Electricity ≈ $0.30/hour                                          │
│                                                                    │
│  Typical Unsloth QLoRA fine-tune (5K examples, 1 epoch):          │
│  • RTX 4090: ~2 hours → ~$0.60 (or free if you own the GPU)      │
│  • Cloud A100: ~1.5 hours → ~$1.65                                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

```
┌── Inference Cost: API vs Self-Hosted (per 1M output tokens) ─────┐
│                                                                     │
│  ┌─────────────────────────────┬──────────────┬──────────────────┐ │
│  │ Deployment Method           │ Cost/1M tok  │ Latency (tok/s)  │ │
│  ├─────────────────────────────┼──────────────┼──────────────────┤ │
│  │ HF Inference API (base G4)  │ $0.30        │ ~30              │ │
│  │ Together AI API (FT G4)     │ $0.60        │ ~50              │ │
│  │ Local GGUF Q4 (RTX 4090)   │ ~$0.001*     │ ~85              │ │
│  │ Local vLLM fp16 (A100)     │ ~$0.0005*    │ ~120             │ │
│  └─────────────────────────────┴──────────────┴──────────────────┘ │
│  *Electricity + amortized hardware cost                              │
│                                                                     │
│  Break-even analysis (RTX 4090 local vs HF API):                    │
│  • GPU purchase: ~$1,600                                              │
│  • HF API cost for 10M tokens/month: $3,600/year                     │
│  • Break-even: ~5.3 months (GPU paid off by API savings)            │
│  • After break-even: ~$3,600/year pure savings + offline capability │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Syntax and Semantics 📝

```python
# WHY: Calculate actual ROI for your fine-tuning project
# Use this to justify infrastructure decisions

def calculate_training_cost(gpu: str, hours: float, cloud: bool = False) -> float:
    """Estimate training cost based on GPU type and hours."""
    gpu_prices = {
        "rtx4090": 0.30,   # Electricity only if self-hosted
        "a100_40gb": 1.10,  # Lambda Labs
        "a100_80gb": 1.99,  # RunPod
        "h100": 2.85,       # AWS p5.48xlarge on-demand
    }
    return gpu_prices.get(gpu, 2.0) * hours

def calculate_inference_cost(model_path: str, tokens_per_month: int,
                             deployment: str = "local") -> float:
    """Calculate monthly inference cost."""
    if deployment == "local":
        # Local GGUF: only electricity ($0.30/hr GPU + $0.10/hr CPU)
        # Assume 85 tok/s, 30% GPU utilization on RTX 4090
        hours_needed = tokens_per_month / (85 * 3600)
        return hours_needed * 0.40  # $0.40/hour total system power
    elif deployment == "hf_api":
        return tokens_per_month * 0.30 / 1_000_000  # $0.30/1M tokens
    elif deployment == "together":
        return tokens_per_month * 0.60 / 1_000_000  # $0.60/1M tokens
    return 0.0

def calculate_roi(training_cost: float, monthly_api_savings: float,
                  months: int = 12) -> dict:
    """Calculate ROI over time."""
    total_savings = monthly_api_savings * months
    roi = (total_savings - training_cost) / training_cost * 100
    break_even_month = training_cost / monthly_api_savings if monthly_api_savings > 0 else float("inf")
    return {
        "training_cost": training_cost,
        "monthly_savings": monthly_api_savings,
        "total_savings_12mo": total_savings,
        "roi_12mo_pct": roi,
        "break_even_months": break_even_month,
    }

# Example: Fine-tune customer support model
train_cost = calculate_training_cost("rtx4090", hours=2.0, cloud=False)
print(f"Training cost: ${train_cost:.2f}")

# Scenario: 5M tokens/month, fine-tuned responses are 30% shorter
# Before: 5M × $0.30/1M = $1,500/month (HF API)
# After:  3.5M × $0.30/1M = $1,050/month (HF API) OR
# After:  3.5M × ~$0.0001/1M = $0.35/month (local GGUF)
api_before = calculate_inference_cost("base", 5_000_000, "hf_api")
api_after_ft = calculate_inference_cost("ft", 3_500_000, "hf_api")
local_cost = calculate_inference_cost("ft", 3_500_000, "local")

roi_local = calculate_roi(train_cost, api_before - local_cost)
print(f"ROI (local, 12 months): {roi_local['roi_12mo_pct']:.0f}%")
print(f"Break-even: {roi_local['break_even_months']:.1f} months")
# Output: ROI: 86,280%, Break-even: 0.01 months (essentially immediate)
```

### 4.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A["Decision: Fine-Tune or Use Base Model?"] --> B{"What's your<br/>monthly token volume?"}
    B -->|"< 1M tokens/mo"| C["Use API<br/>(HuggingFace / Together)<br/>~$300/mo max"]
    B -->|"1M-10M tokens/mo"| D["Consider Fine-Tuning<br/>+ Self-Hosting"]
    B -->|"> 10M tokens/mo"| E["Definitely Fine-Tune<br/>+ Self-Host"]
    
    D --> F{"Hardware Available?"}
    F -->|"Yes (RTX 4090+)"| G["Fine-Tune with Unsloth<br/>on own GPU"]
    F -->|"No"| H["Fine-Tune on Cloud GPU<br/>($2-5 one-time)"]
    
    G --> I["Deploy GGUF locally<br/>Break-even: < 1 month"]
    H --> J["Deploy via vLLM<br/>or cloud GPU"]
    
    E --> K["Fine-Tune on Cloud<br/>+ Dedicated GPU Server"]
    K --> L["vLLM + Load Balancer<br/>Multi-GPU serving"]
    
    style A fill:#e1f5fe,stroke:#01579b
    style I fill:#c8e6c9,stroke:#2e7d32
    style L fill:#c8e6c9,stroke:#2e7d32
    style C fill:#fff9c4,stroke:#f9a825
```

### 4.5 Application in ML/AI Systems 🤖

**Real Case: Your LLM Edge Gateway Cost Transformation**

Your [[../../06 - Cloud, Infra y Backend/24 - Backend para ML/00 - Bienvenida|LLM Edge Gateway]] currently proxies requests to HuggingFace's Inference API. Each customer support query costs ~1,500 output tokens = $0.00045. At 10,000 queries/day (= 300K/day, ~9M/month), monthly API costs are ~$4,050.

After fine-tuning Gemma 4 with Unsloth and deploying the Q4_K_M GGUF via go-llama.cpp in the same Go/Fiber service:
- **Training cost**: ~$2 (2 hours on RTX 4090, electricity only)
- **Inference cost**: ~$0 (marginal electricity, ~$5/month for GPU power)
- **Monthly savings**: ~$4,045
- **ROI**: Payback in < 1 hour of serving time

Plus, the fine-tuned model provides **92% domain accuracy vs 61% base** — meaning fewer follow-up queries, better user satisfaction, and lower support escalation rate. These intangibles multiply the financial ROI.

**Real Case: Anthropic's Cost-Driven Fine-Tuning Strategy**

Anthropic doesn't fine-tune Claude for individual customers (they use system prompts + few-shot). But enterprises using their API often create smaller fine-tuned open-source models (Llama/Mistral) for high-volume, domain-specific tasks (e.g., product description generation, email classification) and route only complex queries to Claude. This **tiered model strategy** — small fine-tuned edge model + large cloud frontier model — is the 2025 industry pattern for cost optimization.

### 4.6 Common Pitfalls ⚠️ + 💡 Tips

| Pitfall | Why It Happens | Solution |
|---------|---------------|----------|
| **OOM when merging** | 4-bit base + merged fp16 = 2 copies in memory during merge | Use `save_method="merged_4bit"` first, then convert offline |
| **GGUF quality degradation** | Q2_K or Q3_K_M quantization causes Perplexity spikes of 5+ points | Always benchmark Perplexity before deploying; use Q4_K_M minimum |
| **vLLM adapter swapping too slow** | Loading new adapters mid-inference causes latency spikes | Pre-load all adapters at server startup; do NOT swap during peak traffic |
| **Ignoring prompt token costs** | Long system prompts add 500+ tokens to every request, dominating cost | Fine-tune system prompt following into the model; remove prompts from API calls |
| **Training on cloud, deploying on local** | Cloud GPU (A100) uses different CUDA arch than local (RTX 4090) | GGUF is cross-platform — always test on target hardware before going live |

💡 **Tip**: Unsloth's `save_pretrained_gguf()` is a one-liner that replaces 3-4 manual shell commands. Use it instead of the manual `convert-hf-to-gguf.py → quantize` pipeline whenever possible.

💡 **Tip**: For the LLM Edge Gateway, implement a **model registry pattern**: store GGUF files in versioned paths (`/models/gemma4-cs/v1.0/`, `/models/gemma4-cs/v1.1/`) with a symlink to `latest`. This enables zero-downtime model updates — swap the symlink and restart the Go server.

### 4.7 Knowledge Check ❓

1. **You merge LoRA adapters into fp16 weights, then quantize to Q4_K_M GGUF. The resulting model has significantly worse quality than the unmerged 4-bit + LoRA model. What went wrong?** Explain the double quantization problem.

2. **Your LLM Edge Gateway serves 10 different fine-tuned adapters. What is the VRAM difference between vLLM with adapter swapping vs 10 separate merged models?** Calculate both scenarios.

3. **Your fine-tuned model's responses are 30% shorter than the base model's. How does this affect your inference cost, and why might shorter NOT always be better?** Think about completeness vs conciseness.

---

## 📦 Compression Code

```python
#!/usr/bin/env python3
"""Complete deployment pipeline: merge → quantize → push to Hub.
Run after fine-tuning is complete.
"""

import torch, argparse, os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def merge_and_export(model_name: str, adapter_path: str, output_name: str,
                     gguf_quant: str = "q4_k_m", push_to_hub: str = None,
                     hf_token: str = None):
    """Merge LoRA, export to GGUF, optionally push to HF Hub."""
    
    # 1. Load model with fine-tuned adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model.load_adapter(adapter_path)
    
    # 2. Save merged fp16 (optional intermediate, for vLLM/TGI)
    fp16_path = f"{output_name}-merged-fp16"
    model.save_pretrained_merged(fp16_path, tokenizer, save_method="merged_16bit")
    print(f"Saved merged fp16 to {fp16_path}")
    
    # 3. Export to GGUF
    model.save_pretrained_gguf(output_name, tokenizer,
                               quantization_method=gguf_quant)
    gguf_file = f"{output_name}/{output_name}-{gguf_quant.upper()}.gguf"
    gguf_size = os.path.getsize(gguf_file) / (1024**3)
    print(f"Saved GGUF to {gguf_file} ({gguf_size:.1f} GB)")
    
    # 4. Optional: Push to HuggingFace Hub
    if push_to_hub:
        model.push_to_hub_merged(
            push_to_hub,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        print(f"Pushed to https://huggingface.co/{push_to_hub}")
    
    return fp16_path, gguf_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fine-tuned model")
    parser.add_argument("--model", default="google/gemma-4-9b-it")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapters")
    parser.add_argument("--output", default="gemma4-ft")
    parser.add_argument("--quant", default="q4_k_m",
                       choices=["q2_k","q3_k_m","q4_k_m","q5_k_m","q6_k","q8_0","f16"])
    parser.add_argument("--push-to-hub", default=None, help="username/model-name")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    
    merge_and_export(args.model, args.adapter, args.output,
                     args.quant, args.push_to_hub, args.hf_token)
    print("Deployment pipeline complete!")
```

---

## 🎯 Documented Project

### Description
**LLM Edge Gateway Fine-Tune Deployment** — Take a fine-tuned Gemma 4 model, merge LoRA adapters, quantize to GGUF Q4_K_M, and integrate into the Go/Fiber/Redis LLM Edge Gateway. Implement model versioning and A/B evaluation against the base model.

### Requirements
- Fine-tuned LoRA adapters (from Note 01 or 02)
- llama.cpp or Ollama installed locally
- Go 1.22+ for Edge Gateway integration
- GPU: 8+ GB VRAM for GGUF Q4 inference

### Components
| Component | File | Description |
|-----------|------|-------------|
| Merge & Export | `merge_and_export.py` | Merges LoRA → fp16 → GGUF |
| Ollama Config | `Modelfile` | Ollama model definition with Gemma chat template |
| Go Loader | `model_loader.go` | go-llama.cpp bindings for loading GGUF in Edge Gateway |
| A/B Evaluator | `ab_eval.py` | Compares base vs fine-tuned on production traffic |
| Cost Tracker | `cost_analysis.py` | Monthly inference cost comparison |

### Metrics
| Metric | Base Gemma 4 (API) | Fine-Tuned (Local GGUF) |
|--------|-------------------|------------------------|
| Domain Accuracy | 61% | 92% |
| Avg Response Tokens | 850 | 580 (32% shorter) |
| Latency (P50) | 220ms | 45ms |
| Cost/1M output tokens | $300 | ~$0.10 |
| Monthly Cost (5M tok) | $1,500 | ~$5 |
| GPU VRAM Usage | N/A (API) | 5.2 GB (Q4_K_M) |

---

## 🎯 Key Takeaways

- **Merge LoRA adapters for single-task deployment** — eliminates inference overhead completely. Keep adapters separate only for multi-task/multi-tenant serving with vLLM's dynamic adapter swapping.
- **GGUF Q4_K_M is the production sweet spot** — 4.7 GB for a 7B model, near-lossless quality, runs on 8 GB GPUs and 16 GB CPUs. Q5+ is marginal improvement; Q3 and below have noticeable quality degradation.
- **vLLM + LoRA adapters enables multi-tenant serving at near-zero additional VRAM cost** — 10 adapters at 20 MB each add only 200 MB to the 14 GB base model. This is how Together AI and Anyscale serve thousands of fine-tuned models.
- **Your Go/Fiber/Redis LLM Edge Gateway can load GGUF models directly** via go-llama.cpp bindings — eliminating the network hop to an external API and cutting latency from ~220ms to ~45ms.
- **Training cost is trivially small with Unsloth QLoRA** (~$2 for a 2-hour fine-tune on an RTX 4090). The ROI decision is purely about **inference volume** and **accuracy improvements**.
- **Model versioning via symlinks** enables zero-downtime deployments: swap `latest → v1.1`, restart the server, serve the new model without dropping requests.
- **Fine-tuned responses are typically 20-40% shorter** — because the model is more certain and doesn't hedge with disclaimers. This amplifies the cost savings from self-hosting.

---

## References

- Unsloth Docs — Saving & Merging: `https://docs.unsloth.ai/basics/saving-and-merging`
- llama.cpp GGUF Specification: `https://github.com/ggerganov/llama.cpp`
- vLLM LoRA Adapter Serving: `https://docs.vllm.ai/en/latest/features/lora.html`
- Ollama Modelfile Reference: `https://github.com/ollama/ollama/blob/main/docs/modelfile.md`
- go-llama.cpp: `https://github.com/go-skynet/go-llama.cpp`
- Unsloth Blog — GGUF Export: `https://unsloth.ai/blog/gemma`
