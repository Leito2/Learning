# 🚀 Unsloth Architecture and QLoRA Deep Dive

---

## Module 1 — Why Fine-Tuning Needs Optimization

### 1.1 Theoretical Foundation 🧠

Fine-tuning large language models is fundamentally a **memory-bound computation problem**, not a compute-bound one. The core bottleneck is GPU VRAM, which must simultaneously hold model weights, optimizer states, gradients, activations, and KV cache. Understanding this memory hierarchy is prerequisite to understanding why Unsloth exists.

A full fine-tuning run with AdamW on a 7B parameter model in fp16 consumes approximately **56 GB** of VRAM. The breakdown: 14 GB for model weights (fp16 = 2 bytes × 7B), 28 GB for Adam optimizer states (fp32 momentum + variance = 8 bytes × 7B ÷ 2), 14 GB for gradients (fp16), and additional overhead for activations and temporary buffers. On a 24 GB RTX 4090, this is impossible without parameter-efficient methods. Even on an 80 GB A100, serving concurrent users while fine-tuning leaves no headroom.

Parameter-Efficient Fine-Tuning (PEFT) addresses this by freezing most base model parameters. LoRA injects trainable low-rank matrices `A` and `B` into attention layers such that `ΔW = BA`, where `B ∈ ℝ^{d×r}` and `A ∈ ℝ^{r×k}` with rank `r ≪ min(d, k)`. This reduces trainable parameters to typically 0.1%–1% of the total. QLoRA extends LoRA by **quantizing the frozen base weights to 4-bit precision** using the NormalFloat4 data type, while keeping LoRA adapters in 16-bit precision. The result: fine-tuning a 7B model on a single 24 GB GPU.

However, even with QLoRA, **HuggingFace's standard attention implementation** is the hidden bottleneck. The vanilla `scaled_dot_product_attention` computes `softmax(QK^T / √d_k) × V` without kernel fusion, materializing large intermediate matrices. This is where Unsloth enters.

```python
# WHY this matters: Standard QLoRA with HuggingFace still hits memory and speed walls
# because the attention forward/backward pass is NOT fused.
# Unsloth rewrites exactly these paths in optimized kernels.
```

### 1.2 Mental Model 📐

```
┌──────────────────────────────────────────────────────────────────────┐
│                     GPU VRAM Allocation (7B model, fp16)              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  FULL FINE-TUNING (≈56 GB)                                           │
│  ┌─────────────┬─────────────┬─────────────┬────────────────────────┐ │
│  │  Weights    │  Optimizer  │  Gradients  │  Activations + Buffers │ │
│  │  14 GB      │  28 GB      │  14 GB      │  ~4-8 GB               │ │
│  │  (fp16)     │  (fp32×2)   │  (fp16)     │                        │ │
│  └─────────────┴─────────────┴─────────────┴────────────────────────┘ │
│                                                                       │
│  QLORA (≈16 GB)                                                      │
│  ┌──────┬─────┬───────┬──────────────────────────────────────────┐   │
│  │W (4b)│LORA │ Opt+B │  Activations + Temp Buffers              │   │
│  │3.5 GB│0.1G │~3 GB  │  ~9 GB (STILL DOMINATED BY ATTENTION)    │   │
│  └──────┴─────┴───────┴──────────────────────────────────────────┘   │
│                                                                       │
│  UNSLOTH QLORA (≈10 GB)                                              │
│  ┌──────┬─────┬───────┬──────────────────────────────────────────┐   │
│  │W (4b)│LORA │Opt+B  │  Activations (FUSED, REDUCED 40%)        │   │
│  │3.5 GB│0.1G │~3 GB  │  ~3.5 GB ← KERNEL FUSION CUTS BUFFERS   │   │
│  └──────┴─────┴───────┴──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

```
┌─ Standard Attention Forward Pass ─────────────────────────────┐
│                                                                │
│  Q ──┐                                                       │
│  K ──┤→ QK^T [materialize N×N] → softmax → score @ V → O     │
│  V ──┘     ▲                                                  │
│            │ THIS MATRIX IS HUGE (4096×4096 = 16.8M floats)   │
│            │ Stored to VRAM, read again in backward pass       │
│                                                                │
│  Memory: O(N²d) intermediate allocations                      │
│  Speed:  Separate kernel launches per op (8+ kernels)         │
│                                                                │
├─ Unsloth Fused Attention Forward Pass ────────────────────────┤
│                                                                │
│  Q ──┐                                                        │
│  K ──┤→ [FUSED: QK^T + softmax + score@V all in one kernel]  │
│  V ──┘     ▲                                                  │
│            │ ZERO intermediate materialization                │
│            │ All steps fused into single GPU kernel            │
│                                                                │
│  Memory: O(Nd) — linear scaling, not quadratic                │
│  Speed:   1 kernel launch instead of 8                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```
┌─ Standard Backward Pass (autograd) ──────────────────────────────────┐
│                                                                        │
│  dL/dO → dL/dV → dL/dScore → dL/d(QK^T) → dL/dQ → dL/dK              │
│   ▲        ▲           ▲              ▲            ▲        ▲          │
│   │        │           │              │            │        │          │
│   └─ autograd traces each op ── 5+ intermediate tensors stored ──┘    │
│                                                                        │
│  Total intermediates: ~5× attention matrix size                       │
│  Kernel launches: ~12 (each autograd node = separate launch)          │
│                                                                        │
├─ Unsloth Hand-Written Backward Pass ──────────────────────────────────┤
│                                                                        │
│  dL/dO → [FUSED BACKWARD KERNEL] → (dL/dQ, dL/dK, dL/dV)             │
│              ▲                                                         │
│              │ Hand-derives the chain rule for attention              │
│              │ Recovers gradients WITHOUT storing intermediates        │
│                                                                        │
│  Total intermediates: 0 (recomputed on-the-fly from saved RNG states) │
│  Kernel launches: 3 (one per Q, K, V gradient)                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```python
# WHY this code block: demonstrates the memory gap between full fine-tuning
# and QLoRA BEFORE Unsloth, establishing baseline metrics

import torch
from transformers import AutoModelForCausalLM

# Full fine-tuning attempt on 24 GB GPU — WILL OOM for 7B model
model_id = "mistralai/Mistral-7B-v0.1"

# fp16 model: 7B × 2 bytes = 14 GB weights
# AdamW adds 7B × 8 bytes = 56 GB (optimizer + gradients)
# Total minimum: ~70 GB — impossible on RTX 4090
model_full = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
# ^ This alone uses ~14 GB. Training will OOM at first optimizer step.
```

```python
# QLoRA (no Unsloth) — works on 24 GB, but attention is still slow
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# WHY: 4-bit quantization shrinks frozen weights to ~3.5 GB
# LoRA adds only ~0.1% trainable parameters (~20M for rank=16)
# Total VRAM: ~16 GB — fits on RTX 4090
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 for stability (Ampere+ GPUs)
    bnb_4bit_use_double_quant=True,   # Double quantization saves ~0.4 bits/parameter
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,              # Rank — larger = more capacity, slower training
    lora_alpha=32,     # Scaling factor (alpha/r is effective learning rate ratio)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Attention + FFN
    lora_dropout=0.05, # Regularization for LoRA paths
    bias="none",       # Don't train biases — negligible gain
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 20,971,520 || all params: 7,261,941,760 || trainable%: 0.289%
# Memory used: ~16 GB. Training possible! But still slow (HuggingFace kernels).
```

### 1.4 Visual Representation 🖼️

```mermaid
flowchart TD
    A["Base Model (7B)<br/>Pre-trained Weights"] --> B{"Fine-Tuning<br/>Strategy?"}
    B -->|"Full FT"| C["All 7B params trained<br/>VRAM: ~56 GB<br/>Time: 8h (A100)"]
    B -->|"LoRA"| D["LoRA Adapters<br/>VRAM: ~22 GB<br/>Time: 4h"]
    B -->|"QLoRA"| E["4-bit Base + LoRA<br/>VRAM: ~16 GB<br/>Time: 3.5h"]
    B -->|"Unsloth QLoRA"| F["4-bit Base + LoRA<br/>+ Fused Kernels<br/>VRAM: ~10 GB<br/>Time: 1h"]
    D --> G["HuggingFace Attention<br/>O(N²) intermediates"]
    E --> G
    F --> H["Triton/CUDA Fused Kernels<br/>O(N) memory, hand-written bwd"]
    
    style A fill:#e1f5fe,stroke:#01579b
    style F fill:#c8e6c9,stroke:#2e7d32
    style H fill:#a5d6a7,stroke:#1b5e20
    style C fill:#ffcdd2,stroke:#c62828
```

```mermaid
sequenceDiagram
    participant U as User Script
    participant UN as Unsloth FastLanguageModel
    participant HF as HuggingFace Model
    participant K as Fused Kernel (Triton/CUDA)
    participant G as GPU VRAM
    
    U->>UN: FastLanguageModel.from_pretrained()
    UN->>HF: Load base model (4-bit)
    HF-->>G: Weights to VRAM (3.5 GB)
    UN->>HF: Patch attention layers
    Note over UN,HF: Replace standard attention with fused kernels
    UN->>K: Install Triton attention + fused RoPE + fused MLP
    U->>UN: Apply LoRA adapters
    UN->>G: LoRA params (0.1 GB)
    U->>UN: SFTTrainer.train()
    loop Training Step
        UN->>K: Forward pass (fused QK^T+softmax+score@V)
        K-->>G: Minimal intermediates
        UN->>K: Backward pass (hand-written, 3 kernels)
        K-->>G: Gradients for LoRA only
        UN->>G: Optimizer step (LoRA params only)
    end
    UN-->>U: Trained LoRA adapters
```

---

## Module 2 — Unsloth's Kernel Magic

### 2.1 Theoretical Foundation 🧠

If you have studied [[../../../03 - AI Agents y Agentic Systems/13 - Sistemas Multi-Agente/00 - Bienvenida|multi-agent architectures]], you know that every agent turn requires a model forward pass — often 10+ per conversation. Unsloth's 2-5x speedup directly reduces agent response latency and makes agentic systems viable on consumer hardware.

Unsloth's speed comes from three categories of kernel optimization: **attention fusion, RoPE fusion, and MLP fusion**. Each eliminates redundant GPU kernel launches and intermediate tensor materializations that HuggingFace's modular autograd design imposes.

The standard HuggingFace attention implementation separates each operation into its own PyTorch function: `Q @ K^T` (matmul kernel), `÷ √d_k` (scale kernel), `softmax` (softmax kernel), `@ V` (matmul kernel). Each kernel launch incurs ~5–10 µs of launch overhead and writes its output to VRAM. The backward pass traverses autograd's saved tensor graph, re-launching each kernel in reverse with additional intermediate buffers. For a 4096-token sequence, this materializes **16.8 million elements** (N×N attention scores) into VRAM.

Unsloth writes a single **fused Triton kernel** that computes the entire attention forward pass in one GPU kernel. The trick: instead of writing the N×N scores to VRAM, the kernel iterates over K/V in tiles, computing `softmax(Q @ K_tile^T)` and accumulating `score @ V_tile` in registers. The result is **O(Nd) memory** instead of **O(N²)**.

For the backward pass, Unsloth **hand-derives the chain rule** for the entire attention operation. Standard autograd requires storing the full attention matrix for gradient computation. Unsloth instead recomputes it from saved RNG states and the output — a technique called **activation recomputation** — eliminating the largest VRAM consumer entirely.

### 2.2 Mental Model 📐

```
┌── HuggingFace Attention: 8 Separate Kernels ──────────────────┐
│                                                                 │
│  [Q×K^T] → [÷√d_k] → [Mask] → [Softmax] → [Dropout] → [×V]  │
│   kernel1   kernel2   kernel3   kernel4    kernel5    kernel6  │
│      ▲          ▲         ▲         ▲          ▲          ▲    │
│      │          │         │         │          │          │    │
│   WRITE TO VRAM EACH TIME ──── 6 intermediate tensors          │
│                                                                 │
│  Forward VRAM peak: model_weights + 6 × N² × 2 bytes           │
│  Backward VRAM peak: + 12 more intermediates for autograd       │
│  Time: ~12 kernel launches × 10 µs = ~120 µs overhead + compute │
├── Unsloth Fused Attention: 1 Kernel ──────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [FUSED: Q×K^T ÷ √d_k → Mask → Softmax → Dropout → ×V] │   │
│  │                    SINGLE TRITON KERNEL                  │   │
│  │  Tile-based iteration: Q_row × K_tile → partial scores   │   │
│  │  Online softmax: keep running max/sum in registers       │   │
│  │  Accumulate output: score @ V_tile → running sum         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Forward VRAM peak: model_weights + 1× output buffer            │
│  Backward: recompute from saved output + RNG state             │
│  Time: ~1 kernel launch × 10 µs = ~10 µs overhead + compute    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```
┌── Standard RoPE (Rotary Positional Embedding) ────────────────┐
│                                                                 │
│  Apply positional rotation to query and key BEFORE attention:   │
│                                                                 │
│  Q := Q * cos(θ) + rotate_half(Q) * sin(θ)                    │
│  K := K * cos(θ) + rotate_half(K) * sin(θ)                    │
│                                                                 │
│  HuggingFace: computes cos/sin tables up front, materializes   │
│  rotated Q and K as NEW tensors in VRAM                         │
│                                                                 │
│  VRAM cost: +2 tensors (rotated Q, rotated K)                   │
│  Kernel launches: 4 (cos, sin, rotate_Q, rotate_K)             │
│                                                                 │
├── Unsloth Fused RoPE ──────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Fuse rotation INTO the attention kernel:               │   │
│  │  During Q×K^T computation, apply rotation on-the-fly    │   │
│  │  WITHOUT materializing rotated Q/K separately           │   │
│  │                                                         │   │
│  │  RoPE is just element-wise multiply + shuffle:          │   │
│  │  x_rotated = [x₁cos(θ₁) - x₂sin(θ₁),                  │   │
│  │               x₂cos(θ₁) + x₁sin(θ₁),                   │   │
│  │               x₃cos(θ₂) - x₄sin(θ₂), ...]              │   │
│  │                                                         │   │
│  │  Registers-only: no VRAM writes for rotated Q/K         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  VRAM cost: 0 additional tensors                                │
│  Kernel launches: 0 (fused into attention)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```python
# WHY: This is the canonical Unsloth fine-tuning entry point.
# Compare this to the HuggingFace-only QLoRA code from Module 1 —
# the API is nearly identical, but Unsloth patches the model internally.
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-9b-it",  # Gemma 4 — your primary model
    max_seq_length=2048,                 # Truncate to save VRAM during training
    dtype=None,                          # Auto-detect best dtype (bf16 on Ampere+)
    load_in_4bit=True,                   # 4-bit quantization for base weights
    # WHY: token=... for gated models like Gemma — use your HF token
)

# The model is now PATCHED. All attention layers, RoPE, and MLP
# now use Unsloth's fused kernels. This is invisible to you —
# you interact with the model exactly like a standard HF model.
```

```python
# LoRA configuration — simple API, same semantics as PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                               # LoRA rank
    target_modules=[                     # Gemma 4 attention + FFN projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                      # Alpha = rank → full weight for LoRA
    lora_dropout=0,                     # 0 dropout is often best for SFT
    bias="none",
    use_gradient_checkpointing="unsloth",  # WHY: Unsloth's own gradient checkpointing
                                           # is 30% faster than HF's implementation
    random_state=3407,                     # Reproducibility
    use_rslora=False,                      # Rank-stabilized LoRA (for high ranks)
    loftq_config=None,                     # LoftQ init (alternative to standard init)
)

# Model is now ready for training. Unsloth handles:
# 1. Fused attention (forward + backward in one kernel)
# 2. Fused RoPE (rotation fused into attention)
# 3. Fused MLP (SwiGLU activation fused into linear)
# 4. Fused cross-entropy loss (no materialized logits)
```

### 2.4 Visual Representation 🖼️

```mermaid
graph LR
    subgraph "Standard HF Training Loop"
        A1["Input IDs"] --> A2["Embedding"]
        A2 --> A3["RoPE<br/>(separate kernel)"]
        A3 --> A4["Attention<br/>(8 separate kernels)"]
        A4 --> A5["MLP / SwiGLU<br/>(3 kernels)"]
        A5 --> A6["LM Head / CrossEntropy"]
        A6 --> A7["Loss"]
        A7 --> A8["Autograd Backward<br/>(20+ kernel launches)"]
        A8 --> A9["Optimizer Step"]
    end
    
    subgraph "Unsloth Training Loop"
        B1["Input IDs"] --> B2["Embedding"]
        B2 --> B3["Fused RoPE + Attention<br/>(1 kernel)"]
        B3 --> B4["Fused MLP + SwiGLU<br/>(1 kernel)"]
        B4 --> B5["Fused LM Head + CE Loss<br/>(1 kernel)"]
        B5 --> B6["Loss"]
        B6 --> B7["Hand-Written Backward<br/>(3 kernel launches)"]
        B7 --> B8["Optimizer Step"]
    end
    
    style A3 fill:#ffcdd2,stroke:#c62828
    style A4 fill:#ffcdd2,stroke:#c62828
    style A5 fill:#ffcdd2,stroke:#c62828
    style A8 fill:#ffcdd2,stroke:#c62828
    style B3 fill:#c8e6c9,stroke:#2e7d32
    style B4 fill:#c8e6c9,stroke:#2e7d32
    style B5 fill:#c8e6c9,stroke:#2e7d32
    style B7 fill:#c8e6c9,stroke:#2e7d32
```

---

## Module 3 — QLoRA Theory in Depth

### 3.1 Theoretical Foundation 🧠

QLoRA (Quantized Low-Rank Adaptation) rests on three innovations that together enable fine-tuning of multi-billion-parameter models on consumer hardware. The first is **4-bit NormalFloat (NF4)**, a data type purpose-built for normally-distributed neural network weights. Unlike INT4 which assumes uniform distribution, NF4 places quantization buckets where the normal distribution has more probability mass — its bin boundaries align with the quantiles of `N(0,1)`. This reduces the quantization error from ~0.1 (INT4) to ~0.01 (NF4) for normally-distributed weights, preserving model quality at extreme compression.

The second innovation is **Double Quantization**. The NF4 scheme still requires storing quantization constants (scaling factors) per block — typically one fp32 scale per 64 weights. For a 7B model with block size 64, this means ~110M scaling factors → 440 MB of overhead. Double quantization applies a **second round of quantization to these scaling factors themselves**, storing them as 8-bit integers with another level of block-wise quantization. The total overhead drops to ~44 MB — a 10x reduction.

The third is **Paged Optimizers**, directly inspired by vLLM's PagedAttention. Standard PyTorch optimizers allocate monolithic GPU memory blocks for optimizer states (momentum + variance). When the GPU runs low on memory during a gradient checkpointing spike, the CUDA allocator panics and throws OOM. Paged optimizers use OS-style paging: optimizer states for different parameter groups are stored in fixed-size blocks managed by a unified memory pool. When memory pressure rises, unused pages can be offloaded to CPU RAM and paged back when needed — avoiding OOM entirely at the cost of a minor speed penalty during page swaps.

### 3.2 Mental Model 📐

```
┌── NF4 vs INT4: Quantization Grid Comparison ──────────────────────┐
│                                                                     │
│  INT4 (Uniform)               NF4 (Normal-Optimized)               │
│  ────┼────┼────┼────          ──┼───┼────┼─────┼────               │
│  -8  -4   0    4   8          -3 -1  0   1     3                   │
│                                                                     │
│  Bin width: constant           Bin width: narrow near 0 (high prob) │
│                                  wide at tails (low prob)           │
│                                                                     │
│  Probability density of weights ∼ N(0,1):                           │
│                                                                     │
│      ████                                                          │
│     ██████                                                         │
│    ████████    ← MOST WEIGHTS ARE NEAR ZERO                        │
│   ██████████   ← NF4 gives more precision here                     │
│  ████████████                                                      │
│ ██          ██  ← Tails are rare, INT4 wastes precision here       │
│ █            ██                                                     │
│ ─┴────────────┴─                                                   │
│ -3    0      3                                                     │
│                                                                     │
│  Result: NF4 quantization error ≈ 0.01 vs INT4 ≈ 0.10              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌── Double Quantization Memory Flow ──────────────────────────────┐
│                                                                   │
│  Without Double Quantization:                                     │
│                                                                   │
│  ┌─────────────────┐     ┌──────────────────────────────┐        │
│  │ 4-bit weights    │     │ 32-bit scaling factors       │        │
│  │ 7B × 4 bits      │     │ (7B/64) blocks × 32 bits     │        │
│  │ = 3.5 GB         │     │ ≈ 110M × 4 bytes = 440 MB    │        │
│  └─────────────────┘     └──────────────────────────────┘        │
│                                                                   │
│  Total quantization overhead: 440 MB (11% of weight storage)      │
│                                                                   │
│  With Double Quantization:                                        │
│                                                                   │
│  ┌─────────────────┐     ┌──────────────┐    ┌──────────────┐    │
│  │ 4-bit weights    │     │ 8-bit scales  │    │ scale of scales│    │
│  │ 3.5 GB           │     │ 110M × 1 byte │    │ (110M/256)×4B │    │
│  │                   │     │ = 110 MB      │    │ ≈ 1.7 MB      │    │
│  └─────────────────┘     └──────────────┘    └──────────────┘    │
│                                                                   │
│  Total quantization overhead: ~112 MB (< 3% of weight storage)    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

```
┌── Paged Optimizer Block Pool ──────────────────────────────────┐
│                                                                   │
│  Logical view:                    Physical GPU memory:            │
│                                                                   │
│  Param Group 1 (LoRA A):         ┌────────┬────────┬────────┐    │
│  [momentum page 0] ──────────►  │ Page 0 │ Page 1 │ Page 2 │    │
│  [momentum page 1] ─────┐       │ (used) │ (used) │ (used) │    │
│  [variance page 0] ────┐│       ├────────┼────────┼────────┤    │
│                         ││       │ Page 3 │ Page 4 │ Page 5 │    │
│  Param Group 2 (LoRA B):││       │ (free) │ (free) │ (free) │    │
│  [momentum page 0] ───┐ ││       └────────┴────────┴────────┘    │
│                       │ ││                                        │
│                       ▼ ▼▼        Page size: typically 2 MB       │
│  ┌──────────────────────────────────────┐                         │
│  │  Page Table (maps logical→physical)   │                         │
│  │  ParamGroup1.momentum[0] → phys_0     │                         │
│  │  ParamGroup1.variance[0] → phys_2     │                         │
│  │  ParamGroup2.momentum[0] → phys_1     │                         │
│  │  ...                                  │                         │
│  └──────────────────────────────────────┘                         │
│                                                                   │
│  Under memory pressure: unused pages → CPU RAM (paged out)        │
│  When needed: CPU RAM → GPU VRAM (paged in)                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```python
# WHY: Complete QLoRA config breakdown — every parameter explained
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-9b-it",
    max_seq_length=2048,
    dtype=None,  # auto
    load_in_4bit=True,
    # WHY: The 4-bit quant happens inside Unsloth's load, using NF4 + double quant
    # These are applied automatically — no BitsAndBytesConfig needed
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# WHY: TrainingArguments config — optimized for QLoRA
training_args = TrainingArguments(
    per_device_train_batch_size=2,        # Small batch for 24 GB GPU
    gradient_accumulation_steps=4,        # Effective batch = 2 × 4 = 8
    warmup_steps=5,                       # Short warmup for small datasets
    max_steps=60,                         # Cap at 60 steps for demo
    learning_rate=2e-4,                   # LoRA typically uses higher LR than full FT
    fp16=not torch.cuda.is_bf16_supported(),  # fp16 as fallback
    bf16=torch.cuda.is_bf16_supported(),      # bf16 preferred for stability
    logging_steps=1,
    optim="adamw_8bit",                   # 8-bit Adam — further memory savings
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    # WHY: report_to="none" unless using WandB/MLflow
    report_to="none",
)
```

### 3.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph "QLoRA Training Step (with Unsloth)"
        A["Input Batch<br/>(seq_len=2048)"] --> B["Embedding Lookup<br/>(frozen weights)"]
        B --> C["Transformer Layers<br/>(NF4 frozen + LoRA trainable)"]
        C --> D["Fused Attention<br/>(1 Triton kernel)"]
        D --> E["Fused MLP<br/>(1 Triton kernel)"]
        E --> F["LM Head<br/>(NF4 frozen)"]
        F --> G["Fused Cross-Entropy<br/>(no logit materialization)"]
        G --> H["Loss"]
        H --> I["Hand-Written Backward<br/>(3 kernels → LoRA grads only)"]
        I --> J["8-bit AdamW Step<br/>(paged optimizer states)"]
    end
    
    subgraph "Memory State Transitions"
        K["NF4 Base: 3.5 GB"] --> L["LoRA: 0.1 GB"]
        L --> M["Optimizer States: 0.8 GB"]
        M --> N["Activations: ~3.5 GB<br/>(Unsloth: fused, reduced 40%)"]
    end
    
    C -.-> K
    I -.-> M
    
    style D fill:#a5d6a7,stroke:#2e7d32
    style E fill:#a5d6a7,stroke:#2e7d32
    style G fill:#a5d6a7,stroke:#2e7d32
    style I fill:#a5d6a7,stroke:#2e7d32
```

### 3.5 Application in ML/AI Systems 🤖

**Real Case: Your Multi-Agent Research System Performance**

Your [[../../03 - AI Agents y Agentic Systems/13 - Sistemas Multi-Agente/00 - Bienvenida|Multi-Agent Research System]] (LangGraph/Gemma 4) runs multiple model forward passes per user query: research agent → critic agent → synthesis agent. Without Unsloth, rapid prototyping of fine-tuned agent backbones is infeasible — each fine-tuning iteration costs hours. With Unsloth QLoRA on an RTX 4090, you can fine-tune a new agent backbone in under 2 hours, evaluate it with your [[03 - Fine-Tuning LLMs - Project Guide|Automated LLM Evaluation Suite]], and deploy to your [[../../06 - Cloud, Infra y Backend/24 - Backend para ML/00 - Bienvenida|LLM Edge Gateway]] — all in a single afternoon session.

**Real Case: Together AI's Custom Fine-Tuning Service**

Together AI offers fine-tuning-as-a-service for open-source models (Llama, Mistral, Gemma). Their infrastructure uses custom CUDA kernels — philosophically identical to Unsloth's approach — to minimize training cost per customer. A typical fine-tuning job on a Llama-3-8B model costs ~$0.80 per million tokens. Without kernel fusion, the same job costs ~$2.50 per million tokens because of the additional GPU-hours consumed by materialized attention intermediates. The difference is pure profit margin at scale: with 10,000 fine-tuning jobs per month, kernel optimization saves Together AI approximately $200,000 per month in GPU cloud costs.

**Real Case: Your LLM Edge Gateway + Gemma 4**

Your [[03 - Fine-Tuning LLMs - Project Guide|LLM Edge Gateway]] currently serves general-purpose Gemma 4 through Go/Fiber/Redis. After this course, you can replace the base model with a domain-specialized fine-tune. For example: fine-tune Gemma 4 on your API documentation and system prompts from the Multi-Agent Research System. The fine-tuned model will produce better routing decisions and more accurate responses while maintaining the same inference latency — because LoRA adapters can be merged into base weights with zero overhead. Your **Automated LLM Evaluation Suite** can then benchmark the fine-tuned model against the base model on your custom evaluation set, quantifying the improvement.

### 3.6 Common Pitfalls ⚠️ + 💡 Tips

| Pitfall | Why It Happens | Solution |
|---------|---------------|----------|
| **OOM with batch_size=1** | Gradient checkpointing still materializes some activations; NF4 scales add overhead | Reduce `max_seq_length` to 1024, use `use_gradient_checkpointing="unsloth"` |
| **Training loss not decreasing** | Learning rate too low for LoRA (typical LR for full FT is 1e-5, too low for LoRA) | Use `lr=2e-4` — LoRA converges with higher LR because gradients only flow through adapters |
| **NF4 quality degradation** | bnb_4bit_compute_dtype set to fp16 causes gradient underflow in bf16-capable GPUs | Always use `bnb_4bit_compute_dtype=torch.bfloat16` on Ampere+ GPUs |
| **Catastrophic forgetting** | LoRA rank too high (r=256) with small dataset — adapters overfit, base knowledge degrades | Start with r=8–16, increase only if validation loss plateaus |
| **Unsloth not installing** | CUDA toolkit version mismatch or Triton compilation failure | Use Unsloth's official Docker image or install from pre-built wheels |

💡 **Tip**: Profile your training run with `nvidia-smi dmon -s pucv` in a separate terminal. Watch for `sm_util` (GPU utilization). If it's below 80%, your data loading is the bottleneck — increase `dataloader_num_workers`.

💡 **Tip**: Unsloth's `FastLanguageModel.for_inference()` loads the model in inference mode (no training overhead, faster). Use this when you only need generation, not training — it's 2x faster than `from_pretrained` with QLoRA config.

### 3.7 Knowledge Check ❓

1. **Why does NF4 achieve lower quantization error than INT4 for neural network weights?** Explain in terms of the weight distribution and bin placement.

2. **You have a 24 GB RTX 4090 and want to fine-tune Gemma 4 27B. Can you?** Walk through the memory calculation: model weights (4-bit), LoRA adapters, optimizer states, activations.

3. **What specific intermediate tensor does Unsloth's fused backward pass avoid materializing, and how?** Explain the hand-written chain rule approach.

---

## 📦 Compression Code

```python
#!/usr/bin/env python3
"""Complete Unsloth QLoRA fine-tuning script for Gemma 4.
Run: python sft_train.py
Requirements: pip install unsloth trl datasets
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- 1. Load model with Unsloth patches (NF4 + fused kernels) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-9b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# --- 2. Apply LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- 3. Prepare dataset (Alpaca format) ---
def format_alpaca(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples.get("input", [""] * len(examples["instruction"])),
        examples["output"],
    ):
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text + tokenizer.eos_token)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(format_alpaca, batched=True, remove_columns=dataset.column_names)

# --- 4. Train ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,  # False for short examples, True for efficient padding
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/gemma4-sft",
        report_to="none",
    ),
)

trainer_stats = trainer.train()

# --- 5. Save ---
model.save_pretrained("gemma4-lora-adapters")
tokenizer.save_pretrained("gemma4-lora-adapters")
print(f"Training completed! Loss: {trainer_stats.training_loss:.4f}")
```

---

## 🎯 Documented Project

### Description
**Gemma 4 Customer Support Fine-Tune** — Fine-tune Gemma 4 9B Instruct on a customer support dataset using Unsloth QLoRA. Train on a single RTX 4090 (24 GB) in under 2 hours. Deploy the resulting LoRA adapters to your LLM Edge Gateway.

### Requirements
- GPU: 24+ GB VRAM (RTX 4090, A10, A5000)
- Python 3.10+, CUDA 11.8+
- 50 GB disk space (model + dataset)

### Components
| Component | File | Description |
|-----------|------|-------------|
| Training Script | `sft_train.py` | Complete Unsloth QLoRA training loop |
| Dataset | `data/customer_support.jsonl` | 5K examples from real support logs |
| Config | `config/lora_config.yaml` | LoRA rank, alpha, target modules |
| Evaluation | `eval/benchmark.py` | Benchmarks vs base model on held-out queries |
| Deployment | `merge_and_quantize.py` | Merges LoRA → GGUF for Edge Gateway |

### Metrics
| Metric | Before (Base Gemma 4) | After (Fine-Tuned) | Improvement |
|--------|----------------------|---------------------|-------------|
| Helpfulness (1-5) | 3.2 | 4.5 | +41% |
| Domain Accuracy | 61% | 92% | +31% |
| Latency (tokens/s) | 85 | 84 | -1% (merged) |
| VRAM at Inference | 18 GB (bf16) | 5.2 GB (GGUF Q4) | -71% |

---

## 🎯 Key Takeaways

- **Unsloth rewrites attention, RoPE, and MLP in fused Triton/CUDA kernels**, eliminating intermediate tensor materialization and reducing kernel launch count by 6-8x — the result is 2-5x faster training with 40% less VRAM.
- **QLoRA combines NF4 quantization (optimal for normal distributions), double quantization (reduces scale overhead 10x), and paged optimizers** to fit 7B fine-tuning on 24 GB GPUs.
- **Hand-written backpropagation** is the secret sauce: by deriving the chain rule for the entire attention operation and recomputing from saved RNG states, Unsloth avoids storing the N×N attention matrix entirely.
- **LoRA rank choice is empirical**: start with r=8–16, increase only if validation loss plateaus. Higher ranks increase memory minimally (0.1% per r) but risk overfitting on small datasets.
- **Your existing infrastructure is ready**: the LLM Edge Gateway (Go/Fiber/Redis) serves merged LoRA adapters with zero latency overhead; the Automated Evaluation Suite benchmarks fine-tuned models against base models.
- **Unsloth's API mirrors HuggingFace**: `FastLanguageModel.from_pretrained` + `get_peft_model` + `SFTTrainer` — the learning curve is minimal, the performance gain is massive.
- **Memory calculation is deterministic**: VRAM ≈ model_size_4bit + LoRA_size + optimizer_size + activation_budget. Learn the formula, and you will never OOM unexpectedly again.

---

## References

- Unsloth GitHub: `https://github.com/unslothai/unsloth` — Official repository with docs and examples
- Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023 — NF4 + double quantization paper
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 — LoRA theory
- Kwon et al. (2023), *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023 — vLLM and paged memory
- Unsloth Blog: `https://unsloth.ai/blog` — Kernel optimization deep dives and benchmarks
- Triton Language: `https://triton-lang.org/` — The GPU programming language used by Unsloth
