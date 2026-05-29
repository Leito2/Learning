# 📐 04 - Back-of-Envelope Estimation — Throughput, Latency, Storage, and Cost

## 🎯 Learning Objectives

- Memorize the 15 essential numbers for ML system design: GPU HBM bandwidth (3.35 TB/s for H100), network latencies (same-AZ 1ms, cross-region 50ms), model sizes (1B params = 2GB FP16 / 1GB INT8), inference throughput (70B ≈ 20 tok/s on H100), and storage bandwidths (SSD 500MB/s, NVMe 3.5GB/s)
- Apply LaTeX estimation templates for the four core dimensions of ML system design: storage, throughput, latency, and cost — deriving orders of magnitude from first principles
- Solve three graded practice problems with full step-by-step solutions: "Design RAG for 1B documents," "How many GPUs for 10K req/s Llama-70B with P99 < 200ms?", and "Monthly cost to host 100K-user recommendation system"
- Demonstrate back-of-envelope work fluently in interviews — the difference between guessing (❌) and deriving (✅) is the difference between L4 and L6 offers

## Introduction

Back-of-envelope estimation is the most undervalued skill in ML system design. An interviewer asks "How many GPUs do we need to serve 10,000 requests per second with Llama-70B at P99 < 200ms?" The weak candidate guesses "maybe 50?" The strong candidate writes on the whiteboard:

$$\text{GPUs} = \frac{\text{RPS} \times \text{avg\_output\_tokens}}{\text{tok/s per GPU}} \times \frac{1}{1 - \text{overhead}}$$

$$= \frac{10{,}000 \times 200}{20} \times \frac{1}{0.7} = 142{,}857 \approx 143 \text{ GPUs}$$

Then explains: "That's ~143 H100 GPUs for throughput. But the P99 < 200ms constraint adds a latency dimension — at 20 tok/s per GPU, 200 output tokens take 10 seconds on a single GPU, so we need batching and maybe 10× more GPUs to meet the latency SLA. Let me refine..."

The interviewer sees structured reasoning, not a memorized number. That is the difference between a senior engineer and someone who read the system design primer.

The term *back-of-envelope* dates to Fermi problems — named after Enrico Fermi, who estimated the yield of the Trinity nuclear test by dropping scraps of paper and measuring how far the shockwave moved them. The principle is the same for ML: you don't need exact numbers; you need the right order of magnitude, derived from first principles, to catch design errors before you spend $1M on GPUs. This note gives you the numbers, the templates, and the practice to do Fermi estimation for ML systems in a 45-minute interview.

---

## Module 1: The Numbers — What to Memorize and Why

### 1.1 Theoretical Foundation 🧠

Effective back-of-envelope estimation requires a set of memorized reference numbers. You derive system-level estimates by combining these numbers with simple multiplication and division. The numbers fall into five categories:

**GPU and Hardware Numbers** (the most important for ML):

| Quantity | Value | Why It Matters |
|----------|-------|----------------|
| H100 HBM3e bandwidth | 3.35 TB/s | Upper bound on data movement for inference/training |
| H100 FP16 TFLOPS | 989 TFLOPS | Max compute throughput (rarely achieved) |
| H100 VRAM | 80 GB | How much model + KV cache fits on one GPU |
| NVMe SSD bandwidth | 3.5 GB/s (read) | Model loading time, training data streaming |
| SATA SSD bandwidth | 500 MB/s | Cheaper storage for warm tier |
| PCIe 5.0 x16 bandwidth | 64 GB/s | CPU ↔ GPU transfer bottleneck |
| NVLink bandwidth | 900 GB/s (H100) | GPU ↔ GPU transfer for tensor parallelism |

**Network Latency Numbers** (critical for distributed systems):

| Path | Latency | Use Case |
|------|---------|----------|
| Same-AZ (intra-zone) | <1 ms | Inference server → feature store, Redis cache |
| Cross-AZ (inter-zone) | 1-3 ms | Replicated feature stores, quorum writes |
| Cross-region (US East → US West) | 50-80 ms | Global load balancing, disaster recovery |
| Cross-region (US → Europe) | 80-120 ms | Multi-region deployment, CDN miss to origin |
| Cross-region (US → Asia-Pacific) | 120-200 ms | Global user serving |

**Model Size Numbers**:

| Model Scale | FP16 Size | INT8 Size | INT4 Size | Example |
|-------------|-----------|-----------|-----------|---------|
| 1B params | 2 GB | 1 GB | 0.5 GB | TinyLlama |
| 7B params | 14 GB | 7 GB | 3.5 GB | Mistral-7B, Llama-3-8B |
| 13B params | 26 GB | 13 GB | 6.5 GB | Llama-2-13B |
| 70B params | 140 GB | 70 GB | 35 GB | Llama-3-70B |
| 405B params | 810 GB | 405 GB | 202 GB | Llama-3.1-405B |
| 1T params | 2 TB | 1 TB | 500 GB | Future scale |

General formula for model size:

$$S_{\text{model}} = N_{\text{params}} \times B_{\text{precision}}$$

Where $B_{\text{precision}}$ is bytes per parameter: 2 for FP16/BF16, 1 for INT8, 0.5 for INT4. Add 20-30% overhead for optimizer states (training only) and metadata.

**Inference Throughput Numbers** (prefill + decode, batch size = 1):

| Model | Hardware | Throughput (tok/s) | Context |
|-------|----------|--------------------|----------|
| Llama-7B | H100, FP16 | ~80 tok/s | 2048 |
| Llama-13B | H100, FP16 | ~45 tok/s | 2048 |
| Llama-70B | H100, FP16 | ~20 tok/s | 2048 |
| Llama-70B | H100, INT8 | ~35 tok/s | 2048 |
| Llama-70B | 4×H100 (TP=4) | ~60 tok/s | 2048 |
| Llama-3.1-405B | 8×H100 (TP=8) | ~15 tok/s | 2048 |

Note: These are single-request sequential throughput numbers. With continuous batching (vLLM, SGLang), effective throughput increases 5-20× because the GPU processes multiple requests in parallel during the decode phase.

**Storage and Cost Numbers**:

| Storage Tier | Cost | Bandwidth | Latency |
|-------------|------|-----------|---------|
| GPU VRAM (HBM3e) | ~$720/GB/month | 3.35 TB/s | ~1 μs |
| System RAM (DDR5) | ~$4/GB/month | 100 GB/s | ~100 ns |
| NVMe SSD | ~$0.10/GB/month | 3.5 GB/s | ~100 μs |
| S3 Standard | ~$0.023/GB/month | 500 MB/s | ~100 ms |
| S3 Glacier | ~$0.004/GB/month | Minutes | Hours |
| H100 GPU (on-demand) | ~$3.00/GPU-hour | — | — |
| H100 GPU (reserved 1yr) | ~$1.50/GPU-hour | — | — |
| H100 GPU (reserved 3yr) | ~$1.00/GPU-hour | — | — |

### 1.2 Mental Model 📐

```
┌─── Back-of-Envelope Number Hierarchy ───────────────────────────┐
│                                                                   │
│  Layer 1: Hardware Primitives (MEMORIZE THESE)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ H100 VRAM: 80GB       NVMe: 3.5 GB/s        HBM: 3.35 TB/s │ │
│  │ Same-AZ: 1ms           Cross-region: 50ms    PCIe5: 64 GB/s │ │
│  │ 1B params: 2GB FP16   70B tok/s: 20          GPU cost: $3/hr│ │
│  └─────────────────────────────────────────────────────────────┘ │
│       │                                                            │
│       ▼                                                            │
│  Layer 2: Derived Quantities (COMPUTE FROM L1)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Model fit on 1 GPU: VRAM / (params × bytes_per_param)       │ │
│  │ Model load time: size / NVMe_bandwidth                       │ │
│  │ KV cache per token: 2 × layers × d_model × bytes_per_elem   │ │
│  │ GPU-hours / month: GPUs × 730 hours                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│       │                                                            │
│       ▼                                                            │
│  Layer 3: System Estimates (DERIVE FROM L1+L2)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Storage: docs × tokens × dimension × bytes_per_elem         │ │
│  │ Throughput: GPUs × tok/s / avg_output_tokens                 │ │
│  │ Latency: TTFT + output_tokens × time_per_token              │ │
│  │ Cost: GPU_hours × price × 730 + storage × price              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 Knowledge Check ❓

1. How many Llama-70B (INT8) model instances fit on a single H100 (80GB) GPU? What about INT4?
2. What is the minimum latency for a feature store read in same-AZ vs cross-region? How does this affect model SLA budget?
3. A training dataset is 10TB on S3. How long does it take to copy to a local NVMe drive? Is this acceptable for a 1-hour training job?

---

## Module 2: Estimation Templates — Storage, Throughput, Latency, Cost

### 2.1 Theoretical Foundation 🧠

Every ML system design interview requires estimating four dimensions. The templates below are the LaTeX derivations you should be able to reproduce on a whiteboard.

**Storage Estimation:**

For embedding-based systems (RAG, recommendation, semantic search), storage is dominated by the vector index:

$$S_{\text{vectors}} = N_{\text{docs}} \times d_{\text{dim}} \times b_{\text{elem}}$$

Where $N_{\text{docs}}$ is the number of documents, $d_{\text{dim}}$ is the embedding dimension, and $b_{\text{elem}}$ is bytes per element (4 for float32, 2 for float16, 1 for int8).

For the full index (including graph structure for HNSW):

$$S_{\text{index}} = S_{\text{vectors}} \times (1 + M \times b_{\text{edge}})$$

Where $M$ is the HNSW connectivity parameter (typically 16-64) and $b_{\text{edge}}$ is bytes per graph edge (~4 for int32 IDs). The index overhead is typically 100-200% of raw vector storage.

For LLM KV cache:

$$S_{KV} = 2 \times L \times d_{\text{model}} \times b_{\text{param}} \times T_{\text{total}}$$

Where $L$ is the number of layers, $d_{\text{model}}$ is the model dimension, $b_{\text{param}}$ is bytes per parameter (2 for FP16), and $T_{\text{total}}$ is the total tokens (prompt + generated). The factor of 2 accounts for both keys and values.

**Throughput Estimation:**

For LLM inference with continuous batching:

$$R_{\text{max}} = \frac{N_{\text{GPU}} \times R_{\text{tok/s/GPU}}}{L_{\text{output}}}$$

Where $R_{\text{max}}$ is max requests per second, $N_{\text{GPU}}$ is GPU count, $R_{\text{tok/s/GPU}}$ is tokens per second per GPU, and $L_{\text{output}}$ is average output tokens per request.

The key insight: throughput is normalized by output length because each output token requires a full forward pass. Short outputs → high request throughput. Long outputs → low request throughput.

**Latency Estimation:**

End-to-end inference latency is the sum of time to first token (TTFT, or prefill) and generation time:

$$L_{\text{total}} = T_{\text{prefill}} + T_{\text{output}} \times L_{\text{output}}$$

Where $T_{\text{prefill}}$ is the time to process the prompt (prefill phase, processes all prompt tokens in parallel): $T_{\text{prefill}} \approx 0.5 - 2$ seconds for 2000 tokens on H100. $T_{\text{output}}$ is time per output token: for 70B llama, $T_{\text{output}} \approx 50$ ms (20 tok/s = 50 ms per token).

For batched inference, P99 latency increases with batch size because tokens from different requests arrive at different times. With continuous batching (vLLM), the latency impact is amortized across the batch.

**Cost Estimation:**

Total monthly cost has two components: compute (GPU-hours) and storage (GB-months). The monthly GPU cost:

$$C_{\text{compute}} = N_{\text{GPU}} \times P_{\text{GPU-hour}} \times 730$$

Where $P_{\text{GPU-hour}}$ is the on-demand or reserved GPU price. Add 20-30% for networking, load balancers, and other infrastructure.

$$C_{\text{total}} = C_{\text{compute}} + C_{\text{storage}} + C_{\text{network}}$$

### 2.2 Mental Model 📐

```
┌─── Four Estimation Dimensions — The Interview Loop ─────────────┐
│                                                                   │
│  For every system design question, cycle through these four:      │
│                                                                   │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ STORAGE │ →  │THROUGHPUT│ →  │ LATENCY  │ →  │   COST   │   │
│  │         │    │          │    │          │    │          │   │
│  │ How     │    │ How many │    │ What's   │    │ How much │   │
│  │ much    │    │ requests │    │ the end- │    │ does it  │   │
│  │ data?   │    │ can we   │    │ to-end   │    │ cost per │   │
│  │         │    │ serve?   │    │ latency? │    │ month?   │   │
│  └────┬────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘   │
│       │              │               │               │          │
│       ▼              ▼               ▼               ▼          │
│  S = N × D × B  R = GPUs × tok/s   L = Σ components  C = GPU × hrs│
│                              ─────────                  + storage │
│                            output length                         │
│                                                                   │
│  ❌/✅: The difference in an interview:                           │
│  ❌ "I think we need about 100 GPUs"                              │
│  ✅ "At 20 tok/s per H100, 200 avg output tokens per request,     │
│      10000 RPS requires 10000x200/20 = 100K GPUs for throughput.  │
│      But with continuous batching at 10x effective throughput,   │
│      that's 10K GPUs. At $1.50/GPU-hr reserved, that's           │
│      $10.95M/month. Can we reduce output tokens or use a          │
│      smaller model?"                                              │
└───────────────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```python
"""
estimation_templates.py — Reusable back-of-envelope estimation functions.
Use these as a reference during interviews or when sizing real systems.
"""

from dataclasses import dataclass
from typing import Optional
import math


# ═══ Model Size Estimation ═══

@dataclass
class ModelSizeEstimate:
    params_billions: float
    fp16_size_gb: float
    int8_size_gb: float
    int4_size_gb: float
    kv_cache_per_token_mb: float
    fits_on_h100: bool
    gpus_needed: int


def estimate_model_size(
    params_billions: float,
    num_layers: int,
    hidden_dim: int,
    vocab_size: int = 128_000,
) -> ModelSizeEstimate:
    """Estimate model memory footprint at different precisions."""
    bytes_fp16 = params_billions * 1e9 * 2
    bytes_int8 = params_billions * 1e9 * 1
    bytes_int4 = params_billions * 1e9 * 0.5

    # KV cache per token: 2 (K+V per layer) x layers x hidden_dim x 2 bytes (FP16)
    kv_cache_bytes = 2 * num_layers * hidden_dim * 2
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    # H100 has 80GB VRAM; leave 5GB for CUDA context, batch overhead
    usable_vram_gb = 75.0

    # GPUs needed for tensor parallelism (FP16 model)
    gpus_needed = math.ceil((bytes_fp16 / 1e9) / (usable_vram_gb * 0.85))

    return ModelSizeEstimate(
        params_billions=params_billions,
        fp16_size_gb=round(bytes_fp16 / 1e9, 1),
        int8_size_gb=round(bytes_int8 / 1e9, 1),
        int4_size_gb=round(bytes_int4 / 1e9, 1),
        kv_cache_per_token_mb=round(kv_cache_mb, 3),
        fits_on_h100=(bytes_int8 / 1e9) < usable_vram_gb,
        gpus_needed=max(1, gpus_needed),
    )


# ═══ Storage Estimation ═══

@dataclass
class StorageEstimate:
    total_tb: float
    vector_index_tb: float
    index_overhead_tb: float
    raw_text_tb: float
    approximate_cost_monthly: float
    storage_tier: str


def estimate_rag_storage(
    num_docs: int,
    avg_tokens_per_doc: int = 512,
    embedding_dim: int = 1536,
    bytes_per_elem: int = 4,
    hnsw_m: int = 32,
) -> StorageEstimate:
    """Estimate storage for a RAG system with N documents."""
    # Raw vectors
    vector_bytes = num_docs * embedding_dim * bytes_per_elem
    vector_tb = vector_bytes / 1e12

    # HNSW index: M edges per node, each edge is 4 bytes (int32)
    edge_bytes = num_docs * hnsw_m * 4
    edge_tb = edge_bytes / 1e12

    total_index_tb = vector_tb + edge_tb

    # Original text (4 bytes per token, UTF-8)
    text_bytes = num_docs * avg_tokens_per_doc * 4
    text_tb = text_bytes / 1e12

    total_tb = total_index_tb + text_tb

    # Cost: S3 Standard $0.023/GB/month
    cost_s3 = total_tb * 1024 * 0.023

    # Determine tier recommendation
    if total_tb < 1:
        tier = "hot"
    elif total_tb < 50:
        tier = "warm"
    else:
        tier = "cold"

    return StorageEstimate(
        total_tb=round(total_tb, 2),
        vector_index_tb=round(total_index_tb, 2),
        index_overhead_tb=round(edge_tb, 2),
        raw_text_tb=round(text_tb, 2),
        approximate_cost_monthly=round(cost_s3, 2),
        storage_tier=tier,
    )


# ═══ Throughput Estimation ═══

@dataclass
class ThroughputEstimate:
    rps_max: float
    tok_s_total: float
    gpus_required: int
    gpus_for_latency_sla: int
    total_gpus: int
    effective_batch_throughput_multiplier: float


def estimate_llm_throughput(
    target_rps: int,
    avg_output_tokens: int,
    tok_s_per_gpu: float = 20.0,
    batching_efficiency: float = 10.0,
    peak_multiplier: float = 1.5,
    latency_sla_ms: Optional[float] = None,
    time_per_output_token_ms: float = 50.0,
) -> ThroughputEstimate:
    """Estimate GPU requirements for LLM inference."""
    effective_tok_s = tok_s_per_gpu * batching_efficiency
    total_tok_s = target_rps * avg_output_tokens
    gpus_throughput = math.ceil(total_tok_s / effective_tok_s * peak_multiplier)

    gpus_latency = 0
    if latency_sla_ms is not None:
        single_req_ms = time_per_output_token_ms * avg_output_tokens
        if single_req_ms > latency_sla_ms:
            concurrent_requests = target_rps * single_req_ms / 1000
            max_batch = 32
            gpus_latency = math.ceil(concurrent_requests / max_batch)
        else:
            gpus_latency = gpus_throughput

    total_gpus = max(gpus_throughput, gpus_latency)

    return ThroughputEstimate(
        rps_max=target_rps,
        tok_s_total=total_tok_s,
        gpus_required=gpus_throughput,
        gpus_for_latency_sla=gpus_latency,
        total_gpus=total_gpus,
        effective_batch_throughput_multiplier=batching_efficiency,
    )


# ═══ Cost Estimation ═══

@dataclass
class CostEstimate:
    compute_monthly: float
    storage_monthly: float
    network_monthly: float
    total_monthly: float
    gpu_count: int
    gpu_hours_month: int
    hourly_rate: float


def estimate_monthly_cost(
    gpu_count: int,
    gpu_hourly_rate: float = 3.00,
    storage_tb: float = 0.0,
    storage_cost_per_gb: float = 0.023,
    network_tb_egress: float = 0.0,
    network_cost_per_gb: float = 0.02,
    reserved_discount: float = 0.0,
) -> CostEstimate:
    """Estimate monthly cost for an ML inference deployment."""
    effective_rate = gpu_hourly_rate * (1 - reserved_discount)
    hours_per_month = 730

    compute = gpu_count * effective_rate * hours_per_month
    storage = storage_tb * 1024 * storage_cost_per_gb
    network = network_tb_egress * 1024 * network_cost_per_gb

    total = compute + storage + network

    return CostEstimate(
        compute_monthly=round(compute, 2),
        storage_monthly=round(storage, 2),
        network_monthly=round(network, 2),
        total_monthly=round(total, 2),
        gpu_count=gpu_count,
        gpu_hours_month=gpu_count * hours_per_month,
        hourly_rate=effective_rate,
    )


# ═══ Latency Budget Estimation ═══

@dataclass
class LatencyBudget:
    components: dict[str, float]
    total_ms: float
    sla_ms: float
    within_sla: bool
    slack_ms: float


def build_latency_budget(
    components: dict[str, float],
    sla_ms: float,
) -> LatencyBudget:
    """Build a latency budget by summing all components."""
    total = sum(components.values())
    within = total <= sla_ms
    slack = sla_ms - total

    return LatencyBudget(
        components=components,
        total_ms=round(total, 2),
        sla_ms=sla_ms,
        within_sla=within,
        slack_ms=round(slack, 2),
    )
```

### 2.4 Application in ML/AI Systems 🤖

**Caso real: Stripe's Radar latency budget.** Stripe's fraud detection system must return a decision within 50ms to avoid adding latency to the payment flow. Their latency budget:

| Component | Allocated | Actual (P99) |
|-----------|-----------|---------------|
| Network ingress | 2ms | 0.8ms |
| Feature fetch (Redis) | 5ms | 3.2ms |
| Feature transform + preprocess | 3ms | 2.1ms |
| Model inference (GPU) | 20ms | 15.0ms |
| Decision + rule engine | 5ms | 2.5ms |
| Network egress | 2ms | 0.8ms |
| **Total** | **37ms** | **24.4ms** |

Every component is measured and tracked. When feature fetch P99 exceeds 5ms, on-call is paged. Without this budget, latency would degrade silently until transactions started timing out. This is the same approach used in the fraud detection walkthrough in [[05 - Interview Walkthrough - Design Systems End-to-End|Note 05]].

### 2.5 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Confusing throughput (requests/second) with latency (milliseconds per request). A system with high throughput can have terrible latency (1000 req/s at 10 seconds each via massive batching).

💡 **Tip**: Always estimate both. State "this design achieves X req/s throughput at Y ms P99 latency." This demonstrates you understand the throughput-latency tradeoff inherent in batching.

⚠️ **Pitfall**: Neglecting continuous batching efficiency. Estimating GPU requirements with sequential throughput (20 tok/s for 70B) without accounting for batching leads to 5-20× over-estimation of GPU needs.

💡 **Tip**: Use a batching efficiency multiplier (5-20×) when estimating LLM throughput. State the assumption explicitly: "Assuming 10× effective throughput from continuous batching, we need..."

### 2.6 Knowledge Check ❓

1. A RAG system has 100M documents, 1536-dim embeddings, float32. Estimate storage for vectors + HNSW index. What tier of storage is appropriate?
2. An LLM inference service handles 5000 RPS with average 300 output tokens. Using Llama-70B on H100 (20 tok/s sequential, 10× batching multiplier), how many GPUs are needed?
3. Build a latency budget for a fraud detection system with 30ms SLA. Include network, feature fetch, inference, and post-processing.

---

## Module 3: Practice Problems — From Zero to Solution

### 3.1 Problem 1: Design RAG for 1B Documents ⭐

**Problem Statement**: Design a RAG system for 1 billion documents. Each document averages 500 tokens. Embeddings are 1536-dimensional float32. Daily query volume: 10M queries. P99 latency: <500ms. Estimate storage, throughput, and cost.

**Solution — Step by Step**:

**Storage:**

$$S_{\text{vectors}} = 10^9 \times 1536 \times 4 = 6.144 \text{ TB}$$

HNSW index with M=32:

$$S_{\text{edges}} = 10^9 \times 32 \times 4 = 128 \text{ GB}$$

$$S_{\text{index}} = 6.144 + 0.128 \approx 6.3 \text{ TB}$$

Original text (for display, chunk-level):

$$S_{\text{text}} = 10^9 \times 500 \times 4 = 2.0 \text{ PB}$$

The vector index is 6.3 TB — manageable on S3 or sharded across NVMe SSDs (~2 NVMe drives at 3.2TB each). The full text is 2 PB — must live in S3 (cold tier).

**Throughput:**

10M queries/day = 10M / 86400 ≈ 116 queries/second. Retrieval at 116 QPS is trivially handled by a single GPU (HNSW search on GPU handles 10K+ QPS on billion-scale indices). The LLM generation component is the bottleneck — not retrieval.

**Cost:**

- Vector index (S3): 6.3 TB × 1024 × $0.023 = $148/month
- Raw text (S3): 2000 TB × 1024 × $0.023 = $47,104/month — this dominates
- GPU for generation: $3/hr × 730 = $2,190/month per GPU

**Optimization:** Store documents compressed (gzip, ~5× compression), reducing text storage to ~400 TB and cost to ~$9,420/month. Use S3 Intelligent Tiering to reduce rarely-accessed chunks to lower-cost tiers. Cache hot document chunks on NVMe (top 1% = 20TB).

### 3.2 Problem 2: How Many GPUs for 10K Req/s Llama-70B with P99 < 200ms? ⭐⭐

**Problem Statement**: Serve Llama-70B on H100 GPUs. Target: 10,000 requests/second with average 200 output tokens. P99 latency must be < 200ms. Streaming is allowed (TTFT can be ~50ms). Estimate GPU count and monthly cost.

**Solution — Step by Step**:

This is the core interview hot spot. Let's break it down:

**Step 1: Sequential throughput (no batching)**

At 20 tok/s per GPU and 200 output tokens per request:

$$\text{RPS per GPU} = 20 / 200 = 0.1 \text{ requests/s}$$

For 10,000 RPS: $10{,}000 / 0.1 = 100{,}000$ GPUs. Obviously wrong — this is why batching exists.

**Step 2: With continuous batching**

With batching, the GPU processes multiple requests simultaneously. Effective throughput increases 5-20×. Using 10× multiplier:

$$\text{Effective tok/s per GPU} = 20 \times 10 = 200$$

$$\text{RPS per GPU} = 200 / 200 = 1.0 \text{ requests/s}$$

$$\text{GPUs for throughput} = 10{,}000 / 1.0 = 10{,}000$$

Still a lot. Let's refine:

**Step 3: Latency SLA dimension**

Single request at 20 tok/s takes 200/20 = 10 seconds. To get P99 < 200ms, we need massive parallelism. The time-per-token is ~50ms. For 200 tokens, that's 10 seconds of wall time.

But here is the key: each GPU can process many requests in parallel via continuous batching (up to 32-64 concurrent requests). At 50ms per token and 32 concurrent requests, each GPU delivers 32 × 20 = 640 effective tok/s. However, a request that arrives must wait for its token slot in the batch cycle. With 32 concurrent requests and 50ms per batch iteration, the average wait is ~32 × 50 / 2 = 800ms — already exceeding the 200ms SLA.

To meet P99 < 200ms, we need smaller batches. With batch size 4:
- 4 concurrent requests × 20 tok/s / 4 = 20 effective tok/s per request
- Wait time: 4 × 50ms / 2 = 100ms (within SLA)

But at batch size 4, effective throughput drops to 4 × 20 / 200 = 0.4 RPS per GPU. For 10,000 RPS: 10,000 / 0.4 = 25,000 GPUs.

This reveals the central tradeoff: **throughput demands large batches; latency demands small batches**. The realistic answer is: you cannot serve 10K RPS of Llama-70B at <200ms P99 on H100s. You need to either:
1. Relax the SLA (200ms → 2 seconds)
2. Use a smaller model (Llama-7B)
3. Use speculative decoding (draft model generates fast, large model verifies)
4. Cache aggressively (60%+ cache hit rate reduces GPU load proportionally)

**Step 4: Realistic design with optimizations**

With 60% cache hit rate: 4,000 RPS to GPU, 6,000 RPS from cache.
With Llama-8B as draft model + Llama-70B as verifier: 3× effective speedup.
Effective: 4,000 / 3 = 1,333 RPS-equivalent.
At batch size 16 (400ms wait, close to SLA with streaming TTFT < 50ms):
1,333 RPS × 200 tokens / (16 × 20 tok/s) = 833 GPUs.

Round to ~1,000 H100 GPUs. At $1.50/hr reserved: 1,000 × $1.50 × 730 = **$1,095,000/month**.

¡Sorpresa! The throughput-only estimate of 33 GPUs (no latency SLA) is off by 30× from the latency-constrained estimate of 1,000 GPUs. This is why back-of-envelope must always consider both throughput AND latency — the constraint that drives the design is usually latency, not throughput.

### 3.3 Problem 3: Monthly Cost for 100K-User Recommendation System ⭐

**Problem Statement**: Host a recommendation system for 100K DAU. Each user sees 50 recommendations per session, 3 sessions/day. Embedding model: 256-dim float32. Ranking model: XGBoost (CPU). Feature store: 200 features per user, Redis-backed. Estimate monthly cost.

**Solution — Step by Step**:

**Request volume:**

100K DAU × 3 sessions × 50 recs = 15M recommendation calls/day.
15M / 86400 ≈ 174 RPS average. Peak: 174 × 3 = 522 RPS.

**Storage:**

- User embeddings: 100K × 256 × 4 bytes = 102 MB — negligible
- Item embeddings: Assume 1M items: 1M × 256 × 4 = 1 GB
- Feature store: 100K users × 200 features × 128 bytes avg = 2.56 GB
Total: ~4 GB — fits entirely in a single Redis instance.

**Infrastructure:**

- Redis (feature store + embedding cache): 1 × `cache.t3.medium` (~$30/month)
- Inference server (FastAPI + XGBoost): 3 × `c5.xlarge` (~$120/month each) for HA — $360/month
- No GPU needed — XGBoost runs on CPU, embeddings are pre-computed
- Load balancer (ALB): ~$25/month
- Model artifact storage (S3): ~$0.50/month

**Total: ~$415/month.**

The key insight: small-scale ML doesn't need GPUs. The most common mistake is over-engineering — deploying a GPU cluster for a workload that fits on a single CPU instance. This is why back-of-envelope estimation is essential: you catch scale mismatches before they become infrastructure decisions.

---

## Module 4: Interview Strategy — Deriving, Not Guessing

### 4.1 Theoretical Foundation 🧠

In a FAANG+ ML system design interview, your back-of-envelope work is the difference between "this candidate seems promising" and "this candidate knows exactly what they're doing." The interviewer is not evaluating whether you know the exact number of GPUs for Llama-70B — they are evaluating your ability to derive sensible numbers from first principles.

❌/✅ **Guessing vs Showing Back-of-Envelope Work**:

❌ **Guessing**: "I think we need about 50 GPUs."
- The interviewer has no insight into your reasoning.
- They cannot tell if you understand batching, VRAM constraints, or latency budgets.
- This answer is indistinguishable from a lucky guess.

✅ **Deriving**: "At 20 tok/s per H100 for Llama-70B, 200 output tokens per request, and 10× batching efficiency, each GPU serves 1 req/s. For 1000 RPS, we need 1000 GPUs for throughput. But the P99 < 500ms SLA means we need smaller batches which reduce throughput. Let me calculate the latency-constrained GPU count..."

- The interviewer sees every assumption.
- They can engage with specific parts ("Why 10× batching efficiency?").
- This demonstrates senior engineering judgment.

**The interview script:**

1. **State assumptions aloud**: "I'll assume 20 tok/s per H100 for 70B, 200 avg output tokens, 10× batching multiplier."
2. **Write the formula**: "GPUs = (RPS × tokens) / (tok/s × batch_multiplier)"
3. **Plug in numbers**: "(1000 × 200) / (20 × 10) = 1000 GPUs"
4. **Sense-check**: "That's 1000 GPUs for 1000 RPS — each GPU serves 1 req/s. At $1.50/hr, that's $1.1M/month. Is that reasonable? Maybe we need a smaller model or caching."
5. **Iterate**: "With 60% cache hit rate, only 400 RPS hit GPU: 400 GPUs. With speculative decoding, 200 GPUs."

### 4.2 Mental Model 📐

```
┌─── Interview Estimation Flow — What to Say Aloud ──────────────────┐
│                                                                       │
│  1. Clarify requirements:                                             │
│     "What's the scale? 500M DAU? What's the SLA? <200ms P99?"        │
│                                                                       │
│  2. Write numbers on the board:                                       │
│     ┌─────────────────────────────────────────────┐                  │
│     │ H100: 80GB VRAM, 20 tok/s (70B), $3/hr      │                  │
│     │ Network: same-AZ 1ms, cross-region 50ms      │                  │
│     │ Storage: NVMe 3.5GB/s, S3 $0.023/GB/month    │                  │
│     │ Model: 70B = 140GB FP16, 70GB INT8, 35GB INT4│                  │
│     └─────────────────────────────────────────────┘                  │
│                                                                       │
│  3. Derive step by step (DO NOT SKIP):                                │
│     "Daily requests = 500M DAU x 10 requests = 5B/day"               │
│     "RPS = 5B / 86400 ~ 57,870 RPS"                                  │
│     "GPUs = RPS x output_tokens / (tok/s x batch_mult)"              │
│                                                                       │
│  4. Sanity-check each number:                                         │
│     "5B requests/day / 86400 seconds = 57,870. At 50ms each,         │
│      that's 2,893 seconds of compute per second — need at least       │
│      2,893 concurrent workers."                                      │
│                                                                       │
│  5. Discuss tradeoffs explicitly:                                     │
│     "The throughput estimate says X GPUs, but latency says Y GPUs.    │
│      The constraint is Y > X, so we size for latency. At $Z/month,    │
│      is this acceptable? Can we reduce requirements?"                 │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.3 Knowledge Check ❓

1. An interviewer asks "How many GPUs for 1M DAU ChatGPT clone?" Walk through the estimation steps: requests/day, RPS, GPUs, cost. What key assumption changes the answer most?
2. You estimate 1000 GPUs at $1.1M/month. The interviewer says "That sounds expensive." How do you respond? What optimizations do you propose?
3. A teammate says "I don't need back-of-envelope — I'll just benchmark." Critique this approach. When is benchmarking necessary, and when is estimation sufficient?

---

## 📦 Código de Compresión

```python
"""
back_of_envelope.py — Complete back-of-envelope estimation library
for ML system design. Covers: model sizing, storage, throughput,
latency, cost, and GPU count.

Quick reference:
  Model:   1B params = 2GB (FP16) = 1GB (INT8) = 0.5GB (INT4)
  GPU:     H100 = 80GB VRAM, 3.35 TB/s HBM, 20 tok/s (70B FP16)
  Network: same-AZ = 1ms, cross-region = 50ms
  Storage: NVMe = 3.5 GB/s ($0.10/GB/mo), S3 = 500MB/s ($0.023/GB/mo)
  Cost:    H100 = $3/hr on-demand, $1.50/hr reserved 1yr
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ═══ Constants (memorize these) ═══

BYTES_PER_PARAM = {"fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

H100_VRAM_GB = 80.0
H100_HBM_BANDWIDTH_TB_S = 3.35
H100_TOK_S_70B_FP16 = 20.0
H100_TOK_S_70B_INT8 = 35.0
H100_PRICE_ON_DEMAND = 3.00
H100_PRICE_RESERVED_1YR = 1.50
H100_PRICE_RESERVED_3YR = 1.00

NVME_BANDWIDTH_GB_S = 3.5
NVME_COST_GB_MONTH = 0.10
S3_BANDWIDTH_MB_S = 500
S3_COST_GB_MONTH = 0.023

NETWORK_SAME_AZ_MS = 1.0
NETWORK_CROSS_REGION_MS = 50.0
NETWORK_CROSS_CONTINENT_MS = 100.0

HOURS_PER_MONTH = 730


# ═══ Model Memory Estimator ═══

@dataclass
class ModelMemory:
    name: str
    params_b: float
    layers: int
    hidden_dim: int
    fp16_gb: float
    int8_gb: float
    int4_gb: float
    kv_cache_mb_per_token: float
    max_tokens_h100: int
    fits_single_h100_fp16: bool
    fits_single_h100_int8: bool
    gpus_for_tp_fp16: int


def assess_model(name: str, params_b: float, layers: int,
                 hidden_dim: int) -> ModelMemory:
    """Complete model memory assessment for H100 deployment."""
    fp16_gb = params_b * 1e9 * 2 / 1e9
    int8_gb = params_b * 1e9 * 1 / 1e9
    int4_gb = params_b * 1e9 * 0.5 / 1e9

    # KV cache: 2 (K+V) x layers x hidden_dim x 2 bytes (FP16)
    kv_mb_per_token = (2 * layers * hidden_dim * 2) / (1024 * 1024)

    # Usable VRAM: 80GB - 5GB CUDA overhead
    usable = H100_VRAM_GB - 5.0

    # Max tokens of KV cache on H100 (INT8 model)
    remaining = usable - int8_gb
    max_tokens = int(remaining * 1024 / kv_mb_per_token) if remaining > 0 else 0

    return ModelMemory(
        name=name, params_b=params_b, layers=layers, hidden_dim=hidden_dim,
        fp16_gb=round(fp16_gb, 1), int8_gb=round(int8_gb, 1),
        int4_gb=round(int4_gb, 1),
        kv_cache_mb_per_token=round(kv_mb_per_token, 3),
        max_tokens_h100=max_tokens,
        fits_single_h100_fp16=fp16_gb <= usable,
        fits_single_h100_int8=int8_gb <= usable,
        gpus_for_tp_fp16=max(1, math.ceil(fp16_gb / (usable * 0.85))),
    )


# ═══ Inference Capacity Planner ═══

@dataclass
class InferencePlan:
    target_rps: int
    avg_output_tokens: int
    avg_prompt_tokens: int
    model_memory: ModelMemory
    tok_s_per_gpu: float
    batching_multiplier: float
    rps_per_gpu_throughput: float = 0.0
    gpus_throughput: int = 0
    gpus_total: int = 0
    cost_monthly_reserved: float = 0.0

    def __post_init__(self):
        eff = self.tok_s_per_gpu * self.batching_multiplier
        self.rps_per_gpu_throughput = eff / self.avg_output_tokens
        self.gpus_throughput = math.ceil(
            self.target_rps / self.rps_per_gpu_throughput
        )
        self.gpus_total = self.gpus_throughput
        self.cost_monthly_reserved = (
            self.gpus_total * H100_PRICE_RESERVED_1YR * HOURS_PER_MONTH
        )

    def summary(self) -> str:
        return (
            f"{self.model_memory.name}: {self.target_rps} RPS, "
            f"{self.avg_output_tokens} tok/req, "
            f"{self.gpus_total} GPUs, "
            f"${self.cost_monthly_reserved:,.0f}/mo"
        )


# ═══ Vector Storage Estimator ═══

@dataclass
class VectorStorage:
    num_vectors: int
    dim: int
    bytes_per_dim: int
    hnsw_m: int = 32
    raw_gb: float = 0.0
    index_gb: float = 0.0
    total_gb: float = 0.0
    s3_cost_monthly: float = 0.0

    def __post_init__(self):
        self.raw_gb = (self.num_vectors * self.dim * self.bytes_per_dim) / 1e9
        self.index_gb = (self.num_vectors * self.hnsw_m * 4 + self.num_vectors * 200) / 1e9
        self.total_gb = self.raw_gb + self.index_gb
        self.s3_cost_monthly = self.total_gb * S3_COST_GB_MONTH


# ═══ Quick Estimators (one-liners) ═══

def quick_gpu_count(rps: int, output_tokens: int = 200,
                    tok_s_per_gpu: float = 20.0,
                    batching_mult: float = 10.0) -> int:
    """One-line GPU count for LLM inference."""
    return math.ceil(rps * output_tokens / (tok_s_per_gpu * batching_mult))


def quick_cost(gpus: int, hourly_rate: float = 1.50) -> float:
    """One-line monthly cost estimate."""
    return gpus * hourly_rate * 730


def quick_rag_storage(docs: int, dim: int = 1536,
                      bytes_per: int = 4) -> float:
    """One-line RAG vector storage estimate (GB)."""
    return round(docs * dim * bytes_per / 1e9, 2)


# ═══ Demonstration ═══

if __name__ == "__main__":
    llama70b = assess_model("Llama-3-70B", 70.0, 80, 8192)
    print(f"\n{llama70b.name}:")
    print(f"  FP16: {llama70b.fp16_gb} GB, INT8: {llama70b.int8_gb} GB")
    print(f"  KV cache/token: {llama70b.kv_cache_mb_per_token:.2f} MB")
    print(f"  Max tokens H100 (INT8): {llama70b.max_tokens_h100:,}")
    print(f"  TP GPUs (FP16): {llama70b.gpus_for_tp_fp16}")

    plan = InferencePlan(5000, 200, 1000, llama70b, 20.0, 10.0)
    print(f"\nInference Plan: {plan.summary()}")

    rag = VectorStorage(100_000_000, 1536, 4)
    print(f"\nRAG Storage (100M docs):")
    print(f"  Total: {rag.total_gb:.1f} GB, S3: ${rag.s3_cost_monthly:.2f}/mo")
```

---

## References

- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly. Chapter 8: Deployment and Prediction Serving.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. Chapter 1: Reliable, Scalable, and Maintainable Applications.
- Donnemartin. (2024). *System Design Primer — How to approach a system design interview*. https://github.com/donnemartin/system-design-primer
- NVIDIA. (2024). *H100 Tensor Core GPU Datasheet*. https://www.nvidia.com/en-us/data-center/h100/
- Stripe Engineering. (2023). "How Stripe Radar detects fraud in real time." https://stripe.com/radar
- [[01 - CAP Theorem and Consistency Models in ML Workloads|CAP Theorem]]
- [[02 - Caching, CDNs and Storage Architectures for ML|Caching for ML]]
- [[03 - Load Balancing, Sharding and Scaling ML Systems|Load Balancing]]
- [[05 - Interview Walkthrough - Design Systems End-to-End|Interview Walkthrough]]
