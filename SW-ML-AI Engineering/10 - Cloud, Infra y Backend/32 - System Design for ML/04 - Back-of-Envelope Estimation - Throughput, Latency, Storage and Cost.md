# 📐 04 - Back-of-Envelope Estimation — Throughput, Latency, Storage, and Cost

## 🎯 Learning Objectives

- Memorize the 15 essential numbers for ML system design: GPU HBM bandwidth (3.35 TB/s for H100), network latencies (same-AZ 1 ms, cross-region 50 ms), model sizes (1B params = 2 GB FP16 / 1 GB INT8), inference throughput (70B ≈ 20 tok/s on H100), and storage costs
- Apply LaTeX estimation templates for storage, throughput, latency, and cost — deriving orders of magnitude from first principles
- Solve three graded practice problems: "Design RAG for 1B documents," "How many GPUs for 10K req/s Llama-70B with P99 < 200 ms?", and "Monthly cost for 100K-user recommendation system"
- Demonstrate back-of-envelope work fluently in interviews — the difference between guessing (❌) and deriving (✅)

## Introduction

Back-of-envelope estimation is the most undervalued skill in ML system design. An interviewer asks "How many GPUs for 10,000 req/s with Llama-70B at P99 < 200 ms?" The weak candidate guesses "maybe 50?" The strong candidate writes:

$$\text{GPUs} = \frac{\text{RPS} \times \text{avg\_output\_tokens}}{\text{tok/s per GPU}} \times \frac{1}{1 - \text{overhead}}$$

$$= \frac{10{,}000 \times 200}{20} \times \frac{1}{0.7} = 142{,}857 \approx 143 \text{ GPUs}$$

Then explains: "That's ~143 H100 GPUs for throughput-only. But P99 < 200 ms adds a latency dimension — at 20 tok/s, 200 output tokens take 10 seconds on a single GPU, so we need batching and far more GPUs to meet the SLA. Let me refine..."

The term *back-of-envelope* dates to Fermi problems — Enrico Fermi estimated the yield of the Trinity nuclear test by dropping scraps of paper and measuring the blast. The principle is the same for ML: you don't need exact numbers; you need the right order of magnitude, derived from first principles, to catch design errors before you spend $1M on GPUs.

---

## 1. The Numbers — What to Memorize

| Quantity | Value | Why It Matters |
|----------|-------|----------------|
| H100 HBM3e bandwidth | 3.35 TB/s | Upper bound on data movement |
| H100 VRAM | 80 GB | How much model + KV cache fits |
| NVMe SSD bandwidth | 3.5 GB/s | Model loading, training data streaming |
| NVLink bandwidth | 900 GB/s (H100) | GPU-GPU tensor parallelism |
| Same-AZ latency | <1 ms | Inference → feature store, Redis |
| Cross-region (US↔EU) | 80-120 ms | Multi-region deployment |
| 1B params FP16/INT8/INT4 | 2 GB / 1 GB / 0.5 GB | Model memory sizing |
| Llama-70B H100 FP16 | ~20 tok/s | Single-request sequential throughput |
| H100 on-demand | ~$3.00/hr | Compute cost baseline |
| GPU VRAM cost | ~$720/GB/month | Why KV cache efficiency matters |
| NVMe SSD cost | ~$0.10/GB/month | Hot storage |
| S3 Standard | ~$0.023/GB/month | Cold storage |
| S3 Glacier | ~$0.004/GB/month | Archive storage |
| Network egress | ~$0.02/GB | Data transfer cost |

**Model size formula**: $S_{\text{model}} = N_{\text{params}} \times B_{\text{precision}}$ where $B$ is bytes per param (2 for FP16, 1 for INT8, 0.5 for INT4). Add 20-30% overhead for optimizer states (training) and metadata.

**KV cache per token**: $S_{KV} = 2 \times L \times d_{\text{model}} \times B_{\text{param}}$ — for Llama-70B (80 layers, 8192 hidden dim, FP16): ~2.6 MB per token. For a 4000-token conversation: ~10.4 GB of VRAM just for cache.

**Inference throughput**: With continuous batching (vLLM, SGLang), effective throughput increases 5-20× because the GPU processes multiple requests in parallel during the decode phase. Each request in the batch shares the GPU's forward pass, amortizing the cost across users.

---

## 2. Estimation Templates

Every ML system design requires estimating four dimensions. These are the LaTeX derivations you should be able to reproduce on a whiteboard.

**Storage** for embedding-based systems (RAG, recommendation, semantic search):

$$S_{\text{vectors}} = N_{\text{docs}} \times d_{\text{dim}} \times b_{\text{elem}}$$

where $N_{\text{docs}}$ is the number of documents, $d_{\text{dim}}$ is the embedding dimension, and $b_{\text{elem}}$ is bytes per element (4 for float32, 2 for float16, 1 for int8).

Full HNSW index including graph structure:

$$S_{\text{index}} = S_{\text{vectors}} \times (1 + M \times b_{\text{edge}})$$

where $M$ is the HNSW connectivity parameter (typically 16-64) and $b_{\text{edge}}$ is bytes per graph edge (~4 for int32 IDs). Index overhead is typically 100-200% of raw vector storage.

**Throughput** for LLM inference with continuous batching:

$$R_{\text{max}} = \frac{N_{\text{GPU}} \times R_{\text{tok/s/GPU}}}{L_{\text{output}}}$$

Key insight: throughput is normalized by output length because each output token requires a full forward pass. Short outputs → high request throughput. Long outputs → low request throughput.

**Latency** — end-to-end is sum of time-to-first-token (prefill) and generation:

$$L_{\text{total}} = T_{\text{prefill}} + T_{\text{output}} \times L_{\text{output}}$$

For Llama-70B on H100: $T_{\text{prefill}} \approx 0.5$-$2$ seconds for 2000 tokens, $T_{\text{output}} \approx 50$ ms per token (20 tok/s). For batched inference, P99 latency increases with batch size. With continuous batching (vLLM), the latency impact is amortized across the batch.

**Cost** — monthly total has compute + storage + network components:

$$C_{\text{compute}} = N_{\text{GPU}} \times P_{\text{GPU-hour}} \times 730$$

$$C_{\text{total}} = C_{\text{compute}} + C_{\text{storage}} + C_{\text{network}}$$

Add 20-30% for networking, load balancers, and other infrastructure. At $1.50/hr reserved, a single H100 costs ~$1,095/month. A cluster of 1000 costs $1.1M/month.

❌/✅ **Guessing vs deriving**: ❌ "I think we need ~100 GPUs" — indistinguishable from a lucky guess. ✅ "At 20 tok/s per H100, 200 avg output tokens, 10K RPS: 100K GPUs for throughput. With 10× batching: 10K GPUs. At $1.50/hr reserved: $10.95M/month. Can we reduce output tokens or cache aggressively?"

**Caso real: Stripe's Radar latency budget.** Stripe's fraud detection must return a decision within 50 ms. Their budget: network ingress (2 ms) → feature fetch from Redis (5 ms) → feature transform (3 ms) → model inference (20 ms) → rule engine (5 ms) → network egress (2 ms) = total 37 ms budget, giving 13 ms slack. Every component is measured and tracked — when feature fetch P99 exceeds 5 ms, on-call is paged. Without this budget, latency degrades silently until transactions time out. This same budget-driven approach applies to any latency-sensitive ML system ([[05 - Interview Walkthrough - Design Systems End-to-End|Note 05]]).

⚠️ **Pitfall**: Confusing throughput (req/s) with latency (ms/request). A system with high throughput can have terrible latency — 1000 req/s at 10 seconds each via massive batching is useless for real-time.

💡 **Tip**: Always estimate both. State "this design achieves X req/s throughput at Y ms P99 latency." This demonstrates understanding of the throughput-latency tradeoff inherent in batching.

⚠️ **Pitfall**: Estimating GPU requirements with sequential throughput (20 tok/s for 70B) without continuous batching leads to 5-20× over-estimation of GPU needs.

💡 **Tip**: State your batching assumption explicitly: "Assuming 10× effective throughput from continuous batching, we need X GPUs."

---

## 3. Practice Problems

### Problem 1: Design RAG for 1B Documents

**Storage**: 1B docs × 1536-dim float32 embeddings:

$$S_{\text{vectors}} = 10^9 \times 1536 \times 4 = 6.144 \text{ TB}$$

HNSW index with M=32: $S_{\text{edges}} = 10^9 \times 32 \times 4 = 128 \text{ GB}$, total index ~6.3 TB. Raw text (500 tokens each): $S_{\text{text}} = 10^9 \times 500 \times 4 = 2 \text{ PB}$.

**Throughput**: 10M queries/day = 116 QPS. Retrieval is trivially handled by a single GPU (HNSW on GPU handles 10K+ QPS at billion scale). The LLM generation is the bottleneck.

**Cost**: Vector index on S3: 6.3 × 1024 × $0.023 = $148/mo. Raw text: 2000 × 1024 × $0.023 = $47,104/mo — this dominates. GPU for generation: $3/hr × 730 = $2,190/mo per GPU.

**Optimization**: Compress text (gzip, ~5×) → 400 TB → ~$9,420/mo. Cache hot document chunks on NVMe (top 1% = 20 TB). Use S3 Intelligent Tiering for rarely-accessed chunks.

### Problem 2: 10K Req/s Llama-70B with P99 < 200 ms

**Step 1 — Sequential throughput**: At 20 tok/s per GPU, 200 output tokens/req: 0.1 req/s per GPU → 100,000 GPUs. Obviously wrong — this is why batching exists.

**Step 2 — With continuous batching (10×)**: 1.0 req/s per GPU → 10,000 GPUs for throughput.

**Step 3 — Latency dimension**: Single request at 20 tok/s takes 10 seconds. To get P99 < 200 ms, we need massive parallelism. With continuous batching, each GPU processes 32 concurrent requests. But at 50 ms per token, average wait is ~32 × 50 / 2 = 800 ms — exceeding the SLA. With batch size 4: wait = 4 × 50 / 2 = 100 ms (within SLA), but throughput drops to 0.4 req/s per GPU → 25,000 GPUs.

This reveals the central tradeoff: **throughput demands large batches; latency demands small batches**. The realistic answer: you cannot serve 10K RPS of Llama-70B at <200 ms on H100s. You need to either relax the SLA, use a smaller model, use speculative decoding, or cache aggressively.

**Step 4 — With optimizations**: 60% cache hit rate → 4,000 RPS to GPU. Speculative decoding (3× speedup) → 1,333 RPS-equivalent. Batch size 16 (400 ms wait, acceptable with streaming TTFT < 50 ms) → ~833 GPUs. At $1.50/hr reserved: 833 × $1.50 × 730 = **$912,000/month**.

¡Sorpresa! The throughput-only estimate (10,000 GPUs with batching) is off by 10× from the latency-constrained estimate of 833 GPUs with optimizations. The constraint that drives the design is usually latency, not throughput.

### Problem 3: Monthly Cost for 100K-User Recommendation System

**Request volume**: 100K DAU × 3 sessions × 50 recs = 15M recommendation calls/day. 15M/86400 ≈ 174 RPS average, 522 peak.

**Storage**: User embeddings (100K × 256 × 4 = 102 MB), item embeddings (1M × 256 × 4 = 1 GB), feature store (100K × 200 feats × 128 B = 2.56 GB). Total ~4 GB — fits entirely in a single Redis instance.

**Infrastructure**: Redis feature store + embedding cache: ~$30/month. Inference server (FastAPI + XGBoost): 3 × `c5.xlarge` ($120/mo each) = $360/month for HA. No GPU needed — XGBoost runs on CPU, embeddings are pre-computed. Load balancer: ~$25/month. **Total: ~$415/month.**

Key insight: small-scale ML doesn't need GPUs. The most common mistake is over-engineering — deploying a GPU cluster for a workload that fits on a single CPU instance. Back-of-envelope estimation catches scale mismatches before they become infrastructure decisions.

---

## 4. Interview Strategy — Deriving, Not Guessing

In a FAANG+ ML system design interview, your back-of-envelope work is the difference between "this candidate seems promising" and "this candidate knows exactly what they're doing." The interviewer evaluates your ability to derive sensible numbers from first principles.

**The interview script:**

1. **State assumptions aloud**: "I'll assume 20 tok/s per H100 for 70B, 200 avg output tokens, 10× batching multiplier."
2. **Write the formula**: $\text{GPUs} = \frac{\text{RPS} \times \text{tokens}}{\text{tok/s} \times \text{batch\_mult}}$
3. **Plug in numbers**: $\frac{1000 \times 200}{20 \times 10} = 1000 \text{ GPUs}$
4. **Sense-check**: "That's 1000 GPUs for 1000 RPS — each GPU serves 1 req/s. At $1.50/hr, that's $1.1M/month. Is that reasonable? Maybe we need a smaller model or caching."
5. **Iterate**: "With 60% cache hit rate, only 400 RPS hit GPU: 400 GPUs. With speculative decoding, 200 GPUs."

❌ **Guessing**: "I think we need about 50 GPUs." The interviewer has no insight into your reasoning. This answer is indistinguishable from a lucky guess.

✅ **Deriving**: "At 20 tok/s per H100 for Llama-70B, 200 output tokens per request, and 10× batching efficiency, each GPU serves 1 req/s. For 1000 RPS, we need 1000 GPUs for throughput. But the P99 < 500 ms SLA means we need smaller batches which reduce throughput. Let me calculate the latency-constrained GPU count..."

⚠️ **Pitfall**: Forgetting to state your assumptions. The interviewer can't evaluate what they can't see. Every silent assumption is a potential gotcha.

💡 **Tip**: Write every number and formula on the whiteboard. The visual progression — assumptions → formula → substitution → result → sanity check — is what interviewers reward.

**Caso real: Uber's ML platform team** uses back-of-envelope as their first pass for every infrastructure decision. Before provisioning any GPU cluster, the team estimates storage, throughput, latency, and cost from first principles. If the estimate says 500 GPUs and a benchmark later says 480, the estimate saved weeks of over-provisioning. If the estimate says 500 and the benchmark says 5000, the estimate caught a design flaw before the purchase order.

---

### Pitfall Patterns When Numbers Lie

Even with memorized numbers, certain estimation traps catch experienced engineers:

**The throughput-latency inversion**: You calculate 10K RPS throughput and assume you need X GPUs. But with 200 output tokens at 20 tok/s, a single request takes 10 seconds. If the SLA is 200 ms, you need 50× more GPUs than the throughput estimate. This is the most common mistake in ML system design interviews — solving for throughput and forgetting latency.

**The batch multiplier trap**: Continuous batching gives 5-20× throughput improvement, but only at batch sizes that destroy latency. At batch size 32, the wait time alone exceeds most real-time SLAs. Always verify that your batch size assumption is compatible with your latency SLA.

**The egress surprise**: Most cost estimates focus on compute and storage. But at scale, network egress can dominate. At $0.02/GB, serving 1M requests/day with 10 KB responses costs $6/day in egress alone. For streaming LLM responses (10-50 KB per output), this adds up to $100-500/day for moderate traffic.

**Caso real: A well-funded startup** provisioned 500 H100 GPUs based on a throughput-only estimate for their chatbot. After deployment, the P99 latency was 12 seconds — 60× their 200 ms SLA. The latency-constrained GPU requirement was 30,000 GPUs. They had to rewrite their architecture to use speculative decoding, prefix caching, and a smaller draft model. The throughput-only estimate was off by 60× because it didn't account for the sequential nature of autoregressive generation. Back-of-envelope with latency would have caught this before the $4.5M GPU purchase.

---

## 🎯 Key Takeaways

- Memorize the 5-tier hierarchy: hardware primitives (HBM, NVMe, PCIe) → derived quantities (model size, KV cache) → system estimates (storage, throughput, latency, cost)
- Always estimate both throughput and latency separately — the binding constraint is almost always latency, not throughput
- Continuous batching multiplies effective throughput by 5-20×; state this assumption explicitly in every estimate
- Latency budgeting (component-by-component allocation) catches silent degradation before it reaches users
- Storage cost for raw text in RAG systems often dominates total cost — use compression and tiered storage
- The back-of-envelope interview script: state assumptions → write formula → plug numbers → sense-check → iterate
- Back-of-envelope is a design tool, not an exact science — orders of magnitude catch 10× errors, which is what matters

## References

- Huyen, C. (2022). *Designing Machine Learning Systems*. Chapter 8.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. Chapter 1.
- Donnemartin. (2024). *System Design Primer*.
- NVIDIA. (2024). *H100 Tensor Core GPU Datasheet*.
- Stripe Engineering. (2023). "How Stripe Radar detects fraud in real time."
- [[01 - CAP Theorem and Consistency Models in ML Workloads|CAP Theorem]]
- [[02 - Caching, CDNs and Storage Architectures for ML|Caching for ML]]
- [[03 - Load Balancing, Sharding and Scaling ML Systems|Load Balancing]]
- [[05 - Interview Walkthrough - Design Systems End-to-End|Interview Walkthrough]]

## 📦 Código de compresión

```python
"""
back_of_envelope.py — ML system design estimation utility.
Quick reference: model sizing, storage, throughput, latency, cost.
Usage: python back_of_envelope.py to see example estimates.
"""
from dataclasses import dataclass
from typing import Optional
import math

# Constants
BYTES_PER_PARAM = {"fp16": 2, "int8": 1, "int4": 0.5}
H100_VRAM = 80.0
H100_TOK_S_70B = 20.0
H100_PRICE_ON_DEMAND = 3.00
H100_PRICE_RESERVED = 1.50
NVME_GB_S, NVME_COST = 3.5, 0.10
S3_COST = 0.023
HOURS_PER_MONTH = 730

# Model Memory
@dataclass
class ModelMemory:
    name: str; params_b: float; layers: int; hidden_dim: int
    fp16_gb: float; int8_gb: float; int4_gb: float
    kv_cache_mb_per_token: float; max_tokens_h100: int
    gpus_for_tp_fp16: int

def assess_model(name: str, params_b: float, layers: int, hidden_dim: int) -> ModelMemory:
    fp16 = params_b * 2
    int8 = params_b * 1
    int4 = params_b * 0.5
    kv_mb = (2 * layers * hidden_dim * 2) / (1024 ** 2)
    usable = H100_VRAM - 5.0
    remaining = usable - int8
    max_tok = int(remaining * 1024 / kv_mb) if remaining > 0 else 0
    return ModelMemory(name, params_b, layers, hidden_dim,
                       round(fp16, 1), round(int8, 1), round(int4, 1),
                       round(kv_mb, 3), max_tok,
                       max(1, math.ceil(fp16 / (usable * 0.85))))

# Inference Planning
@dataclass
class InferencePlan:
    target_rps: int; avg_output_tokens: int
    batching_multiplier: float = 10.0
    rps_per_gpu: float = 0.0; gpus: int = 0; cost_monthly: float = 0.0

    def __post_init__(self):
        eff = H100_TOK_S_70B * self.batching_multiplier
        self.rps_per_gpu = eff / self.avg_output_tokens
        self.gpus = math.ceil(self.target_rps / self.rps_per_gpu)
        self.cost_monthly = self.gpus * H100_PRICE_RESERVED * HOURS_PER_MONTH

# Storage
@dataclass
class VectorStorage:
    num_vectors: int; dim: int; bytes_per_dim: int = 4; hnsw_m: int = 32
    raw_gb: float = 0.0; index_gb: float = 0.0; total_gb: float = 0.0
    s3_cost: float = 0.0

    def __post_init__(self):
        self.raw_gb = (self.num_vectors * self.dim * self.bytes_per_dim) / 1e9
        self.index_gb = (self.num_vectors * self.hnsw_m * 4) / 1e9
        self.total_gb = self.raw_gb + self.index_gb
        self.s3_cost = self.total_gb * S3_COST

# Latency Budget
@dataclass
class LatencyBudget:
    components: dict[str, float]
    total_ms: float; sla_ms: float; within_sla: bool; slack_ms: float

def build_latency_budget(components: dict[str, float], sla_ms: float) -> LatencyBudget:
    total = sum(components.values())
    return LatencyBudget(components, round(total, 2), sla_ms,
                         total <= sla_ms, round(sla_ms - total, 2))

# Quick one-liners
def quick_gpus(rps: int, out_tok: int = 200, batch_mult: float = 10.0) -> int:
    return math.ceil(rps * out_tok / (H100_TOK_S_70B * batch_mult))

def quick_cost(gpus: int, rate: float = H100_PRICE_RESERVED) -> float:
    return gpus * rate * HOURS_PER_MONTH

def quick_rag_storage(docs: int, dim: int = 1536) -> float:
    return round(docs * dim * 4 / 1e9, 2)

# Full system estimate
@dataclass
class SystemEstimate:
    name: str
    storage_tb: float = 0.0
    gpu_count: int = 0
    cost_monthly: float = 0.0
    rps: int = 0

    def summary(self) -> str:
        return (f"{self.name}: {self.storage_tb:.1f}TB storage, "
                f"{self.gpu_count} GPUs, ${self.cost_monthly:,.0f}/mo, "
                f"{self.rps:,} RPS")

def estimate_rag_system(docs_millions: int, queries_per_day: int) -> SystemEstimate:
    """Quick RAG system estimate from high-level numbers."""
    docs = docs_millions * 1_000_000
    storage = docs * 1536 * 4 / 1e12  # TB
    qps = queries_per_day / 86400
    gpus = max(1, math.ceil(qps / 50))  # ~50 QPS per GPU for generation
    cost = gpus * H100_PRICE_RESERVED * HOURS_PER_MONTH
    return SystemEstimate(f"RAG-{docs_millions}M", storage, gpus, cost, int(qps))

# Demonstration
if __name__ == "__main__":
    print("=== Model Assessment ===")
    m = assess_model("Llama-3-70B", 70.0, 80, 8192)
    print(f"  {m.name}: FP16={m.fp16_gb}GB, INT8={m.int8_gb}GB, "
          f"KV={m.kv_cache_mb_per_token:.2f}MB/tok, "
          f"max_ctx={m.max_tokens_h100:,}tok")

    print("\n=== Inference Plans ===")
    for rps in [1000, 5000, 10000]:
        plan = InferencePlan(rps, 200)
        print(f"  {rps:>5} RPS: {plan.gpus} GPUs, ${plan.cost_monthly:,.0f}/mo")
        plan_low = InferencePlan(rps, 50)
        print(f"  {rps:>5} RPS (50 tok): {plan_low.gpus} GPUs, ${plan_low.cost_monthly:,.0f}/mo")

    print("\n=== Storage Estimates ===")
    for docs in [10, 100, 1000]:
        rag = VectorStorage(docs * 1_000_000, 1536)
        print(f"  {docs}M docs: {rag.total_gb:.0f}GB, ${rag.s3_cost:.0f}/mo")

    print("\n=== Latency Budgets ===")
    budget = build_latency_budget(
        {"network": 5, "feature_fetch": 10, "inference": 30, "post": 5}, 50)
    print(f"  Fraud detection: {budget.total_ms}ms/{budget.sla_ms}ms SLA — "
          f"{'WITHIN' if budget.within_sla else 'EXCEEDED'}")

    budget2 = build_latency_budget(
        {"network": 5, "candidate_gen": 30, "features": 10, "ranking": 50}, 200)
    print(f"  Video recsys: {budget2.total_ms}ms/{budget2.sla_ms}ms SLA — "
          f"{'WITHIN' if budget2.within_sla else 'EXCEEDED'}")

    print("\n=== Quick One-Liners ===")
    print(f"  quick_gpus(5000, 200) = {quick_gpus(5000, 200)}")
    print(f"  quick_cost(1000) = ${quick_cost(1000):,.0f}/mo")
    print(f"  quick_rag_storage(100M) = {quick_rag_storage(100_000_000)}GB")
```
