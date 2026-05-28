# 🏷️ Welcome to ColBERT, SGLang and Next-Gen Inference

## 🎯 Learning Objectives
- Understand why 2025-2026 inference is defined by structured generation and late-interaction retrieval
- Map the 5 vectors of next-gen inference covered in this course
- Identify the historical constraints that ColBERT and SGLang overcome
- Navigate the module's 11 notes and their dependencies

## Introduction

The landscape of LLM inference has bifurcated into two distinct frontiers: how we *generate* tokens and how we *search* over them. **ColBERT** — Contextualized Late Interaction over BERT — solves the retrieval side by occupying a sweet spot between fast-but-lossy bi-encoders and accurate-but-slow cross-encoders. It encodes queries and documents independently (like a bi-encoder) but defers their interaction to a lightweight token-level scoring step, giving cross-encoder accuracy at bi-encoder speed. This matters because every production RAG system built on dense embeddings alone leaves measurable relevance on the table; ColBERT reclaims it without sacrificing latency.

**SGLang** — Structured Generation Language — attacks the generation side. Traditional LLM APIs treat each request as an isolated string, forcing multi-step agents and LLM-as-Judge pipelines to recompute the same system prompt and conversation history for every turn. SGLang models LLM calls as first-class programming primitives (`gen()`, `select()`, `fork()`) and introduces **RadixAttention**, which caches KV states in a prefix-aware radix tree. When 100 evaluations share the same rubric prompt, only new tokens are computed. This is not an incremental optimization — it is a paradigm shift from "prompt engineering" to "LLM programming."

Both technologies represent the maturation of inference from research curiosities into production-grade infrastructure. This course sits at the intersection of retrieval-augmented generation, structured decoding, speculative execution, and disaggregated serving — all themes you'll recognize from [[06 - Production RAG]] and [[06 - vLLM and Advanced RAG]].

![ColBERT MaxSim illustration — query and document token embeddings interact via max-similarity](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/ColBERT_Late_Interaction.svg/1280px-ColBERT_Late_Interaction.svg.png)

---

## 1. The Five Vectors of Next-Gen Inference

Next-generation inference in 2025-2026 is driven not by a single breakthrough but by the convergence of five independent vectors, each addressing a distinct bottleneck in the LLM serving stack:

| Vector | Problem | Solution | Notes |
|--------|---------|----------|-------|
| **Inference-Time Scaling** | Fixed compute per query wastes FLOPs on easy cases | Adaptive compute budgets (budget forcing, repeated sampling) | Notes 05-06 |
| **Speculative Decoding 2.0** | Autoregressive generation is memory-bandwidth bound | Draft models (Eagle, Medusa) predict multiple tokens per step | Notes 07-08 |
| **MLA + KV Cache Compression** | KV cache is O(batch × seq_len × heads × head_dim) | Multi-head Latent Attention compresses KV to 1/8–1/16 size | Note 09 |
| **FP8 Hybrid Precision** | BF16 KV cache dominates GPU memory | KV cache in FP8, attention in BF16; DeepSeek-V3 design | Note 09 |
| **Disaggregated Serving + Edge** | Monolithic inference servers couple prefill and decode | Separate prefill/decode nodes; edge speculative decoding | Notes 05-06, 11 |

Each vector compounds: compressed KV caches enable larger batches, which make speculative decoding more effective, which reduces the pressure on disaggregated scheduling. Nothing operates in isolation.

---

## 2. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome | This overview |
| 01 | ColBERT: Token-Level Late Interaction | MaxSim theory, architecture |
| 02 | ColBERT in Production: PLAID and Vector Integration | Two-stage retrieval, scaling to billions |
| 03 | SGLang: Structured Generation and RadixAttention | DSL design, prefix caching |
| 04 | SGLang in Production: Programs, Agents and Benchmarks | Agents, function calling, benchmarks |
| 05 | Inference-Time Compute Scaling | Budget forcing, best-of-N, process rewards |
| 06 | Speculative Decoding Architectures | Draft models, tree attention, acceptance rates |
| 07 | Eagle and Medusa: Next-Gen Speculative Decoding | Multi-token prediction heads |
| 08 | Multi-head Latent Attention and KV Compression | DeepSeek-V2/V3 MLA architecture |
| 09 | FP8 Quantization and Hybrid Precision Inference | KV cache compression, activation quantization |
| 10 | Capstone: High-Performance RAG with ColBERT and SGLang | End-to-end pipeline |
| 11 | Disaggregated Serving and the Edge Frontier | Prefill/decode separation, edge inference |

---

## 3. Prerequisites

You should be comfortable with:

- **Python** and the `transformers` library — the code examples use HuggingFace idioms throughout
- **Embeddings** and vector search — you've worked with dense retrievers in [[10 - Vector Databases and Semantic Search]]
- **RAG fundamentals** — the retrieval-augmented generation pipeline from [[06 - Production RAG]]
- **LLM serving basics** — PagedAttention and continuous batching from [[06 - vLLM and Advanced RAG]]
- **Transformer internals** — attention mechanisms and the KV cache from [[06 - HuggingFace Transformers Deep Dive]]

💡 If you haven't read [[06 - vLLM and Advanced RAG]] recently, skim the PagedAttention section before Note 03 — RadixAttention builds directly on that concept.

---

## 4. Production Reality

The technologies in this course are not speculative research. Their production footprint by mid-2026 includes:

- **Microsoft Bing**: ColBERT for web-scale passage retrieval (see Note 01)
- **LMSYS Chatbot Arena**: SGLang for rubric-based LLM evaluation at scale
- **Databricks**: SGLang for SQL generation quality assessment
- **DeepSeek**: MLA and FP8 hybrid precision deployed on V2 and V3
- **Notion AI**: ColBERT-based two-stage retrieval for enterprise search

⚠️ These systems demand GPU memory proportional to model size × batch × sequence length. The techniques here *reduce* that pressure but do not eliminate it. A 70B model with 128K context will still require 4+ A100s regardless of KV cache compression.

---

## 🎯 Key Takeaways
- ColBERT bridges the bi-encoder/cross-encoder gap with token-level late interaction, delivering cross-encoder accuracy at bi-encoder speed
- SGLang treats LLM calls as programming primitives, not isolated strings — this eliminates redundant KV cache computation across multi-step workflows
- The 5 vectors of next-gen inference compound: compressed KV caches → larger batches → better speculative decoding → more efficient scheduling
- RadixAttention is the core insight behind SGLang's 2-5× speedup on structured workloads like LLM-as-a-Judge
- Every technique in this course is production-deployed at scale by Microsoft, DeepSeek, Databricks, or LMSYS
- The course assumes fluency with HuggingFace, vector search, and vLLM — review [[06 - HuggingFace Transformers Deep Dive]] if needed
- Notes 01-04 cover ColBERT and SGLang (the "what"); Notes 05-09 cover the scaling infrastructure (the "how"); Note 10 ties them together

---

## References
- ColBERT paper: Khattab & Zaharia (2020), "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", SIGIR 2020
- ColBERTv2: Santhanam, Khattab et al. (2021), "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", NAACL 2022
- SGLang paper: Zheng et al. (2024), "SGLang: Efficient Execution of Structured Language Model Programs", NeurIPS 2024
- DeepSeek-V2: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024)
- [[06 - Production RAG]]
- [[06 - vLLM and Advanced RAG]]
- [[06 - HuggingFace Transformers Deep Dive]]
- [[10 - Vector Databases and Semantic Search]]
- [[07 - MCP and Agentic Protocols]]

---

## 📦 Código de Compresión

```python
"""
Mini proof-of-concept: demonstrates the conceptual gap between
bi-encoder (fast, single vector) and cross-encoder (slow, pair-wise)
that ColBERT's late interaction fills.
"""
import time
import numpy as np

def bi_encoder_similarity(q_embed, d_embed):
    """Single vector dot product — fast but lossy."""
    return float(np.dot(q_embed, d_embed))

def cross_encoder_score(query, document, model_fn):
    """Concatenated encoding — accurate but O(N·M) expensive."""
    return model_fn(query + " [SEP] " + document)

def colbert_maxsim(q_tokens, d_tokens):
    """Token-level late interaction — balanced."""
    total = 0.0
    for qt in q_tokens:  # ¡Sorpresa! Each query token independently matches
        scores = [float(np.dot(qt, dt)) for dt in d_tokens]
        total += max(scores)  # MaxSim: best match per query token
    return total

# Simulated embeddings
np.random.seed(42)
q_tokens = [np.random.randn(128) for _ in range(8)]
d_tokens = [np.random.randn(128) for _ in range(64)]
q_vec = np.mean(q_tokens, axis=0)  # pooled representation
d_vec = np.mean(d_tokens, axis=0)

t0 = time.perf_counter()
for _ in range(100):
    bi_encoder_similarity(q_vec, d_vec)
t_bi = (time.perf_counter() - t0) / 100

t0 = time.perf_counter()
for _ in range(100):
    colbert_maxsim(q_tokens, d_tokens)
t_colbert = (time.perf_counter() - t0) / 100

print(f"Bi-encoder:  {t_bi*1e6:.0f} µs per query/doc pair")
print(f"ColBERT:     {t_colbert*1e6:.0f} µs per query/doc pair")
print(f"Speed ratio: {t_colbert/t_bi:.1f}× (exact ratio depends on token counts)")
print(f"\nColBERT keeps per-token information; bi-encoder collapses to one vector.")
print(f"Cross-encoders (not shown) would be O(q_len × d_len) full-attention —")
print(f"orders of magnitude slower on long documents.")
```