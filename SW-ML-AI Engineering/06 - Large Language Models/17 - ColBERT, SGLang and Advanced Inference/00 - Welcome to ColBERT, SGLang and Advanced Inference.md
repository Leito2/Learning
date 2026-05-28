# 🏷️ Welcome to ColBERT, SGLang and Advanced Inference

## 🎯 Learning Objectives

- Understand what makes **late interaction** radically different from bi-encoders and cross-encoders.
- Learn how **token-level retrieval** (ColBERT) achieves cross-encoder accuracy at bi-encoder speed.
- Master **structured generation programs** (SGLang) and why KV cache reuse via RadixAttention changes LLM serving economics.
- Connect advanced inference techniques to production RAG and agent pipelines.

---

## What the Names Mean

**ColBERT** — **CO**ntextualized **L**ate **I**nteraction over **BERT**. The critical word is *Late*: the query and the document do not interact while they are being encoded. Instead, each is encoded independently, and their token embeddings only "meet" at retrieval time, token-by-token. This is the opposite of a *bi-encoder* (interaction happens inside the model but only at the CLS token, early and crude) and of a *cross-encoder* (interaction happens via full cross-attention, accurate but devastatingly slow).

**SGLang** — **S**tructured **G**eneration **Lang**uage. It is a domain-specific language (DSL) that treats LLM calls as first-class primitives inside a program. You write `gen()`, `select()`, `fork()`, and the runtime schedules them with automatic batching, prefix caching, and structured decoding constraints.

**Advanced Inference** — The umbrella of techniques that make LLM serving economically viable at scale: speculative decoding (draft-then-verify), continuous batching, quantization (AWQ/GPTQ), and prefix-aware caching.

## Why These Technologies Are Cutting-Edge (2025–2026)

- **ColBERT** is the only retrieval architecture that gives cross-encoder quality with bi-encoder latency. In production RAG, it sits at the sweet spot where accuracy directly impacts user trust and latency impacts cost. It outperforms dense retrieval on MS MARCO and BEIR by large margins while remaining 100× faster than cross-encoders.
- **SGLang** is the only inference engine optimized for *programs*, not single prompts. Its RadixAttention builds a radix tree of KV caches by content hash, enabling automatic reuse across multi-turn agents, LLM-as-a-Judge pipelines, and structured JSON generation. Benchmarks show 2–5× speedups over vLLM for structured workloads because vLLM's PagedAttention does not share caches across independent requests.
- **Advanced Inference** techniques like speculative decoding reduce per-token cost by 2–3×, and 4-bit quantization enables 70B-parameter models on single A100s. These are not academic curiosities; they are the margin between profit and loss in LLM SaaS.

## Course Modules

1. **00 — Welcome** (this note): Etymology, motivation, and course map.
2. **01 — ColBERT: Token-Level Late Interaction** ([[01 - ColBERT - Token-Level Late Interaction]]): Theory, MaxSim, and the late-interaction design pattern.
3. **02 — ColBERT in Production: PLAID and Vector Integration** ([[02 - ColBERT in Production - PLAID and Vector Integration]]): Indexing, clustering, and two-stage retrieval pipelines.
4. **03 — SGLang: Structured Generation and RadixAttention** ([[03 - SGLang - Structured Generation and RadixAttention]]): DSL, runtime, radix-tree KV caching, and comparisons with vLLM.
5. **Speculative Decoding & Continuous Batching**: Draft-model acceleration and in-flight request scheduling.
6. **Quantization for Serving**: AWQ, GPTQ, and GGUF trade-offs in latency vs. accuracy.
7. **End-to-End Production Patterns**: Wiring ColBERT + SGLang into [[06 - Large Language Models/12 - Production RAG/04 - Production RAG System]] and agentic loops.

## Prerequisites

- Python 3.9+ and PyTorch/TensorFlow fundamentals.
- Familiarity with transformers, embeddings, and vector search basics ([[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search]]).
- Exposure to inference serving concepts from [[06 - Large Language Models/13 - vLLM and Advanced RAG/00 - Welcome to vLLM and Advanced RAG]].

## How to Navigate This Course

Each core note follows a fixed pedagogical structure: theoretical foundations first, mental models second, code and visuals third, and production considerations last. Do not skip the "Theoretical Foundation" sections—they explain *why* the technology was invented, which is essential for debugging and tuning. The "Compression Code" at the end of each note is a standalone script that you can run to verify your understanding.

We recommend studying the notes in order. Note 01 and 02 form a retrieval pair: first you learn how ColBERT thinks, then you learn how to make it fast enough for real traffic. Note 03 shifts to the inference side of the stack and shows how serving economics change when you treat LLM calls as program primitives rather than isolated API requests.

## Prerequisites

- Python 3.9+ and PyTorch/TensorFlow fundamentals.
- Familiarity with transformers, embeddings, and vector search basics ([[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search]]).
- Exposure to inference serving concepts from [[06 - Large Language Models/13 - vLLM and Advanced RAG/00 - Welcome to vLLM and Advanced RAG]].
- Basic linear algebra: dot products, L2 normalization, and softmax.

## Related Vault Courses

- [[06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/00 - Welcome to HuggingFace Transformers Deep Dive]]
- [[06 - Large Language Models/13 - vLLM and Advanced RAG/00 - Welcome to vLLM and Advanced RAG]]
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search]]
- [[06 - Large Language Models/12 - Production RAG/04 - Production RAG System]]
