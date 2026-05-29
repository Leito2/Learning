# 🏷️ Welcome to TensorRT-LLM

## 🎯 Learning Objectives

- Understand what TensorRT-LLM is and why it occupies a distinct niche from vLLM and SGLang
- Map the compilation pipeline from PyTorch model to GPU-optimized engine
- Compare TensorRT-LLM's hardware-first optimization philosophy against memory-first (vLLM) and prefix-first (SGLang) approaches
- Navigate the 2 core notes and their dependencies

## Introduction

**TensorRT-LLM** is NVIDIA's maximum-throughput LLM inference engine. Unlike every other inference framework, TensorRT-LLM is a **compiler** — it takes a PyTorch or HuggingFace model, applies graph optimizations (layer fusion, constant folding, dead-code elimination), auto-tunes kernel parameters for your specific GPU architecture, and emits an optimized binary (`.engine` file). This compilation takes 5–60 minutes depending on model size, but the result runs 2–5x faster than framework-native inference on the same hardware.

Where vLLM optimizes **HOW** you use GPU memory (PagedAttention, block-level KV cache management), and SGLang optimizes **WHEN** you compute (RadixAttention, prefix-aware KV cache sharing), TensorRT-LLM optimizes **WHAT** runs on the GPU (fused CUDA kernels, hardware-specific instruction scheduling). It's the go-to engine when every millisecond and every GPU cycle matters — and you're willing to invest compilation time upfront.

![TensorRT architecture overview — optimizer partitions graph, fuses layers, auto-tunes kernels](https://developer-blogs.nvidia.com/wp-content/uploads/2023/10/TensorRT-LLM-stack.png)

> **Figure 1**: The NVIDIA inference stack. TensorRT-LLM compiles models into optimized engines; Triton Inference Server handles networking, dynamic batching, and multi-model serving.

---

## The Three Engines: A Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE PHILOSOPHIES                     │
├──────────────┬───────────────────┬───────────────────┬───────────────┤
│              │     vLLM          │     SGLang        │ TensorRT-LLM  │
├──────────────┼───────────────────┼───────────────────┼───────────────┤
│ Optimization │ Memory            │ Prefix reuse      │ HARDWARE      │
│ target       │ (PagedAttention)  │ (RadixAttention)  │ (kernel fuse) │
├──────────────┼───────────────────┼───────────────────┼───────────────┤
│ Compilation  │ Zero              │ Zero              │ 5–60 min      │
│ time         │                   │                   │               │
├──────────────┼───────────────────┼───────────────────┼───────────────┤
│ Peak         │ ★★★★☆            │ ★★★★☆            │ ★★★★★         │
│ throughput   │                   │                   │               │
├──────────────┼───────────────────┼───────────────────┼───────────────┤
│ Hardware     │ NVIDIA, AMD,      │ NVIDIA, AMD,      │ NVIDIA-only   │
│ support      │ Intel, Apple      │ Intel             │               │
├──────────────┼───────────────────┼───────────────────┼───────────────┤
│ Best for     │ General serving,  │ Multi-turn agents,│ Latency-SLA   │
│              │ rapid prototyping │ structured gen    │ production    │
└──────────────┴───────────────────┴───────────────────┴───────────────┘
```

---

## Course Map

| # | Note | Focus | Time |
|---|------|-------|------|
| 01 | TensorRT-LLM Architecture — Model Compilation and GPU Scheduling | Compilation pipeline, builder optimization profiles, layer fusion, in-flight batching, multi-GPU strategies | 4-6 h |
| 02 | TensorRT-LLM in Production — Triton Integration and Multi-GPU | Triton backend config, Kubernetes deployment, `genai-perf` benchmarking, engine comparison | 4-6 h |

Each note is self-contained. Note 01 covers the *compiler internals*; Note 02 covers the *serving infrastructure*.

---

## Prerequisites

| Concept | Expected Knowledge | Review If Needed |
|---------|-------------------|-----------------|
| vLLM fundamentals | PagedAttention, continuous batching, block-table memory management | [[../13 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving.md\|vLLM and Production-Grade LLM Serving]] |
| SGLang fundamentals | RadixAttention, structured generation, prefix caching | [[../17 - ColBERT, SGLang and Next-Gen Inference/03 - SGLang - Structured Generation and RadixAttention.md\|SGLang: Structured Generation and RadixAttention]] |
| GPU concepts | CUDA cores, VRAM, tensor cores, NVLink bandwidth | NVIDIA developer docs |
| LLM Serving | Batch processing, KV cache, TTFT/ITL metrics | [[../09 - Sistemas de LLMs en Produccion/03 - Serving y Batch Processing.md\|Serving y Batch Processing]] |
| NVIDIA Triton | Model repository, config.pbtxt, dynamic batching | [[../../10 - MLOps y Edge AI/29 - NVIDIA Triton Inference Server.md\|NVIDIA Triton]] |

> 💡 **Tip**: If you're new to GPU inference, read [[../13 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving.md\|the vLLM note]] first. TensorRT-LLM's concepts build on understanding why naive inference is slow.

---

## Production Reality

TensorRT-LLM is not a research project. Its production footprint by 2026 includes:

- **NVIDIA NIM** (NVIDIA Inference Microservices): TensorRT-LLM engines packaged as Docker containers with Triton, deployed on every major cloud
- **RunPod**: Serverless GPU endpoints powered by TensorRT-LLM engines — pre-compiled for A100, H100, and L40S
- **Perplexity AI**: H100 clusters running TensorRT-LLM + Triton for sub-100ms p50 latency on Llama-70B
- **NVIDIA DGX Cloud**: Every DGX instance ships with pre-compiled TensorRT-LLM engines for Llama, Mistral, and Mixtral

⚠️ TensorRT-LLM engines are **GPU-architecture-specific**. An engine compiled for A100 (SM80) will NOT run on H100 (SM90). You must recompile for each target architecture. This is the price of hardware-specific optimization.

---

## 🎯 Key Takeaways

- TensorRT-LLM is a **compiler** for LLM inference — it produces GPU-specific binary engines that run 2–5x faster than framework-native PyTorch
- vLLM optimizes memory (HOW), SGLang optimizes prefix reuse (WHEN), TensorRT-LLM optimizes GPU execution (WHAT)
- Compilation time (5–60 min) is the tradeoff: you invest time upfront for maximum throughput at runtime
- TensorRT-LLM is NVIDIA-only; for multi-vendor GPU deployments, vLLM or SGLang are required
- The full NVIDIA inference stack = TensorRT-LLM (GPU execution) + Triton Inference Server (networking/batching/routing)

## References

- TensorRT-LLM GitHub: https://github.com/NVIDIA/TensorRT-LLM
- TensorRT-LLM Documentation: https://nvidia.github.io/TensorRT-LLM
- NVIDIA Triton Inference Server: https://github.com/triton-inference-server
- [[../13 - vLLM and Advanced RAG/01 - vLLM and Production-Grade LLM Serving.md]]
- [[../17 - ColBERT, SGLang and Next-Gen Inference/00 - Welcome to ColBERT, SGLang and Next-Gen Inference.md]]
- [[../../10 - MLOps y Edge AI/29 - NVIDIA Triton Inference Server.md]]
- [[../09 - Sistemas de LLMs en Produccion/03 - Serving y Batch Processing.md]]
