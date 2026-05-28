# 🤗 Welcome to HuggingFace Transformers Deep Dive

## 🎯 Learning Objectives

- Understand the architecture and design philosophy of the Hugging Face ecosystem.
- Master model loading, tokenization, training, and generation APIs in depth.
- Connect LLM tooling to the broader ML engineering stack inside this vault.

## 📚 Course Roadmap

This deep dive is structured into 10 focused modules:

1. **[[00 - Welcome to HuggingFace Transformers Deep Dive|00 - Welcome]]** — Course overview, philosophy, and prerequisites.
2. **[[01 - The from_pretrained Ecosystem|01 - from_pretrained Ecosystem]]** — How models are discovered, downloaded, cached, and initialized.
3. **[[02 - Tokenizers and Data Processing|02 - Tokenizers & Data Processing]]** — Fast Rust-backed tokenization and dataset engineering.
4. **[[03 - Trainer, TrainingArguments, and Distributed Training|03 - Trainer & Distributed Training]]** — Production-grade training loops and scaling strategies.
5. **[[04 - Generation, Decoding, and Structured Output|04 - Generation & Structured Output]]** — Autoregressive inference, decoding, and controlled generation.
6. **05 - PEFT and Efficient Fine-Tuning** — LoRA, adapters, and parameter-efficient methods.
7. **06 - Quantization and ONNX Export** — GPTQ, AWQ, and model compression for inference.
8. **07 - Inference Servers and vLLM** — [[06 - Large Language Models/13 - vLLM Deep Dive|vLLM]], TGI, and throughput optimization.
9. **08 - Multimodal Transformers** — Vision-language and speech models in the HF ecosystem.
10. **09 - Building End-to-End Pipelines** — Integration with [[09 - MLOps y Produccion|MLOps]], [[10 - FastAPI|FastAPI]], and [[14 - Rust Engineering|Rust/Candle]].

## 🛠️ Prerequisites

- **Python**: Solid grasp of classes, decorators, context managers, and dunder methods.
- **PyTorch**: Familiarity with `nn.Module`, `Tensor` operations, `DataLoader`, and `torch.cuda`.
- **Docker**: Used for distributed training and reproducible environments.
- **Math**: Basic linear algebra, probability, and gradient-based optimization.
- **Foundations**: Completion of [[05 - Deep Learning y CV|Deep Learning fundamentals]] and exposure to [[06 - Large Language Models|LLM theory]].
- **Transformers Library**: Basic familiarity with `pipeline()` and `AutoModel` is helpful but not required.
- **Git**: Comfortable cloning repos, branching, and resolving merge conflicts.

## 🌍 Philosophy

The Hugging Face ecosystem democratizes AI by combining open model weights with open-source tooling. `transformers`, `datasets`, and `tokenizers` form the backbone of modern NLP engineering. Understanding these libraries at a deep level—beyond copy-pasting snippets—is what separates notebook experiments from production systems that serve millions of requests.

This course assumes you want to build, not just consume. We focus on WHY each abstraction exists, HOW it is implemented, and WHERE it connects to neighboring domains like [[09 - MLOps y Produccion|MLOps]], [[14 - Rust Engineering|Rust/Candle]], and [[06 - Large Language Models/13 - vLLM Deep Dive|vLLM inference]].

## 🔗 Related Vault Modules

| Module | Connection |
|--------|------------|
| [[06 - Large Language Models]] | Foundational LLM theory and architecture. |
| [[06 - Large Language Models/13 - vLLM Deep Dive|vLLM Deep Dive]] | High-throughput inference serving. |
| [[06 - Large Language Models/14 - Unsloth Deep Dive|Unsloth Deep Dive]] | Memory-efficient fine-tuning. |
| [[14 - Rust Engineering]] | [[14 - Rust Engineering/Candle Deep Dive|Candle]] for zero-overhead Rust inference. |
| [[09 - MLOps y Produccion]] | Model deployment, monitoring, and CI/CD. |
| [[10 - FastAPI]] | Serving models via REST and WebSocket APIs. |
| [[13 - Go Engineering]] | Microservices and high-performance gateways. |

## 📝 How to Use These Notes

Each note is designed as a standalone deep dive, but they follow a logical progression from loading models to training them at scale and serving them with controlled generation. Read them in order if you are new to the ecosystem, or jump directly to the module that addresses your current bottleneck. Every note includes a **Compression Code** block that you can paste into a script to see all concepts in action, and a **Documented Project** specification that connects the theory to a realistic engineering task.

We recommend setting up a dedicated Python environment with `transformers`, `datasets`, `tokenizers`, `accelerate`, and `huggingface_hub` installed before starting. Many snippets assume access to a CUDA-capable GPU, though most concepts translate to CPU or MPS backends.

---
*Happy learning! May your loss curves converge smoothly.*
