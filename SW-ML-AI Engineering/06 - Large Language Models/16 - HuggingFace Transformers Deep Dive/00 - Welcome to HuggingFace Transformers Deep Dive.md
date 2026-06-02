# 🤗 Welcome to HuggingFace Transformers Deep Dive

## 🎯 Learning Objectives

- Master model loading, tokenization, training, and generation APIs in depth
- Understand the architecture and design philosophy of the Hugging Face ecosystem
- Connect LLM tooling to the broader ML engineering stack inside this vault

## 📚 Course Roadmap

| Note | Title | What You'll Master |
|------|-------|-------------------|
| `00` | **Welcome** | Course overview, philosophy, prerequisites |
| `01` | **The from_pretrained Ecosystem** | Model discovery, download, caching, initialization, configs |
| `02` | **Tokenizers and Data Processing** | Fast Rust-backed tokenization, dataset engineering, `datasets` library |
| `03` | **Trainer, TrainingArguments, and Distributed Training** | Production-grade training loops, scaling strategies, Accelerate |
| `04` | **Generation, Decoding, and Structured Output** | Autoregressive inference, beam/search, grammar-constrained output |
| `05` | **Vision, Audio, and Multimodal Transformers** | ViT, Whisper, CLIP, multimodal architectures in HF |
| `06` | **Export, Optimization, and Production Serving** | ONNX export, quantization (GPTQ/AWQ), Optimum, TGI |
| `07` | **Diffusers I — Stable Diffusion Fundamentals** | Diffusion core, pipelines, schedulers, memory optimization |
| `08` | **Diffusers II — Advanced Pipelines and ControlNet** | Image-to-image, ControlNet, LoRA, custom pipelines |
| `09` | **Capstone — End-to-End HF Pipeline** | Full stack: training → export → serving → monitoring |
| `10` | **FSDP Deep Dive** *(advanced extension)* | PyTorch-native sharded data parallelism: memory math, `ShardingStrategy` enum, `auto_wrap_policy`, YAML config, FSDP2 (`fully_shard`), FSDP + LoRA / PEFT integration |

## 🛠️ Prerequisites

- **Python**: Classes, decorators, context managers, dunder methods
- **PyTorch**: `nn.Module`, `Tensor`, `DataLoader`, `torch.cuda`
- **Docker**: For distributed training and reproducible environments
- **Math**: Linear algebra, probability, gradient-based optimization
- **Foundations**: Completion of [[05 - Deep Learning y Computer Vision]] and exposure to [[06 - Large Language Models]]
- **Git**: Comfortable with branching and merge conflicts

## 🌍 Philosophy

The Hugging Face ecosystem democratizes AI by combining open model weights with open-source tooling. `transformers`, `datasets`, and `tokenizers` form the backbone of modern NLP engineering. Understanding these libraries at a deep level—beyond copy-pasting snippets—is what separates notebook experiments from production systems that serve millions of requests.

This course assumes you want to build, not just consume. We focus on **why** each abstraction exists, **how** it is implemented, and **where** it connects to neighboring domains like [[09 - MLOps y Produccion]], [[10 - Cloud, Infra y Backend]], and [[06 - Large Language Models]].

## 🔗 Related Vault Modules

| Module | Connection |
|--------|------------|
| [[06 - Large Language Models]] | Foundational LLM theory and architecture |
| [[06 - Large Language Models/13 - vLLM and Advanced RAG]] | High-throughput inference serving |
| [[06 - Large Language Models/14 - Unsloth]] | Memory-efficient fine-tuning |
| [[09 - MLOps y Produccion]] | Model deployment, monitoring, CI/CD |
| [[10 - Cloud, Infra y Backend]] | Serving infrastructure, FastAPI, Docker |

## 📝 How to Use These Notes

Each note is designed as a standalone deep dive following a logical progression: loading models → processing data → training → generation → export/deplyment. Every note includes a **Compression Code** block (runnable script) and inline warnings with production tips.

Set up a dedicated Python environment with `transformers`, `datasets`, `tokenizers`, `accelerate`, and `huggingface_hub` installed before starting. Many snippets assume CUDA, though most concepts translate to CPU or MPS.
