# 🏷️ Welcome to Unsloth and Efficient Fine-Tuning

## 🎯 Learning Objectives

- Master **Unsloth's kernel optimization** architecture and understand why hand-written Triton/CUDA kernels achieve 2-5x speedup over native HuggingFace fine-tuning
- Implement **QLoRA-based fine-tuning** with 4-bit NormalFloat quantization, double quantization, and paged optimizers to train Gemma 4 on a single GPU
- Execute **SFT (Supervised Fine-Tuning)** and **DPO (Direct Preference Optimization)** with Unsloth — from dataset formatting to hyperparameter tuning to evaluation
- Design and curate **high-quality fine-tuning datasets** using synthetic data generation (Self-Instruct, Evol-Instruct) and deduplication strategies
- Deploy fine-tuned models through **quantization (GGUF)**, vLLM serving, and production inference gateways — connecting to your existing LLM Edge Gateway infrastructure

---

## Why Unsloth Changes the Fine-Tuning Game

Full fine-tuning of a 7B-parameter model demands approximately 56 GB of GPU memory with AdamW — prohibitive for most practitioners. QLoRA reduced this to ~16 GB, but still required the overhead of HuggingFace's standard attention kernels. Unsloth takes it further: by **rewriting the attention mechanism and feedforward layers** in optimized Triton/CUDA kernels, you fine-tune 2-5x faster while using 50-70% less VRAM than stock HuggingFace+QLoRA. The result is training runs that finished in 4 hours now completing in 45 minutes on the same hardware.

This course directly connects to your existing projects. Your [[../../projects/03 - Fine-Tuning LLMs - Project Guide.md|Fine-Tuning LLMs project]] taught the PEFT fundamentals; here we go deeper into the kernel-level optimizations that make those techniques truly efficient. Your **LLM Edge Gateway** (Go/Fiber/Redis) will serve the Gemma 4 models you fine-tune here. Your **Automated LLM Evaluation Suite** (Python/AWS/GCP) will benchmark them. Your **Multi-Agent Research System** (LangGraph/Gemma 4) will consume them as specialized agent backbones. This is the course that closes the loop.

If you are preparing for AI/ML Engineer interviews, understanding Unsloth-level optimization — QLoRA, kernel fusion, hand-written backprop — separates you from candidates who only know `trainer.train()`. Companies like Anthropic, Cohere, and Together AI invest heavily in custom inference and training kernels. This knowledge is tested.

---

## Course Map

| # | Note | Core Focus | Time |
|---|------|-----------|------|
| 01 | [[01 - Unsloth Architecture and QLoRA Deep Dive.md\|Unsloth Architecture and QLoRA Deep Dive]] | Kernel optimization, memory math, Gemma 4 QLoRA setup | 3-5 h |
| 02 | [[02 - SFT, DPO, and Preference Alignment with Unsloth.md\|SFT, DPO, and Preference Alignment with Unsloth]] | SFT workflows, DPO theory, preference alignment, evaluation | 3-5 h |
| 03 | [[03 - Dataset Preparation and Curation for Fine-Tuning.md\|Dataset Preparation and Curation for Fine-Tuning]] | Synthetic data generation, deduplication, format conversion | 2-4 h |
| 04 | [[04 - Deployment and Optimization.md\|Deployment and Optimization]] | LoRA merging, GGUF quantization, vLLM serving, cost analysis | 3-4 h |

Each note is self-contained and includes theory BEFORE code, ASCII mental models, Mermaid diagrams, compression-ready scripts, and knowledge checks. Notes 01–02 are the core; notes 03–04 complete the production pipeline.

---

## Prerequisites

| Concept | Expected Knowledge | Review If Needed |
|---------|-------------------|-----------------|
| LLM fundamentals | Transformer architecture, attention mechanism, tokenization | [[../../02 - Large Language Models/06 - Fundamentos de LLMs/00 - Bienvenida\|LLM Fundamentals]] |
| PEFT concepts | LoRA, QLoRA, Adapters, gradient checkpointing | [[../03 - Fine-Tuning LLMs.md\|Fine-Tuning LLMs course]] |
| PyTorch training loop | Autograd, DataLoader, optimizer state dict | [[../../01 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida\|Deep Learning with PyTorch]] |
| HuggingFace ecosystem | `transformers`, `datasets`, `peft`, `trl` (SFTTrainer, DPOTrainer) | HF documentation + [[../03 - Fine-Tuning LLMs.md\|Fine-Tuning LLMs]] |
| GPU fundamentals | CUDA devices, VRAM, mixed precision (fp16/bf16) | NVIDIA Ampere architecture docs |
| Python ≥ 3.10 | Type hints, dataclasses, `torch.compile` | [[../../Advanced Python/00 - Bienvenida\|Advanced Python]] |
| Gemma 4 familiarity | Model family, tokenizer, chat template | Google Gemma docs + your portfolio projects |
| Linux + pip/conda | Environment management, CUDA toolkit installation | System administration basics |

> ⚠️ **GPU Requirement**: A GPU with ≥ 24 GB VRAM (RTX 4090, A10, A5000) is sufficient for Gemma 4 9B QLoRA fine-tuning. A 48 GB GPU (A6000, L40S) handles 27B comfortably. Connect to your [[../../06 - Cloud, Infra y Backend/22 - Cloud Computing/00 - Bienvenida\|Cloud Computing]] resources for on-demand access.

---

## Project Outcome

By the end of this course you will produce a versioned repository:

```
gemma4-unsloth-finetune/
├── notebooks/
│   ├── 01_qlora_finetune.ipynb
│   ├── 02_dpo_alignment.ipynb
│   └── 03_evaluation.ipynb
├── scripts/
│   ├── sft_train.py
│   ├── dpo_train.py
│   ├── merge_and_quantize.py
│   └── data_curation.py
├── configs/
│   ├── lora_config.yaml
│   └── deployment_config.yaml
├── models/
│   └── README.md  (HF links)
├── eval/
│   ├── mt_bench_results.json
│   └── alpaca_eval_results.json
└── README.md
```

This repository is **interview-ready**. Push it to GitHub, reference it when asked "How have you fine-tuned LLMs efficiently?", and connect it to your [[../../projects/00 - Project Planning Guide for ML\|ML Project Planning]] portfolio.

---

## How to Use These Notes

1. **Read the theory section first** — every module explains *why* before showing *how*. The memory formulas, kernel diagrams, and mental models are designed to build lasting intuition.
2. **Copy the compression code** — each note includes complete, production-ready scripts that you can paste into a file and execute immediately.
3. **Study the documented project** — the project walkthrough explains every design decision, from LoRA rank selection to dataset composition ratios.
4. **Link to your other vault notes** — when you see `[[...]]` links, click them to reinforce connections across courses. Knowledge compounds when you connect it.

> 💡 **Tip**: Start with Note 01 even if you have fine-tuned models before. The kernel-level explanation of *why* Unsloth is fast will change how you think about GPU training.
