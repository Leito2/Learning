# 🔧 Fine-Tuning LLMs — Project Guide

## Overview

Why this project matters for a first job.

LLM fine-tuning is one of the most in-demand skills for 2025-2026 ML engineering roles. Companies need engineers who can adapt open-source models to private data without retraining billions of parameters from scratch. This guide teaches parameter-efficient fine-tuning (LoRA/QLoRA) and how to publish your work for recruiters to see.

## Prerequisites

- PyTorch basics
- HuggingFace `transformers` and `datasets` familiarity
- A GPU (Colab, Kaggle, or local CUDA)
- Optional: HuggingFace Hub token for publishing models

## Learning Objectives

1. Understand LoRA and QLoRA mechanics and when to use each.
2. Prepare and tokenize custom datasets for causal language modeling.
3. Fine-tune a decoder model with the HuggingFace PEFT and TRL libraries.
4. Evaluate fine-tuned models with generation metrics and human inspection.
5. Publish reproducible training code and the model card to HuggingFace Hub.

## Official Resources & Links

| Resource | Type | URL | Why It Matters |
|----------|------|-----|----------------|
| HuggingFace | Platform | https://huggingface.co | Central hub for models, datasets, and inference APIs |
| PEFT | Library | https://github.com/huggingface/peft | Parameter-efficient fine-tuning (LoRA, QLoRA, adapters) |
| TRL | Library | https://github.com/huggingface/trl | Transformer Reinforcement Learning; includes `SFTTrainer` |
| Unsloth | Library | https://github.com/unslothai/unsloth | 2-5x faster fine-tuning with 70% less memory |
| DeepSpeed | Library | https://www.deepspeed.ai | ZeRO optimization for large-model training on limited GPUs |
| LLM Course | Course | https://huggingface.co/learn/nlp-course | Free, comprehensive NLP course from HuggingFace |

## Architecture & Planning

### Fine-Tuning Pipeline

```mermaid
flowchart LR
    A[Base LLM<br/>e.g. Llama-3-8B] --> B[Quantization<br/>4-bit / 8-bit]
    B --> C[LoRA Adapter<br/>Rank r, Alpha]
    C --> D[Custom Dataset<br/>Tokenize + Format]
    D --> E[SFT Training<br/>TRL / PEFT]
    E --> F[Evaluation<br/>Perplexity + Generation]
    F --> G[Merge Adapter<br/>Optional]
    G --> H[Publish to Hub<br/>Model Card]
```

### Key Decisions

- **LoRA vs QLoRA**: Use QLoRA when GPU memory is under 24 GB; it quantizes the base model to 4-bit and trains adapters in 16-bit.
- **Rank (r)**: Start with `r=8` or `r=16`. Increase if underfitting.
- **Dataset format**: Instruction-following datasets need a chat template (e.g., Alpaca, ShareGPT).
- **Merge adapters**: Merge only if you need a single file for serving; keep adapters separate for modularity.

## Step-by-Step Implementation Guide

1. **Set up the environment**
   - What: Install `transformers`, `peft`, `trl`, `bitsandbytes`, and `accelerate`.
   - Why: These libraries form the standard open-source LLM fine-tuning stack.
   - Code:
     ```bash
     pip install transformers peft trl bitsandbytes accelerate
     ```
   - Expected output: All imports succeed without version conflicts.

2. **Load and quantize the base model**
   - What: Load a causal LM in 4-bit using `BitsAndBytesConfig`.
   - Why: Enables large-model fine-tuning on consumer GPUs.
   - Code: See the Guide Class below.
   - Expected output: A model loaded with ~5-7 GB VRAM for an 8B parameter model.

3. **Prepare the dataset**
   - What: Load a HuggingFace dataset or a local JSON/CSV, then format prompts.
   - Why: LLMs are fine-tuned on sequences; the prompt format directly impacts output quality.
   - Code:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
     ```
   - Expected output: A `Dataset` object with a `text` column containing formatted instructions.

4. **Configure LoRA**
   - What: Define target modules (usually `q_proj`, `v_proj`) and hyperparameters.
   - Why: Only 1-2% of parameters are trained, dramatically reducing compute.
   - Code:
     ```python
     from peft import LoraConfig
     lora_config = LoraConfig(
         r=16,
         lora_alpha=32,
         target_modules=["q_proj", "v_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type="CAUSAL_LM",
     )
     ```
   - Expected output: A `LoraConfig` object ready for `get_peft_model`.

5. **Train with SFTTrainer**
   - What: Use `trl`'s `SFTTrainer` for supervised fine-tuning.
   - Why: Handles dataset packing, tokenization, and PEFT integration in one API.
   - Code: See the Guide Class below.
   - Expected output: Training loss decreasing steadily; validation perplexity stable or improving.

6. **Evaluate generations**
   - What: Run the model on held-out prompts and inspect outputs.
   - Why: Quantitative metrics (perplexity) do not capture instruction-following quality.
   - Code:
     ```python
     inputs = tokenizer("Translate to Spanish: Hello world", return_tensors="pt").to("cuda")
     outputs = model.generate(**inputs, max_new_tokens=50)
     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
     ```
   - Expected output: A coherent, relevant response.

7. **Save and publish**
   - What: Push adapters (or merged model) and a model card to HuggingFace Hub.
   - Why: Creates a permanent portfolio artifact that recruiters can test instantly.
   - Code:
     ```python
     model.push_to_hub("your-username/my-lora-adapter")
     tokenizer.push_to_hub("your-username/my-lora-adapter")
     ```
   - Expected output: A public repo on HuggingFace Hub with a working inference widget.

## Guide Class / Example

Complete copy-pasteable code.

```python
"""finetune_lora.py — Complete LoRA fine-tuning script."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ---------------------------
# 1. Configuration
# ---------------------------
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Replace with any causal LM
DATASET_NAME = "tatsu-lab/alpaca"
OUTPUT_DIR = "./llama2-lora-alpaca"

# ---------------------------
# 2. Quantization config
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------
# 3. Load model and tokenizer
# ---------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# ---------------------------
# 4. LoRA config
# ---------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ---------------------------
# 5. Dataset
# ---------------------------
dataset = load_dataset(DATASET_NAME, split="train[:1000]")


def format_prompt(example):
    """Alpaca-style prompt formatting."""
    if example["input"]:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


formatted_dataset = dataset.map(format_prompt)

# ---------------------------
# 6. Training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    group_by_length=True,
)

# ---------------------------
# 7. Train
# ---------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

model.config.use_cache = False
trainer.train()

# ---------------------------
# 8. Save
# ---------------------------
trainer.save_model(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# ---------------------------
# 9. Inference test
# ---------------------------
prompt = "### Instruction:\nWrite a haiku about machine learning.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Common Pitfalls & Checklist

⚠️ **Forgetting pad_token** — Most base LLMs lack a pad token. Set it to `eos_token` or training will crash.

⚠️ **Too high learning rate** — LoRA is sensitive. Start with `2e-4` and reduce if loss spikes.

⚠️ **Not using gradient checkpointing** — Without it, 4-bit models can still OOM on long sequences.

⚠️ **Evaluating on training prompts** — Always test on unseen instructions to measure generalization.

| Checkpoint | Status |
|------------|--------|
| Base model loads in 4-bit without OOM | ☐ |
| Dataset formatted with a clear instruction template | ☐ |
| LoRA config targets correct projection layers | ☐ |
| Training loss decreases over 50+ steps | ☐ |
| Model generates coherent text on held-out prompts | ☐ |
| Adapter pushed to HuggingFace Hub with model card | ☐ |

## Deployment & Portfolio Integration

How to deploy and present for recruiters.

- **HuggingFace Hub**: Publish the adapter and create a detailed model card (dataset, training config, hardware used, evaluation samples).
- **HuggingFace Space**: Build a Gradio chat interface that loads your adapter for live demos.
- **GitHub**: Push the training script, requirements, and a `README` with reproduction instructions and sample outputs.
- **Resume**: "Fine-tuned Llama-2-7B with QLoRA on Alpaca dataset, reducing training memory by 75% while maintaining instruction-following accuracy."
- **Blog Post**: Explain one technical choice (e.g., why you picked `r=16` or how you handled context length).

## Next Steps

- [[00 - Project Planning Guide for ML and AI Engineering]]
- [[01 - Kaggle Competitions — Project Guide]]
- [[02 - End-to-End ML Project — Project Guide]]
