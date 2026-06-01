# 🏗️ Capstone Project - End-to-End HF Transformers Pipeline

## 🎯 Learning Objectives

- Architect a complete MLOps pipeline using the HuggingFace ecosystem from ingestion to serving
- Load, tokenize, and fine-tune a model with `datasets`, `AutoTokenizer`, and `Trainer`
- Evaluate model performance with the `evaluate` library and structured metrics
- Export fine-tuned models to ONNX via `optimum` for hardware-agnostic inference
- Build a production-grade FastAPI service with sub-200ms p99 latency
- Integrate a `diffusers` image generation endpoint as a multimodal bonus
- Containerize the entire stack with Docker Compose for reproducible deployment

---

## Introduction

This capstone is the culmination of the HuggingFace Transformers Deep Dive. Where [[00 - Welcome to HuggingFace Transformers Deep Dive]] introduced the `from_pretrained` philosophy and [[02 - Tokenizers and Data Processing]] covered preprocessing, this project fuses every layer into a deployable system. You will not write isolated snippets; you will build a cohesive pipeline that data scientists and ML engineers actually ship to production.

The pipeline follows a canonical MLOps pattern: **ingestion → transformation → training → evaluation → optimization → serving**. We intentionally reuse patterns from [[03 - Trainer, TrainingArguments, and Distributed Training]] for fine-tuning, [[06 - Export, Optimization, and Production Serving]] for ONNX export, and [[07 - Diffusers I - Stable Diffusion Fundamentals]] plus [[08 - Diffusers II - Advanced Pipelines and ControlNet]] for the multimodal bonus. By the end, you will have a Docker Compose stack you can run locally or deploy to a cloud VM.

The guiding principle is **contractual interfaces**: each stage produces artifacts with well-defined schemas that the next stage consumes. When these contracts are explicit, teams can iterate in parallel without breaking downstream stages.

---

## 1. Pipeline Architecture and Contracts

### The DAG Abstraction

An end-to-end ML pipeline is more than a training script. It is a system of four contracts:

- **Data contract**: Dataset schema, splits, and column semantics
- **Model contract**: Input shapes, output logits, and label mapping
- **Evaluation contract**: Metrics, baselines, and acceptance criteria
- **Serving contract**: Latency SLAs, throughput targets, and request/response schemas

The HuggingFace ecosystem provides canonical abstractions for each contract. `datasets` enforces the data contract through `DatasetDict` splits with typed features. `transformers` enforces the model contract via `forward()` signatures that define input/output tensor shapes. `evaluate` supplies benchmark-aligned metrics for the evaluation contract. And `optimum` preserves the model contract while switching the runtime from PyTorch to ONNX.

Treat the pipeline as a directed acyclic graph (DAG). The training node consumes tokenized datasets and produces checkpoints; the export node consumes checkpoints and produces ONNX graphs; the API node consumes ONNX and exposes REST endpoints. This DAG mentality enables CI/CD: change the tokenizer, and all downstream nodes must be invalidated and re-run.

```python
# config.py — centralized configuration enforces contracts across stages
from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    model_id: str = "distilbert-base-uncased"
    dataset_id: str = "emotion"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    onnx_path: str = "./onnx_model/model.onnx"
```

❌ **Antipattern**: Hardcoding `max_length=128` in the training script and `max_length=256` in the serving script. The ONNX graph was traced with sequence length 128; requests with longer inputs cause shape mismatch errors or silent truncation.

✅ **Correct**: Use a single `PipelineConfig` dataclass imported by every stage. The frozen dataclass ensures immutability—no stage can accidentally override a parameter.

### Dataset Contract

Load from HuggingFace Hub with explicit revision pinning:

```python
from datasets import load_dataset

dataset = load_dataset("emotion")
label_names = dataset["train"].features["label"].names
print(f"Classes: {label_names}, Splits: {list(dataset.keys())}")
```

❌ **Antipattern**: Loading a dataset without pinning the revision. The Hub dataset may update its splits or column names, silently breaking your `map()` call.

✅ **Correct**: Use `load_dataset(..., revision="abc123")` to pin a specific version. Validate expected columns at runtime with `assert "text" in dataset["train"].column_names`.

---

## 2. Training and Evaluation

### The Training Loop

The `Trainer` API from [[03 - Trainer, TrainingArguments, and Distributed Training]] abstracts the training loop, but you must still understand what happens inside: collation, forward pass, cross-entropy reduction, and optimization. The cross-entropy loss for sequence classification is:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(f_{\theta}(x_i)_{y_i})}{\sum_{c=1}^C \exp(f_{\theta}(x_i)_c)}$$

where $B$ is the batch size, $C$ is the number of classes, $f_{\theta}(x_i)_c$ is the logit for class $c$, and $y_i$ is the ground truth label.

`TrainingArguments` is the control surface. Misconfigured `logging_steps` or `evaluation_strategy` leads to silent failures or excessive compute. The `DataCollatorWithPadding` dynamically pads each batch to the longest sequence, avoiding wasted compute on static padding:

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
import evaluate
import wandb
from config import PipelineConfig

cfg = PipelineConfig()

dataset = load_dataset(cfg.dataset_id)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
label_names = dataset["train"].features["label"].names
model = AutoModelForSequenceClassification.from_pretrained(
    cfg.model_id, num_labels=len(label_names)
)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

tokenized = dataset.map(preprocess, batched=True, remove_columns=["text"])

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.batch_size,
    num_train_epochs=cfg.num_epochs,
    logging_steps=50,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

wandb.init(project="hf-capstone", config=cfg.__dict__)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./hf_model")
```

❌ **Antipattern**: Forgetting `remove_columns` during `Dataset.map`. Leftover string columns cause the default collator to crash because it cannot tensorize text.

✅ **Correct**: Always pass `remove_columns=dataset["train"].column_names` after tokenization. Only keep tensor-safe columns (`input_ids`, `attention_mask`, `labels`).

💡 **Tip**: Use `weighted` F1 as the primary metric for imbalanced datasets. Accuracy is misleading when one class dominates—a model predicting only the majority class achieves high accuracy while being useless.

### Evaluation Contract

Evaluation is not an afterthought. The `evaluate` library mirrors academic benchmarks. For the `emotion` dataset (6 classes with class imbalance), F1 is more reliable than accuracy. The weighted F1 accounts for class support:

$$\text{F1}_{\text{weighted}} = \sum_{c=1}^C w_c \cdot \frac{2 \cdot \text{precision}_c \cdot \text{recall}_c}{\text{precision}_c + \text{recall}_c}$$

where $w_c$ is the proportion of class $c$ in the dataset. A model achieving F1 > 0.85 on the emotion dataset is considered production-ready.

Logging to Weights & Biases creates an audit trail linking hyperparameters to metrics, essential for reproducibility:

```python
wandb.init(project="hf-capstone", config=cfg.__dict__)
```

**Caso real**: A customer support startup uses this exact pipeline to triage incoming tickets into 12 categories. They fine-tune DistilBERT on their labeled dataset and use the weighted F1 metric because 40% of tickets are "billing" (majority class) while "security" has only 2% support. Accuracy would hide the model's failure on rare but critical classes; F1 reveals it.

---

## 3. Optimization and Serving

### ONNX Export

Training produces a PyTorch checkpoint, but PyTorch is rarely the optimal runtime for production inference. ONNX defines a standard graph representation that executes on ONNX Runtime, TensorRT, OpenVINO, or browser-based runtimes. `optimum` from [[06 - Export, Optimization, and Production Serving]] wraps torch-to-ONNX tracing into a single API.

The key abstraction during export is the "dummy input." Tracing executes the model once with fake inputs and records all operations into a static graph. Because transformer graphs are dynamic in sequence length, we fix ONNX input shapes to `batch_size` and `max_length`, trading flexibility for predictable latency and memory:

```python
# export.py
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from config import PipelineConfig

cfg = PipelineConfig()

model = ORTModelForSequenceClassification.from_pretrained("./hf_model", export=True)
tokenizer = AutoTokenizer.from_pretrained("./hf_model")

model.save_pretrained("./onnx_model")
tokenizer.save_pretrained("./onnx_model")
```

### FastAPI Serving

Serving requires more than a model file. A production API needs input validation, health checks, latency monitoring, and error handling. FastAPI provides async request handling and automatic OpenAPI docs. Paired with ONNX Runtime, a single CPU core serves DistilBERT in under 100 ms:

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from diffusers import StableDiffusionPipeline
import torch
import time
from config import PipelineConfig

app = FastAPI(title="HF Capstone API")
cfg = PipelineConfig()

tokenizer = AutoTokenizer.from_pretrained("./onnx_model")
ort_model = ORTModelForSequenceClassification.from_pretrained("./onnx_model")

sd_pipe = None
if torch.cuda.is_available():
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=cfg.max_length)
    outputs = ort_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    confidence, pred_id = torch.max(probs, dim=-1)
    latency = (time.perf_counter() - start) * 1000
    label_names = ort_model.config.id2label
    return PredictResponse(
        label=label_names[pred_id.item()],
        confidence=confidence.item(),
        latency_ms=round(latency, 2),
    )

@app.get("/health")
def health():
    return {"status": "ok", "onnx_loaded": ort_model is not None}

@app.post("/generate")
def generate_image(prompt: str):
    if sd_pipe is None:
        return {"error": "Stable Diffusion not loaded (GPU unavailable)"}
    image = sd_pipe(prompt, num_inference_steps=25).images[0]
    path = f"/tmp/{hash(prompt)}.png"
    image.save(path)
    return {"image_path": path}
```

❌ **Antipattern**: Recreating the ONNX `InferenceSession` on every request. Session creation parses the graph, allocates memory, and applies optimizations—it can take hundreds of milliseconds.

✅ **Correct**: Load models as module-level singletons or in FastAPI's lifespan handler. The model is loaded once at startup and reused for every request.

❌ **Antipattern**: Serving PyTorch directly in production without ONNX export. PyTorch's eager execution and Python overhead add 30–50% latency compared to ONNX Runtime, and the dependency chain (CUDA, PyTorch version) makes deployment brittle.

✅ **Correct**: Export to ONNX for any production deployment. The minimal viable pattern is FastAPI + ONNX Runtime on CPU, which achieves p99 < 200 ms for BERT-family models.

![HuggingFace ecosystem overview](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Hugging_Face_logo.svg/2560px-Hugging_Face_logo.svg.png)

💡 **Tip**: Containerize the API with Docker and use Docker Compose to orchestrate the stack. A `docker-compose.yml` with the API service, Redis cache, and Prometheus metrics creates a production-ready deployment:

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./onnx_model:/app/onnx_model
  redis:
    image: redis:7-alpine
```

**Caso real**: A real-estate startup deploys this exact stack on a single AWS t3.large (2 vCPU, 8 GB RAM) to classify property descriptions into categories (apartment, house, commercial). With ONNX Runtime, p99 latency is 120 ms on CPU, 60 ms on ARM Graviton. The Docker Compose stack takes 30 seconds to deploy via ECS, and the `id2label` config ensures label consistency across re-deployments.

---

## 📦 Compression Code

```python
#!/usr/bin/env python3
"""End-to-end HF capstone: load -> tokenize -> train -> evaluate -> export -> serve."""
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from optimum.onnxruntime import ORTModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import torch, time, evaluate, wandb

@dataclass(frozen=True)
class Cfg:
    model_id: str = "distilbert-base-uncased"
    dataset_id: str = "emotion"
    max_length: int = 128
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 3

CFG = Cfg()
app = FastAPI()

ds = load_dataset(CFG.dataset_id)
tok = AutoTokenizer.from_pretrained(CFG.model_id)
model = AutoModelForSequenceClassification.from_pretrained(CFG.model_id, num_labels=6)

def preprocess(b):
    return tok(b["text"], truncation=True, max_length=CFG.max_length)

tok_ds = ds.map(preprocess, batched=True, remove_columns=["text"])
acc, f1 = evaluate.load("accuracy"), evaluate.load("f1")

def metrics(ep):
    logits, labels = ep
    preds = logits.argmax(-1)
    return {"acc": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]}

args = TrainingArguments("./results", evaluation_strategy="epoch", save_strategy="epoch",
                         learning_rate=CFG.lr, per_device_train_batch_size=CFG.batch_size,
                         num_train_epochs=CFG.epochs, logging_steps=50, report_to="wandb",
                         load_best_model_at_end=True, metric_for_best_model="f1")
wandb.init(project="hf-capstone", config=CFG.__dict__)
trainer = Trainer(model, args, train_dataset=tok_ds["train"], eval_dataset=tok_ds["validation"],
                  tokenizer=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train(); trainer.save_model("./hf_model")

ort = ORTModelForSequenceClassification.from_pretrained("./hf_model", export=True)
ort.save_pretrained("./onnx_model"); tok.save_pretrained("./onnx_model")

ort_model = ORTModelForSequenceClassification.from_pretrained("./onnx_model")
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

class Req(BaseModel): text: str
class Resp(BaseModel): label: str; confidence: float; latency_ms: float

@app.post("/predict", response_model=Resp)
def predict(r: Req):
    t0 = time.perf_counter()
    inp = tokenizer(r.text, return_tensors="pt", truncation=True, max_length=CFG.max_length)
    out = ort_model(**inp)
    prob = torch.softmax(out.logits, dim=-1)
    conf, pid = torch.max(prob, dim=-1)
    lat = (time.perf_counter() - t0) * 1000
    return Resp(label=ort_model.config.id2label[pid.item()], confidence=conf.item(), latency_ms=round(lat, 2))

@app.get("/health")
def health(): return {"status": "ok"}
```

## 🎯 Key Takeaways

- A single `PipelineConfig` dataclass prevents training/serving contract mismatches and should be imported by every pipeline stage.
- `Trainer` + `DataCollatorWithPadding` + `evaluate` forms a complete, reproducible training loop with minimal custom code.
- Always pin dataset revisions and validate column schemas to prevent silent data contract breaks.
- `optimum` abstracts ONNX export complexity, but tokenization shapes (`max_length`) must be identical between training and inference.
- FastAPI + ONNX Runtime achieves p99 < 200 ms on CPU for BERT-family classification models.
- Weighted F1 is safer than accuracy for imbalanced classification—never deploy with accuracy as the sole evaluation metric.
- Docker Compose turns a collection of Python scripts into a reproducible system that any team member can run with `docker compose up`.

## References

- HuggingFace Transformers Documentation: https://huggingface.co/docs/transformers
- HuggingFace Datasets Documentation: https://huggingface.co/docs/datasets
- HuggingFace Optimum ONNX Export: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models
- FastAPI Documentation: https://fastapi.tiangolo.com
- ONNX Runtime: https://onnxruntime.ai
- Weights & Biases: https://docs.wandb.ai
- HuggingFace Diffusers: https://huggingface.co/docs/diffusers
- `emotion` Dataset: https://huggingface.co/datasets/emotion
