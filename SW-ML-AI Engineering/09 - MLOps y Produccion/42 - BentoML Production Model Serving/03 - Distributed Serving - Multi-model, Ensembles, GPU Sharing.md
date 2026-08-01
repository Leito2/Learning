# 🎯 03 - Distributed Serving — Multi-Model, Ensembles, GPU Sharing

> **Scale out your BentoML service. Multi-model platforms, ensembles, GPU sharing, dynamic batching, and the patterns that make inference scale.**

## 🎯 Learning Objectives
- Serve multiple models in one BentoML service
- Build model ensembles (voting, averaging, stacking)
- Share GPUs across services with fractional GPU allocation
- Implement dynamic batching for LLM and vision models
- Use the runner pattern for async + batched inference
- Build a multi-model platform with model routing
- Apply to portfolio projects: the StayBot recommendation engine

## Introduction

Most production ML services don't serve one model — they serve **multiple models** for different tasks, ensembles for higher accuracy, and complex pipelines that combine models. BentoML handles all of this through:

| Pattern | Use case |
|---------|----------|
| **Multi-model service** | One service serves 10 models (e.g., one per tenant) |
| **Model ensemble** | Combine 3 classifiers via voting for higher accuracy |
| **GPU sharing** | Run 4 models on 1 GPU with fractional allocation |
| **Dynamic batching** | GPU models with adaptive batch size |
| **Model router** | Pick the right model per request |

This note covers the patterns.

---

## 1. Multi-Model in One Service

```python
import bentoml
from pydantic import BaseModel


class PredictRequest(BaseModel):
    task: str  # "sentiment", "translation", "summarization"
    text: str


@bentoml.service
class MultiModelService:
    sentiment_model = bentoml.models.BentoModel("sentiment_classifier:latest")
    translation_model = bentoml.models.BentoModel("en_es_translator:latest")
    summarization_model = bentoml.models.BentoModel("summarizer:latest")
    
    def __init__(self):
        # Load all models at startup
        from transformers import pipeline
        self.sentiment = pipeline("sentiment-analysis", model=self.sentiment_model.path)
        self.translator = pipeline("translation_en_to_es", model=self.translation_model.path)
        self.summarizer = pipeline("summarization", model=self.summarization_model.path)
    
    @bentoml.api(batchable=True, max_batch_size=32)
    def predict(self, requests: list[PredictRequest]) -> list[dict]:
        results = []
        for req in requests:
            if req.task == "sentiment":
                result = self.sentiment(req.text)[0]
                results.append({"task": "sentiment", "label": result["label"], "score": result["score"]})
            elif req.task == "translation":
                result = self.translator(req.text)[0]
                results.append({"task": "translation", "translation": result["translation_text"]})
            elif req.task == "summarization":
                result = self.summarizer(req.text, max_length=130, min_length=30, do_sample=False)[0]
                results.append({"task": "summarization", "summary": result["summary_text"]})
            else:
                results.append({"error": f"unknown task: {req.task}"})
        return results
```

The service auto-batches requests for the same model. Three models in one container.

---

## 2. Model Ensembles for Higher Accuracy

```python
import bentoml
import numpy as np


@bentoml.service
class EnsembleService:
    model_a = bentoml.models.BentoModel("xgboost_classifier:latest")
    model_b = bentoml.models.BentoModel("random_forest:latest")
    model_c = bentoml.models.BentoModel("neural_net:latest")
    
    def __init__(self):
        import joblib
        self.a = joblib.load(self.model_a.path_of("model.pkl"))
        self.b = joblib.load(self.model_b.path_of("model.pkl"))
        self.c = joblib.load(self.model_c.path_of("model.pkl"))
    
    @bentoml.api(batchable=True, max_batch_size=64)
    def predict(self, input_data: list[list[float]]) -> list[dict]:
        # Get predictions from each model
        preds_a = self.a.predict_proba(input_data)
        preds_b = self.b.predict_proba(input_data)
        preds_c = self.c.predict_proba(input_data)
        
        # Weighted average
        weights = [0.4, 0.3, 0.3]  # tuned via validation
        ensemble = (
            weights[0] * preds_a +
            weights[1] * preds_b +
            weights[2] * preds_c
        )
        
        return [
            {
                "prediction": int(ensemble[i].argmax()),
                "confidence": float(ensemble[i].max()),
                "model_votes": [
                    {"model": "xgboost", "vote": int(preds_a[i].argmax())},
                    {"model": "random_forest", "vote": int(preds_b[i].argmax())},
                    {"model": "neural_net", "vote": int(preds_c[i].argmax())},
                ],
            }
            for i in range(len(input_data))
        ]
```

The ensemble gives:
- Higher accuracy than any single model
- Robustness (if one model fails, others still vote)
- Information about model disagreement (useful for uncertainty)

---

## 3. GPU Sharing with Fractional Allocation

For expensive GPUs, run multiple services on one GPU:

```yaml
# Service A: 30% of GPU memory
# bentoml serve --resources gpu=0.3 service_a:Svc

# Service B: 30% of GPU memory
# bentoml serve --resources gpu=0.3 service_b:Svc

# Reserved for OS: 40%
```

Or via BentoML config:

```python
@bentoml.service(
    resources={"gpu": "1"},  # full GPU
    # Or fractional: {"gpu": "0.5"} for half a GPU
)
class BigModelService:
    model = bentoml.models.BentoModel("big_model:latest")
    
    def __init__(self):
        # Load into a specific GPU
        import torch
        self.device = torch.device("cuda:0")
        # ...
    
    @bentoml.api
    def predict(self, input_data):
        # Use only 50% of the GPU
        with torch.cuda.device(self.device):
            return self.model_runner.predict(input_data)
```

For Kubernetes, use NVIDIA's Multi-Instance GPU (MIG) or fractional GPU allocation via the device plugin.

---

## 4. Dynamic Batching for LLMs and Vision Models

For GPU-bound models, batch multiple requests to amortize GPU overhead:

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "16Gi"},
)
class VisionModelService:
    model = bentoml.models.BentoModel("vit_base_patch16:latest")
    
    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=50,  # wait up to 50ms to fill the batch
    )
    def predict(self, images: list[Image]) -> list[dict]:
        """Vision transformer batched inference."""
        # BentoML groups requests into batches of up to 32
        # OR triggers after 50ms (whichever first)
        batched_tensor = self.preprocess(images)
        
        with torch.no_grad():
            outputs = self.model_runner.predict(batched_tensor)
        
        return self.postprocess(outputs)
```

For LLM inference, dynamic batching is the difference between:

| Pattern | Throughput |
|---------|------------|
| Single request | 1 request / 50ms = 20 RPS |
| Dynamic batch (8) | 8 requests / 100ms = 80 RPS |
| Dynamic batch (32) | 32 requests / 200ms = 160 RPS |

---

## 5. The Runner Pattern

The `bentoml.models.BentoModel` returns a reference; the runner loads and manages the model:

```python
@bentoml.service
class RunnerService:
    model = bentoml.models.BentoModel("model:latest")
    
    def __init__(self):
        # Load the runner (handles GPU, async, batching)
        self.runner = bentoml.sklearn.load_model_runner(self.model)
    
    @bentoml.api(batchable=True)
    async def predict(self, inputs: list[list[float]]) -> list[list[float]]:
        # Use the runner (not the raw model)
        # Runner handles batching + GPU + async automatically
        return await self.runner.async_predict(inputs)
```

The runner:
- Handles GPU placement (single GPU, multi-GPU, fractional)
- Supports async + sync modes
- Automatic batching
- Lifecycle management

---

## 6. Multi-Model Platform with Routing

A model router directs requests to the right model:

```python
from typing import Annotated
import bentoml
from fastapi import Header, HTTPException


@bentoml.service
class ModelPlatform:
    # Load all available models
    sentiment = bentoml.models.BentoModel("sentiment:latest")
    translation = bentoml.models.BentoModel("translation:latest")
    summarization = bentoml.models.BentoModel("summarization:latest")
    
    def __init__(self):
        from transformers import pipeline
        self.models = {
            "sentiment": pipeline("sentiment-analysis", model=self.sentiment.path),
            "translation": pipeline("translation_en_to_es", model=self.translation.path),
            "summarization": pipeline("summarization", model=self.summarization.path),
        }
    
    @bentoml.api
    def predict(self, model_name: str, text: str) -> dict:
        """Route to the requested model."""
        if model_name not in self.models:
            raise HTTPException(404, f"Unknown model: {model_name}")
        
        result = self.models[model_name](text)
        return {"model": model_name, "result": result}
```

The platform serves many models; users pick which to call.

---

## 7. Hot-Swap Models Without Restart

For A/B testing or model rollouts:

```python
class SwapService:
    def __init__(self):
        # Load default model
        self.active_model = bentoml.sklearn.load_model("iris_classifier:prod")
        self.active_version = "prod"
    
    @bentoml.api
    def set_active_model(self, version: str) -> dict:
        """Swap the active model at runtime (no restart)."""
        # Validate version exists
        try:
            new_model = bentoml.sklearn.load_model(f"iris_classifier:{version}")
        except Exception:
            raise HTTPException(404, f"Version {version} not found")
        
        # Atomic swap
        old_model = self.active_model
        old_version = self.active_version
        self.active_model = new_model
        self.active_version = version
        
        # Return old model to GC
        del old_model
        
        return {"old_version": old_version, "new_version": self.active_version}
    
    @bentoml.api
    def predict(self, input_data: list[list[float]]) -> list[int]:
        return self.active_model.predict(input_data).tolist()
```

Call `set_active_model` to swap; subsequent `predict` calls use the new model. No restart, no downtime.

---

## 8. Scaling BentoML Services

### 8.1 Horizontal scaling (multiple replicas)

```bash
# Kubernetes
bentoml deploy iris_service:latest --platform kubernetes --replicas 3
```

### 8.2 Vertical scaling (more resources per replica)

```bash
bentoml deploy iris_service:latest --platform kubernetes --replicas 1 --cpu 8 --memory 16Gi --gpu 1
```

### 8.3 Auto-scaling on load

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 9. Antipatterns

### 9.1 Antipattern 1: One service per model

```python
# ❌ One service for sentiment, another for translation
# Deploy 10 services for 10 models — operational nightmare

# ✅ One service with all models
@bentoml.service
class MultiModelService:
    sentiment = bentoml.models.BentoModel("sentiment:latest")
    translation = bentoml.models.BentoModel("translation:latest")
    # ... 10 more models ...
```

### 9.2 Antipattern 2: Not using the runner

```python
# ❌ Direct model use; no batching
@bentoml.api
def predict(self, input_data):
    return self.model.predict(input_data)

# ✅ Use the runner for batching, async, GPU management
@bentoml.api(batchable=True)
def predict(self, input_data):
    return self.runner.predict(input_data)
```

### 9.3 Antipattern 3: Loading models on every request

```python
# ❌ Slow; loads model each time
@bentoml.api
def predict(self, input_data):
    model = bentoml.sklearn.load_model("iris_classifier:latest")
    return model.predict(input_data)

# ✅ Load once at startup (or use the runner)
@bentoml.service
class IrisService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("iris_classifier:latest")
    
    @bentoml.api
    def predict(self, input_data):
        return self.model.predict(input_data)
```

### 9.4 Antipattern 4: Sequential ensemble

```python
# ❌ Sequential model calls; latency = sum of model latencies
result_a = self.model_a.predict(input_data)
result_b = self.model_b.predict(input_data)
result_c = self.model_c.predict(input_data)

# ✅ Parallel via async
import asyncio
result_a, result_b, result_c = await asyncio.gather(
    self.model_a.async_predict(input_data),
    self.model_b.async_predict(input_data),
    self.model_c.async_predict(input_data),
)
```

### 9.5 Antipattern 5: No GPU memory management

```python
# ❌ Two services on one GPU both try to use 100%
# OOM kills; service restarts

# ✅ Fractional GPU allocation
@bentoml.service(resources={"gpu": "0.5"})
class Service1:
    pass

@bentoml.service(resources={"gpu": "0.5"})
class Service2:
    pass
```

---

## 🎯 Key Takeaways

- Multi-model services: load all models at startup; route by task.
- Ensembles for higher accuracy: weighted average of model predictions.
- GPU sharing: allocate fractional GPUs; NVIDIA MIG for hard isolation.
- Dynamic batching: adaptive batch size + latency target.
- Runner pattern handles GPU placement, async, batching automatically.
- Model router for multi-model platforms; A/B test without restart.
- Hot-swap models for zero-downtime rollouts.
- Avoid one-service-per-model, no runner, loading per request, sequential ensembles, no GPU memory management.

## References

- BentoML Service patterns — [docs.bentoml.com/en/latest/guides/services](https://docs.bentoml.com/en/latest/guides/services.html)
- BentoML Runners — [docs.bentoml.com/en/latest/concepts/runner](https://docs.bentoml.com/en/latest/concepts/runner.html)
- BentoML distributed — [docs.bentoml.com/en/latest/guides/distributed](https://docs.bentoml.com/en/latest/guides/distributed.html)
- BentoML examples — [github.com/bentoml/BentoML/tree/main/examples](https://github.com/bentoml/BentoML/tree/main/examples)
- [[09 - MLOps y Produccion/32 - KServe and Knative|KServe]] — K8s-native serving alternative
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — multi-cloud deploy
- [[09 - MLOps y Produccion/42 - BentoML Production Model Serving/04 - Deployment Targets - Kubernetes, Lambda, SageMaker, Cloud Run|Note 04 — Deployment Targets]]
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM]] — LLM serving comparison