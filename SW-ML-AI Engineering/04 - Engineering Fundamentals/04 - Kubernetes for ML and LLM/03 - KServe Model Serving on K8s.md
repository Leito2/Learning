# KServe: Model Serving on K8s

## 1. What is KServe?

KServe (formerly KFServing) is a K8s CRD-based model serving platform. It provides:

- **InferenceService** CRD: model deployment with built-in scaling, canary, and monitoring
- **Built-in predictors**: TensorFlow, PyTorch, HuggingFace, scikit-learn, xgboost, custom
- **Knative integration**: scale-to-zero, serverless inference
- **Istio integration**: traffic splitting, canary deployments
- **Autoscaling**: CPU, memory, GPU utilization, concurrency

## 2. InferenceService YAML

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: bert-classifier
spec:
  predictor:
    pytorch:
      storageUri: s3://models/bert-classifier/
      resources:
        limits:
          nvidia.com/gpu: 1
      readinessProbe:
        httpGet:
          path: /v1/models/bert-classifier
          port: 8080
```

```bash
kubectl apply -f inference-service.yaml
kubectl get inferenceservice bert-classifier
kubectl get inferenceservice bert-classifier -o yaml  # status.url
```

## 3. Custom Predictor (Any Framework)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: custom-model
spec:
  predictor:
    containers:
    - image: my-registry/custom-server:latest
      ports:
      - containerPort: 8080
      env:
      - name: MODEL_PATH
        value: "/mnt/models"
      resources:
        limits:
          nvidia.com/gpu: 1
```

The container must implement KServe's V2 inference protocol (`/v2/models/{name}/infer`).

## 4. V2 Inference Protocol

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    inputs: list[dict]

class InferenceResponse(BaseModel):
    outputs: list[dict]

@app.post("/v2/models/custom/infer")
async def infer(request: InferenceRequest):
    results = model.predict(request.inputs)
    return InferenceResponse(outputs=results)

@app.get("/v2/models/custom")
async def model_info():
    return {"name": "custom", "ready": True}
```

## 5. Canary Deployment

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-model
spec:
  predictor:
    canaryTrafficPercent: 10
    pytorch:
      storageUri: s3://models/v2/
```

70% traffic to stable, 10% to canary:

```bash
kubectl apply -f canary.yaml
kubectl get inferenceservice my-model -o yaml | grep traffic
```

Promote to 100%:

```yaml
spec:
  predictor:
    pytorch:
      storageUri: s3://models/v2/
```

## 6. Autoscaling Configuration

```yaml
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 1  # target concurrent requests per replica
    scaleMetric: concurrency
    containers:
    - resources:
        limits:
          nvidia.com/gpu: 1
```

Scale metrics: `concurrency` (default), `cpu`, `memory`, `rps` (requests per second), `gpu-utilization` (DCGM).

## 7. Multi-Model Serving

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ensemble
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: s3://models/ensemble/
---
# Or use ModelMesh for many models on one GPU
```

ModelMesh allows hundreds of models on a single GPU by loading/unloading models on demand.

## 8. Storage URI Patterns

```
# S3
storageUri: s3://bucket/models/model-name/

# GCS
storageUri: gs://bucket/models/model-name/

# Azure
storageUri: azure://container/models/model-name/

# PVC (local)
storageUri: pvc://models-pvc/model-name/

# HTTP
storageUri: https://storage.example.com/models/model-name/
```

## 9. Observability

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/path: /metrics
    prometheus.io/port: "8080"
spec:
  predictor:
    containers:
    - env:
      - name: KSERVE_OPENMETRICS
        value: "true"
```

## Key Takeaways

- `InferenceService`: single CRD for deployment, scaling, canary, monitoring
- V2 inference protocol: `/v2/models/{name}/infer` (standard, framework-agnostic)
- Canary: `canaryTrafficPercent` + route traffic percentage
- Autoscaling: `scaleMetric: concurrency` for inference-specific scaling
- Multi-model: ModelMesh for many models on one GPU
- Storage: S3/GCS/Azure/PVC/HTTP URIs
- Custom predictor: any container implementing V2 protocol
