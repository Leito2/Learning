# Capstone: Multi-Model Inference Platform

## Objective

Deploy a production-grade inference platform on K8s with: two models (BERT classifier + GPT embedder), GPU scheduling, autoscaling, canary deployments, and monitoring.

## 1. Architecture

```
[Ingress] → [KServe InferenceService]
                ├── bert-classifier (GPU, 1 replica)
                │     └── canary: bert-v2 (10% traffic)
                ├── gpt-embedder (GPU, 3-10 autoscale)
                └── [ModelMesh] → model pool (shared GPU)
```

## 2. Namespace Setup

```bash
kubectl create namespace inference-platform
kubectl label namespace inference-platform istio-injection=enabled
```

## 3. BERT Classifier

```python
# predictor.py
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()
classifier = pipeline("text-classification", model="bert-base-uncased")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    score: float

@app.get("/v2/models/bert")
async def info():
    return {"name": "bert", "ready": True}

@app.post("/v2/models/bert/infer")
async def infer(req: Request):
    result = classifier(req.text)[0]
    return Response(label=result["label"], score=result["score"])
```

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: bert-classifier
  namespace: inference-platform
spec:
  predictor:
    containers:
    - image: registry/bert-predictor:latest
      ports:
      - containerPort: 8080
      resources:
        limits:
          nvidia.com/gpu: 1
      readinessProbe:
        httpGet:
          path: /v2/models/bert
          port: 8080
    minReplicas: 1
    maxReplicas: 3
    scaleMetric: concurrency
    scaleTarget: 5
```

## 4. BERT Canary (10% traffic)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: bert-classifier
spec:
  predictor:
    canaryTrafficPercent: 10
    containers:
    - image: registry/bert-v2-predictor:latest
      resources:
        limits:
          nvidia.com/gpu: 1
```

```bash
kubectl apply -f bert-canary.yaml
# Monitor error rate; promote after 15 min
kubectl apply -f bert-promote.yaml  # remove canaryTrafficPercent
```

## 5. GPT Embedder with KEDA Autoscaling

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gpt-embedder
spec:
  predictor:
    containers:
    - image: registry/gpt-embedder:latest
      resources:
        limits:
          nvidia.com/gpu: 1
    minReplicas: 2
    maxReplicas: 10
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gpt-embedder-scaler
  namespace: inference-platform
spec:
  scaleTargetRef:
    apiVersion: serving.kserve.io/v1beta1
    kind: InferenceService
    name: gpt-embedder
  minReplicaCount: 2
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: gpt_queue_depth
      query: |
        sum(rate(http_requests_total{service="gpt-embedder"}[2m]))
      threshold: "100"
```

## 6. GPU Node Pool

```bash
# Create GPU node pool with taint
gcloud container node-pools create inference-gpu \
  --machine-type a2-highgpu-1g \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --node-taints nvidia.com/gpu=true:NoSchedule \
  --num-nodes 2 \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autoscaling

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator --set driver.enabled=true
```

## 7. Inference Deploy

```yaml
apiVersion: v1
kind: Pod
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 1
```

## 8. Monitoring Stack

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: inference-monitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      serving.kserve.io/component: predictor
  endpoints:
  - port: http
    path: /metrics
```

Prometheus alerts:

```yaml
groups:
- name: inference
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m])) > 2
    for: 5m
  - alert: GPUMemoryPressure
    expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL > 0.9
    for: 2m
  - alert: GPUNotSchedulable
    expr: up{job="gpu-operator"} == 0
    for: 1m
```

## 9. Deploy Commands

```bash
# Build and push
docker build -t registry/bert-predictor:latest .
docker push registry/bert-predictor:latest

# Deploy
kubectl apply -k kustomize/overlays/production

# Test
INGRESS_HOST=$(kubectl get ingress -n istio-system -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')
curl -H "Host: inference-platform.com" http://$INGRESS_HOST/v2/models/bert/infer \
  -d '{"text": "This movie was fantastic!"}'

# Monitor
kubectl describe inferenceservice bert-classifier
kubectl get pods -n inference-platform -w
kubectl logs -n inference-platform -l serving.kserve.io/inferenceservice=bert-classifier
```

## Summary

| Component | Config | Key Detail |
|-----------|--------|------------|
| BERT classifier | KServe + custom container | GPU, canary 10% |
| GPT embedder | KServe + KEDA | autoscale on queue depth |
| GPU node pool | GKE + GPU Operator | A100, tainted, autoscaled |
| Monitoring | Prometheus + ServiceMonitor | latency + GPU alerts |
| Canary | KServe built-in | traffic split via CRD |

## Key Takeaways

- KServe `InferenceService` unifies deployment, scaling, canary, monitoring
- KEDA for event-driven scaling (queue depth → inference replicas)
- GPU node pool with taints ensures GPU pods land on GPU nodes
- Canary: `canaryTrafficPercent` for safe model rollouts
- Monitoring: Prometheus + DCGM for GPU metrics, latency SLOs
- Full deploy: `kubectl apply` + `curl` test + `kubectl logs` debug
