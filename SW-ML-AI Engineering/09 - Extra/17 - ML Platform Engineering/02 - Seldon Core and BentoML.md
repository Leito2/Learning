# 🚀 Seldon Core and BentoML: Specialized Model Serving

## Introduction

While Ray Serve and Triton focus on latency/throughput, Seldon Core and BentoML focus on the serving lifecycle: packaging, monitoring, explainability, A/B testing, and multi-model deployments. They sit on top of Kubernetes and provide ML-specific abstractions that generic API frameworks (FastAPI, Flask) lack.

---

## Seldon Core

### Key Features

| Feature | What It Does |
|---|---|
| **Multi-model graphs** | Compose models into inference DAGs (preprocess → model → postprocess → combiner) |
| **Explainability** | Built-in Alibi Explain integration (SHAP, Anchor, Integrated Gradients) |
| **Outlier detection** | Detect anomalous inputs before they reach the model |
| **A/B testing + Multi-armed bandits** | Route traffic between model versions, auto-select best |
| **Payload logging** | Log requests/responses to Kafka/Elasticsearch for monitoring |

### Model Composition Graph

```
          Request
             │
             ▼
    ┌───────────────┐
    │  Preprocessor  │
    └───────┬───────┘
            │
    ┌───────▼──────────────────┐
    │  Router (A/B Test)       │
    │  90% → Model v1          │
    │  10% → Model v2          │
    └───┬──────────┬──────────┘
        │          │
   ┌────▼───┐ ┌───▼────┐
   │ Model  │ │ Model  │
   │   v1   │ │   v2   │
   └────┬───┘ └───┬────┘
        │         │
   ┌────▼─────────▼───┐
   │    Combiner       │
   └────────┬─────────┘
            ▼
         Response
```

---

## BentoML

BentoML packages models into standardized "Bentos" — portable serving units that include the model, its dependencies, and its serving configuration.

### BentoML vs Seldon

| Aspect | BentoML | Seldon Core |
|---|---|---|
| **Focus** | Model packaging + serving | Advanced serving patterns (A/B, explainability) |
| **Deployment** | BentoCloud, Docker, K8s, cloud functions | K8s-native |
| **Complexity** | Lower | Higher (requires K8s) |
| **Model composition** | Via Service API | Via inference graph CRD |
| **Portability** | Any Docker/K8s/cloud | K8s only |

### BentoML Example

```python
import bentoml

@bentoml.service
class FraudDetector:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("fraud_model:latest")

    @bentoml.api
    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
```

Then deploy: `bentoml deploy .` → runs on BentoCloud, K8s, or as a Docker container.

---

## References

- [Seldon Core Documentation](https://docs.seldon.io/)
- [BentoML Documentation](https://docs.bentoml.org/)
