# 🏷️ Welcome — BentoML Production Model Serving

![Banner del Curso BentoML Production Model Serving](<bentoml-course-banner.svg>)

## 🎯 Learning Objectives
- Use BentoML to package and serve any ML model (sklearn, PyTorch, XGBoost, custom)
- Build REST and gRPC inference services with type-safe I/O
- Deploy Bentos to Kubernetes, Lambda, SageMaker, Cloud Run
- Scale BentoML with adaptive batching, model ensembles, and async inference
- Implement custom runners for complex inference pipelines
- Integrate with vLLM, Ray Serve, and the broader ML serving ecosystem
- Apply to portfolio projects: standalone model serving, multi-model platforms

## Introduction

**BentoML** is the standard framework for production ML model serving. It wraps any model (sklearn, PyTorch, XGBoost, custom neural nets) in a portable artifact called a **Bento**, deployable to any platform.

| Model | Without BentoML | With BentoML |
|-------|----------------|--------------|
| scikit-learn | Custom Flask + pickle + Dockerfile | `bentoml.SKLearnModel` |
| PyTorch | Custom FastAPI + torch.load + handler | `bentoml.pytorch.save_model` |
| XGBoost | Custom FastAPI + xgb.Booster + retraining logic | `bentoml.xgboost.save_model` |
| Custom NN | Weeks of FastAPI + Docker + gRPC + autoscaling | `bentoml.Service` |

The killer features:
- **Multi-framework** — sklearn, PyTorch, TensorFlow, XGBoost, LightGBM, HuggingFace, custom
- **Multi-target** — Docker, Kubernetes, SageMaker, Lambda, Cloud Run, Azure ML
- **Adaptive batching** — automatic batching for low-latency inference
- **Service composition** — chain models, run ensembles
- **Production defaults** — health checks, metrics, environment management

This note teaches BentoML end-to-end, from saving models to deploying them.

---

## 1. The BentoML Workflow

```python
# 1. Train your model (any framework)
import bentoml
from sklearn import svm
import numpy as np

X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])

model = svm.SVC()
model.fit(X_train, y_train)

# 2. Save the model as a Bento
bentoml.sklearn.save_model(
    "iris_classifier",
    model,
    signatures={"predict": {"batchable": True, "batch_dim": 0}},
    metadata={"framework": "sklearn", "version": "1.0"},
)

# 3. Build a service
@bentoml.service(
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 30},
)
class IrisService:
    iris_model = bentoml.models.BentoModel("iris_classifier:latest")
    
    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.iris_model)
    
    @bentoml.api
    def predict(self, input_data: list[list[float]]) -> list[int]:
        return self.model.predict(input_data).tolist()

# 4. Build the Bento (container)
# bentoml build

# 5. Deploy
# bentoml serve
# bentoml deploy --platform kubernetes
```

What you get:
- A Bento (`.bento` file) — model + code + environment
- A Docker image — production-ready
- REST + gRPC endpoints — auto-generated
- Adaptive batching — automatic
- Health checks — built-in
- Metrics — Prometheus out of the box
- Multi-target deployment — Docker, K8s, Lambda, SageMaker

---

## 2. BentoML vs Other Serving Frameworks

| Framework | Multi-framework | Multi-target | Adaptive batching | Composition |
|-----------|------------------|---------------|-------------------|-------------|
| **BentoML** | ✅ | ✅ | ✅ | ✅ |
| **vLLM** | ❌ (LLM only) | Single | ✅ | ❌ |
| **TorchServe** | ✅ (PyTorch) | Single | ✅ | ❌ |
| **TF Serving** | ❌ (TF only) | Single | ✅ | ❌ |
| **Triton** | ✅ | Single | ✅ | ✅ |
| **Custom FastAPI** | ✅ | ❌ | ❌ | ❌ |

For **non-LLM models** (sklearn, XGBoost, PyTorch, custom neural nets), BentoML is the most productive framework. For **LLMs only**, vLLM is better.

---

## 3. Course Map

| Note | Title | Focus |
|------|-------|-------|
| 00 | Welcome — BentoML Production Model Serving | This overview |
| 01 | BentoML Fundamentals — Saving and Loading Models | The basic pipeline |
| 02 | Service Patterns — REST, gRPC, Async, Batching | Inference APIs |
| 03 | Distributed Serving — Multi-model, Ensembles, GPU Sharing | Scale-out |
| 04 | Deployment Targets — Kubernetes, Lambda, SageMaker, Cloud Run | Production deploy |
| 05 | Capstone — Production BentoML Platform | End-to-end |

---

## 4. Cross-Module Connections

| Vault Module | Connection |
|--------------|-----------|
| [[06 - Large Language Models/13 - vLLM and Advanced RAG\|vLLM]] | LLM-specific serving |
| [[09 - MLOps y Produccion/32 - KServe and Knative\|KServe]] | K8s-native serving |
| [[10 - Cloud, Infra y Backend/22 - Cloud Computing\|Cloud Computing]] | Multi-cloud deploy |
| [[10 - Cloud, Infra y Backend/31 - FastAPI for ML\|FastAPI for ML]] | Custom FastAPI alternative |
| [[09 - MLOps y Produccion/27 - Feast and Feature Stores\|Feast]] | Online feature store integration |
| [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems\|Incident Response]] | Model rollback |

---

## 5. What You Will Build

By Note 05, you will have:

- A BentoML service for an sklearn model (Iris classifier)
- A multi-model BentoML service (chained models)
- A Kubernetes deployment with auto-scaling
- A CI/CD pipeline for retraining → rebuild → deploy
- Adaptive batching for low-latency inference
- A complete production deployment

This is the **seventeenth portfolio project**: production ML model serving.

---

## 6. The Cutting Edge in 2026

Three frontiers:

1. **BentoML 2.0** — new async-native API; better LLM serving support
2. **BentoML Cloud** — managed deployment platform (alternative to AWS SageMaker)
3. **Ray Serve + BentoML** — for distributed multi-model serving

These map directly onto your portfolio: the **StayBot** could use BentoML for the recommendation model; the **LLM Edge Gateway** could use BentoML for non-LLM routing; the **Multi-Agent Research System** could use BentoML for the orchestrator service.

---

⚠️ The BentoML API changes between versions. The **patterns** in this course are stable; the **specific API calls** will need updating. Always cross-check against [docs.bentoml.com](https://docs.bentoml.com) before deploying.