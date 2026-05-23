# ⚙️ Kubeflow: ML Workflows on Kubernetes

## Introduction

Kubeflow is the open-source ML platform that runs end-to-end ML workflows on Kubernetes. It provides a unified interface for experiment tracking (Katib), pipeline orchestration (Kubeflow Pipelines), notebook serving, model training (Training Operator), and model serving (KServe). For organizations already invested in Kubernetes, Kubeflow is the natural ML platform — it extends K8s primitives (pods, deployments, CRDs) into ML-specific concepts (experiments, pipelines, inference services).

---

## Architecture

```
┌────────────────────────── KUBEFLOW ──────────────────────────┐
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Notebooks   │  │   Pipelines  │  │    Katib     │       │
│  │  (Jupyter)   │  │   (Argo WF)  │  │  (HPO/AutoML)│       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Training   │  │    KServe    │  │   Metadata   │       │
│  │  Operator    │  │  (Serving)   │  │   (MLMD)     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                   Kubernetes (K8s)                   │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Alternative |
|---|---|---|
| **Kubeflow Pipelines** | DAG-based ML workflow orchestration on Argo | Airflow, Prefect, Dagster |
| **Katib** | Hyperparameter optimization and neural architecture search | Ray Tune, Optuna |
| **Training Operator** | Run distributed PyTorch/TF/XGBoost jobs on K8s | Ray Train |
| **KServe** | Serverless model serving with canary, explainability | Ray Serve, Triton, BentoML |
| **Notebooks** | Jupyter notebooks in K8s pods with GPU access | Vertex AI Workbench |
| **ML Metadata** | Track artifacts, lineage, and experiment metadata | MLflow |

---

## When to Use Kubeflow

- Your organization already runs Kubernetes
- You need multi-tenant ML platform (multiple DS teams)
- You want a unified platform (no stitching Airflow + MLflow + Seldon together manually)
- You need Kubernetes-native everything (RBAC, resource quotas, pod security)

---

## ⚠️ Challenges

- **K8s complexity:** Kubeflow inherits all K8s complexity. If your team doesn't know K8s, Kubeflow will be painful.
- **Heavy footprint:** Full Kubeflow deployment requires significant cluster resources before running any ML workload.
- **Upgrade friction:** Kubeflow version upgrades are notoriously complex due to tight coupling between components.

---

## References

- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
