# 🏗️ ML Platform Engineering Overview

## Introduction

ML Platform Engineering is the discipline of building internal, self-serve infrastructure that enables data scientists and ML engineers to ship models faster — without each team reinventing feature pipelines, experiment tracking, model serving, and monitoring. An ML platform engineer builds the paved road; data scientists drive on it.

---

## What an ML Platform Provides

| Capability | Tools | Problem It Solves |
|---|---|---|
| **Feature Store** | Feast, Tecton, Databricks FS | "Where do I get features for my model?" |
| **Experiment Tracking** | MLflow, W&B, Vertex AI Experiments | "Which hyperparameters produced this accuracy?" |
| **Model Registry** | MLflow, Vertex AI Registry | "Which model version is in production?" |
| **Pipeline Orchestration** | Airflow, Kubeflow, Dagster | "Run training weekly with retry and alerting" |
| **Model Serving** | Triton, KServe, Ray Serve, Seldon | "Serve this model with <50ms latency and auto-scaling" |
| **Monitoring** | Evidently AI, NannyML, WhyLogs | "Did data drift degrade my model?" |
| **Compute** | K8s, Ray, Spark | "Give me GPUs for training without ticket requests" |

---

## The Platform Maturity Model

```
Level 1 — Scripts: Each DS runs training.py on their laptop
  ↓
Level 2 — Managed Services: GCP/AWS managed services (SageMaker, Vertex AI)
  ↓
Level 3 — Unified Platform: Internal platform integrating 3-5 tools (MLflow + K8s + Airflow)
  ↓
Level 4 — Self-Serve: DS clicks "Train" in UI → platform handles everything
  ↓
Level 5 — Fully Autonomous: Platform detects drift → auto-retrains → auto-deploys
```

---

## Key Design Principles

- **Self-serve:** Data scientists should never file a Jira ticket to get a GPU.
- **Paved road, not guardrails:** Make the right thing easy, not the wrong thing impossible.
- **Composable, not monolithic:** Choose best-in-class tools (MLflow for tracking, Triton for serving) and integrate, don't build everything from scratch.
- **Observable by default:** Every model deployed through the platform gets monitoring dashboards automatically.

---

## References

- [MLOps Maturity Model (Google Cloud)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Uber Michelangelo (ML Platform)](https://eng.uber.com/michelangelo-machine-learning-platform/)
