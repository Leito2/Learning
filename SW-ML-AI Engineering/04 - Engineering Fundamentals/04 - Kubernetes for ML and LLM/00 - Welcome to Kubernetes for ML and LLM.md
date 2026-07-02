# Welcome to Kubernetes for ML and LLM

## Why K8s for ML

Kubernetes is the standard orchestration layer for production ML. Every major ML platform (KServe, Seldon, Ray, vLLM) deploys on K8s. GPU scheduling, autoscaling, canary deployments, and multi-model serving are K8s-native patterns.

## Current State in the Vault

K8s for ML is scattered across 5+ courses: vLLM Production Serving (06/20), CI-CD for ML (09/29), KServe (09/32), Distributed ML Infrastructure (10/29), ML Platform Engineering (09/26). This course unifies everything into one focused module.

## Course Map

| # | Module | Lines |
|---|--------|:-----:|
| 00 | Welcome — Why K8s for ML | 60 |
| 01 | K8s Fundamentals for ML Engineers | 150 |
| 02 | GPU Scheduling: NVIDIA Operator, MIG, GPU Sharing | 150 |
| 03 | KServe: Model Serving on K8s | 180 |
| 04 | Autoscaling: HPA, VPA, KEDA for Inference | 150 |
| 05 | Cluster Ops: Node pools, taints, tolerations for ML | 120 |
| 06 | Capstone: Multi-Model Inference Platform | 200 |
| **Total** | | **~1,010** |

## Pre-requisites

- Docker — [[../../02 - Docker Profesional/00 - Bienvenida|02]]
- Python Production Flow for MLOps — [[../03 - Python Production Flow for MLOps/00 - Welcome|04/03]]
- vLLM Production Serving — [[../../06 - Large Language Models/20 - vLLM Production Serving/00 - Bienvenida|06/20]]

## Cross-links in the Vault

- KServe + Knative: [[../../09 - MLOps y ML Platform/32 - KServe and Knative/00 - Welcome|09/32]]
- CI/CD for ML (ArgoCD): [[../../09 - MLOps y ML Platform/29 - CI-CD for ML/00 - Welcome|09/29]]
- Ray on K8s: [[../../10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome|10/29]]
- ML Platform Engineering: [[../../09 - MLOps y ML Platform/26 - ML Platform Engineering/00 - Welcome|09/26]]
