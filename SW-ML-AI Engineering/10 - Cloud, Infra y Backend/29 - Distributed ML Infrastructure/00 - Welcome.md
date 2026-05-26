# 🌐 Welcome to Distributed ML Infrastructure

Production ML systems are not models running on a laptop — they are distributed architectures spanning streaming ingestion, resource scheduling, inference serving, and pipeline orchestration. This course covers the six infrastructure technologies that appear in every large-scale ML deployment: Apache Kafka for event streaming, Ray for distributed compute, NVIDIA Triton for GPU inference, Airflow/Prefect for workflow orchestration, dbt for data transformation, and Dagster for asset-aware ML pipelines.

---

## Course Index

1. [[01 - Apache Kafka|Apache Kafka for ML Event Streaming]]
2. [[02 - Ray and Ray Serve|Ray: Distributed Compute and Model Serving]]
3. [[03 - NVIDIA Triton|NVIDIA Triton Inference Server]]
4. [[04 - Airflow and Prefect|Workflow Orchestration: Airflow and Prefect]]
5. [[05 - dbt|dbt for ML Data Transformation]]
6. [[06 - Dagster|Dagster: Asset-Aware ML Orchestration]]

---

## Why These Six Technologies?

| Technology | Problem It Solves | Without It |
|---|---|---|
| **Kafka** | Real-time event ingestion | Batch-only pipelines, stale features |
| **Ray** | Distributed Python at scale | Single-machine Python limits |
| **Triton** | GPU-optimized inference serving | High latency, low throughput |
| **Airflow/Prefect** | Scheduled, monitored pipelines | Manual script execution |
| **dbt** | Reliable data transformations | Ad-hoc SQL with no testing |
| **Dagster** | ML asset orchestration | Pipeline failures with no dependency tracking |
