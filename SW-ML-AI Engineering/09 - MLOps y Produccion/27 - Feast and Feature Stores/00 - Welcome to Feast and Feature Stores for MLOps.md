# 🏷️ Welcome to Feast and Feature Stores for MLOps

## 🎯 Learning Objectives
- Articulate the training-serving skew problem and how feature stores resolve it
- Decompose Feast architecture into registry, offline store, online store, and SDK
- Implement online feature serving with Redis at sub-10ms latency
- Build point-in-time correct training datasets via historical retrieval
- Deploy Feast on AWS (S3 + DynamoDB/Redis + EKS) and GCP (BigQuery + Memorystore + GKE)
- Apply feature registry governance: ownership, lineage, TTL, validation, monitoring
- Integrate Feast into a complete MLOps stack with MLflow, FastAPI, and vLLM

## Introduction

Feature stores are the **missing bridge** between data engineering and model serving. Data engineers build ETL pipelines that compute features offline — yet data scientists train models with point-in-time correct snapshots, and ML engineers serve those same features online at sub-10ms latency. Without a feature store, these three worlds operate in silos, causing the notorious **training-serving skew**: features used at inference differ from those used at training. This silent killer of ML models costs companies millions in degraded predictions and undetected data leakage.

You already operate Redis at the heart of your LLM Edge Gateway for semantic caching. Feature stores represent the **logical extension** of that infrastructure: Redis becomes the online serving layer, while Feast provides the abstraction — defining features once, retrieving them consistently whether training or serving. If you have completed [[../../05 - MLOps y Produccion/19 - Feature Engineering y Feature Stores/...]], you understand feature engineering principles; Feast operationalizes them at scale. Combined with [[../../05 - MLOps y Produccion/18 - Experiment Tracking y Model Registry/...]] (MLflow model registry), [[../17 - ML Platform Engineering/...]] (platform engineering patterns), and [[../../06 - Cloud, Infra y Backend/...]] (infrastructure), you assemble a complete MLOps platform — the end-to-end thinking that distinguishes senior ML Engineers in technical interviews.

Feature stores are **interview gold**. Airbnb, Uber, Spotify, Stripe, and DoorDash all built internal feature stores, and candidates who discuss point-in-time correctness, online/offline consistency, and feature registry governance stand out dramatically. This course bridges your Redis expertise into the MLOps feature store domain, completing the stack you partially cover in your portfolio projects. By the end, you will design, deploy, and govern a production feature store integrated with your existing LLM systems.

---

## 🗺️ Course Map

| # | Note | Focus Area | Duration |
|---|---|---|---|
| 00 | Welcome | Course overview, prerequisites, roadmap | 15 min |
| 01 | Feature Store Theory and Architecture | Training-serving skew, Feast vs Tecton vs Hopsworks, component deep dive | 60 min |
| 02 | Feast Feature Serving Online and Offline | Feature views, Redis online retrieval, point-in-time historical joins | 60 min |
| 03 | Feast in Production AWS and GCP | Cloud deployments, materialization, pipeline integration with Airflow/FastAPI | 60 min |
| 04 | Advanced Feature Engineering and Registry Governance | Ownership, lineage, streaming transforms, validation with Great Expectations, full MLOps stack | 60 min |

## 📋 Prerequisites

| Skill | Required Level | Where to Refresh |
|---|---|---|
| Python | Intermediate (classes, decorators, type hints) | [[../../01 - Python/]] |
| Redis | Basic (SET/GET, sorted sets, streams) | Your LLM Edge Gateway project |
| ML Model Serving | Concepts (online/batch inference, latency budgets) | [[../../05 - MLOps y Produccion/17 - Model Serving/]] |
| Docker & Kubernetes | Basic (Dockerfile, pods, services) | [[../../06 - Cloud, Infra y Backend/]] |
| Feature Engineering | Concepts (aggregations, embeddings, time windows) | [[../../05 - MLOps y Produccion/19 - Feature Engineering y Feature Stores/]] |

## 🔧 Tools You Will Use

| Tool | Role | License |
|---|---|---|
| **Feast** | Feature store core (registry, SDK, serving) | Apache 2.0 |
| **Redis** | Online store (low-latency feature serving) | BSD / Redis Source Available License |
| **BigQuery / S3** | Offline store (batch feature storage) | Cloud-managed |
| **Great Expectations** | Feature validation and monitoring | Apache 2.0 |
| **FastAPI** | Serving middleware (Feast dependency injection) | MIT |
| **Dagster / Airflow** | Orchestration (materialization jobs) | Apache 2.0 |

---

## 📦 Compression Code

```python
"""feature_store_bootstrap.py — Single-script Feast + Redis bootstrap for MLOps.

WHY: Demonstrates end-to-end Feast workflow in one file: entity definition,
     feature view creation, online materialization to Redis, and retrieval
     of both online and historical features with point-in-time correctness.
"""
from datetime import datetime, timedelta
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, UnixTimestamp

store = FeatureStore(repo_path=".")

entity = Entity(name="user_id", join_keys=["user_id"], description="User identifier")

source = FileSource(
    path="gs://my-bucket/user_features.parquet",
    timestamp_field="event_timestamp",
)

user_features = FeatureView(
    name="user_features",
    entities=[entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="total_orders_7d", dtype=Int64),
        Field(name="avg_order_value_30d", dtype=Float32),
        Field(name="last_login_hours_ago", dtype=Int64),
    ],
    source=source,
    tags={"team": "growth", "tier": "production"},
)

store.apply([entity, user_features])

store.materialize(start_date=datetime.now() - timedelta(days=1), end_date=datetime.now())

features = store.get_online_features(
    features=["user_features:total_orders_7d", "user_features:avg_order_value_30d"],
    entity_rows=[{"user_id": "user_42"}],
).to_dict()

dataset = store.get_historical_features(
    entity_df='SELECT user_id, event_timestamp FROM events WHERE date >= "2024-01-01"',
    features=["user_features:total_orders_7d", "user_features:avg_order_value_30d"],
).to_df()
```

## 🎯 Documented Project

**Project Description:** Build a full-featured feature store for a recommendation system serving 10k RPS, using Feast with Redis for online serving and BigQuery for offline storage, integrated into an existing FastAPI inference pipeline.

**5 Functional Requirements:**
1. Define 15+ feature views across user, item, and context entity types
2. Serve online features via Redis with p99 latency < 5ms
3. Generate point-in-time correct training datasets spanning 90 days
4. Materialize features from BigQuery to Redis every 15 minutes
5. Validate feature freshness and distribution drift before each materialization

**Main Components:**
- Feast Core (registry, gRPC API, SDK)
- Redis Cluster (online store, connection pooling, pipelining)
- BigQuery (offline store, historical feature warehouse)
- Great Expectations suite (feature validation at materialization boundary)
- FastAPI middleware (Feast dependency injection for inference requests)

**3 Success Metrics:**
- Feature freshness lag < 15 minutes for 99% of feature views
- Online serving p99 latency < 5ms under 10k RPS load
- Zero training-serving skew incidents over 30-day measurement period

## 🎯 Key Takeaways
- Feature stores eliminate training-serving skew by providing a **single source of truth** for feature definitions — define once, retrieve consistently across training and inference
- Feast provides the abstraction layer; Redis (or DynamoDB/Bigtable) provides the low-latency online muscle — your existing Redis expertise transfers directly
- Point-in-time correctness is the killer feature: naive SQL JOINs leak future information into training data, producing models that perform deceptively well in testing but fail in production
- Feature registry governance (ownership, lineage, TTL, validation) is not optional — it is what distinguishes a toy feature store from a production MLOps platform
- Integrating Feast into your existing stack (LLM Gateway, MLflow, FastAPI) completes the MLOps journey from experiment to production at scale

## References
- Feast Documentation: https://docs.feast.dev
- Feast GitHub: https://github.com/feast-dev/feast
- Tecton Feature Store: https://www.tecton.ai
- Hopsworks Feature Store: https://www.hopsworks.ai
- "Feast: Feature Store for Machine Learning" (KDD 2021): https://dl.acm.org/doi/10.1145/3447548.3470803
- Uber Michelangelo: https://eng.uber.com/michelangelo-machine-learning-platform/
- Airbnb Zipline: https://medium.com/airbnb-engineering/zipline-airbnbs-machine-learning-data-management-platform
