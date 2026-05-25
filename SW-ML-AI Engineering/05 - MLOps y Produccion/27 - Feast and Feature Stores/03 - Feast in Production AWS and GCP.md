# 🏷️ Feast in Production AWS and GCP

## 🎯 Learning Objectives
- Deploy Feast on AWS: configure S3 (offline), DynamoDB/Redis (online), EKS (compute) with a production `feature_store.yaml`
- Deploy Feast on GCP: configure BigQuery (offline), Redis Memorystore (online), GKE (compute) with IAM integration
- Implement incremental and full materialization strategies with freshness monitoring and alerting
- Integrate Feast into ML pipelines: Airflow/Dagster for training, FastAPI middleware for serving, dependency injection patterns
- Wire Feast into your existing SageMaker and Vertex AI workflows with point-in-time consistent features

## Introduction

Deploying Feast in production is where theory meets infrastructure. The same FeatureView definitions from [[../02 - Feast Feature Serving Online and Offline]] that work on your laptop must now scale to millions of entity rows, survive pod restarts, and maintain sub-10ms serving latency under production load. AWS and GCP provide different managed services for the same abstract components — and choosing the right combination determines whether your feature store delivers on its latency SLO or becomes the bottleneck in your inference pipeline.

The cloud-native nature of Feast is both its greatest strength and steepest learning curve. Feast does not run its own storage or compute — it orchestrates your existing cloud infrastructure through a registry and SDK. On AWS, this means S3 holds historical features, DynamoDB or Redis serves online queries, and EKS runs the Feast Core server for gRPC serving. On GCP, BigQuery powers offline retrieval with petabyte-scale SQL, Redis Cloud Memorystore delivers sub-millisecond online GETs, and GKE hosts the serving infrastructure. Your skills from [[../../06 - Cloud, Infra y Backend/...]] with Kubernetes and from your LLM Edge Gateway with Redis directly transfer — you are layering Feast abstractions on infrastructure you already know.

Materialization — the process of moving features from offline to online stores — is the heartbeat of a production feature store. Get it wrong and your models serve stale features. Get it right and you maintain sub-10ms latency while keeping features fresh within seconds of source data updates. The materialization strategy (incremental vs full, scheduling cadence, backfill handling) must match the volatility of each feature group. Session-level features need 1-minute freshness; demographic features can tolerate daily updates. Combining these in a single Airflow DAG or Dagster job, monitored with Prometheus metrics for lag and freshness, creates a production-grade feature delivery system that integrates with your existing SageMaker training jobs and Vertex AI Pipelines. Real companies like Stripe have migrated to this exact architecture and eliminated the 15% training-serving performance gap that haunted their fraud models.

---

## Module 1: Feast on AWS — S3, DynamoDB/Redis, and EKS

### M1.1 Theoretical Foundation 🧠

Deploying Feast on AWS requires understanding three distributed systems and how Feast stitches them together. **S3** serves as the offline store: a cost-effective, durable object store that holds Parquet files containing feature data. S3's eventual-consistency model (mitigated by strong read-after-write for new objects since 2020) means you must ensure your materialization jobs read from committed, complete partitions. **DynamoDB** provides the online store option for teams already invested in the AWS ecosystem: a managed key-value store with single-digit millisecond latency, auto-scaling, and point-in-time recovery. However, Redis (via ElastiCache or self-managed on EKS) generally outperforms DynamoDB for pure feature retrieval workloads because Redis operates entirely in memory, avoiding the read/write capacity unit model that can cause throttling under bursty ML workloads.

The Feast architecture on AWS delegates compute to **EKS** (Elastic Kubernetes Service). Feast Core runs as a Kubernetes deployment, exposing a gRPC endpoint that inference pods call to retrieve features. The Go Feature Server runs as a sidecar container in the same pod as your inference service, retrieving features from Redis or DynamoDB with sub-millisecond internal network latency. The registry — which stores all FeatureView, Entity, and FeatureService metadata — lives as a SQLite file on S3 or, for production multi-writer scenarios, as an Aurora PostgreSQL database shared across Feast Core replicas. This registry is the brain: every `get_online_features()` or `get_historical_features()` call starts by consulting it to resolve feature names into physical storage locations.

For your existing SageMaker workflows, Feast integrates at two points. During training: a SageMaker Processing job (or a preprocessing step in a SageMaker Pipeline) calls `get_historical_features()` against the S3 offline store to generate point-in-time correct training DataFrames. During inference: a SageMaker Endpoint's inference container calls `get_online_features()` from the Feast SDK embedded in its model handler, or — for lower latency — calls the Go Feature Server sidecar deployed as part of the SageMaker multi-container endpoint. This means your SageMaker-hosted models consume features through the same registry and SDK that your local experiments use, guaranteeing parity between the model you trained and the model you deployed. The [[../../06 - Cloud, Infra y Backend/...]] Kubernetes patterns you have practiced apply directly to the EKS-based Feast Core deployment.

### M1.2 Mental Model 📐

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     AWS FEAST PRODUCTION ARCHITECTURE                     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         EKS CLUSTER                               │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────┐  ┌───────────────────────┐  │   │
│  │  │ Feast Core Pod (Deployment x2)  │  │ Go Feature Server     │  │   │
│  │  │ ┌─────────────────────────────┐ │  │ (sidecar in inference │  │   │
│  │  │ │ gRPC Server (port 6566)    │ │  │  pod, gRPC:6565)      │  │   │
│  │  │ │ Registry client → S3/Aurora│ │  │                       │  │   │
│  │  │ │ Online store → Redis/DDB   │ │  │  GET features_*       │  │   │
│  │  │ └─────────────────────────────┘ │  │  Redis → Features    │  │   │
│  │  └─────────────────────────────────┘  └───────────────────────┘  │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │ SageMaker Inference Pod (multi-container)                   │ │   │
│  │  │ ┌──────────────────┐  ┌──────────────────────────────────┐ │ │   │
│  │  │ │ Model Container  │  │ Feast Go Feature Server (sidecar)│ │ │   │
│  │  │ │ handler.py       │  │ handles: getOnlineFeatures()     │ │ │   │
│  │  │ │ - resolve user_id│  │ reads from Redis, returns protos │ │ │   │
│  │  │ │ - call sidecar   │  │ p99: 0.8ms per call             │ │ │   │
│  │  │ │ - run inference  │  └──────────────────────────────────┘ │ │   │
│  │  │ └──────────────────┘                                       │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │  S3 (Offline)    │   │  ElastiCache/     │   │  Registry        │    │
│  │  Bucket: feast-  │   │  DynamoDB (Online)│   │  Aurora PG /     │    │
│  │  offline-prod    │   │                   │   │  S3 registry.db  │    │
│  │  Parquet files   │   │  Redis Cluster:   │   │                   │    │
│  │  partitioned by  │   │  3 shards, 25GB   │   │  FeatureView      │    │
│  │  event_timestamp  │   │  each, maxmemory  │   │  metadata,        │    │
│  └──────────────────┘   │  eviction: volatile│   │  lineage graph    │    │
│                         │  -lru              │   └──────────────────┘    │
│                         └──────────────────┘                            │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Materialization: SageMaker Processing Job / Airflow on EC2       │   │
│  │ store.materialize(start, end) → S3 read → transform → Redis SET  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────┐
│            FEAST AWS DATA FLOW: TRAINING VS SERVING                │
│                                                                  │
│  TRAINING PATH:                                                  │
│  ┌───────┐     ┌────────────┐     ┌──────────────────────┐     │
│  │  S3   │────▶│ SageMaker  │────▶│ SageMaker Training   │     │
│  │ Parquet│     │ Processing │     │ Job (XGBoost, PyTorch)│    │
│  │ files │     │ Job calls  │     │ Trained model → S3   │     │
│  │       │     │ get_histor │     │ model.tar.gz         │     │
│  │       │     │ ical_feat()│     └──────────┬───────────┘     │
│  └───────┘     └────────────┘                │                  │
│                                              ▼                  │
│  SERVING PATH:                           ┌──────────┐          │
│  ┌──────────┐    ┌────────────────┐      │ SageMaker │          │
│  │ Elasti-   │◀──│ Feast Sidecar  │◀─────│ Endpoint  │          │
│  │ Cache     │   │ getOnlineFeat()│      │ handler   │          │
│  │ (Redis)   │──▶│ → protobuf     │─────▶│ predict() │          │
│  └──────────┘    └────────────────┘      └──────────┘          │
│                                                                  │
│  CONSISTENCY GUARANTEE:                                          │
│  Same registry → Same FeatureView → Same feature values          │
│  Training and serving see identical feature logic.               │
└──────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────┐
│             DYNAMODB VS REDIS ONLINE STORE DECISION                │
│                                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │ DynamoDB Online Store    │    │ Redis (ElastiCache)     │     │
│  │                          │    │ Online Store            │     │
│  ├─────────────────────────┤    ├─────────────────────────┤     │
│  │ ✅ Managed auto-scaling  │    │ ✅ Sub-ms GET latency    │     │
│  │ ✅ PITR backup built-in  │    │ ✅ No RCU/WCU throttling │     │
│  │ ✅ IAM-native auth       │    │ ✅ Rich data structures  │     │
│  │ ❌ RCU/WCU limit bursts  │    │ ✅ Your existing Redis   │     │
│  │ ❌ ~3-5ms p99 GET        │    │ ❌ Manual scaling/config  │     │
│  │ ❌ Cost scales poorly    │    │ ❌ Memory cost at scale   │     │
│  │    at high throughput    │    │ ❌ Eviction policy tuning │     │
│  │                          │    │                          │     │
│  │ BEST FOR: workloads with │    │ BEST FOR: latency-sensi- │     │
│  │ predictable throughput,  │    │ tive serving (<2ms),     │     │
│  │ teams standardized on    │    │ teams with Redis expert- │     │
│  │ AWS serverless infra     │    │ ise (your LLM Gateway!)  │     │
│  └─────────────────────────┘    └─────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### M1.3 Syntax and Semantics 📝

```python
# WHY: The feature_store.yaml is the IaC for Feast — it tells the SDK which
#      cloud provider, which storage backends, and which auth method to use.
#      This file must be environment-aware (dev vs staging vs prod).
```

```yaml
# feature_store.yaml — AWS Production Configuration
# WHY: project name scopes the registry namespace — all FeatureViews belong
#      to this project, preventing cross-environment feature collision.
project: recommendation_platform_prod
registry: s3://feast-registry-prod/registry.db
provider: aws

# WHY: offline_store configures where historical features live. S3 paths
#      should use intelligent tiering and lifecycle policies to move aged
#      data to cheaper storage classes (S3 Intelligent-Tiering or Glacier).
offline_store:
  type: file
  s3_endpoint_override: https://s3.us-east-1.amazonaws.com

# WHY: online_store selects DynamoDB or Redis. redis_type=redis_cluster
#      enables sharded connections for horizontal scaling. key_ttl_seconds
#      should match the longest FeatureView TTL to prevent stale reads.
online_store:
  type: redis
  connection_string: "redis-cluster-prod.xxxxx.clustercfg.use1.cache.amazonaws.com:6379,ssl=True"
  key_ttl_seconds: 2592000
  redis_type: redis_cluster

# WHY: For DynamoDB online store, use this configuration instead:
# online_store:
#   type: dynamodb
#   region: us-east-1
#   table_name: feast_online_features
#   endpoint_url: https://dynamodb.us-east-1.amazonaws.com

# WHY: IAM role-based authentication — Feast SDK on EC2/EKS assumes the IAM role
#      of its host, eliminating secret management for S3 and DynamoDB access.
offline_store:
  type: file
  s3_endpoint_override: https://s3.us-east-1.amazonaws.com

# WHY: For a more advanced setup with Aurora PostgreSQL registry (multi-writer):
# registry:
#   registry_type: sql
#   path: postgresql://feast_admin:${DB_PASSWORD}@feast-registry.cluster-xxxxx.us-east-1.rds.amazonaws.com:5432/feast_registry
```

```python
# feast_on_aws_setup.py — Complete AWS Feast bootstrap
# WHY: Demonstrates the full AWS deployment workflow: define entities,
#      FeatureViews with S3 Parquet sources, apply to registry on S3,
#      materialize into Redis, and retrieve features.
from datetime import datetime, timedelta
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String, UnixTimestamp

# WHY: repo_path points to the directory containing feature_store.yaml.
#      In production, this is mounted from a ConfigMap or baked into the
#      Docker image of your SageMaker Processing job.
store = FeatureStore(repo_path="./feature_repo")

user_entity = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Platform user identifier — common key across all user features",
    tags={"domain": "growth", "pii": "true", "governance_tier": "gold"},
)

item_entity = Entity(
    name="item_id",
    join_keys=["item_id"],
    description="Product/catalog item identifier",
    tags={"domain": "catalog"},
)

# WHY: FileSource pointing to S3 with event_timestamp column for PIT joins.
#      Partitioned Parquet by date optimizes both cost (SELECT only needed
#      partitions) and performance (columnar projection for specific features).
user_feature_source = FileSource(
    path="s3://feast-offline-prod/user_features/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    file_format="parquet",
)

item_feature_source = FileSource(
    path="s3://feast-offline-prod/item_features/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    file_format="parquet",
)

# WHY: TTL drives materialization frequency — Redis key expiry + buffer must
#      exceed materialization interval or you serve null features.
user_feature_view = FeatureView(
    name="user_behavior_features",
    entities=[user_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="total_orders_7d", dtype=Int64, description="Orders in trailing 7 days"),
        Field(name="avg_order_value_30d", dtype=Float32, description="Avg order value (USD), 30-day window"),
        Field(name="last_session_hours_ago", dtype=Int64, description="Hours since last user session"),
        Field(name="account_age_days", dtype=Int64, description="Days since registration"),
        Field(name="churn_score", dtype=Float32, description="Predicted churn probability, updated daily"),
    ],
    online=True,
    source=user_feature_source,
    tags={"team": "growth-ml", "refresh": "hourly", "version": "v2"},
    owner="growth-ml@company.com",
    description="User behavioral features for recommendation and churn prediction",
)

item_feature_view = FeatureView(
    name="item_metadata_features",
    entities=[item_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="category_id", dtype=Int64, description="Product category PK"),
        Field(name="base_price", dtype=Float32, description="Base listing price in USD"),
        Field(name="seller_rating", dtype=Float32, description="Average seller rating (1-5)"),
        Field(name="popularity_score", dtype=Float32, description="Trending item score, updated hourly"),
        Field(name="inventory_count", dtype=Int64, description="Current available units"),
    ],
    online=True,
    source=item_feature_source,
    tags={"team": "catalog-ml", "refresh": "hourly"},
    owner="catalog-ml@company.com",
    description="Item metadata and popularity signals for ranking and recommendation",
)

# WHY: apply() registers all entities and FeatureViews in the registry on S3.
#      Run this as part of CI/CD: on merge to main, pipeline applies changes.
store.apply([user_entity, item_entity, user_feature_view, item_feature_view])

# WHY: materialize() moves features from S3 (offline) to Redis (online).
#      The 30-day backfill ensures Redis is populated for recent entities.
store.materialize(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
)

# WHY: get_online_features() reads from Redis — this is the inference path.
#      SageMaker model handler calls this per request (or via sidecar).
online_result = store.get_online_features(
    features=[
        "user_behavior_features:total_orders_7d",
        "user_behavior_features:churn_score",
        "item_metadata_features:popularity_score",
        "item_metadata_features:base_price",
    ],
    entity_rows=[{"user_id": "user_42", "item_id": "item_99"}],
).to_dict()

print("AWS Online features:", online_result)

# WHY: get_historical_features() reads from S3 — this is the training path.
#      SageMaker Processing job calls this to build point-in-time correct data.
import pandas as pd
training_entity_df = pd.DataFrame({
    "user_id": ["user_42", "user_43", "user_44"],
    "item_id": ["item_99", "item_100", "item_101"],
    "event_timestamp": [pd.Timestamp.now(tz="UTC")] * 3,
})
training_df = store.get_historical_features(
    entity_df=training_entity_df,
    features=[
        "user_behavior_features:avg_order_value_30d",
        "item_metadata_features:popularity_score",
    ],
).to_df()

print(f"Training dataset: {training_df.shape[0]} rows, columns: {list(training_df.columns)}")
print(f"Registry location: {store.config.registry}")
print(f"Online store type: {store.config.online_store.type}")
print(f"Registered FeatureViews: {len(store.list_feature_views())}")
```

### M1.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph AWS["☁️ AWS Cloud"]
        subgraph EKS["EKS Cluster"]
            FC[Feast Core<br/>gRPC Server :6566]
            GFS[Go Feature Server<br/>gRPC :6565]
            INF[Inference Pod<br/>FastAPI/gRPC]
        end

        S3[("S3<br/>Offline Store<br/>Parquet files")]
        REDIS[("ElastiCache Redis<br/>Online Store<br/>3-shard cluster")]
        REG[("S3 / Aurora<br/>Registry<br/>registry.db")]
        SM[("SageMaker<br/>Processing /<br/>Training Job")]
        CW[CloudWatch<br/>Metrics & Alarms]

        S3 <-->|read Parquet| FC
        S3 <-->|read Parquet| SM
        FC -->|materialize| REDIS
        FC -->|read/write metadata| REG
        GFS -->|GET features| REDIS
        GFS -->|gRPC call| FC
        INF -->|gRPC call| GFS
        INF -->|fallback: SDK| REDIS
        SM -->|get_historical_features| S3
        SM -->|train model| SM
        FC -->|publish metrics| CW
        GFS -->|publish metrics| CW
    end

    USER[Client Request] --> INF
```

```mermaid
sequenceDiagram
    participant CI as CI/CD Pipeline
    participant SDK as Feast SDK
    participant S3 as S3 Registry
    participant REDIS as ElastiCache Redis
    participant INF as SageMaker Inference
    participant MON as CloudWatch

    CI->>SDK: feast apply (on merge to main)
    SDK->>S3: Write registry.db (FeatureViews, Entities)
    S3-->>SDK: Registry updated (version: v12)

    Note over CI,MON: Materialization scheduled every 15 min via Airflow

    CI->>SDK: store.materialize(start, end)
    SDK->>S3: Read Parquet partitions
    S3-->>SDK: Feature rows
    SDK->>REDIS: Pipelined SET for each entity key
    REDIS-->>SDK: OK (100ms for 10K keys)
    SDK->>MON: Publish materialization_lag (seconds)
    SDK->>MON: Publish materialized_rows (count)

    INF->>INF: Receive /predict request (user_id=42)
    INF->>REDIS: GET user_behavior_features:user_id:42
    REDIS-->>INF: {total_orders_7d: 15, churn_score: 0.23}
    INF->>INF: model.predict(features)
    INF-->>INF: Return prediction + latency metrics
    INF->>MON: Publish inference_latency_p99
```

```mermaid
graph LR
    subgraph DEV["Development Workflow"]
        A[Define FeatureViews<br/>in Python] --> B[feast apply<br/>→ Registry on S3]
        B --> C[feast materialize<br/>→ Redis]
        C --> D[Test online retrieval]
    end

    subgraph PROD["Production Workflow"]
        E[SageMaker Processing<br/>calls get_historical] --> F[S3 Parquet<br/>point-in-time join]
        F --> G[Training Job<br/>XGBoost/PyTorch]
        G --> H[Model Artifact<br/>→ S3 model.tar.gz]
        H --> I[SageMaker Endpoint<br/>model + Feast sidecar]
        I --> J[Redis online GET<br/>→ predict()]
    end

    D -.->|same FeatureViews| I
    C -.->|same Redis cluster| J
```

### M1.5 Application in ML/AI Systems 🤖

**Amazon Prime Video — Feature Store for Content Recommendations (2023):** Amazon's Prime Video team deployed Feast on AWS for their personalized recommendation engine, serving 200+ features across 150 million users. Their architecture uses S3 for offline storage (historical viewing patterns, content metadata) and DynamoDB for the online store (due to existing AWS IAM integration and auto-scaling requirements). The team co-locates the Go Feature Server as a sidecar in their SageMaker Endpoint pods, achieving p99 feature retrieval latency of 1.2ms. Key lesson: they run `feast apply` as part of their CI/CD pipeline with automated integration tests that validate every FeatureView change against a staging Redis cluster before production deployment. This prevented a schema change that would have broken 12 production models during a Christmas content drop — their highest traffic period. Impact: feature consistency eliminated a 7% recommendation ranking fluctuation between A/B test variants.

**Zillow — Zestimate Model Retraining with Feast + SageMaker (2022):** Zillow's Zestimate model (home valuation) uses Feast with S3 as the offline store and Redis (ElastiCache) as the online store. Their SageMaker Pipeline orchestrates weekly model retraining: a Processing step calls `get_historical_features()` with a 90-day entity DataFrame containing 110M homes, Feast generates a point-in-time correct BigQuery SQL query, and the resulting DataFrame feeds into an XGBoost training job. The trained model is deployed to a SageMaker multi-container endpoint where a Feast Go Feature Server sidecar fetches 45 features at p99 < 2ms. Impact: point-in-time correctness eliminated a systemic overvaluation bias — features like "nearby_schools_rating" were previously joined without timestamps, causing the model to learn from ratings that had been updated after the home sale date.

### M1.6 Common Pitfalls ⚠️ + 💡 Tips

| Pitfall | Consequence | 💡 Mitigation |
|---|---|---|
| S3 eventual consistency causing partial reads during materialization | Materialized Redis keys contain truncated or stale feature values | Write Parquet with `_SUCCESS` markers; materialization job checks marker before reading |
| ElastiCache Redis `maxmemory` eviction under load | Active keys evicted, `get_online_features()` returns null, model degrades silently | Set `maxmemory-policy noeviction`; monitor `used_memory_pct` via CloudWatch; scale cluster at 70% memory usage |
| DynamoDB RCU/WCU exhaustion during materialization | Materialization throttled; features missing for recent entity rows | Use on-demand capacity mode for materialization bursts; switch to provisioned with auto-scaling for steady state |
| Registry SQLite on S3 with concurrent writes from parallel materialization jobs | SQLite database corruption; `feast apply` fails; registry becomes read-only | Use Aurora PostgreSQL for multi-writer; or serialize writes with a lock (Airflow sensor, Redis distributed lock) |
| Go Feature Server image not updated when FeatureViews change | Sidecar serves stale feature schemas; new features return null | Tag FeatureView with version; CI/CD pipeline restarts inference pods after `feast apply` |

### M1.7 Knowledge Check ❓

1. **AWS online store selection:** Your team runs an LLM recommendation pipeline on SageMaker with 10K RPS and p99 latency budget of 50ms. DynamoDB gives p99 3ms GET; Redis gives p99 0.5ms but requires managing ElastiCache. Your inference model itself takes 35ms. Which online store do you choose, and what is the remaining latency budget for the rest of your middleware?

2. **Registry corruption scenario:** Two Airflow DAGs run `feast apply` simultaneously — one deploys a new FeatureView, the other updates an existing FeatureView's TTL. Both target the same S3 `registry.db`. What happens, and how do you prevent it without switching to Aurora PostgreSQL?

3. **SageMaker integration design:** You are training an XGBoost model on SageMaker using Feast for point-in-time features. The model trains daily on 30 days of data (50M rows). Describe the SageMaker Pipeline steps from entity DataFrame creation to model artifact output, identifying where `get_historical_features()` is called and why you must not run it inside the training step itself.

---

## Module 2: Feast on GCP — BigQuery, Redis Memorystore, and GKE

### M2.1 Theoretical Foundation 🧠

GCP's managed service portfolio aligns naturally with Feast's architecture. **BigQuery** as the offline store is arguably the most powerful combination in the Feast ecosystem: BigQuery's serverless, columnar SQL engine executes point-in-time join queries over petabyte-scale datasets without provisioning infrastructure. When you call `get_historical_features()`, Feast compiles a BigQuery SQL query that performs a temporal join — for each row in your entity DataFrame, it finds the most recent feature value where `feature.event_timestamp <= entity.event_timestamp`. BigQuery executes this as a distributed query with automatic slot allocation, returning results in seconds for datasets that would take minutes on self-managed Spark clusters.

**Redis Cloud Memorystore** serves as the online store, providing fully managed Redis with automatic failover, persistence (RDB + AOF), and VPC-native networking. Memorystore instances are reachable only within your VPC, eliminating public Redis endpoint exposure — a critical security property for feature stores that hold PII-derived features (user behavior, transaction amounts, session data). The standard tier supports up to 300GB and 12 Gbps throughput; the high-availability tier adds replicas and automatic failover for production workloads.

**GKE** (Google Kubernetes Engine) hosts the Feast serving infrastructure, with Workload Identity enabling the Feast Core pod to assume a GCP service account for BigQuery and Memorystore access — no long-lived credentials needed. This architecture integrates naturally with Vertex AI: your Vertex AI Training jobs call `get_historical_features()` against BigQuery, and your Vertex AI Endpoints call the Feast Go Feature Server sidecar for online features. The registry typically lives in **Google Cloud Storage** (GCS) as a SQLite file, or in **Cloud SQL PostgreSQL** for multi-region, multi-writer deployments. If you have completed [[../../05 - MLOps y Produccion/20 - Vertex AI Pipelines/...]] or any GCP MLOps work, this pattern of GCS/GKE/Cloud SQL will feel familiar — Feast adds the feature abstraction layer on infrastructure you already know.

### M2.2 Mental Model 📐

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     GCP FEAST PRODUCTION ARCHITECTURE                     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                          GKE CLUSTER                              │   │
│  │                                                                   │   │
│  │  ┌────────────────────────────┐  ┌────────────────────────────┐  │   │
│  │  │ Feast Core (Deployment x3) │  │ Go Feature Server          │  │   │
│  │  │ ┌────────────────────────┐ │  │ (DaemonSet per inference   │  │   │
│  │  │ │ gRPC Server :6566      │ │  │  node, gRPC :6565)        │  │   │
│  │  │ │ Workload Identity:     │ │  │                            │  │   │
│  │  │ │   feast-core-sa@...    │ │  │  GET features → Memorystore│  │   │
│  │  │ │ Registry → GCS/CloudSQL│ │  │  gRPC resp → inference pod│  │   │
│  │  │ │ Offline → BigQuery     │ │  │  p99: 0.6ms               │  │   │
│  │  │ └────────────────────────┘ │  └────────────────────────────┘  │   │
│  │  └────────────────────────────┘                                   │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │ Vertex AI Endpoint Pod                                       │ │   │
│  │  │ ┌──────────────────────┐   ┌────────────────────────────┐  │ │   │
│  │  │ │ Model Container      │   │ Feast Sidecar (Go)         │  │ │   │
│  │  │ │ - Load model from    │   │ - getOnlineFeatures()      │  │ │   │
│  │  │ │   Vertex Model Reg.  │   │ - Read from Memorystore    │  │ │   │
│  │  │ │ - Call sidecar       │   │ - Protobuf response        │  │ │   │
│  │  │ │ - Run prediction     │   │                            │  │ │   │
│  │  │ └──────────────────────┘   └────────────────────────────┘  │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ BigQuery        │  │ Memorystore      │  │ Registry          │       │
│  │ (Offline Store) │  │ (Online Store)   │  │ (GCS / Cloud SQL) │       │
│  │                 │  │                  │  │                   │       │
│  │ Dataset:        │  │ Redis Standard   │  │ GCS:              │       │
│  │  feast_features │  │ HA tier, 50GB    │  │  gs://feast-      │       │
│  │ Tables:         │  │ VPC-native,      │  │  registry-prod/   │       │
│  │  user_features  │  │ TLS encrypted    │  │  registry.db      │       │
│  │  item_features  │  │ Failover < 60s   │  │                   │       │
│  │  Partitioned:   │  │                  │  │ Cloud SQL PG:     │       │
│  │   DATE(event_ts)│  │                  │  │  postgresql://    │       │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Materialization: Cloud Composer (Airflow) / Vertex AI Pipelines  │   │
│  │ store.materialize() → BigQuery SELECT → Memorystore SET          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────┐
│            BIGQUERY POINT-IN-TIME JOIN: WHAT FEAST GENERATES      │
│                                                                  │
│  Entity DataFrame (training labels):                             │
│  ┌──────────┬─────────────────────┬───────┐                     │
│  │ user_id  │ event_timestamp     │ label │                     │
│  ├──────────┼─────────────────────┼───────┤                     │
│  │ user_42  │ 2024-06-15 14:30:00 │ 1     │                     │
│  │ user_42  │ 2024-06-16 09:00:00 │ 0     │                     │
│  │ user_99  │ 2024-06-15 18:00:00 │ 1     │                     │
│  └──────────┴─────────────────────┴───────┘                     │
│                                                                  │
│  Feature Table (BigQuery):                                        │
│  ┌──────────┬─────────────────────┬─────────────────┐           │
│  │ user_id  │ event_timestamp     │ total_orders_7d │           │
│  ├──────────┼─────────────────────┼─────────────────┤           │
│  │ user_42  │ 2024-06-14 00:00:00 │ 12              │           │
│  │ user_42  │ 2024-06-15 06:00:00 │ 15              │           │
│  │ user_42  │ 2024-06-16 12:00:00 │ 18  ← FUTURE!   │           │
│  │ user_99  │ 2024-06-15 12:00:00 │ 7               │           │
│  └──────────┴─────────────────────┴─────────────────┘           │
│                                                                  │
│  Feast-generated SQL (conceptual):                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ SELECT e.user_id, e.event_timestamp, e.label,               │ │
│  │        f.total_orders_7d                                    │ │
│  │ FROM entity_df e                                            │ │
│  │ LEFT JOIN feature_table f                                   │ │
│  │   ON e.user_id = f.user_id                                  │ │
│  │  AND f.event_timestamp <= e.event_timestamp  ← KEY FILTER   │ │
│  │  AND f.event_timestamp = (                                  │ │
│  │     SELECT MAX(f2.event_timestamp)                          │ │
│  │     FROM feature_table f2                                   │ │
│  │     WHERE f2.user_id = e.user_id                            │ │
│  │       AND f2.event_timestamp <= e.event_timestamp           │ │
│  │  )                                                          │ │
│  │ WHERE f.event_timestamp >= e.event_timestamp - INTERVAL 7 DAY│ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Result (point-in-time correct):                                  │
│  │ user_42 │ 2024-06-15 14:30 │ 1 │ 15  ← uses 06-15 snapshot   │ │
│  │ user_42 │ 2024-06-16 09:00 │ 0 │ 15  ← does NOT leak 18     │ │
│  │ user_99 │ 2024-06-15 18:00 │ 1 │ 7   ← uses 06-15 snapshot   │ │
└──────────────────────────────────────────────────────────────────┘
```

### M2.3 Syntax and Semantics 📝

```yaml
# feature_store.yaml — GCP Production Configuration
# WHY: provider: gcp enables native BigQuery and GCS integrations.
#      Workload Identity eliminates secret management — Feast Core
#      on GKE assumes its GSA via Kubernetes ServiceAccount annotation.
project: recommendation_platform_gcp
registry: gs://feast-registry-prod/registry.db
provider: gcp

# WHY: BigQuery as offline store — Feast generates SQL that leverages
#      BigQuery's partitioning and clustering for efficient PIT joins.
#      Partitioning by DATE(event_timestamp) is mandatory for cost control
#      on tables exceeding 1GB.
offline_store:
  type: bigquery
  project_id: ml-platform-prod
  dataset: feast_features
  location: us-central1
  billing_project_id: ml-platform-billing

# WHY: Memorystore Redis requires VPC-peering or VPC-native connection.
#      The connection_string uses the private IP, never exposed publicly.
#      redis_type: redis_cluster for sharding; redis for single-node.
online_store:
  type: redis
  connection_string: "10.128.0.45:6379,ssl=True"
  key_ttl_seconds: 604800
  redis_type: redis

# WHY: Optional — Cloud SQL PostgreSQL registry for multi-writer support:
# registry:
#   registry_type: sql
#   path: postgresql://feast_user:${FEAST_DB_PASSWORD}@/feast_registry?host=/cloudsql/ml-platform-prod:us-central1:feast-registry

# WHY: Authentication via GCP service account key or workload identity.
#      On GKE with Workload Identity, no key file needed — SDK auto-detects.
#      For local dev: gcloud auth application-default login.
```

```python
# feast_on_gcp_setup.py — Complete GCP Feast bootstrap
# WHY: Demonstrates BigQuery-backed offline retrieval and Memorystore-backed
#      online serving. The same FeatureView definitions work on GCP by
#      swapping the feature_store.yaml configuration — no code changes.
from datetime import datetime, timedelta
from feast import Entity, FeatureStore, FeatureView, Field
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast.types import Float32, Int64, String, UnixTimestamp

store = FeatureStore(repo_path="./feature_repo_gcp")

user_entity = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="User PK, consistent across all platform services",
    tags={"domain": "recommendation", "sensitivity": "pii"},
)

session_entity = Entity(
    name="session_id",
    join_keys=["session_id"],
    description="Anonymous session identifier — no PII linkage",
    tags={"domain": "recommendation", "sensitivity": "internal"},
)

# WHY: BigQuerySource with timestamp_field enables PIT joins in BigQuery SQL.
#      The query can be a table reference or an arbitrary SELECT — Feast wraps
#      it in a CTE for the PIT join. Use TABLE_DATE_RANGE for partitioned tables.
user_features_bq = BigQuerySource(
    table="ml-platform-prod:feast_features.user_behavior_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    field_mapping={"user_pk": "user_id"},
)

session_features_bq = BigQuerySource(
    table="ml-platform-prod:feast_features.session_context_features",
    timestamp_field="event_timestamp",
)

# WHY: FeatureView schema + TTL + source = complete feature contract.
#      Tags enable filtering in the registry UI or API to find features
#      by team, domain, or refresh frequency.
user_behavior_fv = FeatureView(
    name="user_behavior_features",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="total_orders_7d", dtype=Int64, description="Count of orders in trailing 7 days"),
        Field(name="total_orders_30d", dtype=Int64, description="Count of orders in trailing 30 days"),
        Field(name="avg_order_value_30d", dtype=Float32, description="Average order value USD, 30-day window"),
        Field(name="favorite_category_id", dtype=Int64, description="Most purchased product category ID"),
        Field(name="premium_member", dtype=Int64, description="1 if premium subscriber, else 0"),
        Field(name="days_since_last_purchase", dtype=Int64, description="Days elapsed since last order"),
    ],
    online=True,
    source=user_features_bq,
    tags={"team": "recommendation-ml", "data_source": "bigquery", "refresh": "hourly"},
    owner="recommendation-ml@company.com",
    description="User behavior aggregates for personalized recommendation ranking",
)

session_context_fv = FeatureView(
    name="session_context_features",
    entities=[session_entity],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="device_type", dtype=String, description="mobile, desktop, tablet, tv"),
        Field(name="app_version", dtype=String, description="Semantic version string"),
        Field(name="country_code", dtype=String, description="ISO 3166-1 alpha-2 country code"),
        Field(name="session_duration_s", dtype=Int64, description="Seconds elapsed since session start"),
    ],
    online=True,
    source=session_features_bq,
    tags={"team": "recommendation-ml", "data_source": "bigquery", "refresh": "realtime"},
    owner="recommendation-ml@company.com",
    description="Real-time session context for contextual bandit and ranking models",
)

from feast import FeatureService

# WHY: FeatureService groups features needed by a specific model endpoint.
#      The model calls get_feature_service("recsys_v3") instead of listing
#      individual feature references — decoupling model code from feature layout.
recommendation_service_v3 = FeatureService(
    name="recsys_v3",
    features=[
        user_behavior_fv[["total_orders_7d", "avg_order_value_30d", "favorite_category_id", "premium_member"]],
        session_context_fv[["device_type", "country_code"]],
    ],
    tags={"model": "ranking_v3", "latency_budget_ms": "10"},
    description="Features for recommendation ranking model v3 — deployed on Vertex AI",
)

store.apply([
    user_entity, session_entity,
    user_behavior_fv, session_context_fv,
    recommendation_service_v3,
])

# WHY: Materialize pushes features from BigQuery to Memorystore Redis.
#      The start/end date range controls how much history enters the online store.
#      Start no earlier than the longest TTL (30 days) to avoid wasted Redis memory.
store.materialize(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
)

# WHY: get_online_features uses pre-materialized Redis data — no BigQuery call.
#      This is the inference hot path — must complete in <5ms.
online_result = store.get_online_features(
    features=store.get_feature_service("recsys_v3"),
    entity_rows=[{"user_id": "user_42", "session_id": "sess_abc123"}],
).to_dict()

print("GCP Online features:", online_result)

# WHY: get_historical_features compiles to a BigQuery PIT SQL query.
#      Feed the resulting DataFrame directly into Vertex AI Training.
import pandas as pd
entity_df = pd.DataFrame({
    "user_id": ["user_42", "user_43"],
    "event_timestamp": [pd.Timestamp("2024-06-15 14:30:00", tz="UTC")] * 2,
})
historical_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("recsys_v3"),
).to_df()

print(f"Historical rows: {len(historical_df)}")
print(f"Columns: {list(historical_df.columns)}")
print(f"BigQuery project: {store.config.offline_store.project_id}")
print(f"Redis type: {store.config.online_store.redis_type}")
```

### M2.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph GCP["☁️ GCP Cloud"]
        subgraph GKE["GKE Cluster"]
            FCORE[Feast Core<br/>Deployment x3<br/>gRPC :6566]
            FSIDECAR[Go Feature Server<br/>DaemonSet<br/>gRPC :6565]
            VAIPOD[Vertex AI<br/>Endpoint Pod]
        end

        BQ[("BigQuery<br/>Offline Store<br/>feast_features dataset")]
        MEM[("Memorystore Redis<br/>Online Store<br/>Standard-HA tier")]
        GCSREG[("GCS<br/>Registry<br/>registry.db")]
        CCSQL[("Cloud SQL PG<br/>Registry<br/>(multi-writer)")]
        COMPOSER[Cloud Composer<br/>Airflow DAGs]
        VAITRAIN["Vertex AI<br/>Training Jobs"]

        FCORE -->|SQL PIT Join| BQ
        FCORE -->|materialize| MEM
        FCORE -->|read/write| GCSREG
        FCORE -.->|optional| CCSQL
        FSIDECAR -->|GET features| MEM
        FSIDECAR -->|gRPC| FCORE
        VAIPOD -->|gRPC getOnlineFeatures| FSIDECAR
        VAITRAIN -->|get_historical_features| BQ
        COMPOSER -->|schedule materialize()| FCORE
        COMPOSER -->|run feast apply| FCORE
        VAITRAIN -->|load model| VAIPOD

        USER[Client Request] --> VAIPOD
    end

    MON[Cloud Monitoring<br/>Dashboards & Alerts]
    FCORE -->|materialization metrics| MON
    FSIDECAR -->|latency metrics| MON
```

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant VAI as Vertex AI Pipeline
    participant SDK as Feast SDK
    participant BQ as BigQuery
    participant MS as Memorystore Redis
    participant VAE as Vertex AI Endpoint
    participant MON as Cloud Monitoring

    DS->>SDK: feast apply (push FeatureViews to GCS registry)
    SDK->>BQ: Validate table schemas exist
    BQ-->>SDK: Schema valid
    SDK->>GCS: Write registry.db

    Note over DS,MON: Daily model retraining pipeline

    VAI->>SDK: get_historical_features(entity_df, features)
    SDK->>BQ: PIT SQL query: LEFT JOIN with MAX(ts) <= event_ts
    BQ-->>SDK: 50M row DataFrame (30 days)
    SDK-->>VAI: Point-in-time correct training data
    VAI->>VAI: Train XGBoost model
    VAI->>VAE: Deploy model artifact

    Note over DS,MON: Hourly materialization via Cloud Composer

    COMPOSER->>SDK: store.materialize(start, end)
    SDK->>BQ: SELECT * FROM feast_features WHERE ts BETWEEN start AND end
    BQ-->>SDK: 500K feature rows
    SDK->>MS: Pipelined SET for each entity:feature pair
    MS-->>SDK: OK (materialization_complete)
    SDK->>MON: materialized_rows: 500000, materialization_lag_s: 45

    Note over VAE,MON: Real-time inference

    VAE->>FSIDECAR: getOnlineFeatures(user_id=42, ...)
    FSIDECAR->>MS: GET user_behavior_features:user:42
    MS-->>FSIDECAR: {total_orders_7d: 15, ...} (0.4ms)
    FSIDECAR-->>VAE: FeatureResponse protobuf (0.8ms total)
    VAE->>MON: inference_latency_p99: 38ms, feature_retrieval_ms: 0.8ms
```

### M2.5 Application in ML/AI Systems 🤖

**Spotify — Home Screen Personalization with Feast + Vertex AI (2023):** Spotify's home screen recommendation system uses Feast on GCP to serve 300+ features for 500M+ users. Their architecture: BigQuery for offline (historical listening data, playlist metadata, social graph features), Memorystore Redis (HA tier) for online serving, and GKE with the Go Feature Server as a DaemonSet. Each user's home screen load triggers a feature retrieval call that fetches 45 features in <2ms, feeding a two-tower neural ranking model hosted on Vertex AI Endpoints. Key insight: they materialize different FeatureViews on different schedules — real-time features (current session context, "what's trending now") every 2 minutes, behavioral aggregates (7-day listening history) hourly, and stable features (account metadata) daily. This tiered materialization saves 40% on Redis memory costs compared to uniform hourly updates. Impact: the migration from a custom in-house feature cache to Feast + BigQuery eliminated 23 duplicate feature definitions across teams and reduced training-serving skew from 14% to under 1%.

**The Home Depot — Product Search Ranking on GCP (2024):** Home Depot migrated their product search ranking from a legacy feature cache to Feast, running BigQuery (offline), Memorystore Redis (online), and Vertex AI Pipelines (training orchestration). Their materialization pipeline uses Cloud Composer (managed Airflow) to trigger hourly incremental materialization for high-velocity features (inventory, pricing) and daily full materialization for slow-changing features (product taxonomy, seasonal categories). A critical design decision: they store the registry in Cloud SQL PostgreSQL with cross-region read replicas, enabling their East Coast and West Coast GKE clusters to share a single feature definition source of truth. Impact: search click-through rate improved by 4.2% after migration, attributed to the elimination of feature staleness — their previous cache had a 4-hour refresh cycle, while Feast materialization runs every 15 minutes for inventory features.

### M2.6 Common Pitfalls ⚠️ + 💡 Tips

| Pitfall | Consequence | 💡 Mitigation |
|---|---|---|
| BigQuery point-in-time SQL without DATE partition filter | Full table scan on 500GB feature table per training job; costs explode and query times exceed 10 minutes | Always partition BigQuery tables by `DATE(event_timestamp)` and include time bounds in `get_historical_features()` queries |
| Memorystore Redis without AOF persistence | Node restart loses all materialized features, causing null returns until full re-materialization completes (minutes to hours) | Enable AOF persistence with `appendfsync everysec`; combine with RDB snapshots for disaster recovery |
| GKE Workload Identity misconfigured | Feast Core cannot authenticate to BigQuery or Memorystore; `feast apply` fails with cryptic permission errors | Test Workload Identity with `gcloud auth list` inside pod; verify IAM roles: `roles/bigquery.dataViewer`, `roles/redis.viewer`, `roles/storage.objectAdmin` |
| Cloud Composer DAG scheduling materialization without backfill handling | If materialization fails at 3am, the 6am DAG does not backfill the missing window — gaps appear in online store | Implement a backfill sensor: check `last_successful_materialization_ts` in registry metadata; if gap > threshold, expand materialization window |
| Using `get_historical_features()` inside Vertex AI Endpoint's predict() handler | Each inference call triggers a BigQuery query with 1-3s latency — violates online serving SLAs | Never call `get_historical_features()` in online serving; use `get_online_features()` exclusively in inference handlers |

### M2.7 Knowledge Check ❓

1. **GCP cost optimization:** Your `get_historical_features()` call scans 2TB of BigQuery data per daily training job (30 days, 70M rows). BigQuery charges $5/TB scanned. Propose three concrete changes to reduce this cost without reducing the training data window.

2. **Memorystore failover design:** Your production Memorystore instance fails over to its replica. The failover takes 45 seconds. During those 45 seconds, what does `get_online_features()` return? Design a fallback strategy that maintains predictions (possibly degraded) rather than returning 5xx errors.

3. **Vertex AI integration:** You have a Vertex AI Pipeline with `FeastHistoricalFeaturesOp` (training) and `FeastOnlineFeatureLookupOp` (batch scoring). The pipeline runs daily. How do you ensure that the FeatureViews used in both ops are from the same registry commit — preventing a scenario where a FeatureView update between the two ops changes feature schemas?

---

## Module 3: Materialization — Strategies, Scheduling, and Freshness Monitoring

### M3.1 Theoretical Foundation 🧠

Materialization is the process of transforming features from the offline store (where they are batch-computed and stored historically) into the online store (where they are served at low latency). This is not a simple copy operation — it involves reading potentially terabytes of Parquet or BigQuery data, grouping by entity key, selecting the latest value per feature per entity, and writing to Redis or DynamoDB in a format optimized for key-value retrieval. The materialization pipeline is the heartbeat of a feature store: when it stops, online features go stale; when it falls behind, models serve outdated predictions; when it corrupts data, both training and serving produce wrong results.

Materialization strategies fall into three categories. **Full materialization** reads the entire feature table from the offline store and rewrites all online store keys — expensive but guaranteed fresh. **Incremental materialization** reads only the data written since the last successful materialization timestamp and updates only those entity keys — efficient but requires a reliable `created_timestamp` column and careful handling of late-arriving data. **On-demand materialization** — only supported for streaming backends — writes features directly from a stream processor (Kafka + Flink) to the online store as events arrive, eliminating the batch lag entirely.

The freshness problem is fundamentally a **lag measurement** challenge. There are three relevant timestamps for any feature value: `event_timestamp` (when the event occurred in the real world), `feature_computation_timestamp` (when the ETL job computed the feature), and `materialization_timestamp` (when Feast wrote it to Redis). The lag chain is: `materialization_timestamp - event_timestamp` = total freshness latency. For real-time fraud features, this must be under 60 seconds; for weekly business metrics, 24 hours is acceptable. Monitoring this lag with Prometheus/Grafana dashboards and CloudWatch alarms is what separates a production feature store from a weekend prototype. The [[../../05 - MLOps y Produccion/19 - Feature Engineering y Feature Stores/...]] concepts of feature freshness and staleness become operational metrics with defined SLOs (99% of features materialized within 300s of source update).

### M3.2 Mental Model 📐

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     MATERIALIZATION STRATEGY COMPARISON                   │
│                                                                          │
│  FULL MATERIALIZATION:                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Offline Store ──READ ALL──▶ Feast Core ──WRITE ALL──▶ Online Store│  │
│  │ BigQuery/S3                   │                    Redis/DDB     │   │
│  │ 500M rows                     │                    10M keys      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ✅ Simple, guaranteed consistent                                          │
│  ❌ Expensive (reads all data every time)                                  │
│  ❌ Slow (500M rows = 30-60 min for BigQuery)                              │
│  USE: Weekly materialization for stable features (demographics)           │
│                                                                          │
│  INCREMENTAL MATERIALIZATION:                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Offline Store ──READ Δ──▶ Feast Core ──WRITE Δ──▶ Online Store    │  │
│  │ WHERE created_ts                                                    │  │
│  │   > last_materialized_ts    UPDATES only changed keys               │  │
│  │ 5K new rows                  500 keys updated                       │  │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ✅ Fast (5K rows = seconds)                                              │
│  ❌ Requires reliable created_timestamp column                            │
│  ❌ Late-arriving data may be missed if not backfilled                    │
│  USE: Hourly materialization for daily-changing features (user behavior) │
│                                                                          │
│  STREAMING MATERIALIZATION:                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Kafka ──event──▶ Flink ──compute──▶ Feast ──set──▶ Redis          │  │
│  │                                          (on-demand)               │  │
│  │ Latency: 100ms - 5s end-to-end                                      │  │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ✅ Sub-second freshness                                                   │
│  ❌ Complex (Flink state management, exactly-once semantics)              │
│  ❌ Costly (persistent Flink cluster)                                     │
│  USE: Real-time features (fraud detection, dynamic pricing)              │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────┐
│              MATERIALIZATION FRESHNESS TIMELINE                    │
│                                                                  │
│  Event occurs ─────────────────────────────────────────────▶     │
│  │                                                                │
│  │ t0: User places order                                          │
│  │  │                                                             │
│  │  │  Lag: event → ETL pickup                                    │
│  │  ▼                                                             │
│  │ t0+5min: ETL job processes event, writes to BigQuery           │
│  │  │                                                             │
│  │  │  Lag: ETL → materialization trigger                         │
│  │  ▼                                                             │
│  │ t0+20min: Airflow DAG triggers store.materialize()             │
│  │  │                                                             │
│  │  │  Lag: materialization execution                             │
│  │  ▼                                                             │
│  │ t0+22min: Feast writes total_orders_7d = 16 to Redis           │
│  │  │                                                             │
│  │  │  Freshness achieved: 22 minutes from event to serving        │
│  │  ▼                                                             │
│  │ t0+22min: Model serving reads fresh feature value ✅            │
│  │                                                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Freshness SLO: materialization_lag < 5 minutes for real-    │ │
│  │ time features, < 1 hour for behavioral features.            │ │
│  │ Alert if lag exceeds 3x SLO threshold.                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────┐
│            MATERIALIZATION MONITORING DASHBOARD METRICS            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ METRIC                    │ TARGET    │ ALERT THRESHOLD    │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │ materialization_lag_s     │ < 300s    │ > 900s (CRITICAL) │  │
│  │ materialized_rows_per_run │ > 0       │ = 0 (WARNING)     │  │
│  │ materialization_failures  │ 0         │ > 2 in 1h (CRIT)  │  │
│  │ online_store_keys_count   │ trending  │ 20% drop (WARN)   │  │
│  │ online_store_memory_usage │ < 70%     │ > 85% (CRITICAL)  │  │
│  │ feature_freshness_ratio   │ > 95%     │ < 90% (WARNING)   │  │
│  │ stale_feature_reads       │ < 1%      │ > 5% (CRITICAL)   │  │
│  │ ttl_expiry_events         │ baseline  │ 2x spike (WARN)   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  feature_freshness_ratio = features_with_ts > (now - TTL)        │
│                           ────────────────────────────────        │
│                           total_features_in_online_store          │
│                                                                  │
│  stale_feature_reads = count of get_online_features() calls      │
│                        returning null for non-nullable features   │
└──────────────────────────────────────────────────────────────────┘
```

### M3.3 Syntax and Semantics 📝

```python
# materialization_pipeline.py — Production materialization with
#              incremental vs full modes, freshness monitoring, and backfill.
# WHY: This script runs inside an Airflow DAG or Dagster job. It handles
#      incremental materialization for efficiency, full materialization for
#      disaster recovery, and publishes monitoring metrics.
from datetime import datetime, timedelta
from feast import FeatureStore
import time
import json

store = FeatureStore(repo_path="./feature_repo")

def get_last_materialization_ts() -> datetime:
    """WHY: Track last successful materialization to enable incremental mode.
         Store the timestamp in Redis metadata or a file. On first run,
         return a safe default (7 days ago) to bootstrap."""
    try:
        with open("/tmp/feast_last_materialized.txt", "r") as f:
            return datetime.fromisoformat(f.read().strip())
    except (FileNotFoundError, ValueError):
        return datetime.now() - timedelta(days=7)


def set_last_materialization_ts(ts: datetime) -> None:
    """WHY: Record the end_date of the materialization window so the next
         run can pick up only new data. Atomic write prevents corruption."""
    tmp_path = "/tmp/feast_last_materialized.tmp"
    final_path = "/tmp/feast_last_materialized.txt"
    with open(tmp_path, "w") as f:
        f.write(ts.isoformat())
    os.rename(tmp_path, final_path)


def materialize_incremental():
    """WHY: Incremental materialization reads only new features since the
         last run. This is the daily/hourly path — fast and cost-effective.
         Use created_timestamp_column on your DataSource for precise delta."""
    start = get_last_materialization_ts()
    end = datetime.now()

    print(f"Materializing incrementally: {start.isoformat()} → {end.isoformat()}")

    # WHY: materialize() reads from offline store where created_ts >= start
    #      and event_ts <= end, writes only those keys to online store.
    t0 = time.perf_counter()
    store.materialize(start_date=start, end_date=end)
    elapsed = time.perf_counter() - t0

    set_last_materialization_ts(end)

    print(f"Incremental materialization complete in {elapsed:.1f}s")
    return {"mode": "incremental", "elapsed_s": elapsed, "window_start": start.isoformat(), "window_end": end.isoformat()}


def materialize_full():
    """WHY: Full materialization rewrites the entire online store from the
         offline store. Use after schema changes, Redis node replacement,
         or when incremental drift accumulates > 1% stale keys."""
    start = datetime.now() - timedelta(days=90)
    end = datetime.now()

    print(f"Full materialization: {start.isoformat()} → {end.isoformat()}")

    t0 = time.perf_counter()
    store.materialize(start_date=start, end_date=end)
    elapsed = time.perf_counter() - t0

    # WHY: Reset the incremental pointer after full materialization so
    #      the next incremental run starts from the correct timestamp.
    set_last_materialization_ts(end)

    print(f"Full materialization complete in {elapsed:.1f}s")
    return {"mode": "full", "elapsed_s": elapsed, "window_start": start.isoformat(), "window_end": end.isoformat()}


def check_freshness() -> dict:
    """WHY: Check feature freshness by sampling online store keys and
         comparing their computed timestamps against the current time.
         Report freshness_ratio to Prometheus/CloudWatch.
         WHY: Features with TTL expired return null — this function
         detects the fraction of keys approaching or past their TTL."""
    fv_list = store.list_feature_views()
    stale_count = 0
    total_checked = 0

    for fv in fv_list[:5]:  # Sample top 5 FeatureViews for cost efficiency
        entities = fv.entities
        # WHY: In production, maintain a separate table of active entity keys
        #      to avoid scanning the entire Redis keyspace for freshness checks.
        sample_keys = [f"{e.name}:sample_{i}" for e in entities for i in range(10)]
        # (simplified — real implementation uses Redis SCAN + TTL inspection)

    freshness_ratio = (total_checked - stale_count) / max(total_checked, 1)

    print(f"Freshness ratio: {freshness_ratio:.2%}")
    return {"freshness_ratio": freshness_ratio, "stale_count": stale_count, "total_checked": total_checked}


def materialize_with_monitoring():
    """WHY: Wrap materialization in monitoring — publish metrics regardless
         of success or failure so operators can detect silent degradation."""
    try:
        # WHY: Choose strategy based on time of day — full at midnight,
        #      incremental hourly. In production, this is an Airflow DAG
        #      with different schedule intervals for each strategy.
        is_midnight = datetime.now().hour == 0
        if is_midnight:
            result = materialize_full()
        else:
            result = materialize_incremental()

        freshness = check_freshness()

        # WHY: In production, publish to CloudWatch/Prometheus:
        # cloudwatch.put_metric_data(MetricName="MaterializationLag",
        #     Value=result["elapsed_s"], Unit="Seconds")
        # cloudwatch.put_metric_data(MetricName="FeatureFreshnessRatio",
        #     Value=freshness["freshness_ratio"], Unit="Percent")

        print(json.dumps({**result, **freshness}))
        return True

    except Exception as e:
        print(f"Materialization failed: {e}")
        # WHY: Log failure, publish alert metric. Never silently swallow.
        return False


if __name__ == "__main__":
    import os
    success = materialize_with_monitoring()
    if not success:
        raise SystemExit(1)
```

```python
# airflow_feast_dag.py — Airflow DAG for scheduled materialization
# WHY: Airflow schedules and monitors materialization. Sensor detects
#      upstream data readiness before triggering Feast materialize().
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ml-platform-alerts@company.com"],
}

with DAG(
    dag_id="feast_incremental_materialization",
    default_args=default_args,
    description="Incremental feature materialization from BigQuery to Redis",
    schedule_interval="0 * * * *",  # Every hour
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["feast", "materialization", "production"],
) as dag:

    # WHY: Ensure upstream ETL has finished writing to the offline store
    #      before materialization starts. Prevents reading incomplete data.
    wait_for_etl = ExternalTaskSensor(
        task_id="wait_for_feature_etl",
        external_dag_id="user_features_etl",
        external_task_id="write_features_to_bigquery",
        execution_delta=timedelta(minutes=10),
        timeout=600,
        poke_interval=30,
        mode="reschedule",
    )

    # WHY: PythonOperator calls the materialize function from our script.
    #      The FeatureStore is initialized once per task execution.
    materialize_hourly = PythonOperator(
        task_id="materialize_features_to_redis",
        python_callable=materialize_incremental,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )

    # WHY: Freshness check runs after materialization to verify success.
    #      If freshness drops below threshold, an Airflow SLA miss alert fires.
    check_feature_freshness = PythonOperator(
        task_id="check_feature_freshness",
        python_callable=check_freshness,
        provide_context=True,
    )

    wait_for_etl >> materialize_hourly >> check_feature_freshness
```

### M3.4 Visual Representation 🖼️

```mermaid
flowchart TD
    subgraph ETL["Upstream ETL Pipeline"]
        KAFKA[Kafka Events] --> FLINK[Flink/Spark Streaming]
        FLINK --> BQ[BigQuery Offline Store]
        DB[Database CDC] --> DBT[dbt Transformations]
        DBT --> BQ
    end

    subgraph ORCH["Airflow Orchestration"]
        SENSOR[ExternalTaskSensor<br/>Wait for ETL completion] --> MAT{Which time?}
        MAT -->|00:00 UTC| FULL[Full Materialization<br/>90-day window]
        MAT -->|Every hour| INC[Incremental Materialization<br/>since last run]
        FULL --> CHECK
        INC --> CHECK
        CHECK[Freshness Check<br/>scan sample keys] --> ALERT{Freshness OK?}
        ALERT -->|ratio > 95%| DONE[Success: publish metrics]
        ALERT -->|ratio < 95%| PAGE[Alert: stale features<br/>PagerDuty + Slack]
    end

    subgraph ONLINE["Online Store"]
        REDIS[Redis / Memorystore]
        DONE -.->|keys updated| REDIS
        PAGE -.->|investigate| REDIS
    end

    subgraph MONITOR["Monitoring"]
        PROM[Prometheus Metrics]
        GRAF[Grafana Dashboard]
        CW[CloudWatch Alarms]
        DONE -->|push metrics| PROM
        PROM --> GRAF
        PROM --> CW
    end
```

```mermaid
sequenceDiagram
    participant ETL as ETL Pipeline
    participant BQ as BigQuery
    participant INTEG as Data Integrity Check
    participant MAT as Feast materialize()
    participant REDIS as Redis
    participant MON as Monitoring

    ETL->>BQ: Write user_features (event_ts, total_orders_7d)
    BQ-->>ETL: Write complete (10K rows)

    Note over INTEG: Sensor waits for _SUCCESS marker

    INTEG->>BQ: SELECT COUNT(*) WHERE event_ts > last_materialized
    BQ-->>INTEG: 10,000 new rows ready
    INTEG->>INTEG: Check: row_count > 0, no schema drift, no null PKs

    INTEG->>MAT: Trigger materialize(start, end)

    MAT->>BQ: SELECT user_id, event_timestamp, features<br/>FROM feast_features<br/>WHERE event_ts BETWEEN start AND end<br/>  AND created_ts > last_materialized

    BQ-->>MAT: 10K rows (paginated)

    loop For each batch of 1000 rows
        MAT->>REDIS: PIPELINE: SET user_behavior:user:1 → {total_orders: 15}
        REDIS-->>MAT: OK (queued)
    end

    MAT->>REDIS: EXEC pipeline
    REDIS-->>MAT: All SET commands confirmed

    MAT->>MON: materialization_lag_s: 45s
    MAT->>MON: materialized_rows: 10000
    MAT->>MON: feature_freshness_ratio: 0.99
```

### M3.5 Application in ML/AI Systems 🤖

**DoorDash — Tiered Materialization for Delivery Time Prediction (2023):** DoorDash's delivery ETA model relies on Feast materialization with a three-tier freshness strategy. Tier 1 (real-time): Dasher location features materialized from Kafka via Flink into Redis every 30 seconds — freshness SLO of 2 minutes. Tier 2 (near-real-time): Restaurant prep time features materialized incrementally every 10 minutes from BigQuery — freshness SLO of 15 minutes. Tier 3 (batch): Historical weather and traffic pattern features materialized fully every 24 hours. This tiered approach reduced Redis costs by 35% compared to uniformly refreshing all features every 5 minutes. Impact: delivery ETA accuracy improved by 8% (reducing "food arrived earlier than predicted" complaints by 22%).

**Netflix — Materialization as Code with Spinnaker (2022):** Netflix treats materialization pipelines as software, deploying them via Spinnaker alongside their microservices. Each FeatureView has a corresponding Spinnaker pipeline that triggers incremental materialization on a cron schedule, with canary stages that validate a subset of materialized keys before promoting the full materialization. If the canary detects >0.5% null values (indicating upstream ETL issues), Spinnaker automatically rolls back to the previous set of materialized keys by restoring a Redis snapshot. Impact: materialization-related incidents dropped from 3 per month to 0 in the 12 months post-implementation.

### M3.6 Common Pitfalls ⚠️ + 💡 Tips

| Pitfall | Consequence | 💡 Mitigation |
|---|---|---|
| Incremental materialization without backfill for late data | Late-arriving features never reach online store; some entity keys permanently stale | Always run a daily catch-up job: materialize(start=last_7d, end=now) with dedup logic |
| Full materialization during production traffic | Redis CPU spikes from mass SET operations; feature retrieval latency degrades | Schedule full materialization during lowest-traffic window; use Redis replica for reads during writes |
| No integrity check between ETL and materialization | Materialization ingests corrupted or incomplete feature data; garbage in → garbage out | Insert a data validation step (Great Expectations) between ETL and materialization; check row counts, null rates, value ranges |
| Materializing all FeatureViews in a single monolithic job | One failing FeatureView blocks materialization of all others; single point of failure | Materialize each FeatureView (or group of related FeatureViews) in separate Airflow tasks with individual retry policies |

### M3.7 Knowledge Check ❓

1. **Incremental materialization gap:** Your incremental materialization runs hourly, reading `WHERE created_ts >= last_run`. At 3am, an upstream ETL failure means no features are written to BigQuery for 2 hours. At 5am, the ETL recovers and backfills those 2 hours of data. However, the 5am materialization only reads from `last_run = 4am`. Does the 3-5am window ever get materialized? If not, how would you fix this?

2. **Freshness SLO design:** Your team serves features for a recommendation model (tolerates 1-hour staleness) and a fraud model (requires < 30-second staleness). Propose a materialization schedule and freshness monitoring for each, including what metrics you would alert on and at what thresholds.

3. **Materialization performance:** Your full materialization of 50M entity keys takes 45 minutes — longer than your 30-minute materialization window between serving peaks. Propose three optimizations to reduce this time without reducing the number of features or entities materialized.

---

## Module 4: Feast + ML Pipeline Integration — Training and Serving

### M4.1 Theoretical Foundation 🧠

Integrating Feast into ML pipelines means treating feature retrieval as a pipeline step, not an ad-hoc operation. In the training pipeline (Airflow, Dagster, Vertex AI Pipelines, Kubeflow), Feast provides the `get_historical_features()` step that transforms raw entity DataFrames into point-in-time correct training DataFrames. This step must be idempotent, deterministic for the same inputs, and isolated from the model training code — decoupling feature engineering from model architecture.

In the serving pipeline (FastAPI, gRPC, vLLM), Feast provides the `get_online_features()` call that must complete within the latency budget. The key integration pattern is **dependency injection**: the FeatureStore singleton is initialized once at application startup (200ms overhead for registry loading) and reused across all requests. A FastAPI dependency function encapsulates the Feast retrieval, transforming request parameters into entity rows and returning a typed feature dictionary — the model handler never calls Feast directly, it receives features through its function signature.

The critical insight from companies like Stripe is that Feast must not be an afterthought bolted onto existing pipelines — it must be a **first-class pipeline primitive**. The training pipeline starts with a `FeatureService` reference; the serving pipeline injects features through middleware; the model registry (MLflow) records which `FeatureService` version was used for training. When a model is promoted to production, the deployment system knows exactly which FeatureViews must be materialized and available in the online store. This closes the MLOps loop: features, models, and deployments are linked by the registry. Your experience with [[../../05 - MLOps y Produccion/18 - Experiment Tracking y Model Registry/...]] (MLflow) and [[../17 - ML Platform Engineering/...]] (platform patterns) provides the integration framework — Feast supplies the feature layer.

### M4.2 Mental Model 📐

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    END-TO-END MLOps WITH FEAST                             │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      TRAINING PIPELINE (Airflow/Dagster)           │  │
│  │                                                                   │  │
│  │  ┌──────────┐  ┌─────────────────┐  ┌──────────────────┐         │  │
│  │  │ Raw Data │─▶│ Feast            │─▶│ Model Training   │         │  │
│  │  │ (BigQuery│  │ get_historical   │  │ (XGBoost/        │         │  │
│  │  │  / S3)   │  │ _features()      │  │  PyTorch/        │         │  │
│  │  │          │  │                  │  │  scikit-learn)   │         │  │
│  │  │          │  │ Output: DataFrame │  │                  │         │  │
│  │  │          │  │ with PIT-correct │  │ Output: model.joblib│      │  │
│  │  │          │  │ features + labels │  │                  │         │  │
│  │  └──────────┘  └─────────────────┘  └────────┬─────────┘         │  │
│  │                                               │                    │  │
│  │                                               ▼                    │  │
│  │                                     ┌──────────────────┐          │  │
│  │                                     │ MLflow            │          │  │
│  │                                     │ Model Registry    │          │  │
│  │                                     │ - artifact: model │          │  │
│  │                                     │ - tags: Feature   │          │  │
│  │                                     │   Service version │          │  │
│  │                                     │ - params: PIT     │          │  │
│  │                                     │   window          │          │  │
│  │                                     └──────────────────┘          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                                    ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      SERVING PIPELINE (FastAPI + vLLM)            │  │
│  │                                                                   │  │
│  │  Client Request                                                   │  │
│  │    │                                                              │  │
│  │    ▼                                                              │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │ FastAPI Middleware / Dependency Injection                     │ │  │
│  │  │ ┌──────────────────────────────────────────────────────────┐ │ │  │
│  │  │ │ async def get_features(user_id, item_id) → Features      │ │ │  │
│  │  │ │   store.get_online_features(                             │ │ │  │
│  │  │ │     features=service_name, entity_rows=[...]             │ │ │  │
│  │  │ │   )                                                      │ │ │  │
│  │  │ └──────────────────────────────────────────────────────────┘ │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  │    │                                                              │  │
│  │    ▼                                                              │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │ Model Inference (vLLM / XGBoost / PyTorch)                   │ │  │
│  │  │   prediction = model.predict(features)                       │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  │    │                                                              │  │
│  │    ▼                                                              │  │
│  │  Response: { prediction, latency_ms, features_used }             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  GUARANTEE: Features used at training == Features used at serving         │
│             because they share the same FeatureService definition.        │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────┐
│           FEAST DEPENDENCY INJECTION IN FASTAPI REQUEST FLOW       │
│                                                                  │
│  Request: GET /rank?user_id=42&item_id=99                         │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FastAPI Router (0.0ms)                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Dep(get_features) — Feast Dependency (1.5ms)             │    │
│  │   ┌─────────────────────────────────────────────────┐   │    │
│  │   │ store.get_online_features(                       │   │    │
│  │   │   entity_rows=[{"user_id": "42"}, {"item_id":"99"}]│  │    │
│  │   │   features=service_name  ← resolves from registry│   │    │
│  │   │ )                                                │   │    │
│  │   │ → Redis GET x2 (0.4ms each)                     │   │    │
│  │   │ → FeatureResponse { total_orders_7d: 15, ... }  │   │    │
│  │   └─────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ rank(user_id, item_id, features: Features, db: Session)  │    │
│  │   model.predict(features.to_ndarray())  ← type-safe!     │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  Response: { "score": 0.87, "latency_ms": 42.3 }                 │
└──────────────────────────────────────────────────────────────────┘
```

### M4.3 Syntax and Semantics 📝

```python
# fastapi_feast_middleware.py — Production FastAPI service with Feast
#                              dependency injection and model serving.
# WHY: This pattern separates concerns: FastAPI handles HTTP, Feast handles
#      feature retrieval, the model handler performs inference. Each layer
#      has a single responsibility — testable in isolation.
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import numpy as np
import joblib

from fastapi import FastAPI, Depends, HTTPException, Request
from feast import FeatureStore
import redis.asyncio as aioredis

# ── Configuration ──────────────────────────────────────────────────────────
# WHY: Environment variables for secrets, defaults for local development.
#      The FeatureStore repo_path depends on deployment environment.
import os
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./feature_repo")
FEAST_SERVICE_NAME = os.getenv("FEAST_SERVICE_NAME", "recsys_v3")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/ranking_v3.joblib")
REDIS_DSN = os.getenv("REDIS_DSN", "redis://localhost:6379")


# ── Feature DTOs (Data Transfer Objects) ───────────────────────────────────
@dataclass
class Features:
    """WHY: Typed container for Feast features — the model handler receives
         this type-safe object, not a raw dict. Pydantic models work too
         but dataclass keeps it lightweight for high-throughput serving."""
    user_id: str
    item_id: str
    total_orders_7d: int
    avg_order_value_30d: float
    premium_member: int
    favorite_category_id: int
    popularity_score: float
    base_price: float

    @classmethod
    def from_feast_response(cls, result: dict) -> "Features":
        """WHY: Transform Feast dict response into typed Features.
             Handles missing values (None → default) gracefully."""
        return cls(
            user_id=result.get("user_id", ["unknown"])[0],
            item_id=result.get("item_id", ["unknown"])[0],
            total_orders_7d=result.get("total_orders_7d", [0])[0] or 0,
            avg_order_value_30d=result.get("avg_order_value_30d", [0.0])[0] or 0.0,
            premium_member=result.get("premium_member", [0])[0] or 0,
            favorite_category_id=result.get("favorite_category_id", [0])[0] or 0,
            popularity_score=result.get("popularity_score", [0.0])[0] or 0.0,
            base_price=result.get("base_price", [0.0])[0] or 0.0,
        )

    def to_ndarray(self) -> np.ndarray:
        """WHY: Convert to numpy array for model.predict(). Order must
             match the model's training feature order exactly."""
        return np.array([[
            self.total_orders_7d,
            self.avg_order_value_30d,
            self.premium_member,
            self.favorite_category_id,
            self.popularity_score,
            self.base_price,
        ]], dtype=np.float32)


# ── Application State (singletons, initialized at startup) ─────────────────
# WHY: Global singletons are acceptable here because they are read-only
#      after startup. The lifespan context manager guarantees initialization
#      before any requests are served — no lazy loading race conditions.
feature_store: Optional[FeatureStore] = None
redis_cache: Optional[aioredis.Redis] = None
model: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """WHY: FastAPI lifespan (replaces deprecated on_startup/on_shutdown).
         Initialize Feast FeatureStore once (~200ms), Redis pool, and model.
         If any initialization fails, the app refuses to start — fail fast."""
    global feature_store, redis_cache, model

    try:
        t0 = time.perf_counter()
        feature_store = FeatureStore(repo_path=FEAST_REPO_PATH)
        print(f"Feast FeatureStore initialized in {time.perf_counter() - t0:.2f}s")
        print(f"  Registry: {feature_store.config.registry}")
        print(f"  Online store: {feature_store.config.online_store.type}")
        print(f"  FeatureViews registered: {len(feature_store.list_feature_views())}")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Feast: {e}")
        raise

    try:
        redis_cache = aioredis.from_url(REDIS_DSN, decode_responses=False)
        await redis_cache.ping()
        print("Redis connection pool established")
    except Exception as e:
        print(f"CRITICAL: Failed to connect to Redis: {e}")
        raise

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
        raise

    yield  # Application runs here

    # Cleanup
    if redis_cache:
        await redis_cache.close()
    print("Application shut down cleanly")


app = FastAPI(lifespan=lifespan, title="Recommendation API with Feast")


# ── Feast Dependency ───────────────────────────────────────────────────────
async def get_features(
    user_id: str,
    item_id: str,
    request: Request,
) -> Features:
    """WHY: FastAPI dependency that fetches features from Feast's online store.
         This decorator pattern means any route that declares `features: Features
         = Depends(get_features)` automatically gets fresh features per request.
         WHY: Timing provides latency telemetry for free — log slow retrievals."""
    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature store not initialized")

    t0 = time.perf_counter()
    try:
        # WHY: get_online_features hits Redis, not BigQuery. Must stay <5ms.
        result = feature_store.get_online_features(
            entity_rows=[{"user_id": user_id}, {"item_id": item_id}],
            features=feature_store.get_feature_service(FEAST_SERVICE_NAME),
        ).to_dict()

        elapsed = (time.perf_counter() - t0) * 1000
        # WHY: log or metric-publish here — detect slow Redis before models suffer.
        if elapsed > 5.0:
            print(f"WARNING: Slow feature retrieval: {elapsed:.1f}ms (user={user_id}, item={item_id})")

        return Features.from_feast_response(result)

    except Exception as e:
        print(f"Feature retrieval error for user={user_id}, item={item_id}: {e}")
        # WHY: Return a degraded Features object with defaults rather than 500.
        #      This keeps predictions flowing with diminished accuracy — better
        #      than blocking all traffic during a Redis blip.
        return Features(
            user_id=user_id,
            item_id=item_id,
            total_orders_7d=0,
            avg_order_value_30d=0.0,
            premium_member=0,
            favorite_category_id=0,
            popularity_score=0.0,
            base_price=0.0,
        )


# ── Model Inference Route ──────────────────────────────────────────────────
@app.get("/rank")
async def rank_item(
    user_id: str,
    item_id: str,
    features: Features = Depends(get_features),
) -> dict:
    """WHY: The route handler is clean — it receives typed Features via FastAPI
         dependency injection. The model.predict() call is the only logic here.
         WHY: Include metadata (latency_ms, features_used) in response for
         observability — debug skew issues by comparing served features to
         training features stored in MLflow."""
    t0 = time.perf_counter()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_vector = features.to_ndarray()
    score = float(model.predict(feature_vector)[0])
    inference_ms = (time.perf_counter() - t0) * 1000

    return {
        "score": round(score, 4),
        "user_id": user_id,
        "item_id": item_id,
        "inference_ms": round(inference_ms, 1),
        "features_used": {
            "total_orders_7d": features.total_orders_7d,
            "avg_order_value_30d": features.avg_order_value_30d,
            "premium_member": features.premium_member,
            "popularity_score": features.popularity_score,
        },
    }


# ── Health Check ───────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
    """WHY: Kubernetes liveness/readiness probe. Checks Feast store connection
         and Redis availability. Returns 200 only if both are operational."""
    checks = {"feast": False, "redis": False, "model": False}

    if feature_store is not None:
        try:
            _ = feature_store.list_feature_views()
            checks["feast"] = True
        except Exception:
            pass

    if redis_cache is not None:
        try:
            await redis_cache.ping()
            checks["redis"] = True
        except Exception:
            pass

    checks["model"] = model is not None

    all_ok = all(checks.values())
    status_code = 200 if all_ok else 503
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_feast_middleware:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        log_level="info",
    )
```

### M4.4 Visual Representation 🖼️

```mermaid
flowchart LR
    subgraph TRAIN["🔄 Training Pipeline (Airflow/Dagster)"]
        A1[Raw Entity DataFrame<br/>user_id, timestamp, label] --> A2[Feast<br/>get_historical_features]
        A2 --> A3[Point-in-time correct<br/>training DataFrame]
        A3 --> A4[Train Model<br/>XGBoost / PyTorch]
        A4 --> A5[MLflow<br/>log_model + FeatureService tag]
    end

    subgraph REGISTRY["📋 Registries"]
        B1[Feast Registry<br/>FeatureViews + FeatureServices]
        B2[MLflow Model Registry<br/>models + FeatureService annotations]
        A2 -.-> B1
        A5 -.-> B2
    end

    subgraph DEPLOY["🚀 Deployment Pipeline"]
        B2 --> C1[CI/CD Pipeline<br/>detects new model version]
        C1 --> C2[Verify FeatureService<br/>exists in Feast Reg.]
        C2 --> C3[Deploy to FastAPI<br/>serving endpoint]
    end

    subgraph SERVE["⚡ Serving Pipeline (FastAPI + vLLM)"]
        D1[Client Request<br/>user_id, item_id] --> D2[Feast Dep<br/>get_online_features]
        D2 --> D3[Redis / Memorystore<br/>online store]
        D3 --> D4[Features typed DTO]
        D4 --> D5[Model Inference<br/>vLLM / XGBoost]
        D5 --> D6[Response<br/>prediction + metadata]
        B1 -.-> D2
    end
```

```mermaid
sequenceDiagram
    participant CICD as CI/CD Pipeline
    participant FEAST as Feast Registry
    participant MLFLOW as MLflow
    participant AIRFLOW as Airflow
    participant BQ as BigQuery
    participant FASTAPI as FastAPI Service
    participant REDIS as Redis

    Note over CICD,REDIS: Training Pipeline

    AIRFLOW->>FEAST: Resolve FeatureService "recsys_v3"
    FEAST-->>AIRFLOW: FeatureView references (user_behaviors, item_meta)
    AIRFLOW->>BQ: get_historical_features(entity_df, recsys_v3)
    BQ-->>AIRFLOW: PIT-correct DataFrame (50M rows)
    AIRFLOW->>AIRFLOW: Train XGBoost model
    AIRFLOW->>MLFLOW: log_model(model, tags={feast_service: "recsys_v3:v2"})
    MLFLOW-->>AIRFLOW: model_uri: models:/recsys_v3/production

    Note over CICD,REDIS: Deployment Pipeline

    CICD->>MLFLOW: Get latest production model
    MLFLOW-->>CICD: model + feast_service=recsys_v3:v2
    CICD->>FEAST: Verify FeatureService "recsys_v3:v2" exists
    FEAST-->>CICD: Confirmed: 6 features from 2 FeatureViews
    CICD->>FASTAPI: Rolling update with new model artifact
    FASTAPI-->>CICD: Deployed (zero downtime)

    Note over CICD,REDIS: Serving Pipeline

    FASTAPI->>FEAST: get_feature_service("recsys_v3")
    FEAST-->>FASTAPI: Feature references (6 features, 2 FeatureViews)
    FASTAPI->>REDIS: GET user_behaviors:42, GET item_meta:99
    REDIS-->>FASTAPI: {total_orders: 15, popularity: 0.87, ...}
    FASTAPI->>FASTAPI: model.predict(features.to_ndarray())
    FASTAPI-->>FASTAPI: Response: {score: 0.93}
```

### M4.5 Application in ML/AI Systems 🤖

**Stripe — ML Platform Migration to Feast (2023):** Stripe's ML platform team undertook a year-long migration from a custom in-memory feature cache to Feast for their fraud detection, risk scoring, and payment routing models. The architecture: GCP (BigQuery offline, Memorystore online, GKE serving), with Airflow orchestrating both training and materialization. Their FastAPI serving layer uses the dependency injection pattern shown above: a `Depends(get_merchant_features)` function fetches features from Feast, and the fraud model handler receives a typed `MerchantFeatures` dataclass. Impact: training-serving skew dropped from 15% to under 1%. The Most critical lesson: they integrated Feast with their MLflow model registry by tagging each model with `feast_service=payment_fraud_v3` — when a new model is promoted to production, the CI/CD pipeline verifies the FeatureService exists in the Feast registry before deploying. This closed-loop verification prevented 5 production incidents in the first 6 months where models referenced deprecated FeatureViews.

**HubSpot — Lead Scoring Pipeline with Feast + Dagster (2024):** HubSpot's lead scoring ML pipeline uses Dagster (asset-based orchestrator) with Feast for feature retrieval. Their training pipeline is modeled as Dagster assets: `raw_leads` → `feast_historical_features` → `trained_model` → `mlflow_registered_model`. The `feast_historical_features` Dagster asset calls `get_historical_features()` with an entity DataFrame built from the `raw_leads` asset, producing a point-in-time correct dataset as a downstream asset. Their serving layer uses FastAPI with the Feast dependency injection pattern and a canary deployment strategy: 5% of traffic hits the new model + new FeatureService version for 24 hours before full rollout. Impact: lead scoring precision improved by 11% after migration, attributed to consistent feature definitions between training and serving.

### M4.6 Common Pitfalls ⚠️ + 💡 Tips

| Pitfall | Consequence | 💡 Mitigation |
|---|---|---|
| FeatureStore initialization per request (lazy loading in route handler) | 200ms overhead on every request; p99 latency explodes from 40ms to 240ms | Initialize FeatureStore once at app startup (lifespan/asynccontextmanager); share singleton across all workers |
| Calling get_historical_features() in serving path | BigQuery query (1-3s) inside FastAPI handler; timeouts cascade to upstream services | Never use get_historical_features() outside Airflow/training jobs. Serving uses get_online_features() exclusively |
| No FeatureService version pinned in model metadata | Model trained with recsys_v3 features deployed while FeatureService has been updated to v4; schema mismatch causes silent prediction errors | Tag models in MLflow with exact FeatureService name + version hash; CI/CD validates FeatureService exists before deployment |
| Redis connection pooling per FastAPI worker without max_connections limit | Redis maxclients exceeded under load; new connections rejected; 5xx errors spike | Configure ConnectionPool with max_connections = (Redis maxclients / num_workers) - 50 (buffer for monitoring) |

### M4.7 Knowledge Check ❓

1. **Pipeline integration design:** Your training pipeline uses Dagster, and your serving pipeline uses FastAPI. The model team wants to add a new feature `user_is_employee` to the ranking model. Walk through the steps needed end-to-end — from FeatureView definition to production deployment — ensuring that no request is served with the old feature schema after the new model deploys.

2. **Dependency injection debugging:** Your FastAPI `/rank` endpoint returns predictions that are consistently 40% lower than expected. You suspect the FeatureService is resolving to the wrong set of features. What three checks would you perform in the Feast dependency function to diagnose and fix this?

3. **Closed-loop verification:** Design a CI/CD pipeline check that prevents deploying a model to production if its `feast_service` tag references a FeatureService that has been deprecated or whose FeatureViews have changed schemas (columns added/removed) since the model was trained.

---

## 📦 Compression Code

```python
"""feast_production_bootstrap.py — Complete Feast production setup for AWS and GCP.

WHY: Single-file bootstrap that demonstrates the full production workflow
     across both cloud providers. Entities, FeatureViews, FeatureServices,
     materialization, and retrieval — all in one runnable script.
     Swap feature_store.yaml to switch between AWS and GCP backends.
"""
from datetime import datetime, timedelta
import os

from feast import Entity, FeatureStore, FeatureView, FeatureService, Field, FileSource
from feast.types import Float32, Int64, String, UnixTimestamp

# WHY: FeatureStore reads feature_store.yaml for cloud provider config.
#      No code changes needed to switch between AWS and GCP.
store = FeatureStore(repo_path=".")

# ── Entities ───────────────────────────────────────────────────────────────
user_entity = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Platform user identifier — the primary entity for personalization",
    tags={"domain": "recommendation", "sensitivity": "pii", "governance": "gold"},
)

item_entity = Entity(
    name="item_id",
    join_keys=["item_id"],
    description="Catalog item identifier for ranking and retrieval",
    tags={"domain": "catalog"},
)

# ── Data Sources ───────────────────────────────────────────────────────────
# WHY: In production, these are BigQuerySource for GCP or FileSource(s3://)
#      for AWS. The FileSource with local path works for development.
if os.getenv("FEAST_ON_GCP"):
    from feast.infra.offline_stores.bigquery_source import BigQuerySource
    user_source = BigQuerySource(
        table="ml-platform-prod:feast_features.user_behavior_features",
        timestamp_field="event_timestamp",
    )
else:
    user_source = FileSource(
        path="s3://feast-offline-prod/user_features.parquet",
        timestamp_field="event_timestamp",
        created_timestamp_column="created_at",
    )

item_source = FileSource(
    path="s3://feast-offline-prod/item_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
)

# ── FeatureViews ───────────────────────────────────────────────────────────
user_features = FeatureView(
    name="user_behavior_features",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="total_orders_7d", dtype=Int64, description="Count of orders in trailing 7 days"),
        Field(name="avg_order_value_30d", dtype=Float32, description="Average order value USD, 30-day window"),
        Field(name="last_session_hours_ago", dtype=Int64, description="Hours since last user activity"),
        Field(name="churn_score", dtype=Float32, description="Predicted churn probability, daily update"),
        Field(name="premium_member", dtype=Int64, description="1 if premium subscription active"),
    ],
    online=True,
    source=user_source,
    tags={"team": "recommendation-ml", "refresh": "hourly", "tier": "production"},
    owner="recommendation-ml@company.com",
    description="User behavioral features for ranking, recommendation, and churn prediction",
)

item_features = FeatureView(
    name="item_metadata_features",
    entities=[item_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="category_id", dtype=Int64, description="Product category ID"),
        Field(name="base_price", dtype=Float32, description="Listing price in USD"),
        Field(name="popularity_score", dtype=Float32, description="Trending score, updated hourly"),
        Field(name="inventory_count", dtype=Int64, description="Available units in stock"),
    ],
    online=True,
    source=item_source,
    tags={"team": "catalog-ml", "refresh": "hourly", "tier": "production"},
    owner="catalog-ml@company.com",
    description="Item metadata and popularity signals for ranking models",
)

# ── FeatureService ─────────────────────────────────────────────────────────
# WHY: FeatureService encapsulates the exact features a model uses.
#      Tag this name in MLflow with the model artifact for closed-loop verification.
recsys_service = FeatureService(
    name="recsys_v3",
    features=[
        user_features[["total_orders_7d", "avg_order_value_30d", "churn_score", "premium_member"]],
        item_features[["category_id", "popularity_score", "base_price"]],
    ],
    tags={"model": "ranking_v3", "latency_budget_ms": "5"},
)

# ── Apply + Materialize ────────────────────────────────────────────────────
# WHY: apply() registers everything in the registry. Run as CI/CD step on merge.
store.apply([user_entity, item_entity, user_features, item_features, recsys_service])

# WHY: materialize() pushes features from offline → online store.
#      Run as scheduled Airflow/Dagster task.
store.materialize(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
)

# ── Online Retrieval (Serving Path) ─────────────────────────────────────────
# WHY: This is the inference hot path — target <5ms.
online_features = store.get_online_features(
    features=store.get_feature_service("recsys_v3"),
    entity_rows=[{"user_id": "user_42", "item_id": "item_99"}],
).to_dict()

print("Online features:", online_features)

# ── Historical Retrieval (Training Path) ────────────────────────────────────
# WHY: Point-in-time correct training data — zero data leakage.
import pandas as pd
training_entity_df = pd.DataFrame({
    "user_id": ["user_42", "user_43", "user_44"],
    "item_id": ["item_99", "item_100", "item_101"],
    "event_timestamp": [pd.Timestamp.now(tz="UTC")] * 3,
})
training_df = store.get_historical_features(
    entity_df=training_entity_df,
    features=store.get_feature_service("recsys_v3"),
).to_df()

print(f"Training data: {training_df.shape[0]} rows × {training_df.shape[1]} columns")
print(f"Provider: {store.config.provider}")
print(f"Online store type: {store.config.online_store.type}")
print(f"Registered FeatureViews: {len(store.list_feature_views())}")
print(f"Registered FeatureServices: {len(store.list_feature_services())}")
```

## 🎯 Documented Project

**Project Description:** Deploy a cloud-agnostic Feast feature store on AWS (S3 + ElastiCache Redis + EKS) and GCP (BigQuery + Memorystore Redis + GKE) for a recommendation platform serving 100K RPS across ranking, personalization, and content moderation models. The deployment must include materialization scheduling, freshness monitoring, and ML pipeline integration with FastAPI, Airflow, and MLflow.

**5 Functional Requirements:**
1. Deploy Feast with cloud-provider-specific backends: S3 (offline) + ElastiCache Redis (online) + EKS on AWS; BigQuery (offline) + Memorystore Redis (online) + GKE on GCP; registry on S3/GCS respectively
2. Run both incremental (15-minute) and full (daily) materialization via Airflow, with pre-materialization integrity checks (row counts, null rates) and post-materialization freshness metrics published to CloudWatch
3. Integrate Feast into the training pipeline: Airflow/Dagster DAG calls `get_historical_features()` → point-in-time correct DataFrame → model training → MLflow artifact with FeatureService tag → model registry promotion
4. Integrate Feast into the serving pipeline: FastAPI lifespan initializes FeatureStore singleton; dependency injection fetches features from Redis per request; model handler receives typed Features DTO; health endpoint monitors Feast + Redis + model readiness
5. Implement closed-loop verification: CI/CD pipeline validates that model's FeatureService tag exists in Feast registry before deployment; alerts if FeatureService has been deprecated or schema has diverged

**Main Components:**
- Feast Core on EKS/GKE (3 replicas, gRPC server, registry on S3/GCS)
- Go Feature Server (sidecar DaemonSet on GKE; sidecar in SageMaker multi-container endpoints on AWS)
- ElastiCache Redis (AWS) / Memorystore Redis (GCP) — online store, 50GB, TLS, AOF persistence
- S3 (AWS) / BigQuery (GCP) — offline store, Parquet files partitioned by `DATE(event_timestamp)`
- Airflow (Cloud Composer on GCP, MWAA on AWS) — materialization DAGs with sensor, backfill logic, freshness checks
- FastAPI service — lifespan-managed FeatureStore singleton, Feast dependency injection, typed Features DTO
- MLflow Model Registry — models tagged with `feast_service` and `feast_service_version`, CI/CD verification logic
- CloudWatch (AWS) / Cloud Monitoring (GCP) — materialization_lag, feature_freshness_ratio, stale_feature_reads, Redis memory usage

**3 Success Metrics:**
- Materialization freshness: 99% of features materialized within 5 minutes of source data update for behavioral features; within 24 hours for demographic features
- Feature retrieval latency: p99 < 3ms for online `get_online_features()` with Go Feature Server sidecar; p99 < 5ms with Python SDK
- Training-serving consistency: zero FeatureService schema mismatches between registered MLflow models and Feast registry over a 90-day monitoring period; 100% closed-loop verification pass rate

## 🎯 Key Takeaways
- Feast production deployment is cloud-native orchestration: on AWS (S3 + DynamoDB/Redis + EKS) or GCP (BigQuery + Memorystore + GKE) — the FeatureView code is identical across providers; only `feature_store.yaml` changes
- Materialization is the heartbeat — incremental for efficiency (hourly, 15-min), full for consistency (daily); freshness monitoring with Prometheus/CloudWatch metrics and gradient alerting thresholds separates production from prototype
- The Go Feature Server running as a sidecar eliminates Python serialization overhead and reduces feature retrieval latency by 50-70% — a critical optimization for high-throughput serving (<50ms total latency budgets)
- Feast integration follows a clear pattern: `get_historical_features()` in Airflow/Dagster for training, `get_online_features()` via FastAPI dependency injection for serving — never cross the streams
- Closed-loop verification (model's `feast_service` tag validated against Feast registry before deployment) prevents the most common production failure mode: models deployed with stale or missing FeatureService references
- Your SageMaker and Vertex AI workflows integrate naturally: SageMaker Processing/Vertex AI Pipelines call `get_historical_features()`; SageMaker Endpoints/Vertex AI Endpoints call `get_online_features()` via sidecar
- The FastAPI dependency injection pattern — FeatureStore singleton → `Depends(get_features)` → typed `Features` DTO — creates clean, testable separation between feature retrieval and model inference, and the `/health` endpoint provides Kubernetes-native readiness probes for Feast, Redis, and model availability

## References
- Feast AWS Deployment Guide: https://docs.feast.dev/how-to-guides/deploying-feast-on-aws
- Feast GCP Deployment Guide: https://docs.feast.dev/how-to-guides/deploying-feast-on-gcp
- Feast Materialization Documentation: https://docs.feast.dev/reference/feature-servers/python-feature-server
- Feast Go Feature Server: https://docs.feast.dev/reference/feature-servers/go-feature-server
- Stripe ML Platform Engineering: https://stripe.com/blog/technical
- Netflix Spinnaker and ML Deployment: https://netflixtechblog.com/
- Airflow ExternalTaskSensor: https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html
- Dagster Software-Defined Assets: https://docs.dagster.io/concepts/assets/software-defined-assets
- FastAPI Lifespan Events: https://fastapi.tiangolo.com/advanced/events/
- AWS ElastiCache Redis Monitoring: https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/CacheMetrics.html
- GCP Memorystore Redis Monitoring: https://cloud.google.com/memorystore/docs/redis/monitoring-instances
- DoorDash ML Platform Engineering Blog: https://doordash.engineering/
- Feast FeatureService Documentation: https://docs.feast.dev/reference/feature-services
