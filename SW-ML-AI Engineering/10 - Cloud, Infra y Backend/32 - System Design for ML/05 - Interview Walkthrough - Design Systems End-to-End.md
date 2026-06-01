# 🎯 05 - Interview Walkthrough — Design Systems End-to-End

## 🎯 Learning Objectives

- Execute two complete FAANG-style ML system design interviews from requirements through back-of-envelope, API design, data model, architecture, scaling, and tradeoffs
- Walk through a YouTube-style video recommendation system serving 500M DAU with <200 ms latency — demonstrating two-stage candidate generation (CPU) + ranking (GPU) to reduce GPU needs 200×
- Walk through a real-time fraud detection system handling 10K tx/s with <50 ms SLA and strong consistency (CP) — demonstrating latency budgets and graceful degradation
- Demonstrate the interview rhythm: requirements → scale → API → data model → architecture → scaling → tradeoffs — the same pattern FAANG interviewers expect

## Introduction

The system design interview is the capstone of the FAANG+ ML engineer hiring process. You have 45 minutes to design a production ML system from scratch, clarify requirements, estimate scale, sketch an architecture, and defend every tradeoff. This note provides two complete walkthroughs that demonstrate the end-to-end process across the previous four notes: CAP classification ([[01 - CAP Theorem and Consistency Models in ML Workloads|Note 01]]), caching ([[02 - Caching, CDNs and Storage Architectures for ML|Note 02]]), load balancing and sharding ([[03 - Load Balancing, Sharding and Scaling ML Systems|Note 03]]), and back-of-envelope estimation ([[04 - Back-of-Envelope Estimation - Throughput, Latency, Storage and Cost|Note 04]]).

Each walkthrough follows the interview rhythm:

1. **Requirements** — Clarify functional and non-functional: scale (DAU, RPS), latency SLA, consistency needs, availability targets
2. **Back-of-Envelope** — Derive storage, throughput, GPU count, cost from first principles
3. **API Design** — Define the service interface: endpoints, request/response schemas, protocol (REST/gRPC/streaming)
4. **Data Model** — Feature store schema, database choices, embedding storage, training data organization
5. **Architecture** — Whiteboard with components, data flow, separation of concerns
6. **Scaling Plan** — Sharding strategy, load balancing algorithm, auto-scaling metrics, CDN usage
7. **Tradeoffs** — What degrades gracefully, what breaks, what you would change with more time

---

## Walkthrough 1: YouTube-Style Video Recommendation

### 1. Requirements

*Interviewer*: "Design a video recommendation system like YouTube's homepage. 500 million daily active users. Recommendations should be personalized and appear within 200 ms of page load."

**Clarify**:
- Functional: Personalized feed based on watch history, search history, and engagement patterns. Each user sees ~50 videos per session, ~3 sessions/day.
- Non-functional: P99 < 200 ms end-to-end. 99.95% availability (AP — stale recommendations > blank screen). Freshness: new uploads within 10 min, user behavior within 1 min. Global users, multi-region deployment.

### 2. Back-of-Envelope

75B recommendation calls/day:

$$RPS_{\text{avg}} = \frac{75 \times 10^9}{86400} \approx 868K, \quad RPS_{\text{peak}} \approx 1.7M$$

**Storage**: User embeddings (500M × 256-dim float32 = 512 GB), video embeddings (1B × 256-dim = 1 TB), user features (500M × 200 feats × 128 B = 12.8 TB), video metadata (1B × 500 B = 500 GB), training data (75B impressions/day × 1 KB = 75 TB/day → petabyte-scale data lake).

**Two-stage architecture justification**: Naive = 1.7M RPS × one GPU inference per request = 1.7M GPUs. Instead: (1) Candidate generation on CPU — ANN over 1B videos → 1000 candidates per request (~10 ms). (2) Ranking on GPU — scores 1000 candidates in one batch forward pass (50 ms). The two-stage design reduces GPU load by 200×.

**GPU estimation**: 1.7M RPS × 50 ms ranking = 85,000 GPU-seconds/s → 85,000 GPUs. With prediction cache (60% hit rate, [[02|Note 02]]), only 680K RPS hit ranking → 34,000 GPU-seconds/s. With model distillation (10× faster) + continuous batching (10×): 340 GPUs. Candidate generation on CPU: 1.7M RPS × 10 ms = 17,000 CPU-seconds/s → ~266 × 64-core instances.

### 3. API Design

```python
GET /api/v1/recommendations?user_id={id}&count={n}&page_token={token}
Response: {"videos": [{"video_id", "title", "channel_name", "score": 0.942,
           "reason": "Based on your interest in Distributed Systems"}],
           "next_page_token": "..."}

# Internal gRPC
service RecommendationService {
  rpc GetRecommendations(RecommendationRequest) returns (RecommendationResponse);
  rpc RecordImpression(ImpressionEvent) returns (google.protobuf.Empty);
}
```

### 4. Data Model

**Feature store** (Redis, sharded by user_id): `feature:user:{user_id}:{feature_name}` with TTL per freshness tier. User embedding (float32[256], 24h TTL), watch_history_7d (string[], 1h TTL), subscribed_channels (string[], 24h), language (string, 24h), region (string, 24h), session_active (bool, 60s TTL).

**Video index**: FAISS sharded across 32 nodes by video_id hash. **Candidate store**: `candidate:user:{user_id}:{session_id}` → set of 1000 video IDs (TTL 5 min). **Training data**: S3 partitioned by date in Parquet: `user_id, video_id, impression_time, clicked, watched_duration, position_in_feed`.

### 5. Architecture

Key data flow: User app → CloudFront CDN (thumbnails, static) → Global LB (geo-routed) → API Gateway (FastAPI, async) → Prediction Cache (Redis, 60% hit rate) → FAISS ANN (on miss, 32 shards, parallel search + merge) → Feature Store (Redis, consistent hashing by user_id) → GPU Ranking (distilled model, batch 1000 candidates, 50 ms) → Filter + Diversity (remove watched, diversify channels, enforce freshness) → Re-rank + Page (business rules, final 50) → Response.

**Offline pipeline**: Kafka ingests impression events → Spark Streaming computes user/video features → GPU training cluster retrains ranking model weekly → nightly candidate gen model updates → model artifacts → S3 → CDN → serving replicas.

Components: CloudFront CDN, Envoy LB, FastAPI servers, Redis prediction cache + feature store, 32 FAISS shards, GPU ranking cluster (A100s with custom-metric HPA), Kafka, Spark Streaming, GPU training cluster, DynamoDB model registry.

### 6. Scaling Plan

- **Sharding**: User features by `user_id` (consistent hashing, 64 Redis nodes). Video embeddings by `video_id` hash (32 FAISS shards, parallel query + merge coordinator). Prediction cache on same Redis ring.
- **Load balancing**: API layer — round-robin (stateless). Candidate gen — consistent hashing by `user_id` to FAISS shard. Ranking — GPU-aware routing ([[03|Note 03]]) by VRAM and model type.
- **Auto-scaling**: API servers — CPU-based HPA (CPU-bound). Ranking — custom-metric HPA on GPU utilization + queue depth + P99 latency ([[03|Note 03]]). CDN-distributed model artifacts minimize cold start from 120 s to ~15 s.
- **Multi-region**: Full stack in us-east, eu-west, ap-southeast. Geo-routed. Video catalog replicated (eventual consistency — AP). User data pinned to home region for GDPR.

### 7. Tradeoffs

**CAP profile**: AP. Stale features → slightly worse recommendations. Unavailability → blank screen → user closes app.

**Graceful degradation**: Feature store down → cached features (5-min TTL). Ranking down → candidate gen scores only (no re-rank but still relevant). Candidate gen down → trending/popular fallback (non-personalized). Model registry fails → keep last known good version in local cache.

**Breaks**: Model registry inconsistency → different users see different rankings → A/B test contamination (fix: DynamoDB strong reads). Feature shard failure → subset of users get empty recommendations (fix: replica sets, each shard has 3 replicas).

**With more time**: Multi-armed bandit for exploration (5% random traffic to discover new content), real-time Flink streaming instead of batch Spark for fresher features, two-tower model architecture (user tower + item tower) for end-to-end differentiable training, A/B test infrastructure for online experimentation.

---

## Walkthrough 2: Real-Time Fraud Detection

### 1. Requirements

*Interviewer*: "Design a real-time fraud detection system for a payment processor. 10,000 transactions per second. Must return a fraud decision within 50 ms. 99.9% availability. Fraud missed = financial loss."

**Clarify**: Credit card transaction fraud. Each payment: ACCEPT/REJECT/REVIEW with risk score. 10K TPS (peak 15K). SLA: <50 ms P99. Availability: 99.9% — CP during partition (reject txns rather than risk approving fraud). Strong consistency for features (stale feature = missed fraud). Recall > 95%, precision > 30%.

### 2. Back-of-Envelope

10K TPS × 50 ms = 500 concurrent transactions.

**Latency budget**: Network ingress (2 ms) → DynamoDB feature fetch (10 ms, strong consistent read) → feature transform (3 ms) → model inference (20 ms, gradient-boosted trees on CPU) → rule engine (5 ms) → network egress (2 ms) = **42 ms total**, 8 ms slack. This budget is tight but feasible — requires same-AZ deployment and a lightweight model.

**GPU estimation**: Gradient-boosted trees run in <5 ms on CPU. At 10K TPS × 20 ms = 200,000 ms/s = 200 seconds of compute per second → 200 CPU cores or ~4 GPU equivalents (if using a small NN).

**Storage**: Feature store (500M users × 50 features × 128 B = 3.2 TB). Transaction log: 10K TPS × 1 KB = 10 MB/s → 864 GB/day → 26 TB/month in S3.

### 3. CAP Analysis — CP at Every Layer

During a network partition, reject transactions rather than risk approving fraudulent ones. The cost asymmetry:

$$\text{Cost}(CP) = P(\text{partition}) \times \text{lost\_revenue\_per\_rejected\_txn}$$
$$\text{Cost}(AP) = P(\text{partition}) \times P(\text{fraud}) \times \text{chargeback\_cost}$$

With chargebacks at $50-500/txn and fraud at 0.1-1%, AP is 100-1000× more expensive during partitions.

| Component | CAP Mode | Storage |
|-----------|----------|---------|
| Online feature store | CP | DynamoDB (strong reads) |
| Model registry | CP | DynamoDB (strong reads) |
| Transaction log | CP | Kafka + S3 (immutable) |
| Training data | CP | S3 (snapshot isolation) |
| Rule engine config | CP | DynamoDB |
| A/B experiment config | CP | DynamoDB |
| Monitoring dashboards | AP | Redis (stale acceptable) |

### 4. API & Data Model

```python
POST /api/v1/fraud/check
Request: {transaction_id, merchant_id, user_id, amount_cents, currency,
          card_last4, card_bin, ip_address, device_fingerprint,
          billing_address, shipping_address, timestamp_ms}
Response: {decision: "ACCEPT|REJECT|REVIEW", risk_score: 0.0-1.0,
           decision_time_ms: 28, model_version: "fraud_v4.2.1",
           rules_triggered: ["velocity_check_ok"], review_reason: null}
```

DynamoDB table `fraud_features` (PK: user_id, GSIs on device_fingerprint, card_bin). Attributes: purchase_velocity_1h/24h, avg_amount_7d, distinct_merchants_24h, distinct_cards_7d, device_risk_score, device_seen_before, card_velocity_1h, ip_geolocation_risk, ip_is_proxy, billing_shipping_match. Read consistency: `ConsistentRead=True`.

Transaction log: Kafka topic `fraud.transactions` (64 partitions, keyed by user_id). Avro schema with transaction_id, user_id, amount, decision, risk_score, model_version, features_snapshot. 7-day Kafka retention, permanent S3.

### 5. Architecture

Flow: Payment Processor → gRPC (50 ms timeout, auto-REJECT on timeout) → Layer 7 LB (gRPC health checks, circuit breaker) → Fraud API Server → Feature Engineering (transforms raw input: 3 ms) → DynamoDB strong read (10 ms, same-AZ) → Gradient-Boosted Trees (in-process, 20 ms, ~500 trees) → Rule Engine (deterministic business rules: amount > $10K + risk > 0.1 → REVIEW, blocklist → REJECT, sanctioned IP → REJECT, 3+ distinct cards in 1h → REVIEW) → Decision → Response.

Async: Transaction logged to Kafka → Flink streaming computes velocity features (1h sliding window, updates every 10 s; 24h tumbling window, every 5 min) → writes to DynamoDB → daily model retraining pipeline.

**Feature freshness**: Writing to online store (DynamoDB) first, then offline store (S3 for training), guarantees $V_{\text{offline}} \leq V_{\text{online}}$ — serving features are always at least as fresh as training features. This prevents the write-order trick problem ([[01|Note 01]]).

**Never block transactions**: Always process with a weaker check rather than block entirely. False positives lose revenue; false negatives lose money; system downtime loses all revenue.

### 6. Scaling Plan

DynamoDB auto-shards by primary key. Kafka partitioned by `user_id` (64 partitions) for in-order Flink processing. Inference servers are stateless — any server handles any request.

Load balancing: Layer 7 with gRPC health checks (`/healthz` returns 200 if model loaded, FS reachable, latency < 40 ms). Circuit breaker: > 50% timeouts in 30 s → mark unhealthy.

Auto-scaling on P99 latency and TPS (not CPU — CPU is always low for fast inference). Min 10, max 50 replicas. Scale-up trigger: P99 > 30 ms for 30 s.

Multi-region: 3 regions for HA. Each region has its own DynamoDB table (global tables are eventually consistent). User data pinned to home region with cross-region proxying for strong consistency.

### 7. Tradeoffs

**Model complexity tradeoff**: Gradient-boosted trees (5-10 ms CPU, explainable, no GPU needed) — best for 20 ms budget. Small NN (10-15 ms GPU, better on feature interactions, less explainable). Large transformer (50-200 ms GPU, best for sequence features) — only viable for async post-transaction review.

**Graceful degradation**: Model down → rules-only (catches 70% of fraud). Feature store slow → cached features (60 s TTL, catches 85%). Kafka down → buffered logs (5 min buffer, no data loss). Rule engine down → model-only (95% of fraud, misses deterministic patterns).

**Breaks**: Feature store corruption → wrong user's features → wrong decisions (mitigated by checksums and user_id verification). Model registry inconsistency → systematically wrong risk scores (mitigated by DynamoDB strong reads). Full region down → cross-region failover (latency 50→150 ms, but continues).

¡Sorpresa! The most impactful design decision in fraud systems is not the model architecture — it's feature store consistency and grace of degradation. A perfect model with stale features is worse than a simple model with fresh features. Senior fraud designers spend 80% of time on feature freshness and degradation paths, and 20% on the model itself.

---

## 🎯 Key Takeaways

- Always clarify requirements before designing — scale, SLA, consistency, and availability fundamentally shape the architecture; never skip this step
- The two-stage pattern (candidate generation on cheap CPU infra, ranking on expensive GPU) reduces compute needs by 200×+ in recsys
- Latency budgeting component-by-component catches silent degradation before it affects users — allocate, measure, and alert on every component
- Fraud detection systems are CP at every layer — the cost asymmetry between rejecting legitimate txns and approving fraudulent ones is 100-1000×
- Graceful degradation paths (model → rules → cached features) are more important than model accuracy for production reliability
- The 7-step interview script is a forcing function for structured thinking — practice it until it becomes automatic in 45-minute sessions
- Cross-reference architectural decisions with earlier notes: CAP (Note 01), caching (Note 02), load balancing (Note 03), estimation (Note 04)

## References

- Huyen, C. (2022). *Designing Machine Learning Systems*. Chapters 7-9.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. Chapters 5, 6, 9.
- YouTube Engineering. (2016). "Deep Neural Networks for YouTube Recommendations." *ACM RecSys*.
- Stripe Engineering. (2023). "Radar: How Stripe's ML detects fraud."
- DoorDash Engineering. (2023). "Building DoorDash's ML Platform — Feature Store Design."
- [[01 - CAP Theorem and Consistency Models in ML Workloads|CAP Theorem]]
- [[02 - Caching, CDNs and Storage Architectures for ML|Caching for ML]]
- [[03 - Load Balancing, Sharding and Scaling ML Systems|Load Balancing]]
- [[04 - Back-of-Envelope Estimation - Throughput, Latency, Storage and Cost|Back-of-Envelope]]

## 📦 Código de compresión

```python
"""
interview_checklist.py — The 7-step ML system design interview checklist.
Use during mock interviews to stay on track. Covers both walkthroughs.
"""
from dataclasses import dataclass, field
from enum import Enum

class CapMode(Enum):
    CP = "Consistent during partition — reject requests"
    AP = "Available during partition — serve stale data"

@dataclass
class SystemDesign:
    """Template for 45-min ML system design interview. Fill this during the call.

    Time budget:
    - Requirements:    5 min
    - Back-of-envelope: 5 min
    - API Design:       5 min
    - Data Model:       5 min
    - Architecture:    10 min (diagram + walkthrough)
    - Scaling Plan:     8 min
    - Tradeoffs:        7 min
    """
    name: str
    cap_mode: CapMode
    latency_sla_ms: float
    peak_rps: float

    scale: str = ""
    functional_reqs: list[str] = field(default_factory=list)
    non_functional_reqs: list[str] = field(default_factory=list)

    storage_tb: float = 0.0
    gpu_count: int = 0
    cost_monthly: float = 0.0
    latency_budget: dict[str, float] = field(default_factory=dict)

    endpoints: list[str] = field(default_factory=list)
    protocol: str = "REST + gRPC"
    stores: list[str] = field(default_factory=list)
    key_features: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    data_flow: str = ""

    sharding: str = ""
    load_balancing: str = ""
    auto_scaling: str = ""
    regions: int = 1

    graceful_degradation: list[str] = field(default_factory=list)
    break_points: list[str] = field(default_factory=list)
    with_more_time: list[str] = field(default_factory=list)


def interview_script(sd: SystemDesign) -> str:
    parts = []
    parts.append(f"=== {sd.name} ===")
    parts.append(f"1. REQ: Scale={sd.scale}, SLA={sd.latency_sla_ms}ms, "
                 f"CAP={sd.cap_mode.value}")
    parts.append(f"2. EST: Storage={sd.storage_tb}TB, GPUs={sd.gpu_count}, "
                 f"Cost=${sd.cost_monthly:,.0f}/mo, Budget={sd.latency_budget}")
    parts.append(f"3. API: {sd.protocol}, {sd.endpoints}")
    parts.append(f"4. DATA: Stores={sd.stores}, Features={sd.key_features[:3]}")
    parts.append(f"5. ARCH: Components={len(sd.components)}, "
                 f"Flow={sd.data_flow}")
    parts.append(f"6. SCALE: Shard={sd.sharding}, LB={sd.load_balancing}, "
                 f"Auto={sd.auto_scaling}, Regions={sd.regions}")
    parts.append(f"7. TRADE: Degrade={sd.graceful_degradation[:2]}, "
                 f"Break={sd.break_points[:2]}, Future={sd.with_more_time[:2]}")
    return "\n".join(parts)


# YouTube RecSys
youtube = SystemDesign(
    name="YouTube-Style Video Recommendation", cap_mode=CapMode.AP,
    latency_sla_ms=200.0, peak_rps=1_736_000,
    scale="500M DAU, 75B recs/day, 1.7M peak RPS",
    functional_reqs=["Personalized feed", "50 videos per page",
                     "Real-time impression logging"],
    non_functional_reqs=["P99 < 200ms", "99.95% avail", "Multi-region",
                         "Freshness: 10min uploads, 1min behavior"],
    storage_tb=750.0, gpu_count=340, cost_monthly=850_000,
    latency_budget=dict(net=5, cache=2, cand_gen=30, feat_fetch=10,
                        ranking=50, filter=5, egress=5),
    endpoints=["GET /recommendations", "POST /impressions"],
    protocol="REST + gRPC internal",
    stores=["Redis (features + cache)", "FAISS (embeddings, 32 shards)",
            "S3 (training data)", "DynamoDB (model registry)"],
    key_features=["user_embedding", "watch_history_7d", "subscribed_channels"],
    components=["CDN", "API GW", "Prediction Cache", "FAISS ANN",
                "Feature Store", "GPU Ranking", "Filter+Diversity",
                "Kafka", "Spark Streaming", "Training Cluster"],
    data_flow="User->CDN->API->Cache->FAISS->Features->GPU->Filter->Response",
    sharding="user_id consistent hashing (Redis + FAISS)",
    load_balancing="CHash for ranking, round-robin for API",
    auto_scaling="GPU utilization + queue depth + P99 latency",
    regions=3,
    graceful_degradation=["Ranking down -> candidate scores only",
                          "FS down -> cached features (5min TTL)",
                          "Candidate gen down -> trending/popular"],
    break_points=["Registry inconsistency -> A/B contamination",
                  "Shard failure -> empty recs for subset"],
    with_more_time=["MAB exploration (5% traffic)",
                    "Real-time Flink streaming",
                    "Two-tower model for end-to-end training"],
)

# Fraud Detection
fraud = SystemDesign(
    name="Real-Time Fraud Detection", cap_mode=CapMode.CP,
    latency_sla_ms=50.0, peak_rps=15_000,
    scale="10K TPS (15K peak), <50ms per txn",
    functional_reqs=["ACCEPT/REJECT/REVIEW", "Risk score 0-1",
                     "Real-time velocity features"],
    non_functional_reqs=["P99 < 50ms", "99.9% CP during partition",
                         "Strong consistency for features"],
    storage_tb=3.2, gpu_count=0, cost_monthly=15_000,
    latency_budget=dict(net_in=2, feat_fetch=10, transform=3,
                        inference=20, rules=5, net_out=2),
    endpoints=["POST /fraud/check"],
    protocol="gRPC (sync, 50ms timeout, auto-REJECT on timeout)",
    stores=["DynamoDB (features, CP)", "Kafka (txn log, 64 partitions)",
            "S3 (training data)"],
    key_features=["purchase_velocity_1h", "device_risk_score",
                  "card_velocity_1h", "ip_geolocation_risk"],
    components=["Payment GW", "gRPC LB", "Fraud API", "Feature Engine",
                "DynamoDB", "GB Trees (CPU)", "Rule Engine",
                "Kafka", "Flink Streaming", "Training Pipeline"],
    data_flow="Payment->API->Transform->DynamoDB->Model->Rules->Decision",
    sharding="DynamoDB auto-shard by user_id",
    load_balancing="gRPC health checks + circuit breaker",
    auto_scaling="P99 latency > 30ms for 30s",
    regions=3,
    graceful_degradation=["Model slow -> rules-only",
                          "FS down -> cached features (60s TTL)",
                          "Kafka down -> buffered logs (5min)"],
    break_points=["FS corruption -> wrong decisions",
                  "Registry inconsistency -> wrong scores"],
    with_more_time=["GNN for transaction network",
                    "Sequence transformer for session fraud",
                    "Real-time model updates via online learning"],
)

if __name__ == "__main__":
    print(interview_script(youtube))
    print()
    print(interview_script(fraud))
```
