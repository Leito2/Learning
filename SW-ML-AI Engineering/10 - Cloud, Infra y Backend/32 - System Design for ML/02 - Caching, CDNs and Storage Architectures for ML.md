# 💾 02 - Caching, CDNs, and Storage Architectures for ML

## 🎯 Learning Objectives

- Design multi-tiered caching strategies: embedding caches, prediction caches, feature caches, and semantic caches with correct TTL and invalidation
- Architect model artifact distribution via CDN edge caching with versioned immutable URLs and measure latency-to-cost tradeoffs
- Apply tiered storage economics (Hot/Warm/Cold) to ML data — GPU VRAM, NVMe SSDs, S3 — and right-size each data class by access frequency
- Implement and benchmark a Redis semantic cache for LLM inference
- Evaluate KV cache reuse (SGLang's RadixAttention) as the highest-leverage caching mechanism in LLM serving

## Introduction

Caching is the single most cost-effective optimization in ML system design. A cache hit costs ~1ms (in-memory Redis) while a cache miss costs ~100ms (GPU inference) — two orders of magnitude difference. Yet most ML teams under-invest in caching because they treat predictions as inherently uncacheable ("every user is different, every query is unique"). This note demonstrates that well-designed caching can reduce ML infrastructure costs by 30-80% while improving latency, and provides concrete patterns for each layer of the ML stack.

The word *cache* enters English from French *cacher* (to hide), ultimately from Latin *coacticare* (to constrain, store up). In computing, a cache hides the latency of a slower storage layer behind a faster one. For ML systems, the cache hierarchy is deeper than for traditional software: GPU VRAM (fastest, most expensive) > system RAM > NVMe SSD > S3/CDN (slowest, cheapest). Each layer trades access latency for storage cost, and right-sizing data placement across these tiers is the central problem of ML storage architecture.

This note builds on caching patterns from [[../../31 - FastAPI for ML/00 - Welcome to FastAPI for ML|FastAPI]] and extends them to ML-specific cache types. The KV cache inside LLM inference engines is covered as a system-level pattern. Tiered storage connects to [[../../23 - Infrastructure as Code/00 - Welcome to Infrastructure as Code|IaC]].

![Memory hierarchy](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/ComputerMemoryHierarchy.svg/640px-ComputerMemoryHierarchy.svg.png)

## 1. Embedding and Semantic Caching

Embedding computation dominates RAG (Retrieval-Augmented Generation) and semantic search costs. A single query requires: tokenization → embedding model forward pass → vector normalization — typically 50-200ms on GPU and $0.0001-0.001 in API costs. If the same query (or a semantically similar one) arrives repeatedly, recomputing the embedding wastes GPU cycles and budget.

### Exact-Match LRU Cache

The simplest form. Hash the input text → check if embedding exists in cache → return cached or compute. Python's `@lru_cache` provides this for free:

$$\text{hit\_ratio} = \frac{\text{unique\_queries\_repeated}}{\text{total\_queries}}$$

Real production traffic shows 30-50% of queries are exact duplicates within a time window (users searching the same thing, bots, retries). An LRU cache of 10,000 entries captures these with <1ms latency and ~60MB memory for 768-dim float32 embeddings.

### Semantic Cache

More sophisticated. Cache {query_embedding → [document_ids]} instead of {query_text → embedding}. This allows approximate matching: a query semantically similar to a cached query (cosine similarity > 0.95) can reuse cached results without computing anything. Redis with vector similarity search (RediSearch) enables this at scale.

```python
import hashlib, json, time
import numpy as np
import redis

class SemanticCache:
    """L2 semantic cache using Redis vector similarity search.

    Cache key: query_embedding (768-dim float32)
    Cache value: [document_ids] returned by the RAG pipeline
    Similarity threshold: cosine > 0.95 (tunable)
    TTL: 1 hour by default
    """

    def __init__(self, threshold: float = 0.95, ttl: int = 3600):
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=False)
        self.threshold = threshold
        self.ttl = ttl
        self._ensure_index()

    def _ensure_index(self):
        try:
            self.redis.execute_command(
                "FT.CREATE", "idx:emb", "ON", "HASH",
                "SCHEMA", "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32", "DIM", "768", "DISTANCE_METRIC", "COSINE",
            )
        except redis.ResponseError:
            pass

    def search(self, query_emb: np.ndarray):
        query_vec = query_emb.astype(np.float32).tobytes()
        results = self.redis.execute_command(
            "FT.SEARCH", "idx:emb",
            "@embedding:[VECTOR_RANGE 0.05 $vec]",
            "PARAMS", "2", "vec", query_vec,
            "LIMIT", "0", "5", "SORTBY", "__embedding_score", "ASC",
            "RETURN", "2", "doc_ids", "__embedding_score", "DIALECT", "2",
        )
        if results and len(results) > 1:
            doc_ids_raw = results[1][1]
            score = float(results[1][3])
            if score < 1.0 - self.threshold:  # COSINE distance < 0.05
                return json.loads(doc_ids_raw)
        return None

    def store(self, query_emb: np.ndarray, doc_ids: list[str]):
        key = f"sem:{hashlib.md5(query_emb.tobytes()).hexdigest()}"
        self.redis.hset(key, mapping={
            "embedding": query_emb.astype(np.float32).tobytes(),
            "doc_ids": json.dumps(doc_ids),
            "stored_at": str(time.time()),
        })
        self.redis.expire(key, self.ttl)
```

Combined hit ratio for a 2-level cache:

$$\text{effective\_hit} = \text{L1\_hit} + (1 - \text{L1\_hit}) \times \text{L2\_hit}$$

$$= 0.40 + 0.60 \times 0.50 = 0.70$$

A 70% effective hit rate means 70% fewer GPU embedding calls and 70% cost savings.

💡 **Tip**: Benchmark your similarity threshold against a human-labeled dataset. For 100 query pairs, ask "would the same answer satisfy both?" Tune until false-positive rate is <5%. Typical sweet spot: 0.92-0.95 for general text, 0.95-0.98 for technical text.

⚠️ **Pitfall**: Setting the threshold too low (e.g., 0.85) returns cached responses for tangentially related queries, degrading user experience.

**Caso real — LLM Edge Gateway**: A Redis semantic cache intercepting OpenAI/Anthropic API requests achieved 30-40% cost reduction. Cosine threshold of 0.92 was tuned empirically — 0.85 increased hit rate but returned irrelevant responses for different code generation tasks, 0.98 preserved quality but halved hit rate. The threshold directly controls the API cost vs quality tradeoff.

## 2. Prediction and Feature Caching

### Prediction Cache

ML inference is a deterministic function of its inputs (for fixed model weights). Given the same input features, the model produces identical predictions. If the same input arrives repeatedly — and input distributions typically follow a power law — caching predictions eliminates redundant GPU computation.

The cache key: `SHA256(serialized input features + model_version)`. The TTL strategy depends on model update frequency. If the model retrains daily, a 24-hour TTL is safe. If the model updates in real-time, the TTL must be shorter and the cache key must include model version.

```python
import hashlib, json
from collections import OrderedDict

class PredictionCache:
    """Two-level prediction cache: L1 (in-memory LRU) + L2 (Redis).

    Cache key includes model_version for automatic invalidation
    on model deploy. State-of-the-art pattern used by Google Search
    ranking and DoorDash recommendation systems.
    """

    def __init__(self, redis_client, model_version="latest",
                 ttl_seconds=86400, l1_maxsize=10000):
        self.redis = redis_client
        self.model_version = model_version
        self.ttl = ttl_seconds
        self.l1 = OrderedDict()
        self.l1_maxsize = l1_maxsize

    def _key(self, **features):
        canonical = json.dumps(features, sort_keys=True)
        fingerprint = hashlib.sha256(
            f"{canonical}|{self.model_version}".encode()
        ).hexdigest()
        return f"pred:{fingerprint}"

    def get(self, **features):
        key = self._key(**features)
        # L1: In-memory LRU (sub-millisecond)
        if key in self.l1:
            self.l1.move_to_end(key)
            return self.l1[key]
        # L2: Redis (~1ms, shared across replicas)
        cached = self.redis.get(key)
        if cached is not None:
            result = json.loads(cached)
            self.l1[key] = result  # Promote to L1
            if len(self.l1) > self.l1_maxsize:
                self.l1.popitem(last=False)
            return result
        return None  # Cache miss

    def set(self, prediction, **features):
        key = self._key(**features)
        self.redis.setex(key, self.ttl, json.dumps(prediction))
        self.l1[key] = prediction
        if len(self.l1) > self.l1_maxsize:
            self.l1.popitem(last=False)

    def invalidate(self, new_version):
        """Implicitly invalidates all cached predictions."""
        self.model_version = new_version
        self.l1.clear()
```

💡 **Tip**: Always include `model_version` in every cache key. This makes model deployment the cache invalidation event. Version changes → new keys → old keys expire naturally via TTL. No explicit flush required.

⚠️ **Pitfall**: Caching predictions without model version in the key. When the model updates, the cache silently serves predictions from the old model — a subtle correctness bug.

### Feature Cache

Feature stores are the most heavily queried component in ML inference. Every prediction request requires 10-100 feature reads. Feature caching exploits the fact that many features change slowly relative to request frequency. A user's age, country, and subscription tier change on timescales of weeks to years.

| Feature Category | TTL | Examples | % of Reads |
|-----------------|-----|----------|------------|
| Static | 24h | user_age, user_country | 30% |
| Slow-changing | 1h | subscription_tier, item_price | 35% |
| Behavioral | 5min | purchase_count_7d | 25% |
| Real-time | 60s | session_active | 10% |

With per-feature TTLs, 65% of feature reads hit the cache. The prediction pipeline becomes:

```python
async def predict_with_feature_cache(user_id, item_id, model, feature_store, cache):
    features = await cache.mget("user", user_id, required_user_features)
    missing = [f for f, v in features.items() if v is None]
    if missing:
        from_store = await feature_store.get(user_id=user_id, feature_names=missing)
        await cache.mset("user", user_id, from_store)
        features.update(from_store)
    return model.predict(features)
```

**Caso real — DoorDash**: Measured empirical staleness tolerance for each model. Fraud model requires 60s TTL max. Menu recommendation tolerates 15-minute stale features → 15-minute TTL. Result: 65% cache hit rate, ~$50K/month GPU cost reduction. The key insight: measure staleness impact via A/B testing before setting TTLs. Don't guess — run the experiment.

❌ **Antipattern**: Single global TTL for all features. Forces real-time features to be cached too long (stale predictions) or static features too short (unnecessary misses).

✅ **Best practice**: Per-feature TTLs based on natural rate of change. A/B test different TTLs and monitor prediction quality to find the optimal value for each feature.

❌ **Antipattern**: No cache pre-warming on deploy. Cold cache causes a thundering herd on the feature store.

✅ **Best practice**: Pre-warm the feature cache on deploy by loading the top-N most active users' features. A 10-minute pre-warm with 1M users covers 90%+ of initial traffic.

## 3. Memory Hierarchy: GPU VRAM → CDN

### Tiered Storage Economics

ML systems generate petabytes of data across dramatically different access patterns. The storage tier for each data class is determined by access frequency, latency requirements, and cost tolerance:

$$T_{\text{access}} \propto \frac{1}{\text{frequency}} \quad \text{and} \quad \text{cost} \propto \text{speed}$$

| Tier | Latency | Bandwidth | Cost/GB/month | Capacity | Example Data |
|------|---------|-----------|---------------|----------|--------------|
| **Hot** (GPU VRAM) | ~1μs | 3.35 TB/s (H100) | ~$720 | 80GB/GPU | Active weights, KV cache |
| **Warm** (RAM/NVMe) | 100ns-100μs | 100 GB/s (DDR5) | ~$0.10 | TBs/node | Embeddings, checkpoints |
| **Cold** (S3/Blob) | 100ms-5s | 500 MB/s | ~$0.023 | PBs | Training data, archives |

Hot is 31,000× more expensive than Cold per month ($720/GB vs $0.023/GB). Every gigabyte moved from Hot to Warm saves ~$700/month. The right-tiering principle: data lives at the cheapest tier that satisfies its latency requirement.

**Caso real — Anthropic's Claude training**: Training data in S3 (Cold); active shards pre-fetched to NVMe SSDs on training nodes (Warm); current training batch and model weights in GPU VRAM (Hot). Without this tiered architecture, training would be I/O-bound — GPUs waiting for data from S3.

### Model Artifact CDN Distribution

Model binaries (.pt, .onnx, .safetensors) range from MBs to hundreds of GBs. When a new serving replica starts, it downloads the model before serving. Download latency directly contributes to cold start.

Serving from S3 is suboptimal. A 5GB Llama model takes ~12.5s from S3 vs ~2s from CloudFront CDN edge:

$$T_{\text{download}} = \frac{S_{\text{model}}}{B_{\text{effective}}}$$

CDN benefit: $B_{\text{effective}}$ jumps from ~400 MB/s (S3 single-region) to ~2500 MB/s (CloudFront edge cache hit). The 6× speedup makes auto-scaling practical.

CDN caching works because versioned model artifacts are immutable. A model at `/models/v3.1.0/model.onnx` never changes — no cache invalidation problem. Versioned URLs turn model files into perfectly cacheable static assets.

**Caso real — Hugging Face**: Distributes thousands of models through CloudFront CDN. Llama-70B (~140GB) downloads in ~45s with 16 parallel connections vs ~7 minutes single-threaded S3. This enables the tight iteration cycles that ML researchers depend on.

```python
# Kubernetes init container pattern: download from CDN to local SSD
def get_download_command(model_url: str, sha256: str) -> str:
    """Generate download command with parallel connections and verification."""
    return (
        f"aria2c -x16 -s16 --dir=/models '{model_url}' && "
        f"echo '{sha256}  /models/{model_url.rsplit('/', 1)[-1]}' | sha256sum -c"
    )
```

❌ **Antipattern**: Mutable URLs (`/models/latest/model.onnx`). "Latest" changes unpredictably, destroying CDN cache efficiency — every change forces a cache miss.

✅ **Best practice**: Versioned immutable URLs (`/models/v3.1.0/model.onnx`). For "latest," maintain a redirect service that 302-redirects to the versioned URL. The CDN caches the 302 briefly; the actual download is a 200 with infinite TTL.

### KV Cache and RadixAttention

The KV (Key-Value) cache is the hidden gem of LLM inference optimization. During autoregressive generation, the model computes key and value tensors for every token. These KV pairs are reused for subsequent token generation — without this, generation would be $O(n^2)$ compute instead of $O(n)$.

Per-token KV cache size:

$$S_{KV} = 2 \times N_{\text{layers}} \times d_{\text{model}} \times \text{bytes\_per\_elem}$$

For Llama-70B: $80 \times 8192 \times 2 \times 2 = 2.6$ MB/token. A conversation with 4096 context + 512 generated tokens = 4608 tokens × 2.6 MB = 12 GB — ~15% of an H100's 80 GB VRAM.

**RadixAttention (SGLang)** recognizes that requests share common prompt prefixes. If two users start with the same system prompt and conversation history prefix, their KV caches for the shared prefix are identical. RadixAttention uses a radix tree (prefix tree) to store KV caches, reusing nodes for shared prefixes. This reduces KV cache memory by 30-50%.

Without sharing: two requests with 1500 tokens each = $2 \times 1500 \times 2.6\text{MB} = 7.8$ GB.
With RadixAttention: one shared 1000-token prefix ($2.6$ GB) + two 500-token suffixes ($1.3$ GB each) = 5.2 GB — a 33% memory savings that scales with concurrent users.

⚠️ **Pitfall**: Ignoring KV cache memory in capacity planning. A 70B model on H100 may have 35 GB (INT8 weights) leaving ~45 GB for KV cache. At 2.6 MB/token, that's ~17K tokens shared across ALL concurrent requests. Over-subscription causes preemption or OOM.

💡 **Tip**: Plan GPU VRAM as: `model_weights + KV_cache = available_VRAM`. Compute max concurrent sequences = `available_VRAM - model_weights / S_KV`. Monitor KV cache utilization and set limits accordingly.

⚠️ **Pitfall**: Using round-robin load balancing for LLM inference, destroying KV cache locality. Each request from the same user hits a different GPU, forcing full KV recomputation.

💡 **Tip**: Use consistent-hashing routing (see [[03 - Load Balancing, Sharding and Scaling ML Systems|Note 03]]) to pin users to specific GPU replicas. Combined with RadixAttention, this maximizes KV cache reuse.

## 🎯 Key Takeaways

- A 2-level embedding cache (L1 LRU + L2 Redis semantic) reduces embedding compute by 40-60% for RAG workloads
- Prediction caches require model-version-aware keys — TTL matches retrain frequency, not arbitrary values
- Per-feature TTLs (static → 24h, behavioral → 5min, real-time → 60s) reduce feature store load by 50-80%
- CDN distribution with versioned immutable URLs turns model artifacts into infinitely cacheable static assets — 6× faster downloads than direct S3
- Hot tier (GPU VRAM) is 31,000× more expensive than Cold (S3) — every GB moved down saves ~$700/month
- RadixAttention (KV cache prefix sharing) is the highest-leverage cache in LLM serving, reducing memory by 30-50%
- Consistent-hashing routing preserves KV cache locality — round-robin destroys it, forcing full recomputation on every request

## References

- Redis. (2024). *RediSearch Documentation — Vector Similarity*. https://redis.io/docs/latest/develop/interact/search-and-query/query/vector-search/
- SGLang. (2024). *RadixAttention: Efficient LLM Serving with Prefix Sharing*. https://github.com/sgl-project/sglang
- DoorDash Engineering. (2023). "Building DoorDash's ML Platform." https://doordash.engineering/
- Hugging Face. (2024). *Model Hub Infrastructure*. https://huggingface.co/docs/hub/models-downloading
- AWS. (2024). *Amazon CloudFront Developer Guide*. https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly.
- [[03 - Load Balancing, Sharding and Scaling ML Systems|Load Balancing for ML]]
- [[01 - CAP Theorem and Consistency Models in ML Workloads|CAP Theorem]]
- [[../../31 - FastAPI for ML/00 - Welcome to FastAPI for ML|FastAPI for ML]]

## 📦 Código de compresión

```python
"""Unified tiered ML cache: L0 (LRU) → L1 (Redis) → L2 (GPU compute).
Supports embedding, prediction, and feature caching in a single interface.
"""
import hashlib, json, time, threading
from collections import OrderedDict
from typing import Any, Optional

class TieredMLCache:
    """Production-grade unified cache for ML inference."""

    def __init__(self, redis=None, l0_size=10000,
                 pred_ttl=86400, feat_ttl=300, emb_ttl=3600):
        self.redis = redis
        self.l0 = OrderedDict()
        self.l0_size, self._lock = l0_size, threading.Lock()
        self.ttl = {"pred": pred_ttl, "feat": feat_ttl, "emb": emb_ttl}
        self.stats = {"l0_hits": 0, "l1_hits": 0, "misses": 0}

    def _l0_get(self, key):
        with self._lock:
            if key in self.l0:
                self.l0.move_to_end(key)
                self.stats["l0_hits"] += 1
                return self.l0[key]
        return None

    def _l0_set(self, key, val):
        with self._lock:
            self.l0[key] = val
            if len(self.l0) > self.l0_size:
                self.l0.popitem(last=False)

    def _l1_get(self, key):
        if self.redis:
            val = self.redis.get(key)
            if val is not None:
                self.stats["l1_hits"] += 1
                return json.loads(val)
        return None

    def _l1_set(self, key, val, ttl_type):
        if self.redis:
            self.redis.setex(key, self.ttl[ttl_type], json.dumps(val))

    def get_prediction(self, model_ver: str, **features) -> Optional[dict]:
        key = f"pred:{model_ver}:{hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()}"
        cached = self._l0_get(key) or self._l1_get(key)
        if cached is not None:
            self._l0_set(key, cached)
            return cached
        self.stats["misses"] += 1
        return None

    def set_prediction(self, model_ver: str, prediction: dict, **features):
        key = f"pred:{model_ver}:{hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()}"
        self._l1_set(key, prediction, "pred")
        self._l0_set(key, prediction)

    @property
    def hit_rate(self):
        total = sum(self.stats.values())
        return (self.stats["l0_hits"] + self.stats["l1_hits"]) / total if total else 0.0

if __name__ == "__main__":
    cache = TieredMLCache(redis=None)  # In-memory only for demo
    cache.set_prediction("v1", {"score": 0.85}, user="123", item="abc")
    result = cache.get_prediction("v1", user="123", item="abc")
    print(f"Result: {result}, Hit rate: {cache.hit_rate:.0%}")
    # Output: Result: {'score': 0.85}, Hit rate: 100%
```
