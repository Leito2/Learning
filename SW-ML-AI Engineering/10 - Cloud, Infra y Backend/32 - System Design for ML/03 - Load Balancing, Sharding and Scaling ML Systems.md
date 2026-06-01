# ⚖️ 03 - Load Balancing, Sharding, and Scaling ML Systems

## 🎯 Learning Objectives

- Contrast load balancing algorithms and justify consistent hashing as the default for ML inference where KV cache locality dominates
- Design GPU-aware load balancers that route based on free VRAM, model type, and batch composition — not connection count
- Implement feature store sharding by `user_id` with consistent hashing for single-digit-millisecond reads
- Configure auto-scaling for GPU services using custom metrics (queue depth, GPU utilization) and articulate why CPU-based HPA fails for GPU workloads

## Introduction

Load balancing in traditional systems is a solved problem — round-robin across stateless replicas. None of this works for ML inference. An ML serving replica is not stateless: its GPU VRAM holds model weights, KV caches, and active inference batches. Sending a user's second request to a different GPU than their first destroys KV cache locality, forcing recomputation of the entire attention context. Sending a batch to a GPU whose VRAM is saturated causes OOM errors. Auto-scaling on CPU misses GPU saturation entirely.

The etymology of *load balancing* is straightforward — balance load across workers. But in ML systems, "load" is not requests per second. It is VRAM occupancy, batch composition, and KV cache pressure. *Sharding* — partitioning data across nodes — comes from database design but takes on new meaning: feature store sharding by user_id, embedding table sharding by hash ring, and model parallelism. These concepts connect to CAP classification ([[01 - CAP Theorem and Consistency Models in ML Workloads|Note 01]]) and caching architectures ([[02 - Caching, CDNs and Storage Architectures for ML|Note 02]]).

---

## 1. Consistent Hashing for ML Inference

Three algorithms dominate load balancing, and their suitability for ML workloads diverges dramatically:

**Round-Robin**: Each request to the next replica in circular order. Works for stateless web servers. For ML inference, catastrophic: two consecutive requests from the same user land on different GPUs, destroying KV cache locality. The model must recompute the full context.

**Least-Connections**: Request to the replica with fewest active connections. Connection count is a misleading proxy for GPU load — a single connection running a 2000-token LLM generation occupies 15 GB of VRAM, while 100 connections running 50-token generations may occupy the same.

**Consistent Hashing**: A hash ring maps keys (user IDs, session IDs) to replicas. The ring is a circle of size $2^{32}$, and each replica is assigned multiple positions (virtual nodes). A request's key is hashed, and the ring is walked clockwise to the next replica. When a replica is added or removed, only keys between its predecessor and itself need reassignment — this property is *monotonicity* and is the defining advantage for ML workloads.

For ML inference with KV cache, consistent hashing on `hash(user_id)` ensures all requests from the same user route to the same GPU for the duration of their session, preserving KV cache locality and reducing memory usage by 30-50%.

$$\text{Miss rate}_{\text{round-robin}} = \frac{N-1}{N} \quad \text{vs} \quad \text{Miss rate}_{\text{consistent}} = 0$$

For $N = 8$ GPUs, round-robin destroys cache for 87.5% of consecutive requests. Consistent hashing preserves it for 100%.

❌/✅ **Round-robin for LLM inference**: The canonical anti-pattern. Every request from the same user hits a different GPU, wasting 30-50% of GPU time on recomputation. Always use consistent hashing on `user_id` for ML inference with stateful caches. The small load imbalance (~5-10% variance) is negligible vs the throughput gain from cache reuse.

**Caso real: Notion AI** uses consistent-hash routing for LLM inference. When a user starts a conversation, the load balancer hashes their user ID and routes all subsequent requests to the same GPU replica, preserving the KV cache for the entire conversation history — a 4000-token context that would cost 10.4 GB of VRAM to recompute on every request. Notion reported that consistent hashing reduced their GPU fleet size by 35% compared to round-robin.

```python
"""
consistent_hash_router.py — Pins user sessions to specific GPU replicas
for KV cache locality in ML inference.
"""
import hashlib, bisect
from typing import Optional
from dataclasses import dataclass

@dataclass
class GPUReplica:
    id: str
    vram_total_gb: float
    vram_used_gb: float = 0.0
    model: str = "llama-70b"
    healthy: bool = True

    @property
    def vram_available_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

class ConsistentHashRouter:
    def __init__(self, virtual_nodes_per_gpu: int = 150):
        self.vnodes_per_gpu = virtual_nodes_per_gpu
        self.ring: list[tuple[int, GPUReplica]] = []
        self.replicas: dict[str, GPUReplica] = {}

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)

    def add_replica(self, replica: GPUReplica):
        self.replicas[replica.id] = replica
        for i in range(self.vnodes_per_gpu):
            pos = self._hash(f"{replica.id}:vnode:{i}")
            bisect.insort(self.ring, (pos, replica), key=lambda x: x[0])

    def remove_replica(self, replica_id: str):
        self.replicas.pop(replica_id, None)
        self.ring = [(p, r) for p, r in self.ring if r.id != replica_id]

    def get_replica(self, key: str) -> Optional[GPUReplica]:
        if not self.ring:
            return None
        pos = self._hash(key)
        positions = [p for p, _ in self.ring]
        idx = bisect.bisect_right(positions, pos) % len(self.ring)
        for _ in range(len(self.ring)):
            replica = self.ring[idx][1]
            if replica.healthy:
                return replica
            idx = (idx + 1) % len(self.ring)
        return None
```

💡 **Tip**: Use 100-200 virtual nodes per physical GPU. Too few (e.g., 10) creates severe imbalance when a node joins or leaves. Too many adds memory overhead. With 150 vnodes per GPU, the law of large numbers ensures roughly equal key distribution.

⚠️ **Pitfall**: Using `hash(user_id) % N` instead of consistent hashing. When N changes (scale-up), ALL keys get reassigned, causing a 100% cache invalidation event. Consistent hashing reassigns only ~1/(N+1) of keys.

---

## 2. GPU-Aware Load Balancing

Traditional load balancers operate on connection counts — send to the server with fewest open connections. For GPU inference, connections are the wrong metric. A GPU can be "idle" (0 connections) but have 78/80 GB VRAM occupied. Conversely, a GPU with 50 active connections processing short completions may have plenty of VRAM headroom.

A GPU-aware load balancer must consider:

1. **VRAM availability**: Route to the GPU with the most free VRAM. Free VRAM = total − model weights − KV cache − batch activations.
2. **Batch composition**: A 50-token completion uses minimal KV cache; a 4000-token conversation uses 10.4 GB. Route heavy requests to GPUs with headroom.
3. **Model type**: In multi-model deployments, each GPU runs one model. Route only to GPUs running the requested model.
4. **Backpressure**: When all GPUs are saturated, queue requests rather than routing to unavailable GPUs. This prevents cascading failures where saturated GPUs return errors → client retries → more saturation.

The VRAM-aware routing algorithm:

$$R_{\text{target}} = \arg\max_{r \in R_{\text{model}}} \left( \text{VRAM}_{\text{free}}(r) - \text{VRAM}_{\text{estimated}}(request) \right)$$

If no GPU has sufficient free VRAM, the request is queued until a GPU becomes available or times out.

❌/✅ **Least-connections for GPU inference**: Anti-pattern. A GPU with 0 connections may have 100% VRAM utilization from a single long-running batch. A GPU with 100 connections may have 20% VRAM utilization. Always route based on VRAM, not connections.

**Caso real: Google Vertex AI Prediction** uses GPU-aware load balancing. Their prediction service measures VRAM usage per GPU in real-time and routes requests to the GPU with the most available VRAM. For multi-model deployments, each GPU registers its model, and the load balancer performs model-level dispatch before VRAM-level routing. When all GPUs for a model are saturated, Vertex AI returns HTTP 429 with `Retry-After` headers — implementing backpressure at the API level.

```python
"""
gpu_aware_lb.py — VRAM-based routing with backpressure.
"""
import time, threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUState:
    id: str
    model: str
    vram_total_gb: float = 80.0
    vram_kv_cache_gb: float = 0.0
    active_requests: int = 0
    max_concurrent: int = 32

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - (35.0 + self.vram_kv_cache_gb)

    @staticmethod
    def estimate_vram(prompt_tokens: int, max_output_tokens: int) -> float:
        kv_cache_mb = (prompt_tokens + max_output_tokens) * 2.6
        return (kv_cache_mb + prompt_tokens * 1.0) / 1024.0

class GPUAwareLoadBalancer:
    def __init__(self, request_timeout_ms: int = 2000):
        self.gpus: dict[str, GPUState] = {}
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self.request_timeout = request_timeout_ms / 1000.0

    def route(self, model: str, prompt_tokens: int, max_output_tokens: int) -> dict:
        with self._lock:
            est = GPUState.estimate_vram(prompt_tokens, max_output_tokens)
            candidates = [g for g in self.gpus.values() if g.model == model]
            best = max((g for g in candidates if g.vram_free_gb >= est),
                       key=lambda g: g.vram_free_gb, default=None)
            if best:
                best.vram_kv_cache_gb += est
                best.active_requests += 1
                return {"status": "routed", "gpu_id": best.id}
            if len(self._queue) < 200:
                self._queue.append((model, prompt_tokens, max_output_tokens, time.time()))
                return {"status": "queued", "position": len(self._queue)}
            return {"status": "rejected", "reason": "circuit_breaker"}
```

⚠️ **Pitfall**: No backpressure mechanism. When GPUs are saturated, requests pile up in TCP connection queues, timeout after 30 seconds, and clients retry — creating a retry storm that makes saturation worse.

💡 **Tip**: Implement explicit backpressure: return HTTP 429 immediately when queue depth exceeds 80% of max capacity. Include `Retry-After` headers. This gives clients clear signals to back off and prevents retry amplification.

---

## 3. Feature Store Sharding by User ID

A feature store serving millions of users per second cannot fit all features in a single Redis instance. The dataset must be partitioned (sharded). The optimal strategy is **sharding by user_id using consistent hashing**. Every feature for a given user — demographic, behavioral, embedding — lives on the same shard. When a model requests features for `user_123`, the client hashes `user_123`, resolves to a shard, and reads all features in a single `MGET`. No scatter-gather. No cross-shard coordination.

The alternative — sharding by feature name (all `user_age` on shard A, all `purchase_count` on shard B) — forces every read to scatter across multiple shards. For 20 features per request: 20 parallel reads, each potentially to a different shard, with P99 latency determined by the slowest shard.

For multi-entity features: user features sharded by `user_id`, item features by `item_id`, context features (time of day) are small enough to replicate across all shards.

❌/✅ **Sharding by feature name**: Anti-pattern. Scatters every read across multiple shards, amplifying P99 latency by the slowest-shard problem. Always shard by entity ID so all features for an entity live on one shard.

**Caso real: DoorDash** shards their feature store by `user_id` using consistent hashing. With 500M users and 200 features per user (~3 KB total = ~1.5 TB), sharded across 12 Redis nodes (128 GB each), each node holds ~125 GB. When a new node is added, only ~1/13 (~7.7%) of users migrate. Contrast with sharding by feature name: 200 features would require reading from up to 200 different shards per request, with P99 driven by the slowest shard (~10 ms). Sharding by user_id reduces this to a single shard read (~0.5 ms) — a 20× improvement.

```python
"""
feature_store_sharding.py — Sharded feature store with consistent hashing.
All features for a user on a single shard ⇒ single MGET.
"""
import hashlib, bisect, json
from typing import Any

class ShardedFeatureStore:
    def __init__(self, shard_addresses: list[str], vnodes: int = 150):
        self.vnodes = vnodes
        self.ring: list[tuple[int, int]] = []
        self.shards = []
        for idx, addr in enumerate(shard_addresses):
            host, port = addr.split(":")
            import redis
            self.shards.append(redis.Redis(host=host, port=int(port), decode_responses=True))
            self._add_shard_to_ring(idx)

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)

    def _add_shard_to_ring(self, idx: int):
        for i in range(self.vnodes):
            pos = self._hash(f"shard:{idx}:v:{i}")
            bisect.insort(self.ring, (pos, idx), key=lambda x: x[0])

    def _get_shard(self, user_id: str):
        pos = self._hash(user_id)
        positions = [p for p, _ in self.ring]
        idx = bisect.bisect_right(positions, pos) % len(self.ring)
        return self.shards[self.ring[idx][1]]

    def get_user_features(self, user_id: str, feature_names: list[str]) -> dict[str, Any]:
        shard = self._get_shard(user_id)
        keys = [f"feature:{user_id}:{name}" for name in feature_names]
        values = shard.mget(keys)
        result = {}
        for name, value in zip(feature_names, values):
            result[name] = json.loads(value) if value else None
        return result
```

⚠️ **Pitfall**: Sharding by feature name (or hash of feature value). Scatters every read across shards, amplifying P99 latency.

💡 **Tip**: Shard by entity ID. All features for entity X live on shard Y. Reads are single-shard `MGET`s with predictable sub-millisecond latency.

---

## 4. Auto-Scaling: GPU Metrics, Not CPU

Kubernetes HPA scales pods based on CPU utilization by default. For web servers this works. For GPU inference, it fails catastrophically: a GPU inference pod shows 5-10% CPU utilization while its GPU is at 100% — the CPU is just passing data to the GPU and waiting. CPU-based HPA will never trigger a scale-up for a GPU-saturated service.

❌/✅ **CPU HPA for GPU services**: The canonical anti-pattern. CPU is idle; GPU is saturated; HPA is blind to the actual bottleneck.

The correct metrics for GPU auto-scaling:

1. **GPU utilization (%)** — `DCGM_FI_DEV_GPU_UTIL`. Scale up when sustained > 80%.
2. **GPU memory utilization (%)** — `DCGM_FI_DEV_FB_USED_PERCENT`. Scale up when > 85%.
3. **Queue depth** — Number of requests in the backpressure queue. The most direct signal of insufficient capacity.
4. **Request latency P99** — When P99 exceeds SLA, scale up.

**Caso real: A major ride-sharing company's ML inference platform** migrated from CPU-based HPA to GPU-custom-metric HPA after a production incident. During peak hours, their fraud detection model experienced 3-second P99 latency while CPU utilization stayed at 8%. The CPU-based HPA maintained minimum replicas. After migrating to GPU-utilization and queue-depth-based HPA, the system scaled proactively. The cost impact was counter-intuitive: custom-metric HPA **reduced** GPU costs by 25% — because CPU-based HPA kept minimum replicas running 24/7 while GPU-metric HPA scaled down aggressively during off-peak hours.

Key HPA configuration considerations:

- `scaleUp.stabilizationWindowSeconds`: Set to 60-120 seconds to wait for cold starts before re-evaluating.
- `scaleDown.stabilizationWindowSeconds`: Set to 300 seconds — scale down 1 pod every 60 seconds.
- Cold starts for 70B models take 30-120 seconds (model download + VRAM load). Use CDN-distributed model artifacts to minimize this ([[02 - Caching, CDNs and Storage Architectures for ML|Note 02]]).

⚠️ **Pitfall**: Scaling up too aggressively during bursts. Adding 10 GPUs simultaneously triggers 10 cold starts. During cold start, new replicas contribute zero capacity, and the scale-up may trigger again — creating oscillation.

⚠️ **Pitfall**: Scaling down too aggressively during traffic dips, then encountering a spike. Over-provisioning by 1-2 replicas costs $3-6/hour; under-provisioning during a spike loses users.

---

## 🎯 Key Takeaways

- Consistent hashing is the default load balancing algorithm for ML inference — it preserves KV cache locality, reducing GPU memory usage by 30-50% vs round-robin
- GPU-aware routing must consider VRAM availability, batch composition, and model type, not connection counts
- Feature stores should shard by entity ID using consistent hashing — this guarantees single-shard `MGET` reads under 1 ms
- CPU-based HPA is blind to GPU saturation; use custom metrics (GPU utilization, queue depth, P99 latency) for inference auto-scaling
- Backpressure with circuit breaking prevents retry storms during GPU saturation
- Cold start latency for model loading (30-120 seconds) must inform auto-scaling cooldown windows
- The same consistent hashing ring can serve both inference routing and feature store sharding

## References

- Karger, D. et al. (1997). "Consistent Hashing and Random Trees." *STOC '97*.
- SGLang. (2024). *RadixAttention: Efficient LLM Serving with Prefix Sharing*.
- NVIDIA. (2024). *DCGM User Guide*.
- Kubernetes. (2024). *Horizontal Pod Autoscaling with Custom Metrics*.
- Notion Engineering. (2024). "Scaling Notion AI's Inference Infrastructure."
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. Chapter 5.
- [[01 - CAP Theorem and Consistency Models in ML Workloads|CAP Theorem]]
- [[02 - Caching, CDNs and Storage Architectures for ML|Caching for ML]]
- [[04 - Back-of-Envelope Estimation - Throughput, Latency, Storage and Cost|Back-of-Envelope]]

## 📦 Código de compresión

```python
"""
ml_load_balancer.py — Complete ML load balancing reference: consistent
hashing ring, GPU-aware routing, feature store sharding, auto-scaling.
"""
import hashlib, bisect, json, time, threading, math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

KV_CACHE_MB_PER_TOKEN = 2.6

@dataclass
class Replica:
    id: str; model: str; vram_total_gb: float = 80.0
    vram_used_gb: float = 0.0; healthy: bool = True

class ConsistentHashRing:
    def __init__(self, vnodes: int = 150):
        self.vnodes, self.ring, self.nodes = vnodes, [], {}
    def _hash(self, k: str) -> int:
        return int(hashlib.sha256(k.encode()).hexdigest(), 16) % (2**32)
    def add_node(self, r: Replica):
        self.nodes[r.id] = r
        for i in range(self.vnodes):
            p = self._hash(f"{r.id}:v:{i}")
            bisect.insort(self.ring, (p, r), key=lambda x: x[0])
    def remove_node(self, rid: str):
        self.nodes.pop(rid, None)
        self.ring = [(p, r) for p, r in self.ring if r.id != rid]
    def get_node(self, key: str) -> Optional[Replica]:
        if not self.ring: return None
        ps = [p for p, _ in self.ring]
        i = bisect.bisect_right(ps, self._hash(key)) % len(self.ring)
        for _ in range(len(self.ring)):
            r = self.ring[i][1]
            if r.healthy: return r
            i = (i + 1) % len(self.ring)
        return None

@dataclass
class GPUMetrics:
    gpu_id: str; model: str; gpu_util_pct: float
    vram_free_gb: float; active_requests: int; p99_latency_ms: float

class GPUAwareRouter:
    def __init__(self, ring: ConsistentHashRing):
        self.ring, self.metrics, self._queue, self._lock = ring, {}, deque(), threading.Lock()
    def update_metrics(self, m: GPUMetrics):
        with self._lock: self.metrics[m.gpu_id] = m
    def route(self, user_id: str, model: str, prompt: int, output: int) -> dict:
        with self._lock:
            est = ((prompt + output) * KV_CACHE_MB_PER_TOKEN) / 1024.0
            candidates = [m for m in self.metrics.values() if m.model == model]
            best = max((g for g in candidates if g.vram_free_gb >= est),
                       key=lambda g: g.vram_free_gb, default=None)
            if best: return {"status": "routed", "gpu_id": best.gpu_id}
            if len(self._queue) < 200:
                self._queue.append((user_id, model, prompt, output, time.time()))
                return {"status": "queued", "position": len(self._queue)}
            return {"status": "rejected", "reason": "queue_full"}

@dataclass
class Shard:
    address: str; redis: Any = None

class ShardedFeatureStore:
    def __init__(self, addrs: list[str], vnodes: int = 150):
        self.ring: list[tuple[int, int]] = []
        self.shards = []
        for i, addr in enumerate(addrs):
            import redis
            self.shards.append(redis.Redis.from_url(f"redis://{addr}"))
            for j in range(vnodes):
                pos = int(hashlib.sha256(f"shard:{i}:v:{j}".encode()).hexdigest(), 16) % (2**32)
                bisect.insort(self.ring, (pos, i), key=lambda x: x[0])
    def _shard_for(self, uid: str) -> Any:
        ps = [p for p, _ in self.ring]
        i = bisect.bisect_right(ps, int(hashlib.sha256(uid.encode()).hexdigest(), 16) % (2**32)) % len(self.ring)
        return self.shards[self.ring[i][1]]
    def get_features(self, uid: str, features: list[str]) -> dict:
        shard = self._shard_for(uid)
        vals = shard.mget([f"feat:{uid}:{f}" for f in features])
        return {f: json.loads(v) if v else None for f, v in zip(features, vals)}

class AutoScaler:
    def __init__(self, min_r=2, max_r=50, util_up=80.0, util_down=40.0):
        self.min_r, self.max_r, self.util_up, self.util_down = min_r, max_r, util_up, util_down
        self.last_scale, self.replicas = 0.0, min_r
    def evaluate(self, metrics: list[GPUMetrics], queue: int) -> str:
        if not metrics: return f"HOLD:{self.replicas}"
        avg = sum(m.gpu_util_pct for m in metrics) / len(metrics)
        now = time.time()
        if avg > self.util_up and self.replicas < self.max_r and now - self.last_scale > 60:
            self.replicas = min(self.max_r, self.replicas + max(1, queue // 50))
            self.last_scale = now
            return f"UP:{self.replicas}"
        if avg < self.util_down and queue < 10 and self.replicas > self.min_r and now - self.last_scale > 300:
            self.replicas -= 1
            self.last_scale = now
            return f"DOWN:{self.replicas}"
        return f"HOLD:{self.replicas}"

if __name__ == "__main__":
    ring = ConsistentHashRing()
    ring.add_node(Replica("gpu-0", "llama-70b"))
    ring.add_node(Replica("gpu-1", "llama-70b"))
    router = GPUAwareRouter(ring)
    router.update_metrics(GPUMetrics("gpu-0", "llama-70b", 45, 50.0, 8, 120))
    router.update_metrics(GPUMetrics("gpu-1", "llama-70b", 85, 10.0, 24, 450))
    print(router.route("user_123", "llama-70b", 500, 200))
    scaler = AutoScaler()
    print(scaler.evaluate(list(router.metrics.values()), len(router._queue)))
```
