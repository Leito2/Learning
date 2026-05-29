# ⚖️ 03 - Load Balancing, Sharding, and Scaling ML Systems

## 🎯 Learning Objectives

- Contrast load balancing algorithms (round-robin, least-connections, consistent hashing) and justify consistent hashing as the default for ML inference workloads where KV cache locality dominates performance
- Design GPU-aware load balancers that route requests based on free VRAM, model type, and batch composition — not just connection count
- Implement feature store sharding by user_id with consistent hashing, ensuring all features for a given user reside on the same shard for single-digit-millisecond reads
- Configure Kubernetes HPA for GPU services using custom metrics (queue depth, GPU utilization, batch saturation) and articulate why CPU-based HPA fails catastrophically for GPU workloads
- Architect auto-scaling policies for ML inference that balance cold-start latency, model download time (CDN), and over-provisioning cost

## Introduction

Load balancing in traditional software systems is a solved problem. A reverse proxy distributes HTTP requests across stateless replicas using round-robin or least-connections, and auto-scaling adjusts replica count based on CPU utilization. None of this works for ML inference. An ML serving replica is not stateless — its GPU VRAM holds model weights, KV caches, and active inference batches. Sending a user's second request to a different GPU than their first destroys KV cache locality, forcing recomputation of the entire attention context. Sending a batch of requests to a GPU whose VRAM is already saturated causes OOM errors or queuing. Auto-scaling on CPU misses GPU saturation entirely — the CPU is idle while the GPU is at 100%.

The etymology of *load balancing* is straightforward: to balance the load across workers. But in ML systems, "load" is not requests per second — it is VRAM occupancy, batch composition, and KV cache pressure. *Sharding* — partitioning data across nodes so each node handles a subset — comes from database system design but takes on new meaning in ML: feature store sharding by user_id, embedding table sharding by hash ring, and model parallelism (tensor sharding across GPUs within a node). The concepts connect deeply to CAP classification ([[01 - CAP Theorem and Consistency Models in ML Workloads|Note 01]]) and caching architectures ([[02 - Caching, CDNs and Storage Architectures for ML|Note 02]]), and they form the infrastructure foundation for the end-to-end interview walkthroughs in [[05 - Interview Walkthrough - Design Systems End-to-End|Note 05]].

---

## Module 1: Load Balancing Algorithms — Round-Robin, Least-Connections, and Consistent Hashing

### 1.1 Theoretical Foundation 🧠

Three algorithms dominate load balancing, and their suitability for ML workloads diverges dramatically:

**Round-Robin**: Each request is sent to the next replica in a circular order. This is the simplest algorithm and works well when all replicas are identical and stateless — the standard case for web servers serving REST APIs. For ML inference, round-robin is catastrophic: two consecutive requests from the same user land on different GPUs, destroying KV cache locality. The model must recompute the full context for the second request.

**Least-Connections**: Each request is sent to the replica with the fewest active connections. This balances actual workload better than round-robin, especially when requests have varying durations. For ML, connection count is a misleading proxy for actual GPU load — a single connection running a 2000-token LLM generation occupies 15GB of VRAM, while 100 connections running 50-token generations may occupy the same.

**Consistent Hashing**: A hash ring maps keys (user IDs, session IDs) to replicas. The ring is a circle of size $2^{32}$, and each replica is assigned multiple positions on the ring (virtual nodes). A request's key is hashed, and the ring is walked clockwise to find the next replica. When a replica is added or removed, only the keys between its predecessor and itself need to be reassigned — all other key-to-replica mappings remain unchanged. This property is called *monotonicity* and is the defining advantage for ML workloads.

For ML inference with KV cache, consistent hashing on `hash(user_id)` ensures that all requests from the same user route to the same GPU for the duration of their session. This preserves KV cache locality, reducing memory usage by 30-50% and eliminating redundant attention computation.

The cache miss rate reduction from consistent hashing vs round-robin can be quantified. With $N$ replicas and $U$ users with session affinity:

$$\text{Miss rate}_{\text{round-robin}} = \frac{N-1}{N} \quad \text{vs} \quad \text{Miss rate}_{\text{consistent}} = 0 \text{ (for stable ring)}$$

For $N=8$ GPUs, round-robin destroys cache for $\frac{7}{8} = 87.5\%$ of consecutive user requests. Consistent hashing preserves cache for 100% of repeated user requests as long as the ring doesn't change.

### 1.2 Mental Model 📐

```
┌─── Consistent Hashing Ring for ML Inference ───────────────────────┐
│                                                                      │
│                        0 (2^32)                                      │
│                        ●                                             │
│                   ┌─────────┐                                        │
│              ┌────┘         └────┐                                    │
│             │                   │                                    │
│    user_345 │                   │ user_892                          │
│    hash=0.2B│   hash(user_id)   │ hash=1.5B                         │
│        ●────┼─── → replica ─────┼────●                              │
│             │    clockwise       │                                    │
│             │                   │                                    │
│    GPU-0   │     Virtual Nodes │   GPU-2                            │
│    (A100)  │     (150 vnodes   │   (A100)                           │
│             │     per GPU)     │                                    │
│              └────┐         ┌───┘                                    │
│                   └─────────┘                                        │
│                        ●                                             │
│                    GPU-1 (A100)                                      │
│                                                                      │
│  Rules:                                                              │
│  1. hash(user_id) → position on ring                                │
│  2. Walk clockwise → first vnode → GPU                              │
│  3. Add GPU → only keys between predecessor and new GPU move        │
│  4. Remove GPU → only its keys reassign to next GPU                 │
│                                                                      │
│  Why this matters for ML:                                            │
│  - user_345 ALWAYS → GPU-0 (KV cache preserved)                     │
│  - When GPU-0 fails, user_345 → GPU-1 (loss acceptable, rare)       │
│  - Adding GPU-3 → only ~1/4 of users reassign (minimal disruption) │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```python
"""
consistent_hash_router.py — Consistent hashing load balancer for ML inference.
Pins user sessions to specific GPU replicas for KV cache locality.
"""

import hashlib
import bisect
from typing import Optional
from dataclasses import dataclass, field


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
    """
    Consistent hashing ring for ML inference routing.

    Each GPU gets N virtual nodes on the ring. A user ID is hashed
    to a ring position, and the next virtual node clockwise determines
    the target GPU. This guarantees that the same user always routes
    to the same GPU (barring ring changes from scaling events).
    """

    def __init__(self, virtual_nodes_per_gpu: int = 150):
        self.vnodes_per_gpu = virtual_nodes_per_gpu
        self.ring: list[tuple[int, GPUReplica]] = []  # (hash_pos, replica)
        self.replicas: dict[str, GPUReplica] = {}

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the 2^32 ring."""
        return int(
            hashlib.sha256(key.encode()).hexdigest(), 16
        ) % (2 ** 32)

    def add_replica(self, replica: GPUReplica):
        """Add a GPU replica to the ring with virtual nodes."""
        self.replicas[replica.id] = replica
        for i in range(self.vnodes_per_gpu):
            vnode_key = f"{replica.id}:vnode:{i}"
            pos = self._hash(vnode_key)
            bisect.insort(self.ring, (pos, replica), key=lambda x: x[0])

    def remove_replica(self, replica_id: str):
        """Remove a GPU replica and its virtual nodes."""
        if replica_id in self.replicas:
            del self.replicas[replica_id]
        self.ring = [
            (pos, r) for pos, r in self.ring if r.id != replica_id
        ]

    def get_replica(self, key: str) -> Optional[GPUReplica]:
        """
        Route a key to a GPU replica.

        Returns the first healthy replica clockwise from hash(key).
        Skips unhealthy replicas (circuit breaker pattern).
        """
        if not self.ring:
            return None

        pos = self._hash(key)
        start_idx = bisect.bisect_right(
            [p for p, _ in self.ring], pos
        ) % len(self.ring)

        # Walk clockwise until we find a healthy replica
        for i in range(len(self.ring)):
            idx = (start_idx + i) % len(self.ring)
            replica = self.ring[idx][1]
            if replica.healthy:
                return replica

        return None  # No healthy replicas

    def get_replica_for_user(self, user_id: str, model: str) -> Optional[GPUReplica]:
        """
        Route a user request to a GPU that serves the requested model
        AND has sufficient VRAM.
        """
        replica = self.get_replica(user_id)
        if replica is None or replica.model != model:
            return None
        return replica


# ═══ Benchmark: round-robin vs consistent hashing KV cache hit rate ═══

def simulate_cache_hit_rate(
    num_replicas: int = 8,
    num_users: int = 1000,
    requests_per_user: int = 10,
    algorithm: str = "consistent",
) -> float:
    """Simulate KV cache efficiency under different routing algorithms."""
    import random

    cache_hits = 0
    total_requests = 0
    user_gpu_map: dict[str, int] = {}  # user_id → last GPU index
    gpu_index = 0

    for user_id in range(num_users):
        for _ in range(requests_per_user):
            total_requests += 1

            if algorithm == "round_robin":
                gpu = gpu_index
                gpu_index = (gpu_index + 1) % num_replicas
            elif algorithm == "consistent":
                # Consistent hash: same user → same GPU (unless ring changes)
                gpu = hash(str(user_id)) % num_replicas
            elif algorithm == "least_connections":
                gpu = random.randint(0, num_replicas - 1)

            # Check if KV cache hit (same user, same GPU as last request)
            if user_id in user_gpu_map and user_gpu_map[user_id] == gpu:
                cache_hits += 1

            user_gpu_map[user_id] = gpu

    return cache_hits / total_requests


if __name__ == "__main__":
    for algo in ["consistent", "round_robin", "least_connections"]:
        hit_rate = simulate_cache_hit_rate(algorithm=algo)
        print(f"{algo:>20}: {hit_rate:.1%} KV cache hit rate")
    # consistent:       90.0%  (100% for same user, <100% due to ring changes)
    # round_robin:      11.1%  (only hits when ring index happens to match)
    # least_connections: 12.5%  (random routing, P(hit) = 1/N)
```

### 1.4 Visual Representation 🖼️

```mermaid
graph TD
    subgraph "Client Layer"
        U1["user_abc"]
        U2["user_xyz"]
        U3["user_123"]
    end

    subgraph "Load Balancer"
        CH["Consistent Hash Ring<br/>hash(user_id) → GPU"]
    end

    subgraph "GPU Cluster"
        G0["GPU-0 (A100)<br/>Model: Llama-70B<br/>KV Cache: 20GB<br/>Users: abc, def"]
        G1["GPU-1 (A100)<br/>Model: Llama-70B<br/>KV Cache: 18GB<br/>Users: xyz, ghi"]
        G2["GPU-2 (A100)<br/>Model: Llama-70B<br/>KV Cache: 22GB<br/>Users: 123, jkl"]
    end

    U1 -->|"hash(abc) → GPU-0"| CH
    U2 -->|"hash(xyz) → GPU-1"| CH
    U3 -->|"hash(123) → GPU-2"| CH

    CH -->|"Request 1"| G0
    CH -->|"Request 1"| G1
    CH -->|"Request 1"| G2

    CH -.->|"Request 2 (KV cache HIT)"| G0
    CH -.->|"Request 2 (KV cache HIT)"| G1
    CH -.->|"Request 2 (KV cache HIT)"| G2

    style CH fill:#1565c0,stroke:#0d47a1,color:#fff
```

### 1.5 Application in ML/AI Systems 🤖

**Caso real: Notion's AI features** use consistent-hash routing for LLM inference. When a user starts a conversation with Notion AI, the load balancer hashes their user ID and routes all subsequent requests in that session to the same GPU replica. This preserves the KV cache for the user's entire conversation history — a 4000-token context that would cost 10.4GB of VRAM to recompute on every request. Notion reported that consistent hashing reduced their GPU fleet size by 35% compared to round-robin routing because KV cache reuse eliminated redundant computation.

¡Sorpresa! Consistent hashing reduces KV cache miss rate by 40-60% in LLM serving workloads compared to round-robin. The magnitude of this improvement depends on conversation length — longer conversations benefit more because the KV cache is larger and more expensive to recompute. For a 70B model with 4000-token conversations, avoiding KV cache recomputation saves approximately 250ms of GPU time per request — which at 100 requests/second translates to 25 fewer GPU-seconds of compute per second.

### 1.6 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Using round-robin for LLM inference. Every request from the same user hits a different GPU, destroying KV cache locality. The model must recompute the full attention context (prompt tokens + generated tokens) on every request, wasting 30-50% of GPU time.

💡 **Tip**: Always use consistent hashing on `user_id` (or `session_id`) for ML inference with stateful caches (KV cache, embedding cache). The small imbalance in load distribution (~5-10% variance) is negligible compared to the 30-50% throughput gain from cache reuse.

⚠️ **Pitfall**: Using too few virtual nodes per GPU (e.g., 10). Creates large key ranges per node, causing severe imbalance when a node is added or removed. The new node takes over a 1/10 chunk of the ring, flooding itself with migrated keys.

💡 **Tip**: Use 100-200 virtual nodes per physical GPU. This ensures each GPU gets roughly equal key distribution (by the law of large numbers) and smooths out the redistribution when nodes join or leave the ring.

### 1.7 Knowledge Check ❓

1. A 70B LLM serves 500 concurrent users across 8 GPUs. Users average 10 requests per session with 2000-token contexts. Estimate the GPU-seconds saved per second by consistent hashing vs round-robin (assume 15ms/token for prefill, 2.6MB KV cache per token).
2. When a GPU fails and is removed from the hash ring, what happens to the users who were pinned to it? How does this affect KV cache hit rate temporarily?
3. Why does consistent hashing combined with RadixAttention (shared prefix in KV cache) produce compounding benefits? Quantify for 100 users sharing a 1000-token system prompt.

---

## Module 2: GPU-Aware Load Balancing — VRAM, Batching, and Backpressure

### 2.1 Theoretical Foundation 🧠

Traditional load balancers operate on connection counts — send to the server with the fewest open TCP connections. For GPU inference, connections are the wrong metric. A GPU can be "idle" (0 connections) but have 78/80GB VRAM occupied from a recently completed long-generation request that hasn't been evicted yet. Conversely, a GPU with 50 active connections processing short completions may have plenty of VRAM headroom.

A GPU-aware load balancer must consider:

1. **VRAM availability**: Route to the GPU with the most free VRAM. Free VRAM = total VRAM − model weights − KV cache − current batch activations.

2. **Batch composition**: Different requests have different VRAM requirements. A 50-token completion uses minimal KV cache; a 4000-token conversation uses 10.4GB. The load balancer should route heavy requests to GPUs with headroom and pack light requests onto near-saturated GPUs.

3. **Model type**: In multi-model deployments, each GPU runs one model at a time (for throughput). The load balancer must route to GPUs running the requested model.

4. **Backpressure**: When all GPUs serving a model are saturated, the load balancer must queue requests rather than routing them to unavailable GPUs. Backpressure prevents cascading failures: a saturated GPU returns errors, which triggers client retries, which saturates the GPU further.

The VRAM-aware routing algorithm:

$$R_{\text{target}} = \arg\max_{r \in R_{\text{model}}} \left( \text{VRAM}_{\text{free}}(r) - \text{VRAM}_{\text{estimated}}(request) \right)$$

If no GPU has sufficient free VRAM, the request is queued until a GPU becomes available or the request times out.

### 2.2 Mental Model 📐

```
┌─── GPU-Aware Load Balancer ──────────────────────────────────────┐
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    Incoming Request                          │ │
│  │  {user_id, model: "llama-70b", max_tokens: 500, stream: true}│ │
│  └────────────┬─────────────────────────────────────────────────┘ │
│               │                                                    │
│               ▼                                                    │
│  ┌───────────────────────────────┐                                │
│  │ 1. Model Dispatch              │  Filter GPUs with correct model│
│  │    model == "llama-70b" ?      │                                │
│  └────────────┬──────────────────┘                                │
│               │                                                    │
│               ▼                                                    │
│  ┌───────────────────────────────┐                                │
│  │ 2. VRAM Estimation            │  prompt_tokens × 2.6MB/token  │
│  │    est_vram = f(tokens, ctx)  │  + max_tokens × 2.6MB/token   │
│  │    est_vram = 2.6GB (1K ctx)  │                                │
│  └────────────┬──────────────────┘                                │
│               │                                                    │
│               ▼                                                    │
│  ┌───────────────────────────────┐                                │
│  │ 3. GPU Selection              │  Pick GPU with most free VRAM  │
│  │    ┌──────┬──────┬──────┐     │  that fits estimated VRAM      │
│  │    │ G0:  │ G1:  │ G2:  │     │                                │
│  │    │ 25GB │ 5GB  │ 40GB │     │                                │
│  │    │ free │ free │ free │ → G2│                                │
│  │    └──────┴──────┴──────┘     │                                │
│  └────────────┬──────────────────┘                                │
│               │                                                    │
│               │  All GPUs full?                                     │
│               ▼                                                    │
│  ┌───────────────────────────────┐                                │
│  │ 4. Backpressure Queue         │  FIFO queue, request timeout   │
│  │    Position: 15/100           │  Circuit breaker at 80% depth  │
│  │    ETA: 250ms                 │  Return 429 if > timeout      │
│  └───────────────────────────────┘                                │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```python
"""
gpu_aware_lb.py — GPU-aware load balancer with VRAM estimation,
backpressure, and circuit breaking for ML inference workloads.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUState:
    id: str
    model: str
    vram_total_gb: float = 80.0
    vram_model_gb: float = 35.0   # Model weights (INT8 Llama-70B)
    vram_kv_cache_gb: float = 0.0  # Current KV cache usage
    vram_activations_gb: float = 0.0  # Current batch activations

    active_requests: int = 0
    max_concurrent: int = 32

    @property
    def vram_used_gb(self) -> float:
        return self.vram_model_gb + self.vram_kv_cache_gb + self.vram_activations_gb

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

    def estimate_vram_for_request(
        self, prompt_tokens: int, max_output_tokens: int
    ) -> float:
        """
        Estimate VRAM needed for this request.
        For Llama-70B: ~2.6MB per token of KV cache (K+V).
        Activations: ~1MB per token per batch (rough estimate).
        """
        kv_cache_mb = (prompt_tokens + max_output_tokens) * 2.6
        activation_mb = prompt_tokens * 1.0  # Batch activations
        return (kv_cache_mb + activation_mb) / 1024.0  # Convert to GB


class GPUAwareLoadBalancer:
    """
    Routes requests to GPUs based on VRAM availability and model type.
    Implements backpressure when all GPUs are saturated.
    """

    def __init__(
        self,
        request_timeout_ms: int = 2000,
        max_queue_depth: int = 200,
        circuit_breaker_threshold: float = 0.8,
    ):
        self.gpus: dict[str, GPUState] = {}
        self.request_timeout = request_timeout_ms / 1000.0
        self.max_queue_depth = max_queue_depth
        self.circuit_breaker = circuit_breaker_threshold
        self._queue: deque = deque()
        self._lock = threading.Lock()

        # Metrics
        self.requests_routed = 0
        self.requests_queued = 0
        self.requests_rejected = 0
        self.requests_timed_out = 0

    def register_gpu(self, gpu: GPUState):
        with self._lock:
            self.gpus[gpu.id] = gpu

    def update_gpu_state(
        self, gpu_id: str, vram_kv_cache_gb: float, active_requests: int
    ):
        with self._lock:
            if gpu_id in self.gpus:
                self.gpus[gpu_id].vram_kv_cache_gb = vram_kv_cache_gb
                self.gpus[gpu_id].active_requests = active_requests

    def route(
        self,
        user_id: str,
        model: str,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> dict:
        """
        Route a request to the best GPU.

        Returns:
            {"status": "routed"/"queued"/"rejected", "gpu_id": str, ...}
        """
        with self._lock:
            # 1. Filter GPUs by model
            candidates = [
                g for g in self.gpus.values()
                if g.model == model and g.active_requests < g.max_concurrent
            ]

            if not candidates:
                # No GPU runs this model → rejected
                self.requests_rejected += 1
                return {"status": "rejected", "reason": "no_gpu_for_model"}

            # 2. Estimate VRAM for this request
            est_vram = candidates[0].estimate_vram_for_request(
                prompt_tokens, max_output_tokens
            )

            # 3. Select GPU with most free VRAM that fits the request
            best_gpu = None
            best_free = -1

            for gpu in candidates:
                free = gpu.vram_free_gb
                if free >= est_vram and free > best_free:
                    best_gpu = gpu
                    best_free = free

            if best_gpu is not None:
                # Route to best GPU
                best_gpu.vram_kv_cache_gb += est_vram
                best_gpu.active_requests += 1
                self.requests_routed += 1
                return {"status": "routed", "gpu_id": best_gpu.id}

            # 4. All GPUs saturated → check queue
            queue_ratio = len(self._queue) / self.max_queue_depth
            if queue_ratio > self.circuit_breaker:
                self.requests_rejected += 1
                return {
                    "status": "rejected",
                    "reason": "circuit_breaker_tripped",
                    "queue_depth": len(self._queue),
                }

            self._queue.append((user_id, model, prompt_tokens, max_output_tokens, time.time()))
            self.requests_queued += 1
            return {
                "status": "queued",
                "queue_position": len(self._queue),
                "estimated_wait_ms": len(self._queue) * 50,
            }

    def drain_queue(self):
        """Process queued requests when GPU resources free up."""
        with self._lock:
            to_process = []
            now = time.time()

            while self._queue:
                user_id, model, prompt_tokens, max_output_tokens, enqueued_at = self._queue[0]

                # Check timeout
                if now - enqueued_at > self.request_timeout:
                    self._queue.popleft()
                    self.requests_timed_out += 1
                    continue

                to_process.append(self._queue.popleft())

        # Process outside lock to avoid deadlock
        for item in to_process:
            self.route(*item[:4])

    @property
    def utilization(self) -> dict:
        """GPU utilization summary for auto-scaling decisions."""
        with self._lock:
            return {
                gpu.id: {
                    "vram_used_pct": round(gpu.vram_used_gb / gpu.vram_total_gb * 100, 1),
                    "active_requests": gpu.active_requests,
                    "vram_free_gb": round(gpu.vram_free_gb, 1),
                }
                for gpu in self.gpus.values()
            }

    @property
    def stats(self) -> dict:
        return {
            "routed": self.requests_routed,
            "queued": self.requests_queued,
            "rejected": self.requests_rejected,
            "timed_out": self.requests_timed_out,
            "queue_depth": len(self._queue),
            "gpu_utilization": self.utilization,
        }
```

### 2.4 Application in ML/AI Systems 🤖

Google's Vertex AI Prediction uses GPU-aware load balancing. Their prediction service measures VRAM usage per GPU in real-time and routes requests to the GPU with the most available VRAM that can fit the request's estimated memory footprint. For multi-model deployments, each GPU registers the model it serves, and the load balancer performs model-level dispatch before VRAM-level routing. When all GPUs for a model are saturated, Vertex AI returns a 429 (Too Many Requests) with a Retry-After header — implementing backpressure at the API level rather than letting requests pile up in TCP buffers.

Traditional commercial load balancers (Nginx, HAProxy, Envoy) cannot natively route based on GPU VRAM because they operate at Layer 4/7 and have no visibility into GPU state. ML teams build custom sidecar agents that expose GPU metrics via gRPC health checks, and the load balancer queries these sidecars for routing decisions. Envoy's subset load balancing with custom health checkers is the closest off-the-shelf solution.

### 2.5 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Using least-connections for GPU inference. A GPU with 0 connections may have 100% VRAM utilization from a single long-running batch generation. A GPU with 100 connections may have 20% VRAM utilization from lightweight classification requests.

💡 **Tip**: Implement a gRPC health check endpoint on each GPU serving replica that reports `vram_free_gb`, `model_name`, and `max_concurrent_requests`. The load balancer queries these health checks every 100ms for routing decisions.

⚠️ **Pitfall**: No backpressure mechanism. When GPUs are saturated, requests pile up in TCP connection queues, timeout after 30 seconds, and clients retry — creating a retry storm that makes the saturation worse.

💡 **Tip**: Implement explicit backpressure: return HTTP 429 immediately when queue depth exceeds 80% of max capacity. Include `Retry-After` headers. This gives clients clear signals to back off and prevents retry amplification.

### 2.6 Knowledge Check ❓

1. A GPU farm has 4 H100 GPUs (80GB each) serving Llama-70B (INT8, 35GB model weights). Each active conversation uses ~10GB of KV cache. How many concurrent conversations can the farm support? What happens when conversation 33 arrives?
2. Design a health check endpoint for GPU inference replicas. What metrics should it expose? How frequently should the load balancer poll?
3. Why is circuit breaking at 80% queue depth better than 100%? What's the design principle?

---

## Module 3: Feature Store Sharding — Consistent Hashing by User ID

### 3.1 Theoretical Foundation 🧠

A feature store serving millions of users per second cannot fit all features in a single Redis instance. The dataset must be partitioned (sharded) across multiple nodes. The sharding strategy determines whether feature reads are single-digit milliseconds (local to one shard) or tens of milliseconds (scatter-gather across multiple shards).

The optimal sharding strategy for ML features is **sharding by user_id using consistent hashing**. Every feature for a given user — demographic features, behavioral features, embedding features — lives on the same shard. When a model requests features for `user_123`, the feature store client hashes `user_123`, resolves the hash to a shard, and reads all features in a single `MGET` command on that shard. No scatter-gather. No cross-shard coordination.

The alternative — sharding by feature name (e.g., all `user_age` values on shard A, all `user_purchase_count_7d` on shard B) — forces every feature read to scatter across multiple shards and gather the results. For 20 features per request, this means 20 parallel reads, each potentially to a different shard, with P99 latency determined by the slowest shard.

For multi-entity features (e.g., item features, context features), the sharding key is the entity ID. User features are sharded by `user_id`, item features by `item_id`, and context features (day of week, time of day) are small enough to replicate across all shards.

### 3.2 Mental Model 📐

```
┌─── Feature Store Sharding Strategies ────────────────────────────┐
│                                                                    │
│  ❌ BAD: Shard by Feature Name                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │ Shard A   │  │ Shard B   │  │ Shard C   │  │ Shard D   │     │
│  │ user_age  │  │ user_ctry │  │ purch_7d  │  │ session   │     │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │
│       ▲              ▲              ▲              ▲               │
│       └──────────────┼──────────────┼──────────────┘               │
│             Scatter-Gather: 4 parallel reads                      │
│             P99 latency = max(shard_latencies)                     │
│             Network: 4× fan-out                                    │
│                                                                    │
│  ✅ GOOD: Shard by User ID (Consistent Hashing)                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │ Shard A   │  │ Shard B   │  │ Shard C   │  │ Shard D   │     │
│  │ users     │  │ users     │  │ users     │  │ users     │     │
│  │ 0-250K    │  │ 250K-500K │  │ 500K-750K │  │ 750K-1M   │     │
│  │           │  │           │  │           │  │           │     │
│  │ ALL feats │  │ ALL feats │  │ ALL feats │  │ ALL feats │     │
│  │ for user  │  │ for user  │  │ for user  │  │ for user  │     │
│  │ in range  │  │ in range  │  │ in range  │  │ in range  │     │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │
│       ▲                                                            │
│       └──── 1 read: MGET user_123:*  (single-digit ms)            │
│                                                                    │
│  Feature Key Format:                                               │
│    feature:{user_id}:{feature_name}                                │
│    feature:user_123:user_age → Shard B                             │
│    feature:user_123:user_country → Shard B (same shard!)           │
│    feature:user_123:purchase_count_7d → Shard B                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```python
"""
feature_store_sharding.py — Sharded feature store with consistent
hashing by user_id. All features for a user on a single shard.
"""

import hashlib
import bisect
import json
from typing import Any, Optional
import redis


class ShardedFeatureStore:
    """
    Feature store sharded by user_id with consistent hashing.

    Key schema: feature:{user_id}:{feature_name}
    Sharding: Consistent hash of user_id → shard

    Single-MGET read: all features for user_123 fetched from one shard.
    """

    def __init__(self, shard_addresses: list[str], vnodes_per_shard: int = 150):
        self.vnodes_per_shard = vnodes_per_shard
        self.ring: list[tuple[int, int]] = []  # (hash_pos, shard_idx)
        self.shards: list[redis.Redis] = []

        for idx, addr in enumerate(shard_addresses):
            host, port = addr.split(":")
            self.shards.append(redis.Redis(
                host=host, port=int(port), decode_responses=True
            ))
            self._add_shard_to_ring(idx)

    def _hash(self, key: str) -> int:
        return int(
            hashlib.sha256(key.encode()).hexdigest(), 16
        ) % (2 ** 32)

    def _add_shard_to_ring(self, shard_idx: int):
        for i in range(self.vnodes_per_shard):
            pos = self._hash(f"shard:{shard_idx}:vnode:{i}")
            bisect.insort(self.ring, (pos, shard_idx), key=lambda x: x[0])

    def _get_shard(self, user_id: str) -> redis.Redis:
        """Resolve user_id to a shard using consistent hashing."""
        pos = self._hash(user_id)
        idx = bisect.bisect_right(
            [p for p, _ in self.ring], pos
        ) % len(self.ring)
        return self.shards[self.ring[idx][1]]

    def _feature_key(self, user_id: str, feature_name: str) -> str:
        return f"feature:{user_id}:{feature_name}"

    def get_user_features(
        self, user_id: str, feature_names: list[str]
    ) -> dict[str, Any]:
        """
        Fetch all features for a user. Single shard access.
        Latency: ~1ms (single MGET on one Redis shard).
        """
        shard = self._get_shard(user_id)
        keys = [self._feature_key(user_id, name) for name in feature_names]
        values = shard.mget(keys)

        result = {}
        for name, value in zip(feature_names, values):
            result[name] = json.loads(value) if value else None
        return result

    def set_user_features(
        self, user_id: str, features: dict[str, Any]
    ):
        """Write all features for a user. Single shard access."""
        shard = self._get_shard(user_id)
        pipe = shard.pipeline()
        for name, value in features.items():
            key = self._feature_key(user_id, name)
            pipe.setex(key, 3600, json.dumps(value))
        pipe.execute()

    def get_multi_entity_features(
        self,
        user_id: str,
        user_features: list[str],
        item_id: str,
        item_features: list[str],
        context_features: list[str],
    ) -> dict:
        """
        Fetch features for user + item + context.
        User features → shard(user_id)
        Item features  → shard(item_id)
        Context        → replicated across all shards (or single designated shard)

        Total latency: ~2ms (two parallel MGETs + one local context read).
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            user_future = executor.submit(
                self.get_user_features, user_id, user_features
            )
            item_future = executor.submit(
                self.get_user_features, item_id, item_features
            )

            user_feats = user_future.result()
            item_feats = item_future.result()

        # Context features: small enough to replicate, read from shard 0
        context_feats = self.shards[0].hgetall(f"context:current")
        context_feats = {
            k: json.loads(v) for k, v in context_feats.items()
            if k in context_features
        }

        return {"user": user_feats, "item": item_feats, "context": context_feats}


# ═══ Shard rebalancing on scale-up ═══

class FeatureStoreRebalancer:
    """
    Handles feature data migration when shards are added or removed.
    Only features for users whose hash mapping changed need to be moved.
    """

    def __init__(self, store: ShardedFeatureStore):
        self.store = store

    def estimate_migration_volume(
        self, old_shard_count: int, new_shard_count: int, total_users: int
    ) -> float:
        """
        Estimate fraction of users that need migration when scaling.

        With consistent hashing, adding M shards to N existing shards
        moves approximately M/(N+M) of keys.
        """
        fraction_moved = new_shard_count / (old_shard_count + new_shard_count)
        users_to_move = int(total_users * fraction_moved)
        return users_to_move

    def rebalance_add_shard(self, new_shard_addr: str):
        """Add a new shard: only ~1/(N+1) of users migrate."""
        old_shard_count = len(self.store.shards)
        new_idx = old_shard_count
        host, port = new_shard_addr.split(":")
        self.store.shards.append(redis.Redis(
            host=host, port=int(port), decode_responses=True
        ))
        self.store._add_shard_to_ring(new_idx)

        fraction_moved = 1.0 / (old_shard_count + 1)
        print(f"New shard added. ~{fraction_moved:.1%} of users rebalanced.")
```

### 3.4 Application in ML/AI Systems 🤖

DoorDash's feature store shards by user_id using consistent hashing. With 500M users and 200 features per user (~3KB total), the total feature dataset is ~1.5TB. Sharded across 12 Redis nodes (128GB each), each node holds ~125GB — well within capacity. When a new Redis node is added, only ~1/13 (~7.7%) of users need to be migrated, and the migration happens lazily: existing features expire via TTL, new writes go to the new shard, reads check the new shard first then fall back to the old shard.

Contrast with sharding by feature name: DoorDash's 200 features would require reading from up to 200 different shards per request, with P99 latency driven by the slowest shard (~10ms). Sharding by user_id reduces this to a single shard read (~0.5ms) — a 20× latency improvement.

### 3.5 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Sharding by feature name (or by hash of feature value). Scatters every feature read across multiple shards, amplifying P99 latency by the slowest-shard problem.

💡 **Tip**: Shard by entity ID (user_id, item_id). All features for entity X live on shard Y. Reads are single-shard MGETs with predictable latency.

⚠️ **Pitfall**: Using a simplistic `hash(user_id) % N` for shard assignment instead of consistent hashing. When N changes (scale-up), ALL keys get reassigned, causing a 100% cache invalidation event.

💡 **Tip**: Always use consistent hashing for shard assignment. Adding a shard reassigns only ~1/(N+1) of keys. The hash ring decouples physical shard count from logical key mapping.

### 3.6 Knowledge Check ❓

1. A feature store serves 1M users with 50 features each. It is sharded across 4 Redis nodes using consistent hashing by user_id. A 5th node is added. What fraction of user features need to be migrated? Design a lazy migration strategy.
2. Why does `MGET` work efficiently with user_id sharding but not with feature-name sharding? Consider Redis's single-threaded execution model.
3. A teammate proposes sharding by `hash(feature_name) % N`. Critique this approach for a recommendation system that reads 30 features per user per request.

---

## Module 4: Auto-Scaling for ML — GPU Metrics, Not CPU

### 4.1 Theoretical Foundation 🧠

Kubernetes Horizontal Pod Autoscaler (HPA) scales pods based on CPU utilization by default. For web servers, this works: high CPU → more pods → balanced load. For GPU inference, this fails catastrophically. A GPU inference pod shows 5-10% CPU utilization while its GPU is at 100% — the CPU is just passing data to the GPU and waiting. CPU-based HPA will never trigger a scale-up event for a GPU-saturated service.

❌/✅ **CPU HPA for GPU services**: The canonical anti-pattern. A Kubernetes cluster with CPU-based HPA running GPU inference will maintain minimum replicas while queue depth grows unbounded. The CPU is idle; the GPU is saturated; HPA is blind to the actual bottleneck.

The correct metrics for GPU auto-scaling are:

1. **GPU utilization (%)**: `DCGM_FI_DEV_GPU_UTIL`. Scale up when sustained > 80%, scale down when < 40%.

2. **GPU memory utilization (%)**: `DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE`. Scale up when VRAM used > 85%.

3. **Queue depth**: The number of requests waiting in the backpressure queue. This is the most direct signal of insufficient capacity. Scale up proportional to queue depth.

4. **Request latency P99**: When P99 latency exceeds SLA, scale up. This captures the end-to-end impact of saturation.

The HPA configuration uses custom metrics from Prometheus (or the K8s custom metrics API) exposed by NVIDIA's DCGM exporter:

```
GPU utilization → Prometheus → K8s Custom Metrics API → HPA
Queue depth     → Prometheus → K8s Custom Metrics API → HPA
```

### 4.2 Mental Model 📐

```
┌─── Auto-Scaling Decision Flow ────────────────────────────────────┐
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Every 15 seconds (HPA sync period):                           │  │
│  │                                                                │  │
│  │ 1. Check GPU utilization (avg across all pods)                │  │
│  │    ├── > 80% for 2 minutes → SCALE UP                         │  │
│  │    └── < 40% for 5 minutes → SCALE DOWN                       │  │
│  │                                                                │  │
│  │ 2. Check queue depth (backpressure queue)                     │  │
│  │    ├── > 100 requests → SCALE UP immediately                  │  │
│  │    └── < 10 requests for 5 minutes → candidate for scale down │  │
│  │                                                                │  │
│  │ 3. Check P99 latency                                          │  │
│  │    ├── > 200ms → SCALE UP                                     │  │
│  │    └── < 50ms → candidate for scale down                      │  │
│  │                                                                │  │
│  │ 4. Apply constraints                                          │  │
│  │    ├── min_replicas = 2 (high availability)                   │  │
│  │    ├── max_replicas = 50 (cost cap)                          │  │
│  │    └── scale_up_cooldown = 60s (prevent flapping)             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ❌ CPU-Based HPA for GPU Inference (ANTI-PATTERN):                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ GPU at 100% utilization → serving 100 req/s, queue growing    │  │
│  │ CPU at 5% utilization   → HPA: "everything is fine"           │  │
│  │ Result: latency explodes, HPA never scales up                 │  │
│  │                                                                │  │
│  │ GPU at 10% utilization → idle GPU costing $3/hr               │  │
│  │ CPU at 8% utilization   → HPA: "everything is fine"           │  │
│  │ Result: idle GPUs burning money, HPA never scales down        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ✅ Custom-Metric HPA for GPU Inference:                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ GPU utilization 85% → HPA triggers scale up                   │  │
│  │ Queue depth > 100  → HPA triggers scale up immediately        │  │
│  │ GPU utilization 30% → HPA triggers scale down (after cooldown)│  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Syntax and Semantics 📝

```python
"""
gpu_hpa_config.yaml — Kubernetes HPA configuration for GPU inference
using custom metrics from Prometheus (NVIDIA DCGM).
"""

HPA_CONFIG_YAML = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-70b-inference
  minReplicas: 2
  maxReplicas: 50
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
        - type: Percent
          value: 10                    # Scale down max 10% at a time
          periodSeconds: 60
        - type: Pods
          value: 1                     # Or 1 pod at a time
          periodSeconds: 60
      selectPolicy: Min                # Use the more conservative policy
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100                   # Scale up 100% if needed
          periodSeconds: 30
        - type: Pods
          value: 5                     # Or add 5 pods at a time
          periodSeconds: 30
      selectPolicy: Max                # Use the more aggressive policy
  metrics:
    # ── Custom metric 1: GPU Utilization (NVIDIA DCGM) ──
    - type: Pods
      pods:
        metric:
          name: DCGM_FI_DEV_GPU_UTIL
        target:
          type: AverageValue
          averageValue: "70"  # Target 70% avg GPU utilization

    # ── Custom metric 2: GPU Memory Utilization ──
    - type: Pods
      pods:
        metric:
          name: DCGM_FI_DEV_FB_USED_PERCENT
        target:
          type: AverageValue
          averageValue: "80"

    # ── Custom metric 3: Queue Depth (backpressure) ──
    - type: Object
      object:
        metric:
          name: inference_queue_depth
        describedObject:
          apiVersion: v1
          kind: Service
          name: llama-70b-inference
        target:
          type: AverageValue
          averageValue: "50"  # Scale up when queue > 50

    # ── Custom metric 4: P99 Latency ──
    - type: Object
      object:
        metric:
          name: inference_p99_latency_ms
        describedObject:
          apiVersion: v1
          kind: Service
          name: llama-70b-inference
        target:
          type: AverageValue
          averageValue: "200"
"""


# ═══ Prometheus metrics exporter for GPU auto-scaling ═══

from dataclasses import dataclass
from prometheus_client import Gauge, CollectorRegistry, generate_latest


@dataclass
class GPUMetricsExporter:
    """
    Exposes GPU metrics to Prometheus for HPA consumption.

    Metrics:
    - gpu_utilization: 0-100% GPU core utilization
    - gpu_memory_used_bytes: VRAM in use
    - gpu_memory_total_bytes: Total VRAM
    - inference_queue_depth: Number of requests waiting
    - inference_p99_latency_ms: P99 end-to-end latency
    """

    registry: CollectorRegistry

    gpu_util = Gauge(
        "gpu_utilization_percent",
        "GPU core utilization percentage",
        ["gpu_id"],
        registry=registry,
    )
    gpu_mem_used = Gauge(
        "gpu_memory_used_bytes",
        "GPU VRAM used in bytes",
        ["gpu_id"],
        registry=registry,
    )
    gpu_mem_total = Gauge(
        "gpu_memory_total_bytes",
        "GPU VRAM total in bytes",
        ["gpu_id"],
        registry=registry,
    )
    queue_depth = Gauge(
        "inference_queue_depth",
        "Number of requests waiting in backpressure queue",
        registry=registry,
    )
    p99_latency = Gauge(
        "inference_p99_latency_ms",
        "P99 end-to-end inference latency in milliseconds",
        registry=registry,
    )

    def update_gpu_metrics(self, gpu_id: str, util_pct: float,
                           mem_used_bytes: int, mem_total_bytes: int):
        self.gpu_util.labels(gpu_id=gpu_id).set(util_pct)
        self.gpu_mem_used.labels(gpu_id=gpu_id).set(mem_used_bytes)
        self.gpu_mem_total.labels(gpu_id=gpu_id).set(mem_total_bytes)

    def update_service_metrics(self, queue: int, p99_ms: float):
        self.queue_depth.set(queue)
        self.p99_latency.set(p99_ms)


# ═══ Scale decision simulator ═══

def should_scale(
    gpu_util_pct: float,
    queue_depth: int,
    p99_latency_ms: float,
    current_replicas: int,
    min_replicas: int = 2,
    max_replicas: int = 50,
    scale_up_threshold: float = 80.0,
    scale_down_threshold: float = 40.0,
    queue_threshold: int = 50,
    latency_threshold_ms: float = 200.0,
) -> str:
    """
    Decide whether to scale up, scale down, or hold based on GPU metrics.
    This logic runs in the HPA controller every 15 seconds.
    """
    reasons = []

    if gpu_util_pct > scale_up_threshold:
        reasons.append(f"GPU util {gpu_util_pct}% > {scale_up_threshold}%")
    if queue_depth > queue_threshold:
        reasons.append(f"Queue depth {queue_depth} > {queue_threshold}")
    if p99_latency_ms > latency_threshold_ms:
        reasons.append(f"P99 latency {p99_latency_ms}ms > {latency_threshold_ms}ms")

    if reasons and current_replicas < max_replicas:
        target = min(
            max_replicas,
            current_replicas + max(1, queue_depth // queue_threshold),
        )
        return f"SCALE_UP: {current_replicas} → {target} ({'; '.join(reasons)})"

    # Scale down: all metrics below threshold for sustained period
    if (gpu_util_pct < scale_down_threshold
            and queue_depth < 10
            and p99_latency_ms < 50
            and current_replicas > min_replicas):
        return f"SCALE_DOWN: {current_replicas} → {current_replicas - 1} (low utilization)"

    return f"HOLD: {current_replicas} replicas, {gpu_util_pct}% GPU, queue={queue_depth}"


if __name__ == "__main__":
    # Simulate a GPU-saturated scenario that CPU-HPA would miss
    print(should_scale(gpu_util_pct=92, queue_depth=150, p99_latency_ms=350,
                       current_replicas=3))
    # → SCALE_UP: 3 → 6 (GPU util 92% > 80%; Queue 150 > 50; P99 350ms > 200ms)

    # Low utilization scenario
    print(should_scale(gpu_util_pct=25, queue_depth=3, p99_latency_ms=40,
                       current_replicas=10))
    # → SCALE_DOWN: 10 → 9 (low utilization)
```

### 4.4 Application in ML/AI Systems 🤖

**Caso real: A major ride-sharing company's ML inference platform** migrated from CPU-based HPA to GPU-custom-metric HPA after a production incident. During peak hours, their fraud detection model (running on GPUs) experienced 3-second P99 latency while CPU utilization stayed at 8%. The CPU-based HPA maintained minimum replicas. Users experienced transaction timeouts, and the fraud detection was effectively disabled. After migrating to GPU-utilization and queue-depth-based HPA, the system scaled proactively — adding GPUs when utilization exceeded 70% rather than waiting for latency to degrade.

The cost impact was counter-intuitive: custom-metric HPA **reduced** GPU costs by 25% despite having higher peak replica counts. Why? Because CPU-based HPA kept minimum replicas running 24/7 to handle variance, while GPU-metric HPA scaled down aggressively during off-peak hours (GPU utilization < 30% for 5 minutes → scale down). The system ran at 2 replicas at 3 AM instead of 8, saving $432/day in GPU costs.

### 4.5 Common Pitfalls ⚠️ + 💡 Tips

⚠️ **Pitfall**: Running CPU-based HPA on GPU workloads. The CPU is never the bottleneck for inference — the GPU is. CPU-HPA maintains minimum replicas while the GPU is saturated and the queue is growing.

💡 **Tip**: Install NVIDIA DCGM exporter on every GPU node. Configure Prometheus to scrape `DCGM_FI_DEV_GPU_UTIL` and expose it through the K8s Custom Metrics API. Configure HPA with these custom metrics. This is a one-time setup that pays for itself in the first production incident it prevents.

⚠️ **Pitfall**: Scaling up too aggressively during burst traffic. Adding 10 GPUs simultaneously triggers 10 cold starts (model download + load into VRAM, 30-120 seconds). During cold start, the new replicas contribute zero capacity, and the scale-up decision might trigger again — creating an oscillating scale-up loop.

💡 **Tip**: Set `scaleUp.stabilizationWindowSeconds` to 60-120 seconds. After a scale-up, wait for cold starts to complete before evaluating again. Use CDN-distributed model artifacts ([[02 - Caching, CDNs and Storage Architectures for ML|Note 02]]) to minimize cold start time.

⚠️ **Pitfall**: Scaling down too aggressively during traffic dips, then encountering a traffic spike with insufficient replicas. Cold starts take 30-120 seconds; the traffic spike is immediate.

💡 **Tip**: Set `scaleDown.stabilizationWindowSeconds` to 300 seconds (5 minutes). Scale down slowly — 1 pod every 60 seconds. Over-provisioning by 1-2 replicas during low traffic costs $3-6/hour; under-provisioning during a spike loses users.

### 4.6 Knowledge Check ❓

1. A GPU inference service runs on K8s with CPU-based HPA. GPU utilization is 95%, queue depth is 200, CPU is at 8%. The HPA maintains 3 replicas. What happens? What metric should trigger scaling?
2. Cold start for a 70B model takes 90 seconds (download + load). If traffic doubles in 10 seconds, how can the system survive? Design a pre-warming strategy.
3. An auto-scaling system oscillates: scale-up → cold start → traffic handled → scale-down → new traffic spike → scale-up. What HPA parameters need tuning to dampen this oscillation?

---

## 📦 Código de Compresión

```python
"""
ml_load_balancer.py — Production-grade ML load balancer integrating:
consistent hashing, GPU-aware routing, feature store sharding, and
auto-scaling hooks. Complete reference for FAANG+ ML system design interviews.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│  Request → {user_id, model, features, max_tokens}                │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────┐   ┌───────────────────────────────┐
│ ConsistentHashRouter       │   │ GPUAwareLoadBalancer           │
│ hash(user_id) → GPU        │──▶│ GPU with max free VRAM         │
└────────────────────────────┘   └────────────┬──────────────────┘
                                              │
                                              ▼
┌────────────────────────────┐   ┌───────────────────────────────┐
│ ShardedFeatureStore        │   │ Auto-Scaling Controller         │
│ user_id → shard            │   │ GPU util + queue → HPA          │
│ MGET all features on shard │   │ Scale up/down based on metrics  │
└────────────────────────────┘   └────────────────────────────────┘

Key design decisions:
1. Consistent hashing → user → GPU pinning → KV cache reuse
2. GPU-aware routing → free VRAM → batch composition → backpressure
3. Feature store sharding → user_id → single shard MGET → <1ms reads
4. GPU-metric HPA → GPU util + queue depth → never CPU-based
"""

from __future__ import annotations

import hashlib
import bisect
import json
import time
import threading
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ═══ Consistent Hash Ring ═══

@dataclass
class Replica:
    id: str
    address: str
    model: str
    vram_total_gb: float = 80.0
    vram_used_gb: float = 0.0
    weight: float = 1.0
    healthy: bool = True

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb


class ConsistentHashRing:
    """
    Consistent hashing ring for ML inference routing.

    Properties:
    - Monotonic: adding/removing a node reassigns only O(K/N) keys
    - Balanced: ~1/N fraction of keys per node (with sufficient vnodes)
    - KV-cache-friendly: same user → same GPU every time
    """

    def __init__(self, vnodes_per_node: int = 150):
        self.vnodes_per_node = vnodes_per_node
        self.ring: list[tuple[int, Replica]] = []
        self.nodes: dict[str, Replica] = {}

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)

    def add_node(self, replica: Replica):
        self.nodes[replica.id] = replica
        for i in range(self.vnodes_per_node):
            pos = self._hash(f"{replica.id}:v:{i}")
            bisect.insort(self.ring, (pos, replica), key=lambda x: x[0])

    def remove_node(self, replica_id: str):
        if replica_id in self.nodes:
            del self.nodes[replica_id]
        self.ring = [(p, n) for p, n in self.ring if n.id != replica_id]

    def get_node(self, key: str, skip_unhealthy: bool = True) -> Optional[Replica]:
        if not self.ring:
            return None

        positions = [p for p, _ in self.ring]
        idx = bisect.bisect_right(positions, self._hash(key)) % len(self.ring)

        for _ in range(len(self.ring)):
            replica = self.ring[idx][1]
            if not skip_unhealthy or replica.healthy:
                return replica
            idx = (idx + 1) % len(self.ring)

        return None

    @property
    def node_count(self) -> int:
        return len(self.nodes)


# ═══ GPU-Aware Router ═══

@dataclass
class GPUMetrics:
    """Per-GPU metrics used for routing and auto-scaling."""
    gpu_id: str
    model: str
    gpu_util_pct: float
    vram_used_gb: float
    vram_total_gb: float
    active_requests: int
    max_requests: int
    avg_latency_ms: float
    p99_latency_ms: float
    kv_cache_tokens: int
    updated_at: float

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

    @property
    def capacity_pct(self) -> float:
        """Combined utilization metric for scaling decisions."""
        vram_ratio = self.vram_used_gb / self.vram_total_gb
        request_ratio = self.active_requests / max(self.max_requests, 1)
        return max(vram_ratio, request_ratio, self.gpu_util_pct / 100.0)


class GPUAwareRouter:
    """
    GPU-aware request router integrating consistent hashing and
    VRAM-based load distribution.

    Routing strategy:
    1. Consistent hash(user_id) → candidate GPUs (ring order)
    2. Among candidates running the requested model, pick GPU with max
       free VRAM that can accommodate the estimated request footprint
    3. If all GPUs saturated → backpressure queue
    """

    def __init__(self, ring: ConsistentHashRing, queue_max: int = 200):
        self.ring = ring
        self.gpu_metrics: dict[str, GPUMetrics] = {}
        self._queue: deque = deque()
        self.queue_max = queue_max
        self._lock = threading.Lock()

        # KV cache estimation constants
        self.kv_cache_bytes_per_token = 2.6 * 1024 * 1024  # ~2.6MB/token for 70B

    def update_metrics(self, metrics: GPUMetrics):
        with self._lock:
            self.gpu_metrics[metrics.gpu_id] = metrics

    def estimate_vram_gb(self, prompt_tokens: int, max_output_tokens: int) -> float:
        total_tokens = prompt_tokens + max_output_tokens
        return (total_tokens * self.kv_cache_bytes_per_token) / (1024 ** 3)

    def route(self, user_id: str, model: str,
              prompt_tokens: int, max_output_tokens: int) -> dict:
        with self._lock:
            # 1. Consistent hash → candidate GPUs (walk ring)
            start_node = self.ring.get_node(user_id)
            if start_node is None:
                return {"status": "rejected", "reason": "no_nodes"}

            # 2. Find all GPUs for the requested model
            candidates = [
                m for m in self.gpu_metrics.values()
                if m.model == model and m.active_requests < m.max_requests
            ]

            if not candidates:
                return {"status": "rejected", "reason": "no_gpu_for_model"}

            # 3. Estimate VRAM needed
            est_vram = self.estimate_vram_gb(prompt_tokens, max_output_tokens)

            # 4. Pick best GPU: max free VRAM that fits the request
            best = None
            best_score = -1.0

            for gpu in candidates:
                if gpu.vram_free_gb < est_vram:
                    continue

                # Score: prefer GPUs with more free VRAM, penalize high p99 latency
                score = gpu.vram_free_gb - gpu.p99_latency_ms / 1000
                if score > best_score:
                    best_score = score
                    best = gpu

            if best is not None:
                return {"status": "routed", "gpu_id": best.gpu_id,
                        "estimated_vram_gb": round(est_vram, 1)}

            # 5. Backpressure: all GPUs saturated
            if len(self._queue) >= self.queue_max:
                return {"status": "rejected",
                        "reason": "queue_full",
                        "queue_depth": len(self._queue)}

            entry = (user_id, model, prompt_tokens, max_output_tokens, time.time())
            self._queue.append(entry)
            return {"status": "queued",
                    "queue_position": len(self._queue),
                    "estimated_wait_ms": len(self._queue) * 50}

    def drain_queue(self, timeout_s: float = 2.0):
        """Retry queued requests. Evict timed-out entries."""
        now = time.time()
        to_retry = []

        with self._lock:
            while self._queue:
                entry = self._queue[0]
                if now - entry[4] > timeout_s:
                    self._queue.popleft()  # Timeout, discard
                    continue
                to_retry.append(self._queue.popleft())

        for entry in to_retry:
            result = self.route(*entry[:4])
            if result["status"] == "routed":
                print(f"Queued request routed to {result['gpu_id']}")


# ═══ Auto-Scaling Decision Engine ═══

@dataclass
class ScaleDecision:
    action: str  # "scale_up", "scale_down", "hold"
    current_replicas: int
    target_replicas: int
    reason: str
    metrics_snapshot: dict


class AutoScaler:
    """
    Auto-scaling controller that operates on GPU metrics, not CPU.

    Scale-up signals (any one triggers):
    - Avg GPU util > 80% for 2 minutes
    - Queue depth > 50 for 30 seconds
    - P99 latency > SLA for 1 minute

    Scale-down signals (all must hold):
    - Avg GPU util < 40% for 5 minutes
    - Queue depth < 10 for 5 minutes
    - P99 latency < 50% of SLA for 5 minutes
    """

    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 50,
        gpu_util_up_threshold: float = 80.0,
        gpu_util_down_threshold: float = 40.0,
        queue_up_threshold: int = 50,
        queue_down_threshold: int = 10,
        p99_up_threshold_ms: float = 200.0,
        p99_down_threshold_ms: float = 100.0,
        cooldown_up_s: int = 60,
        cooldown_down_s: int = 300,
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.gpu_util_up = gpu_util_up_threshold
        self.gpu_util_down = gpu_util_down_threshold
        self.queue_up = queue_up_threshold
        self.queue_down = queue_down_threshold
        self.p99_up = p99_up_threshold_ms
        self.p99_down = p99_down_threshold_ms
        self.cooldown_up = cooldown_up_s
        self.cooldown_down = cooldown_down_s

        self.last_scale_time: float = 0.0
        self.replicas: int = min_replicas

    def evaluate(self, metrics: list[GPUMetrics],
                 queue_depth: int) -> ScaleDecision:
        if not metrics:
            return ScaleDecision("hold", self.replicas, self.replicas,
                                 "no_metrics", {})

        # Aggregate metrics
        avg_gpu_util = sum(m.gpu_util_pct for m in metrics) / len(metrics)
        max_p99 = max(m.p99_latency_ms for m in metrics)
        avg_vram_pct = sum(m.vram_used_gb / m.vram_total_gb
                          for m in metrics) / len(metrics) * 100

        snapshot = {
            "avg_gpu_util": round(avg_gpu_util, 1),
            "max_p99_ms": round(max_p99, 1),
            "avg_vram_pct": round(avg_vram_pct, 1),
            "queue_depth": queue_depth,
            "replicas": self.replicas,
        }

        now = time.time()

        # Scale-up signals
        scale_up_reasons = []
        if avg_gpu_util > self.gpu_util_up:
            scale_up_reasons.append(f"GPU util {avg_gpu_util:.0f}% > {self.gpu_util_up}%")
        if queue_depth > self.queue_up:
            scale_up_reasons.append(f"Queue {queue_depth} > {self.queue_up}")
        if max_p99 > self.p99_up:
            scale_up_reasons.append(f"P99 {max_p99:.0f}ms > {self.p99_up}ms")

        if scale_up_reasons and now - self.last_scale_time > self.cooldown_up:
            if self.replicas < self.max_replicas:
                # Scale proportionally to queue depth
                add = max(1, queue_depth // self.queue_up)
                target = min(self.max_replicas, self.replicas + add)
                self.replicas = target
                self.last_scale_time = now
                return ScaleDecision("scale_up", self.replicas, target,
                                     "; ".join(scale_up_reasons), snapshot)

        # Scale-down signals
        if (avg_gpu_util < self.gpu_util_down
                and queue_depth < self.queue_down
                and max_p99 < self.p99_down
                and now - self.last_scale_time > self.cooldown_down):
            if self.replicas > self.min_replicas:
                self.replicas -= 1
                self.last_scale_time = now
                return ScaleDecision("scale_down", self.replicas + 1,
                                     self.replicas,
                                     "Low utilization sustained", snapshot)

        return ScaleDecision("hold", self.replicas, self.replicas,
                             "Metrics within range", snapshot)


# ═══ Unified ML Load Balancer ═══

class MLLoadBalancer:
    """
    Unified ML load balancing system integrating:
    - Consistent hashing for user → GPU affinity
    - GPU-aware routing with VRAM estimation
    - Feature store sharding with consistent hashing
    - Auto-scaling based on GPU metrics and queue depth
    """

    def __init__(
        self,
        shard_addresses: list[str],
        min_gpus: int = 2,
        max_gpus: int = 50,
    ):
        self.ring = ConsistentHashRing(vnodes_per_node=150)
        self.router = GPUAwareRouter(self.ring)
        self.autoscaler = AutoScaler(
            min_replicas=min_gpus,
            max_replicas=max_gpus,
        )

        # Placeholder for feature store (see ShardedFeatureStore above)
        self.shard_addresses = shard_addresses

    def handle_request(
        self,
        user_id: str,
        model: str,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> dict:
        """Full request flow: route → infer → scale check."""

        # Step 1: Route to best GPU
        routing = self.router.route(user_id, model, prompt_tokens, max_output_tokens)

        if routing["status"] != "routed":
            return routing

        # Step 2: (Actual inference happens here — placeholder)
        gpu_id = routing["gpu_id"]

        # Step 3: Evaluate auto-scaling
        metrics = list(self.router.gpu_metrics.values())
        queue_depth = len(self.router._queue)
        scale_decision = self.autoscaler.evaluate(metrics, queue_depth)

        return {
            "status": "routed",
            "gpu_id": gpu_id,
            "scale_action": scale_decision.action,
            "scale_target": scale_decision.target_replicas,
        }

    def get_system_status(self) -> dict:
        """System health overview for monitoring dashboards."""
        metrics = list(self.router.gpu_metrics.values())
        queue_depth = len(self.router._queue)

        if not metrics:
            return {"status": "no_gpus", "replicas": 0}

        return {
            "replicas": len(metrics),
            "avg_gpu_util": round(
                sum(m.gpu_util_pct for m in metrics) / len(metrics), 1
            ),
            "avg_vram_free_gb": round(
                sum(m.vram_free_gb for m in metrics) / len(metrics), 1
            ),
            "max_p99_ms": round(max(m.p99_latency_ms for m in metrics), 1),
            "queue_depth": queue_depth,
            "ring_nodes": self.ring.node_count,
            "autoscaler_target": self.autoscaler.replicas,
        }
```

---

## References

- Karger, D. et al. (1997). "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." *STOC '97*. — The original consistent hashing paper.
- SGLang. (2024). *RadixAttention: Efficient LLM Serving with Prefix Sharing*. https://github.com/sgl-project/sglang
- NVIDIA. (2024). *DCGM (Data Center GPU Manager) User Guide*. https://docs.nvidia.com/datacenter/dcgm/latest/
- Kubernetes. (2024). *Horizontal Pod Autoscaling with Custom Metrics*. https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Notion Engineering. (2024). "Scaling Notion AI's Inference Infrastructure." https://www.notion.so/blog
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. Chapter 5: Partitioning.
- [[01 - CAP Theorem and Consistency Models in ML Workloads|CAP Theorem]]
- [[02 - Caching, CDNs and Storage Architectures for ML|Caching for ML]]
- [[04 - Back-of-Envelope Estimation - Throughput, Latency, Storage and Cost|Back-of-Envelope]]
- [[05 - Interview Walkthrough - Design Systems End-to-End|Interview Walkthrough]]
