# ⚖️ Service Discovery and Load Balancing

## 🎯 Learning Objectives

- Understand client-side vs server-side service discovery and their trade-offs.
- Master load balancing algorithms: round-robin, least-connections, consistent hashing.
- Implement circuit breakers, retries, and rate limiters in Go.
- Analyze service mesh architectures for resilient ML inference pipelines.

## Introduction

In cloud-native architectures, services are ephemeral. Containers spin up and down, and auto-scaling groups resize fleets dynamically. Service discovery and load balancing allow distributed components to find each other and share work efficiently. Without them, microservices collapse under hardcoded endpoints and single points of failure.

For ML/AI systems, these patterns are existential requirements. Inference services run on heterogeneous GPU clusters, model serving requires session affinity for embedding caches, and training pipelines orchestrate thousands of transient workers. A failed model server behind a stale DNS cache silently degrades recommendations for millions. Understanding these patterns enables ML engineers to build fault-tolerant, high-performance inference pipelines.

This module explores theoretical foundations, implements production-ready patterns in Go, and examines how Uber handles millions of discovery lookups per second. Connections to [[05 - Infrastructure as Code with Pulumi|🏗️ 05 - IaC with Pulumi]] and [[06 - Cloud Networking and Observability|📡 06 - Observability]] are essential for deploying and monitoring these patterns.

## Module 1: Service Discovery

### 1.1 Theoretical Foundation 🧠

Service discovery decouples consumers from provider locations. DNS (1983) was the first large-scale mechanism. In distributed systems theory, discovery is a **name service problem** governed by the CAP theorem: registries must choose consistency (all clients see the same list) or availability (clients always read a list, even if stale). Client-side discovery favors availability; server-side discovery enforces stronger consistency. Modern registries like Consul and etcd use Raft consensus. Kubernetes DNS provides a declarative, controller-based registry where Services aggregate Pod endpoints automatically.

### 1.2 Mental Model 📐

Client-side discovery:

```
┌─────────────────────────────────────────┐
│           Client Application            │
│  ┌─────────────┐    ┌───────────────┐  │
│  │ Service     │───>│ Local Cache   │  │
│  │ Registry    │    │ (with TTL)    │  │
│  │ Client      │    └───────────────┘  │
│  │ (Consul)    │           │           │
│  └─────────────┘           ▼           │
│                      ┌───────────────┐ │
│                      │ Load Balancer │ │
│                      │ (Round Robin) │ │
│                      └───────┬───────┘ │
└──────────────────────────────┼─────────┘
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ Backend │          │ Backend │          │ Backend │
   │  :8081  │          │  :8082  │          │  :8083  │
   └─────────┘          └─────────┘          └─────────┘
```

Server-side discovery:

```
┌─────────┐     ┌─────────────────────────────┐     ┌─────────┐
│ Client  │────>│     Load Balancer / Proxy   │────>│ Healthy │
│         │     │  (Nginx, Envoy, HAProxy)    │     │ Backend │
└─────────┘     │         ┌─────────┐         │     └─────────┘
                │         │ Service │         │     ┌─────────┐
                │         │ Registry│         │────>│ Healthy │
                │         │ (Watch) │         │     │ Backend │
                │         └─────────┘         │     └─────────┘
                └─────────────────────────────┘           │
                                                          ▼
                                                    ┌─────────┐
                                                    │  Failed │
                                                    │ Backend │
                                                    └─────────┘
```

Kubernetes DNS hybrid:

```
┌─────────┐      ┌──────────────┐      ┌─────────────┐
│   Pod   │─────>│ kube-dns     │─────>│  Service    │
│ (Client)│      │ (CoreDNS)    │      │  ClusterIP  │
└─────────┘      └──────────────┘      └──────┬──────┘
              ┌───────────────────────────────┼────────────────────┐
              ▼                               ▼                    ▼
         ┌─────────┐                    ┌─────────┐         ┌─────────┐
         │ Pod IP  │                    │ Pod IP  │         │ Pod IP  │
         │ 10.0.1.5│                    │ 10.0.1.6│         │ 10.0.1.7│
         └─────────┘                    └─────────┘         └─────────┘
```

### 1.3 Syntax and Semantics 📝

Go program demonstrating Consul client-side discovery with TTL-based caching.

```go
package main

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	consul "github.com/hashicorp/consul/api"
)

// WHY: Stale endpoints cause black-holes; TTL forces refresh.
type ServiceCache struct {
	mu        sync.RWMutex
	endpoints []string
	expires   time.Time
	ttl       time.Duration
}

func (c *ServiceCache) Refresh(client *consul.Client, service string) error {
	catalog := client.Catalog()
	services, _, err := catalog.Service(service, "", nil)
	if err != nil {
		return err
	}
	var eps []string
	for _, s := range services {
		eps = append(eps, fmt.Sprintf("http://%s:%d", s.ServiceAddress, s.ServicePort))
	}
	c.mu.Lock()
	c.endpoints = eps
	c.expires = time.Now().Add(c.ttl)
	c.mu.Unlock()
	return nil
}

// WHY: Lazy refresh avoids thundering herd on cache expiry.
func (c *ServiceCache) Get(client *consul.Client, service string) ([]string, error) {
	c.mu.RLock()
	valid := time.Now().Before(c.expires) && len(c.endpoints) > 0
	out := make([]string, len(c.endpoints))
	copy(out, c.endpoints)
	c.mu.RUnlock()
	if valid {
		return out, nil
	}
	if err := c.Refresh(client, service); err != nil {
		return nil, err
	}
	c.mu.RLock()
	out = make([]string, len(c.endpoints))
	copy(out, c.endpoints)
	c.mu.RUnlock()
	return out, nil
}

func main() {
	client, _ := consul.NewClient(consul.DefaultConfig())
	cache := &ServiceCache{ttl: 30 * time.Second}
	eps, _ := cache.Get(client, "ml-inference")
	fmt.Println("Discovered:", eps)
	// WHY: Background refresher hides latency for hot services.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		t := time.NewTicker(30 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				_ = cache.Refresh(client, "ml-inference")
			case <-ctx.Done():
				return
			}
		}
	}()
	http.ListenAndServe(":8080", nil)
}
```

### 1.4 Visual Representation 🖼️

```mermaid
graph TB
	subgraph "Client-Side"
		C[Go Client] -->|Query| R[Consul Registry]
		R -->|Endpoints| Cache[(Local Cache)]
		Cache -->|Select| LB[Client LB]
		LB -->|Request| B1[Backend A]
		LB -->|Request| B2[Backend B]
	end
	subgraph "Server-Side"
		C2[Go Client] -->|Request| P[Envoy Proxy]
		P -->|Lookup| R2[Service Registry]
		R2 -->|Healthy List| P
		P -->|Forward| B3[Backend C]
		P -->|Forward| B4[Backend D]
	end
```

```mermaid
sequenceDiagram
	participant C as Go Client
	participant R as Consul Agent
	participant B1 as Backend 1
	C->>R: Query /v1/catalog/service/api
	R-->>C: [10.0.1.5:8080, 10.0.1.6:8080]
	C->>B1: POST /predict
	B1-->>C: Result
	C->>R: Health check
	R-->>C: Backend 2 unhealthy
```

**Wikimedia Commons Reference:**

![DNS Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Diagram_of_the_Internet.svg/800px-Diagram_of_the_Internet.svg.png)

![Client-Server](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Client-server-model.svg/800px-Client-server-model.svg.png)

### 1.5 Application in ML/AI Systems 🤖

| Case Study | Discovery Pattern | Load Balancer | ML/AI Impact |
|---|---|---|---|
| **OpenAI GPT-4 Fleet** | Client-side control plane | Weighted round-robin on GPU memory | Routes A100 vs H100; 40% latency reduction |
| **Netflix Recommendations** | Server-side Envoy + Eureka | Consistent hashing on user ID | Embeddings stay warm; 95% cache hit rate |
| **Uber Michelangelo** | Hybrid DNS + sidecar | Least-loaded queue depth | 10k+ microservices; <1ms discovery |
| **Hugging Face API** | Kubernetes DNS + Istio | Round-robin topology-aware | Multi-region GPU scheduling; 99.9% availability |

### 1.6 Common Pitfalls ⚠️

> ⚠️ **Warning — Stale Cache Black-Holes:** Client-side caches without TTL or health checks route traffic to failed instances. In ML serving, this causes inference timeouts cascading to batch pipelines. Always pair caching with background health checks.

> ⚠️ **Warning — Thundering Herd on Cache Expiry:** When a popular cache expires, all clients query the registry simultaneously, overwhelming it. Implement `singleflight` for deduplicated refreshes.

> 💡 **Tip:** In Kubernetes, prefer server-side discovery via ClusterIP Services for stateless Go microservices. It eliminates sidecar complexity and integrates natively with kube-proxy rules.

### 1.7 Knowledge Check ❓

1. **Why does the CAP theorem force service registries to choose between consistency and availability?** Describe when an AP registry is preferable for an ML inference gateway.
2. **What is the primary advantage of consistent hashing over round-robin for distributed embedding caches?**
3. **In the Go code, why does `Get()` first acquire an `RLock()` before calling `Refresh()`?** Explain how this prevents writer starvation.

## Module 2: Load Balancing and Resilience

### 2.1 Theoretical Foundation 🧠

Load balancing distributes workloads to optimize utilization and minimize latency. Roots trace to queueing theory (Erlang, 1909) and the **balls-into-bins** problem. The **power of two choices** (Azar et al., 1994) shows that choosing the least loaded of two random backends dramatically reduces maximum load versus pure random assignment. Consistent hashing (Karger et al., 1997) powers Amazon Dynamo and CDN edge selection: when a node changes, only $1/n$ keys remap. Resilience patterns include the circuit breaker (Nygard, 2007), a nonlinear control system with Closed, Open, and Half-Open states. Retry with exponential backoff derives from Ethernet CSMA/CD; rate limiting uses the **token bucket algorithm** from traffic shaping.

### 2.2 Mental Model 📐

Round-robin scheduler:

```
┌──────────────────────────────────────┐
│         Round Robin Scheduler        │
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │ Counter │  │ Backends│  │ Next│ │
│  │    2    │  │ [A,B,C] │──>│  C  │ │
│  └────┬────┘  └─────────┘  └─────┘ │
│       │ (increment mod n)           │
│       ▼                             │
│  ┌─────────┐  (next request goes A) │
│  │    0    │                        │
│  └─────────┘                        │
└──────────────────────────────────────┘
```

Consistent hashing ring:

```
        0 (360°)
         │
    ┌────┴────┐
    │  Hash   │◄─────────────────────┐
    │  Ring   │                      │
    │  ●N1    │  ●Key1 ──────> ●N1  │
    │  ●N2────┼──●Key2 ──────> ●N2  │
    │  ●N3    │  ●Key3 ──────> ●N3  │
    │  ●N1(v) │  ●Key4 ──────> ●N1  │
    └────┬────┘                      │
         └───────────────────────────┘
```

Circuit breaker states:

```
┌──────────────────────────────────────────┐
│         Circuit Breaker States           │
│    ┌─────────┐      Failure >= N        │
│    │ CLOSED  │────────────────────────┐  │
│    │ (Normal)│                        │  │
│    └───┬─────┘                        │  │
│        │ Success                      ▼  │
│        │                         ┌────────┐
│        │                         │  OPEN  │
│        │                         │(Blocked)│
│        │                         └───┬────┘
│        │         Timeout expires     │
│        │<────────────────────────────┘
│        ▼
│   ┌─────────┐    Test fails
│   │HALF-OPEN│────────────────────> OPEN
│   │ (Probe) │
│   └────┬────┘
│        │ Test succeeds
│        └────────────────────────> CLOSED
└──────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

Go implementation of round-robin, consistent hashing, circuit breaker, and token bucket.

```go
package main

import (
	"context"
	"fmt"
	"hash/crc32"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Backend represents an upstream instance.
type Backend struct {
	URL     *url.URL
	Active  atomic.Int64
	Healthy atomic.Bool
}

// RoundRobin cycles through backends atomically.
// WHY: Atomic increment avoids lock contention.
type RoundRobin struct {
	backends []*Backend
	current  atomic.Uint64
	mu       sync.RWMutex
}

func (rr *RoundRobin) Next() *Backend {
	rr.mu.RLock()
	n := len(rr.backends)
	rr.mu.RUnlock()
	if n == 0 {
		return nil
	}
	for i := 0; i < n; i++ {
		idx := int(rr.current.Add(1) % uint64(n))
		if rr.backends[idx].Healthy.Load() {
			return rr.backends[idx]
		}
	}
	return nil
}

// ConsistentHash maps keys to backends using a sorted hash ring.
// WHY: Adding/removing a backend only remaps 1/n keys, preserving cache warmth.
type ConsistentHash struct {
	backends []*Backend
	replicas int
	ring     []uint32
	hashMap  map[uint32]*Backend
	mu       sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
	return &ConsistentHash{replicas: replicas, hashMap: make(map[uint32]*Backend)}
}

func (ch *ConsistentHash) Add(b *Backend) {
	ch.mu.Lock()
	defer ch.mu.Unlock()
	for i := 0; i < ch.replicas; i++ {
		h := crc32.ChecksumIEEE([]byte(b.URL.String() + string(rune(i))))
		ch.ring = append(ch.ring, h)
		ch.hashMap[h] = b
	}
	sort.Slice(ch.ring, func(i, j int) bool { return ch.ring[i] < ch.ring[j] })
	ch.backends = append(ch.backends, b)
}

func (ch *ConsistentHash) Next(key string) *Backend {
	if len(ch.ring) == 0 {
		return nil
	}
	h := crc32.ChecksumIEEE([]byte(key))
	ch.mu.RLock()
	defer ch.mu.RUnlock()
	idx := sort.Search(len(ch.ring), func(i int) bool { return ch.ring[i] >= h })
	if idx == len(ch.ring) {
		idx = 0
	}
	return ch.hashMap[ch.ring[idx]]
}

// CircuitBreaker isolates unhealthy dependencies.
type CircuitBreaker struct {
	failures    atomic.Int32
	threshold   int32
	timeout     time.Duration
	state       atomic.Int32
	lastFailure atomic.Int64
	mu          sync.Mutex
}

func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
	cb := &CircuitBreaker{threshold: int32(threshold), timeout: timeout}
	cb.state.Store(0)
	return cb
}

func (cb *CircuitBreaker) Allow() bool {
	switch cb.state.Load() {
	case 0:
		return true
	case 1:
		if time.Since(time.Unix(0, cb.lastFailure.Load())) > cb.timeout {
			cb.mu.Lock()
			if cb.state.Load() == 1 {
				cb.state.Store(2)
			}
			cb.mu.Unlock()
			return true
		}
		return false
	case 2:
		return true
	}
	return false
}

func (cb *CircuitBreaker) Record(success bool) {
	if success {
		cb.failures.Store(0)
		cb.state.Store(0)
		return
	}
	if cb.failures.Add(1) >= cb.threshold {
		cb.lastFailure.Store(time.Now().UnixNano())
		cb.state.Store(1)
	}
}

// TokenBucket rate limiter.
// WHY: Burst capacity + smooth rate protects inference endpoints.
type TokenBucket struct {
	tokens   float64
	capacity float64
	rate     float64
	last     time.Time
	mu       sync.Mutex
}

func NewTokenBucket(capacity, rate float64) *TokenBucket {
	return &TokenBucket{tokens: capacity, capacity: capacity, rate: rate, last: time.Now()}
}

func (tb *TokenBucket) Allow(tokens float64) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	now := time.Now()
	elapsed := now.Sub(tb.last).Seconds()
	tb.tokens = min(tb.tokens+elapsed*tb.rate, tb.capacity)
	tb.last = now
	if tb.tokens >= tokens {
		tb.tokens -= tokens
		return true
	}
	return false
}

func main() {
	cb := NewCircuitBreaker(3, 5*time.Second)
	tb := NewTokenBucket(10, 5)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !tb.Allow(1.0) {
			http.Error(w, "Rate limited", http.StatusTooManyRequests)
			return
		}
		if !cb.Allow() {
			http.Error(w, "Circuit open", http.StatusServiceUnavailable)
			return
		}
		err := Retry(r.Context(), 3, func() error { return nil })
		cb.Record(err == nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	http.ListenAndServe(":8080", handler)
}

// WHY: Jitter decorrelates retries from multiple clients.
func Retry(ctx context.Context, max int, fn func() error) error {
	var err error
	base := 100 * time.Millisecond
	for i := 0; i <= max; i++ {
		err = fn()
		if err == nil {
			return nil
		}
		if i == max {
			break
		}
		delay := base * time.Duration(1<<i)
		jitter := time.Duration(float64(delay) * float64(time.Now().UnixNano()%100) / 100.0)
		select {
		case <-time.After(delay + jitter):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return fmt.Errorf("retry exhausted: %w", err)
}
```

### 2.4 Visual Representation 🖼️

```mermaid
graph LR
	subgraph "Round Robin"
		C1[Client] -->|Req 1| A[Backend A]
		C1 -->|Req 2| B[Backend B]
		C1 -->|Req 3| C[Backend C]
		C1 -->|Req 4| A
	end
	subgraph "Consistent Hash"
		C2[Client] -->|Key=User:42| H{Hash Ring}
		H -->|Clockwise| A2[Backend A]
		C2 -->|Key=User:99| H
		H -->|Clockwise| B2[Backend B]
	end
```

```mermaid
stateDiagram-v2
	[*] --> Closed
	Closed --> Closed : Success
	Closed --> Open : Failures >= Threshold
	Open --> HalfOpen : Timeout Expired
	HalfOpen --> Closed : Success
	HalfOpen --> Open : Failure
	Open --> Open : Request Rejected
```

**Wikimedia Commons Reference:**

![Load Balancing](https://upload.wikimedia.org/wikipedia/commons/7/74/Load_balancing_diagram.svg)

![Circuit Breaker](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Circuit_breaker.svg/800px-Circuit_breaker.svg.png)

### 2.5 Application in ML/AI Systems 🤖

| Case Study | Algorithm/Pattern | Why Chosen | ML/AI Impact |
|---|---|---|---|
| **Google TPU Pods** | Weighted round-robin | TPU v4 vs v5 FLOPS differ | 25% step-time improvement |
| **Pinterest Cache** | Consistent hashing | User embeddings stay warm | 98% cache hit; p99 < 5ms |
| **OpenAI API** | Circuit breaker + retry | Per-model breaker | 99.99% availability |
| **AWS SageMaker** | Token bucket rate limit | Per-tenant buckets | Fair GPU sharing; zero noisy neighbors |

### 2.6 Common Pitfalls ⚠️

> ⚠️ **Warning — Unbounded Connection Growth:** Least-connections balancers that fail to decrement active counts on timeout eventually blacklist healthy backends. In Go, always defer `active.Add(-1)`.

> ⚠️ **Warning — Hotspots in Consistent Hashing:** Without virtual replicas (50-200 per node), load distribution is uneven. Always use sufficient virtual replicas.

> 💡 **Tip:** For GPU clusters, weight backends by available VRAM. An 80GB A100 should receive 4x the traffic of a 20GB node. Export breaker state as a Prometheus gauge; alerting triggers automatic canary rollback.

### 2.7 Knowledge Check ❓

1. **Explain why the "power of two choices" makes least-connections superior to pure random selection.**
2. **In the consistent hashing code, why is binary search used instead of linear scan?**
3. **Describe a scenario where round-robin is strictly better than consistent hashing for ML inference.**

## 📦 Compression Code

Go script to compress health check batches for cache storage:

```go
package main

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"fmt"
)

type Endpoint struct {
	Address string `json:"address"`
	Healthy bool   `json:"healthy"`
	Latency int64  `json:"latency_ms"`
}

type HealthBatch struct {
	Service   string     `json:"service"`
	Endpoints []Endpoint `json:"endpoints"`
}

// WHY: Reduces discovery sync payload by 80% for large fleets.
func main() {
	batch := HealthBatch{
		Service: "ml-inference",
		Endpoints: []Endpoint{
			{Address: "10.0.1.10:8080", Healthy: true, Latency: 12},
			{Address: "10.0.1.11:8080", Healthy: true, Latency: 15},
			{Address: "10.0.1.12:8080", Healthy: false, Latency: 5000},
		},
	}
	data, _ := json.Marshal(batch)
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	gw.Write(data)
	gw.Close()
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("Original: %d bytes\n", len(data))
	fmt.Printf("Compressed+Encoded: %d bytes\n", len(encoded))
}
```

## 🎯 Documented Project

### Description

Develop **BalanceGo**, a production-ready HTTP reverse proxy and load balancer in Go. It supports round-robin, least-connections, and weighted algorithms, active health checks, and circuit breakers. Targets ML inference workloads with heterogeneous GPU capacities.

### Functional Requirements

1. Reverse proxy forwarding HTTP requests to configured backends.
2. Round-robin, least-connections, and weighted round-robin selectable via config.
3. Active health checks every 5 seconds via HTTP `/health`.
4. Circuit breaker: opens after 5 consecutive failures, half-opens after 30 seconds.
5. Prometheus metrics for requests, latency, backend health, and breaker state.

### Main Components

- `cmd/proxy/main.go` — Entry point
- `pkg/balancer/` — Algorithms
- `pkg/health/` — Health check goroutines
- `pkg/breaker/` — Circuit breaker state machine
- `pkg/metrics/` — Prometheus instrumentation
- `config.yaml` — Backend list and thresholds

### Success Metrics

- 10,000 concurrent connections with <5 ms added latency.
- Unhealthy backends removed within 10 seconds.
- Circuit breaker opens within 1 second of threshold breach.
- `/metrics` exposes `proxy_requests_total`, `proxy_backend_health`, `breaker_state`.
- Health check accuracy >99%.

### References

- [Consul Service Discovery](https://developer.hashicorp.com/consul/docs/concepts/service-discovery)
- [Envoy Proxy Architecture](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview)
- [AWS Elastic Load Balancing](https://docs.aws.amazon.com/elasticloadbalancing/)
- [[05 - Infrastructure as Code with Pulumi|🏗️ 05 - IaC with Pulumi]]
- [[06 - Cloud Networking and Observability|📡 06 - Observability]]
