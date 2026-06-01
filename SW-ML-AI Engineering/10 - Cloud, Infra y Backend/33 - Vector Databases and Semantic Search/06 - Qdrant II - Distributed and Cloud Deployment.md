# 🔍 6 - Qdrant II - Distributed and Cloud Deployment

## 🎯 Learning Objectives
- Architect Qdrant clusters using Raft consensus, sharding, and replication for horizontal scalability
- Configure write consistency levels (`ALL`, `MAJORITY`, `ONE`) and understand their trade-offs for ML pipelines
- Implement snapshot and backup strategies for distributed vector collections — both collection-level and full storage snapshots
- Deploy on Qdrant Cloud and self-hosted Kubernetes with production best practices
- Integrate Qdrant with LangChain and LlamaIndex for RAG and agentic workflows
- Benchmark Qdrant performance with concurrent clients and identify bottlenecks in distributed settings

## Introduction

A single-node Qdrant instance is sufficient for prototyping and many mid-scale production workloads (up to ~10M vectors). But modern AI systems — especially RAG platforms serving millions of users, global e-commerce search, and cross-region recommendation engines — require horizontal scalability, fault tolerance, and geographic distribution. Qdrant's distributed mode provides these capabilities through a **Raft-based consensus layer** for cluster metadata, **sharded collections** for data partitioning, and **configurable replication** for fault tolerance and read scaling.

This note is the capstone of the course. It combines the algorithmic foundations from [[01 - Vector Search Fundamentals]] and [[02 - Indexing Algorithms Deep Dive]], the database mechanics from [[05 - Qdrant I - Architecture and Collections]], and operational practices from [[04 - pgvector II - Production and Hybrid Search]] to show how vector search scales to cloud-native deployments. We also cover integrations with the two dominant LLM orchestration frameworks — LangChain and LlamaIndex — because vector databases in production are almost always consumed through these interfaces.

This module connects to [[15 - Docker and Kubernetes]] (container orchestration for stateful workloads), [[18 - MLOps and Model Serving]] (production SLAs and observability), and [[06 - Large Language Models]] (downstream RAG consumption of vector retrieval results).

---

## 1. Cluster Architecture: Raft, Sharding, and Replication

### Theory

Qdrant's distributed architecture has three core concepts:

1. **Consensus (Raft)**: A consensus group of peer nodes maintains cluster metadata — which collections exist, how they are sharded, and where replicas live. Raft ensures that metadata changes (creating a collection, scaling shards, adding/removing nodes) are committed by a majority of nodes before taking effect. This prevents split-brain scenarios during network partitions. Raft operates on a separate port (6335) and handles only metadata, not vector data — data replication uses a separate streaming protocol.

2. **Sharding**: A collection is partitioned into `shard_count` shards. Each shard is an independent subset of points with its own HNSW graph, payload storage, and RocksDB instance. When you upsert a point, Qdrant hashes the point ID using a consistent hash function to determine the target shard. Shards can be distributed across nodes, allowing a single collection to exceed the RAM or disk capacity of any single machine.

3. **Replication**: Each shard has `replication_factor` copies (replicas) distributed across different nodes. One replica is the **primary** — all writes go to the primary first, then propagate to secondaries. Reads can be served by any replica, enabling horizontal read scaling. If a node fails, replicas on surviving nodes continue to serve queries, and the Raft consensus group detects the failure and promotes a new primary for orphaned shards after a configurable timeout.

The interplay of these concepts means Qdrant scales along three axes: **collection count** (horizontal isolation), **shard count** (partitioning per collection), and **replication factor** (read throughput and fault tolerance). A well-architected cluster balances all three.

```python
from qdrant_client import QdrantClient

# Connect to any node in the cluster — the client discovers topology.
client = QdrantClient(
    url="http://qdrant-node-1:6333",
    prefer_grpc=True,
)

# Create a distributed collection with sharding and replication.
# WHY: shard_number determines partitioning; replication_factor
#      determines fault tolerance and read capacity.
client.create_collection(
    collection_name="distributed_docs",
    vectors_config={"size": 768, "distance": "Cosine"},
    shard_number=6,
    replication_factor=2,
    hnsw_config={"m": 16, "ef_construct": 128},
)

# Write consistency levels.
# - ALL: wait for all replicas (strongest consistency, slowest)
# - MAJORITY: wait for quorum of replicas (default, balanced)
# - ONE: wait for primary only (fastest, weakest consistency)
client.upsert(
    collection_name="distributed_docs",
    points=[...],  # your PointStruct list
    wait=True,
)
```

```yaml
# Docker Compose snippet for a 3-node Qdrant cluster (dev/testing).
version: "3.8"
services:
  qdrant-node1:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__PEERS_URL=http://qdrant-node2:6335,http://qdrant-node3:6335
    volumes:
      - ./qdrant-node1:/qdrant/storage
  qdrant-node2:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__PEERS_URL=http://qdrant-node1:6335,http://qdrant-node3:6335
    volumes:
      - ./qdrant-node2:/qdrant/storage
  qdrant-node3:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__PEERS_URL=http://qdrant-node1:6335,http://qdrant-node2:6335
    volumes:
      - ./qdrant-node3:/qdrant/storage
```

⚠️ **Using `ALL` write consistency for high-throughput ingestion pipelines creates a tail-latency bottleneck.** If one replica is slow (GC pause, network blip, disk saturation), every write stalls. Use `MAJORITY` for production writes and `ONE` only for offline batch jobs where eventual consistency is acceptable.

💡 **"ONE for batches, MAJORITY for serving, ALL for money."** — Reserve `ALL` consistency for critical metadata changes, not bulk vector upserts. Your weekend backfill job does not need every replica to acknowledge every point.

**Caso real — Klarna merchant search:** Klarna uses a multi-node Qdrant cluster to power semantic search across their transaction and merchant documentation. Sharding by `merchant_id` hash keeps related embeddings co-located for efficient filtered search, while replication factor 3 ensures their AI assistant remains operational during rolling upgrades and single-node failures. Their write consistency is `MAJORITY` for customer-facing queries and `ONE` for their nightly re-embedding batch job (50M+ documents), which runs 3× faster with the relaxed consistency.

❌ **Antipattern: Creating too many shards.** More shards = more parallelism, but also more network overhead for cross-shard queries and more memory for per-shard HNSW graphs. A rule of thumb: `shard_number = node_count × 2`. For a 5-node cluster, start with 10 shards and benchmark before increasing.

✅ **Use consistent hashing — not range partitioning — for shard assignment.** Range partitioning (by ID or timestamp) creates hot spots where one shard receives most writes. Hashing distributes points uniformly across shards, and Qdrant's internal routing handles the mapping transparently.

---

## 2. Snapshots, Backups, and Qdrant Cloud

### Theory

Vector databases are stateful infrastructure, and their state must be protected. Qdrant provides **snapshots** at two levels:

1. **Collection snapshots**: A point-in-time dump of a single collection's vectors, payloads, and indexes. Snapshots are portable — you can create one on a self-hosted cluster and restore it on Qdrant Cloud, or vice versa. They are compressed tar archives containing the collection's raw data files and index structures.

2. **Full storage snapshots**: A filesystem-level snapshot of the entire Qdrant storage directory. These are faster (no Qdrant-side serialization) but less portable and require coordinated filesystem support (LVM, ZFS, or cloud volume snapshots). They capture all collections, cluster metadata, and Raft state.

For managed deployments, **Qdrant Cloud** provides a control plane for cluster management, scaling, monitoring, and automated snapshots. Crucially, Qdrant Cloud runs the same open-source binary — there is no proprietary fork. This means zero vendor lock-in: you can export a snapshot from Cloud and restore it on a self-hosted cluster at any time.

Backup strategy recommendations:
- **Daily collection snapshots** for RPO of 24 hours
- **Pre-deployment snapshots** before schema migrations or index parameter changes
- **Cross-region snapshot replication** for disaster recovery in cloud environments
- **Quarterly restore drills** to validate backup integrity

```python
# Create a collection snapshot via the Python client.
# WHY: Snapshots are atomic — they capture the HNSW graph, vectors,
#      and payloads in a consistent state without stopping the server.
client.create_snapshot(collection_name="distributed_docs")

# List snapshots.
snapshots = client.list_snapshots(collection_name="distributed_docs")
for snap in snapshots:
    print(snap.name, snap.creation_time, snap.size)

# Restore snapshot into a new collection (non-destructive).
client.recover_snapshot(
    collection_name="distributed_docs_backup",
    location=snapshots[0].url,  # or S3 URL / file path
)
```

```bash
# REST API for snapshot creation (useful in CI/CD scripts).
curl -X POST \
  'http://localhost:6333/collections/distributed_docs/snapshots'

# Download snapshot for offsite backup.
curl -O \
  'http://localhost:6333/collections/distributed_docs/snapshots/latest'
```

```yaml
# Kubernetes CronJob for daily snapshot to S3.
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qdrant-snapshot
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: snapshotter
            image: curlimages/curl:latest
            command:
            - sh
            - -c
            - |
              curl -X POST "http://qdrant:6333/collections/distributed_docs/snapshots" && \
              aws s3 cp "http://qdrant:6333/collections/distributed_docs/snapshots/latest" \
                s3://my-backup-bucket/qdrant/daily-$(date +%Y%m%d).snap
          restartPolicy: OnFailure
```

⚠️ **Restoring a snapshot from a different Qdrant version can cause silent corruption.** Snapshot formats can change between minor versions. Always tag snapshots with the Qdrant binary version and the embedding model version used to generate the vectors.

💡 **"Snapshot the version, not just the data."** — Store the Qdrant version and embedding model name alongside each snapshot. A simple naming convention: `docs-2025-06-01--qdrant-v1.12--e5-large-v2.snap`.

**Caso real — Health-tech startup disaster recovery:** A health-tech startup running Qdrant Cloud takes automated snapshots every 6 hours. During a buggy deployment that corrupted payload indexes across all replicas, they restored the previous snapshot to a new collection in under 10 minutes, validated data integrity by comparing total point count and a checksum of the first 1000 vectors, and switched DNS traffic to the new collection. The snapshot portability between Qdrant Cloud and their on-premises DR cluster was critical for their hybrid cloud compliance strategy.

❌ **Antipattern: Relying only on Kubernetes volume snapshots without Qdrant-level snapshots.** A K8s volume snapshot captures the storage directory while Qdrant is running, but it may contain partially written data or inconsistent index state. Qdrant-level collection snapshots are guaranteed consistent and can be restored to any cluster. Use both: K8s snapshots for quick node recovery, Qdrant snapshots for cross-cluster migration.

✅ **Test snapshot restore in a non-production environment before every major deployment.** Knowing that your RPO (Recovery Point Objective) is 6 hours is useless if the snapshot restore process takes 8 hours due to a network bottleneck. Measure end-to-end restore time quarterly.

---

## 3. Client Integration and Performance Benchmarking

### Theory

Vector databases do not exist in isolation — they are consumed by applications written in Python, Go, Rust, and JavaScript. Qdrant provides first-party clients for Python (`qdrant-client`) and Go (`qdrant/go-client`), with community clients for other languages. In the LLM ecosystem, Qdrant is a standard backend for **LangChain** and **LlamaIndex**, the two dominant retrieval frameworks.

Performance benchmarking is not optional for distributed deployments. Key metrics:
- **Latency**: p50, p95, p99 per query (single and batch), measured at the client, not the server
- **Throughput**: queries per second (QPS) at a target p99 latency
- **Recall@k**: compared against exact brute-force results on a held-out query set
- **Resource utilization**: CPU, RAM, disk I/O, network bandwidth per node

For distributed clusters, benchmark both local (single-shard) and global (cross-node) queries, as network hops add latency. A query that hits a single shard on the same node may be 2–5× faster than one that requires scatter-gather across 3 nodes.

```python
# LangChain integration.
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Qdrant(
    client=client,
    collection_name="distributed_docs",
    embeddings=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever,
)
answer = qa_chain.invoke("What is the Raft consensus algorithm?")
```

```python
# LlamaIndex integration.
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = QdrantVectorStore(
    client=client,
    collection_name="distributed_docs",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
query_engine = index.as_query_engine()
response = query_engine.query("How does HNSW indexing work?")
```

```python
# Performance benchmarking harness.
import time
import numpy as np
from statistics import quantiles

def benchmark_search(client, queries: list[list[float]], k: int = 10):
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        client.search("distributed_docs", q, limit=k)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    return {
        "p50": latencies[len(latencies)//2],
        "p99": latencies[int(len(latencies) * 0.99)],
        "qps": len(queries) / (sum(latencies) / 1000),
    }

results = benchmark_search(client, [[0.1]*768 for _ in range(500)])
print(f"p50={results['p50']:.1f}ms  p99={results['p99']:.1f}ms  QPS={results['qps']:.0f}")
```

⚠️ **Benchmarking with a single-threaded client on a multi-node cluster produces misleading results.** A single Python process with synchronous `client.search()` calls cannot stress a distributed cluster — you will observe artificially low QPS and conclude the cluster is underloaded. Use concurrent workers (Python `ThreadPoolExecutor`, `asyncio`, or Go goroutines) that match or exceed your expected production concurrency.

💡 **"One client, one shard; many clients, real numbers."** — Benchmark with concurrency equal to or greater than shard_count × replication_factor to see true cluster behavior under load.

**Caso real — Vercel AI SDK RAG template:** Vercel uses Qdrant as the retrieval layer for their AI SDK RAG starter template. The template demonstrates a full production pipeline: Next.js frontend → API route → LangChain retriever → Qdrant Cloud → OpenAI embeddings → GPT-4o response. The Go client is used internally for a high-throughput ingestion service that processes millions of documentation pages concurrently, using gRPC streaming to saturate the network link. Their key optimization: batch upserts of 1000 points per gRPC call, achieving 50,000 points/second/node.

❌ **Antipattern: Using default batch sizes for upserts.** The default batch size in `qdrant-client` is optimized for developer convenience, not throughput. For production ingestion, batch at least 500–1000 points per `upsert()` call. Measure throughput at different batch sizes — the sweet spot is typically between 500 and 2000 points per call, depending on payload size.

✅ **Use `asyncio` with `qdrant-client` for concurrent search and ingest.** The Python client supports async operations via `httpx` and `grpcio`. For workloads with 100+ QPS, the async client achieves 2–3× throughput over the synchronous version by overlapping I/O operations.

---

## 🎯 Key Takeaways

- **Raft consensus** ensures cluster metadata consistency; vector data is sharded and replicated independently — Raft handles only metadata, not data plane operations
- **Sharding** partitions collections horizontally via consistent hashing; **replication** provides fault tolerance and read scaling — size shards so each fits in available RAM on a single node
- Use `MAJORITY` write consistency for production serving (balanced), `ONE` for offline batch ingestion (fastest), and `ALL` only for critical metadata changes
- **Collection snapshots** are portable and version-tagged; use them for backup, migration, and environment promotion between Qdrant Cloud and self-hosted clusters
- **Qdrant Cloud** runs the same open-source binary, enabling zero-vendor-lock-in — export snapshots to migrate at any time
- **LangChain** and **LlamaIndex** integrations enable rapid RAG prototyping; tune `search_kwargs` (k, score_threshold) and retrieval strategy per use case
- Benchmark with concurrent client count ≥ shard_count × replication_factor to see true distributed performance
- Batch upserts at 500–1000 points per call for maximum ingest throughput; use gRPC with async I/O for production workloads

## References

- Qdrant Distributed Deployment: https://qdrant.tech/documentation/guides/distributed_deployment/
- Qdrant Cloud: https://qdrant.tech/cloud/
- Qdrant Snapshots: https://qdrant.tech/documentation/concepts/snapshots/
- LangChain Qdrant Integration: https://python.langchain.com/docs/integrations/vectorstores/qdrant/
- LlamaIndex Qdrant Integration: https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/
- [[05 - Qdrant I - Architecture and Collections]] — Core Qdrant concepts and single-node operations
- [[15 - Docker and Kubernetes]] — Container orchestration for stateful workloads
- [[06 - Large Language Models]] — Downstream RAG consumption

## 📦 Código de compresión

```python
"""
Qdrant II — Compression Script
Summarizes: clustering, replication, snapshots, cloud, LangChain, benchmarking.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import time
import numpy as np

class QdrantClusterManager:
    def __init__(self, url: str):
        self.client = QdrantClient(url=url, prefer_grpc=True)

    def create_distributed_collection(self, name: str, dim: int = 768):
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            shard_number=6,
            replication_factor=2,
            hnsw_config={"m": 16, "ef_construct": 128},
        )

    def snapshot_and_upload(self, name: str):
        snap = self.client.create_snapshot(collection_name=name)
        print(f"Snapshot created: {snap.name} ({snap.size / 1e6:.1f} MB)")
        return snap

    def benchmark(self, name: str, queries: list[list[float]], k: int = 10):
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            self.client.search(name, q, limit=k)
            latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()
        return {
            "p50": latencies[len(latencies)//2],
            "p99": latencies[int(len(latencies)*0.99)],
            "qps": len(queries) / (sum(latencies) / 1000),
        }

    def upsert_batch(self, name: str, n: int = 10000, dim: int = 768):
        batch_size = 500
        for i in range(0, n, batch_size):
            points = [
                PointStruct(
                    id=i + j,
                    vector=np.random.randn(dim).astype(float).tolist(),
                    payload={"idx": i + j, "group": f"g{(i+j) % 10}"},
                )
                for j in range(min(batch_size, n - i))
            ]
            self.client.upsert(collection_name=name, points=points, wait=True)
            if (i // batch_size) % 4 == 0:
                print(f"  Upserted {i + len(points)}/{n}")

if __name__ == "__main__":
    mgr = QdrantClusterManager("http://localhost:6333")
    mgr.create_distributed_collection("bench_docs")
    mgr.upsert_batch("bench_docs", n=5000)
    # Wait for background indexing.
    time.sleep(2)
    queries = [np.random.randn(768).tolist() for _ in range(100)]
    results = mgr.benchmark("bench_docs", queries, k=10)
    print(f"Benchmark: p50={results['p50']:.1f}ms  p99={results['p99']:.1f}ms  QPS={results['qps']:.0f}")
```
