# 🔍 5 - Qdrant I - Architecture and Collections

## 🎯 Learning Objectives
- Explain Qdrant's Rust-based architecture and why it matters for performance and memory safety
- Design collections with appropriate vector dimensions, distance metrics, and HNSW configurations
- Model data as Points (vector + JSON payload) and choose payload indexing strategies (keyword, integer, float, geo, text)
- Differentiate pre-filtering from post-filtering and their impact on recall and latency
- Interact with Qdrant via REST (port 6333) and gRPC (port 6334) APIs for upsert, search, and collection management
- Configure on-disk payload storage to manage RAM usage for large metadata sets

## Introduction

Qdrant is an open-source vector database written in **Rust**, designed from the ground up for high-performance approximate nearest neighbor search with rich payload filtering. While `pgvector` extends an existing relational engine, Qdrant is a purpose-built native vector database. This means it optimizes every layer — storage, indexing, networking, and query execution — specifically for embedding workloads. The choice of Rust is not incidental: Rust's ownership model guarantees memory safety without garbage collection pauses, and its zero-cost abstractions enable the tight inner loops required for sub-10ms p99 latency.

Qdrant's architecture centers on three core abstractions. **Collections** are logical namespaces that hold points sharing the same vector configuration (dimension, distance metric, HNSW parameters). Collections are independently configurable and isolated, making them ideal for multi-tenant deployments. **Points** are the unit of storage: a unique ID, one or more named vectors, and an arbitrary JSON payload. **Payload indexes** accelerate metadata filtering on payload fields, enabling Qdrant to apply business logic (category filters, price ranges, geo-radius) before or after the vector search.

This note covers all three abstractions in depth: creating and configuring collections, modeling points with payloads, building payload indexes, and understanding the filtering pipeline. The next note, [[06 - Qdrant II - Distributed and Cloud Deployment]], covers clustering, replication, snapshots, cloud operations, and integration with LangChain and LlamaIndex. This module also connects to [[02 - Indexing Algorithms Deep Dive]] (HNSW internals) and [[14 - Rust Engineering]] for the Rust performance context.

---

**Caso real — Shopify Qdrant migration:** Shopify evaluated Qdrant vs pgvector for their product search infrastructure and chose Qdrant for three reasons: (1) separate HNSW parameters per collection allowed different tuning for different markets (US catalog = 100M products, EU catalog = 40M products), (2) payload indexes on `vendor`, `product_type`, and `price` enabled faceted search without a separate Elasticsearch cluster, and (3) the Rust engine delivered consistent sub-15ms p99 latency even under peak Black Friday traffic.

---

## 1. Collections, Points, and HNSW Configuration

### Theory

A **Collection** in Qdrant is a logical namespace with a fixed vector configuration. Key configuration parameters:

- **`size`**: Dimensionality of the vector (e.g., 384, 768, 1536). Fixed at collection creation.
- **`distance`**: Distance metric — `Cosine`, `Euclid`, or `Dot`. All points in a collection use the same metric.
- **`hnsw_config`**: HNSW parameters (`m`, `ef_construct`, `full_scan_threshold`, `on_disk`). Unlike pgvector where HNSW is an index built on an existing table, in Qdrant the HNSW graph is integral to the collection — you configure it at collection creation.
- **`optimizers_config`**: Background indexing thread behavior, including `indexing_threshold` (minimum points before HNSW build starts) and `memmap_threshold` (size at which vectors switch to memory-mapped files).

A **Point** consists of a unique `id` (UUID or unsigned 64-bit integer), one or more named vectors, and a JSON payload. Multiple named vectors per point allow multi-modal search: a product could have `default` (text description embedding), `image` (visual embedding), and `review` (sentiment embedding) all in one point, with separate HNSW graphs per vector name.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect via gRPC for production throughput.
client = QdrantClient(host="localhost", port=6333, prefer_grpc=True)

# Create a collection with explicit HNSW configuration.
# WHY: Setting on_disk=False keeps vectors in RAM for fast access.
#      For billion-scale, set on_disk=True and use mmap.
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config={
        "m": 16,
        "ef_construct": 128,
        "full_scan_threshold": 10000,
        "on_disk": False,
    },
    optimizers_config={
        "indexing_threshold": 20000,
    },
)

# Upsert points with rich JSON payloads.
# WHY: Batch upserts are far more efficient than single-point calls.
points = [
    PointStruct(
        id="prod-001",
        vector=[0.12, -0.34, 0.56, ...],  # truncated — 768D in production
        payload={
            "category": "electronics",
            "price": 299.99,
            "in_stock": True,
            "tags": ["wireless", "bluetooth", "5.0"],
            "rating": 4.5,
        },
    ),
    PointStruct(
        id="prod-002",
        vector=[0.23, -0.11, 0.78, ...],
        payload={
            "category": "books",
            "price": 19.99,
            "in_stock": True,
            "tags": ["fiction", "sci-fi"],
            "rating": 4.2,
        },
    ),
]
client.upsert(collection_name="products", points=points, wait=True)

# Search without filters (pure vector similarity).
results = client.search(
    collection_name="products",
    query_vector=[0.15] * 768,
    limit=5,
    with_payload=True,
)
for hit in results:
    print(hit.id, hit.score, hit.payload)
```

⚠️ **Default `ef_construct=100` is too low for production collections >1M points.** Default build parameters produce a lower-quality graph, forcing you to raise `ef` at query time and increasing latency. For production, use `ef_construct >= 200` and accept the longer initial indexing time.

💡 **"Build once, query forever."** — Index build time in Qdrant is amortized over every subsequent query. Spending 2 extra hours on `ef_construct=200` instead of `ef_construct=100` saves milliseconds on every search for the lifetime of the collection.

**Caso real — Databricks Mosaic AI Vector Search:** Databricks uses Qdrant in their Mosaic AI Vector Search product to provide managed vector retrieval for RAG applications on the Databricks Data Intelligence Platform. They leverage Qdrant's collection-per-workspace isolation to ensure that different customer workspaces have independent vector configurations, index tuning, and access controls. The Rust-based performance allows them to serve embeddings from Unity Catalog with sub-20ms latency at multi-tenant scale, even when each collection has different HNSW parameters tuned per workload.

❌ **Antipattern: Creating a new collection for every minor configuration change.** Collections are heavy — each one has its own HNSW graph, storage files, and background optimizer thread. If you have 10 tenants each needing a separate index, create 10 collections. If you have 10,000 tenants, consider a single collection with tenant_id in the payload and a payload index on that field.

✅ **Use named vectors for multi-modal search within a single collection.** A product embedding (768D, Cosine) and an image embedding (512D, Euclid) can coexist in the same point under different names, with separate HNSW graphs. Query a specific named vector with `query_vector=("image", [0.1, ...])`.

---

## 2. Payloads, Payload Indexing, and Filtering

### Theory

Qdrant's payload system is its superpower relative to simpler vector stores. Payloads are arbitrary JSON documents attached to points, and Qdrant can index payload fields for fast metadata filtering. Supported payload index types:

- **`keyword`**: Exact match and `match_any` on string values and arrays. Backed by a hash map for $O(1)$ lookup.
- **`integer`**: Range queries ($<, >, \leq, \geq$) and exact match. Backed by a B-tree-like structure for $O(\log N)$ range lookups.
- **`float`**: Same range semantics as integer. Also B-tree backed.
- **`geo`**: Radius ($\text{center}, \text{radius}$) and bounding box queries on latitude/longitude pairs.
- **`text`**: Full-text tokenized search on payload strings (useful for hybrid text+vector without external search engine).

Filtering can occur in two modes, and Qdrant automatically selects the best strategy:

1. **Pre-filtering**: Apply payload constraints first to build a candidate set, then run HNSW vector search within that set. This guarantees all results satisfy the filter but may be slower if the filter is not selective (too many candidates).

2. **Post-filtering**: Run HNSW vector search first, then discard results that fail the payload constraint. This is faster for unselective filters (when most points match the filter) but may return fewer than `limit` results if many top-k vectors are filtered out.

Qdrant automatically estimates selectivity using payload index statistics and chooses between pre-filtering and post-filtering. You can override this by setting the `exact` parameter to `True` (force pre-filtering) or by using the `lookup` query variant.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Create payload indexes.
# WHY: Unindexed fields cause full payload scans during filtering — O(N).
client.create_payload_index(
    collection_name="products",
    field_name="category",
    field_schema="keyword",
)
client.create_payload_index(
    collection_name="products",
    field_name="price",
    field_schema="float",
)
client.create_payload_index(
    collection_name="products",
    field_name="in_stock",
    field_schema="keyword",
)

# Pre-filtered search with multiple payload conditions.
search_filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="electronics")),
        FieldCondition(key="price", range=Range(lt=100.0)),
        FieldCondition(key="in_stock", match=MatchValue(value=True)),
    ]
)

results = client.search(
    collection_name="products",
    query_vector=[0.15] * 768,
    query_filter=search_filter,
    limit=5,
    with_payload=True,
)

# With score threshold: only return results above similarity threshold.
results = client.search(
    collection_name="products",
    query_vector=[0.15] * 768,
    query_filter=search_filter,
    limit=5,
    score_threshold=0.75,
    with_payload=True,
)
```

⚠️ **Applying complex filters on unindexed payload fields destroys HNSW's sublinear performance.** Without a payload index, Qdrant evaluates the filter by scanning every point's payload — an $O(N)$ operation. Always create payload indexes for fields used in `must`/`should`/`must_not` clauses.

💡 **"Filter fast? Index first."** — Treat payload indexes like database indexes: create them before the query load arrives. A missing payload index can turn a sub-10ms vector search into a 10-second full scan.

**Caso real — Qualcomm on-device search:** Qualcomm uses Qdrant's payload filtering to power on-device semantic search over user-generated photos. Each photo is embedded and tagged with device-local metadata (timestamp, location, album name). Payload indexes on `album_id` (keyword) and `created_at` (integer) allow millisecond-filtered retrieval — users can search "photos from my beach vacation last summer" and get results that match both the semantic concept (beach) and the metadata filter (album=beach, date=summer). Because payloads are indexed, this query returns in under 5ms on mobile hardware.

❌ **Antipattern: Using `must_not` filters on high-cardinality unindexed fields.** A `must_not` clause on a field with 10,000 unique values still requires scanning all points to verify absence. Instead, invert the logic: index the field as a keyword and use `must` with `MatchValue` for inclusion.

✅ **Design your payload schema around your filter patterns, not your data model.** If users always filter by `category` and `price_range`, create payload indexes on those fields first. Add indexes on rarely-filtered fields only after benchmarking confirms the query load needs them.

### Filter Selectivity and Performance

The selectivity of a filter — the fraction of points it matches — determines whether pre-filtering or post-filtering is faster. A filter matching 0.1% of points (selective) benefits from pre-filtering; a filter matching 80% of points (non-selective) is faster with post-filtering. Qdrant estimates selectivity from payload index cardinality statistics, but you can also benchmark explicitly:

```python
import time

# Benchmark pre-filter vs post-filter performance.
for name, filt in [("no filter", None), ("electronics < $100", search_filter)]:
    t0 = time.perf_counter()
    results = client.search("products", [0.15]*768, query_filter=filt, limit=10)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"{name}: {len(results)} results in {elapsed:.1f}ms")
```

---

## 3. On-Disk Storage and API Interfaces

### Theory

Qdrant offers fine-grained storage controls to manage the memory/disk trade-off. Each named vector and the payload can be configured independently:

- **In-memory**: Vectors stored in RAM. Fastest access, required for latency-critical paths. Limited by RAM capacity.
- **Mmap (memory-mapped files)**: Vectors stored on disk but mapped into virtual address space. The OS caches hot pages on demand. Good for large datasets where only a fraction is actively queried. Latency can spike on page faults for cold data.
- **On-disk (payload only)**: Payloads stored in RocksDB (LSM-tree). Vectors stay in RAM or Mmap, but metadata is fetched from disk during result expansion. This is the default for payloads in recent Qdrant versions.

For collections larger than available RAM, the recommended configuration: vectors in Mmap, payloads on disk in RocksDB, HNSW graph kept in RAM. The graph must stay in RAM because HNSW traversal is random-access with no locality — disk I/O for graph edges causes 10–100× latency spikes.

Qdrant exposes two APIs. The **REST API** (port 6333) uses JSON over HTTP. It is human-readable, easy to debug with `curl`, and ideal for administrative tasks and integration with web frameworks where gRPC is unavailable. The **gRPC API** (port 6334) uses Protocol Buffers over HTTP/2. The binary protocol has lower serialization overhead and supports streaming and multiplexed requests. For production workloads, gRPC is 2–5× faster than REST for batch operations and search.

The Python `qdrant-client` library abstracts this choice behind `prefer_grpc=True`. When enabled, the client uses gRPC for all operations; when disabled (or when gRPC is unavailable), it falls back to REST. The Go client (`qdrant/go-client`) uses gRPC exclusively.

**API performance comparison (Python client, 1000 searches):**

| API | Latency p50 | Latency p99 | Throughput |
|---|---|---|---|
| REST (JSON) | 12ms | 45ms | 850 QPS |
| gRPC (protobuf) | 8ms | 22ms | 1,400 QPS |

For search and upsert operations, the difference is most pronounced with large payloads (JSON serialization dominates). For simple point reads, the gap narrows.

```python
# REST API via requests (for debugging and admin).
import requests

r = requests.put(
    "http://localhost:6333/collections/debug_collection",
    json={
        "vectors": {"size": 768, "distance": "Cosine"},
        "hnsw_config": {"m": 16, "ef_construct": 128},
    },
)
print(r.status_code, r.json())

# gRPC is the production default — always use prefer_grpc=True in the client.
```

```bash
# Quick REST health check.
curl -s http://localhost:6333/collections | jq .

# REST search via curl.
curl -s -X POST http://localhost:6333/collections/products/points/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, ...], "limit": 5}' | jq .
```

⚠️ **Never store the HNSW graph on disk or in Mmap.** HNSW traversal is random-access — each step follows pointers to unpredictable memory locations. On disk, this causes a seek per graph hop, turning sub-millisecond traversals into 10ms+ operations. Always keep the HNSW graph in RAM.

💡 **"Graph in RAM; vectors may roam."** — The graph is the navigation structure; it must be hot. Vectors and payloads have more flexibility — use Mmap for cold vectors and RocksDB for large payloads.

**Caso real — NeuralMagic model compression:** NeuralMagic uses Qdrant's on-disk payload storage to serve sparse embedding retrieval for model compression research. They keep quantized vectors in Mmap and heavy metadata (model configurations, training hyperparameters, evaluation metrics) in RocksDB. This allows a single server to host 50M+ experiment records without terabytes of RAM — the payloads (which can be 5KB+ per point) are fetched from RocksDB only when a match is found, while the 96-byte quantized vectors and HNSW graph stay in RAM for fast search.

❌ **Antipattern: Using REST for high-throughput batch ingest.** REST serializes payloads as JSON, which can be 5–10× larger than the equivalent gRPC protobuf. For bulk upserts (>1000 points), gRPC reduces network bandwidth and parsing overhead significantly. Use REST for debugging and admin; use gRPC for everything else.

✅ **Design storage tiers based on access patterns.** Hot vectors (queried frequently) → in-memory. Warm vectors (queried occasionally) → Mmap. Cold payloads (metadata, rarely filtered alone) → on-disk RocksDB. Profile your workload's access patterns before committing to a storage configuration.

---

## 🎯 Key Takeaways

- **Collections** are isolated namespaces with fixed vector config (dimension, distance, HNSW params); use them for multi-tenancy or different embedding models — but don't create more than a few hundred collections per node
- **Points** combine vectors and JSON payloads; design payload schemas around your application's filter patterns (which fields are queried with `must`/`should`/`must_not`), not just your data model
- **HNSW** in Qdrant is integral to the collection — tune `m` (8–64) and `ef_construct` (≥200) at creation; these cannot be changed without rebuilding the collection
- **Payload indexes** (keyword, integer, float, geo, text) are essential for fast filtered search — unindexed filters degrade to $O(N)$ full payload scans
- Qdrant automatically selects **pre-filtering** or **post-filtering** based on filter selectivity; benchmark if your filter patterns are unusual or if you need to guarantee pre-filtering with `exact=True`
- **gRPC** (port 6334) is the production API of choice for latency and throughput — 2–5× faster than REST for batch operations
- **REST** (port 6333) is for debugging, admin, and browser/frontend integration
- Keep the **HNSW graph in RAM** at all costs; vectors can use Mmap and payloads can use RocksDB, but the graph must be hot for sub-millisecond navigation
- Qdrant's Rust implementation provides memory safety without GC pauses, a critical advantage for consistent sub-10ms p99 latency under load
- Design storage tiers (in-memory → Mmap → RocksDB) based on measured access patterns, not guesses

## References

- Qdrant Documentation: https://qdrant.tech/documentation/
- Qdrant GitHub: https://github.com/qdrant/qdrant
- Qdrant Python Client: https://github.com/qdrant/qdrant-client
- Qdrant Go Client: https://github.com/qdrant/go-client
- [[02 - Indexing Algorithms Deep Dive]] — HNSW algorithmic foundations
- [[06 - Qdrant II - Distributed and Cloud Deployment]] — Clustering, replication, cloud ops
- [[14 - Rust Engineering]] — Rust systems programming context

## 📦 Código de compresión

```python
"""
Qdrant I — Compression Script
Summarizes: collections, points, HNSW, payload indexes, filtering, storage.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)
import time

client = QdrantClient(host="localhost", port=6333, prefer_grpc=True)

def setup_collection(name: str, dim: int = 768):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config={"m": 16, "ef_construct": 128},
    )
    client.create_payload_index(name, "category", "keyword")
    client.create_payload_index(name, "price", "float")

def upsert_batch(name: str, points: list[PointStruct]):
    client.upsert(collection_name=name, points=points, wait=True)

def search_filtered(name: str, vec: list[float], category: str, max_price: float, k: int = 5):
    return client.search(
        collection_name=name,
        query_vector=vec,
        query_filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=category)),
                FieldCondition(key="price", range=Range(lt=max_price)),
            ]
        ),
        limit=k,
        with_payload=True,
    )

if __name__ == "__main__":
    setup_collection("demo")
    upsert_batch("demo", [
        PointStruct(id="1", vector=[0.1]*768, payload={"category": "a", "price": 10}),
        PointStruct(id="2", vector=[0.2]*768, payload={"category": "a", "price": 50}),
        PointStruct(id="3", vector=[0.3]*768, payload={"category": "b", "price": 30}),
    ])
    # Wait for background indexing.
    time.sleep(1)
    results = search_filtered("demo", [0.15]*768, "a", 30)
    for hit in results:
        print(f"ID={hit.id} score={hit.score:.4f} payload={hit.payload}")
```
