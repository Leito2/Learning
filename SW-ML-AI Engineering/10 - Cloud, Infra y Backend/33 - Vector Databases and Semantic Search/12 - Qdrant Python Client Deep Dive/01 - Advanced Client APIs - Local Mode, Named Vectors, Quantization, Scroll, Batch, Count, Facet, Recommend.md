# 🧰 Advanced Client APIs — Local Mode, Named Vectors, Quantization, Scroll, Batch, Count, Facet, Recommend

## 🎯 Learning Objectives

- Run Qdrant **in-process** with `QdrantClient(path=...)` for unit tests, CI, and single-tenant edge deployments — no Docker required
- Configure **named vectors** (multi-vector per point) for ColBERT, late chunking, and multi-modal embeddings in a single collection
- Configure **quantization at collection creation** (`ScalarQuantization`, `ProductQuantization`) to cut memory by 4× with <2% recall loss
- Use the **scroll API** to iterate over millions of points without loading the full collection into memory
- Apply the **search_batch** and **count** APIs for offline evaluation and cardinality estimation
- Build **faceted search** with the `facet` API to return per-category counts alongside hits
- Implement **recommend** with positive and negative examples to power "more like this, less like that" UX
- Distinguish the **`search` (deprecated) vs `query_points` (current)** API and migrate code

---

## Introduction

The fundamentals course taught you to call `QdrantClient("localhost:6333").search(...)` with a single vector and a payload filter. That is the right starting point, and it covers 80% of RAG and 60% of recommendation workloads. The remaining 40% of production code reaches for client APIs the basics never showed you.

This note is a **reference**, not a narrative. Each section introduces one advanced API: when to reach for it, the model, the call shape, a runnable example, and the failure modes. Read it linearly the first time; jump by section thereafter. By the end you will know that `query_points` is the current name for what used to be `search` (Qdrant 1.10, mid-2024), that `recommend` is a different mathematical operation than `search` even though the response shape is similar, and that the local mode and server mode share 95% of the surface — but the 5% that differs is exactly what makes the difference between a test suite that runs in 3 seconds and one that takes 3 minutes.

We connect to [[../05 - Qdrant I - Architecture and Collections]] (basics), [[../10 - Advanced Patterns and Observability]] (hybrid search uses the named-vector surface we cover here), and `[[../../35 - Vector Quantization/01 - Product Quantization]]` (the math behind the `ProductQuantization` config). The capstone `[[../12 - Qdrant Python Client Deep Dive/03 - Capstone]]` shows these APIs in a full RAG production stack.

---

## 1. Local Mode — In-Process Qdrant for Tests and Edge

### The problem with the test pyramid

The basics taught you to point at `localhost:6333` — which is fine for development, fatal for unit tests. CI pipelines either need a sidecar Qdrant container (slow, fragile) or a heavy mock (drifts from real behavior). Local mode solves this by embedding the Qdrant engine in the Python process. The same `QdrantClient` API works against a local directory; tests run in 3 seconds and exercise the real query planner.

```python
from qdrant_client import QdrantClient

# Production: HTTP/gRPC server
client = QdrantClient(host="qdrant.prod.internal", port=6333, prefer_grpc=True)

# Tests / single-tenant edge: in-process
client = QdrantClient(path="./qdrant_data")  # local mode, no Docker

# In-memory only (no disk persistence)
client = QdrantClient(location=":memory:")
```

### What works in local mode and what does not

| Capability | Local mode | Server mode |
|------------|:---:|:---:|
| CRUD on collections / points | ✓ | ✓ |
| `search`, `query_points`, `scroll`, `count`, `facet`, `recommend` | ✓ | ✓ |
| `Batch` (multi-segment upload) | ✓ | ✓ |
| Named vectors, quantization config | ✓ | ✓ |
| Sharded collections (`shard_number>1`) | ❌ (single-node only) | ✓ |
| Replication | ❌ | ✓ |
| Distributed deployment | ❌ | ✓ |

> **Caso real**: StayBot's `tests/` directory uses `QdrantClient(location=":memory:")` for unit tests, and the embedded edge build (`pip install staybot[edge]`) ships with `QdrantClient(path="/var/lib/staybot/qdrant")` for single-tenant Raspberry Pi deployments. Both reuse the same `retrieve.py` module that production uses against a 3-node cluster. No code branches per deployment.

### Persistence and cleanup

```python
import shutil
from pathlib import Path

@pytest.fixture
def qdrant():
    data_dir = Path("./test_qdrant")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    client = QdrantClient(path=str(data_dir))
    yield client
    client.close()
    shutil.rmtree(data_dir)
```

`QdrantClient.close()` releases the in-process engine. Without it, the process hangs on test exit because the embedded server holds a thread that does not get joined by pytest.

---

## 2. Named Vectors — Multi-Vector per Point

### Why named vectors exist

The basics taught you to store a single vector per point. The model breaks when you need to combine:

- A **dense** embedding (e.g., 768-dim `bge-base`)
- A **sparse** embedding (BM25-style, for hybrid search)
- A **late-interaction** vector (ColBERT, 128-dim-per-token × N tokens)
- A **cross-modal** embedding (CLIP, 512-dim for text and images)

You could concatenate them into one 2,000-dim vector, but you lose the ability to query against a single embedding type and you cannot update one without rewriting the point. Named vectors solve this by giving each vector a name:

```python
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, SparseVectorParams,
)

client.create_collection(
    collection_name="multimodal_docs",
    vectors_config={
        "dense_bge":    VectorParams(size=768,  distance=Distance.COSINE),
        "colbert":      VectorParams(size=128,  distance=Distance.COSINE, multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)),
    },
    sparse_vectors_config={
        "bm25": SparseVectorParams(),
    },
)

client.upsert(
    collection_name="multimodal_docs",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense_bge":   [0.12] * 768,
                "colbert":     [[0.05] * 128, [0.07] * 128, [0.09] * 128],  # 3 tokens
            },
            sparse_vectors={"bm25": {"indices": [101, 340, 1023], "values": [1.5, 2.0, 0.8]}},
            payload={"title": "Late Chunking with ColBERT", "year": 2024},
        ),
    ],
)
```

Querying against one named vector:

```python
results = client.query_points(
    collection_name="multimodal_docs",
    query=[0.10] * 768,                # dense_bge query
    using="dense_bge",                  # tell Qdrant which vector to use
    limit=10,
).points
```

> ⚠️ **Advertencia**: if you query with a single flat list but the collection has named vectors, you get `ValueError: expected named vectors`. The shape must match.

### Multi-vectors for late interaction (ColBERT)

ColBERT produces **one vector per token** instead of one vector per document. Qdrant supports this natively via `MultiVectorConfig`:

```python
from qdrant_client.models import MultiVectorConfig, MultiVectorComparator

VectorParams(
    size=128,
    distance=Distance.COSINE,
    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
)
```

The `MAX_SIM` comparator implements the ColBERT scoring: sum the maximum similarity of each query token to any document token. This is the basis of the high-recall late-interaction retrieval covered in `[[../../../06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference/10 - Capstone - High-Performance RAG with ColBERT and SGLang]]`.

---

## 3. Quantization Configuration — Cut Memory 4× at <2% Recall Loss

### Why quantize at all

A 768-dim `float32` vector is 3 KB. A 100M-vector collection is 300 GB. A 1B-vector collection is 3 TB. At those scales, the HBM in your serving nodes becomes the bottleneck. Quantization compresses each vector to fewer bits while keeping the inner-product math fast enough for real-time retrieval. The trade-off is recall: at very high compression (4-bit or 1-bit), you lose 2-5% recall@10.

### Scalar vs Product quantization

| Type | Bits per dim | Compression | Recall loss | Build time |
|------|-------------:|------------:|------------:|-----------:|
| None (float32) | 32 | 1× | 0% | 1× |
| `ScalarQuantization` (`int8`) | 8 | 4× | <1% | 1.1× |
| `ScalarQuantization` (`int4`) | 4 | 8× | 1-2% | 1.2× |
| `ProductQuantization` (m=8, nbits=8) | ~8 | 4× | <1% | 3-5× |
| `BinaryQuantization` | 1 | 32× | 3-5% | 1.3× |

The math is in `[[../../35 - Vector Quantization/01 - Product Quantization]]`. The code is straightforward:

```python
from qdrant_client.models import (
    ScalarQuantization, ScalarType, QuantizationSearchParams,
    ProductQuantization, CompressionRatio,
)

# Scalar int8 — the production default
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarType.INT8,
        quantile=0.99,           # ignore outliers; 99% of vectors fit in 8 bits
        always_ram=True,        # keep the quantized vectors in RAM, raw on disk
    ),
)

# Product quantization — better for very large collections
client.create_collection(
    collection_name="web_pages",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    quantization_config=ProductQuantization(
        product=CompressionRatio.X4,   # 4× compression
        always_ram=True,
    ),
)
```

### When to keep raw vectors vs always_ram

- `always_ram=True` (default for `int8`): the quantized vectors live in RAM for fast scoring; raw vectors live on disk. Best for serving p99 < 10ms.
- `always_ram=False`: both quantized and raw on disk. Best for cold storage with infrequent re-ranking.

### Rescoring with `QuantizationSearchParams`

When the user has time budget, the best pattern is **search with quantization + rescore with raw vectors**:

```python
results = client.query_points(
    collection_name="products",
    query=[0.1] * 768,
    search_params=QuantizationSearchParams(
        ignore=False,
        rescore=True,             # re-rank top-K candidates with raw vectors
        oversampling=2.0,         # retrieve 2× more candidates from quantized index
    ),
    limit=10,
)
```

`rescore=True` re-ranks the top candidates with the raw float32 vectors. Recall goes back to 99%+ with the same 4× memory reduction. This is the production answer for high-recall workloads.

> 💡 **Tip**: Pair `ScalarQuantization(int8, always_ram=True)` with `rescore=True` for the best 2025-2026 trade-off. Avoid `BinaryQuantization` unless you really need 32× compression and can tolerate 3-5% recall loss.

---

## 4. Scroll, Batch, Count, Facet, Recommend

### `scroll` — paginated iteration over millions of points

`scroll` returns points in batches with a `next_page_offset` token. Use it to back up a collection, stream re-embeddings, or build offline pipelines.

```python
offset = None
while True:
    points, next_offset = client.scroll(
        collection_name="products",
        scroll_filter=Filter(must=[FieldCondition(key="year", range=Range(gte=2024))]),
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False,        # skip vectors for memory efficiency
    )
    process_batch(points)
    if next_offset is None:
        break
    offset = next_offset
```

### `search_batch` and `query_batch`

For offline evaluation, you often have thousands of queries to run. `search_batch` runs them in a single gRPC call (much faster than 1000 sequential `search` calls).

```python
queries = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
results = client.query_batch_points(
    collection_name="products",
    queries=[
        QueryRequest(query=q, limit=10) for q in queries
    ],
)
# results is a list of lists of ScoredPoint — same shape as `search`, batched
```

### `count` — cardinality estimation

```python
n = client.count(
    collection_name="products",
    count_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="books"))]),
)
print(f"Books in the index: {n.count}")
```

Internally, this is a fast O(payload-index) operation when the filter uses indexed payload fields. Faster than `scroll(...).len()` by 100×.

### `facet` — per-category counts alongside hits

The "faceted search" pattern: user types a query, server returns both the top hits and the count of hits per category so the UI can show "1,234 results across 23 categories":

```python
hits = client.query_points(
    collection_name="products",
    query=[0.1] * 768,
    limit=10,
    facet={"key": "category", "limit": 20},     # top 20 categories
)
# hits.facets is a dict: {"category": {"books": 1234, "movies": 567, ...}}
```

### `recommend` — "more like this, less like that"

Different math from `search`: instead of a query vector, you give positive examples (vectors of things the user liked) and negative examples (vectors of things the user did not like). The result is the centroid-based "average of positives minus average of negatives" vector, scored against the collection.

```python
hits = client.recommend(
    collection_name="products",
    positive=[42, 88, 1024],          # IDs the user liked
    negative=[17, 99],               # IDs the user disliked
    limit=10,
    with_payload=True,
)
```

`recommend` is the right answer for:

- "More papers like this one" (multi-agent research agent)
- "Customers who bought X also bought Y" (e-commerce)
- "Other tracks by this artist" (music recommendation)

⚠️ **Advertencia**: `recommend` is **not** a substitute for `search` with a query vector. They have different math, different recall characteristics, and different latency. Use `search` for explicit semantic queries; use `recommend` for collaborative-filtering-style UX.

### `query_points` vs `search` (the API migration)

Qdrant 1.10 (mid-2024) renamed `search` to `query_points` and unified all the query operations (search, recommend, discover, context) under a single API. `search` still works as a deprecated alias:

```python
# Old (still works, deprecated in 1.10, removed in 1.13+)
results = client.search(collection_name="products", query_vector=[0.1]*768, limit=10)

# New (current, use this)
results = client.query_points(collection_name="products", query=[0.1]*768, limit=10).points
```

`recommend`, `discover`, and `context` are also accessible via `query_points` with different `query` types. This is a forward-compatible migration: convert when you touch the file.

---

## 5. Production Reality

### Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| `client.search` raises `AttributeError` after upgrade to 1.13+ | Removed in 1.13 | Replace with `client.query_points(...).points` |
| Forgetting `using=` with named vectors | `ValueError: expected named vectors` | Pass `using="dense_bge"` to `query_points` |
| `always_ram=False` for int8 quantization | p99 > 50ms | Set `always_ram=True` to keep quantized vectors in RAM |
| `scroll` returning all vectors (3 KB × 100M) | OOM in worker | Set `with_vectors=False` unless you need the raw vectors |
| `recommend` used as semantic search | Bad recall | Use `query_points` for explicit semantic queries |
| Local mode in production by accident | No replication, single-node failure | Use `host=`/`port=` in production, `path=` only in tests/edge |
| `MultivectorConfig` not set on ColBERT collection | Late-interaction scoring fails silently | Explicitly set `MultiVectorConfig(comparator=MAX_SIM)` |

### Cost economics

Quantization is the single highest-leverage cost lever in Qdrant:

| Collection size | Raw float32 | int8 (4×) | int4 (8×) | PQ X4 |
|----------------|------------:|----------:|----------:|------:|
| 10M vectors (768-dim) | 30 GB | 7.5 GB | 4 GB | 7.5 GB |
| 100M vectors | 300 GB | 75 GB | 40 GB | 75 GB |
| 1B vectors | 3 TB | 750 GB | 400 GB | 750 GB |

At 1B vectors, the int4 path saves $1,500/month in HBM-equipped serving nodes (rough order of magnitude on AWS `r6id.32xlarge` or `x2idn.32xlarge` with NVMe).

### The 1.10 / 1.12 / 1.13 API migration path

| Version | Status | Migration |
|---------|--------|-----------|
| 1.8 (2023) | Released with `search` | Reference implementation |
| 1.10 (mid-2024) | `query_points` introduced as the unified API; `search` deprecated | New code uses `query_points` |
| 1.12 (early 2025) | Current stable; `search` still works with a deprecation warning | All examples in this note target 1.12 |
| 1.13+ (planned) | `search` removed | Migrate legacy code |

The migration is mechanical: `client.search(c, query_vector=q, ...)` → `client.query_points(c, query=q, ...).points`. Nothing else changes. The vault examples in [[../05 - Qdrant I - Architecture and Collections]] and [[../06 - Qdrant II - Distributed and Cloud Deployment]] use the old API for clarity; production code should use `query_points`.

---

## 6. Code in Practice

```python
# 📦 Qdrant advanced client — single file, exercises every API in this note.
# Requires: qdrant-client>=1.12.0, numpy

import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range,
    ScalarQuantization, ScalarType, QuantizationSearchParams,
    MultiVectorConfig, MultiVectorComparator, SparseVectorParams,
    QueryRequest, MultiQueryRequest,
)

# 1. Local mode (in-process) for tests
client = QdrantClient(location=":memory:")

# 2. Named vectors + quantization at collection creation
client.create_collection(
    collection_name="multimodal_docs",
    vectors_config={
        "dense_bge": VectorParams(
            size=768, distance=Distance.COSINE,
            quantization_config=ScalarQuantization(scalar=ScalarType.INT8, quantile=0.99, always_ram=True),
        ),
        "colbert": VectorParams(
            size=128, distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    },
    sparse_vectors_config={"bm25": SparseVectorParams()},
)

# 3. Insert 1000 synthetic multimodal points
rng = np.random.default_rng(42)
points = []
for i in range(1000):
    points.append(PointStruct(
        id=i,
        vector={
            "dense_bge": rng.normal(size=768).tolist(),
            "colbert":   [rng.normal(size=128).tolist() for _ in range(5)],   # 5 tokens
        },
        sparse_vectors={"bm25": {"indices": [int(x) for x in rng.integers(0, 10_000, 8)], "values": rng.uniform(0.5, 2.0, 8).tolist()}},
        payload={"doc_id": i, "category": ["books", "movies", "music"][i % 3], "year": 2020 + (i % 6)},
    ))

client.upsert(collection_name="multimodal_docs", points=points, wait=True)

# 4. query_points with named vector + rescore (int8 quantization + raw re-rank)
hits = client.query_points(
    collection_name="multimodal_docs",
    query=rng.normal(size=768).tolist(),
    using="dense_bge",
    query_filter=Filter(must=[FieldCondition(key="year", range=Range(gte=2024))]),
    search_params=QuantizationSearchParams(rescore=True, oversampling=2.0),
    limit=5,
    with_payload=True,
).points
print(f"Top-5 hits (with rescore): {[(h.id, h.score, h.payload['title'] if 'title' in h.payload else h.payload['category']) for h in hits]}")

# 5. facet — per-category counts alongside the hits
hits_with_facets = client.query_points(
    collection_name="multimodal_docs",
    query=rng.normal(size=768).tolist(),
    using="dense_bge",
    facet={"key": "category", "limit": 10},
    limit=5,
)
print(f"Top-5 hits + facet: {hits_with_facets.points[:1]}, facets={dict(list(hits_with_facets.facets.items())[:1]) if hits_with_facets.facets else None}")

# 6. scroll — paginated iteration
offset = None
total = 0
while True:
    batch, offset = client.scroll(
        collection_name="multimodal_docs",
        scroll_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="books"))]),
        limit=500, offset=offset, with_payload=True, with_vectors=False,
    )
    total += len(batch)
    if offset is None:
        break
print(f"Books in index (via scroll): {total}")

# 7. count — fast cardinality estimation
n_books = client.count(
    collection_name="multimodal_docs",
    count_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="books"))]),
)
print(f"Books (via count): {n_books.count}")

# 8. query_batch_points — bulk offline evaluation
queries = [rng.normal(size=768).tolist() for _ in range(100)]
batch_results = client.query_batch_points(
    collection_name="multimodal_docs",
    queries=[QueryRequest(query=q, using="dense_bge", limit=5) for q in queries],
)
print(f"Batch results: {len(batch_results)} queries × {len(batch_results[0])} hits each")

# 9. recommend — "more like these, less like those"
recs = client.recommend(
    collection_name="multimodal_docs",
    positive=[0, 1, 2],
    negative=[3, 4],
    limit=5,
    with_payload=True,
)
print(f"Recommendations: {[r.id for r in recs]}")

# 10. Tear down
client.delete_collection(collection_name="multimodal_docs")
client.close()
```

> **Caso real**: **Multi-Agent Research System** uses this exact pattern. The research agent's `vector_store.py` wraps `QdrantClient(location=":memory:")` for unit tests, `QdrantClient(host="qdrant.research.svc.cluster.local", prefer_grpc=True)` for the staging cluster, and a per-pipeline `QdrantClient(path=...)` for offline batch re-embeddings. Named vectors carry the dense embedding (for initial retrieval), the ColBERT multi-vector (for re-ranking), and the BM25 sparse vector (for keyword fallback). Quantization at int8 cuts the cluster's HBM footprint from 2.4 TB to 600 GB.

---

## 🎯 Key Takeaways

- **`QdrantClient(path=...)` runs Qdrant in-process** for unit tests, CI, and single-tenant edge deployments. Same API as the server client for 95% of operations.
- **Named vectors** let a single point hold dense + sparse + ColBERT embeddings under typed names, instead of concatenating into one mega-vector.
- **`ScalarQuantization(int8, always_ram=True)` cuts memory 4× with <1% recall loss**. Pair with `rescore=True` for 99%+ recall.
- **`scroll` paginates over millions of points** without loading the full collection. Use `with_vectors=False` unless you need the raw vectors.
- **`query_batch_points` and `query_points` are the new unified API** (`search` is deprecated, removed in 1.13+). Migrate when you touch the file.
- **`count` and `facet` are O(payload-index)** operations — 100× faster than `scroll(...).len()`.
- **`recommend` is collaborative filtering**, not semantic search. Use `query_points` for explicit semantic queries; use `recommend` for "more like this" UX.
- **`MultiVectorConfig(comparator=MAX_SIM)` is the ColBERT scoring** — required for late-interaction retrieval in Qdrant.

## References

- Qdrant Python client docs: https://python-client.qdrant.tech/
- Qdrant v1.12 release notes: https://github.com/qdrant/qdrant/releases/tag/v1.12.0
- Scalar and Product quantization internals: `[[../../35 - Vector Quantization/01 - Product Quantization - Theory, Code and Reconstruction Error]]`
- ColBERT and multi-vector retrieval: `[[../../../06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference/10 - Capstone - High-Performance RAG with ColBERT and SGLang]]`
- Related Vault: [[../05 - Qdrant I - Architecture and Collections]]
- Related Vault: [[../10 - Advanced Patterns and Observability]]
- Related Vault: [[../../../14 - Rust Engineering/04 - Rust for ML and AI/07 - Vector Databases in Rust (Qdrant, pgvector)]]
