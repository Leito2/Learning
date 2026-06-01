# 🔍 2 - Indexing Algorithms Deep Dive

## 🎯 Learning Objectives
- Understand the design rationale behind Flat, IVF, HNSW, PQ, OPQ, DiskANN, and ScaNN indices
- Build a decision framework for selecting an index based on build time, query time, memory, and recall
- Explain how HNSW navigable graphs achieve polylogarithmic search complexity $O(\log N)$
- Compare quantization methods (PQ, OPQ) and their impact on memory vs accuracy
- Evaluate DiskANN and ScaNN as solutions for billion-scale and GPU-accelerated search
- Tune index parameters like `nlist`, `nprobe`, $M$, $\text{efConstruction}$, and $\text{efSearch}$ with confidence
- Quantify memory budgets for each index type using formulas, not guesswork

## Introduction

If vector search is the engine of modern AI retrieval, indexing algorithms are its transmission. Without them, every query would trigger a brute-force scan, making real-time semantic search, recommendation, and RAG impossible at scale. A naive linear scan over 100M 768D vectors takes seconds on modern hardware; a well-tuned HNSW index on the same data answers queries in milliseconds. This is not a marginal improvement — it is the difference between a product that ships and one that times out.

This note dissects the major algorithmic families — space partitioning (IVF), graph navigation (HNSW), vector quantization (PQ/OPQ), and hardware-aware compression (DiskANN, ScaNN) — that transform the $O(Nd)$ KNN problem into sublinear or near-constant time lookups. We analyze each algorithm through the same lens: **build cost, query cost, memory footprint, and recall**. Every production vector database (pgvector, Qdrant, Weaviate, Pinecone, Milvus) implements one or more of these algorithms under the hood.

This module follows [[01 - Vector Search Fundamentals]] (where we defined the KNN/ANN problem and distance metrics) and precedes [[03 - pgvector I - Core Operations and Indexing]] and [[05 - Qdrant I - Architecture and Collections]] (where these algorithms are packaged into production database engines). Understanding the algorithms *independently* of any database is critical: it allows you to tune parameters like `ef_construction`, `nlist`, and $M$ with confidence rather than guesswork. The same tuning logic applies whether you are using Faiss, pgvector, Qdrant, or Weaviate.

The core trade-off every index makes is between four resources: **build time** (how long to create the index), **query time** (latency per search), **memory** (RAM consumed), and **recall** (fraction of true nearest neighbors returned). No index optimizes all four simultaneously — your job is to pick the right trade-off for your workload. A batch deduplication pipeline may tolerate 1-second queries and high memory, while a real-time recommendation API needs sub-10ms latency and moderate recall. Understanding these trade-offs mathematically (not just intuitively) is what separates a tuned production deployment from a default-configuration prototype that breaks under load.

Throughout this note, we will use Faiss (Facebook AI Similarity Search) for code examples because it is the reference implementation for most indexing algorithms. However, every concept applies directly to pgvector's `ivfflat` and `hnsw`, Qdrant's built-in HNSW, and managed services like Pinecone. The parameter names differ slightly (`nlist` vs `lists`, `efConstruction` vs `ef_construct`), but the underlying mathematics is identical.

---

## 1. Flat Index and Inverted File Index (IVF)

### Theory

The **Flat index** is the brute-force baseline. It stores vectors in raw float32 format and computes exact distances at query time. Query latency is $O(Nd)$ with no build time and perfect recall (1.0). Flat indices are useful as gold-standard benchmarks for recall measurement and for small collections (<50k vectors) where simplicity trumps speed. In production, Flat is rarely used as the primary index but is indispensable as a validation tool.

The **Inverted File Index (IVF)** addresses scalability by partitioning the vector space into `nlist` clusters using k-means. At build time, the algorithm learns `nlist` centroids $\{c_1, c_2, \ldots, c_{\text{nlist}}\} \subset \mathbb{R}^d$ via k-means clustering on the training data. Each vector $v_i$ is assigned to its nearest centroid, creating inverted lists $L_j = \{v_i : \arg\min_k \|v_i - c_k\| = j\}$. At query time, the system computes distances from query $q$ to all centroids, selects the `nprobe` nearest centroids, and searches only the vectors in those lists. This reduces the candidate set from $N$ to approximately $(N / \text{nlist}) \times \text{nprobe}$ vectors.

The cost breakdown is: centroid lookup is $O(\text{nlist} \cdot d)$, per-probe list scanning is $O((N/\text{nlist}) \cdot d \cdot \text{nprobe})$, so total query cost is approximately $O(d \cdot (\text{nlist} + N \cdot \text{nprobe} / \text{nlist}))$. This is minimized when $\text{nlist} \approx \sqrt{N \cdot \text{nprobe}}$. The heuristic $\text{nlist} = \sqrt{N}$ is a common starting point.

```python
import faiss
import numpy as np

dim = 768
nlist = 100
nprobe = 10
N = 1_000_000

xb = np.random.randn(N, dim).astype('float32')
xb /= np.linalg.norm(xb, axis=1, keepdims=True)

flat_index = faiss.IndexFlatIP(dim)
flat_index.add(xb)

quantizer = faiss.IndexFlatIP(dim)
ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
ivf_index.train(xb)
ivf_index.add(xb)
ivf_index.nprobe = nprobe

xq = np.random.randn(1, dim).astype('float32')
xq /= np.linalg.norm(xq)

D_flat, I_flat = flat_index.search(xq, k=10)
D_ivf, I_ivf = ivf_index.search(xq, k=10)
recall = len(set(I_flat[0]) & set(I_ivf[0])) / 10.0
print(f"IVF recall@10 with nprobe={nprobe}: {recall:.2f}")

# Systematic nprobe tuning.
for probe in [1, 5, 10, 20, 50, 100]:
    ivf_index.nprobe = probe
    _, I = ivf_index.search(xq, k=10)
    r = len(set(I_flat[0]) & set(I[0])) / 10.0
    print(f"  nprobe={probe:3d}  recall={r:.2f}")
```

⚠️ **Train IVF on a representative sample — never on empty or tiny datasets.** k-means centroids learned on skewed data create imbalanced inverted lists where one cluster holds 10× more vectors than others, effectively nullifying the pruning benefit. Use at least $30 \times \text{nlist}$ training vectors.

💡 **IVF is static; Flat is dynamic.** Adding vectors after IVF is built does not update centroids — new vectors are assigned to the nearest existing centroid, which may be a poor fit. For dynamic datasets, prefer HNSW or rebuild IVF periodically.

**Caso real — Meta (Facebook) photo search:** Early versions of Meta's photo search used IVF-based indices over visual feature vectors extracted from CNN classifiers. IVF was chosen because it offered a simple memory-to-recall trade-off and integrated well with their existing k-means clustering infrastructure. Their engineering team tuned `nprobe` per workload: *exploratory search* (user browsing photos) used `nprobe=20` for higher recall, while *batch deduplication* (finding near-duplicate uploads) used `nprobe=5` for speed. They monitored inverted list size distribution as a quality metric — a coefficient of variation >1.0 triggered index rebuild. A critical lesson from their deployment: IVF with imbalanced clusters (one list holding 40% of data) performed worse than a brute-force scan because the centroid lookup overhead plus the skewed list scan was slower than linear scan on their hardware.

**Memory accounting for IVF:**
- Raw vectors: $N \times d \times 4$ bytes (float32)
- Centroids: $\text{nlist} \times d \times 4$ bytes
- Inverted list pointers: $N \times 4$ bytes (int32 per vector-to-list assignment)
- Total for 1M vectors, 768D, nlist=100: ~3.07 GB + negligible overhead
- No extra graph edges (unlike HNSW), which makes IVF attractive for memory-constrained deployments

❌ **Antipattern: Assuming `nlist = sqrt(N)` is optimal for all datasets.** This heuristic assumes uniform data distribution. For clustered data (e.g., embeddings grouped by language or category), fewer centroids may suffice because each cluster is already dense. For uniformly distributed data, more centroids help. Benchmark 3–4 `nlist` values (e.g., `sqrt(N)/2`, `sqrt(N)`, `sqrt(N)*2`) before choosing.

✅ **Validate IVF with a train-test split.** Hold out 10% of your data as a query set, build IVF on 90%, and measure recall@k against exact Flat results on the full dataset. This gives you an unbiased estimate of production recall before you commit to index parameters.

✅ **Always keep a Flat index as your recall baseline.** Run a weekly comparison of IVF recall@k against the Flat exact results. A drop >0.05 indicates data drift and should trigger retraining.

---

## 2. Hierarchical Navigable Small World (HNSW)

### Theory

HNSW is a graph-based ANN algorithm introduced by Malkov and Yashunin in 2016. It builds a **multi-layer navigable small-world graph** where each layer $L_i$ is a sparse subgraph of the layer below. Layer 0 contains every vector and its local neighborhood edges (degree $M$); higher layers contain exponentially fewer nodes — a node appears in layer $L_i$ with probability $p^i$ where $p$ is typically 0.5 (inverse of $M_L$ parameter). This creates a hierarchy where the top layer has only a handful of nodes acting as "highways" for long-distance jumps.

The search algorithm is elegant: starting at a random entry point in the top layer, it greedily traverses to the nearest neighbor within the current layer by evaluating all neighbors at each step. When a local minimum is reached, it drops to the next layer using the current node as the entry point, repeating until Layer 0. At Layer 0, a beam search (controlled by $\text{efSearch}$) refines the result by maintaining a dynamic list of candidates and exploring their neighbors. The greedy descent ensures logarithmic scaling: in expectation, each layer reduces the distance to the query by a constant factor, yielding $O(\log N \cdot M \cdot \text{efSearch})$ complexity.

Three parameters control the trade-off:
- **$M$**: Maximum edges per node (bidirectional). Range: 8–64. Higher $M$ = better recall, more memory (~$M \times 8$ bytes per node for edge storage), slower build.
- **$\text{efConstruction}$**: Beam width during insertion. Range: 100–500. Higher = better graph quality, slower build. This is the single most important parameter for production quality.
- **$\text{efSearch}$**: Beam width at query time. Range: 64–512. Higher = better recall, slower query.

```python
import faiss
import numpy as np

dim = 768
M = 16
efConstruction = 200
efSearch = 128
N = 500_000

xb = np.random.randn(N, dim).astype('float32')
xb /= np.linalg.norm(xb, axis=1, keepdims=True)

index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = efConstruction
index.add(xb)
index.hnsw.efSearch = efSearch

xq = np.random.randn(1, dim).astype('float32')
xq /= np.linalg.norm(xq)
D, I = index.search(xq, k=10)
print(f"HNSW top-10 neighbors: {I[0]}")

# Memory accounting.
vec_mb = index.ntotal * dim * 4 / 1e6
graph_mb = index.ntotal * M * 2 * 4 / 1e6
print(f"Vectors: {vec_mb:.1f} MB | Graph edges: {graph_mb:.1f} MB")

# efSearch sweep.
for ef in [16, 32, 64, 128, 256, 512]:
    index.hnsw.efSearch = ef
    _, I = index.search(xq, k=10)
    r = len(set(I_flat[0]) & set(I[0])) / 10.0
    print(f"  efSearch={ef:3d}  recall={r:.2f}")
```

⚠️ **Default `efConstruction` (40-64) is too low for production.** It produces a low-quality graph with poor connectivity. Increasing `efConstruction` from 40 to 200 may double build time but dramatically improves the recall-vs-latency Pareto frontier for all subsequent queries. You cannot fix a bad graph at query time. A poorly built HNSW graph may require `efSearch=512` to achieve the same recall that a well-built graph achieves at `efSearch=128` — a 4× latency penalty for a one-time build-time optimization that costs seconds.

**Caso real — Yandex search infrastructure:** The creators of HNSW originally deployed it at Yandex for image similarity search across hundreds of millions of web images. Their production configuration uses `M=32` and `efConstruction=400`, tolerating a build time of several hours for a 200M-point index. This investment pays off: the index serves 10,000+ queries per second with p99 latency under 15ms at 99% recall. Yandex's key insight was that HNSW build time is a one-time cost amortized over millions of queries — spending extra hours during construction saved milliseconds on every query forever.

💡 **"Build wide, search wide."** — `efConstruction` is a one-time investment amortized over millions of queries. Spend the build time.

**Caso real — Pinecone and Qdrant:** Every major vector database uses HNSW as their default index. Pinecone, Weaviate, Qdrant, and pgvector all implement variants. Spotify uses HNSW with `M=32` and `efSearch=200` for music recommendation, achieving sub-10ms latency at 95%+ recall on 50M+ track embeddings. HNSW's incremental insertion (no retraining needed for new data) was the deciding factor over IVF — Spotify ingests 20,000+ new tracks daily, and IVF would require periodic k-means retraining to avoid centroid drift.

❌ **Antipattern: Setting `efSearch` too high for interactive workloads.** At `efSearch=500`, recall may hit 99%, but p99 latency can spike to 150ms+. For user-facing search, set `efSearch` to achieve the latency SLA first, then measure recall. You can always increase `efSearch` for offline batch queries.

✅ **Per-query `efSearch` tuning is a superpower.** Your exploratory UI can use `efSearch=64` (fast, ~90% recall), while your RAG pipeline uses `efSearch=256` (slower, ~97% recall). Databases like Qdrant and pgvector support setting this per request.

---

## 3. Product Quantization (PQ), OPQ, DiskANN, and ScaNN

### Theory

**Product Quantization (PQ)** attacks the memory bottleneck of storing full float32 vectors. The core idea: split each $d$-dimensional vector into $m$ sub-vectors of size $d/m$, and quantize each sub-vector independently using a small codebook of $k^*$ centroids learned via k-means on that subspace:

$$\text{PQ}(v) = [q_1(v^{(1)}), q_2(v^{(2)}), \ldots, q_m(v^{(m)})]$$

Each $q_i(v^{(i)})$ returns a centroid index (1 byte if $k^*=256$). The full vector is represented by $m$ bytes — for $d=768$ and $m=96$, compression is $768 \times 4 / 96 = 32\times$. Distance computation uses *asymmetric distance computation* (ADC): the query $q$ remains in full precision, and for each database vector $v$, the distance is computed as:

$$D(q, v) \approx \sum_{j=1}^{m} \|q^{(j)} - c_j^{\text{idx}_j}\|^2$$

where $c_j^{\text{idx}_j}$ is the centroid for the $j$-th sub-vector. This is computed via table lookup rather than full vector operations, making it extremely fast.

**Optimized Product Quantization (OPQ)** adds a learned rotation matrix $R \in \mathbb{R}^{d \times d}$ before PQ:

$$\text{OPQ}(v) = \text{PQ}(Rv)$$

The rotation aligns the data distribution with the quantization grid, making the subspaces more independent and reducing quantization error. OPQ consistently outperforms raw PQ and is the default in Faiss production pipelines.

**DiskANN** (Microsoft Research) solves billion-scale search when RAM is insufficient. It uses a **Vamana graph** (a variant of HNSW with a robust pruning algorithm) kept in RAM, while PQ-compressed vectors reside on SSD. Search navigates the graph in RAM and fetches compressed vectors from SSD on demand, decompressing them for distance computation. The key numbers: a billion 768D vectors in PQ format take ~96GB (vs 3TB raw), the Vamana graph needs ~16GB of RAM, and a single NVMe SSD can store the vectors for ~$100.

**ScaNN** (Google Research) takes a hardware-aware approach using *anisotropic vector quantization* (different quantization error tolerance per dimension) and coarse clustering with asymmetric hashing, heavily optimized for AVX2/AVX-512 SIMD instructions. ScaNN dominates the ANN benchmarks leaderboard for batch query throughput on modern CPUs.

```python
import faiss
import numpy as np

dim = 768
N = 1_000_000
xb = np.random.randn(N, dim).astype('float32')

M = 16
nbits = 8
coarse_nlist = 1024

quantizer = faiss.IndexFlatL2(dim)
ivf_pq = faiss.IndexIVFPQ(quantizer, dim, coarse_nlist, M, nbits)
opq = faiss.OPQMatrix(dim, M)
index = faiss.IndexPreTransform(opq, ivf_pq)

index.train(xb)
index.add(xb)
index.nprobe = 50

xq = np.random.randn(1, dim).astype('float32')
D, I = index.search(xq, k=10)
print(f"IVF+OPQ+PQ top-10: {I[0]}")

flat_bytes = N * dim * 4
pq_bytes = N * M
print(f"Flat: {flat_bytes/1e6:.0f} MB | PQ: {pq_bytes/1e6:.0f} MB | Ratio: {flat_bytes/pq_bytes:.0f}x")

# Reconstruction error check.
reconstructed = np.zeros((10, dim), dtype=np.float32)
for i in range(10):
    codes = ivf_pq.sa_decode(ivf_pq.sa_encode(xb[i:i+1]))
    reconstructed[i] = codes[0]
mse = np.mean((xb[:10] - reconstructed) ** 2)
print(f"PQ reconstruction MSE: {mse:.6f}")
```

⚠️ **PQ needs room to breathe per subspace.** If $M$ is too large (e.g., 64 for 768D), each subspace is only 12D and k-means finds unstable centroids. If `nbits` is too small (e.g., 4), the codebook has only 16 entries and quantization error is severe. Aim for subspace dimension $\geq 24$ and `nbits=8` (256 centroids).

💡 **Always benchmark reconstruction MSE before accepting a PQ config.** A rule of thumb: MSE should be <0.01 for normalized unit vectors. Monitor MSE on a held-out validation set; degradation indicates the quantization is destroying ranking fidelity. A more direct validation: measure recall@10 of PQ-approximate distances vs exact distances on a query set. If recall drops below 0.90, reduce $m$ (fewer sub-quantizers, higher memory, better fidelity).

**Caso real — Microsoft Bing's DiskANN:** Bing uses DiskANN to power vector search over billions of web document embeddings on single commodity servers. By keeping only the Vamana graph (~8GB for 1B points) in RAM and PQ-compressed vectors (~96GB) on NVMe SSD, they achieve sub-5ms median latency. The cost savings vs all-in-RAM are dramatic: 3TB of DRAM would cost ~$30,000/server; DiskANN's approach uses ~$100 of NVMe storage. This cost structure made billion-scale semantic search economically viable for production rather than a research curiosity.

**Caso real — Google's ScaNN deployment in Search:** Google uses ScaNN internally for batch re-ranking pipelines where thousands of queries are scored simultaneously against billions of documents. ScaNN's anisotropic quantization is specifically designed for this batch setting: it allocates more bits to dimensions that are more important for ranking, improving recall at the same memory budget. In Google's production benchmarks, ScaNN achieves 2-3x higher QPS than IVF+OPQ at equivalent recall on TPU-hosted serving infrastructure.

❌ **Antipattern: Using raw PQ without OPQ.** Raw PQ assumes data distribution aligns with Cartesian axes, which is rarely true for real embeddings (the semantic manifold is not axis-aligned). OPQ's learned rotation always reduces quantization error with negligible overhead — always use OPQ over raw PQ.

✅ **Algorithm selection decision tree:**
- $N < 100k$? → **Flat** (simple, exact, no build time)
- RAM tight, static data, batch queries? → **IVF + PQ/OPQ** (train once, query efficiently)
- Real-time, frequent inserts/updates, high recall? → **HNSW** (incremental, graph-based)
- $N > 10^9$, SSD available? → **DiskANN** (RAM-efficient, SSD-scalable)
- TPU/GPU cluster for batch inference? → **ScaNN** (SIMD-optimized, batch throughput)

### Additional Considerations

**Memory budget sizing:** A Flat index on 100M vectors of 768D requires $100M \times 768 \times 4 = 307$ GB of RAM — more than most production servers carry. IVF with the same vectors uses identical memory (raw float32 storage). HNSW adds ~$N \times M \times 16$ bytes for graph edges: with $M=16$, that is ~25 GB of additional overhead. PQ with $m=96$ reduces vector storage to $100M \times 96 = 9.6$ GB plus negligible codebook storage. This is why PQ-based indices dominate billion-scale deployments.

| Index | Memory Formula (N vectors, d dimensions) | Example: 100M × 768D |
|---|---|---|
| Flat | $N \cdot d \cdot 4$ | 307 GB |
| IVF-Flat | $N \cdot d \cdot 4 + \text{nlist} \cdot d \cdot 4$ | ~307 GB |
| HNSW | $N \cdot d \cdot 4 + N \cdot M \cdot 16$ | ~332 GB (M=16) |
| IVF+PQ (m) | $N \cdot m + \text{codebooks}$ | ~9.6 GB (m=96) |
| DiskANN (graph+PQ) | $N \cdot M \cdot 8 + N \cdot m$ | ~17.6 GB (M=32, m=96) |

**Build time comparison:** Flat = instant (zero build). IVF = $O(N \cdot \text{nlist} \cdot d \cdot \text{iterations})$ for k-means plus $O(N \cdot d)$ for assignment. HNSW = $O(N \cdot (M \cdot \text{efConstruction} + \log N))$ for graph construction — at $M=16, \text{efConstruction}=200$, this is roughly 2–5× slower than IVF for the same dataset. PQ/OPQ adds codebook learning: $O(N \cdot d \cdot m \cdot k^*)$ for k-means in $m$ subspaces. DiskANN's Vamana build is similar to HNSW but with an additional robust pruning step that can double build time.

### Index Selection Decision Matrix

| Workload Profile | Best Index | Why |
|---|---|---|
| <50k vectors, prototyping | Flat | Zero build time, 100% recall, simple |
| 50k–10M, static data, memory tight | IVF + PQ/OPQ | Good compression, trained once, balanced speed |
| 50k–100M, dynamic inserts, high recall | HNSW | Incremental updates, best recall-vs-latency Pareto |
| 100M+, SSD available, single node | DiskANN | RAM-efficient graph + SSD vectors, billion-scale |
| Batch re-ranking, TPU/GPU cluster | ScaNN | SIMD-optimized, best batch throughput |
| Need exact recall at any scale | Flat subset + tiered search | Search small candidate set exactly, prune with ANN

---

## 🎯 Key Takeaways

- **Flat** is the exact baseline; use it for small datasets and recall validation — never for production at scale beyond 100k vectors
- **IVF** partitions space via k-means into `nlist` clusters; tune `nlist` (~$\sqrt{N}$) and `nprobe` for your latency/recall target; requires periodic retraining for dynamic data because centroids become stale as distribution shifts
- **HNSW** is the production gold standard — graph-based navigation with $O(\log N)$ search complexity and incremental (online) inserts; invest in `efConstruction` at build time (≥200); you cannot fix a bad graph at query time
- **PQ/OPQ** compress vectors by 10–50× via sub-vector quantization; always use OPQ's learned rotation; benchmark reconstruction MSE before accepting a configuration
- **DiskANN** extends graph search to SSD with the Vamana algorithm, enabling billion-vector search on a single machine without terabytes of RAM — the defining innovation for cost-effective large-scale deployment
- **ScaNN** is purpose-built for batch throughput on SIMD/GPU hardware; ideal for cloud-scale reranking and offline scoring pipelines
- Memory budget is often the binding constraint in production; use the formulas in this note to estimate before selecting hardware
- No index is universally optimal — benchmark recall, latency, and memory on your specific data distribution before committing to a decision
- Document index parameters alongside your schema with reasoning comments (e.g., why you chose $M=32$ over $M=16$, or why PQ $m=96$ was selected given your 64GB RAM budget)

## References

- Y. Malkov, D. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE TPAMI, 2018
- H. Jégou et al. "Product Quantization for Nearest Neighbor Search." IEEE TPAMI, 2011
- S. Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS, 2019
- Google Research. "ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization." ICML, 2020
- Faiss documentation: https://github.com/facebookresearch/faiss/wiki
- Microsoft Research. "DiskANN: Graph-based Indices on SSD." https://github.com/microsoft/DiskANN
- [[01 - Vector Search Fundamentals]] — ANN motivation and distance metrics
- [[03 - pgvector I - Core Operations and Indexing]] — pgvector's ivfflat and hnsw implementation
- [[05 - Qdrant I - Architecture and Collections]] — HNSW in Qdrant's production engine

## 📦 Código de compresión

```python
"""
Indexing Algorithms Deep Dive — Compression Script
Summarizes: Flat, IVF, HNSW, PQ, OPQ trade-offs with recall benchmarking.
"""
import faiss
import numpy as np
import time

class VectorIndexBenchmark:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.indices = {}

    def build_flat(self, vectors: np.ndarray):
        t0 = time.time()
        idx = faiss.IndexFlatIP(self.dim)
        idx.add(vectors)
        self.indices["flat"] = idx
        return time.time() - t0

    def build_ivf(self, vectors: np.ndarray, nlist: int = 100, nprobe: int = 10):
        t0 = time.time()
        q = faiss.IndexFlatIP(self.dim)
        idx = faiss.IndexIVFFlat(q, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(vectors)
        idx.add(vectors)
        idx.nprobe = nprobe
        self.indices["ivf"] = idx
        return time.time() - t0

    def build_hnsw(self, vectors: np.ndarray, M: int = 16, efConstruction: int = 200):
        t0 = time.time()
        idx = faiss.IndexHNSWFlat(self.dim, M)
        idx.hnsw.efConstruction = efConstruction
        idx.add(vectors)
        idx.hnsw.efSearch = 128
        self.indices["hnsw"] = idx
        return time.time() - t0

    def recall_vs_exact(self, query: np.ndarray, k: int = 10):
        _, exact_I = self.indices["flat"].search(query, k)
        results = {}
        for name, idx in self.indices.items():
            if name == "flat":
                continue
            _, I = idx.search(query, k)
            overlap = len(set(exact_I[0]) & set(I[0]))
            results[name] = overlap / k
        return results

if __name__ == "__main__":
    dim, N = 768, 200_000
    vecs = np.random.randn(N, dim).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bench = VectorIndexBenchmark(dim)
    print(f"Flat build: {bench.build_flat(vecs):.2f}s")
    print(f"IVF build:  {bench.build_ivf(vecs):.2f}s")
    print(f"HNSW build: {bench.build_hnsw(vecs):.2f}s")
    q = np.random.randn(1, dim).astype("float32")
    q /= np.linalg.norm(q)
    print("Recall vs Flat:", bench.recall_vs_exact(q, k=10))
```
