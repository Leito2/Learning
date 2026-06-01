# 🔍 1 - Vector Search Fundamentals

## 🎯 Learning Objectives
- Master the concept of dense vector embeddings and their geometric interpretation in $\mathbb{R}^n$
- Compare and contrast Euclidean (L2), Cosine, Dot Product, and Manhattan (L1) distance metrics with mathematical precision
- Understand the trade-off between exact KNN ($O(Nd)$) and approximate ANN search
- Explain the curse of dimensionality and its impact on nearest-neighbor queries
- Evaluate embedding quality through distance distribution analysis and recall benchmarking
- Map vector search use cases to ML pipelines: semantic search, recommendation, anomaly detection, RAG

## Introduction

Modern AI systems do not search by keywords; they search by **meaning**. When a user asks "What is the best way to relax after work?" a traditional inverted index fails because the word "relax" may not appear in relevant documents. Vector databases solve this by translating text, images, or audio into high-dimensional dense vectors — embeddings — where semantic similarity becomes geometric proximity. This paradigm shift has enabled a new generation of applications: Retrieval-Augmented Generation (RAG) for LLMs, real-time recommendation systems, visual search, drug discovery, and anomaly detection.

This note establishes the mathematical and conceptual bedrock for the entire course. Every indexing algorithm, database extension, and deployment pattern we will study — from IVF cluster pruning to HNSW graph navigation to DiskANN's SSD-backed search — is an optimization of a single primitive: **given a query vector $q \in \mathbb{R}^d$, find its $k$ nearest neighbors in a collection of $N$ vectors**. Understanding *why* this problem is hard in high dimensions, and *how* distance metrics encode different notions of similarity, is essential before we touch any database syntax or index configuration.

This module connects directly to [[22 - NLP and Transformers]] (where embeddings are produced via transformer models like BERT and Sentence-T5) and [[06 - Large Language Models]] (where vector search powers Retrieval-Augmented Generation). We will also reference concepts from [[03 - Mathematics for ML]] when discussing norms, high-dimensional geometry, and the concentration of measure phenomenon.

---

## 1. Embeddings and Vector Spaces

### Theory

The idea of representing discrete objects as continuous vectors is not new. In 2013, word2vec demonstrated that neural networks could learn vector spaces where analogies like $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ held geometrically. The breakthrough of transformers (BERT, GPT, T5) scaled this from words to sentences, paragraphs, and multimodal objects (images, audio, video, graphs). An embedding is a learned function $f: X \to \mathbb{R}^n$ that maps an object $x$ into an $n$-dimensional space such that semantically similar objects map to nearby points.

The dimensionality $n$ typically ranges from 384 (sentence-transformers `all-MiniLM-L6-v2`) to 1536 (OpenAI `text-embedding-3-large`) or even 2048+ for vision models like CLIP. Why such high dimensions? Because natural language is combinatorially explosive — a 768-dimensional space can partition the semantic manifold with enough granularity to distinguish subtle meanings. The embedding space forms a **semantic manifold**: a lower-dimensional surface embedded in $\mathbb{R}^n$ where geodesic distance corresponds to semantic difference.

### Core Distance Metrics

The fundamental operation in vector search is: given query $q \in \mathbb{R}^d$ and collection $C = \{v_1, v_2, \ldots, v_N\}$ where $v_i \in \mathbb{R}^d$, find the $k$ vectors nearest to $q$ under distance metric $D$:

$$\text{KNN}(q, C, k) = \arg\min_{S \subseteq C, |S|=k} \sum_{v \in S} D(q, v)$$

Four distance metrics dominate production use:

- **Euclidean (L2)**: $D_{L2}(a, b) = \sqrt{\sum_{i=1}^d (a_i - b_i)^2}$. This is the straight-line distance in space. Best when absolute magnitude matters, such as image pixel embeddings or sensor data where scale carries information.

- **Cosine distance**: $D_{\cos}(a, b) = 1 - \frac{a \cdot b}{\|a\|\|b\|}$. This captures pure orientation, ignoring magnitude. Essential for text embeddings where document length varies but semantic direction is what matters.

- **Dot Product**: $D_{\text{dot}}(a, b) = -a \cdot b$ (negated for ascending ordering). Equivalent to cosine when vectors are normalized. The fastest metric to compute, used extensively in OpenAI's embedding API where outputs are pre-normalized.

- **Manhattan (L1)**: $D_{L1}(a, b) = \sum_{i=1}^d |a_i - b_i|$. Robust to outliers because it does not square differences. Useful in sparse or high-noise domains like genomics or network traffic analysis.

```python
import numpy as np
from numpy.linalg import norm

query = np.array([0.1, -0.3, 0.8, 0.02], dtype=np.float32)
collection = np.array([
    [0.2, -0.1, 0.9, 0.05],
    [-0.5, 0.8, 0.1, -0.3],
    [0.15, -0.25, 0.85, 0.0]
], dtype=np.float32)

def euclidean(a, b):
    return norm(a - b)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def dot_product(a, b):
    return np.dot(a, b)

def manhattan(a, b):
    return np.sum(np.abs(a - b))

distances = [euclidean(query, vec) for vec in collection]
k = 2
nearest_indices = np.argsort(distances)[:k]
print(f"Nearest neighbors (exact KNN): {nearest_indices}")

# Unit norm equivalence proof: if ||a|| = ||b|| = 1, then
# ||a - b||^2 = 2 - 2 cos(a,b), so L2 and cosine are monotonic.
a = query / norm(query)
b = collection[0] / norm(collection[0])
l2_sq = np.sum((a - b)**2)
cos = np.dot(a, b)
print(f"L2^2 = {l2_sq:.4f}, 2-2*cos = {2 - 2*cos:.4f}")  # identical
```

⚠️ **Use float32, not float64**, for embeddings. A 768D vector costs 3KB in float32 vs 6KB in float64. At 100M vectors, this saves 300GB of RAM with no measurable recall degradation.

💡 **Proof shortcut:** For unit vectors, $\|a - b\|^2 = 2 - 2\cos(a,b)$. Minimizing L2 is strictly equivalent to maximizing cosine similarity. This is why you can use either metric on normalized vectors — the rankings are identical.

**Caso real — Spotify Discover Weekly:** Spotify uses vector search to power its Discover Weekly playlist, serving 100M+ active users. Audio tracks are embedded into 768-dimensional vectors using convolutional neural networks trained on raw audio waveforms. When a user likes a song, Spotify performs ANN search to find tracks with similar acoustic fingerprints — not just similar genre tags or metadata. Cosine similarity is chosen over L2 because track duration varies from 30-second snippets to 10-minute epics; the orientation of the embedding vector captures timbre and rhythm, while magnitude correlates with recording volume and production loudness. By normalizing all embeddings to unit length, Spotify ensures that a quiet acoustic ballad and a loud rock anthem can still match on style.

❌ **Antipattern: Using L2 on unnormalized text embeddings.** Text embeddings from sentence-transformers are optimized for cosine space. L2 will over-penalize longer documents because their vectors have larger magnitudes, burying semantically relevant long articles under short irrelevant ones.

✅ **Always normalize text embeddings to unit length** before computing cosine or dot product. Most embedding APIs (OpenAI, Cohere, Voyage) return normalized vectors by default, but check your model's documentation.

---

## 2. ANN vs Exact KNN and the Curse of Dimensionality

### Theory

Exact K-Nearest Neighbor search has $O(Nd)$ time complexity per query for $N$ vectors of dimension $d$. For small $N$ (thousands), this is acceptable. But in production ML systems, $N$ is millions or billions. A single brute-force scan over 100M vectors in 768D requires computing 76.8 billion multiply-add operations — far too slow for interactive applications demanding <50ms latency.

Approximate Nearest Neighbor (ANN) search accepts a trade-off: **sacrifice a small amount of recall in exchange for orders-of-magnitude speedup**. ANN algorithms build data structures (indices) that partition the vector space or compress the vectors so that only a small subset must be examined. The speedup factor can be 100–1000× with recall@10 still above 95%.

The mathematical justification for approximation comes from the **curse of dimensionality**, studied formally as the *concentration of measure* phenomenon. In high-dimensional spaces, the contrast between near and far points vanishes: for random points in $\mathbb{R}^d$, the ratio of maximum to minimum pairwise distance converges toward 1 as $d \to \infty$:

$$\lim_{d \to \infty} \frac{\text{dist}_{\max}}{\text{dist}_{\min}} = 1$$

This means that in 768D space, almost all pairs of random vectors are roughly equidistant. The volume of a unit sphere relative to its circumscribed cube shrinks to zero exponentially in $d$:

$$\frac{V_{\text{sphere}}(d)}{V_{\text{cube}}(d)} = \frac{\pi^{d/2}}{2^d \cdot \Gamma(d/2 + 1)} \to 0$$

Consequence: exact distance computations become less informative because there is no "close" vs "far" — everything is far. This makes ANN surprisingly effective, because "close enough" is often indistinguishable from "exactly closest" in terms of downstream task performance (retrieval accuracy, recommendation relevance, etc.).

```python
import numpy as np
import time

np.random.seed(42)
N = 500_000
d = 768
query = np.random.randn(d).astype(np.float32)
query /= np.linalg.norm(query)

collection = np.random.randn(N, d).astype(np.float32)
collection /= np.linalg.norm(collection, axis=1, keepdims=True)

# Exact KNN: full scan using BLAS-optimized matrix-vector multiply.
start = time.time()
scores = collection @ query
top_k_exact = np.argsort(-scores)[:10]
exact_time = time.time() - start
print(f"Exact KNN time: {exact_time:.3f}s")

# Simulated ANN: random subsampling (crude demo).
# Real ANN uses smarter structures like HNSW, IVF, or PQ.
sample_size = 5_000
start = time.time()
sample_idx = np.random.choice(N, sample_size, replace=False)
sample_scores = collection[sample_idx] @ query
top_k_ann = sample_idx[np.argsort(-sample_scores)[:10]]
ann_time = time.time() - start

recall = len(set(top_k_exact) & set(top_k_ann)) / 10.0
print(f"ANN time: {ann_time:.3f}s | Recall@10: {recall:.1f}")

# Demonstration of distance concentration.
import math
dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for dim in dimensions:
    pts = np.random.randn(1000, dim).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    dists = np.linalg.norm(pts[:1] - pts[1:], axis=1)
    ratio = dists.max() / dists.min()
    print(f"d={dim:4d}  max/min ratio={ratio:.3f}")
```

⚠️ **"ANN is fast, but verify recall first."** — Always measure recall@k against an exact baseline on a held-out query set before deploying an ANN index to production. Recall is data-dependent: highly clustered data may exhibit boundary effects where true neighbors straddle partition boundaries.

💡 **The recall benchmark is your safety net.** Write it as a CI test that runs nightly. If recall drops below threshold after a reindex, your build pipeline should fail before the change reaches production.

**Caso real — Pinterest Visual Search:** Pinterest's visual search system lets users upload photos to find similar pins across billions of images. Their engineering team evaluated exact KNN and found it impossible at their scale — a single query would take minutes. They adopted ANN with HNSW indexing, achieving sub-100ms latency with 95%+ recall@10. Critically, they discovered that the choice of distance metric was as important as the index algorithm: cosine similarity captured style similarity (a red sneaker matches other red sneakers), while L2 better captured color composition (a red sneaker matches images with similar RGB histograms). They exposed both options to users via a "Style" vs "Color" toggle.

❌ **Antipattern: Deploying ANN without a recall regression harness.** A stakeholder demanding 100% recall on 1B vectors needs education: exact KNN at that scale requires $10^{12}$ operations per query and is physically impossible at interactive latency. Negotiate a recall target like 97% and document the trade-off.

✅ **Start with exact KNN for prototyping (<50k vectors), then migrate to ANN** once latency requirements become strict. This validates embedding quality before adding index complexity. Use the exact results as ground truth for ANN recall benchmarking.

---

## 3. Evaluating Embedding Quality

### Theory

Not all embeddings are created equal. The quality of your vector search system depends critically on the quality of the embeddings. Three properties matter most:

**Dimensionality vs. information density:** Higher dimensions capture more nuanced semantics but suffer more from the curse of dimensionality and require more storage. The optimal dimensionality depends on your task: 384D from `all-MiniLM-L6-v2` is sufficient for many retrieval tasks, while 1536D from `text-embedding-3-large` excels at fine-grained semantic distinctions.

**Intra-class vs. inter-class distance distribution:** A good embedding model produces tight clusters for semantically similar items and well-separated clusters for dissimilar items. You can measure this with the *silhouette score*: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance. Values near 1 indicate well-separated clusters.

**Robustness to distribution shift:** Embeddings from models fine-tuned on one domain often fail on another. A legal document embedder fine-tuned on case law will produce poor embeddings for medical abstracts. Always evaluate on your specific data distribution.

```python
import numpy as np
from numpy.linalg import norm

def silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Simplified silhouette score for embedding quality assessment."""
    unique_labels = np.unique(labels)
    scores = []
    for i, vec in enumerate(vectors):
        same_cluster = vectors[labels == labels[i]]
        other_clusters = [vectors[labels == l] for l in unique_labels if l != labels[i]]
        a = np.mean([norm(vec - v) for v in same_cluster if not np.allclose(v, vec)])
        b = min(np.mean([norm(vec - v) for v in cluster]) for cluster in other_clusters)
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))

np.random.seed(42)
n_per_class = 50
class_a = np.random.randn(n_per_class, 64) + np.array([2.0] * 64)
class_b = np.random.randn(n_per_class, 64) + np.array([-2.0] * 64)
vectors = np.vstack([class_a, class_b]).astype(np.float32)
vectors /= norm(vectors, axis=1, keepdims=True)
labels = np.array([0]*n_per_class + [1]*n_per_class)
print(f"Silhouette score: {silhouette_score(vectors, labels):.3f}")
# Score near 1 = well-separated clusters → good embedding quality.
```

⚠️ **Silhouette scores >0.5 indicate reasonable separation**, but this is not sufficient for production. Always measure downstream task performance (retrieval NDCG, classification F1) after evaluating embedding quality.

💡 **The embedding model is the most important component of your vector search system.** A better embedding model with a simple Flat index often outperforms a worse model with the most sophisticated HNSW index. Invest in embedding quality before index tuning.

**Caso real — Shopify product search:** Shopify evaluated multiple embedding models for their semantic product search feature. The 384D `all-MiniLM-L6-v2` achieved 89% recall@10, while 768D `e5-large` achieved 94% — but at 4× the storage cost and 3× the query latency. They chose a middle ground: `e5-base` (768D) with PQ compression to reduce memory footprint. The lesson: embedding quality, dimensionality, and system cost form a trilemma that must be balanced for each use case.

❌ **Antipattern: Using raw embedding model outputs without normalization or centering.** Many embedding spaces have non-zero mean vectors. Subtracting the mean (centering) and normalizing to unit length often improves retrieval quality, especially for cosine-based search.

✅ **Run a quick embedding quality check before building your index.** Compute silhouette score on a labeled subset. If <0.3, consider a different embedding model or fine-tuning on your domain.

---

## 4. Distance Metrics Selection Guide

### Theory

Choosing the right distance metric is a design decision that impacts recall, latency, and even the embedding model you select. Here is a mathematical and practical framework:

**Cosine similarity** is the default for any text embedding. Sentence transformers, OpenAI embeddings, Cohere, and Voyage all optimize for cosine space. The range is $[-1, 1]$ for raw similarity and $[0, 2]$ for cosine distance ($1 - \cos$). When vectors are normalized to unit length, dot product produces identical rankings to cosine.

**Euclidean distance** is appropriate when the magnitude of the embedding carries meaning. For example, image embeddings from CNNs often encode intensity or confidence in the vector magnitude. L2 is also the natural metric for clustering algorithms like k-means, which minimizes within-cluster L2 variance.

**Dot product** is preferred when you want raw inner product, such as in matrix factorization recommendation systems where the score represents user-item affinity. OpenAI's `text-embedding-3-small` and `text-embedding-3-large` models output normalized vectors by default, making dot product equivalent to cosine.

**Manhattan distance** is useful in high-noise domains where the squaring in L2 amplifies outlier dimensions. In bioinformatics (gene expression profiles) and network anomaly detection, L1 often outperforms L2.

```python
import numpy as np
from numpy.linalg import norm

def metric_comparison(a: np.ndarray, b: np.ndarray):
    """Compare all four metrics on the same pair of vectors."""
    results = {
        "L2 (Euclidean)": float(norm(a - b)),
        "L1 (Manhattan)": float(np.sum(np.abs(a - b))),
        "Cosine distance": float(1 - np.dot(a, b) / (norm(a) * norm(b))),
        "Dot product": float(np.dot(a, b)),
    }
    for name, val in results.items():
        print(f"  {name:20s} = {val:.4f}")
    return results

v1 = np.array([0.9, 0.1, 0.0])
v2 = np.array([0.1, 0.9, 0.0])
print("v1 vs v2 (different orientation):")
metric_comparison(v1, v2)

v3 = np.array([1.8, 0.2, 0.0])  # v1 scaled by 2x
print("\nv1 vs v3 (same orientation, different magnitude):")
metric_comparison(v1, v3)
```

⚠️ **Watch out for metric-operator mismatch in databases.** For example, PostgreSQL's pgvector uses `<->` for L2, `<=>` for cosine, and `<#>` for negative inner product. Using the wrong operator silently degrades results.

💡 **When in doubt, default to cosine.** It works well across text, image, and multimodal embeddings. Only switch to L2 when you have a concrete reason (magnitude matters) and benchmark to validate.

**Caso real — Netflix Recommendations:** Netflix uses dot product similarity on user and content embeddings trained via collaborative filtering. Their model outputs are not normalized because the magnitude encodes user engagement level (power users have larger embedding norms). Switching to cosine would lose this signal. Their serving stack uses dot product as the primary scoring metric, then blends with contextual features (time of day, device type) in a secondary ranking pass.

❌ **Antipattern: Using the same metric for embedding training and search.** Your embedding model was trained with a specific loss function (contrastive loss, triplet loss, etc.) that implies a distance structure. Always check the model card for the recommended similarity metric.

✅ **Benchmark 2–3 metrics before committing.** Run exact KNN on a validation set with candidate metrics and measure downstream task accuracy (retrieval precision, NDCG, etc.), not just recall@k of the metric against itself.

---

### Summary of Metric Selection

| Scenario | Recommended Metric | Why |
|---|---|---|
| Text search (BERT, Sentence-T5) | Cosine or Dot | Models optimized for cosine space; magnitude irrelevant |
| Image similarity (CLIP, ViT) | Cosine | CLIP trained with contrastive loss on normalized vectors |
| Image composition (color histograms) | L2 | Magnitude carries intensity information |
| Recommendation (matrix factorization) | Dot | User-item affinity is raw inner product |
| Anomaly detection (sensor logs) | L1 or L2 | Robust to outliers (L1) or sensitive to extreme values (L2) |
| Genomic sequences | L1 | High noise; squaring in L2 amplifies artifacts |

This table should be your starting point, not your final answer. Always validate with data.

---

## 🎯 Key Takeaways

- Embeddings map discrete objects into continuous vector spaces where geometric proximity equals semantic similarity; this is the foundation of all modern semantic search and RAG systems
- **Cosine similarity** is the default for normalized text embeddings; **Euclidean (L2)** is common for image, audio, and sensor embeddings where magnitude matters
- **Dot product** is the fastest metric and equivalent to cosine on normalized vectors; **Manhattan (L1)** is robust to outliers in high-noise domains
- Exact KNN is $O(Nd)$ and becomes infeasible beyond ~100k vectors in high dimensions; ANN is the production standard at larger scales
- The **curse of dimensionality** causes distances to concentrate in high dimensions, making exact distinctions less meaningful and approximate methods more justifiable
- Always benchmark ANN recall on your specific data distribution before committing to an index type or distance metric
- Normalize text embeddings to unit length before computing cosine or dot product — document length should not affect similarity scores
- Monitor embedding model drift; upgrading from `text-embedding-3-small` to `text-embedding-3-large` changes the vector space and may require reindexing
- Document the embedding model version and distance metric alongside your vector schema to prevent silent mismatches during reindexing or migration
- Evaluate embedding quality with silhouette scores before building production indices; a better embedding model with a simple index beats a poor model with optimal tuning

## References

- J. Johnson, M. Douze, H. Jégou. "Billion-scale similarity search with GPUs." arXiv:1702.08734, 2017
- T. Mikolov et al. "Efficient Estimation of Word Representations in Vector Space." ICLR, 2013
- A. Andoni, P. Indyk. "Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions." Communications of the ACM, 2008
- OpenAI Embeddings Documentation: https://platform.openai.com/docs/guides/embeddings
- Faiss Wiki (ANN fundamentals): https://github.com/facebookresearch/faiss/wiki
- [[22 - NLP and Transformers]] — Embedding model internals
- [[03 - Mathematics for ML]] — Norms and high-dimensional geometry

## 📦 Código de compresión

```python
"""
Vector Search Fundamentals — Compression Script
Summarizes: embeddings, distance metrics, ANN vs KNN, dimensionality curse.
"""
import numpy as np
from numpy.linalg import norm

class VectorSearchBasics:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.random.randn(self.dim).astype(np.float32)
        return vec / norm(vec)

    def distance(self, a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
        if metric == "euclidean":
            return float(norm(a - b))
        if metric == "cosine":
            return float(1 - np.dot(a, b) / (norm(a) * norm(b)))
        if metric == "dot":
            return float(-np.dot(a, b))
        if metric == "manhattan":
            return float(np.sum(np.abs(a - b)))
        raise ValueError(f"Unknown metric: {metric}")

    def knn(self, query: np.ndarray, collection: np.ndarray, k: int = 10):
        scores = collection @ query
        top_k = np.argsort(-scores)[:k]
        return top_k, scores[top_k]

    def recall_at_k(self, exact_top_k, ann_top_k):
        return len(set(exact_top_k) & set(ann_top_k)) / len(exact_top_k)

    def distance_concentration(self, dims: list[int], n_pts: int = 1000):
        for d in dims:
            pts = np.random.randn(n_pts, d).astype(np.float32)
            pts /= norm(pts, axis=1, keepdims=True)
            dists = norm(pts[:1] - pts[1:], axis=1)
            print(f"d={d:4d}  max/min ratio={dists.max()/dists.min():.3f}")

if __name__ == "__main__":
    engine = VectorSearchBasics(dim=128)
    q = engine.embed("vector databases")
    coll = np.array([engine.embed(f"doc {i}") for i in range(1000)])
    top_k, scores = engine.knn(q, coll, k=5)
    print("Top-5 indices:", top_k, "Scores:", scores)
    print("\nDistance concentration:")
    engine.distance_concentration([2, 4, 8, 16, 32, 64, 128, 256, 512])
```
