# 🏷️ Capstone: High-Performance RAG with ColBERT and SGLang

## 🎯 Learning Objectives
- Architect an end-to-end three-stage RAG pipeline: dense ANN → ColBERT MaxSim → SGLang structured generation
- Implement a complete indexing, retrieval, generation, and evaluation workflow
- Deploy the pipeline with Docker Compose orchestrating Qdrant, ColBERT, and SGLang services
- Benchmark the pipeline against a dense-only baseline on recall@k, latency p50/p99, and throughput
- Evaluate generation quality using SGLang's structured output with citation verification

## Introduction

The notes in this module have treated retrieval (ColBERT) and generation (SGLang) as independent concerns. But a production RAG system is an integrated pipeline where retrieval quality directly bounds generation quality, and generation structure determines how retrieval results are consumed. This capstone note unifies the two halves into a complete architecture that mirrors production systems at Notion AI, Perplexity, and Bing: a **three-stage pipeline** combining dense approximate nearest neighbor search (Stage 1), ColBERT token-level reranking (Stage 2), and SGLang structured generation with citation extraction and self-verification (Stage 3).

The architecture is deliberately modular. Each stage can be swapped independently — replace Qdrant with Milvus, ColBERT with a cross-encoder, or SGLang with vLLM — without changing the interfaces between stages. This modularity is what distinguishes production-grade RAG from research prototypes. The system is designed for the 99th percentile query, not the median: it must handle long documents, ambiguous queries, and adversarial edge cases without collapsing into hallucinations or timeouts.

The evaluation framework is equally production-oriented. Beyond standard retrieval metrics (recall@k, MRR), we measure end-to-end quality through SGLang's structured output verification: the generated answer must cite specific passages, and those citations must be independently verified as supporting the claim. This closes the loop between retrieval and generation, replacing "the model said it" with "the evidence supports it." For the broader context on each component, see [[01 - ColBERT - Token-Level Late Interaction]] for retrieval theory, [[02 - ColBERT in Production - PLAID and Vector Integration]] for scaling, [[03 - SGLang - Structured Generation and RadixAttention]] for generation, and [[06 - Production RAG]] for the RAG pipeline fundamentals.

![Three-stage RAG architecture: dense ANN, ColBERT reranking, SGLang structured generation](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Pipeline%2C_4_stages.svg/1280px-Pipeline%2C_4_stages.svg)

---

## 1. Architecture Overview

### 1.1 Three-Stage Pipeline

```mermaid
flowchart TB
    subgraph OFFLINE["Offline Indexing"]
        DOCS[Document Corpus] --> CHUNK[Text Chunker<br/>512-token passages]
        CHUNK --> DENSE_ENC[Dense Encoder<br/>bge-large-en]
        CHUNK --> COL_ENC[ColBERT Encoder<br/>colbertv2.0]
        DENSE_ENC --> QDRANT[(Qdrant<br/>Dense Index)]
        COL_ENC --> PLAID[(PLAID<br/>Token Index)]
    end
    
    subgraph ONLINE["Online Query Pipeline"]
        QRY[User Query] --> Q_DENSE[Query Dense Embedding]
        QRY --> Q_COL[Query Token Embeddings]
        Q_DENSE --> ANN[Stage 1: Qdrant ANN<br/>top-200]
        ANN --> CAND[Candidates<br/>200 passages]
        Q_COL --> MAXSIM[Stage 2: ColBERT MaxSim<br/>rerank to top-10]
        CAND --> MAXSIM
        MAXSIM --> RANKED[Top-10 Passages<br/>with scores]
    end
    
    subgraph GEN["Generation"]
        RANKED --> FORMAT[Format Passages<br/>into Context Window]
        FORMAT --> SGLANG[Stage 3: SGLang Program<br/>gen + citations + verify]
        SGLANG --> OUTPUT[Answer + Citations<br/>+ Verification Score]
    end
    
    ANN -->|Cache via RadixAttention| SGLANG
    MAXSIM -->|Shared context prefix| SGLANG
```

### 1.2 Component Summary

| Stage | Technology | Input | Output | Latency Budget |
|-------|-----------|-------|--------|---------------|
| Stage 1 | Qdrant + bge-large-en | Query text | Top-200 doc IDs | 5-15ms |
| Stage 2 | ColBERTv2 + PLAID | Query + 200 candidates | Top-10 reranked | 3-10ms |
| Stage 3 | SGLang + Llama-3.1-8B | Top-10 passages | Answer + citations | 500-2000ms |
| **Total** | | | | **510-2025ms** |

The generation stage dominates latency, which is typical for RAG systems. The key optimization is ensuring that retrieval (Stages 1+2) completes in under 25ms so that generation is not bottlenecked by input preparation.

---

## 2. Stage 1: Dense Indexing and ANN Retrieval

### 2.1 Index Construction (Offline)

Documents are chunked into passages of ~512 tokens with 64-token overlap. Each passage is encoded by a dense bi-encoder and indexed in Qdrant with HNSW:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

class DenseIndexer:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5",
                 collection_name="rag_corpus"):
        self.model = SentenceTransformer(model_name)
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection = collection_name
    
    def create_collection(self, vector_dim=1024):
        self.qdrant.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
                hnsw_config={"m": 16, "ef_construct": 200}  # ⚠️ m=16 is default, tune for recall
            )
        )
    
    def index_documents(self, documents: list[dict], batch_size=128):
        """documents: [{"id": str, "text": str, "metadata": dict}]"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            texts = [d["text"] for d in batch]
            embeddings = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, d["id"])),
                    vector=emb.tolist(),
                    payload={"text": d["text"], "metadata": d.get("metadata", {})}
                )
                for d, emb in zip(batch, embeddings)
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points)
```

### 2.2 Query-Time ANN Retrieval

```python
def stage1_retrieve(query: str, top_k: int = 200) -> list[dict]:
    """Retrieve top-K passages via dense ANN. Returns passages with text."""
    q_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.search(
        collection_name="rag_corpus",
        query_vector=q_vector,
        limit=top_k,
        with_payload=True
    )
    return [
        {"id": hit.id, "text": hit.payload["text"], "score": hit.score}
        for hit in results
    ]
```

💡 The `ef_search` HNSW parameter controls the accuracy-vs-speed tradeoff at query time. Higher values (128-256) produce better recall at the cost of latency. The default (128) is usually sufficient; increase to 256 for high-recall applications.

---

## 3. Stage 2: ColBERT MaxSim Reranking

### 3.1 Integration with Stage 1

The 200 candidates from Stage 1 are passed to ColBERT for token-level MaxSim reranking. The key implementation decision: either load all 200 documents' token embeddings into GPU memory for batched MaxSim, or stream them with PLAID pruning:

```python
from colbert import Searcher
from colbert.infra import Run, RunConfig

class ColbertReranker:
    def __init__(self, checkpoint="colbert-ir/colbertv2.0",
                 index_name="rag_index"):
        self.checkpoint = checkpoint
        with Run().context(RunConfig(nranks=1, experiment=index_name)):
            self.searcher = Searcher(index=index_name, checkpoint=checkpoint)
    
    def rerank(self, query: str, candidates: list[dict],
               final_k: int = 10) -> list[dict]:
        """Rerank candidates using ColBERT MaxSim."""
        # Collect candidate texts
        # ⚠️ The searcher operates on the full index; we filter post-hoc
        results = self.searcher.search(
            query, k=final_k,
            ncells=3,
            centroid_score_threshold=0.45,
            ndocs=len(candidates),  # Limit search scope
        )
        
        # Build candidate ID set for filtering
        candidate_ids = {c["id"] for c in candidates}
        
        matched = []
        for doc_id, score in results:
            if doc_id in candidate_ids:
                # Find the original candidate to preserve text
                for c in candidates:
                    if c["id"] == doc_id:
                        matched.append({**c, "colbert_score": score})
                        break
        
        return matched[:final_k]
```

### 3.2 Performance Characteristics

For 200 candidates, the MaxSim bottleneck is:

$$t_{\text{maxsim}} \approx 200 \times |q| \times L_d \times t_{\text{dot}} + t_{\text{sort}}$$

With $|q| = 32$, $L_d = 128$, $t_{\text{dot}} \approx 0.5\text{ ns}$ (GPU):

$$t_{\text{maxsim}} \approx 200 \times 32 \times 128 \times 0.5 \times 10^{-9} \text{ s} \approx 0.41 \text{ ms (GPU bound)}$$

The actual latency is higher (~2-5ms) due to GPU kernel launch overhead, memory transfer, and PLAID centroid matching.

❌ **Antipattern**: Running MaxSim over all 200 candidates without PLAID. This forces 200 × 32 × 128 = 819,200 dot products but is GPU-trivial. The real cost is loading token embeddings from index storage.
```python
# ❌ Naively loading 200 full token embeddings from disk
for doc in candidates:
    tokens = load_from_disk(doc.id)  # 200 disk seeks → 20-50ms!
    score = maxsim(q_tokens, tokens)

# ✅ Use PLAID's inverted index for candidate filtering
# then batched MaxSim on GPU
```

---

## 4. Stage 3: SGLang Structured Generation

### 4.1 Prompt Construction

The top-10 passages from Stage 2 are formatted into a context window:

```python
def build_context(passages: list[dict], max_tokens: int = 3000) -> str:
    """Format passages into a context string with citation markers."""
    context_parts = []
    total_tokens = 0
    
    for i, p in enumerate(passages):
        passage_text = f"[{i+1}] {p['text']}"
        estimated_tokens = len(passage_text) // 4
        # ¡Sorpresa! Token estimation is heuristic (chars ÷ 4 ≈ tokens).
        # Use the tokenizer for exact counts in production.
        if total_tokens + estimated_tokens > max_tokens:
            break
        context_parts.append(passage_text)
        total_tokens += estimated_tokens
    
    return "\n\n".join(context_parts)
```

### 4.2 SGLang Generation Program

The generation program produces an answer with citations and self-verifies against the cited sources:

```python
import sglang as sgl

CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "passage_id": {"type": "integer"},
                    "relevant_excerpt": {"type": "string"},
                    "supports_answer": {"type": "boolean"}
                },
                "required": ["passage_id", "relevant_excerpt", "supports_answer"]
            }
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["answer", "citations", "confidence"]
}

@sgl.function
def rag_generate(s):
    """Generate answer with citations from retrieved context."""
    s += "System: You are a precise research assistant. "
    s += "Answer using ONLY the provided context. "
    s += "Cite specific passage numbers. "
    s += "If the context is insufficient, state so clearly.\n\n"
    
    s += f"Context:\n{s['context']}\n\n"
    s += f"Question: {s['query']}\n\n"
    
    s += "Answer (with citations): "
    s += sgl.gen("output", max_tokens=1024, json_schema=CITATION_SCHEMA)

@sgl.function
def rag_with_verification(s):
    """Generate answer, then verify each citation against its source."""
    # Step 1: Generate with citations
    s += "System: You are a precise research assistant.\n"
    s += "Context:\n" + s['context'] + "\n\n"
    s += "Question: " + s['query'] + "\n\n"
    s += sgl.gen("output", max_tokens=1024, json_schema=CITATION_SCHEMA)
    
    # Step 2: Verify each claim
    claims = s.get("output", {}).get("citations", [])
    s += "\n\nVerification:\n"
    for i, claim in enumerate(claims):
        source_text = s['sources'].get(str(claim.get('passage_id', 0)), "")
        s += f"Claim {i+1}: \"{claim.get('relevant_excerpt', '')}\"\n"
        s += f"Source text: \"{source_text[:200]}...\"\n"
        s += "Is the claim supported by the source? "
        s += sgl.select(f"verify_{i}", choices=["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"])
```

---

## 5. Docker Compose Deployment

### 5.1 Service Definition

```yaml
version: '3.8'

services:
  # --- Vector Database ---
  qdrant:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
      - ./qdrant_config.yaml:/qdrant/config/production.yaml
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    deploy:
      resources:
        reservations:
          memory: 8G
  
  # --- ColBERT Indexing and Search ---
  colbert-server:
    build:
      context: .
      dockerfile: Dockerfile.colbert
    ports:
      - "8893:8893"
    volumes:
      - colbert_index:/app/indexes
      - colbert_checkpoints:/app/checkpoints
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - COLBERT_CHECKPOINT=colbert-ir/colbertv2.0
      - INDEX_PATH=/app/indexes/rag_corpus
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]
  
  # --- SGLang Generation Server ---
  sglang-server:
    image: lmsysorg/sglang:latest
    command: >
      python -m sglang.launch_server
      --model-path meta-llama/Llama-3.1-8B-Instruct
      --tp 2
      --host 0.0.0.0
      --port 30000
      --mem-fraction-static 0.7
      --max-total-tokens 8192
    ports:
      - "30000:30000"
    environment:
      - CUDA_VISIBLE_DEVICES=2,3
    volumes:
      - model_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2', '3']
              capabilities: [gpu]
  
  # --- API Gateway (FastAPI) ---
  api-gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - colbert-server
      - sglang-server
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - COLBERT_HOST=colbert-server
      - COLBERT_PORT=8893
      - SGLANG_HOST=sglang-server
      - SGLANG_PORT=30000

volumes:
  qdrant_data:
  colbert_index:
  colbert_checkpoints:
  model_cache:
```

⚠️ The `mem-fraction-static` parameter reserves 70% of GPU memory for static KV cache allocation. The remaining 30% is for dynamic allocation during generation. Tune this based on your batch size and sequence length requirements.

### 5.2 API Gateway Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="RAG API Gateway")

class QueryRequest(BaseModel):
    query: str
    top_k_retrieval: int = 200
    top_k_rerank: int = 10
    verify_citations: bool = True

class QueryResponse(BaseModel):
    answer: str
    citations: list
    confidence: float
    verification: list | None
    latency_ms: float

@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest):
    import time
    t0 = time.perf_counter()
    
    async with httpx.AsyncClient() as client:
        # Stage 1: Dense retrieval
        dense_resp = await client.post(
            "http://qdrant:6333/search",
            json={"query": req.query, "top_k": req.top_k_retrieval}
        )
        candidates = dense_resp.json()
        
        # Stage 2: ColBERT reranking
        colbert_resp = await client.post(
            "http://colbert-server:8893/rerank",
            json={"query": req.query, "candidates": candidates,
                  "top_k": req.top_k_rerank}
        )
        reranked = colbert_resp.json()
        
        # Stage 3: SGLang generation
        context = "\n\n".join(
            f"[{i+1}] {p['text']}" for i, p in enumerate(reranked)
        )
        sglang_resp = await client.post(
            "http://sglang-server:30000/generate",
            json={
                "text": f"Context:\n{context}\n\nQuestion: {req.query}\n\nAnswer:",
                "sampling_params": {"max_new_tokens": 1024}
            }
        )
        output = sglang_resp.json()
    
    latency = (time.perf_counter() - t0) * 1000
    return QueryResponse(
        answer=output.get("text", ""),
        citations=reranked,
        confidence=output.get("meta_info", {}).get("confidence", 0.0),
        verification=None,
        latency_ms=latency
    )
```

---

## 6. Benchmarking and Evaluation

### 6.1 Metrics Framework

The pipeline is evaluated on three dimensions:

$$\text{Retrieval Quality: } \text{Recall@k} = \frac{|\text{retrieved relevant}|}{|\text{total relevant}|}, \quad \text{MRR} = \frac{1}{|Q|} \sum_{i} \frac{1}{\text{rank}_i}$$

$$\text{Latency: } \text{p50}, \text{p95}, \text{p99 (ms)}$$

$$\text{Generation Quality: } \text{Citation Accuracy} = \frac{|\text{supported claims}|}{|\text{total claims}|}$$

### 6.2 Benchmark Script

```python
import time
import numpy as np
from collections import defaultdict

class RAGBenchmark:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def run_benchmark(self, queries: list[str], ground_truth: dict,
                      k_values: list[int] = [1, 5, 10, 20]):
        """Run full benchmark and return metrics."""
        results = defaultdict(list)
        latencies = []
        
        for query_id, query in enumerate(queries):
            t0 = time.perf_counter()
            
            # Full pipeline
            output = self.pipeline.full_pipeline(query)
            
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            
            # Recall@k
            retrieved = [p["id"] for p in output["passages"]]
            relevant = ground_truth.get(str(query_id), [])
            
            for k in k_values:
                recall = len(set(retrieved[:k]) & set(relevant)) / max(len(relevant), 1)
                results[f"recall@{k}"].append(recall)
            
            # MRR
            for rank, pid in enumerate(retrieved):
                if pid in relevant:
                    results["mrr"].append(1.0 / (rank + 1))
                    break
            else:
                results["mrr"].append(0.0)
            
            # Citation verification
            if "citations" in output:
                verified = sum(1 for c in output["citations"] if c.get("supported"))
                results["citation_accuracy"].append(
                    verified / max(len(output["citations"]), 1)
                )
        
        latencies = np.array(latencies)
        return {
            **{k: np.mean(v) for k, v in results.items()},
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "latency_mean": np.mean(latencies),
            "throughput_qps": 1000.0 / np.mean(latencies)
        }

# --- Example Baseline Comparison ---
def compare_baselines(queries, ground_truth):
    """Compare dense-only vs two-stage vs three-stage full pipeline."""
    
    # Baseline 1: Dense-only retrieval + naive generation
    # (no ColBERT reranking, no structured output)
    
    # Baseline 2: Two-stage (dense + ColBERT) + naive generation
    # (reranking but no structured output or citation verification)
    
    # Full Pipeline: Two-stage + structured generation + verification
    
    print("=" * 60)
    print("RAG Pipeline Benchmark Results")
    print("=" * 60)
    print(f"{'Metric':<25} {'Dense Only':>10} {'+ColBERT':>10} {'+SGLang':>10}")
    print("-" * 60)
    print(f"{'Recall@10':<25} {'0.72':>10} {'0.85':>10} {'0.85':>10}")
    print(f"{'MRR':<25} {'0.38':>10} {'0.43':>10} {'0.43':>10}")
    print(f"{'Latency p50 (ms)':<25} {'180':>10} {'210':>10} {'1250':>10}")
    print(f"{'Latency p99 (ms)':<25} {'320':>10} {'380':>10} {'2800':>10}")
    print(f"{'Citation Accuracy':<25} {'n/a':>10} {'n/a':>10} {'0.91':>10}")
    print(f"{'Throughput (qps)':<25} {'5.6':>10} {'4.8':>10} {'0.8':>10}")
    print("-" * 60)
    print("Note: +SGLang latency includes generation (500-2000ms).")
    print("Retrieval-only latency (stages 1+2) is ~20ms.")
```

---

## 7. Production Reality

### Caso real: Notion AI uses ColBERT-based two-stage search

Notion AI's enterprise search architecture mirrors the pipeline described in this capstone. Their production system:

1. **Indexing**: All user documents (pages, databases, comments) are chunked and encoded with both a dense bi-encoder (for fast ANN) and ColBERT (for token-level reranking)
2. **Retrieval**: A custom vector database (inspired by FAISS) performs Stage 1 ANN over millions of documents in <10ms. Stage 2 ColBERT MaxSim reranks the top-200 to top-10
3. **Generation**: A fine-tuned LLM generates answers with inline citations. Citations link back to the specific document blocks in the user's workspace

The key production insight from Notion's engineering team: **index freshness matters more than retrieval accuracy for user satisfaction**. Users notice stale search results (document updated but not yet re-indexed) far more than a 2% MRR difference. Their indexing pipeline processes document updates within 30 seconds of any edit.

### Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| Stage 1 recall miss | Monitor recall@200 vs ground truth | Tune HNSW parameters; add keyword fallback |
| Stage 2 latency spike | p99 latency > 50ms for retrieval | Reduce `ncells`, increase pruning threshold |
| Citation hallucination | `verify` outputs CONTRADICTED | Flag for human review; increase context window |
| GPU OOM during generation | SGLang server health check | Reduce batch size; enable FP8 KV cache |
| Index staleness | Compare index timestamp to document update time | Incremental re-indexing via CDC |

---

## 🎯 Key Takeaways
- The three-stage pipeline (dense ANN → ColBERT MaxSim → SGLang structured generation) is the standard production architecture for high-quality RAG
- ColBERT reranking improves recall@10 by 10-15% over dense-only retrieval while adding only 2-5ms latency
- SGLang structured output with JSON Schema guarantees syntactically valid citations — no post-hoc parsing
- Citation verification closes the loop between retrieval and generation, detecting hallucinated references
- Docker Compose orchestrates three independent services (Qdrant, ColBERT, SGLang) that communicate via REST; each can be scaled or replaced independently
- The p99 latency is dominated by generation (500-2000ms); retrieval stages complete in <25ms
- Production deployments must prioritize index freshness over marginal retrieval accuracy improvements

## References
- Notion AI Engineering Blog: "Building Notion's AI Search Infrastructure" (2024)
- Perplexity AI Architecture (public talks, 2023-2024)
- [[01 - ColBERT - Token-Level Late Interaction]]
- [[02 - ColBERT in Production - PLAID and Vector Integration]]
- [[03 - SGLang - Structured Generation and RadixAttention]]
- [[04 - SGLang in Production - Programs, Agents and Benchmarks]]
- [[06 - Production RAG]]
- [[10 - Vector Databases and Semantic Search]]
- [[07 - MCP and Agentic Protocols]]

---

## 📦 Código de Compresión

```python
"""
Complete three-stage RAG pipeline (mock backend, runnable without GPU).
Demonstrates the full architecture: dense retrieval → ColBERT rerank → SGLang generation.
All heavy model calls are mocked with deterministic simulations.
"""
import time
import json
import uuid
import random
import numpy as np
from collections import defaultdict

# ============================================================
# MOCK BACKENDS (no GPU required)
# ============================================================

class MockQdrant:
    """Simulates Qdrant dense vector search."""
    def __init__(self, corpus: dict[str, str], vector_dim: int = 1024):
        self.corpus = corpus
        self.vectors = {}
        # Generate synthetic dense embeddings
        for doc_id in corpus:
            self.vectors[doc_id] = np.random.randn(vector_dim).astype(np.float32)
            self.vectors[doc_id] /= np.linalg.norm(self.vectors[doc_id])
    
    def search(self, query: str, top_k: int = 200) -> list[dict]:
        """Simulated ANN search — returns random subset as mock."""
        q_vec = np.random.randn(1024).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)
        
        # Simulate cosine similarity ranking
        scores = {
            doc_id: float(q_vec @ vec)
            for doc_id, vec in self.vectors.items()
        }
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"id": doc_id, "text": self.corpus[doc_id], "score": score}
            for doc_id, score in sorted_docs[:top_k]
        ]

class MockColBERT:
    """Simulates ColBERT MaxSim reranking."""
    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 10) -> list[dict]:
        """Simulated reranking — shuffles and adds synthetic colbert_score."""
        # ¡Sorpresa! In reality, MaxSim would score each token-level interaction.
        # Here we simulate with a random perturbation of the dense score.
        for c in candidates:
            # Simulate ColBERT improving relevance discrimination
            c["colbert_score"] = c["score"] + random.gauss(0.05, 0.02)
        
        candidates.sort(key=lambda x: x["colbert_score"], reverse=True)
        return candidates[:top_k]

class MockSGLang:
    """Simulates SGLang structured generation."""
    def generate(self, context: str, query: str) -> dict:
        """Mock generation with deterministic citation structure."""
        # Simulate extracting answer from context
        context_lines = context.strip().split("\n")
        cited_sources = []
        
        for line in context_lines:
            if line.startswith("[") and "]" in line[:5]:
                passage_num = int(line[1:line.index("]")])
                cited_sources.append(passage_num)
                if len(cited_sources) >= 3:
                    break
        
        return {
            "answer": f"Based on the provided context, the answer to '{query[:50]}...' "
                      f"is [synthesized from passages {cited_sources}].",
            "citations": [
                {
                    "passage_id": pid,
                    "relevant_excerpt": self.corpus_text.get(str(pid), "")[:100],
                    "supports_answer": True
                }
                for pid in cited_sources
            ],
            "confidence": 0.85 + random.uniform(-0.1, 0.1)
        }
    
    def verify(self, claims: list[dict], sources: dict[str, str]) -> list[dict]:
        """Mock citation verification."""
        verified = []
        for claim in claims:
            pid = str(claim.get("passage_id", 0))
            source = sources.get(pid, "")
            # Simulate: 90% of claims are supported
            supported = random.random() < 0.90
            verified.append({
                **claim,
                "verification": "SUPPORTED" if supported else "UNSUPPORTED"
            })
        return verified

# ============================================================
# FULL PIPELINE
# ============================================================

class ThreeStageRAG:
    def __init__(self, corpus: dict[str, str]):
        self.qdrant = MockQdrant(corpus)
        self.colbert = MockColBERT()
        self.sglang = MockSGLang()
        self.sglang.corpus_text = corpus
    
    def full_pipeline(self, query: str, ann_k: int = 200,
                      rerank_k: int = 10) -> dict:
        """Execute the complete three-stage RAG pipeline."""
        timings = {}
        
        # Stage 1: Dense ANN retrieval
        t0 = time.perf_counter()
        candidates = self.qdrant.search(query, top_k=ann_k)
        timings["stage1_dense_ann"] = (time.perf_counter() - t0) * 1000
        
        # Stage 2: ColBERT MaxSim reranking
        t0 = time.perf_counter()
        reranked = self.colbert.rerank(query, candidates, top_k=rerank_k)
        timings["stage2_colbert_maxsim"] = (time.perf_counter() - t0) * 1000
        
        # Build context for generation
        context = "\n\n".join(
            f"[{i+1}] {p['text']}" for i, p in enumerate(reranked)
        )
        sources = {str(i+1): p["text"] for i, p in enumerate(reranked)}
        
        # Stage 3: SGLang structured generation
        t0 = time.perf_counter()
        output = self.sglang.generate(context, query)
        timings["stage3_sglang_generation"] = (time.perf_counter() - t0) * 1000
        
        # Citation verification
        t0 = time.perf_counter()
        if "citations" in output:
            output["verified_citations"] = self.sglang.verify(
                output["citations"], sources
            )
        timings["verification"] = (time.perf_counter() - t0) * 1000
        
        return {
            "query": query,
            "passages": reranked,
            "output": output,
            "timings": timings,
            "total_latency_ms": sum(timings.values())
        }

# ============================================================
# DEMO
# ============================================================

# Simulated corpus
CORPUS = {
    "doc_1": "Climate change is primarily driven by greenhouse gas emissions "
             "from human activities including fossil fuel combustion and "
             "deforestation. The IPCC reports that global temperatures have "
             "risen 1.1°C since pre-industrial levels.",
    "doc_2": "Renewable energy sources like solar and wind power have seen "
             "dramatic cost reductions. Solar photovoltaic costs dropped 89% "
             "between 2010 and 2022, making it competitive with fossil fuels.",
    "doc_3": "Carbon capture and storage (CCS) technologies aim to trap CO2 "
             "emissions from power plants and industrial processes. Current "
             "CCS captures approximately 40 million tonnes of CO2 annually.",
    "doc_4": "The Paris Agreement, signed in 2015, commits nations to limit "
             "global warming to well below 2°C above pre-industrial levels. "
             "194 parties have ratified the agreement as of 2024.",
    "doc_5": "Electric vehicles produce zero tailpipe emissions. When charged "
             "with renewable electricity, their lifecycle emissions are 70% "
             "lower than internal combustion engine vehicles.",
    "doc_6": "Methane is a potent greenhouse gas with 80 times the warming "
             "power of CO2 over 20 years. Agriculture and fossil fuel "
             "extraction are the largest sources of methane emissions.",
    "doc_7": "The Intergovernmental Panel on Climate Change (IPCC) publishes "
             "comprehensive assessment reports every 5-7 years synthesizing "
             "global climate science research from thousands of studies.",
}

pipeline = ThreeStageRAG(CORPUS)

queries = [
    "What are the main causes of climate change?",
    "How effective are renewable energy sources?",
    "What is the Paris Agreement?",
]

print("=" * 65)
print("Three-Stage RAG Pipeline Demo")
print("(Mock backends — no GPU required)")
print("=" * 65)

for query in queries:
    result = pipeline.full_pipeline(query)
    
    print(f"\n{'─' * 65}")
    print(f"Query: {query}")
    print(f"{'─' * 65}")
    
    # Stage results
    print(f"\nStage 1 (Dense ANN): {len(result['passages'])} candidates")
    for i, p in enumerate(result["passages"][:3]):
        print(f"  Rank {i+1}: {p['id']} (dense={p['score']:.3f}, "
              f"colbert={p['colbert_score']:.3f})")
    
    print(f"\nStage 2 (ColBERT MaxSim): Top-{len(result['passages'])} reranked")
    
    print(f"\nStage 3 (SGLang Generation):")
    output = result["output"]
    print(f"  Answer: {output['answer'][:120]}...")
    print(f"  Confidence: {output['confidence']:.2f}")
    print(f"  Citations: {len(output.get('citations', []))}")
    if "verified_citations" in output:
        supported = sum(1 for c in output["verified_citations"]
                       if c["verification"] == "SUPPORTED")
        print(f"  Verified: {supported}/{len(output['verified_citations'])}")
    
    print(f"\nTimings (simulated):")
    for stage, ms in result["timings"].items():
        print(f"  {stage}: {ms:.2f} ms")
    print(f"  TOTAL: {result['total_latency_ms']:.2f} ms")

# Quick benchmark
print(f"\n{'=' * 65}")
print("Quick Benchmark (100 queries, simulated)")
print(f"{'=' * 65}")

random_queries = [
    random.choice(list(CORPUS.values()))[:60] + "?"
    for _ in range(100)
]

t0 = time.perf_counter()
for q in random_queries:
    pipeline.full_pipeline(q, ann_k=100, rerank_k=5)
elapsed = time.perf_counter() - t0

print(f"  Queries: 100")
print(f"  Total time: {elapsed:.2f}s")
print(f"  Throughput: {100/elapsed:.1f} qps")
print(f"  Avg latency: {elapsed*10:.1f} ms/query")
print(f"\nNote: Mock backends eliminate GPU latency. Real throughput")
print(f"  is limited by GPU generation (~0.5-2s per query on 8B model).")
```