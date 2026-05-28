# 🏷️ Capstone: High-Performance RAG with ColBERT and SGLang

## 🎯 Learning Objectives

- Architect a complete RAG pipeline that combines dense retrieval, ColBERT late-interaction reranking, and SGLang structured generation.
- Implement a two-stage retrieval system: dense ANN for candidate generation and ColBERT MaxSim for precise reranking.
- Deploy SGLang as the generation backend with structured programs for answer generation, citation validation, and LLM-as-a-Judge quality scoring.
- Evaluate end-to-end system performance against a dense-only baseline using recall, latency, throughput, and accuracy metrics.
- Containerize the stack with Docker Compose for production deployment.

## Introduction

**What "RAG" means.** **R**etrieval-**A**ugmented **G**eneration is a paradigm where a language model generates answers conditioned on retrieved documents, rather than relying solely on parametric knowledge. It grounds the model in external, updatable, and verifiable information. **ColBERT** stands for **Contextualized Late Interaction over BERT**—a retrieval architecture that tokenizes queries and documents into fine-grained embeddings and scores relevance with MaxSim, avoiding the compression losses of single-vector dense retrieval.

**What this capstone is.** In plain terms, you will build a search-and-answer engine that is both *precise* and *fast*. It finds the best documents using two complementary retrieval stages and then uses a programmable inference engine (SGLang) to generate answers that cite their sources and judge their own quality. This is not a toy demo: it is a blueprint for production AI search used by companies like Bing, Perplexity, and internal enterprise knowledge bases.

**Why this is cutting-edge technology today.** Standard RAG systems use a single dense embedding model to retrieve documents and a generic LLM API to generate answers. This approach has three critical weaknesses: (1) dense retrieval compresses entire documents into a single vector, losing fine-grained token-level relevance signals; (2) generation is stateless and slow for multi-step validation; (3) there is no systematic self-evaluation of answer quality. This capstone solves all three: ColBERT's late-interaction preserves token-level matching, SGLang's RadixAttention accelerates multi-step structured programs, and LLM-as-a-Judge provides automated quality control. No other open-source stack combines these three advances into a unified, deployable system.

This capstone integrates knowledge from [[06 - Large Language Models/12 - Production RAG/04 - Production RAG System]], [[06 - Large Language Models/13 - vLLM and Advanced RAG/00 - Welcome to vLLM and Advanced RAG]], [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search]], and [[07 - AI Agents/15 - MCP and Agentic Prot/00 - Welcome to MCP and Agentic Protocols]].

---

## Module 1: Dual-Stage Indexing and Retrieval

### 1.1 Theoretical Foundation 🧠

Retrieval quality determines the ceiling of any RAG system. Dense retrieval (bi-encoder) encodes queries and documents into fixed-length vectors and uses approximate nearest neighbor (ANN) search. It is fast and scalable but suffers from the "information bottleneck": a 768-dimensional vector must represent an entire document, inevitably losing nuance. ColBERT solves this with **late interaction**: instead of compressing the document into one vector, it stores embeddings for *every token* in the document. At query time, it computes a **MaxSim** score: for each query token, find the most similar document token, sum these maxima across query tokens. This captures fine-grained lexical and semantic matches (e.g., "neural network training" matching "training of deep neural nets") that dense retrieval misses.

The trade-off is computation: MaxSim over all documents is too slow. The solution is a **two-stage cascade**. Stage 1 uses fast dense ANN (e.g., Qdrant or FAISS) to retrieve the top-100 candidates. Stage 2 reranks these 100 with ColBERT MaxSim to produce the top-10. This gives the precision of ColBERT at the speed of ANN. PLAID (Performance-Optimized Late Interaction Driver) further accelerates ColBERT with pruning and quantization, making stage 2 feasible at millisecond latency.

### 1.2 Mental Model 📐

```
┌─────────────────────────────────────────────────────────────┐
│              Two-Stage Retrieval Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: "neural network optimization"                       │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────┐                                    │
│  │ Stage 1: Dense ANN  │                                    │
│  │  (Qdrant / FAISS)   │                                    │
│  │  Top-100 candidates │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ Stage 2: ColBERT    │                                    │
│  │  MaxSim Reranking   │                                    │
│  │  Top-100 → Top-10   │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  Top-10 passages sent to SGLang for generation              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Syntax and Semantics 📝

```python
# WHY: Stage 1 uses dense embeddings for fast ANN retrieval.
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

def dense_retrieve(query: str, top_k: int = 100):
    # WHY: Dense vector is fast but lossy; we over-fetch to compensate for lower precision.
    vector = encoder.encode(query).tolist()
    results = qdrant.search(
        collection_name="passages",
        query_vector=vector,
        limit=top_k
    )
    return [r.payload["text"] for r in results]

# WHY: Stage 2 uses ColBERT's fine-grained token embeddings.
from colbert import Searcher

colbert_searcher = Searcher(index="plaid_index")

def colbert_rerank(query: str, candidates: list, top_k: int = 10):
    # WHY: MaxSim scores every query token against every
    # document token, capturing precise local relevance.
    scores = colbert_searcher.search(query, k=top_k, filter_fn=candidates)
    return scores
```

### 1.4 Visual Representation 🖼️

```mermaid
graph LR
    Q[Query] --> E[Dense Encoder]
    E --> F[ANN Index]
    F -->|Top-100| C[ColBERT MaxSim]
    C -->|Top-10| G[SGLang Generator]
```

```mermaid
graph TD
    subgraph Dense["Dense Retrieval (Fast, Lossy)"]
        D1[Document] --> D2[Single Vector 768d]
    end
    subgraph ColBERT["ColBERT Late Interaction (Slow, Precise)"]
        C1[Document] --> C2[Token Vectors N×128]
        C2 --> C3[MaxSim with Query Tokens]
    end
```

### 1.5 Application in ML/AI Systems 🤖

Real case: **Stanford NLP** uses ColBERT in their IR course materials and open-source search demos, reporting 15-25% recall improvements over dense-only systems on MS MARCO.

| ML Use Case             | This Concept             | Impact                              |
|-------------------------|--------------------------|-------------------------------------|
| Legal document search   | Dense + ColBERT cascade  | +20% recall on precise entity matches |
| Scientific literature RAG| PLAID-accelerated MaxSim | Sub-100ms reranking on 100 candidates |
| E-commerce search       | Token-level matching     | Better brand + attribute alignment  |

### 1.6 Common Pitfalls ⚠️

⚠️ **Pitfall:** Using ColBERT as the primary retriever on large corpora without an initial filter. Root cause: MaxSim is O(query_tokens × doc_tokens), which is infeasible at billion-document scale.

💡 **Tip:** Always cascade: dense ANN first, ColBERT second. Think: "Filter wide, score deep."

### 1.7 Knowledge Check ❓

1. Why does a single dense vector lose information compared to token-level embeddings?
2. In the MaxSim operation, what does "late interaction" mean?
3. If dense retrieval has 60% recall@10 and ColBERT reranking improves it to 75%, what is the relative gain?

---

## Module 2: SGLang Structured Generation and LLM-as-a-Judge

### 2.1 Theoretical Foundation 🧠

Once the top-10 passages are retrieved, the LLM must synthesize an answer. Standard generation APIs produce free text, which may hallucinate or fail to cite sources. Structured generation forces the model to emit answers in a predictable format (e.g., JSON with `answer` and `citations` fields). SGLang enforces this at the token level, meaning invalid tokens are never sampled. This is faster and more reliable than generating raw text and parsing it afterwards.

Beyond generation, quality control is essential. The **LLM-as-a-Judge** pattern uses a rubric to score the generated answer. In a traditional system, each judgment is an independent API call with the full rubric prompt. SGLang's RadixAttention caches the rubric KV state, so judging 100 answers reuses the rubric prefix entirely. When combined into a single SGLang *program*, the flow becomes: generate answer → validate citations → judge quality, all within one structured execution graph with full KV-cache reuse between steps.

### 2.2 Mental Model 📐

```
┌─────────────────────────────────────────────────────────────┐
│           SGLang RAG Program (Single Execution)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Generate Answer                                    │
│  ┌─────────────────────────────────────────┐                │
│  │ Context: [Passage1] [Passage2] ...      │                │
│  │ Query:  "Explain X"                     │                │
│  │ gen("answer", json_schema=...)          │                │
│  └─────────────────────────────────────────┘                │
│                    │                                        │
│                    ▼                                        │
│  Step 2: Validate Citations                                │
│  ┌─────────────────────────────────────────┐                │
│  │ Does answer.citations[] exist in context?│               │
│  │ If not, flag hallucination.             │                │
│  └─────────────────────────────────────────┘                │
│                    │                                        │
│                    ▼                                        │
│  Step 3: Judge Quality                                      │
│  ┌─────────────────────────────────────────┐                │
│  │ Rubric: "Accuracy, Coverage, Clarity"   │                │
│  │ select("score", [1,2,3,4,5])            │                │
│  │ (Rubric KV cached from prior runs)      │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Syntax and Semantics 📝

```python
import sglang as sgl

@sgl.function
def rag_pipeline(s, query: str, passages: list, rubric: str):
    # WHY: Build context from retrieved passages.
    context = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    s += f"Context:\n{context}\n\nQuery: {query}\n\n"

    # WHY: Generate structured answer with inline citations.
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["answer", "citations"]
    }
    s += "Answer (JSON): "
    s += sgl.gen("response", max_tokens=512, json_schema=schema)

    # WHY: Validate that citations point to real passage indices.
    answer = json.loads(s["response"])
    valid_citations = all(1 <= c <= len(passages) for c in answer["citations"])
    s += f"\nValidation: citations_valid={valid_citations}\n"

    # WHY: Judge quality. The rubric prefix is cached by RadixAttention
    # when this program is run in a batch across many queries.
    s += f"Rubric: {rubric}\nQuality Score: "
    s += sgl.select("score", choices=["1", "2", "3", "4", "5"])
```

### 2.4 Visual Representation 🖼️

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gateway
    participant R as Retriever
    participant S as SGLang
    participant RT as Radix Tree

    U->>G: Query
    G->>R: Retrieve passages
    R-->>G: Top-10 passages
    G->>S: Run rag_pipeline()
    S->>S: Generate structured answer
    S->>S: Validate citations
    S->>RT: Cache rubric KV
    S->>S: Judge with select()
    S-->>G: Answer + Score
    G-->>U: Final response
```

```mermaid
graph TD
    A[Query + Passages] --> B[Generate JSON Answer]
    B --> C[Validate Citations]
    C -->|Valid| D[Judge Quality]
    C -->|Invalid| E[Flag Hallucination]
    D --> F[Return Answer + Score]
    E --> F
```

### 2.5 Application in ML/AI Systems 🤖

Real case: **Perplexity.ai** uses a similar cascade (retrieval + structured answer + citation validation) to provide cited, trustworthy answers in real time.

| ML Use Case            | This Concept                  | Impact                             |
|------------------------|-------------------------------|------------------------------------|
| Cited answer generation| Structured JSON gen           | 100% valid citation format         |
| Quality assurance      | LLM-as-a-Judge + caching      | 3x faster batched evaluation       |
| Multi-step reasoning   | Single SGLang program         | Reuses KV across all steps         |

### 2.6 Common Pitfalls ⚠️

⚠️ **Pitfall:** Allowing the model to cite passages that were not actually retrieved. Root cause: LLMs are prone to hallucinating plausible-looking citation numbers.

💡 **Tip:** Always validate citation indices against the retrieved passage list before returning to the user. Think: "Trust but verify indices."

### 2.7 Knowledge Check ❓

1. Why is token-level JSON generation faster than generate-then-parse?
2. How does RadixAttention help when judging 50 answers with the same rubric?
3. What should happen if a citation index is out of range?

---

## Module 3: End-to-End Evaluation and Docker Deployment

### 3.1 Theoretical Foundation 🧠

A RAG system is only as good as its weakest component. Evaluating the full pipeline requires metrics for each stage and the system as a whole. **Recall@k** measures retrieval quality: what fraction of relevant documents appear in the top-k. **Latency p50/p99** measures user-perceived speed at median and tail. **Throughput (req/s)** measures system capacity. **Answer accuracy** measures generation quality, often via LLM-as-a-Judge or exact match on QA datasets.

The baseline for comparison is a dense-only retrieval system served by standard vLLM. The hypothesis is that ColBERT reranking improves recall, and SGLang structured programs reduce latency for multi-step generation. To validate this, you must hold all other variables constant: same model, same passages, same queries. Only the retrieval and generation backends change.

For deployment, a Docker Compose stack isolates each service: Qdrant for dense vectors, a ColBERT/PLAID container for reranking, SGLang server for generation, and a FastAPI gateway to orchestrate the pipeline. This mirrors production microservice architectures and ensures reproducibility.

### 3.2 Mental Model 📐

```
┌─────────────────────────────────────────────────────────────┐
│              Docker Compose Stack                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │   Qdrant    │   │  ColBERT    │   │  SGLang Server  │   │
│  │  (Vectors)  │   │  (PLAID)    │   │  (LLM, TP=4)    │   │
│  │   :6333     │   │   :8000     │   │    :30000       │   │
│  └──────┬──────┘   └──────┬──────┘   └────────┬────────┘   │
│         │                 │                     │            │
│         └─────────────────┼─────────────────────┘            │
│                           ▼                                 │
│                  ┌─────────────────┐                        │
│                  │  FastAPI Gateway│                        │
│                  │    :8000        │                        │
│                  └─────────────────┘                        │
│                           │                                 │
│                           ▼                                 │
│                        [User]                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Syntax and Semantics 📝

```yaml
# docker-compose.yml
# WHY: Each service is isolated, scalable, and stateless
# except for Qdrant, which persists vector data.
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  colbert:
    build: ./colbert_service
    ports:
      - "8001:8000"
    volumes:
      - ./plaid_index:/index:ro

  sglang:
    image: sglang/sglang:latest
    command: >
      python -m sglang.launch_server
      --model-path meta-llama/Llama-2-70b-chat-hf
      --tp 4 --port 30000
    ports:
      - "30000:30000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]

  gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - colbert
      - sglang

volumes:
  qdrant_storage:
```

```python
# gateway/main.py
from fastapi import FastAPI
import sglang as sgl
import requests

app = FastAPI()
sgl.set_default_backend(sgl.RuntimeEndpoint("http://sglang:30000"))

@app.post("/rag")
def rag(query: str):
    # WHY: Stage 1 - dense retrieval
    r = requests.post("http://qdrant:6333/collections/passages/points/search", json={
        "vector": encode(query),
        "limit": 100
    })
    candidates = [p["payload"]["text"] for p in r.json()["result"]]

    # WHY: Stage 2 - ColBERT rerank
    r = requests.post("http://colbert:8000/rerank", json={
        "query": query,
        "candidates": candidates,
        "top_k": 10
    })
    top_passages = r.json()["passages"]

    # WHY: Stage 3 - structured generation + judge
    state = rag_pipeline.run(query=query, passages=top_passages,
                             rubric="Accuracy and completeness.")
    return {
        "answer": json.loads(state["response"]),
        "score": state["score"],
        "passages": top_passages
    }
```

### 3.4 Visual Representation 🖼️

```mermaid
graph TB
    subgraph Docker["Docker Compose Network"]
        Q[Qdrant :6333]
        C[ColBERT :8001]
        S[SGLang :30000]
        G[FastAPI Gateway :8000]
    end
    U[User] -->|POST /rag| G
    G -->|ANN Search| Q
    G -->|MaxSim Rerank| C
    G -->|Structured Gen| S
```

```mermaid
graph LR
    B[Baseline: Dense + vLLM] -->|Recall@10| BR[60%]
    B -->|Latency p99| BL[1.8s]
    O[Our System: ColBERT + SGLang] -->|Recall@10| OR[78%]
    O -->|Latency p99| OL[0.9s]
```

### 3.5 Application in ML/AI Systems 🤖

Real case: **Enterprise knowledge bases** at a Fortune 500 company use dense+ColBERT cascades with structured generation to answer regulatory compliance queries, achieving >90% citation accuracy.

| ML Use Case              | This Concept              | Impact                         |
|--------------------------|---------------------------|--------------------------------|
| Production RAG serving   | Docker Compose microservices | Scalable, reproducible stack  |
| Latency-sensitive search | ColBERT + SGLang          | <1s end-to-end p99             |
| Quality monitoring       | LLM-as-a-Judge            | Automated answer scoring       |

### 3.6 Common Pitfalls ⚠️

⚠️ **Pitfall:** Failing to warm up the SGLang server before running benchmarks. Root cause: the first request triggers CUDA kernel compilation and model loading, skewing latency measurements.

💡 **Tip:** Send 10-20 warmup requests before recording metrics. Think: "Warm the engine before timing the race."

### 3.7 Knowledge Check ❓

1. Why must the baseline and experimental systems use the same model and passages?
2. What metric best captures the improvement from ColBERT reranking?
3. How would you horizontally scale the gateway if throughput is the bottleneck?

---

## 📦 Compression Code

```python
"""
Complete High-Performance RAG Stack
Demonstrates: two-stage retrieval, structured generation,
citation validation, LLM-as-a-Judge, and Docker orchestration.
"""
import json, requests, sglang as sgl
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

# ─── Retrieval Functions ───
def dense_retrieve(query: str):
    vec = encoder.encode(query).tolist()
    r = requests.post("http://localhost:6333/collections/passages/points/search",
                      json={"vector": vec, "limit": 100})
    return [p["payload"]["text"] for p in r.json()["result"]]

def colbert_rerank(query: str, candidates: list):
    r = requests.post("http://localhost:8001/rerank",
                      json={"query": query, "candidates": candidates, "top_k": 10})
    return r.json()["passages"]

# ─── SGLang RAG Program ───
@sgl.function
def rag_pipeline(s, query: str, passages: list, rubric: str):
    ctx = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    s += f"Context:\n{ctx}\n\nQuery: {query}\n\nAnswer (JSON): "
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["answer", "citations"]
    }
    s += sgl.gen("response", max_tokens=512, json_schema=schema)
    ans = json.loads(s["response"])
    valid = all(1 <= c <= len(passages) for c in ans["citations"])
    s += f"\nCitations valid: {valid}\nRubric: {rubric}\nScore: "
    s += sgl.select("score", choices=["1", "2", "3", "4", "5"])

# ─── End-to-End Query ───
if __name__ == "__main__":
    query = "What is speculative decoding?"
    candidates = dense_retrieve(query)
    top10 = colbert_rerank(query, candidates)
    state = rag_pipeline.run(query=query, passages=top10,
                             rubric="Accuracy and clarity.")
    print("Answer:", state["response"])
    print("Score:", state["score"])
```

## 🎯 Documented Project

### Description
Build and evaluate a production RAG system that outperforms a dense-only baseline on retrieval recall and generation latency. The system uses Qdrant for dense ANN, ColBERT/PLAID for reranking, and SGLang for structured answer generation with self-judgment.

### Functional Requirements
- Index 100K passages with dense embeddings and ColBERT token embeddings.
- Accept user queries and return cited, structured answers.
- Validate all citations against retrieved passages before returning.
- Score answer quality with an LLM-as-a-Judge program.
- Expose a single `/rag` endpoint via FastAPI.

### Main Components
1. **Indexer**: Script to build Qdrant and PLAID indices from passage corpus.
2. **Retriever Service**: Qdrant container + ColBERT rerank container.
3. **Generator Service**: SGLang server with structured program loaded.
4. **API Gateway**: FastAPI orchestrating retrieval and generation.
5. **Evaluator**: Script computing recall@10, latency, throughput, and judge scores.

### Success Metrics
- **ColBERT reranking improves recall@10 by >15%** over dense-only retrieval on the test query set.
- **SGLang reduces multi-turn latency by >40%** vs vLLM for structured judge programs, due to RadixAttention caching.
- **Citation validation achieves 100% precision**: no hallucinated citations pass through.
- **End-to-end p99 latency < 1.5s** for queries with 100 candidates and 10 reranked passages.

## 🎯 Key Takeaways

- **Two-stage retrieval** (dense ANN + ColBERT MaxSim) combines speed and precision, improving recall by >15% over dense-only systems.
- **ColBERT's late interaction** preserves token-level relevance signals that single-vector dense retrieval compresses away.
- **SGLang structured programs** enforce valid JSON citations at the token level, eliminating parse failures and hallucinated formats.
- **RadixAttention** caches rubric KV state across judge evaluations, providing 3-5x speedups for batched quality scoring.
- **Docker Compose** isolates retrieval, reranking, and generation into scalable microservices, enabling reproducible production deployment.
- **End-to-end evaluation** requires controlling for model and data when comparing retrieval backends; metrics must cover recall, latency, throughput, and accuracy.
- **Citation validation** is a mandatory guardrail: always verify that cited passage indices exist in the retrieved set before returning answers to users.

## References

- ColBERT: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction" (2020)
- PLAID: Khattab et al., "PLAID: An Efficient Engine for Late Interaction Retrieval" (2022)
- SGLang: https://github.com/sgl-project/sglang
- Qdrant: https://qdrant.tech/documentation/
- Related: [[06 - Large Language Models/12 - Production RAG/04 - Production RAG System]]
- Related: [[06 - Large Language Models/13 - vLLM and Advanced RAG/00 - Welcome to vLLM and Advanced RAG]]
- Related: [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search]]
- Related: [[07 - AI Agents/15 - MCP and Agentic Prot/00 - Welcome to MCP and Agentic Protocols]]
- Related: [[06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/00 - Welcome to HuggingFace Transformers Deep Dive]]
