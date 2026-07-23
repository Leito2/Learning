# 🎯 04 - Production RAG Patterns — Comparison, Selection, and Integration

> **The decision matrix that determines whether you reach for LangChain, LlamaIndex, Haystack, or txtai. Plus the integration patterns that let you mix them.**

## 🎯 Learning Objectives
- Map production RAG scenarios to the right framework (decision tree)
- Combine LangChain, LlamaIndex, Haystack, and txtai in a single system
- Implement cross-framework retrievers and integrations
- Wire RAG systems with LangFuse cost attribution and Phoenix observability
- Apply cost optimization patterns: caching, hybrid, batching
- Design production RAG architectures that scale to 100M+ documents

## Introduction

The four RAG frameworks covered so far are not interchangeable. Each has a specific sweet spot. The choice depends on:

1. **Data characteristics** — single corpus vs multi-modal vs multi-tenant
2. **Query patterns** — single-hop vs multi-hop vs agentic
3. **Deployment constraints** — small VM vs Kubernetes vs deepset Cloud
4. **Team familiarity** — Python-first vs enterprise Java vs Node.js

This note synthesizes the decision into a clear framework and shows the integration patterns that let you **mix frameworks** when each is the right tool for the job.

For the user's portfolio projects (LLM Edge Gateway, Automated LLM Evaluation Suite, Multi-Agent Research System, StayBot, plus all the capstones from the recent courses), **the right answer is "use multiple frameworks"** — and this note shows how.

![Framework selection matrix](https://example.com/framework-matrix.png)

---

## 1. The Decision Tree

```
START
  │
  ├─ Multi-hop retrieval with knowledge graphs? 
  │   │
  │   YES ── txtai (graph + embeddings native)
  │   │
  │   NO
  │     │
  │     ├─ Single-corpus small (<1M documents)?
  │     │   │
  │     │   YES ── txtai (simplest API)
  │     │   │
  │     │   NO
  │     │     │
  │     │     ├─ Data-heavy ingestion + multi-modal?
  │     │     │   │
  │     │     │   YES ── LlamaIndex (best ingestion)
  │     │     │   │
  │     │     │   NO
  │     │     │     │
  │     │     │     ├─ Enterprise production + deepset Cloud?
  │     │     │     │   │
  │     │     │     │   YES ── Haystack (DAG pipelines + enterprise support)
  │     │     │     │   │
  │     │     │     │   NO
  │     │     │     │     │
  │     │     │     │     └─ Default choice: LangChain (ubiquitous, broad integration)
  │     │     │     │     │
  │     └─ Cross-framework integration required? ── YES: prefer Haystack (built-in adapters to LangChain/LlamaIndex)
```

In practice, **most teams start with LangChain** because of its ubiquity, then specialize. The right mature production stack often uses **2-3 frameworks composed**:

- **LlamaIndex for ingestion** (best loader/splitter pipeline)
- **Haystack for retrieval pipelines** (explicit DAG, deepset Cloud)
- **LangChain for orchestration** (broad LLM integration)

Or **txtai for everything when simplicity matters**.

---

## 2. Framework-Specific Decision Details

### 2.1 LangChain — use when

- You need broad LLM integration (OpenAI, Anthropic, Gemini, Cohere, Mistral, etc.)
- Your team has LangChain experience
- You're building a complex multi-step agent (LangGraph integration)
- Your retrieval needs are standard (semantic + BM25 + reranking)
- Cost is not the primary concern (LangChain's abstraction overhead)

### 2.2 LlamaIndex — use when

- Your data is multi-modal (PDFs, images, audio, video)
- You need sophisticated ingestion (chunking, metadata extraction, multi-index)
- You're building a chat-with-your-data application
- You want agents that use retrieval as a tool
- Vector search is the primary use case

### 2.3 Haystack — use when

- You're building enterprise production RAG
- deepset Cloud is available (Fortune 500 customers)
- You need typed DAG pipelines with explicit validation
- Cross-framework integration (LangChain + LlamaIndex + custom) is required
- Production testing rigor matters
- Compliance / on-prem deployment is required

### 2.4 txtai — use when

- Multi-hop retrieval is a primary requirement
- You want the smallest API surface (5-line setup vs 50+)
- Cold-start matters (serverless deployment, edge inference)
- You're OK without LangChain's broad integration
- Simplicity wins over flexibility

---

## 3. Cross-Framework Integration Patterns

### 3.1 Haystack + LangChain (best of both worlds)

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.langchain import LangChainRetrieverAdapter

# LangChain retriever (broad ecosystem)
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

lc_qdrant = Qdrant.from_documents(documents, OpenAIEmbeddings(), location=":memory:")
lc_retriever = lc_qdrant.as_retriever(search_kwargs={"k": 10})

# Add LangChain reranking
llm_extractor = LLMChainExtractor.from_llm(OpenAI(temperature=0))
lc_rerank_retriever = ContextualCompressionRetriever(
    base_compressor=llm_extractor,
    base_retriever=lc_retriever,
)

# Wrap in Haystack
haystack_retriever = LangChainRetrieverAdapter(retriever=lc_rerank_retriever)

# Build the Haystack pipeline
pipeline = Pipeline()
pipeline.add_component("retriever", haystack_retriever)
pipeline.add_component("prompt_builder", PromptBuilder(template="..."))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))

pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")
```

This composes:
- **LangChain's ContextualCompressionRetriever** (LLM-based reranking)
- **Haystack's typed DAG pipeline**
- **OpenAI's generator** as the LLM

### 3.2 LlamaIndex + Haystack (best for multi-modal)

```python
from haystack import Pipeline
from haystack_integrations.components.retrievers.llama_index import LlamaIndexRetrieverAdapter
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# LlamaIndex for multi-modal ingestion (PDF, images, audio)
documents = SimpleDirectoryReader("data/mixed").load_data()  # auto-detects PDF, images, audio
multi_modal_index = VectorStoreIndex.from_documents(
    documents,
    multi_modal_llm=OpenAIMultiModal(model="gpt-4o"),  # multi-modal embeddings
)

# Wrap the multi-modal retriever in Haystack
haystack_retriever = LlamaIndexRetrieverAdapter(
    query_engine=multi_modal_index.as_query_engine(similarity_top_k=5),
)

# Use in Haystack pipeline
pipeline = Pipeline()
pipeline.add_component("retriever", haystack_retriever)
# ... rest of pipeline
```

This combines:
- **LlamaIndex's multi-modal ingestion** (PDFs, images)
- **Haystack's explicit DAG pipeline**
- OpenAI's multi-modal embeddings

### 3.3 txtai + Haystack (multi-hop + enterprise)

```python
from haystack import Pipeline, component
from haystack.dataclasses import Document


@component
class TxtaiRetriever:
    """Wraps a txtai embeddings index as a Haystack component."""
    
    @component.output_fields(documents=list[Document])
    def run(self, query: str, limit: int = 5, graph_depth: int = 0) -> dict:
        # Query txtai with graph traversal
        results = self.txtai_embeddings.search(
            query,
            limit=limit,
            graph={"depth": graph_depth} if graph_depth > 0 else False,
        )
        documents = [
            Document(
                content=r["text"],
                meta={"score": r["score"], "id": r["id"]},
            )
            for r in results
        ]
        return {"documents": documents}


# Build a hybrid pipeline: txtai for retrieval, Haystack for orchestration
txtai_retriever = TxtaiRetriever(txtai_embeddings=global_txtai_index)

pipeline = Pipeline()
pipeline.add_component("multi_hop_retriever", txtai_retriever)
pipeline.add_component("cross_encoder_ranker", SentenceTransformersSimilarityRanker(top_k=3))
pipeline.add_component("prompt_builder", PromptBuilder(template="..."))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))

pipeline.connect("multi_hop_retriever.documents", "cross_encoder_ranker.documents")
pipeline.connect("cross_encoder_ranker.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")
```

This combines:
- **txtai's multi-hop graph retrieval** for the initial search
- **Haystack's cross-encoder reranking** for precision
- **Haystack's typed pipeline + LLM** for the production synthesis

---

## 4. Cost Optimization Across Frameworks

Cost patterns that work in all four:

### 4.1 Caching

```python
import redis

redis_client = redis.Redis(...)


def cached_search(query: str, search_fn, ttl: int = 3600) -> list[dict]:
    """Generic cache wrapper for any search function."""
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    results = search_fn(query)
    redis_client.setex(cache_key, ttl, json.dumps(results))
    return results


# Wrap any framework
def haystack_search(query):
    return pipeline.run({"text_embedder": {"text": query}})["retriever"]["documents"]

results = cached_search(query, haystack_search)
```

### 4.2 Prefix caching

Most RAG prompts share a long system prompt + retrieved context. Both providers (OpenAI, Anthropic, Together) and the four frameworks support prefix caching:

- **Together AI**: 5-10 minute TTL, 80%+ savings
- **Anthropic prompt caching**: 5 minute TTL, 90% input cost savings
- **OpenAI automatic caching**: partial caching for repeated prefixes

For the user's profile, **prefix caching on Together/Fireworks** is the most cost-effective pattern (covered in [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization/04 - Serverless Cost Optimization and Patterns|Serverless Cost Optimization]]).

### 4.3 Reranking only when needed

```python
@observe()
def smart_rag(query: str) -> dict:
    # Step 1: Cheap retrieval (vector + BM25)
    docs = retriever.search(query, limit=20)
    
    # Step 2: Score-based filtering (no LLM call)
    high_confidence = [d for d in docs if d["score"] > 0.7]
    
    if len(high_confidence) >= 3:
        # High-confidence retrieval → skip reranking
        return {"docs": high_confidence[:5], "reranked": False}
    
    # Step 3: Reranker only when retrieval is uncertain (LLM call)
    reranked = ranker.rerank(query, docs, top_k=5)
    return {"docs": reranked, "reranked": True}
```

This pattern saves 30-50% of reranking costs by only invoking the cross-encoder when retrieval is uncertain.

---

## 5. Observability Across Frameworks

All four frameworks integrate with LangFuse and Phoenix via OpenTelemetry:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Set up Phoenix OTLP exporter
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://phoenix:4317/v1/traces"))
)
```

For LangFuse cost attribution (covered in [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability]]):

```python
from langfuse import observe, langfuse_context

@observe()
def run_hybrid_rag(query: str, tenant_id: str, framework: str) -> dict:
    langfuse_context.update_current_observation(metadata={
        "tenant_id": tenant_id,
        "framework": framework,  # haystack/langchain/txtai/llamaindex
    })
    
    if framework == "haystack":
        result = haystack_pipeline.run({"text_embedder": {"text": query}})
    elif framework == "txtai":
        result = txtai_rag(query)
    elif framework == "llamaindex":
        result = llama_index_query_engine.query(query)
    else:
        result = langchain_chain.invoke({"query": query})
    
    langfuse_context.update_current_observation(metadata={
        "retrieved_count": len(result["documents"]),
        "tokens": calculate_tokens(result["answer"]),
    })
    return result
```

The LangFuse UI shows the framework used, retrieval count, and per-tenant cost.

---

## 6. Production Architectures

### 6.1 Single-framework architectures

**LangChain-only** (most common):
- FastAPI + LangServe + LangChain RAG
- Single deployment unit
- Best for: small to medium teams

**LlamaIndex-only**:
- FastAPI + LlamaIndex QueryEngine
- Best for: data-heavy multi-modal RAG

**Haystack-only**:
- FastAPI + Haystack pipeline
- Best for: enterprise production + deepset Cloud

**txtai-only**:
- FastAPI + txtai API server
- Best for: simplicity + multi-hop

### 6.2 Multi-framework architectures (recommended for production)

The user's portfolio pattern:

```
┌─────────────────────────────────────────────┐
│  FastAPI Gateway                            │
│  ├─ /ingest → LlamaIndex multi-modal       │
│  ├─ /search → txtai multi-hop             │
│  ├─ /query → Haystack hybrid + rerank    │
│  ├─ /agent → LangChain/LangGraph          │
│  └─ /evaluate → RAGAS pipeline            │
└─────────────────────────────────────────────┘
```

Each endpoint uses the best framework for its task. The gateway orchestrates the routing.

### 6.3 Reference architecture for the user

Based on the user's profile (RAG + LangGraph + multi-framework portfolio):

1. **Ingestion**: LlamaIndex multi-modal loaders
2. **Indexing**: Haystack Document Store + Qdrant (canonical)
3. **Retrieval**: Haystack hybrid + cross-encoder reranking
4. **Multi-hop**: txtai graph traversal (when needed)
5. **Generation**: LiteLLM routed across Together/Fireworks/OpenAI
6. **Agent**: LangGraph for cyclic workflows
7. **Observability**: LangFuse + Phoenix
8. **Evaluation**: RAGAS + LangFuse experiments

This stack uses 5+ frameworks but each has a clear responsibility.

---

## 7. Antipatterns

### 7.1 Antipattern 1: Choosing based on popularity

```python
# ❌ "Everyone uses LangChain so we use LangChain"
# But our use case is multi-hop → should use txtai

# ✅ Choose based on requirements
# Use a decision tree, not community consensus
```

### 7.2 Antipattern 2: Framework lock-in

```python
# ❌ Framework A's retriever + Framework B's reranker + Framework C's agent
# Each works alone, but the integration glue is 100s of lines and brittle

# ✅ Use one framework end-to-end UNLESS the cross-framework cost is justified
```

### 7.3 Antipattern 3: Using txtai for complex agents

```python
# ❌ txtai's workflow DSL is limited for complex multi-step agents
workflow = Workflow([Task1, Task2, Task3])  # can't express conditional branching easily

# ✅ Use LangGraph for agents, txtai for retrieval
```

### 7.4 Antipattern 4: Skipping offline evaluation

```python
# ❌ Deploy a new RAG system without measuring retrieval quality
pipeline.deploy()  # retrieval precision might be 40%

# ✅ Always run RAGAS evals on a golden dataset before deployment
results = ragas_eval(pipeline, golden_dataset)
assert results["faithfulness"] > 0.85
```

### 7.5 Antipattern 5: No fallback for failed retrievals

```python
# ❌ System crashes if retrieval returns empty
result = pipeline.run({"query": user_query})
# ... uses result["documents"][0] but list might be empty!

# ✅ Handle empty retrievals gracefully
documents = result.get("documents", [])
if not documents:
    return {"answer": "I don't have enough context to answer.", "documents": []}
```

---

## 🎯 Key Takeaways

- The four RAG frameworks have distinct sweet spots: LangChain (ubiquitous), LlamaIndex (multi-modal), Haystack (enterprise), txtai (multi-hop simplicity).
- The decision tree: multi-hop → txtai; data-heavy → LlamaIndex; enterprise → Haystack; default → LangChain.
- Cross-framework integration: Haystack has built-in adapters for LangChain and LlamaIndex retrievers.
- The user's portfolio pattern: LlamaIndex for ingestion, Haystack for retrieval, LiteLLM for generation, LangGraph for agents.
- Cost optimization: caching (Redis), prefix caching (Together), selective reranking.
- Observability: all four integrate with LangFuse and Phoenix via OpenTelemetry.
- Avoid choosing by popularity, framework lock-in, txtai for complex agents, no evaluation, no empty-retrieval handling.

## References

- [[06 - Large Language Models/12 - Production RAG|Production RAG]] — foundational RAG patterns
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/01 - Haystack Fundamentals - Pipelines, Components, Retrievers|Note 01 — Haystack Fundamentals]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/02 - Haystack Advanced Pipelines - Hybrid Search, Reranking, Agents, and Evaluation|Note 02 — Haystack Advanced]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/03 - txtai Fundamentals - Semantic Search, Graphs, and RAG in One Library|Note 03 — txtai]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/05 - Capstone - Production Hybrid RAG Service|Note 05 — Capstone]]
- [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference\|ColBERT, SGLang]] — token-level retrieval
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider routing
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured RAG responses
- [[06 - Large Language Models/23 - Serverless LLM Platforms and Cost Optimization|Serverless LLM Platforms]] — Together/Fireworks for LLM
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — agentic workflows
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — cost attribution
- [[09 - MLOps y Produccion/20 - RAG Evaluation Deep Dive|RAG Evaluation Deep Dive]] — RAGAS
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant, Milvus, Pinecone
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — service deployment