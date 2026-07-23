# 🎯 02 - Haystack Advanced Pipelines — Hybrid Search, Reranking, Agents, and Evaluation

> **Production-grade RAG patterns: hybrid retrieval, agentic loops, evaluation pipelines, and observability. The patterns that distinguish a real production RAG system from a research demo.**

## 🎯 Learning Objectives
- Implement hybrid retrieval with BM25 + dense + cross-encoder reranking
- Build agentic RAG loops with Haystack's `Agent` component and tool registration
- Wire Haystack with LangChain and LlamaIndex retrievers via the integrations interface
- Set up offline evaluation with RAGAS-style metrics for Haystack pipelines
- Deploy Haystack to deepset Cloud and Kubernetes
- Configure observability via OpenTelemetry, LangFuse, and Phoenix

## Introduction

Note 01 covered the foundation: pipelines, components, retrievers, generators. This note covers the **production patterns** that turn a working pipeline into a system that meets enterprise SLAs: 99.9% uptime, sub-200ms p99 retrieval latency, sub-1% retrieval error rate, full observability.

The five patterns that matter most:

1. **Hybrid retrieval with reranking** — combining keyword and semantic search with cross-encoder refinement
2. **Agentic RAG loops** — multi-step reasoning where the agent decides what to retrieve next
3. **Cross-framework integration** — using LangChain/LlamaIndex retrievers inside Haystack pipelines
4. **Offline evaluation pipelines** — RAGAS-style metrics for golden dataset testing
5. **Production deployment** — deepset Cloud, Kubernetes, observability

Mastering these patterns is the difference between "I built a RAG demo" and "I shipped a production RAG system." Hiring managers at deepset's enterprise customers (banks, insurers, defense) recognize this rigor immediately.

![Hybrid retrieval pipeline](https://haystack.deepset.ai/assets/img/hybrid_retrieval.png)

---

## 1. Hybrid Retrieval with Cross-Encoder Reranking

The canonical production retrieval pipeline:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import PromptBuilder


# Document store with BM25 enabled
document_store = InMemoryDocumentStore(use_bm25=True)


def build_hybrid_pipeline(document_store) -> Pipeline:
    pipeline = Pipeline()
    
    # Embedding model for dense retrieval
    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    
    # Two retrievers: dense (semantic) + sparse (BM25)
    pipeline.add_component(
        "dense_retriever",
        InMemoryEmbeddingRetriever(document_store=document_store, top_k=20),
    )
    pipeline.add_component(
        "sparse_retriever",
        InMemoryBM25Retriever(document_store=document_store, top_k=20),
    )
    
    # Reciprocal Rank Fusion joiner
    pipeline.add_component(
        "joiner",
        DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[0.6, 0.4]),
    )
    
    # Cross-encoder reranker
    pipeline.add_component(
        "ranker",
        SentenceTransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=5,
        ),
    )
    
    # LLM
    template = """Answer the question based on the context.

Context:
{% for doc in documents %}
---
{{ doc.content }}
{% endfor %}

Question: {{ query }}

Provide a detailed answer with citations.
Answer: """
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))
    
    # Connect
    pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
    pipeline.connect("dense_retriever.documents", "joiner.documents")
    pipeline.connect("sparse_retriever.documents", "joiner.documents")
    pipeline.connect("joiner.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    
    return pipeline
```

Key decisions:
- **top_k=20 for retrieval, top_k=5 for reranking**: cast a wide net, refine to high-precision
- **Reciprocal Rank Fusion weights [0.6, 0.4]**: 60% dense, 40% BM25 (tune per dataset)
- **Cross-encoder reranker**: ms-marco-MiniLM is fast; bge-reranker-base is stronger

---

## 2. Reciprocal Rank Fusion vs Other Join Modes

The `DocumentJoiner` supports four join modes:

| Mode | Formula | When to use |
|------|---------|-------------|
| `concatenate` | Just append; preserve order | Simple, fast, doesn't rank merge |
| `merge` | Concatenate then deduplicate | General purpose |
| `reciprocal_rank_fusion` | RRF formula: 1/(k + rank) | Default for hybrid search |
| `distribution_based_rank_fusion` | Score-based | When retriever scores are calibrated |

RRF is the production default. The formula `1/(k + rank)` is robust to different score scales between retrievers (BM25 produces 0-50, dense produces 0-1). The `k` parameter (default 60) controls the impact of low-ranked documents.

Tuning weights:

```python
# Tune via grid search on a held-out dataset
for dense_weight in [0.4, 0.5, 0.6, 0.7]:
    for sparse_weight in [1 - dense_weight]:
        pipeline = build_hybrid_pipeline(weights=[dense_weight, sparse_weight])
        # Evaluate against golden dataset
        # ... pick best ...
```

In practice, dense weight 0.5-0.7 is optimal for most production workloads (semantic similarity matters more than exact keyword matches).

---

## 3. Agentic RAG Loops

For multi-step retrieval where the agent decides what to search next:

```python
from haystack.components.agents import Agent
from haystack.components.tools import tool
from haystack.dataclasses import ChatMessage
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@tool
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search the document store for relevant context.
    
    Args:
        query: The search query.
        top_k: Number of documents to return.
    
    Returns:
        List of {content, score, meta} dicts.
    """
    # The retriever is shared via global state
    documents = global_retriever.run(query=query, top_k=top_k)["documents"]
    return [{"content": d.content, "score": d.score, "meta": d.meta} for d in documents]


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A math expression like '2 + 2'.
    
    Returns:
        The result as a string.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Build the agent
agent = Agent(
    generator=OpenAIGenerator(model="gpt-4o"),
    tools=[search_documents, calculate],
    system_prompt="""You are a research assistant with access to a document store and a calculator.
    Use the tools to answer questions step by step. Cite your sources.""",
    max_steps=10,
)


# Run
result = agent.run(
    messages=[ChatMessage.from_user("What is the revenue growth rate from 2024 to 2025, and what factors contributed?")]
)
print(result["messages"][-1].content)
```

The agent:
1. Receives the question
2. Decides to call `search_documents` with an initial query
3. Reads the results, decides to call again with a refined query
4. Eventually calls `calculate` to compute the growth rate
5. Synthesizes a final answer

This is the **agentic RAG pattern** from [[06 - Large Language Models/12 - Production RAG]] with Haystack's explicit tool registration.

For production, register many tools:

```python
@tool
def search_documents(query: str, top_k: int = 5) -> list[dict]: ...

@tool
def search_emails(query: str, days_back: int = 30) -> list[dict]: ...

@tool
def search_calendar(start_date: str, end_date: str) -> list[dict]: ...

@tool
def search_database(sql_query: str) -> list[dict]: ...

@tool
def create_ticket(title: str, description: str, priority: str) -> dict: ...

@tool
def send_email(to: str, subject: str, body: str) -> dict: ...
```

Each tool wraps a function call. The agent decides which to invoke based on the user's question and prior steps.

---

## 4. Cross-Framework Integration

Haystack 2.x ships integrations for using LangChain and LlamaIndex retrievers inside Haystack pipelines:

### 4.1 LangChain retriever inside Haystack

```python
from haystack import component
from haystack_integrations.components.retrievers.langchain import LangChainRetrieverAdapter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings


# LangChain retriever
lc_qdrant = Qdrant.from_documents(documents, OpenAIEmbeddings(), location=":memory:")
lc_retriever = lc_qdrant.as_retriever(search_kwargs={"k": 5})

# Wrap in a Haystack component
haystack_retriever = LangChainRetrieverAdapter(retriever=lc_retriever)

# Use in a Haystack pipeline
pipeline.add_component("retriever", haystack_retriever)
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
```

### 4.2 LlamaIndex query engine inside Haystack

```python
from haystack_integrations.components.retrievers.llama_index import LlamaIndexRetrieverAdapter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

# Wrap in a Haystack component
haystack_retriever = LlamaIndexRetrieverAdapter(query_engine=query_engine)
```

This lets you **use the best of each framework** — LlamaIndex for ingestion, Haystack for pipelines, LangChain for chains — within a single pipeline.

---

## 5. Offline Evaluation Pipelines

Haystack 2.x ships evaluation primitives:

```python
from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
)
from haystack.evaluation import EvaluationPipeline


# Define evaluators
exact_match = AnswerExactMatchEvaluator()
context_relevance = ContextRelevanceEvaluator(model="gpt-4o-mini")  # LLM-as-judge
faithfulness = FaithfulnessEvaluator(model="gpt-4o-mini")  # LLM-as-judge


# Build evaluation pipeline
eval_pipeline = EvaluationPipeline(
    pipeline=rag_pipeline,  # the production RAG pipeline
    evaluators=[exact_match, context_relevance, faithfulness],
)


# Run against a golden dataset
golden_dataset = [
    {
        "query": "What is Haystack?",
        "ground_truth_answer": "Haystack is a framework for building production RAG systems.",
        "ground_truth_context": ["Haystack is a framework..."],
    },
    # ... 49 more
]

results = eval_pipeline.run(goldens=golden_dataset)
print(f"Exact match: {results['exact_match']:.2%}")
print(f"Context relevance: {results['context_relevance']:.2%}")
print(f"Faithfulness: {results['faithfulness']:.2%}")
```

Three evaluation dimensions:
- **Exact match**: response exactly matches ground truth (rare for RAG; only useful for QA datasets)
- **Context relevance**: retrieved documents are relevant to the query (LLM-as-judge)
- **Faithfulness**: response is supported by the retrieved context (no hallucination)

The faithfulness metric is the most important for production — it catches hallucination. Targets: context relevance >85%, faithfulness >90%.

CI integration (per [[09 - MLOps y Produccion/28 - Testing in ML Systems]]):

```yaml
# .github/workflows/eval.yml
- name: Run Haystack evaluation
  run: |
    python -m pytest tests/test_pipeline_eval.py
    # Fails if context_relevance < 0.85 OR faithfulness < 0.90
```

---

## 6. Streaming Responses

For low-latency UIs, stream the LLM response:

```python
from haystack.components.generators import OpenAIGenerator

# Streaming generator
generator = OpenAIGenerator(
    model="gpt-4o-mini",
    streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),
)

result = pipeline.run({...})
# Tokens stream as they're generated
```

For Server-Sent Events via FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from haystack import Pipeline

app = FastAPI()

@app.post("/query/stream")
async def stream_query(query: str):
    async def generate():
        generator = OpenAIGenerator(
            model="gpt-4o-mini",
            streaming_callback=lambda chunk: None,  # we yield directly
        )
        pipeline_with_streaming = build_pipeline(generator=generator)
        
        result = await pipeline_with_streaming.run_async({...})
        for token in result["llm"]["streaming_chunks"]:
            yield f"data: {token.content}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

This pattern integrates with [[06 - Large Language Models/22 - Instructor and Structured Generation]] and [[10 - Cloud, Infra y Backend/30 - WebSockets and Real-Time ML]] for production streaming.

---

## 7. Production Deployment

### 7.1 deepset Cloud

The fastest deployment path:

```bash
# Install CLI
pip install deepset-cloud-sdk

# Login
deepset-cloud login

# Deploy the pipeline
deepset-cloud deploy pipelines/rag_pipeline.yaml --name production-rag

# Get the URL
deepset-cloud pipeline info production-rag
```

deepset Cloud handles:
- Kubernetes deployment
- Auto-scaling
- SSO + RBAC
- Monitoring + alerting
- Cost attribution per workspace

Pricing: $99/seat/month for Enterprise, custom for self-hosted.

### 7.2 Self-hosted on Kubernetes

For self-hosted, package Haystack as a Docker image and deploy:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY pipeline.yaml .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# app/main.py
from fastapi import FastAPI
from haystack import Pipeline
from haystack.utils import Secret
import yaml

app = FastAPI()

with open("pipeline.yaml") as f:
    pipeline_dict = yaml.safe_load(f)
pipeline = Pipeline.from_dict(pipeline_dict)


@app.post("/query")
async def query(q: str):
    result = pipeline.run({"text_embedder": {"text": q}, ...})
    return {"answer": result["llm"]["replies"][0]}
```

Deploy with Helm:

```bash
helm install haystack-rag ./helm \
    --set image.tag=v1.0.0 \
    --set pipeline.configPath=/app/pipeline.yaml
```

For HA, use 3+ replicas with HPA based on request rate.

---

## 8. Observability

Haystack emits OpenTelemetry-compatible traces for every component:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://phoenix:4317/v1/traces"))
)
```

Every component invocation becomes a span. The Phoenix UI shows the pipeline DAG with timing per component.

For LangFuse cost attribution (covered in [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability]]):

```python
from langfuse.decorators import langfuse_context, observe

@observe()
def run_pipeline(query: str) -> dict:
    langfuse_context.update_current_observation(metadata={"query": query})
    result = pipeline.run({...})
    langfuse_context.update_current_observation(
        metadata={"retrieved_count": len(result["retriever"]["documents"])},
    )
    return result
```

Per-tenant cost tracking is automatic via LangFuse metadata.

---

## 9. Case Studies

### 9.1 Case real 1: Insurance Q&A

A large insurer uses Haystack with hybrid retrieval + reranking for claims adjuster Q&A:

```python
pipeline = build_hybrid_pipeline(document_store=policy_document_store)
# Add compliance filter
pipeline.add_component("compliance_filter", ComplianceFilter(min_clearance="L2"))
pipeline.connect("ranker.documents", "compliance_filter.documents")
pipeline.connect("compliance_filter.documents", "prompt_builder.documents")
```

Every retrieval respects the user's clearance level. The compliance filter is a custom component that drops documents above the user's clearance.

### 9.2 Case real 2: Customer support automation

A SaaS company uses agentic RAG for tier-1 support:

```python
@tool
def search_kb(query: str) -> list[dict]:
    """Search the knowledge base."""
    ...

@tool
def lookup_account(email: str) -> dict:
    """Look up a customer account."""
    ...

@tool
def create_ticket(title: str, description: str, priority: str) -> dict:
    """Create a support ticket."""
    ...

agent = Agent(
    generator=OpenAIGenerator(model="gpt-4o"),
    tools=[search_kb, lookup_account, create_ticket],
    system_prompt="""You are a tier-1 support agent. Try to answer via the knowledge base first.
    If the customer needs human help, create a ticket. Always cite the source.""",
    max_steps=8,
)
```

The agent resolves 60% of tickets without human intervention. The remaining 40% are auto-triaged with full context.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Skipping cross-encoder reranking

```python
# ❌ Dense retriever top-5 straight to LLM
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(top_k=5))
# No reranker → 30-40% noise in context

# ✅ Retrieve more, rerank to fewer
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(top_k=30))
pipeline.add_component("ranker", SentenceTransformersSimilarityRanker(top_k=5))
```

### 10.2 Antipattern 2: Not evaluating faithfulness

```python
# ❌ Deploy without measuring hallucination rate
pipeline.deploy()  # hallucination could be 20%!

# ✅ Always measure faithfulness before deployment
results = eval_pipeline.run(goldens=golden_dataset)
assert results["faithfulness"] > 0.90, "Faithfulness too low"
```

### 10.3 Antipattern 3: Using a single embedder for both indexing and retrieval

```python
# ❌ Embedding mismatch if you change the model
indexed_with = "all-MiniLM-L6-v2"
retrieved_with = "bge-small-en-v1.5"  # different dimensions!
# Documents and queries are in different vector spaces

# ✅ Pin one embedder, version it
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # constant
```

### 10.4 Antipattern 4: Hard-coding the LLM in production

```python
# ❌ Locked to OpenAI; no fallback
OpenAIGenerator(model="gpt-4o-mini")

# ✅ Use LiteLLM for multi-provider routing
from haystack_integrations.components.generators.litellm import LiteLLMGenerator

LiteLLMGenerator(
    model="gpt-4o-mini",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    fallbacks=["anthropic/claude-3-5-sonnet", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"],
)
```

### 10.5 Antipattern 5: Skipping pipeline validation

```python
# ❌ Deploy a pipeline with type mismatches (silent runtime errors)
pipeline.connect("text_embedder.text", "retriever.query_embedding")  # wrong type!

# ✅ Always validate before deployment
pipeline.validate()  # raises on type mismatch
```

---

## 🎯 Key Takeaways

- Hybrid retrieval (BM25 + dense + cross-encoder reranking) is the canonical production pattern.
- Reciprocal Rank Fusion with weights [0.6, 0.4] is the default for dense-heavy workloads.
- Agentic RAG with Haystack's `Agent` enables multi-step reasoning over multiple retrieval tools.
- Cross-framework integration: use LangChain/LlamaIndex retrievers inside Haystack pipelines.
- Offline evaluation with RAGAS-style metrics: faithfulness > 0.90 is the production target.
- Deploy via deepset Cloud (managed) or Kubernetes (self-hosted).
- OpenTelemetry + LangFuse for unified observability across all components.
- Avoid skipping reranking, no faithfulness measurement, embedder mismatches, locked providers, and missing type validation.

## References

- Haystack 2.x docs — [docs.haystack.deepset.ai](https://docs.haystack.deepset.ai/docs/intro)
- Haystack Agent docs — [docs.haystack.deepset.ai/docs/agents](https://docs.haystack.deepset.ai/docs/agents)
- Haystack Evaluators — [docs.haystack.deepset.ai/docs/evaluation](https://docs.haystack.deepset.ai/docs/evaluation)
- deepset Cloud — [cloud.deepset.ai](https://cloud.deepset.ai)
- RAGAS — [docs.ragas.io](https://docs.ragas.io)
- [[06 - Large Language Models/12 - Production RAG|Production RAG]] — foundational RAG patterns
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM and Advanced RAG]] — vLLM as backend
- [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference\|ColBERT, SGLang]] — token-level retrieval
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider routing
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured RAG responses
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/01 - Haystack Fundamentals - Pipelines, Components, Retrievers|Note 01 — Haystack Fundamentals]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/03 - txtai Fundamentals - Semantic Search + RAG|Note 03 — txtai]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — observability
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — Phoenix spans
- [[09 - MLOps y Produccion/20 - RAG Evaluation Deep Dive|RAG Evaluation Deep Dive]] — RAGAS deep dive
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant, Milvus, Pinecone