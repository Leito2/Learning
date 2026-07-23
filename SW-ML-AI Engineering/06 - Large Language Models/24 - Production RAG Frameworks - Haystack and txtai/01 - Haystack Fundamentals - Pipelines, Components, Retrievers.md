# 🎯 01 - Haystack Fundamentals — Pipelines, Components, Retrievers

> **The enterprise RAG framework. Typed DAG pipelines, component-based architecture, production hardening out of the box. deepset's bet on the explicit-pipeline philosophy.**

## 🎯 Learning Objectives
- Understand the Haystack 2.x architecture: Pipelines, Components, Document Stores
- Build a basic RAG pipeline with InMemoryDocumentStore, EmbeddingRetriever, and Generator
- Use the Component DSL: `@component`, input types, output types
- Configure multiple Document Stores: InMemory, Elasticsearch, OpenSearch, Qdrant, Milvus, pgvector
- Implement custom components that wrap any callable
- Debug pipelines with visualization and tracing

## Introduction

Haystack 2.x (released 2024) is a **ground-up rewrite** of the original Haystack by deepset. The new architecture replaces the legacy `Pipeline` / `JoinablePipeline` / `RayPipeline` zoo with a single typed `Pipeline` class and a Component DSL. Every node in the pipeline declares its input types, output types, and validation rules — the runtime checks them at construction time and refuses to connect incompatible components.

The framework is built around three core abstractions:

1. **Components** — units of work that take typed inputs and produce typed outputs. Examples: `Embedder`, `Retriever`, `Ranker`, `Generator`, `PromptBuilder`.
2. **Pipelines** — explicit DAGs that wire components together. Pipelines serialize to YAML, run on local threads, Ray, or remote executors.
3. **Document Stores** — typed storage backends with consistent query interfaces. Backends include InMemory, Elasticsearch, OpenSearch, Qdrant, Milvus, pgvector, Weaviate.

For an enterprise team, the value proposition is **explicit, testable, serializable, deployable**. A Haystack pipeline is a Python object that can be saved to YAML, versioned in Git, tested with `pytest`, deployed to deepset Cloud, and monitored in production. Compare to LangChain's chain-of-functions approach, which is implicit and harder to serialize.

![Haystack 2.x architecture](https://haystack.deepset.ai/assets/img/architecture.png)

---

## 1. The Component DSL

A Component is a Python class with the `@component` decorator:

```python
from haystack import component
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    documents: list[dict]
    query: str


@component
class CustomRetriever:
    """Retrieves documents from a custom source."""
    
    @component.output_fields(documents=list[dict], query=str)
    def run(self, query: str, top_k: int = 5) -> RetrievalResult:
        # Custom logic
        docs = self.search_engine.search(query, k=top_k)
        return {"documents": docs, "query": query}
```

The decorator enforces:
1. **Input/output types** declared via `@component.output_fields(...)` and `@component.input_fields(...)`.
2. **Type checking at runtime** — connecting a component that emits `documents` to one that expects `passages` raises an error.
3. **Serialization** — the pipeline YAML includes the component config (init parameters), not its code.

For backward compatibility with v1.x, the `@component` accepts `init_parameters` that map to dataclass fields:

```python
@component
class MyComponent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
    
    @component.output_fields(result=str)
    def run(self, text: str) -> dict:
        return {"result": f"Processed: {text}"}
```

The `model_name` is serialized to YAML and can be overridden at load time.

---

## 2. A Basic RAG Pipeline

The canonical Hello-World of Haystack 2.x:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document


# 1. Setup document store
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# 2. Embedding model
embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up()

# 3. Index some documents
documents = [
    Document(content="Haystack is a framework for building production RAG systems."),
    Document(content="RAG combines retrieval with generation for grounded answers."),
    Document(content="Deepset is the company behind Haystack."),
]
documents_with_embeddings = embedder.run(documents=documents)["documents"]
document_store.write_documents(documents_with_embeddings)


# 4. Build the RAG pipeline
pipeline = Pipeline()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=3))
pipeline.add_component("prompt_builder", PromptBuilder(template="""Answer the question based on the context.

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer: """))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini", api_key=Secret.from_env_var("OPENAI_API_KEY")))


# 5. Connect the components
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")
pipeline.connect("llm.replies", "output_1")  # output_1 is implicit


# 6. Run
result = pipeline.run({
    "text_embedder": {"text": "What is Haystack?"},
    "prompt_builder": {"query": "What is Haystack?"},
})
print(result["llm"]["replies"][0])
# "Haystack is a framework for building production RAG systems."
```

The pipeline:
1. Embeds the query with `text_embedder`
2. Retrieves top-3 documents from the document store via cosine similarity
3. Builds a prompt with the retrieved documents as context
4. Calls the LLM (gpt-4o-mini) to generate the answer

This is the **minimal end-to-end RAG pipeline** in Haystack 2.x.

---

## 3. Document Stores — The Backend Abstraction

Haystack's `DocumentStore` interface supports 8+ backends with a consistent query API:

| Backend | License | Persistent | Best for |
|---------|---------|:----------:|----------|
| **InMemoryDocumentStore** | Apache 2.0 | ❌ | Prototyping, dev, small datasets |
| **ElasticsearchDocumentStore** | Apache 2.0 | ✅ | Hybrid search, enterprise production |
| **OpenSearchDocumentStore** | Apache 2.0 | ✅ | AWS-native hybrid search |
| **QdrantDocumentStore** | Apache 2.0 | ✅ | High-performance vector search |
| **MilvusDocumentStore** | Apache 2.0 | ✅ | Distributed GPU vector search |
| **pgvectorDocumentStore** | PostgreSQL | ✅ | PostgreSQL-centric stacks |
| **WeaviateDocumentStore** | BSD-3 | ✅ | Hybrid search with built-in reranking |
| **PineconeDocumentStore** | Proprietary | ✅ | Serverless vector search |

For a production deployment, pick based on:

- **Existing infrastructure**: PostgreSQL → pgvector, Elasticsearch → ElasticsearchDocumentStore
- **Hybrid search needs**: Elasticsearch or OpenSearch (built-in BM25 + vector)
- **Performance**: Qdrant (fastest pure vector), Milvus (distributed)
- **Cloud-managed**: Pinecone (no ops)

Configuration example for Qdrant:

```python
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    collection_name="documents",
    embedding_dim=384,  # must match the embedder's output
    similarity="cosine",
    recreate_collection=False,
)
```

The integration packages (`haystack-integrations/qdrant`, `haystack-integrations/pinecone`, etc.) are maintained by deepset + the backend vendors.

---

## 4. Retrievers, Rankers, and Generators

### 4.1 Retrievers

A `Retriever` is a component that takes a query and returns documents:

```python
from haystack.components.retrievers import (
    InMemoryEmbeddingRetriever,
    InMemoryBM25Retriever,
    InMemoryFilterRetriever,
)

# Dense retriever (semantic similarity)
dense_retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=10)

# Sparse retriever (BM25 keyword)
sparse_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=10)

# Filter retriever (metadata only)
filter_retriever = InMemoryFilterRetriever(document_store=document_store)
```

Haystack ships retrievers for every document store: `QdrantEmbeddingRetriever`, `ElasticsearchBM25Retriever`, `PgvectorEmbeddingRetriever`, etc. The pattern is identical: take a query, return `list[Document]` with relevance scores.

### 4.2 Rankers

A `Ranker` re-scores documents after retrieval:

```python
from haystack.components.rankers import (
    SentenceTransformersSimilarityRanker,
    TransformersSimilarityRanker,
    CohereRanker,
)

# Cross-encoder reranker
cross_encoder_ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=3,
)

# Cohere rerank (cloud API)
cohere_ranker = CohereRanker(api_key=Secret.from_env_var("COHERE_API_KEY"), top_k=3)
```

Rankers are typically applied after retrieval to refine the top-k. The pattern is the canonical **dense retrieve → rerank → generate** from [[06 - Large Language Models/12 - Production RAG]].

### 4.3 Generators

A `Generator` is the LLM component:

```python
from haystack.components.generators import (
    OpenAIGenerator,
    AnthropicGenerator,
    HuggingFaceLocalGenerator,
)

# OpenAI
openai_gen = OpenAIGenerator(
    model="gpt-4o-mini",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
)

# Anthropic
anthropic_gen = AnthropicGenerator(
    model="claude-3-5-sonnet-20241022",
    api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
)

# Local model (HuggingFace)
local_gen = HuggingFaceLocalGenerator(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)
```

All three expose `replies: list[str]` as output. The pipeline wires them identically.

---

## 5. Hybrid Retrieval — BM25 + Dense

The production pattern combines keyword and semantic search:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


# 1. Setup document store with BM25 (TF-IDF) index
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine", use_bm25=True)
# Index documents (omitted for brevity)

# 2. Build hybrid pipeline
pipeline = Pipeline()

# Two retrievers: dense and sparse
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipeline.add_component("dense_retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=10))
pipeline.add_component("sparse_retriever", InMemoryBM25Retriever(document_store=document_store, top_k=10))

# Joiner merges results
pipeline.add_component("joiner", DocumentJoiner(join_mode="concatenate"))

# Reranker
pipeline.add_component("ranker", SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3))

# LLM
pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))

# Connect
pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
pipeline.connect("sparse_retriever.documents", "joiner.documents")
pipeline.connect("dense_retriever.documents", "joiner.documents")
pipeline.connect("joiner.documents", "ranker.documents")
pipeline.connect("ranker.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.prompt")

# Run
result = pipeline.run({
    "text_embedder": {"text": "What is Haystack?"},
    "sparse_retriever": {"query": "What is Haystack?"},
    "prompt_builder": {"query": "What is Haystack?"},
})
```

The pipeline:
1. Queries both BM25 (keyword) and dense (semantic) retrievers in parallel
2. Joins the results (with deduplication)
3. Reranks via cross-encoder
4. Sends top-3 to the LLM

This is the **canonical production retrieval pattern**: hybrid + rerank.

---

## 6. Custom Components — Wrap Anything

When the built-in components don't fit, write a custom one:

```python
from haystack import component
from haystack.dataclasses import Document


@component
class PIIRedactor:
    """Redacts PII from documents before retrieval."""
    
    @component.output_fields(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        redacted = []
        for doc in documents:
            text = doc.content
            text = redact_ssn(text)
            text = redact_email(text)
            text = redact_phone(text)
            redacted.append(Document(content=text, meta=doc.meta))
        return {"documents": redacted}


# Use in pipeline
pipeline.add_component("pii_redactor", PIIRedactor())
pipeline.connect("retriever.documents", "pii_redactor.documents")
pipeline.connect("pii_redactor.documents", "prompt_builder.documents")
```

Custom components are reusable across pipelines and serializable to YAML. They integrate with Haystack's tracing and debugging.

---

## 7. Pipeline Visualization and Debugging

Haystack 2.x ships with built-in pipeline visualization:

```python
pipeline.draw(path="pipeline.png")
```

Generates a Mermaid or Graphviz diagram showing components and connections.

For runtime debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.DEBUG)

result = pipeline.run({"text_embedder": {"text": "..."}})
# Logs every component invocation with input/output types and timings
```

For tracing, install Haystack's OpenTelemetry integration:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())
# Every component becomes a span; pre-built Phoenix/LangFuse integration works
```

---

## 8. Pipeline Serialization and Versioning

Save a pipeline to YAML for version control:

```python
pipeline_yaml = pipeline.to_dict()
import yaml
with open("pipeline.yaml", "w") as f:
    yaml.dump(pipeline_yaml, f)
```

Load it back:

```python
with open("pipeline.yaml") as f:
    pipeline_dict = yaml.safe_load(f)
pipeline = Pipeline.from_dict(pipeline_dict)
```

This enables GitOps workflows: pipelines are versioned, reviewed, and deployed via standard Git workflows. Compare to LangChain's `with_config()` patterns, which are Python objects not easily serializable.

---

## 9. Case Studies

### 9.1 Case real 1: deepset Cloud deployment

deepset Cloud is the managed Haystack SaaS. Customers deploy pipelines via web UI; the platform handles Kubernetes, monitoring, and scaling. A typical setup takes ~1 hour vs 1-2 days for self-hosted Kubernetes.

### 9.2 Case real 2: Financial services compliance

A bank uses Haystack for compliance Q&A:

```python
@component
class ComplianceFilter:
    """Filters documents by compliance level."""
    
    @component.output_fields(documents=list[Document])
    def run(self, documents: list[Document], user_clearance: str) -> dict:
        allowed = []
        for doc in documents:
            if doc.meta.get("clearance_level", "public") <= user_clearance:
                allowed.append(doc)
        return {"documents": allowed}
```

The custom component enforces access control at the pipeline level. Every retrieval respects the user's clearance.

### 9.3 Case real 3: Healthcare clinical decision support

A hospital uses Haystack for clinical Q&A:

```python
@component
class CitationEnforcer:
    """Ensures every LLM answer cites its sources."""
    
    @component.output_fields(answer=str)
    def run(self, llm_replies: list[str], documents: list[Document]) -> dict:
        answer = llm_replies[0]
        citations = [doc.meta["source"] for doc in documents]
        # Append citations if not already in the answer
        if "[" not in answer:
            answer += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        return {"answer": answer}
```

The custom component guarantees clinical provenance — every answer includes its sources.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Using InMemoryDocumentStore in production

```python
# ❌ Data lost on restart; not scalable
document_store = InMemoryDocumentStore()

# ✅ Use a persistent backend
document_store = QdrantDocumentStore(url="http://qdrant:6333", collection_name="docs")
```

### 10.2 Antipattern 2: Returning top_k=100 documents to the LLM

```python
# ❌ Context window overflow, slow, expensive
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=100)

# ✅ Retrieve more, rerank to fewer, send few
dense_retriever = InMemoryEmbeddingRetriever(top_k=30)
ranker = SentenceTransformersSimilarityRanker(top_k=3)
# Pipeline: dense retrieve → rerank → generate
```

### 10.3 Antipattern 3: Storing PII in document content

```python
# ❌ PII persists in document store, exposed via traces
Document(content=f"Patient SSN: {ssn}, name: {name}")

# ✅ Hash or redact before storage
Document(content=redact_pii(raw_text))
```

### 10.4 Antipattern 4: Hard-coding model names in pipelines

```python
# ❌ Model change requires code change
OpenAIGenerator(model="gpt-4o-mini")

# ✅ Parameterize
OpenAIGenerator(model=os.getenv("HAYSTACK_GENERATOR_MODEL", "gpt-4o-mini"))
```

### 10.5 Antipattern 5: Not validating pipeline types

```python
# ❌ Connecting incompatible components (silent runtime error)
pipeline.connect("text_embedder.text", "retriever.query_embedding")  # wrong type

# ✅ Always run pipeline validation before deployment
pipeline.validate()
```

---

## 🎯 Key Takeaways

- Haystack 2.x is the enterprise RAG framework: explicit typed DAG pipelines, deepset Cloud, production testing.
- The `@component` DSL enforces type-checked inputs/outputs and serializes to YAML for GitOps.
- Document Stores: InMemory (dev) → Qdrant/Milvus/Pinecone (production) → Elasticsearch (hybrid).
- Hybrid retrieval (BM25 + dense + reranking) is the canonical production pattern.
- Custom components wrap any callable — PII redaction, citation enforcement, compliance filters.
- Pipeline visualization, OpenTelemetry tracing, and YAML serialization enable enterprise workflows.
- Avoid InMemoryDocumentStore in production, top_k=100 to LLM, PII in documents, hard-coded models, and skipping type validation.

## References

- Haystack docs — [haystack.deepset.ai](https://haystack.deepset.ai)
- Haystack GitHub — [github.com/deepset-ai/haystack](https://github.com/deepset-ai/haystack)
- deepset Cloud — [cloud.deepset.ai](https://cloud.deepset.ai)
- Haystack integrations — [haystack.deepset.ai/integrations](https://haystack.deepset.ai/integrations)
- [[06 - Large Language Models/12 - Production RAG|Production RAG]] — foundational RAG patterns
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM and Advanced RAG]] — vLLM as LLM backend
- [[06 - Large Language Models/17 - ColBERT, SGLang and Next-Gen Inference\|ColBERT, SGLang]] — token-level retrieval
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider routing
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Instructor and Structured Generation]] — structured RAG responses
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/02 - Haystack Advanced Pipelines|Note 02 — Haystack Advanced]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/03 - txtai Fundamentals - Semantic Search + RAG|Note 03 — txtai]]
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant, Milvus, Pinecone
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — RAG observability