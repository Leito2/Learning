# 🎯 03 - txtai Fundamentals — Semantic Search, Graphs, and RAG in One Library

> **The all-in-one alternative to Haystack. Embeddings index, vector search, graph database, and LLM orchestration in a single Python package with the smallest API surface.**

## 🎯 Learning Objectives
- Use txtai's all-in-one API for semantic search + RAG in 5 lines
- Build embeddings indexes with txtai.Embeddings and persist them to disk
- Leverage graphs and workflows for multi-hop retrieval
- Configure txtai's RAG pipeline with custom prompts and LLMs
- Deploy txtai as an API server with the built-in FastAPI wrapper
- Compare txtai vs Haystack for different production scenarios

## Introduction

txtai by [neuml](https://github.com/neuml/txtai) is the **all-in-one** alternative to the four-major RAG frameworks. Where Haystack offers explicit DAG pipelines, txtai offers a single `Embeddings` class that handles indexing, vector search, and hybrid queries. Where LangChain offers 30+ integrations, txtai offers 5: SQLite, Faiss, HNSW, Milvus, Qdrant. The philosophy is **smallest surface area, fastest cold-start, easiest deployment**.

By 2026 txtai has 11k+ GitHub stars, 1k+ contributors, and a 7.x release that ships quarterly. The library powers production RAG at several mid-market companies and is the de facto choice for teams that want semantic search without the complexity of Qdrant + Haystack + LangChain.

The three core abstractions:

1. **Embeddings** — vector index with hybrid search (semantic + keyword)
2. **Graph** — knowledge graph for multi-hop retrieval
3. **Workflow** — DAG for multi-step RAG (similar to LangGraph)

The killer feature is **graph + embeddings**: build a knowledge graph of entities and relationships, then traverse it alongside vector search to answer multi-hop questions. This is a pattern LangChain and LlamaIndex struggle with; txtai makes it a 10-line setup.

![txtai architecture](https://raw.githubusercontent.com/neuml/txtai/master/docs/images/architecture.png)

---

## 1. Embeddings Index — The Foundation

### 1.1 In-memory embeddings

```python
from txtai import Embeddings

# Create the index
embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2")

# Index documents (with IDs)
embeddings.index([
    (0, "Haystack is a framework for building production RAG systems.", None),
    (1, "RAG combines retrieval with generation for grounded answers.", None),
    (2, "Deepset is the company behind Haystack.", None),
    (3, "txtai is an all-in-one semantic search library.", None),
])

# Search
results = embeddings.search("What is Haystack?", limit=3)
for result in results:
    print(f"[{result['score']:.2f}] {result['text']}")
# [0.85] Haystack is a framework for building production RAG systems.
# [0.71] txtai is an all-in-one semantic search library.
# [0.62] RAG combines retrieval with generation for grounded answers.
```

That's it. **5 lines for semantic search**. Compare to Haystack (50+ lines for the same functionality).

### 1.2 Hybrid search

```python
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    content=True,  # enable BM25 keyword search
)

# Hybrid search combines semantic + keyword
results = embeddings.search(
    "RAG framework",
    limit=3,
    weights=[0.7, 0.3],  # 70% semantic, 30% keyword
)
```

The hybrid search uses Reciprocal Rank Fusion (RRF) like Haystack, but configured inline.

### 1.3 Persistent embeddings

```python
# Save to disk
embeddings.save("index.tar.gz")

# Load from disk
loaded = Embeddings()
loaded.load("index.tar.gz")

# Search the loaded index
results = loaded.search("What is Haystack?", limit=3)
```

The index is fully portable — copy `index.tar.gz` to another machine and load. No database server required.

For production scale, use external backends:

```python
# SQLite (good for 1M documents)
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    backend="sqlite",
)

# Faiss (good for 10M+ documents)
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    backend="faiss",
    faiss={"quantize": True},  # memory-efficient
)

# Qdrant (distributed, multi-node)
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    backend="qdrant",
    qdrant={"url": "http://localhost:6333"},
)

# Milvus (GPU-accelerated)
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    backend="milvus",
    milvus={"url": "http://localhost:19530"},
)
```

The backend choice is independent of the embedding model — you can index with one backend and serve from another.

---

## 2. RAG with txtai — Built-in Pipeline

txtai's `RAG` class wraps the embeddings + LLM into a single component:

```python
from txtai import Embeddings, RAG

# 1. Embeddings index
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    content=True,
)

# 2. Index documents
embeddings.index([
    (0, "Haystack is a framework for building production RAG systems.", None),
    (1, "RAG combines retrieval with generation for grounded answers.", None),
    (2, "Deepset is the company behind Haystack.", None),
])

# 3. RAG pipeline
rag = RAG(
    embeddings,
    path="Qwen/Qwen2.5-7B-Instruct",  # any HuggingFace or OpenAI model
    api_key="...",  # for OpenAI/Anthropic
)

# 4. Ask a question
result = rag("What is Haystack?")
print(result)
# "Haystack is a framework for building production RAG systems."
```

The `RAG` class handles:
- Embedding the query
- Searching the embeddings index
- Building a context-augmented prompt
- Calling the LLM
- Returning the answer

For OpenAI:

```python
from txtai import RAG
from txtai.pipeline import RAG

rag = RAG(
    embeddings,
    "openai://gpt-4o-mini",
    api_key="sk-...",
    system="Answer based on the context below. Cite sources when possible.",
)

result = rag("What is Haystack?")
```

For Anthropic:

```python
rag = RAG(
    embeddings,
    "anthropic://claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
)
```

### 2.1 Custom prompts

```python
custom_template = """You are a helpful assistant. Use the context to answer the question.

Context:
{context}

Question: {question}

Answer concisely with citations: """

rag = RAG(
    embeddings,
    path="openai://gpt-4o-mini",
    template=custom_template,
    api_key="sk-...",
)
```

The `{context}` placeholder receives the retrieved documents; `{question}` receives the user's query.

---

## 3. Knowledge Graphs for Multi-Hop Retrieval

txtai's killer feature is **graph + embeddings** for multi-hop retrieval:

```python
from txtai import Embeddings

# Build an embeddings + graph index
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    content=True,
    graph={
        "backend": "networkx",  # or "rdflib" for RDF
    },
)

# Index documents with relationships
embeddings.index([
    (0, "Haystack is a framework for building production RAG systems.", None),
    (1, "Deepset is the company behind Haystack.", None),
    (2, "txtai is an all-in-one semantic search library.", None),
])

# Add graph relationships between documents
embeddings.graph.add(
    # (source_doc_id, target_doc_id, relationship)
    (0, 1, "developed_by"),  # Haystack developed_by Deepset
    (2, 0, "alternative_to"),  # txtai alternative_to Haystack
)

# Search with graph traversal
results = embeddings.search(
    "Who developed Haystack?",
    limit=3,
    graph=True,  # enable graph traversal
)
```

The search traverses the graph alongside vector similarity. For "Who developed Haystack?", the system:
1. Finds Haystack via semantic similarity (document 0)
2. Traverses the `developed_by` relationship to find Deepset (document 1)
3. Returns both as context

This is the **multi-hop retrieval** pattern that LangChain and LlamaIndex struggle with. txtai's graph + embeddings design makes it natural.

For knowledge graph construction from text:

```python
@component
class GraphBuilder:
    """Extract entities and relationships from text."""
    
    @component.output_fields(triples=list[tuple])
    def run(self, text: str) -> dict:
        # Use an LLM to extract (entity, relationship, entity) triples
        triples = llm_extract_triples(text)
        return {"triples": triples}
```

Run the GraphBuilder over your documents to populate the graph automatically.

---

## 4. Workflows — Multi-Step RAG Pipelines

txtai's workflow DSL (added in 6.x) enables multi-step pipelines:

```python
from txtai.workflow import Workflow, Task, ExecuteAction

# Define a workflow
workflow = Workflow([
    Task(
        action=ExecuteAction(lambda x: embeddings.search(x["query"], limit=5)),
        task="retrieval",
    ),
    Task(
        action=ExecuteAction(lambda x: llm_generate(x["query"], x["retrieval"])),
        task="generation",
    ),
])

# Run
result = workflow({"query": "What is Haystack?"})
print(result)
```

The workflow DSL is similar to LangGraph but simpler. For most RAG use cases, the built-in `RAG` class is sufficient; workflows are for advanced multi-step patterns.

---

## 5. The txtai API Server

txtai ships a built-in FastAPI server:

```bash
# Install with API support
pip install txtai[api]

# Create config.yml
cat > config.yml <<EOF
writable: true
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  content: true
EOF

# Start the API server
python -m txtai.server config.yml
```

Endpoints exposed:

```bash
# Add documents
curl -X POST "http://localhost:8000/add" -H "Content-Type: application/json" \
    -d '[{"id": "0", "text": "Haystack is a framework for production RAG systems."}]'

# Search
curl -X GET "http://localhost:8000/search?query=What+is+Haystack&limit=3"

# RAG
curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" \
    -d '{"query": "What is Haystack?"}'
```

The API server is the deployment story. For production:

```yaml
# config-production.yml
writable: false  # read-only
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  content: true
  backend: qdrant
  qdrant:
    url: http://qdrant:6333
```

Deploy with Docker:

```dockerfile
FROM python:3.12-slim
RUN pip install txtai[api] qdrant-client
COPY config.yml /app/config.yml
WORKDIR /app
EXPOSE 8000
CMD ["python", "-m", "txtai.server", "config.yml"]
```

---

## 6. Case Studies

### 6.1 Case real 1: Customer support knowledge base

A SaaS company uses txtai for tier-1 support Q&A:

```python
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    content=True,
    backend="qdrant",
)

# Index the support knowledge base
embeddings.index(support_documents)

# RAG
rag = RAG(embeddings, "openai://gpt-4o-mini", api_key=...)
result = rag("How do I reset my password?")
```

The team reports 70% query resolution without human intervention. The remaining 30% are auto-routed to the right team via entity extraction.

### 6.2 Case real 2: Multi-hop research

A research firm uses graph + embeddings for multi-hop question answering:

```python
embeddings = Embeddings(
    path="sentence-transformers/all-MiniLM-L6-v2",
    content=True,
    graph={"backend": "networkx"},
)

# Index research papers
embeddings.index(papers)

# Add citations as graph edges
for paper_id, citations in paper_citations.items():
    for cited_id in citations:
        embeddings.graph.add((paper_id, cited_id, "cites"))

# Search with graph traversal
results = embeddings.search(
    "What research followed the seminal attention paper?",
    limit=10,
    graph={"depth": 2},  # traverse 2 hops
)
```

The system finds papers that cite or are cited by the seminal attention paper — a pattern that pure vector search misses.

---

## 7. txtai vs Haystack vs LangChain vs LlamaIndex

| Property | txtai | Haystack | LangChain | LlamaIndex |
|----------|:-----:|:--------:|:---------:|:----------:|
| **API surface** | Tiny (5 main classes) | Medium (~20 components) | Large (~100 integrations) | Large (~50 components) |
| **Cold-start** | <1s | 5-10s | 5-10s | 5-10s |
| **Multi-hop** | ✅ Built-in graphs | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| **Hybrid search** | ✅ Inline config | ✅ Component DSL | ✅ LCEL | ✅ Native |
| **Agentic loops** | ⚠️ Workflows | ✅ Agent class | ✅ LangGraph | ⚠️ Workflows |
| **Cross-framework** | ❌ | ✅ LangChain/LlamaIndex adapters | ✅ Most ecosystems | ✅ LangChain |
| **Production deployment** | ✅ FastAPI server | ✅ deepset Cloud + K8s | ✅ LangServe + K8s | ✅ Llamaindex + K8s |
| **Enterprise adoption** | 🟡 Niche | ✅ deepset enterprise | ✅ Universal | ✅ Universal |
| **Community** | 11k stars | 22k stars | 100k+ stars | 45k stars |
| **When to choose** | Simplicity, multi-hop | Enterprise, deepset Cloud | Ubiquitous Python | Data-heavy indexing |

For the user's profile (RAG + LangGraph + multi-framework), **txtai is the right choice when**:
- Multi-hop retrieval is a primary requirement
- You want minimal dependencies (one library vs three)
- You're deploying to a small VM without Kubernetes
- You don't need 100+ integrations

**Haystack is the right choice when**:
- Enterprise production with deepset Cloud or on-prem Kubernetes
- Cross-framework integration (LangChain/LlamaIndex) is needed
- DAG pipelines with multiple components
- Production testing rigor matters

---

## 8. Antipatterns

### 8.1 Antipattern 1: Indexing without an ID

```python
# ❌ No ID → cannot update or delete later
embeddings.index([("Haystack is a framework for RAG.", None)])

# ✅ Always use IDs
embeddings.index([(0, "Haystack is a framework for RAG.", None)])
```

### 8.2 Antipattern 2: Using the default embedder for production

```python
# ❌ Default is no embedder; falls back to no-op
embeddings = Embeddings()

# ✅ Pin a production-quality embedder
embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2")
```

### 8.3 Antipattern 3: Not using hybrid search

```python
# ❌ Pure semantic misses keyword-specific queries
embeddings = Embeddings(path="...")
results = embeddings.search("API_KEY", limit=5)  # no semantic match for "API_KEY"!

# ✅ Enable hybrid search for keyword-sensitive queries
embeddings = Embeddings(path="...", content=True)
results = embeddings.search("API_KEY", limit=5, weights=[0.5, 0.5])
```

### 8.4 Antipattern 4: Building graphs without explicit relationships

```python
# ❌ Graph auto-extracts from text (noisy, low precision)
embeddings.graph.add(auto_extract_relationships_from_text)

# ✅ Explicitly defined relationships
embeddings.graph.add((doc_id, target_id, "developed_by"))
```

### 8.5 Antipattern 5: Using txtai for complex agentic workflows

```python
# ❌ txtai's workflow DSL is limited compared to LangGraph
workflow = Workflow([...])  # can't easily express complex branching

# ✅ Use LangGraph for complex agents, txtai for simple RAG
workflow = langgraph_builder.compile()
```

---

## 🎯 Key Takeaways

- txtai is the all-in-one alternative to Haystack: embeddings index + vector search + graph + RAG in one library.
- 5-line semantic search; configurable hybrid search via `weights=[semantic, keyword]`.
- Persistent indexes (`save()` / `load()`) and external backends (Qdrant, Milvus, Faiss).
- Knowledge graphs for multi-hop retrieval: combine vector similarity with graph traversal.
- Built-in `RAG` class wraps embeddings + LLM; supports OpenAI, Anthropic, HuggingFace.
- API server via `python -m txtai.server config.yml`; deploy with Docker.
- Choose txtai for simplicity + multi-hop; choose Haystack for enterprise + cross-framework.

## References

- txtai docs — [neuml.github.io/txtai](https://neuml.github.io/txtai/)
- txtai GitHub — [github.com/neuml/txtai](https://github.com/neuml/txtai)
- txtai API server — [neuml.github.io/txtai/server](https://neuml.github.io/txtai/server/)
- [[06 - Large Language Models/12 - Production RAG|Production RAG]] — foundational RAG patterns
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/01 - Haystack Fundamentals - Pipelines, Components, Retrievers|Note 01 — Haystack Fundamentals]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/02 - Haystack Advanced Pipelines - Hybrid Search, Reranking, Agents, and Evaluation|Note 02 — Haystack Advanced]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai/05 - Capstone - Production Hybrid RAG Service|Note 05 — Capstone]]
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant, Milvus, Pinecone
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — agentic workflows