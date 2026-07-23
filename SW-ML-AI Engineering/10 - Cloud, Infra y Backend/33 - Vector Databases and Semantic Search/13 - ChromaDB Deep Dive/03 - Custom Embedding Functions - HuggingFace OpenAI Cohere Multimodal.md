# 🧠 Custom Embedding Functions — HuggingFace, OpenAI, Cohere, Multi-Modal

Chroma's default embedding model is `all-MiniLM-L6-v2` — a 384-dimensional, 80MB model that runs on CPU. For a prototype it's fine; for a production RAG system it's usually wrong. Your OpenAI embeddings (1536-d) outperform it on retrieval; Cohere's `embed-english-v3.0` is faster and multilingual-capable; a fine-tuned embedding (S-BioBERT for medical, E5 for code) crushes both on the right domain. Chroma's `EmbeddingFunction` Protocol lets you swap the encoder without touching application code.

This note covers the four embedding patterns you'll actually use: OpenAI (managed, high-quality), HuggingFace sentence-transformers (free, local), Cohere (multilingual), and CLIP-style multimodal (text + image). We also build a custom `EmbeddingFunction` for any model that exposes a `encode()` method, including caching for cost control.

## 🎯 Learning Objectives

- Use the built-in embedding functions: `OpenAIEmbeddingFunction`, `HuggingFaceEmbeddingFunction`, `CohereEmbeddingFunction`, `SentenceTransformerEmbeddingFunction`.
- Build custom `EmbeddingFunction` subclasses for fine-tuned or domain-specific models.
- Implement CLIP-style multimodal embeddings for image-augmented RAG.
- Add embedding caching to reduce API costs.
- Pick the right model per use case.

## 1. Why Custom Embeddings Matter

| Model | Dimensions | Quality (BEIR avg) | Cost | Best for |
|-------|:---:|:---:|------|----------|
| `all-MiniLM-L6-v2` (default) | 384 | 0.41 | Free | Prototypes |
| `text-embedding-3-small` (OpenAI) | 1536 | 0.51 | $0.02/1M tokens | Production text |
| `text-embedding-3-large` (OpenAI) | 3072 | 0.55 | $0.13/1M tokens | High-stakes retrieval |
| `embed-english-v3.0` (Cohere) | 1024 | 0.52 | $0.10/1M tokens | English, fast |
| `embed-multilingual-v3.0` (Cohere) | 1024 | 0.50 | $0.10/1M tokens | Multi-language |
| `BAAI/bge-large-en-v1.5` (HuggingFace) | 1024 | 0.54 | Free (self-host) | Self-hosted production |
| `clip-vit-base-patch32` (OpenCLIP) | 512 | varies | Free | Image+text retrieval |

The default `all-MiniLM-L6-v2` is the **2nd worst** on the BEIR benchmark after random. For production RAG, swap it.

## 2. The Built-in Functions

### OpenAI

```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-...",   # or OPENAI_API_KEY env var
    model_name="text-embedding-3-small",
    dimensions=1536,     # optional: reduce for cost (OpenAI supports 256-1536)
)

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=openai_ef,
)

# Embedding is automatic on add/query
collection.add(documents=["hello"], ids=["doc-1"])
results = collection.query(query_texts=["greeting"])
```

The `dimensions` parameter lets you **shrink OpenAI's 1536-d to 256-d** to save storage and speed up search, at a small recall cost.

### HuggingFace (sentence-transformers)

```python
hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=None,    # public models need no key
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda",   # or "cpu", "mps" (Apple Silicon)
    normalize_embeddings=True,
)
```

Local sentence-transformers load on first use (~30s for 400MB model). Then ~100ns/token inference.

> 💡 **Tip:** The HF embedding function calls `sentence-transformers` internally. If you don't have it, install: `pip install sentence-transformers`.

### Cohere

```python
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key="...",   # or COHERE_API_KEY env
    model="embed-english-v3.0",
    input_type="search_document",   # or "search_query" depending on call
)
```

Cohere distinguishes `search_document` (for ingested docs) from `search_query` (for user queries) — using the wrong one degrades retrieval quality. The function handles this automatically based on whether you're embedding a `query` or a `document`.

### Google PaLM / Vertex

```python
palm_ef = embedding_functions.GooglePalmEmbeddingFunction(
    api_key="...",
    model="models/embedding-001",
)
```

### Other Built-ins

Chroma also ships `InstructorEmbeddingFunction`, `ONNXMiniLM_L6_V2`, and `Text2VecEmbeddingFunction`. List at https://docs.trychroma.com/embeddings.

## 3. Custom Embedding Function

```python
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import List
from chromadb.utils import embedding_functions

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """Custom encoder — wraps any model with an encode() method."""

    def __init__(self, model, batch_size: int = 32):
        self._model = model
        self._batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        # input is a list[str]; return a list[list[float]] of equal length
        embeddings: list[list[float]] = []
        for i in range(0, len(input), self._batch_size):
            batch = input[i:i + self._batch_size]
            vecs = self._model.encode(batch, convert_to_numpy=True)
            embeddings.extend(vecs.tolist())
        return embeddings
```

The Protocol is just `__call__(input: list[str]) -> list[list[float]]`. Wrap any model and you're done.

### Wrapping a Fine-Tuned Encoder

```python
from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
model = SentenceTransformer("./my-finetuned-encoder")

# Wrap with custom EF
def create_ef():
    class FineTunedEF(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return model.encode(list(input), convert_to_numpy=True).tolist()
    return FineTunedEF()

collection = client.get_or_create_collection(
    name="medical_docs",
    embedding_function=create_ef(),
)
```

The encoder is loaded once at collection creation and held in memory.

## 4. Caching for Cost Control

For OpenAI, re-embedding the same text wastes money. Add a cache:

```python
from diskcache import Cache

class CachedOpenAIEmbedding(EmbeddingFunction):
    """OpenAI EF with on-disk embedding cache."""

    def __init__(self, model_name: str = "text-embedding-3-small", cache_path: str = "./emb_cache"):
        from chromadb.utils.embedding_functions.openai import OpenAIEmbeddingFunction
        self._inner = OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model_name,
        )
        self._cache = Cache(cache_path)

    def __call__(self, input: Documents) -> Embeddings:
        results: Embeddings = []
        to_fetch: list[tuple[int, str]] = []
        for i, text in enumerate(input):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                to_fetch.append((i, text))

        if to_fetch:
            texts = [t for _, t in to_fetch]
            new_embeddings = self._inner(texts)
            for (i, text), emb in zip(to_fetch, new_embeddings):
                results[i] = emb
                self._cache[text] = emb
        return results
```

Re-ingesting the same corpus twice costs zero. On a typical weekly re-ingest job, this saves 90% of API spend.

## 5. Multi-Modal Embeddings (CLIP)

For image+text RAG, Chroma supports multimodal embeddings via the `image` field of `add()`:

```python
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# CLIP-style multi-modal encoder
clip_ef = OpenCLIPEmbeddingFunction(
    model_name="ViT-B-32",   # small, fast
    pretrained="laion2b_s34b_b79k",
    device="cpu",
)

collection = client.get_or_create_collection(
    name="multimodal",
    embedding_function=clip_ef,
)

# Add images (paths or PIL)
collection.add(
    ids=["img-1", "img-2"],
    images=["./cat.jpg", "./dog.jpg"],   # file paths
    metadatas=[{"type": "animal"}, {"type": "animal"}],
)

# Add text (CLIP shares embedding space with images!)
collection.add(
    ids=["text-1"],
    documents=["A photo of a cat sitting on a couch"],
    metadatas=[{"type": "caption"}],
)

# Query with text → retrieve images
results = collection.query(
    query_texts=["a cat relaxing"],
    n_results=2,
    include=["documents", "metadatas"],
)
```

> ⚠️ **Advertencia:** CLIP text and image encoders **must share embedding space**. Don't mix CLIP for images with OpenAI for text — they're incompatible. One model for the whole collection.

### Multi-Modal for Document RAG

A common pattern: PDF documents have text + figures. Encode the text body with one encoder, encode figures with CLIP, store both in the same collection, query with text → get matching figures + text passages.

```python
# Multi-modal with separate text and image embedding functions
# (Not directly supported in single collection; common approach is two collections)
text_collection = client.get_or_create_collection(
    name="text",
    embedding_function=text_ef,
)
image_collection = client.get_or_create_collection(
    name="images",
    embedding_function=clip_ef,
)

# On query, fan out
def multi_modal_query(query: str) -> dict:
    text_results = text_collection.query(query_texts=[query], n_results=5)
    image_results = image_collection.query(query_texts=[query], n_results=3)
    return {"text": text_results, "images": image_results}
```

## 6. Async Embedding

For high-throughput ingest pipelines, batch embeddings are 10-50× faster than per-document:

```python
# Bad: per-doc API call
for doc in documents:
    collection.add(documents=[doc], ids=[f"doc-{i}"])  # 1 API call per doc

# Good: batched add
collection.add(documents=documents, ids=ids)  # 1 batched call total
```

For OpenAI specifically, the `OpenAIEmbeddingFunction` accepts batches up to 2048 docs per call — exceed this and you get rate limits.

### Async Wrapper

```python
import asyncio
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

class AsyncBatchedEmbedding:
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 2048):
        self._ef = OpenAIEmbeddingFunction(model_name=model_name)
        self._batch = batch_size

    async def __call__(self, input: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        # Run sync EF in executor
        return await loop.run_in_executor(None, self._ef, input)
```

## 7. ❌/✅ Antipatterns

### ❌ Mixing models in the same collection

```python
# ❌ doc-1 is 1536-d (OpenAI), doc-2 is 384-d (HuggingFace)
collection.add(documents=["text"], ids=["doc-1"])  # using OpenAI EF
collection.add(documents=["text"], ids=["doc-2"], embeddings=[list[float]])  # 384-d
# Cosine similarity between different-dim vectors: ERROR
```

### ✅ One model per collection

```python
# ✅ All documents use the same encoder
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=openai_ef,
)
# All adds/queries use this encoder
```

### ❌ Re-embedding unchanged text on every ingest

```python
# ❌ Wasted $$ on already-ingested text
collection.add(documents=all_docs, ids=ids)
```

### ✅ Use `upsert` and embed only new docs

```python
# ✅ Embed only new documents
existing = collection.get(include=[])["ids"]
new_docs = [d for d, i in zip(docs, ids) if i not in existing]
new_embeddings = openai_ef(new_docs)
collection.add(documents=new_docs, embeddings=new_embeddings, ids=new_ids)
```

### ❌ Wrong Cohere input type

```python
# ❌ Using search_query for documents degrades quality
results = cohere_ef(["hello"])  # default is search_document
```

### ✅ Match input type to call

```python
# ✅ Cohere handles this internally based on whether you're calling add or query
collection.add(documents=[...])  # search_document internally
results = collection.query(query_texts=[...])  # search_query internally
```

### ❌ One embedding model for multi-language

```python
# ❌ all-MiniLM-L6-v2 has weak multilingual support
collection.add(documents=["Hola, ¿cómo estás?", "Hello, how are you?"], ...)
```

### ✅ Cohere multilingual or paraphrase-multilingual-MiniLM

```python
# ✅ Cohere embed-multilingual-v3.0
ef = embedding_functions.CohereEmbeddingFunction(model="embed-multilingual-v3.0")
```

## 8. Production Reality

**Caso real — Production RAG Project (portfolio):** Started with `all-MiniLM-L6-v2` (default). After prototyping, swapped to `text-embedding-3-small`. Retrieval recall (RAGAS) improved 12% on the eval set. Add the `CachedOpenAIEmbedding` wrapper next — re-ingestion during evaluation iterations dropped from $3 to $0.30 per cycle.

**Caso real — StayBot property search:** Property images + descriptions. Multi-modal collection with CLIP encoder. Users query "cozy mountain cabin" → returns matching images + descriptions in one ranked list. Without CLIP, image search would be a separate API.

## 📦 Compression Code

```python
# 📦 Compression: custom embedding functions in 80 lines

import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import List

# === Built-in ===
client = chromadb.PersistentClient(path="./chroma_demo")

# 1. OpenAI
oa_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)
coll_oa = client.get_or_create_collection(name="oa", embedding_function=oa_ef)
coll_oa.add(documents=["OpenAI embedding"], ids=["doc-1"])

# 2. HuggingFace
hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    model_name="sentence-transformers/all-mpnet-base-v2",
)
coll_hf = client.get_or_create_collection(name="hf", embedding_function=hf_ef)
coll_hf.add(documents=["HuggingFace embedding"], ids=["doc-1"])

# === Custom wrapper for a fine-tuned model ===
class SentenceTransformerEF(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input)).tolist()

# coll_custom = client.get_or_create_collection(
#     name="custom",
#     embedding_function=SentenceTransformerEF("./my-finetuned-model"),
# )

# === Multimodal (CLIP) ===
clip_ef = embedding_functions.OpenCLIPEmbeddingFunction()  # default = ViT-B-32
coll_clip = client.get_or_create_collection(name="multimodal", embedding_function=clip_ef)
coll_clip.add(
    ids=["img-1"],
    images=["./sample_image.jpg"],
)

# Query with text — returns matching images
results = coll_clip.query(query_texts=["a photo"], n_results=1)
print(results["ids"])
```

## 🎯 Key Takeaways

1. **The default `all-MiniLM-L6-v2` is a prototype convenience, not a production choice.** Swap for OpenAI, Cohere, or a fine-tuned encoder.
2. **`OpenAIEmbeddingFunction`, `HuggingFaceEmbeddingFunction`, `CohereEmbeddingFunction`** are the built-in three. Pick one per collection.
3. **`EmbeddingFunction[Documents]` is the Protocol** — `__call__(list[str]) -> list[list[float]]`. Wrap any model.
4. **Cohere has two input types** (`search_document` vs `search_query`); the built-in handles the distinction.
5. **Caching matters** — `CachedOpenAIEmbedding` saves 90%+ on repeated ingest.
6. **CLIP for image+text retrieval** — one model, shared embedding space.
7. **One model per collection** — mixing 1536-d and 384-d vectors causes silent corruption.

## References

- [[00 - Welcome to ChromaDB|Welcome]] — course map.
- [[01 - Chroma Fundamentals|Fundamentals]] — collections and embedding function slot.
- [[02 - Chroma Server Mode|Server Mode]] — embedding functions work the same on the server.
- [[../../06 - Large Language Models/12 - Production RAG/02 - Vector Databases for RAG - HNSW, IVF, PQ and Filtering.md|Vector Databases for RAG]] — embedding quality comparison.
- Chroma embeddings: https://docs.trychroma.com/embeddings