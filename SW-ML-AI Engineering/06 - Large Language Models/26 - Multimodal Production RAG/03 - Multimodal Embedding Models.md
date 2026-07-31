# 🎯 03 - Multimodal Embedding Models — CLIP, BGE-M3, and Jina CLIP for Image Search

> **The vector space that bridges text and images. CLIP-style embeddings enable image search with text queries, multimodal RAG, and cross-modal retrieval.**

## 🎯 Learning Objectives
- Understand the CLIP architecture (text-image contrastive learning)
- Use modern multimodal embedding models (CLIP, BGE-M3, Jina CLIP v2)
- Implement image search with text queries
- Build a multimodal vector store with text + image embeddings
- Combine multimodal retrieval with text-based RAG
- Apply multimodal embeddings to portfolio projects (StayBot, MARS)
- Benchmark the tradeoffs between models

## Introduction

Multimodal embedding models map **text and images into a shared vector space**. This enables powerful capabilities:

- **Text → Image search**: "Find images of cozy Airbnb bedrooms"
- **Image → Text search**: Given an image, find similar text descriptions
- **Image → Image search**: Find visually similar images
- **Multimodal RAG**: Combine text and image retrieval in a single query

The breakthrough was **CLIP** (OpenAI, 2021). Modern models like **BGE-M3** (BAAI), **Jina CLIP v2** (Jina AI), and **OpenAI CLIP** improved on it.

This note covers the architecture, modern models, and production pipelines.

![Multimodal embeddings](https://example.com/multimodal-embeddings.png)

---

## 1. The CLIP Architecture

CLIP (Contrastive Language-Image Pre-training) trains two encoders jointly:

```
Text → Text Encoder (Transformer) → 512-dim vector
                                    ↑
                               (cosine similarity)
                                    ↓
Image → Image Encoder (ViT or ResNet) → 512-dim vector
```

Training objective: in a batch of (text, image) pairs, maximize cosine similarity between matching pairs and minimize it for non-matching pairs.

The result: text and image embeddings live in the same vector space. The cosine similarity between text and image embeddings measures semantic relevance.

---

## 2. Modern Models

| Model | Embedding dim | Strengths | Weakback | Use case |
|-------|--------------|-----------|----------|----------|
| **OpenAI CLIP ViT-L/14** | 768 | General purpose | English-only | Baseline |
| **OpenCLIP ViT-bigG** | 1280 | SOTA accuracy | Slow | High-quality |
| **BGE-M3 (BAAI)** | 1024 | Multilingual text + image | Newer | Multilingual |
| **Jina CLIP v2** | 768 | Multilingual, fast | Smaller | Production |
| **SigLIP** | 768 | Better than CLIP | Newer | Production |
| **Cohere embed-multimodal-v3** | 1024 | Commercial | API-based | Managed |

### 2.1 OpenAI CLIP via HuggingFace

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()


def embed_text(text: str) -> torch.Tensor:
    """Generate text embedding."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # Normalize for cosine similarity
    return text_features / text_features.norm(dim=-1, keepdim=True)


def embed_image(image: Image.Image) -> torch.Tensor:
    """Generate image embedding."""
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    return image_features / image_features.norm(dim=-1, keepdim=True)
```

### 2.2 Jina CLIP v2 (multilingual)

```python
from transformers import AutoModel, AutoProcessor


model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)


def embed_jina_text(text: str) -> torch.Tensor:
    """Multilingual text embedding via Jina CLIP v2."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    return text_features / text_features.norm(dim=-1, keepdim=True)


def embed_jina_image(image: Image.Image) -> torch.Tensor:
    """Multilingual image embedding."""
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    return image_features / image_features.norm(dim=-1, keepdim=True)
```

Jina CLIP v2 supports 89 languages, including English, Chinese, Spanish, French, German.

### 2.3 BGE-M3 (multimodal + multilingual + dense)

```python
from FlagEmbedding import BGEM3FlagModel


model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def embed_bge_multimodal(text: str = None, image: Image.Image = None) -> torch.Tensor:
    """BGE-M3 multimodal embedding."""
    if text:
        output = model.encode(text=[text], return_dense=True, return_sparse=False)
        return torch.tensor(output["dense_vecs"][0])
    elif image:
        # BGE-VL for image embeddings (related model)
        from transformers import AutoModel
        vl_model = AutoModel.from_pretrained("BAAI/BGE-VL-base")
        # ... encode image ...
```

---

## 3. Building a Multimodal Vector Store

For a production multimodal RAG, store both text and image embeddings in the same vector space.

### 3.1 Qdrant with multimodal support

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


client = QdrantClient(url="http://localhost:6333")


def create_multimodal_collection():
    """Create a Qdrant collection with multimodal vectors."""
    
    # Use a single collection with named vectors
    client.create_collection(
        collection_name="multimodal_rag",
        vectors_config={
            "text": VectorParams(size=768, distance=Distance.COSINE),
            "image": VectorParams(size=768, distance=Distance.COSINE),
        },
    )


def index_document(doc_id: str, text: str, image: Image.Image, metadata: dict):
    """Index a document with both text and image embeddings."""
    
    text_embedding = embed_text(text)
    image_embedding = embed_image(image)
    
    client.upsert(
        collection_name="multimodal_rag",
        points=[
            PointStruct(
                id=doc_id,
                vector={
                    "text": text_embedding.tolist()[0],
                    "image": image_embedding.tolist()[0],
                },
                payload=metadata,
            ),
        ],
    )
```

### 3.2 Search with text query

```python
def text_search(query: str, top_k: int = 5) -> list[dict]:
    """Search by text query."""
    query_embedding = embed_text(query)
    
    results = client.search(
        collection_name="multimodal_rag",
        query_vector=("text", query_embedding.tolist()[0]),
        limit=top_k,
    )
    
    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]


def image_search(image: Image.Image, top_k: int = 5) -> list[dict]:
    """Search by image query."""
    query_embedding = embed_image(image)
    
    results = client.search(
        collection_name="multimodal_rag",
        query_vector=("image", query_embedding.tolist()[0]),
        limit=top_k,
    )
    
    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]


def multimodal_search(
    text_query: str = None,
    image_query: Image.Image = None,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    top_k: int = 5,
) -> list[dict]:
    """Search with both text and image queries (weighted fusion)."""
    
    # Build queries
    query_vectors = {}
    if text_query:
        query_vectors["text"] = embed_text(text_query).tolist()[0]
    if image_query:
        query_vectors["image"] = embed_image(image_query).tolist()[0]
    
    # RRF fusion (Reciprocal Rank Fusion)
    results = client.search_batch(
        collection_name="multimodal_rag",
        requests=[
            SearchRequest(
                vector=("text", query_vectors["text"]),
                limit=top_k * 2,
            ),
        ] if "text" in query_vectors else [
            SearchRequest(
                vector=("image", query_vectors["image"]),
                limit=top_k * 2,
            ),
        ],
    )
    
    # Combine results
    fused_scores = {}
    for result_set in results:
        for rank, r in enumerate(result_set):
            fused_scores[r.id] = fused_scores.get(r.id, 0) + 1 / (60 + rank + 1)
    
    sorted_results = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]
    
    return [{"id": doc_id, "score": score} for doc_id, score in sorted_results]
```

---

## 4. LlamaIndex Multimodal Pipelines

LlamaIndex (covered in [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai|06/24]]) provides high-level multimodal RAG.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.clip import ClipEmbedding
from qdrant_client import QdrantClient


# Multi-modal LLM
mm_llm = OpenAIMultiModal(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# CLIP embeddings
clip_embed = ClipEmbedding(model_name="ViT-B/32")

# Qdrant vector store
qdrant_client = QdrantClient(url="http://localhost:6333")
qdrant_store = QdrantVectorStore(client=qdrant_client, collection_name="multimodal_rag")

# Build multi-modal index
documents = SimpleDirectoryReader("./data/listing_images/").load_data()

storage_context = StorageContext.from_defaults(vector_store=qdrant_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=clip_embed,
)

# Query with text
query_engine = index.as_query_engine(multi_modal_llm=mm_llm)
response = query_engine.query("Show me apartments with cozy bedrooms and modern kitchens")
print(response)
# Includes the retrieved images + LLM-generated description
```

LlamaIndex automatically:
- Extracts images from documents
- Generates CLIP embeddings for each image
- Stores in Qdrant with metadata
- Retrieves images based on text queries
- Uses vision-capable LLM (GPT-4o) to generate responses

---

## 5. Multimodal RAG for the StayBot

For your **StayBot** portfolio project:

```python
def staybot_multimodal_search(query: str) -> dict:
    """Multimodal RAG for Airbnb listings."""
    
    # 1. Text-based retrieval of listing descriptions
    text_results = text_search(query, top_k=5)
    
    # 2. Image-based retrieval of listing photos
    image_results = image_search(query, top_k=5)
    
    # 3. Multimodal LLM generates a response
    context = ""
    for r in text_results:
        context += f"Listing: {r['payload']['description']}\n"
    
    for r in image_results:
        context += f"Image URL: {r['payload']['image_url']}\n"
    
    # Vision-capable LLM
    response = llm.complete(
        f"User query: {query}\n\nListings:\n{context}\n\nRecommend the best listings:",
        images=[r['payload']['image_url'] for r in image_results[:3]],
    )
    
    return {"text_recommendations": response, "matching_images": image_results[:3]}
```

The result: users searching "cozy apartment near the beach" get listings with descriptions AND matching photos, presented by a vision-capable LLM.

---

## 6. Case Study — MARS Multimodal Agent

For your **Multi-Agent Research System**:

```python
class MultimodalResearchAgent:
    """Research agent that processes papers, figures, and tables."""
    
    def __init__(self):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.gpt4_vision = OpenAIMultiModal(model="gpt-4o")
    
    def answer_question(self, question: str, paper_assets: list[dict]) -> str:
        """Answer a question using paper text, figures, and tables."""
        
        # 1. Find relevant figures and tables via CLIP
        if any(asset["type"] == "image" for asset in paper_assets):
            relevant_images = image_search(question, paper_assets)
        else:
            relevant_images = []
        
        # 2. Find relevant text via text embeddings
        relevant_text = text_search(question, paper_assets)
        
        # 3. Use vision-capable LLM
        response = self.gpt4_vision.complete(
            f"Question: {question}\n\n"
            f"Relevant text: {relevant_text}\n\n"
            f"Refer to the figures and tables in the images.",
            images=[img["url"] for img in relevant_images[:3]],
        )
        
        return response
```

---

## 7. The Multimodal Embedding Pipeline

A complete production pipeline:

```python
from pathlib import Path
from PIL import Image


class MultimodalIngestionPipeline:
    """End-to-end multimodal ingestion for a corpus of documents."""
    
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.qdrant = QdrantClient(url="http://localhost:6333")
        self.collection = "multimodal_corpus"
    
    def process_document(self, doc_path: str):
        """Process a document with mixed text + images."""
        
        # 1. Extract text + images
        text_chunks, image_paths = extract_text_and_images(doc_path)
        
        # 2. Embed and index text chunks
        for chunk_id, chunk in enumerate(text_chunks):
            text_embedding = self._embed_text(chunk)
            self._upsert(
                point_id=f"{doc_path}#{chunk_id}",
                text_vector=text_embedding,
                payload={"type": "text", "content": chunk, "source": doc_path},
            )
        
        # 3. Embed and index images
        for image_id, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            image_embedding = self._embed_image(image)
            self._upsert(
                point_id=f"{doc_path}#img{image_id}",
                image_vector=image_embedding,
                payload={"type": "image", "path": image_path, "source": doc_path},
            )
    
    def _embed_text(self, text: str) -> list[float]:
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        return (embedding / embedding.norm(dim=-1, keepdim=True)).tolist()[0]
    
    def _embed_image(self, image: Image.Image) -> list[float]:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        return (embedding / embedding.norm(dim=-1, keepdim=True)).tolist()[0]
    
    def _upsert(self, point_id: str, text_vector=None, image_vector=None, payload=None):
        vectors = {}
        if text_vector:
            vectors["text"] = text_vector
        if image_vector:
            vectors["image"] = image_vector
        
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vectors, payload=payload)],
        )
```

---

## 8. Benchmarking — Which Model Wins?

| Model | MTEB-R | MTEB-I | Lang | Speed (img/s) |
|-------|--------|--------|------|---------------|
| OpenAI CLIP ViT-L/14 | 60.4 | 56.7 | EN | 50 (A100) |
| OpenCLIP ViT-bigG | 65.5 | 62.0 | EN | 15 (A100) |
| Jina CLIP v2 | 60.0 | 59.2 | 89 | 30 (A100) |
| BGE-M3 | 65.0 | - | 100+ | 40 (A100) |
| SigLIP | 64.0 | 60.5 | EN | 45 (A100) |

(MTEB-R = text retrieval, MTEB-I = image retrieval benchmarks)

For most enterprise use cases:
- **Jina CLIP v2** if multilingual is critical
- **OpenAI CLIP ViT-L/14** if speed matters and English-only
- **OpenCLIP ViT-bigG** if accuracy is paramount and compute is available
- **BGE-M3** for dense text + multimodal

---

## 9. Antipatterns

### 9.1 Antipattern 1: Using CLIP for OCR

```python
# ❌ CLIP embeddings aren't trained for text recognition
clip_embedding = embed_image(image_with_text)

# ✅ Use OCR for text; CLIP for visual features
ocr_text = pytesseract.image_to_string(image_with_text)
visual_features = embed_image(image_with_text)
```

### 9.2 Antipattern 2: Normalizing vectors for cosine

```python
# ❌ Forget to normalize; cosine similarity breaks
text_features = model.get_text_features(**inputs)
similarity = text_features @ image_features.T  # wrong scale

# ✅ Normalize for cosine similarity
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
similarity = text_features @ image_features.T  # correct
```

### 9.3 Antipattern 3: Single-vector for multi-modal

```python
# ❌ Single vector combining text + image features loses information
combined = (text_features + image_features) / 2

# ✅ Use named vectors in Qdrant
vectors = {"text": text_features, "image": image_features}
```

### 9.4 Antipattern 4: No image preprocessing

```python
# ❌ Raw images of any size → inconsistent embeddings
image = Image.open("raw_photo.jpg")
embedding = embed_image(image)

# ✅ Resize and normalize
image = Image.open("raw_photo.jpg").convert("RGB").resize((224, 224))
embedding = embed_image(image)
```

### 9.5 Antipattern 5: Ignoring model-specific tokenization

```python
# ❌ Use any text as input to a non-English model
embedding = embed_text("a cozy bedroom")  # works for English, fails for other languages

# ✅ Use language-appropriate text for multilingual models
if jina_clip_v2:
    embedding = embed_text_jina("ein gemütliches Schlafzimmer")  # German OK
```

---

## 🎯 Key Takeaways

- CLIP-style models map text + images into a shared vector space.
- Modern models: OpenAI CLIP, OpenCLIP, BGE-M3, Jina CLIP v2, SigLIP.
- Use named vectors in Qdrant for multimodal search with separate text and image embeddings.
- LlamaIndex provides high-level multimodal RAG with vision-capable LLMs (GPT-4o).
- For 89-language support, use Jina CLIP v2.
- For SOTA accuracy, use OpenCLIP ViT-bigG.
- Always normalize embeddings before cosine similarity.
- Avoid CLIP for OCR, missing normalization, single-vector for multimodal, no preprocessing, language mismatches.

## References

- CLIP paper — [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- OpenCLIP — [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- Jina CLIP v2 — [huggingface.co/jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2)
- BGE-M3 — [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- MTEB benchmark — [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- LlamaIndex multi-modal — [docs.llamaindex.ai/en/stable/module_guides/models/multi_modal](https://docs.llamaindex.ai/en/stable/module_guides/models/multi_modal)
- [[06 - Large Language Models/26 - Multimodal Production RAG/01 - PDF Parsing and Extraction|Note 01 — PDF Parsing]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/02 - Visual Document Retrieval - ColPali and Beyond|Note 02 — ColPali]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/04 - Video and Audio RAG|Note 04 — Video/Audio RAG]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai|Production RAG: Haystack + txtai]]
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]]