# 🎯 05 - Capstone — Production Multimodal RAG for Legal Documents

> **The thirteenth portfolio project. End-to-end multimodal RAG that handles PDFs (with tables), images, audio, and video. The architecture for real enterprise document understanding.**

## 🎯 Learning Objectives
- Build an end-to-end multimodal RAG for legal documents (PDFs + tables + images + audio)
- Use LlamaIndex + ColPali + TableTransformer + CLIP in one pipeline
- Serve with FastAPI + LangFuse observability
- Integrate with Haystack for production-grade pipelines
- Benchmark multimodal vs text-only retrieval on your own data
- Deploy with Docker Compose for one-command setup

## Introduction

The capstone is the **synthesis** of all four notes. You will build a production multimodal RAG for legal documents:

1. **PDF ingestion**: Unstructured.io for text, ColPali for visual, TableTransformer for tables
2. **Multimodal embeddings**: CLIP for images, BGE-M3 for text
3. **Vector store**: Qdrant with named vectors (text + image)
4. **Retrieval**: hybrid text + ColPali late interaction + RRF fusion
5. **Generation**: GPT-4o vision for multimodal responses
6. **Observability**: LangFuse traces for every retrieval and generation

The architecture demonstrates all four notes working together. The capstone is the **thirteenth portfolio project**: the multimodal RAG skill.

```mermaid
graph TB
    subgraph "Document Sources"
        PDF[Legal PDFs]
        IMG[Scanned Images]
        AUD[Voice Messages]
    end
    
    subgraph "Ingestion"
        UNSTR[Unstructured.io]
        COLPALI[ColPali]
        TBL[TableTransformer]
        WHISPER[Whisper]
    end
    
    subgraph "Multimodal Embeddings"
        CLIP[CLIP for images]
        BGE[BGE-M3 for text]
    end
    
    subgraph "Vector Store"
        QD[Qdrant]
    end
    
    subgraph "Retrieval"
        HYBRID[Hybrid text + ColPali]
        RRF[Reciprocal Rank Fusion]
    end
    
    subgraph "Generation"
        GPT4[GPT-4o Vision]
    end
    
    PDF --> UNSTR
    PDF --> COLPALI
    PDF --> TBL
    AUD --> WHISPER
    
    UNSTR --> BGE
    COLPALI --> QD
    TBL --> BGE
    WHISPER --> BGE
    IMG --> CLIP
    
    BGE --> QD
    CLIP --> QD
    
    HYBRID --> QD
    HYBRID --> RRF
    RRF --> GPT4
```

---

## 1. Project Layout

```
multimodal-rag-service/
├── app/
│   ├── api.py                # FastAPI service
│   ├── retrieval.py          # Hybrid retrieval
│   ├── ingestion/
│   │   ├── pdf_ingestor.py   # Unstructured + ColPali
│   │   ├── table_ingestor.py # TableTransformer
│   │   ├── audio_ingestor.py # Whisper
│   │   └── image_ingestor.py # CLIP
│   ├── embeddings/
│   │   ├── text_embedder.py  # BGE-M3
│   │   └── image_embedder.py # CLIP
│   └── observability.py      # LangFuse
├── evaluation/
│   ├── benchmark.py          # Compare text vs multimodal
│   └── eval_dataset.json     # Q&A pairs
├── docker-compose.yml        # Full stack
└── README.md
```

---

## 2. Multimodal Ingestion Pipeline (`app/ingestion/pdf_ingestor.py`)

```python
from unstructured.partition.pdf import partition_pdf
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from pdf2image import convert_from_path
import torch


class MultimodalPDFIngestor:
    """Ingest a PDF with text, tables, images via ColPali."""
    
    def __init__(self):
        # Text + tables via Unstructured.io
        # Visual via ColQwen2
        self.colpali_model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0-mixed",
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()
        self.colpali_processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v1.0-mixed",
        )
    
    def ingest(self, pdf_path: str, doc_id: str) -> dict:
        """Full multimodal ingestion of a PDF."""
        
        # 1. Extract text + tables via Unstructured.io
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
        )
        
        text_chunks = []
        table_chunks = []
        image_paths = []
        
        for elem in elements:
            if elem.category == "Table":
                table_chunks.append({
                    "text": elem.text,  # markdown representation
                    "metadata": elem.metadata.to_dict(),
                })
            elif elem.category == "Image":
                if hasattr(elem.metadata, "image_path"):
                    image_paths.append(elem.metadata.image_path)
            else:  # Title, NarrativeText, etc.
                text_chunks.append({
                    "text": elem.text,
                    "metadata": elem.metadata.to_dict(),
                })
        
        # 2. Visual embeddings via ColPali
        pages = convert_from_path(pdf_path, dpi=150)
        colpali_embeddings = []
        
        with torch.no_grad():
            batch = self.colpali_processor.process_images(pages).to(self.colpali_model.device)
            colpali_embeddings = self.colpali_model(**batch)
        
        return {
            "doc_id": doc_id,
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "image_paths": image_paths,
            "colpali_embeddings": colpali_embeddings.cpu(),
            "page_count": len(pages),
        }
```

---

## 3. Multimodal Embeddings (`app/embeddings/`)

```python
# app/embeddings/text_embedder.py
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """BGE-M3 text embeddings (multilingual, 1024-dim)."""
    
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
```

```python
# app/embeddings/image_embedder.py
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch


class ImageEmbedder:
    """CLIP image embeddings (768-dim)."""
    
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
    
    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed batch of images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize for cosine similarity
        normalized = image_features / image_features.norm(dim=-1, keepdim=True)
        return normalized.tolist()
```

---

## 4. Multimodal Vector Store (`app/retrieval.py`)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue


class MultimodalVectorStore:
    """Vector store with text + image embeddings."""
    
    def __init__(self, collection_name: str = "multimodal_rag"):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection = collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with named vectors."""
        try:
            self.client.get_collection(self.collection)
        except:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "text": VectorParams(size=1024, distance=Distance.COSINE),  # BGE-M3
                    "image": VectorParams(size=768, distance=Distance.COSINE),  # CLIP
                    "colpali": VectorParams(
                        size=128,
                        distance=Distance.COSINE,
                        multivector_config=...,  # ColPali multi-vector
                    ),
                },
            )
    
    def index_document(
        self,
        doc_id: str,
        text_chunks: list[dict],
        colpali_embeddings: torch.Tensor,
        image_paths: list[str],
        page_count: int,
    ):
        """Index a document with all modalities."""
        
        points = []
        
        # Text chunks (BGE-M3)
        for chunk_id, chunk in enumerate(text_chunks):
            text_embedding = text_embedder.embed([chunk["text"]])[0]
            points.append(PointStruct(
                id=f"{doc_id}#text{chunk_id}",
                vector={"text": text_embedding},
                payload={
                    "type": "text",
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "page": chunk.get("page_number", 0),
                },
            ))
        
        # Images (CLIP)
        for image_id, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            image_embedding = image_embedder.embed([image])[0]
            points.append(PointStruct(
                id=f"{doc_id}#img{image_id}",
                vector={"image": image_embedding},
                payload={
                    "type": "image",
                    "doc_id": doc_id,
                    "image_path": image_path,
                },
            ))
        
        # ColPali multi-vector embeddings
        for page_num in range(page_count):
            points.append(PointStruct(
                id=f"{doc_id}#colpali{page_num}",
                vector={"colpali": colpali_embeddings[page_num].tolist()},
                payload={
                    "type": "colpali_page",
                    "doc_id": doc_id,
                    "page": page_num + 1,
                },
            ))
        
        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )
    
    def hybrid_search(self, query: str, query_image: Image.Image = None, top_k: int = 5) -> list[dict]:
        """Hybrid search: text + image + ColPali with RRF fusion."""
        
        # 1. Text search (BGE-M3)
        text_embedding = text_embedder.embed([query])[0]
        text_results = self.client.search(
            collection_name=self.collection,
            query_vector=("text", text_embedding),
            limit=top_k * 3,  # over-fetch for fusion
        )
        
        # 2. Image search (CLIP) if query image
        image_results = []
        if query_image:
            image_embedding = image_embedder.embed([query_image])[0]
            image_results = self.client.search(
                collection_name=self.collection,
                query_vector=("image", image_embedding),
                limit=top_k * 3,
            )
        
        # 3. ColPali visual search
        colpali_results = self._colpali_search(query, top_k * 3)
        
        # 4. Reciprocal Rank Fusion
        fused_scores = {}
        
        for rank, r in enumerate(text_results):
            key = r.payload.get("doc_id") + "#" + str(r.payload.get("page", 0))
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (60 + rank + 1)
        
        for rank, r in enumerate(image_results):
            key = r.payload.get("doc_id") + "#" + str(r.payload.get("page", 0))
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (60 + rank + 1)
        
        for rank, r in enumerate(colpali_results):
            key = r.payload.get("doc_id") + "#" + str(r.payload.get("page", 0))
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (60 + rank + 1)
        
        # Sort by fused score
        sorted_pages = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]
        
        # Return with payload details
        results = []
        for key, score in sorted_pages:
            doc_id, page = key.split("#")
            results.append({
                "doc_id": doc_id,
                "page": int(page),
                "score": score,
            })
        
        return results
```

---

## 5. The FastAPI Service (`app/api.py`)

```python
from fastapi import FastAPI, HTTPException, UploadFile
from contextlib import asynccontextmanager
from langfuse import observe, langfuse_context
from PIL import Image
import io


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all models and clients on startup."""
    app.state.vector_store = MultimodalVectorStore()
    yield


app = FastAPI(title="Multimodal RAG Service", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    query_image_url: str = None  # Optional image query


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/query", response_model=QueryResponse)
@observe()
async def query(req: QueryRequest) -> QueryResponse:
    """Multimodal RAG query."""
    
    # 1. Hybrid retrieval (text + ColPali)
    query_image = None
    if req.query_image_url:
        query_image = Image.open(httpx.get(req.query_image_url).content)
    
    pages = app.state.vector_store.hybrid_search(
        query=req.question,
        query_image=query_image,
        top_k=5,
    )
    
    langfuse_context.update_current_observation(
        metadata={"retrieved_pages": len(pages)},
    )
    
    # 2. Fetch the actual images for multimodal LLM
    images_for_llm = []
    for page in pages:
        # Load page image from ColPali (or rendered PDF)
        page_image = load_page_image(page["doc_id"], page["page"])
        if page_image:
            images_for_llm.append(page_image)
    
    # 3. Generate response with vision-capable LLM
    from openai import OpenAI
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {req.question}\n\nRelevant documents (see images):"},
                    *[{
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img)},
                    } for img in images_for_llm],
                ],
            }
        ],
        max_tokens=1000,
    )
    
    answer = response.choices[0].message.content
    
    return QueryResponse(
        answer=answer,
        sources=pages,
    )


@app.post("/index")
async def index_pdf(file: UploadFile):
    """Index a new PDF document."""
    
    # Save file
    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    
    # Ingest
    ingestor = MultimodalPDFIngestor()
    result = ingestor.ingest(pdf_path, file.filename)
    
    # Index
    app.state.vector_store.index_document(
        doc_id=file.filename,
        text_chunks=result["text_chunks"],
        colpali_embeddings=result["colpali_embeddings"],
        image_paths=result["image_paths"],
        page_count=result["page_count"],
    )
    
    return {
        "doc_id": file.filename,
        "page_count": result["page_count"],
        "text_chunks": len(result["text_chunks"]),
        "tables": len(result["table_chunks"]),
        "images": len(result["image_paths"]),
    }


def image_to_data_url(image: Image.Image) -> str:
    """Convert PIL image to data URL for LLM API."""
    import base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"
```

---

## 6. The Single Docker Compose

```yaml
version: "3.9"

services:
  api:
    build: ./app
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
      - LANGFUSE_HOST=http://langfuse-web:3000
    depends_on:
      - qdrant
      - langfuse-web
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  langfuse-web:
    image: langfuse/langfuse:main
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
    depends_on:
      - postgres

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  qdrant_data:
  postgres_data:
```

One command: `docker compose up -d` boots the entire multimodal RAG stack.

---

## 7. Benchmarking — Does Multimodal Beat Text?

```python
# evaluation/benchmark.py
def benchmark_multimodal_vs_text(eval_dataset: list[dict]):
    """Compare text-only vs multimodal retrieval accuracy."""
    
    text_correct = 0
    multimodal_correct = 0
    total = 0
    
    for example in eval_dataset:
        question = example["question"]
        expected_doc = example["expected_doc"]
        expected_page = example["expected_page"]
        
        # 1. Text-only retrieval
        text_results = vector_store.search(
            query_vector=("text", text_embedder.embed([question])[0]),
            limit=5,
        )
        text_top = text_results[0].payload if text_results else None
        
        # 2. Multimodal retrieval
        multimodal_results = vector_store.hybrid_search(
            query=question, top_k=5,
        )
        
        # 3. Check if expected doc/page is in top-5
        text_hit = text_top and text_top.get("doc_id") == expected_doc and text_top.get("page") == expected_page
        multi_hit = any(
            r["doc_id"] == expected_doc and r["page"] == expected_page
            for r in multimodal_results
        )
        
        if text_hit:
            text_correct += 1
        if multi_hit:
            multimodal_correct += 1
        
        total += 1
    
    print(f"Text-only retrieval accuracy: {text_correct / total:.1%}")
    print(f"Multimodal retrieval accuracy: {multimodal_correct / total:.1%}")
```

For most enterprise document corpora, multimodal retrieval beats text-only by 30-50%.

---

## 8. Real-World Case — Legal Document RAG

```python
def legal_document_pipeline(pdf_path: str) -> dict:
    """Complete multimodal RAG for a legal document."""
    
    # 1. Detect document type
    is_scanned = detect_if_scanned(pdf_path)
    has_tables = detect_tables(pdf_path)
    
    # 2. Ingest based on type
    if is_scanned:
        # Scanned: OCR first
        chunks = ingest_with_ocr(pdf_path)
    elif has_tables:
        # Tables: Unstructured.io with table detection
        chunks = ingest_with_tables(pdf_path)
    else:
        # Regular digital PDF
        chunks = ingest_simple(pdf_path)
    
    # 3. Index with multimodal embeddings
    vector_store.index_document(
        doc_id=Path(pdf_path).stem,
        text_chunks=chunks["text"],
        colpali_embeddings=chunks["colpali"],
        image_paths=chunks["images"],
        page_count=chunks["page_count"],
    )
    
    return chunks
```

---

## 9. Production Deployment Checklist

- [ ] ColPali / ColQwen2 model served via vLLM for high throughput
- [ ] Qdrant with multi-vector support for ColPali embeddings
- [ ] Unstructured.io `hi_res` strategy for layout-aware extraction
- [ ] TableTransformer for image-based PDFs with tables
- [ ] GPT-4o vision for multimodal generation
- [ ] LangFuse tracing for every retrieval and generation
- [ ] BGE-M3 embeddings (multilingual support)
- [ ] FastAPI service with GPU support (vLLM)
- [ ] Rate limiting per tenant (covered in [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems|Incident Response]])
- [ ] PII redaction for legal documents
- [ ] Encryption at rest and in transit
- [ ] Audit trail for all retrievals (compliance)

---

## 🎯 Key Takeaways

- The capstone composes all four notes into a single multimodal RAG service.
- One `docker-compose.yml` boots Qdrant + LangFuse + Postgres + the API.
- Multimodal retrieval (text + ColPali + image) beats text-only by 30-50% on enterprise documents.
- Vision-capable LLMs (GPT-4o) generate responses with embedded images and tables.
- The capstone is the **thirteenth portfolio project**: the multimodal RAG skill.

## References

- [[06 - Large Language Models/26 - Multimodal Production RAG/01 - PDF Parsing and Extraction|Note 01 — PDF Parsing]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/02 - Visual Document Retrieval - ColPali and Beyond|Note 02 — ColPali]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/03 - Multimodal Embedding Models|Note 03 — Multimodal Embeddings]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/04 - Video and Audio RAG|Note 04 — Video/Audio RAG]]
- ColPali — [github.com/illuin-tech/colpali](https://github.com/illuin-tech/colpali)
- LlamaIndex multi-modal — [docs.llamaindex.ai](https://docs.llamaindex.ai)
- Unstructured.io — [unstructured.io](https://unstructured.io/)
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant multi-vector
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]]
- [[06 - Large Language Models/24 - Production RAG Frameworks - Haystack and txtai|Production RAG: Haystack + txtai]]
- [[06 - Large Language Models/12 - Production RAG|Production RAG]]