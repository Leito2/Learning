# 🎯 02 - Visual Document Retrieval — ColPali and the Late Interaction Revolution

> **The breakthrough that changed PDF retrieval. ColPali renders pages as images, encodes them with a vision model, and retrieves by visual similarity. 30%+ accuracy improvement on technical documents.**

## 🎯 Learning Objectives
- Understand the ColPali architecture (vision encoder + late interaction)
- Compare ColPali to traditional text-based PDF retrieval
- Implement ColPali with HuggingFace + vLLM
- Use ColPali v2 with multi-vector late interaction
- Apply visual document retrieval to production RAG systems
- Benchmark ColPali against text-based retrieval on your own data
- Use ColQwen2 for production-grade visual document understanding

## Introduction

Traditional PDF retrieval has a fundamental limitation: it loses information. A chart's axes, a figure's caption, a table's visual structure — all are lost when text is extracted and embedded.

**ColPali** (2024) and **ColPali v2** (2025) changed this. The approach:

1. Render each PDF page as an image
2. Pass the image through a vision encoder (PaliGemma, Qwen2-VL)
3. Get a multi-vector embedding (one embedding per image patch)
4. At query time, embed the query with the same encoder
5. Use **late interaction** (MaxSim) to score the query against each patch

The result: the model **sees the page** rather than reading extracted text. It picks up on visual cues that text extraction misses.

This note covers the ColPali architecture, comparison with text-based retrieval, and how to implement it in production.

![ColPali architecture](https://example.com/colpali.png)

---

## 1. The Traditional PDF Retrieval Problem

The typical pipeline:

```python
# Traditional: extract text, embed, retrieve
text = extract_text_from_pdf(pdf)  # Lost: charts, figures, layout
chunks = chunk_text(text)
embeddings = embed_chunks(chunks)
relevant = retrieve(query, embeddings)
```

**What's lost**:
- A bar chart showing "Q1: 100, Q2: 200, Q3: 150" becomes just the legend text or nothing
- A figure with axis labels and trend lines: zero information extracted
- A table with merged cells: structural relationships lost
- Tables vs prose vs lists: layout context lost

For technical documents (research papers, financial reports, engineering specs), this information loss is catastrophic. ColPali was designed to solve exactly this.

---

## 2. The ColPali Architecture

ColPali (and ColQwen2) follow a four-stage pipeline:

```
PDF page → Image (rendered) → PaliGemma/Qwen2-VL → Multi-vector embedding → MaxSim scoring
```

### 2.1 Stage 1: PDF page rendering

```python
from pdf2image import convert_from_path


def render_pdf_pages(pdf_path: str, dpi: int = 150) -> list[Image.Image]:
    """Render PDF pages as images."""
    return convert_from_path(pdf_path, dpi=dpi)
```

DPI 150 is a sweet spot: high enough for OCR and visual features, low enough for fast processing.

### 2.2 Stage 2: Vision encoding

The image is passed through a vision-language model (PaliGemma-3B for ColPali v1; Qwen2-VL-2B for ColQwen2). The model produces a sequence of **patch embeddings** — one embedding per image patch (typically 32×32 pixels).

```python
from colpali_engine.models import ColPali, ColPaliProcessor


def load_colpali():
    """Load ColPali model + processor."""
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.3-mixed",
        torch_dtype=torch.float16,
        device_map="cuda",  # or "mps" for Apple Silicon, "cpu" for CPU
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-mixed")
    
    return model, processor
```

### 2.3 Stage 3: Multi-vector embeddings

For each page, the model produces ~700-1000 patch embeddings (1024-dim each). This is a "multi-vector" representation — one vector per patch, not one vector per page.

```python
import torch
from PIL import Image


def embed_images(model, processor, images: list[Image.Image]) -> torch.Tensor:
    """Generate multi-vector embeddings for page images."""
    
    batch_queries = processor.process_images(images).to(model.device)
    
    with torch.no_grad():
        embeddings = model(**batch_queries)
    
    # Returns tensor of shape [batch_size, num_patches, embedding_dim]
    return embeddings
```

For a 10-page PDF, you get 10 × ~800 × 1024 = 8M floats. This is the **storage cost** — multi-vector embeddings are larger than single-vector embeddings.

### 2.4 Stage 4: Late interaction (MaxSim scoring)

At query time, the query is embedded with the same encoder:

```python
def embed_query(model, processor, query: str) -> torch.Tensor:
    """Generate multi-vector embedding for query."""
    batch_queries = processor.process_queries([query]).to(model.device)
    
    with torch.no_grad():
        query_embedding = model(**batch_queries)
    
    return query_embedding
```

Then the score is computed via **MaxSim** (Maximum Similarity):

```python
def maxsim_score(query_embedding: torch.Tensor, doc_embedding: torch.Tensor) -> float:
    """Late interaction: max similarity between any query token and any document token."""
    
    # query_embedding: [num_query_tokens, dim]
    # doc_embedding: [num_doc_tokens, dim]
    
    similarity_matrix = query_embedding @ doc_embedding.T  # [num_q, num_d]
    
    # Max over document tokens, then mean over query tokens
    max_per_query = similarity_matrix.max(dim=1).values  # [num_q]
    score = max_per_query.mean().item()
    
    return score
```

For each query token, find the most similar document token. Average these maximums. This is the **late interaction** — instead of compressing to one vector early (single-vector retrieval), the similarity is computed at the patch level.

---

## 3. ColPali v2 and ColQwen2 — The Production Models

ColPali v2 (2025) and ColQwen2 (2025) are the production-ready models.

### 3.1 ColPali v2

```python
from colpali_engine.models import ColPali, ColPaliProcessor


def load_colpali_v2():
    """Load ColPali v2 (multi-vector, late interaction)."""
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.3-mixed",  # v1.3 is the latest stable v2-line
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-mixed")
    
    return model, processor
```

### 3.2 ColQwen2

```python
from colpali_engine.models import ColQwen2, ColQwen2Processor


def load_colqwen2():
    """Load ColQwen2 (Qwen2-VL backbone, better for complex layouts)."""
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0-mixed",
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-mixed")
    
    return model, processor
```

ColQwen2 uses Qwen2-VL (better than PaliGemma for complex layouts). For most enterprise documents, ColQwen2 is the right choice.

---

## 4. Building a ColPali Index

The full pipeline:

```python
import torch
from pathlib import Path
from PIL import Image


def build_colpali_index(pdf_dir: str, output_path: str):
    """Build a ColPali index over a directory of PDFs."""
    
    model, processor = load_colqwen2()
    
    all_embeddings = []
    all_metadata = []
    
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        print(f"Processing {pdf_path.name}")
        
        # 1. Render pages
        images = render_pdf_pages(str(pdf_path), dpi=150)
        
        # 2. Embed pages (multi-vector)
        page_embeddings = embed_images(model, processor, images)
        
        # 3. Store
        for page_num, embedding in enumerate(page_embeddings):
            all_embeddings.append(embedding.cpu())
            all_metadata.append({
                "pdf": pdf_path.name,
                "page": page_num + 1,
                "image": images[page_num],
            })
    
    # Save index
    torch.save({
        "embeddings": all_embeddings,
        "metadata": all_metadata,
    }, output_path)
    
    print(f"Index saved to {output_path}")
```

Storage size: ~10KB per page × N pages = significant. For 10K pages, expect ~100MB.

---

## 5. ColPali with vLLM for High Throughput

For production scale, use **vLLM** (covered in [[06 - Large Language Models/13 - vLLM and Advanced RAG|06/13]]) for high-throughput embedding inference.

```python
from vllm import LLM, SamplingParams


def build_colpali_index_vllm(pdf_dir: str, output_path: str):
    """Build ColPali index using vLLM for high throughput."""
    
    # Load ColPali via vLLM
    llm = LLM(
        model="vidore/colqwen2-v1.0-mixed",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    
    all_embeddings = []
    all_metadata = []
    
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        images = render_pdf_pages(str(pdf_path), dpi=150)
        
        # Process in batch
        # (vLLM specifics depend on ColPali support)
        embeddings = llm.encode(images)
        
        for page_num, embedding in enumerate(embeddings):
            all_embeddings.append(embedding)
            all_metadata.append({
                "pdf": pdf_path.name,
                "page": page_num + 1,
            })
```

For 10K pages, vLLM provides 5-10× throughput vs naive PyTorch inference.

---

## 6. Querying ColPali Index

```python
def query_colpali_index(index_path: str, query: str, top_k: int = 5) -> list[dict]:
    """Query the ColPali index."""
    
    model, processor = load_colqwen2()
    
    index = torch.load(index_path)
    
    # 1. Embed query
    query_embedding = embed_query(model, processor, query)
    
    # 2. Score against all pages (using MaxSim late interaction)
    scores = []
    for i, doc_embedding in enumerate(index["embeddings"]):
        score = maxsim_score(query_embedding[0], doc_embedding)
        scores.append((score, i))
    
    # 3. Top-K
    scores.sort(reverse=True)
    top_indices = [i for _, i in scores[:top_k]]
    
    # 4. Return results
    results = []
    for idx in top_indices:
        metadata = index["metadata"][idx]
        results.append({
            "pdf": metadata["pdf"],
            "page": metadata["page"],
            "score": scores[idx][0] if idx < len(scores) else 0,
        })
    
    return results
```

---

## 7. Accuracy Comparison — Why ColPali Wins

The original ColPali paper (2024) showed substantial improvements on the ViDoRe benchmark:

| Document type | Text-based retrieval | ColPali v1 | ColPali v2 / ColQwen2 |
|--------------|---------------------|------------|------------------------|
| **Research papers** | 60-70% nDCG@5 | 75-85% | 85-92% |
| **Financial reports** | 55-65% | 70-80% | 80-88% |
| **Technical manuals** | 50-60% | 65-75% | 75-85% |
| **Scanned documents** | 20-30% (after OCR) | 75-85% | 80-90% |

ColPali's advantage:
- **Visual cues**: understands layout, figures, charts
- **Multi-vector**: captures fine-grained information
- **Late interaction**: more expressive than single-vector embeddings

For technical documents (research, finance, engineering), ColPali is **30%+ more accurate** than text-based retrieval.

---

## 8. The Hybrid Approach — Text + Visual

For production, combine text-based and visual retrieval:

```python
def hybrid_retrieve(query: str, pdf_index: str, top_k: int = 5) -> list[dict]:
    """Hybrid text + visual retrieval."""
    
    # 1. Text-based retrieval
    text_results = text_based_retrieve(query, pdf_index, top_k=top_k * 2)
    
    # 2. Visual retrieval
    visual_results = query_colpali_index(pdf_index, query, top_k=top_k * 2)
    
    # 3. Score fusion (Reciprocal Rank Fusion)
    fused_scores = {}
    for rank, result in enumerate(text_results):
        fused_scores[result["page_id"]] = fused_scores.get(result["page_id"], 0) + 1 / (60 + rank + 1)
    for rank, result in enumerate(visual_results):
        fused_scores[result["page_id"]] = fused_scores.get(result["page_id"], 0) + 1 / (60 + rank + 1)
    
    # 4. Top-K
    sorted_pages = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]
    
    return [{"page_id": pid, "score": score} for pid, score in sorted_pages]
```

Hybrid retrieval combines the strengths of both:
- Text retrieval: good for prose, simple queries
- Visual retrieval: good for charts, figures, complex layouts
- Reciprocal Rank Fusion: balanced weighting

---

## 9. Real-World Example — Legal Documents

A complete ColPali pipeline for legal documents:

```python
def process_legal_documents(legal_pdf_dir: str) -> dict:
    """Process a directory of legal PDFs with ColPali."""
    
    # 1. Build index
    index_path = "legal_index.pt"
    build_colpali_index(legal_pdf_dir, index_path)
    
    # 2. Sample queries (real legal research)
    sample_queries = [
        "What is the termination clause?",
        "What is the liability cap?",
        "What is the indemnification scope?",
        "What are the payment terms?",
    ]
    
    # 3. Run queries
    results = {}
    for query in sample_queries:
        pages = query_colpali_index(index_path, query, top_k=3)
        results[query] = pages
    
    return results
```

The legal documents contain:
- Multi-page contracts with numbered sections
- Tables with payment schedules
- Definitions in bold formatting
- Footnotes and references

Text-based retrieval misses:
- Table data (extracted poorly)
- Section numbering (often garbled in extraction)
- Visual cues like indentation

ColPali handles all of these naturally.

---

## 10. Production Considerations

### 10.1 Storage

Multi-vector embeddings are large. For 10K pages:
- ~800 vectors × 1024 dims × 4 bytes = 3.2MB per page
- 10K pages × 3.2MB = 32GB total

**Compression options**:
- **ColPali v2 with quantization** — 4-bit quantization reduces to ~800KB per page
- **Product quantization** — further compression to ~100KB per page

### 10.2 Index structure

Store the ColPali index in a vector database that supports multi-vector:

| Vector DB | Multi-vector support | Quantization | Scale |
|-----------|---------------------|--------------|-------|
| **Qdrant** | ✅ Native | ✅ Built-in | Production |
| **Milvus** | ✅ Native | ✅ Built-in | Production |
| **LanceDB** | ✅ Native | ✅ Built-in | Production |
| **pgvector** | ⚠️ Limited | ❌ No | Mid-scale |

For 100K+ pages, use **Qdrant** with built-in quantization.

---

## 11. Antipatterns

### 11.1 Antipattern 1: Text-only retrieval on technical docs

```python
# ❌ Text extraction loses visual cues
text = extract_text(pdf)
relevant = text_retrieve(query, text)

# ✅ Use ColPali for visual docs
embeddings = colpali_embed(pdf_pages)
relevant = maxsim_retrieve(query, embeddings)
```

### 11.2 Antipattern 2: Single-vector embeddings for multi-page docs

```python
# ❌ One embedding per page = loss of detail
doc_embedding = mean_pool(all_patches)  # lost fine-grained info

# ✅ Multi-vector embeddings with late interaction
doc_embedding = all_patches  # keep all patches
score = maxsim(query_embedding, doc_embedding)  # patch-level matching
```

### 11.3 Antipattern 3: No GPU for ColPali inference

```python
# ❌ CPU inference: 10+ seconds per page
model = ColPali.from_pretrained(..., device_map="cpu")

# ✅ GPU inference: 100ms per page
model = ColPali.from_pretrained(..., device_map="cuda")
```

### 11.4 Antipattern 4: Loading the whole PDF as one image

```python
# ❌ Page-sized chunks lose cross-page context
embeddings = embed_one_big_image(all_pages)

# ✅ Page-level embeddings (preserves structure)
for page in render_pdf_pages(pdf):
    embeddings = embed_images([page])
```

### 11.5 Antipattern 5: No reranking after retrieval

```python
# ❌ ColPali alone (may not be enough)
relevant = colpali_retrieve(query, top_k=10)
return relevant[:5]

# ✅ Rerank with a cross-encoder for final precision
relevant = colpali_retrieve(query, top_k=20)
reranked = rerank_with_cross_encoder(query, relevant)
return reranked[:5]
```

---

## 🎯 Key Takeaways

- **ColPali / ColQwen2** render PDF pages as images and embed with a vision encoder.
- **Multi-vector embeddings** (one per patch) preserve fine-grained information.
- **Late interaction (MaxSim)** scores query against patches, not whole documents.
- **30%+ accuracy improvement** over text-based retrieval on technical documents.
- **ColQwen2** (Qwen2-VL backbone) is the production model for most use cases.
- **Hybrid retrieval** (text + visual) combines strengths of both approaches.
- **Multi-vector storage** is large (3MB/page uncompressed); use Qdrant with quantization.
- Avoid text-only retrieval on technical docs, single-vector embeddings, CPU inference, page-merge, no reranking.

## References

- ColPali paper — [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)
- ColPali v2 (ColQwen2) — [huggingface.co/vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0)
- ViDoRe benchmark — [huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)
- colpali_engine — [github.com/illuin-tech/colpali](https://github.com/illuin-tech/colpali)
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM and Advanced RAG]] — high-throughput inference
- [[06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/05 - Vision, Audio, and Multimodal Transformers|06/16/05 Vision, Audio, and Multimodal Transformers]]
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]] — Qdrant multi-vector
- [[06 - Large Language Models/26 - Multimodal Production RAG/01 - PDF Parsing and Extraction|Note 01 — PDF Parsing]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/03 - Multimodal Embedding Models|Note 03 — Multimodal Embeddings]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/05 - Capstone - Production Multimodal RAG|Note 05 — Capstone]]