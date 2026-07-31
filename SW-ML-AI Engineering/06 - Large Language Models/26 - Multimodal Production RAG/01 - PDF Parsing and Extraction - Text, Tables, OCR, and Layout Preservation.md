# 🎯 01 - PDF Parsing and Extraction — Text, Tables, OCR, and Layout Preservation

> **The first step of multimodal RAG. Convert messy PDFs into structured data the LLM can actually use. The difference between an enterprise-grade RAG and a toy demo.**

## 🎯 Learning Objectives
- Parse PDFs that contain text, tables, images, and scanned content
- Use Unstructured.io, PyMuPDF, and pdfplumber for different document types
- Apply OCR with Tesseract and PaddleOCR for scanned documents
- Extract tables with Camelot and TableTransformer
- Preserve document layout (headers, footers, footnotes) for context
- Build an ingestion pipeline that handles the messy real world
- Distinguish which tool to use for which document type

## Introduction

A PDF is not a document. It's a render description — a set of coordinates for text, images, and shapes that, when rendered, looks like a page. Extracting structured content from a PDF is hard because:

- **Text** might be in any order or encoded strangely
- **Tables** are visual structures, not semantic ones
- **Images** are embedded bitmaps that contain information text extraction can't capture
- **Scanned documents** are just images — you need OCR before any extraction
- **Layout** matters: a footnote at the bottom of page 3 belongs to the article on page 1

This note covers the four tools you'll use most: Unstructured.io for general parsing, PyMuPDF for fast text extraction, pdfplumber for table extraction, and Marker for clean markdown conversion. Plus OCR with Tesseract and table extraction with TableTransformer.

![PDF parsing pipeline](https://example.com/pdf-parsing.png)

---

## 1. The PDF Complexity Matrix

| PDF type | Tool | Difficulty |
|----------|------|------------|
| Digital text PDF | PyMuPDF | Easy |
| Multi-column research paper | Unstructured.io or pdfplumber | Medium |
| Tables with complex layouts | Camelot or TableTransformer | Hard |
| Scanned document (image-only) | Tesseract / PaddleOCR | Hard |
| Mixed text + image + table + header/footer | Unstructured.io + Marker | Hard |
| Multi-page with footnotes | Marker + custom logic | Very hard |

Most production systems use **Unstructured.io** as the default, with specialized tools for specialized cases.

---

## 2. PyMuPDF — Fast Text Extraction

```bash
pip install pymupdf
```

```python
import fitz  # PyMuPDF


def extract_text_simple(pdf_path: str) -> str:
    """Simple text extraction."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_with_metadata(pdf_path: str) -> list[dict]:
    """Extract text with positional metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        images = page.get_images()
        
        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "image_count": len(images),
            "block_count": len(page.get_text("blocks")),
        })
    doc.close()
    return pages
```

PyMuPDF is **fast** (10-100× faster than pdfplumber) but **layout-unaware** — multi-column papers come out jumbled.

---

## 3. pdfplumber — Tables and Layout

```bash
pip install pdfplumber
```

```python
import pdfplumber


def extract_text_with_tables(pdf_path: str) -> list[dict]:
    """Extract text + tables page by page."""
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text (layout-aware)
            text = page.extract_text()
            
            # Extract tables
            tables = page.extract_tables()
            
            # Extract images
            images = page.images
            
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "tables": tables,
                "image_count": len(images),
            })
    
    return pages


def extract_specific_table(pdf_path: str, page_num: int, table_settings: dict = None):
    """Extract a specific table with custom settings."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.extract_tables(table_settings=table_settings or {})
        return tables[0] if tables else None
```

pdfplumber is **layout-aware** and **best for tables** — multi-column papers come out correctly.

---

## 4. Unstructured.io — The Generalist

Unstructured.io handles the broadest range of document types. It auto-detects layout, extracts tables, handles images, and supports OCR.

```bash
pip install unstructured[all-docs]  # install all dependencies
```

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


def extract_with_unstructured(pdf_path: str) -> list[dict]:
    """Extract structured elements with Unstructured."""
    
    elements = partition_pdf(
        filename=pdf_path,
        # Strategy for extracting text
        strategy="hi_res",  # high resolution = better layout detection
        # Extract images
        extract_images_in_pdf=True,
        # Infer tables
        infer_table_structure=True,
        # Languages
        languages=["en"],
        # Chunking
        chunking_strategy=chunk_by_title,
        # Include page numbers
        include_page_breaks=True,
    )
    
    # Convert to dicts with metadata
    chunks = []
    for elem in elements:
        chunks.append({
            "type": elem.category,  # "Title", "NarrativeText", "Table", etc.
            "text": elem.text,
            "page": elem.metadata.page_number,
            "image_path": elem.metadata.image_path if hasattr(elem.metadata, "image_path") else None,
        })
    
    return chunks


def extract_only_tables(pdf_path: str) -> list[dict]:
    """Extract only tables from a PDF."""
    
    tables = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        # Skip narrative extraction
        include_page_breaks=False,
    )
    
    return [elem for elem in tables if elem.category == "Table"]
```

Unstructured.io's `hi_res` strategy uses **Detectron2** for layout detection. It's slow (1-2 pages per second) but accurate.

---

## 5. Marker — Clean Markdown Conversion

Marker (released 2024) converts PDFs to clean markdown with embedded images:

```bash
pip install marker-pdf
```

```bash
# CLI usage
marker_single /path/to/pdf.pdf --output_dir /output/
```

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF to markdown using Marker."""
    
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    
    rendered = converter(pdf_path)
    
    # Extract markdown text, images
    markdown_text = rendered.markdown
    images = rendered.images  # {filename: PIL Image}
    
    # Save images
    for filename, img in images.items():
        img.save(f"/output/images/{filename}")
    
    return markdown_text
```

Marker is fast (5-10× faster than Unstructured.io hi_res) and produces clean markdown. Good for documents with equations, tables, and figures.

---

## 6. OCR for Scanned Documents

For scanned PDFs (image-only), you need OCR before any extraction.

### 6.1 Tesseract

```bash
# Install Tesseract
apt-get install tesseract-ocr
pip install pytesseract
```

```python
import pytesseract
from PIL import Image


def ocr_image(image_path: str) -> str:
    """OCR a single image."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="eng")
    return text


def ocr_pdf(pdf_path: str) -> str:
    """OCR a scanned PDF (image-only)."""
    images = convert_pdf_to_images(pdf_path)  # PyMuPDF
    full_text = ""
    for image in images:
        full_text += pytesseract.image_to_string(image) + "\n\n"
    return full_text
```

### 6.2 PaddleOCR (better for non-English)

```bash
pip install paddleocr paddlepaddle
```

```python
from paddleocr import PaddleOCR


def ocr_with_paddle(image_path: str) -> list[tuple]:
    """OCR with PaddleOCR (better for Chinese, Japanese, etc.)."""
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # or "ch", "japan", etc.
    result = ocr.ocr(image_path)
    
    # Returns list of (box, (text, confidence))
    return [(line[0], line[1]) for line in result]


def ocr_pdf_paddle(pdf_path: str) -> str:
    """OCR a scanned PDF with PaddleOCR."""
    images = convert_pdf_to_images(pdf_path)
    full_text = ""
    
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    for image in images:
        result = ocr.ocr(image)
        for line in result:
            full_text += line[1][0] + "\n"
        full_text += "\n\n"
    
    return full_text
```

PaddleOCR is faster and more accurate than Tesseract for non-English documents.

---

## 7. Table Extraction with TableTransformer

For complex tables in PDFs, use **TableTransformer** from Microsoft:

```bash
pip install transformers[torch] timm
```

```python
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import torch
from PIL import Image


class TableExtractor:
    """Extract tables using TableTransformer."""
    
    def __init__(self):
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection",
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection",
        )
    
    def detect_tables(self, image: Image.Image) -> list[dict]:
        """Detect tables in a page image."""
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.7,
            target_sizes=target_sizes,
        )[0]
        
        tables = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            tables.append({
                "bbox": box.tolist(),
                "score": float(score),
                "label": self.model.config.id2label[int(label)],
            })
        
        return tables
    
    def extract_table_image(self, image: Image.Image, bbox: list) -> Image.Image:
        """Crop the table region from the page."""
        x0, y0, x1, y1 = bbox
        return image.crop((int(x0), int(y0), int(x1), int(y1)))
```

Then OCR the cropped table image to extract the actual data:

```python
def extract_table_data(pdf_path: str, page_num: int) -> list[dict]:
    """Full table extraction pipeline."""
    
    images = convert_pdf_to_images(pdf_path)
    page_image = images[page_num - 1]
    
    extractor = TableExtractor()
    tables = extractor.detect_tables(page_image)
    
    all_tables_data = []
    for table_info in tables:
        # Crop table region
        table_image = extractor.extract_table_image(
            page_image,
            table_info["bbox"],
        )
        
        # OCR the cropped table
        data = pytesseract.image_to_data(
            table_image,
            output_type=pytesseract.Output.DATAFRAME,
        )
        
        # Group text by row/column
        structured = structure_ocr_data(data)
        
        all_tables_data.append({
            "bbox": table_info["bbox"],
            "confidence": table_info["score"],
            "data": structured,
        })
    
    return all_tables_data
```

---

## 8. Real-World Example — Legal Document Pipeline

A complete legal document ingestion pipeline:

```python
from typing import Iterator
from pathlib import Path


def process_legal_document(pdf_path: str) -> dict:
    """Process a legal document with full structure."""
    
    # 1. Detect if scanned
    is_scanned = detect_if_scanned(pdf_path)
    
    if is_scanned:
        # Scanned: OCR first, then extract
        pages = []
        for image in convert_pdf_to_images(pdf_path):
            ocr_text = pytesseract.image_to_string(image)
            tables = extract_tables_from_image(image)
            pages.append({
                "text": ocr_text,
                "tables": tables,
                "page_number": len(pages) + 1,
            })
    else:
        # Digital: use Unstructured.io
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
        )
        
        # Group by page
        pages = group_by_page(elements)
    
    # 2. Extract structured data
    document = {
        "file_name": Path(pdf_path).name,
        "page_count": len(pages),
        "is_scanned": is_scanned,
        "pages": pages,
        "tables": [t for p in pages for t in p["tables"]],
    }
    
    # 3. Compute document metadata
    document["word_count"] = sum(len(p["text"].split()) for p in pages)
    document["table_count"] = len(document["tables"])
    
    return document


def detect_if_scanned(pdf_path: str) -> bool:
    """Heuristic: a PDF is scanned if the text layer is empty."""
    doc = fitz.open(pdf_path)
    total_text = sum(len(page.get_text()) for page in doc)
    doc.close()
    return total_text < 100  # Less than 100 chars = likely scanned
```

---

## 9. Case Study — Airbnb Listing Documents

For your **StayBot** portfolio project: Airbnb often provides host documents (PDFs of rules, house manuals, check-in instructions) that are scanned or have unusual layouts.

```python
def extract_staybot_listing_documents(doc_path: str) -> list[dict]:
    """Extract content from Airbnb listing PDFs."""
    
    # Detect document type
    doc_type = detect_listing_doc_type(doc_path)  # "rules", "manual", "house"
    
    if doc_type == "rules":
        # Use Unstructured.io for tables (rules often have tables)
        chunks = extract_with_unstructured(doc_path)
    elif doc_type == "manual":
        # Use Marker for clean markdown (manuals have mixed content)
        chunks = convert_pdf_to_markdown(doc_path).split("\n## ")
    elif doc_type == "scanned_checkin":
        # Use OCR
        chunks = ocr_pdf(doc_path).split("\n\n")
    
    return [{
        "text": chunk,
        "doc_type": doc_type,
        "source": Path(doc_path).name,
    } for chunk in chunks]
```

---

## 10. Antipatterns

### 10.1 Antipattern 1: One-size-fits-all parsing

```python
# ❌ Same parser for all PDFs
def parse_any_pdf(pdf_path: str):
    return extract_text_simple(pdf_path)

# ✅ Choose parser based on document type
def parse_smart(pdf_path: str):
    if is_scanned(pdf_path):
        return ocr_pdf(pdf_path)
    elif has_complex_tables(pdf_path):
        return extract_with_unstructured(pdf_path)
    else:
        return extract_text_simple(pdf_path)
```

### 10.2 Antipattern 2: Ignoring images

```python
# ❌ Extract text only, ignore images
chunks = extract_text_simple(pdf_path)

# ✅ Extract images too (for diagrams, figures)
chunks = extract_with_unstructured(pdf_path, extract_images_in_pdf=True)

# Combine text and image embeddings (multimodal RAG)
for chunk in chunks:
    if chunk["image_path"]:
        # Generate image embedding via CLIP
        image_embedding = clip_model.encode(chunk["image_path"])
        chunk["embedding"] = image_embedding
```

### 10.3 Antipattern 3: Naive table extraction

```python
# ❌ Tables as plain text
table_text = " ".join(cell for row in table for cell in row)
# "Name Age Alice 30 Bob 25" - meaning lost

# ✅ Tables as structured data
table_data = pd.DataFrame(table[1:], columns=table[0])
table_markdown = table_data.to_markdown()
# | Name  | Age |
# |-------|-----|
# | Alice | 30  |
# | Bob   | 25  |
```

### 10.4 Antipattern 4: No OCR fallback

```python
# ❌ Digital-only parsing
text = extract_text_simple(pdf)
chunks.append({"text": text})  # Empty for scanned PDFs!

# ✅ Detect scanned PDFs and OCR
if not extract_text_simple(pdf).strip():
    text = ocr_pdf(pdf)
    chunks.append({"text": text})
```

### 10.5 Antipattern 5: Losing layout context

```python
# ❌ Text extracted in arbitrary order
text = extract_text_simple(pdf)
# Footnote from page 3 may end up before main text

# ✅ Preserve layout / use Unstructured.io for layout awareness
chunks = extract_with_unstructured(pdf)
# Each chunk includes page number, section, position
```

---

## 🎯 Key Takeaways

- PDFs are render descriptions, not documents — extraction requires care.
- **PyMuPDF** is fast for clean text extraction (10-100× faster than pdfplumber).
- **pdfplumber** is the best for tables and layout-aware extraction.
- **Unstructured.io** is the generalist — handles most document types, including tables and images.
- **Marker** is the cleanest markdown converter for equations, tables, and figures.
- **OCR with Tesseract/PaddleOCR** is required for scanned PDFs.
- **TableTransformer** extracts tables from image-based PDFs with high accuracy.
- Choose the parser based on document type; don't use one-size-fits-all.
- Always preserve layout context (page numbers, sections) for better retrieval.
- Combine text extraction with image extraction for multimodal RAG.
- Avoid ignoring images, naive table extraction, no OCR fallback, losing layout.

## References

- PyMuPDF — [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/)
- pdfplumber — [github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber)
- Unstructured.io — [unstructured.io](https://unstructured.io/)
- Marker — [github.com/datalab-to/marker](https://github.com/datalab-to/marker)
- Tesseract OCR — [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
- PaddleOCR — [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- TableTransformer — [huggingface.co/microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection)
- [[06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/05 - Vision, Audio, and Multimodal Transformers|06/16/05 Vision, Audio, and Multimodal Transformers]]
- [[06 - Large Language Models/12 - Production RAG|Production RAG]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/02 - Visual Document Retrieval - ColPali and Beyond|Note 02 — ColPali]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/05 - Capstone - Production Multimodal RAG|Note 05 — Capstone]]