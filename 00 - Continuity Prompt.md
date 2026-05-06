# Continuity Prompt for Next Kimi Session

Copy and paste this entire file into a new Kimi session to continue the project without losing context.

---

## Project Context

We are building an **Obsidian vault** inside `C:\Users\Leito\Documents\Learning`, which is a **Git repository** connected to `https://github.com/Leito2/Learning.git`. The vault contains compressed markdown courses for an **AI/ML Engineer** learning path. The vault is also used to document portfolio projects for landing a first job as an ML/AI Engineer.

**Current Goal:** Build the vault into a comprehensive, job-ready knowledge base. All **new** courses and notes are written in **English** with deep theoretical and practical depth. The user is looking for their first ML/AI Engineer job.

---

## Vault Structure

```
Learning/
├── .obsidian/                          # Obsidian configuration
├── Welcome.md                          # Main vault index (Spanish)
│
├── Curso Markdown/                     # Complete (5 notes)
├── Curso SQL con PostgreSQL/           # Complete (7 notes + assets)
│
├── Advanced Python/                    # Complete (62 notes + images)
│   ├── 01 - Python Basico/             # 13 notes
│   ├── 02 - Python Intermedio/         # 13 notes
│   ├── 03 - Python Avanzado/           # 13 notes
│   ├── 04 - Librerias Basicas de Python/# 8 notes
│   └── 05 - Librerias Especificas/     # 7 notes
│
├── Software Engineering/               # Active area
│   ├── 01 - Docker Profesional/        # Complete (7 notes)
│   ├── extra/                          # NEW (6 notes - English)
│   │   ├── 00 - Welcome to SE Extra.md
│   │   ├── 01 - FastAPI for ML.md
│   │   ├── 02 - System Design for ML.md
│   │   ├── 03 - Testing in ML Systems.md
│   │   ├── 04 - CI-CD for ML.md
│   │   └── 05 - Rust for ML Infra.md
│   └── projects/                       # NEW (6 notes - English)
│       ├── 00 - Project Planning Guide for SE.md
│       ├── 01 - FastAPI for ML - Project Guide.md
│       ├── 02 - System Design for ML - Project Guide.md
│       ├── 03 - Testing in ML Systems - Project Guide.md
│       ├── 04 - CI-CD for ML - Project Guide.md
│       └── 05 - Rust for ML Infra - Project Guide.md
│
├── ML and IA Engineering/              # Main vault for ML courses
│   ├── Welcome.md                      # Sub-vault index (Spanish + English sections)
│   ├── 00 - Indice Maestro de Cursos.md # Master course index
│   │
│   ├── M00 - Fundamentos de Ingenieria/           # Complete + images (23 notes)
│   ├── M01 - Deep Learning y Computer Vision/     # Complete + images (18 notes)
│   ├── M02 - Large Language Models/               # Complete + images (30 notes)
│   ├── M03 - AI Agents y Agentic Systems/         # Complete + images (22 notes)
│   ├── M04 - NLP Avanzado/                        # Complete + images (17 notes)
│   ├── M05 - MLOps y Produccion/                  # Complete + images (24 notes)
│   ├── M06 - Cloud, Infra y Backend/              # Complete + images (24 notes)
│   ├── M07 - Research y Ciencia de Datos/         # Complete + images (24 notes)
│   ├── M08 - Producto, Negocio y Open Source/     # Complete + images (18 notes)
│   │
│   ├── extra/                          # NEW (8 notes - English)
│   │   ├── 00 - Welcome to ML Extra Courses.md
│   │   ├── 01 - Kaggle Competitions.md
│   │   ├── 02 - End-to-End ML Project.md
│   │   ├── 03 - Fine-Tuning LLMs.md
│   │   ├── 04 - Production RAG System.md
│   │   ├── 05 - Computer Vision Pipeline.md
│   │   ├── 06 - Advanced MLOps.md
│   │   └── 07 - Paper Reproduction.md
│   └── projects/                       # NEW (8 notes - English)
│       ├── 00 - Project Planning Guide for ML.md
│       ├── 01 - Kaggle Competitions - Project Guide.md
│       ├── 02 - End-to-End ML Project - Project Guide.md
│       ├── 03 - Fine-Tuning LLMs - Project Guide.md
│       ├── 04 - Production RAG System - Project Guide.md
│       ├── 05 - Computer Vision Pipeline - Project Guide.md
│       ├── 06 - Advanced MLOps - Project Guide.md
│       └── 07 - Paper Reproduction - Project Guide.md
│
├── Transversal Skills/                 # NEW (4 notes - English)
│   ├── 00 - Welcome to Transversal Skills.md
│   ├── 01 - Technical English.md
│   ├── 02 - Technical Communication and Storytelling.md
│   └── 03 - Technical Leadership.md
│
├── Software Engineering/
│   ├── Go Engineering/                 # ✅ COMPLETE (49 notes - English)
│   │   ├── 00 - Welcome to Go Engineering.md
│   │   ├── 01 - Go Fundamentals/       # 7 notes
│   │   ├── 02 - Go for Cloud Native/   # 7 notes
│   │   ├── 03 - Microservices with Go/ # 7 notes
│   │   ├── 04 - DevSecOps and CLI Tools/ # 7 notes
│   │   ├── 05 - Local AI with Go/      # 7 notes
│   │   ├── 06 - Go for ML Backend/     # 7 notes
│   │   ├── projects/                   # 6 notes
│   │   └── extra/                      # 7 notes
│   │
│   └── Rust Engineering/               # ✅ COMPLETE (49 notes - English)
│       ├── 00 - Welcome to Rust Engineering.md
│       ├── 01 - Rust Fundamentals/     # 7 notes
│       ├── 02 - Advanced Rust/         # 7 notes
│       ├── 03 - Rust for Data Engineering/ # 7 notes
│       ├── 04 - Rust for ML and AI/    # 7 notes
│       ├── 05 - WebAssembly and Edge AI/ # 7 notes
│       ├── 06 - Agentic AI with Rust/  # 7 notes
│       ├── projects/                   # 6 notes
│       └── extra/                      # 7 notes
```

**Current total: ~483 notes created** (294 original + 42 SE/ML/Transversal + 49 Go + 49 Rust + 49 restructuring)

**Git Checkpoints:**
- `6de9e0e` — Rust Engineering complete (revert point)
- `f17ca9b` — Go Engineering complete

---

## Language Policy

- **Existing modules (M00-M08, Advanced Python, Docker, Markdown, SQL):** Written in Spanish. **DO NOT modify** unless explicitly requested.
- **All new content (SE extra/projects, ML extra/projects, Transversal, Go Engineering, future Rust Engineering):** Written entirely in **English**.
- **Welcome.md and indices:** May contain both Spanish and English sections to bridge old and new content.

---

## Note Style and Format (New English Content)

Each course note follows this exact structure:

### For Full Course Notes (01-06 within each course folder)

```markdown
# <Title with emoji>

## Introduction
2-3 paragraphs explaining relevance to ML/AI engineering. Use [[...]] internal links.

## 1. <Section Name>
Deep conceptual explanations with:
- Bullet points for key ideas
- "Real case: <specific ML/AI company example>" at least once
- ⚠️ Warnings about common mistakes
- 💡 Tips and mnemonic rules

## 2. <Section Name>
Same depth. Include comparison tables where useful.

## 3. <Section Name>
At least 2-3 images/diagrams per note:
- Mermaid diagrams (flowcharts, decision trees, timelines)
- Wikimedia Commons image URLs: `![description](https://upload.wikimedia.org/wikipedia/commons/thumb/...)`

## 4. <Section Name>
Python/Go code blocks when applicable.

## 5. <Section Name> (if needed)

---

## 📦 Compression Code
A complete script summarizing the entire topic.

## 🎯 Documented Project

### Description
2-3 sentences about a real-world project.

### Functional Requirements
1. ...
2. ...
3. ...
4. ...
5. ...

### Main Components
- ...
- ...
- ...

### Success Metrics
- ...
- ...
- ...

### References
- <Paper or library reference>
- <Tool or framework reference>
```

### For Project Guide Notes (projects/ folder)

```markdown
# <Title with emoji>

## Overview
Why this matters for a first job.

## Prerequisites

## Learning Objectives

## Official Resources & Links
| Resource | Type | URL | Why It Matters |
|----------|------|-----|----------------|
(At least 5 links per note)

## Architecture & Planning
Mermaid diagram + key decisions.

## Step-by-Step Implementation Guide
5-10 numbered steps with what, why, code, expected output.

## Guide Class / Example
Complete copy-pasteable code.

## Common Pitfalls & Checklist
⚠️ At least 3 mistakes
✅ Checklist table

## Deployment & Portfolio Integration
How to deploy and present for recruiters.

## Next Steps
[[...]] internal links.
```

### Formatting Rules
- Use `[[...]]` for internal Obsidian links.
- File names: `## - Descriptive Name.md` (with spaces).
- **IMPORTANT:** Do NOT use `/` in file names. Use `-` instead (Windows restriction).
- One H1 (`#`) per note only.
- Blank lines between paragraphs.
- Markdown tables for comparisons.
- Emojis for sections: 📦 (code), 🎯 (project), 💡 (tip), ⚠️ (warning).

---

## CRITICAL RULE: Subagent (Task Tool) Usage

### Identified Problem
Subagents sometimes complete work (files created on disk) but do not return a result message, appearing "hung." This happens when:
- More than 3 subagents are launched in parallel.
- A subagent must create more than 7 notes at once.
- Prompts are extremely long.

### MANDATORY Workflow for Creating Courses

1. **ALWAYS** use subagents to create batches of notes (creating one by one consumes too much main model context).
2. **Maximum 2 subagents in parallel per batch.**
3. **Maximum 7 notes per subagent.** If a course has more notes, split into 2 sequential subagents (verify the first's filesystem before launching the second).
4. **IMMEDIATELY** after launching subagents, verify the filesystem with bash:
   ```powershell
   Get-ChildItem -Recurse -File "C:\Users\Leito\Documents\Learning\PATH\TO\COURSE" | ForEach-Object { $_.FullName }
   ```
5. If files exist, consider the task completed even if the subagent has not reported anything.
6. If files are missing, re-launch ONLY the missing ones.
7. **The filesystem is the single source of truth.**

---

## Upcoming Work Queue

### 1. Go Engineering (✅ COMPLETE)
### 2. Rust Engineering (✅ COMPLETE)

### 3. Deep Content Expansion — Go (PLANNED — Priority 1)
Expand Go Engineering with deeper ML content. Create additional notes or expand existing ones to cover:

**Gorgonia (Go Deep Learning Framework):**
- Gorgonia architecture: tensors, computation graphs, automatic differentiation
- Building neural networks from scratch in Go
- Gorgonia vs TensorFlow vs PyTorch (when to use Go for DL)
- GPU support via CUDA bindings
- Real case: How Gorgonia powers ML in Go-only environments (no Python allowed)
- Include Go code: MLP, CNN, RNN built with Gorgonia

**LocalAI (Local LLM Server in Go):**
- LocalAI architecture: Go backend, model backends (llama.cpp, whisper.cpp, stable diffusion)
- API compatibility with OpenAI
- Hardware acceleration: CUDA, Metal, Vulkan, CPU
- Model formats: GGUF, GGML, ONNX, TensorRT
- Real case: How LocalAI enables private AI for enterprises
- Include Go code: LocalAI API client, custom backend

**Go ML Ecosystem Deep Dive:**
- gonum: numerical computing, statistics, optimization
- gorgonia.org/tensor: NDArray operations
- gotch: PyTorch bindings for Go
- Go + ONNX: onnxruntime-go for cross-platform inference
- Go + HuggingFace: hfgo for model downloading and inference
- Go + MLflow: tracking experiments from Go

### 4. Deep Content Expansion — Rust (PLANNED — Priority 2)
Expand Rust Engineering with deeper ML content:

**Polars Deep Dive:**
- Polars internals: query optimization, lazy execution, memory mapping
- Custom expressions and UDFs in Polars
- Polars vs DataFusion vs DuckDB (Rust-based alternatives)
- Streaming API for datasets larger than RAM

**Candle Advanced Patterns:**
- Custom model architectures in Candle
- Candle + CUDA: GPU training and inference
- Model quantization with Candle
- Candle vs PyTorch Rust (tch-rs) comparison

**PyO3 Advanced Patterns:**
- Complex type conversions (nested structs, enums)
- Async PyO3 with tokio
- PyO3 + Rayon for parallel Python extensions
- Building production Python wheels with maturin

**Vector Search Deep Dive:**
- HNSW algorithm implementation
- Product Quantization (PQ) and IVF indexes
- Hybrid search: sparse + dense vectors
- Qdrant clustering and sharding

### 5. Portfolio Web Redesign (PLANNED)
Update `leito2.github.io` from basic course index to a professional ML/AI Engineer portfolio showcasing:
- GitHub repos with real projects (Go + Rust ML tools)
- Kaggle profile and competition results
- HuggingFace models (Candle, Ollama integrations)
- Technical blog posts about Go/Rust for ML
- Interactive demo (WASM-based ML model in browser)

---

## Deep Content Reference (Go & Rust for ML)

This section provides reference material for expanding Go and Rust tracks with deeper ML content. Use these when creating new notes or expanding existing ones.

---

### 🐹 Go Deep Content: Gorgonia

**What is Gorgonia?**
Gorgonia is a library for machine learning in Go, similar to TensorFlow/PyTorch but native to Go. It provides:
- Tensor operations (N-D arrays)
- Automatic differentiation (autodiff)
- Neural network building blocks (layers, activations, losses)
- GPU support via CUDA

**Key Concepts to Cover:**
```
Gorgonia Architecture:
├── gorgonia.org/tensor        # NDArray operations (like NumPy)
├── gorgonia.org/gorgonia      # Computational graphs + autodiff
├── gorgonia.org/genfuncs      # Generic functions for tensors
└── gorgonia.org/vm            # Virtual machine for graph execution
```

**API Comparison:**
| Operation | NumPy | PyTorch | Gorgonia |
|-----------|-------|---------|----------|
| Matrix multiply | `np.dot(A, B)` | `torch.matmul(A, B)` | `gorgonia.MatMul(A, B)` |
| ReLU | `np.maximum(0, x)` | `torch.relu(x)` | `gorgonia.Rectify(x)` |
| Softmax | `scipy.special.softmax(x)` | `torch.softmax(x, dim)` | `gorgonia.SoftMax(x)` |
| Backward | N/A (manual) | `loss.backward()` | `gorgonia.Grad(loss)` |

**When to Use Gorgonia:**
- Go-only environments (no Python allowed)
- Embedded systems where Go is the primary language
- Building custom ML frameworks
- Prototyping before porting to PyTorch

**Real Cases:**
- CockroachDB: uses gonum for query optimization
- Various fintech companies: Go-first ML pipelines

---

### 🐹 Go Deep Content: LocalAI

**What is LocalAI?**
LocalAI is an open-source, drop-in replacement for OpenAI API that runs locally. Written in Go by Mudler (Ettore Di Giacinto). It enables:
- Running LLMs, image generation, audio transcription, and embedding models locally
- OpenAI-compatible API (use existing OpenAI SDKs with local models)
- Multiple backends: llama.cpp, whisper.cpp, stable-diffusion.cpp, piper TTS
- Hardware acceleration: CUDA, Metal, Vulkan, CPU

**Architecture:**
```
LocalAI Architecture:
├── API Layer (Go)             # OpenAI-compatible REST API
├── Backend Manager (Go)       # Manages model backends
├── Backends (C/C++/Go):
│   ├── llama.cpp              # LLM inference (LLama, Mistral, etc.)
│   ├── whisper.cpp            # Audio transcription
│   ├── stable-diffusion.cpp   # Image generation
│   ├── piper                  # Text-to-speech
│   └── sentencetransformers   # Embeddings
├── Model Gallery              # One-click model download
└── Protobuf gRPC             # Communication with backends
```

**Key APIs:**
| Endpoint | Function | Backends |
|----------|----------|----------|
| `/v1/chat/completions` | Chat with LLM | llama.cpp |
| `/v1/completions` | Text completion | llama.cpp |
| `/v1/embeddings` | Generate embeddings | sentence-transformers |
| `/v1/images/generations` | Generate images | stable-diffusion |
| `/v1/audio/transcriptions` | Transcribe audio | whisper.cpp |

**Configuration (YAML):**
```yaml
name: llama-3-8b
backend: llama
parameters:
  model: llama-3-8b-Q4_K_M.gguf
  temperature: 0.7
  max_tokens: 2048
context_size: 8192
f16: true
threads: 8
gpu_layers: 35  # Offload to GPU
```

**Real Cases:**
- Enterprises deploying AI without sending data to cloud
- Hobbyists running AI on Raspberry Pi, old laptops
- Developers building local-first AI applications

---

### 🐹 Go Deep Content: ML Ecosystem

**gonum (Numerical Computing):**
```go
import (
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/gonum/optimize"
)

// Matrix operations
A := mat.NewDense(3, 3, []float64{1,2,3,4,5,6,7,8,9})
B := mat.NewDense(3, 3, nil)
B.Mul(A, A.T())  // Matrix multiply

// Statistics
data := []float64{1, 2, 3, 4, 5}
mean := stat.Mean(data, nil)
stddev := stat.StdDev(data, nil)
```

**gotch (PyTorch Bindings):**
```go
import "github.com/sugarme/gotch"

// Create tensor
ts := gotch.NewTensor([][]float32{{1, 2}, {3, 4}})

// Neural network
model := nn.NewLinear(vs, 784, 10, nil)
output := model.Forward(&input)
loss := output.CrossEntropyForLogits(&target)
```

**hfgo (HuggingFace for Go):**
```go
import "github.com/rhasspy-ai/hfgo"

// Download model
model := hfmodel.New("bert-base-uncased", nil)
tokenizer := model.Tokenizer()
tokens := tokenizer.Encode("Hello world")
```

---

### 🦀 Rust Deep Content: Polars Internals

**Polars Architecture:**
```
Polars Execution Model:
├── DataFrame (Eager)          # Immediate execution
├── LazyFrame (Lazy)           # Query planning + optimization
│   ├── Logical Plan           # Abstract query tree
│   ├── Optimizer              # Predicate pushdown, projection pushdown, common subexpression elimination
│   └── Physical Plan          # Optimized execution plan
├── ChunkedArray               # Columnar storage (Apache Arrow)
├── Series                     # 1D typed array
└── DataType                   # Int8-64, Float32/64, Utf8, Boolean, Date, Datetime, List, Struct
```

**Advanced Features to Cover:**

1. **Lazy Evaluation Optimization:**
```rust
// Polars optimizes BEFORE execution
let result = LazyCsvReader::new("data.csv")?
    .filter(col("age").gt(lit(18)))           // Predicate pushdown: filter at read time
    .select([col("name"), col("salary")])      // Projection pushdown: only read needed columns
    .groupby([col("department")])
    .agg([col("salary").mean()])
    .collect()?;                                // Optimization happens here
```

2. **Memory Mapping for Large Files:**
```rust
// Memory-map a Parquet file (zero-copy read)
let df = LazyParquetReader::new("huge_file.parquet")?
    .with_memory_map(true)    // Memory-map instead of loading
    .collect()?;
```

3. **Streaming for Out-of-Core Processing:**
```rust
// Process files larger than RAM
let df = LazyCsvReader::new("100GB_file.csv")?
    .with_streaming(true)     // Process in chunks
    .collect()?;
```

**Performance Formulas:**
```
Polars_Speedup ≈ 10-100× vs Pandas (single-threaded)
Polars_Memory ≈ 0.1× Pandas (Arrow format)
Parallel_Speedup ≈ min(Cores, Chunks) for embarrassingly parallel ops
```

---

### 🦀 Rust Deep Content: Candle Advanced

**Candle Architecture:**
```
Candle ML Framework:
├── candle-core                 # Tensor operations, autodiff
│   ├── CpuDevice              # CPU backend
│   ├── CudaDevice             # NVIDIA GPU (via cudarc)
│   ├── MetalDevice            # Apple Silicon GPU
│   └── DType                  # F16, F32, F64, BF16
├── candle-nn                   # Neural network layers
│   ├── Linear, Conv2d, LSTM
│   ├── BatchNorm, LayerNorm
│   └── Activations (ReLU, GELU, SiLU)
├── candle-transformers         # Pre-built model architectures
│   ├── Llama, Mistral, Phi
│   ├── Stable Diffusion
│   ├── Whisper
│   └── BERT
└── candle-wasm                 # WebAssembly support
```

**Custom Model Example:**
```rust
use candle_core::{Tensor, DType, Device, Result};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

struct MLP {
    l1: Linear,
    l2: Linear,
}

impl MLP {
    fn new(vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: Linear::new(vs.get((128, 784), "l1.weight")?, Some(vs.get(128, "l1.bias")?)),
            l2: Linear::new(vs.get((10, 128), "l2.weight")?, Some(vs.get(10, "l2.bias")?)),
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.l1)?.gelu()?.apply(&self.l2)
    }
}
```

**GPU Acceleration:**
```rust
// Candle automatically uses GPU if available
let device = Device::cuda_if_available(0)?;
let tensor = Tensor::randn(0f32, 1f32, (1024, 1024), &device)?;
```

---

### 🦀 Rust Deep Content: PyO3 Advanced Patterns

**Complex Type Conversion:**
```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// Rust struct ↔ Python dict
#[pyclass]
#[derive(Clone)]
struct MLConfig {
    #[pyo3(get, set)]
    learning_rate: f64,
    #[pyo3(get, set)]
    batch_size: usize,
    #[pyo3(get, set)]
    model_name: String,
}

#[pymethods]
impl MLConfig {
    #[new]
    fn new(lr: f64, bs: usize, name: String) -> Self {
        Self { learning_rate: lr, batch_size: bs, model_name: name }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("learning_rate", self.learning_rate)?;
        dict.set_item("batch_size", self.batch_size)?;
        dict.set_item("model_name", &self.model_name)?;
        Ok(dict.into())
    }
}
```

**Async PyO3 with Tokio:**
```rust
use pyo3::prelude::*;
use tokio::runtime::Runtime;

#[pyfunction]
fn async_predict(py: Python<'_>, input: Vec<f64>) -> PyResult<PyObject> {
    let rt = Runtime::new()?;
    py.allow_threads(|| {
        rt.block_on(async {
            // Async Rust inference
            let result = model.predict_async(input).await?;
            Ok(result)
        })
    })
}
```

---

### 🦀 Rust Deep Content: Vector Search Algorithms

**HNSW (Hierarchical Navigable Small World):**
```
HNSW Index Structure:
├── Layer 0 (dense)           # All vectors connected
├── Layer 1 (sparse)          # Subset of vectors
├── Layer 2 (sparser)         # Fewer connections
└── Layer M (top)             # 1-2 vectors

Search Algorithm:
1. Start at top layer entry point
2. Greedy search: move to neighbor closest to query
3. When no improvement, drop to next layer
4. Repeat until layer 0
5. Return k nearest neighbors

Parameters:
- M: max connections per node (16-64)
- efConstruction: build-time search width (100-500)
- efSearch: query-time search width (10-500)
```

**Formulas:**
```
Recall@k = |Relevant_in_top_k| / |Total_Relevant|
Precision@k = |Relevant_in_top_k| / k
MRR = (1/|Q|) × Σ(1/rank_i)  # Mean Reciprocal Rank
HNSW_Query_Time = O(log(N))   # N = total vectors
```

---

### 🔗 Official Links for Deep Content

| Resource | URL | Use For |
|----------|-----|---------|
| Gorgonia | https://gorgonia.org/ | Go deep learning |
| LocalAI | https://localai.io/ | Local LLM server |
| gonum | https://gonum.org/ | Go numerical computing |
| gotch | https://github.com/sugarme/gotch | Go PyTorch bindings |
| Polars Book | https://pola.rs/polars/book/ | Rust DataFrames |
| Candle Docs | https://huggingface.co/docs/candle | Rust ML framework |
| PyO3 Guide | https://pyo3.rs/ | Rust-Python bindings |
| Qdrant Docs | https://qdrant.tech/documentation/ | Vector database |

---

## Git Commands

```powershell
# Check git status
git -C "C:\Users\Leito\Documents\Learning" status

# Stage, commit, push
git -C "C:\Users\Leito\Documents\Learning" add -A
git -C "C:\Users\Leito\Documents\Learning" commit -m "message"
git -C "C:\Users\Leito\Documents\Learning" push origin master

# Verify course files (USE ALWAYS after subagents)
Get-ChildItem -Recurse -File "C:\Users\Leito\Documents\Learning\PATH\TO\COURSE" | ForEach-Object { $_.FullName }

# Full structure view
Get-ChildItem -Recurse "C:\Users\Leito\Documents\Learning\Software Engineering\Go Engineering"
```

## Repository
- **GitHub:** https://github.com/Leito2/Learning
- **Branch:** master
- **Local path:** C:\Users\Leito\Documents\Learning

---

> **Tip for next session:** Start by verifying the filesystem state of any in-progress work, then proceed with the next batch of subagents following the max-2-parallel, max-7-notes rule.
