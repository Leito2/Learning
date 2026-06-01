# 🦀 Rust for ML Infra

## Introduction

Python dominates machine learning research, but it was never engineered for high-throughput serving, low-latency data pipelines, or memory-safe systems programming. As models grow larger and user expectations shrink to milliseconds, the infrastructure around the model—tokenizers, vector databases, inference servers, feature stores—becomes the bottleneck. Rust offers a compelling alternative: C++-like performance with Python-like ergonomics and compile-time guarantees against memory corruption.

Rust's ownership model eliminates data races and null-pointer dereferences without a garbage collector. This makes it ideal for ML infrastructure that must run for weeks without memory leaks or segmentation faults. In this course, you will learn when to reach for Rust instead of Python, how to expose Rust code to Python via PyO3, and how modern Rust ML frameworks like Candle allow you to run inference without leaving the ecosystem.

Think of Rust not as a replacement for Python training loops, but as a specialized alloy for the hot path of production ML systems.

## 1. Rust Core Principles

Rust's design rests on three pillars relevant to ML infrastructure:

- **Memory safety without GC**: Ownership and borrowing are checked at compile time. There is no unpredictable pause from garbage collection, which is critical for real-time inference servers that must meet strict tail-latency SLAs.
- **Zero-cost abstractions**: High-level iterators, generics, and closures compile down to machine code as efficient as hand-written C. You can write ergonomic data pipeline code without paying a runtime penalty.
- **Fearless concurrency**: The type system prevents data races. If your code compiles, you can safely share state across threads—a stark contrast to Python's GIL-bounded parallelism.

Real case: Polars, a DataFrame library written in Rust, outperforms Pandas by 10–50× on many workloads. Its multithreaded query engine processes terabytes without the memory overhead of Python objects, making it a drop-in replacement for feature engineering pipelines that previously required Spark clusters.

⚠️ **Warning:** Rust's borrow checker is notoriously unforgiving for beginners. Attempting to fight it with excessive `clone()` or `unsafe` blocks defeats the purpose. Invest time in understanding ownership semantics before writing production Rust; otherwise, you will produce slow, idiomatically poor code that is painful to maintain.

💡 **Tip:** Mnemonic for Rust ownership: **O**ne owner, **M**any borrowers, **N**o mutability aliasing. If you remember "OMN", you remember the core rule.

## 2. When Rust Beats Python for ML

Not every ML task benefits from Rust. Use it selectively:

| Task | Python | Rust | Winner |
|---|---|---|---|
| Research & prototyping | Excellent ecosystem | Verbose, slower dev cycle | Python |
| High-throughput inference server | GIL bottleneck, GC pauses | Async, lock-free concurrency | Rust |
| Data preprocessing pipeline | Pandas/Spark mature | Polars/Arrow zero-copy | Rust |
| Embedding store / Vector DB | Bindings only | Native performance | Rust |
| GPU kernel programming | CUDA Python | Rust CUDA (experimental) | Python (for now) |
| WebAssembly deployment | Heavy runtime | Tiny binaries, no GC | Rust |

Rust excels where latency, throughput, or resource efficiency is paramount. A Python FastAPI server might handle 1,000 requests per second; an equivalent Actix-web or Axum server in Rust can handle 100,000+ on the same hardware.

## 3. PyO3 and Python Bindings

You do not need to rewrite your entire stack. PyO3 allows you to write performance-critical Rust functions and call them from Python as native modules.

```rust
use pyo3::prelude::*;

/// Fast batch cosine similarity in Rust
#[pyfunction]
fn batch_cosine(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let mut scores = Vec::with_capacity(a.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        let dot: f32 = av.iter().zip(bv.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = bv.iter().map(|x| x * x).sum::<f32>().sqrt();
        scores.push(dot / (norm_a * norm_b));
    }
    Ok(scores)
}

#[pymodule]
fn rust_ml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_cosine, m)?)?;
    Ok(())
}
```

After compiling with Maturin, you import and use it like any Python package:

```python
from rust_ml import batch_cosine
scores = batch_cosine(user_embeddings, item_embeddings)
```

This pattern is how libraries like `tokenizers` (HuggingFace) and `ruff` achieve their speed: Rust core, Python interface.

## 4. Candle and the Rust ML Ecosystem

Candle is HuggingFace's ML framework written in pure Rust. It enables running LLMs and diffusion models without Python dependencies, making deployment to edge devices and serverless environments radically simpler.

```rust
use candle_core::{Device, Tensor};
use candle_nn::Module;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let data = vec![1.0f32, 2.0, 3.0];
    let tensor = Tensor::new(&data, &device)?;
    let weights = Tensor::new(&vec![0.5f32, 0.5, 0.5], &device)?;
    let output = tensor.mul(&weights)?.sum_all()?;
    println!("Output: {}", output.to_scalar::<f32>()?);
    Ok(())
}
```

Real case: HuggingFace's Candle powers lightweight inference on macOS, Windows, and Linux with a single binary. For ML teams shipping desktop applications or embedded agents, this eliminates the "it works on my conda environment" problem entirely.

The performance gain over equivalent Python can be dramatic:

$$
\text{Speedup} = \frac{T_{\text{python}}}{T_{\text{rust}}}
$$

In vectorized embedding retrieval, speedups of 20–100× are common because Rust avoids Python object overhead and GIL contention.

```mermaid
graph LR
    A[Python Client] -->|PyO3| B[Rust Core Library]
    B --> C[Inference Engine<br/>Candle / ONNX Runtime]
    C --> D[Tokenization<br/>Rust Tokenizers]
    D --> E[Embedding Store<br/>Native Rust]
    E --> F[Arrow / Polars<br/>Zero-copy Data]
    F --> G[Async HTTP Server<br/>Axum / Actix]
    G --> A
```

⚠️ **Warning:** Candle is younger than PyTorch and TensorFlow. Many cutting-edge architectures lack pre-built Rust implementations. Evaluate the ecosystem maturity for your specific model family before committing to a pure Rust inference stack.

![Rust Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Rust_programming_language_black_logo.svg/240px-Rust_programming_language_black_logo.svg.png)

---

## 📦 Compression Code

```rust
// Rust inference microservice template
// Axum + Candle-compatible tensor ops
use axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    score: f32,
    version: String,
}

async fn predict(Json(payload): Json<PredictRequest>) -> Json<PredictResponse> {
    let score: f32 = payload.features.iter().sum::<f32>() / payload.features.len().max(1) as f32;
    Json(PredictResponse {
        score,
        version: "rust-ml-0.1.0".into(),
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/predict", post(predict));
    let addr: SocketAddr = "0.0.0.0:3000".parse().unwrap();
    println!("Listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

