# 🦀 3 - candle-transformers and Pre-built Architectures

## 🎯 Learning Objectives
- Navigate the `candle-transformers` crate and understand its architecture.
- Load and run pre-trained models (BERT, Llama, Whisper) from Hugging Face in Rust.
- Understand how `Config` structs and `VarBuilder` enable safe model deserialization.
- Connect transformer deployment to [[01 - Deep Learning y Computer Vision]] and [[04 - Rust for ML and AI]].

## Introduction

The transformer architecture, introduced by Vaswani et al. in 2017, has become the universal substrate of modern AI. From GPT-4 to Stable Diffusion, nearly every state-of-the-art model relies on attention mechanisms. For engineers, this creates both an opportunity and a challenge: you rarely need to invent a new architecture, but you must be able to load, run, and modify existing ones safely and efficiently.

Python makes this easy with `transformers.AutoModel.from_pretrained()`, but the convenience comes at a cost: hundreds of megabytes of Python libraries, dynamic object instantiation, and implicit state. Candle's `candle-transformers` crate provides Rust-native implementations of popular architectures as plain modules with `Config` structs and `VarBuilder`-based constructors. This note builds on device abstraction from [[02 - GPU Acceleration and Device Abstraction]].

---

## 1. Architecture of candle-transformers

### Design Principles

`candle-transformers` was built with three design goals:

1. **Readability** – Pure Rust with no macro magic, so you can step through the forward pass with a debugger.
2. **Portability** – The same code compiles for CUDA, Metal, and WebAssembly without modification.
3. **Minimality** – No dependency on Python, Conda, or CUDA toolkits at runtime. The crate provides only the model implementation—you bring the tokenizer, the weight files, and the device.

The crate demonstrates a critical design principle: **separation of concerns**. The model code does not know where weights come from. It only knows that `VarBuilder` will provide a tensor of a given shape and name. This decoupling enables hot-swapping between different weight sources (disk, HTTP stream, in-memory buffer) and different quantization levels.

### The Consistent Pattern: Config + VarBuilder + Model

Every model in `candle-transformers` follows the same blueprint:

```mermaid
flowchart LR
    A[config.json] -->|serde_json| B[Config Struct]
    C[model.safetensors] -->|mmap| D[VarBuilder]
    B --> E[Model::load(vb, config)]
    D --> E
    E --> F[Ready Model]
```

1. **`Config`** – A `#[derive(Deserialize)]` struct that mirrors the JSON config from Hugging Face.
2. **`VarBuilder`** – Reads weights from safetensors, lazily and memory-mapped.
3. **`Model`** – A struct implementing `Module`, constructed by `Model::load(vb, config)?`.

❌ **Python habit:** Calling `AutoModel.from_pretrained("bert-base-uncased")` and trusting the magic.
✅ **Candle approach:** Explicitly download config and weights, deserialize the config, construct `VarBuilder`, and load the model. Every step is inspectable and testable.

```rust
use candle_core::{Device, Result};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    // 1. Download artifacts from Hugging Face Hub (cached locally)
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        RepoType::Model, "main".to_string(),
    ));

    // 2. Load config from JSON
    let config: Config = serde_json::from_slice(
        &std::fs::read(repo.get("config.json")?)?
    )?;

    // 3. Memory-map weights (avoids loading 400 MB into RAM)
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[repo.get("model.safetensors")?], DTYPE, &device
        )?
    };

    // 4. Build the model — shapes validated at load time
    let model = BertModel::load(vb, &config)?;

    // 5. Tokenize on the Rust side
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let encoded = tokenizer.encode("Candle is a Rust ML framework.", false)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let ids: Vec<u32> = encoded.get_ids().iter().map(|&id| id as u32).collect();
    let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;

    // 6. Forward pass
    let embeddings = model.forward(&input, None)?;
    let pooled = embeddings.mean(1)?;
    println!("Embedding shape: {:?}", pooled.shape());
    Ok(())
}
```

> 💡 **Mnemonic:** "Config is the blueprint, VarBuilder is the contractor, Model is the house." If the blueprint does not match the materials, construction fails immediately—which is exactly what you want.

⚠️ **Pitfall:** The config and safetensors must come from the **same model revision**. Loading a `bert-base-uncased` config with `bert-large-uncased` weights will panic because vocabulary sizes differ.

### Loading Weights from Byte Buffers (No Filesystem)

For serverless or embedded environments, you may not have a filesystem. `VarBuilder` can also load from in-memory buffers:

```rust
// Download weights as bytes (e.g., from S3 or HTTP)
let weights_bytes: Vec<u8> = fetch_url("https://.../model.safetensors")?;

let vb = VarBuilder::from_buffered_safetensors(
    &weights_bytes,
    candle_core::DType::F32,
    &device,
)?;
let model = BertModel::load(vb, &config)?;
```

This pattern is used by Hugging Face's Inference API to stream model weights directly into GPU memory without writing to disk. It also makes Candle models compatible with `wasm32-unknown-unknown`, where there is no POSIX filesystem.

## 2. Working with Large Language Models (Llama, Mistral)

### Memory-Mapped Weight Loading

LLMs like Llama 7B require ~14 GB in F16. Candle memory-maps the weights so the OS pages them in on demand:

```rust
use candle_transformers::models::llama as llama;
use candle_core::DType;

let config: llama::Config = serde_json::from_slice(&bytes)?;
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(
        &["model-00001-of-00002.safetensors",
          "model-00002-of-00002.safetensors"],
        DType::F16,
        &device,
    )?
};
let model = llama::LlamaModel::load(vb, &config)?;
```

The `VarBuilder` accepts multiple shards. The weight names (e.g., `model.layers.0.self_attn.q_proj.weight`) are matched against the safetensors keys using the prefix scoping system (`vs.pp("model.layers.0.self_attn")`).

### LLM Inference Loop (Text Generation)

For autoregressive models like Llama, inference requires a loop:

```rust
fn generate(model: &llama::LlamaModel, tokenizer: &Tokenizer, prompt: &str, max_len: usize, device: &Device) -> Result<String> {
    let encoded = tokenizer.encode(prompt, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut tokens: Vec<u32> = encoded.get_ids().iter().map(|&id| id as u32).collect();

    for _ in 0..max_len {
        let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let next_token = logits.squeeze(0)?
            .narrow(0, tokens.len() - 1, 1)?
            .argmax(candle_core::D::Minus1)?
            .to_scalar::<u32>()?;
        tokens.push(next_token);
        if next_token == tokenizer.token_to_id("</s>") { break; }
    }

    tokenizer.decode(&tokens, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}
```

The key Candle-specific details: the KV cache must be managed manually (passing `layer_idx` to `forward`), and token generation is a loop that reshapes the logits tensor at each step.

**Caso real:** Notion migrated their semantic search from a 2 GB Python container running sentence-transformers to a 40 MB Rust binary using `candle-transformers`. The binary starts in under 100 ms, embeds documents on CPU, and requires no dependency management beyond `cargo build`.

### Model Zoo Overview

| Model | Candle Module | Key Features |
|-------|--------------|--------------|
| BERT | `bert::BertModel` | Sentence embeddings, classification |
| Llama | `llama::LlamaModel` | Autoregressive text generation |
| Mistral | `mistral::MistralModel` | Sliding window attention |
| Whisper | `whisper::WhisperModel` | Speech-to-text |
| Phi | `phi::PhiModel` | Small LLM for edge devices |
| Stable Diffusion | `stable_diffusion::StableDiffusion` | Text-to-image |

### Quantization for Memory-Constrained Deployment

Candle supports loading quantized weights via the `candle-quant` crate. The most common format is GGML/GGUF, which uses 4-bit or 8-bit block quantization to reduce model size by 4-8x:

```rust
use candle_quant::QuantizedVarBuilder;

let vb = QuantizedVarBuilder::from_gguf("model.gguf", &device)?;
let model = llama::LlamaModel::load_quantized(vb, &config)?;
// Model now uses ~1.75 GB instead of ~14 GB for 7B parameters
```

Quantization trades a small accuracy loss (typically < 1% on perplexity benchmarks) for dramatically reduced memory and bandwidth requirements. On an M2 MacBook with 16 GB RAM, a 4-bit quantized 7B model runs at interactive speeds.

## 3. Tokenizer Considerations

`candle-transformers` provides the model but **not** the tokenizer. You must bring your own—typically the `tokenizers` crate, which is a Rust binding of Hugging Face's fast tokenizer:

```rust
let tokenizer = Tokenizer::from_file("tokenizer.json")
    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
```

❌ **Mistake:** Assuming the tokenizer is embedded in the model file.
✅ **Reality:** The tokenizer is a separate artifact (`tokenizer.json` or `vocab.txt`). Its vocabulary must match the model's `Config.vocab_size`.

The benefit of this separation: you can use the same model with different tokenizers (e.g., a custom WordPiece for a domain-specific BERT without re-downloading the weights).

---

## 🎯 Key Takeaways
- Every model follows `Config + VarBuilder + Model::load()` — learn once, apply to all architectures.
- Weights are memory-mapped via `VarBuilder::from_mmaped_safetensors` — no need to load multi-GB files into RAM.
- The tokenizer is a **separate** dependency — ensure vocabulary alignment with the model config.
- A single Rust binary can replace a multi-GB Python container for transformer inference.

## References
- `candle-transformers` docs: https://huggingface.github.io/candle/candle_transformers/index.html
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- [[02 - GPU Acceleration and Device Abstraction]]
- [[01 - Deep Learning y Computer Vision]]

## 📦 Código de compresión

```rust
use candle_core::{Device, Result, Tensor};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn embed(text: &str, model: &BertModel, tok: &Tokenizer, dev: &Device) -> Result<Tensor> {
    let enc = tok.encode(text, false).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let ids: Vec<u32> = enc.get_ids().iter().map(|&id| id as u32).collect();
    model.forward(&Tensor::new(&ids[..], dev)?.unsqueeze(0)?, None)?.mean(1)
}

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let api = Api::new()?;
    let repo = api.repo(Repo::new("sentence-transformers/all-MiniLM-L6-v2".into(), RepoType::Model));
    let cfg: Config = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[repo.get("model.safetensors")?], DTYPE, &dev)? };
    let model = BertModel::load(vb, &cfg)?;
    let tok = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let v1 = embed("Rust is fast.", &model, &tok, &dev)?;
    let v2 = embed("Rust is safe.", &model, &tok, &dev)?;
    let sim = v1.broadcast_mul(&v2)?.sum_all()? / (v1.sqr()?.sum_all()?.sqrt()? * v2.sqr()?.sum_all()?.sqrt()?)?;
    println!("Similarity: {}", sim.to_scalar::<f32>()?);
    Ok(())
}
```
