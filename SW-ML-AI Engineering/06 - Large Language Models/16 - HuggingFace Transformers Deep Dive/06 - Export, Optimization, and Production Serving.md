# ⚡ Export, Optimization, and Production Serving

## 🎯 Learning Objectives

- Export HuggingFace models to ONNX, TensorRT, and OpenVINO using the `optimum` library
- Apply `torch.compile` to `transformers` models for graph-level acceleration
- Implement quantization strategies: GPTQ, AWQ, and dynamic quantization with `optimum`
- Understand `text-generation-inference` (TGI) architecture and when to self-host it
- Design FastAPI serving patterns with batching, async queues, and streaming
- Compare HF Inference API, TGI, Triton, and vLLM for production workloads
- Build Docker-based deployment pipelines for transformer models at scale

---

## Introduction

Training a state-of-the-art transformer is only half the battle. The other half is serving predictions with millisecond latency under bursty traffic on hardware your finance team approved. This note bridges the gap between the research checkpoint and the production endpoint. While [[06 - Large Language Models]] teaches you to fine-tune, and [[09 - MLOps y Produccion]] covers the CI/CD lifecycle, this module focuses on the mechanical transformation of a PyTorch `nn.Module` into a production artifact.

The HuggingFace ecosystem provides a dedicated optimization library, `optimum`, that acts as a compiler and runtime adapter for transformers. It supports ONNX for framework interoperability, TensorRT for NVIDIA GPU inference, and OpenVINO for Intel hardware. Beyond export, `torch.compile` (PyTorch 2.0+) offers graph compilation with minimal code changes. For serving, `text-generation-inference` (TGI) wraps `transformers` with a Rust-based HTTP server, continuous batching, and flash attention kernels.

Understanding these tools is non-negotiable for ML engineers in [[10 - Cloud, Infra y Backend]] roles. A model taking 2 seconds per request on a naive `model.generate()` loop can often be pushed below 200 ms with the right combination of quantization, compilation, and serving infrastructure.

The three axes of production optimization are:
- **Graph optimization**: Removing Python overhead via static graphs (ONNX, torch.compile)
- **Weight compression**: Reducing numerical precision to lower memory and bandwidth (GPTQ, AWQ)
- **Runtime orchestration**: Batching, scheduling, and memory management (TGI, vLLM)

---

## 1. Model Export with Optimum and ONNX

### The Export Pipeline

Deep learning frameworks like PyTorch are designed for research flexibility, not production efficiency. Dynamic computation graphs, Python overhead, and eager execution introduce latency unacceptable for high-throughput APIs. ONNX (Open Neural Network Exchange) solves this by representing the model as a static computation graph in a framework-agnostic format.

The export process uses **symbolic tracing**: the Python model is invoked once with dummy inputs, and every operation is recorded into a static graph. This traced graph removes all Python control flow, dynamic dispatch, and autograd overhead. The result is a serializable `.onnx` file that can be executed by any ONNX-compliant runtime.

```bash
# Export a BERT model to ONNX for CPU inference
optimum-cli export onnx \
    --model bert-base-uncased \
    --task text-classification \
    ./bert_onnx/

# Export with optimization for GPU via TensorRT
optimum-cli export onnx \
    --model google/vit-base-patch16-224 \
    --task image-classification \
    --optimize O2 \
    ./vit_onnx/
```

The `--task` flag tells the exporter which model head to trace—each task maps to specific input/output tensor names and shapes. Common task values include `text-classification`, `causal-lm`, `seq2seq-lm`, `image-classification`, and `object-detection`.

❌ **Antipattern**: Exporting with static shapes and later receiving requests with different sequence lengths. Without `dynamic_axes`, the ONNX graph hardcodes the sequence length from the dummy input, causing shape mismatch errors at runtime.

✅ **Correct**: Let `optimum-cli` infer dynamic axes automatically. For custom export, explicitly declare `dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}}`.

💡 **Tip**: Dynamic axes trade a small performance penalty for flexibility. If your production traffic has uniform sequence lengths, consider static shapes for maximum throughput.

Loading and running an exported model requires minimal code changes because `optimum` preserves the exact `transformers` API:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained("./bert_onnx/")
tokenizer = AutoTokenizer.from_pretrained("./bert_onnx/")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.argmax(-1))
```

The `ORTModelFor*` classes mirror the PyTorch API exactly. You can drop them into existing code by changing just the import line. Behind the scenes, ONNX Runtime applies graph optimizations automatically—operator fusion, constant folding, and memory layout optimization—that eager PyTorch cannot perform because it lacks the full graph view.

### Backend-Specific Optimizations

Once the ONNX graph is exported, it can be consumed by different backends:

**TensorRT** (NVIDIA) further optimizes the graph for specific GPU architectures. It performs:
- **Layer fusion**: Merging adjacent operations (Conv + Bias + ReLU → one kernel)
- **Kernel autotuning**: Selecting the fastest CUDA kernel for each operation based on the target GPU
- **INT8 calibration**: Quantizing activations to INT8 using a representative calibration dataset, minimizing accuracy loss while maximizing throughput

**OpenVINO** (Intel) targets CPUs, GPUs, and VPUs with:
- **Intermediate Representation (IR)**: A two-file format (`.xml` topology + `.bin` weights)
- **Dynamic shape inference**: Native support for variable-length inputs without recompilation
- **INT8 quantization**: Using the Intel Deep Learning Boost (VNNI) instructions on modern Xeon CPUs

The performance gains from ONNX + backend optimization can be substantial:

| Backend | Latency vs Eager PyTorch | Throughput vs Eager PyTorch |
|---------|--------------------------|-----------------------------|
| ONNX Runtime (CPU) | 2–3x faster | 3–4x higher |
| TensorRT (GPU) | 3–5x faster | 4–6x higher |
| OpenVINO (CPU) | 2–4x faster | 3–5x higher |

![ONNX Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/ONNX_logo_main.png/640px-ONNX_logo_main.png)

**Caso real**: Microsoft Azure Cognitive Services exports internal transformer models to ONNX Runtime to serve billions of NLP requests daily across global regions with sub-100 ms latency. The same ONNX artifact runs on CPU-only VMs in regions without GPU availability, falling back transparently from TensorRT to ONNX Runtime without code changes.

---

## 2. Quantization and torch.compile

### Numerical Precision vs Model Quality

Modern LLMs contain billions of FP16 parameters, requiring 2 GB per billion parameters just for weights. A 70B model consumes 140 GB in FP16—far beyond a single GPU's VRAM. Quantization reduces numerical precision to shrink memory footprints and accelerate matrix multiplications on hardware with low-precision support (NVIDIA Tensor Cores, Intel VNNI).

The quantization operation maps a full-precision weight $w$ to a quantized value $q$:

$$q = \text{clamp}\left(\text{round}\left(\frac{w}{s}\right) + z,\ 0,\ 2^b - 1\right)$$

where $s = \frac{\max(w) - \min(w)}{2^b - 1}$ is the scale factor, $z$ is the zero-point, and $b$ is the bit width. The dequantized weight is $\hat{w} = s \cdot (q - z)$.

The compression ratio achieved by reducing precision from $b_{\text{orig}}$ to $b_{\text{quant}}$ bits:

$$\text{CR} = \frac{b_{\text{orig}}}{b_{\text{quant}}}$$

A model stored in INT4 requires $\frac{16}{4} = 4\times$ less memory than its FP16 equivalent. However, quantization introduces error:

$$\mathcal{L}_{\text{quant}} = \|Wx - Q(W)x\|_2$$

The error depends on the quantization granularity:

| Granularity | Scale per | Memory Overhead | Accuracy |
|-------------|-----------|-----------------|----------|
| Per-tensor | Entire weight matrix | Negligible | Lowest |
| Per-channel | Each output channel | O(d_model) | Good |
| Per-group | Groups of 32–128 weights | O(n_groups) | Best |

Per-group quantization (group size 128) is the standard for GPTQ and AWQ, balancing memory overhead against accuracy preservation.

### Methods: Dynamic, GPTQ, AWQ

**Dynamic quantization** applies INT8 weights at runtime while keeping activations in FP32. No calibration data is needed, making it the simplest method to apply:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", export=True, quantization=True
)
```

Dynamic quantization works well for BERT-size models but degrades for LLMs where activation outliers are larger.

**GPTQ** (Frantar et al., 2022) treats quantization as a layer-wise optimization problem, finding quantized weights $\hat{W}$ that minimize the output error on a calibration dataset:

$$\hat{W} = \arg\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$$

where $W$ is the original weight matrix, $X$ is a set of input activations sampled from the calibration set, and $\|\cdot\|_F$ is the Frobenius norm. GPTQ solves this using approximate second-order information from the Hessian $H = 2XX^T$, quantizing weights in an order determined by the inverse Hessian diagonal. Layers quantized first can be compensated by remaining full-precision weights, reducing cumulative error.

```python
from optimum.gptq import GPTQQuantizer

quantizer = GPTQQuantizer(
    bits=4,
    dataset="c4",          # calibration dataset
    desc_act=False,        # per-group activation ordering
    group_size=128,
)
quantizer.quantize_model(model, tokenizer)
quantizer.save(model, "./llama-7b-gptq/")
```

❌ **Antipattern**: Applying GPTQ to a model that will later be fine-tuned with PEFT. INT4 weights are not differentiable in standard autograd; training through them produces noisy gradients.

✅ **Correct**: Use QLoRA for fine-tuning quantized models. The base model stays frozen in INT4 while LoRA adapters train in FP16.

**AWQ** (Lin et al., 2023) observes that a small fraction ($\sim 1\%$) of weight channels carry disproportionate importance because they correspond to large activation magnitudes. By keeping these salient channels in FP16 while quantizing the rest to INT4, AWQ achieves better accuracy than uniform quantization:

$$\hat{W}_i = \begin{cases}
W_i & \text{if channel } i \text{ is salient} \\
Q(W_i) & \text{otherwise}
\end{cases}$$

Salient channels are identified by measuring the activation distribution on a calibration set—no backpropagation needed. This gives AWQ a speed advantage over GPTQ during quantization while maintaining comparable accuracy.

![Quantization Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Quantization_visualization.svg/640px-Quantization_visualization.svg.png)

### torch.compile

`torch.compile` (PyTorch 2.0) takes a complementary approach. Instead of changing weights, it compiles the Python/PyTorch program into optimized C++ / Triton kernels using a graph compiler pipeline:

1. **TorchDynamo**: Captures the Python bytecode graph at frame boundaries (no source code changes needed)
2. **TorchInductor**: Converts the captured graph into optimized GPU kernels (Triton) or CPU kernels (C++/OpenMP)

The compilation fuses operations, eliminates redundant memory copies, and generates fused kernels like FlashAttention-2:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = torch.compile(model, mode="reduce-overhead")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
```

❌ **Antipattern**: Applying `torch.compile` and then modifying the model's weights afterward. The compiled graph is frozen; changes cause graph breaks or silent correctness errors.

✅ **Correct**: Compile only after all training, loading, and PEFT adaptation is complete. The mnemonic is **COMPILE LAST**.

💡 **Tip**: `torch.compile` offers three modes—`"default"` (best tradeoff of compile time vs runtime speed), `"reduce-overhead"` (low overhead for small models, fast compilation), `"max-autotune"` (slowest compile, fastest runtime). For production APIs receiving steady traffic, `"max-autotune"` amortizes the compile cost over thousands of requests.

**Caso real**: TheBloke's HuggingFace repository hosts thousands of GPTQ and AWQ quantized models, enabling the open-source community to run LLaMA-2-70B on consumer RTX 4090 GPUs (24 GB VRAM). With 4-bit quantization, the 140 GB FP16 model fits in ~35 GB, requiring only CPU RAM offloading for the excess. This single technique democratized access to 70B-class models that previously required A100-80GB clusters.

---

## 3. Production Serving Patterns

### The Memory Wall

LLM inference is fundamentally memory-bandwidth-bound during the decode phase. For each generated token, all model weights must be read from GPU HBM into SRAM for the forward pass. The time per token is dominated by this memory transfer, not by computation:

$$t_{\text{token}} \approx \frac{2 \cdot P \cdot \text{sizeof}(\text{dtype})}{\text{BW}_{\text{mem}}}$$

where $P$ is the number of parameters and $\text{BW}_{\text{mem}}$ is the memory bandwidth. For a 7B FP16 model on an A100 (2 TB/s bandwidth): $t_{\text{token}} \approx \frac{2 \cdot 7 \times 10^9 \cdot 2}{2 \times 10^{12}} \approx 14\ \text{ms per token}$.

Batching reduces this bottleneck by sharing weight reads across requests. With batch size $B$, the effective bandwidth utilization per token increases:

$$t_{\text{token}}(B) \approx \frac{2 \cdot P \cdot \text{sizeof}(\text{dtype})}{\text{BW}_{\text{mem}}} \cdot \frac{1}{B} + t_{\text{compute}}$$

The KV-cache adds another memory burden. For each sequence, the cache stores key and value tensors for all layers across all generated tokens:

$$\text{KV}_{\text{mem}} = 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot s \cdot 2\ \text{bytes}$$

For LLaMA-2-7B ($n_{\text{layers}}=32$, $n_{\text{heads}}=32$, $d_{\text{head}}=128$) at sequence length $s=2048$: $\text{KV}_{\text{mem}} = 2 \cdot 32 \cdot 32 \cdot 128 \cdot 2048 \cdot 2 \approx 1\ \text{GB per sequence}$.

### Continuous Batching and PagedAttention

Naive synchronous serving processes one request at a time, leaving GPU compute and memory underutilized. **Static batching** groups multiple requests but pads shorter sequences to the longest, wasting compute on padding tokens.

**Continuous batching** (in-flight batching) dynamically adds and removes requests from the GPU batch as sequences complete. When one request reaches `max_new_tokens`, it exits and a new one enters, eliminating padding waste entirely. This improves throughput by 2–4x under variable-length workloads.

**PagedAttention** (vLLM) extends this by managing KV-cache like operating system virtual memory. Instead of allocating contiguous blocks for each sequence (which causes fragmentation as sequences grow), it allocates fixed-size "pages" and maps them via a page table. This eliminates internal fragmentation and allows memory overcommitment across sequences.

### Serving Options: TGI, vLLM, FastAPI

TGI is HuggingFace's Rust-based production server with built-in continuous batching, FlashAttention, and PagedAttention:

```bash
docker run --gpus all \
    -p 8080:80 \
    -v $(pwd)/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --quantize gptq \
    --max-batch-prefill-tokens 4096
```

For lightweight or custom serving, FastAPI paired with `asyncio.run_in_executor` provides the minimum viable pattern:

```python
from fastapi import FastAPI
from transformers import pipeline
import asyncio

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    torch_dtype="auto",
    device_map="auto"
)

@app.post("/generate")
async def generate(prompt: str, max_new_tokens: int = 50):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    )
    return {"generated_text": result[0]["generated_text"]}
```

❌ **Antipattern**: Running `model.generate()` directly in a FastAPI endpoint without an executor. This blocks the async event loop, serializing all concurrent requests behind the first one.

✅ **Correct**: Use `asyncio.run_in_executor` to offload GPU inference to a thread pool, or use TGI/vLLM which handle request queuing natively.

💡 **Tip**: For multi-model services serving text, vision, and audio models from the same endpoint, use Triton Inference Server with model ensembles. It routes different model types to separate GPUs and exposes built-in Prometheus metrics for latency and throughput monitoring.

| Serving Solution | Batching | KV-Cache | Memory Mgmt | Best For |
|-----------------|----------|----------|-------------|----------|
| Naive FastAPI | Static | None | Manual | Prototypes, low traffic |
| TGI | Continuous | FlashAttention | Block allocator | Single LLM API |
| vLLM | Continuous | PagedAttention | Virtual memory pages | High-throughput LLM API |
| Triton + TRT-LLM | Custom ensembles | Custom | Custom slabs | Multi-model, enterprise |

**Caso real**: Perplexity.ai uses vLLM with continuous batching to serve open-source LLMs to millions of users with sub-200 ms latency. vLLM's PagedAttention eliminates KV-cache fragmentation, allowing the system to pack 2–3x more simultaneous requests per GPU than prior approaches, directly reducing their inference cost per query.

---

## 📦 Compression Code

```python
"""Export, quantize, compile, and serve a transformer model."""
import subprocess
from fastapi import FastAPI
from transformers import pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch
import asyncio

subprocess.run(["optimum-cli", "export", "onnx",
    "--model", "distilbert-base-uncased-finetuned-sst-2-english",
    "--task", "text-classification", "./onnx_model/"])

ort_model = ORTModelForSequenceClassification.from_pretrained("./onnx_model/")

gen_model = pipeline("text-generation", model="gpt2", device_map="auto")
gen_model.model = torch.compile(gen_model.model, mode="reduce-overhead")

app = FastAPI()

@app.post("/classify")
async def classify(text: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: ort_model(text))
    return {"label": result}

@app.post("/generate")
async def generate(prompt: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: gen_model(prompt, max_new_tokens=50)
    )
    return {"text": result[0]["generated_text"]}
```

## 🎯 Key Takeaways

- `optimum-cli export onnx` converts HuggingFace checkpoints to framework-agnostic static graphs for cross-platform deployment.
- TensorRT and OpenVINO are backend-specific optimizers that fuse layers and select hardware-appropriate kernels.
- Quantization (INT8, INT4) reduces memory by 2–4x; GPTQ and AWQ minimize accuracy loss via layer-wise optimization and salient channel preservation.
- `torch.compile` accelerates PyTorch via TorchDynamo + TorchInductor graph compilation without changing weights—always compile last.
- TGI provides production-grade text generation with continuous batching, FlashAttention, and an OpenAI-compatible API.
- FastAPI + `asyncio.run_in_executor` is the minimum viable pattern for non-blocking Python inference servers.
- For high-throughput LLM serving, use vLLM's PagedAttention to eliminate KV-cache fragmentation and maximize GPU utilization.

## References

- HuggingFace Optimum Documentation: https://huggingface.co/docs/optimum
- PyTorch 2.0 torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Frantar et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR.
- Lin et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys.
- Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP.
- TGI Repository: https://github.com/huggingface/text-generation-inference
- ONNX Runtime Documentation: https://onnxruntime.ai/
- HuggingFace Inference Documentation: https://huggingface.co/docs/inference-endpoints
