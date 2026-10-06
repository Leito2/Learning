# 🔱 02 - Triton Deep Dive

In-process ONNX Runtime gives a single scorer the lowest possible latency. But when ten services need the same model, when a GPU must be shared by six models, when requests from many clients should be batched together, or when a model update must not redeploy every consumer, you want a **model server**. NVIDIA Triton Inference Server is the most general one: many frameworks, one server, one protocol, with dynamic batching and detailed metrics. The question for real-time ML is precise — what does Triton add, and what does the extra network hop cost?

## 🎯 Learning Objectives
- Describe Triton's architecture: model repository, backends, schedulers, protocols, metrics
- Write `config.pbtxt` files: shapes, `max_batch_size`, **instance groups**, **dynamic batching**, version policy
- Serve tree models with the **FIL backend** and neural/ONNX models with the **ONNX Runtime backend**
- Model **dynamic batching** as a queue and choose `max_queue_delay_microseconds` with numbers
- Decompose a Triton request's latency (network, queue, compute) using its metrics
- Decide when Triton beats in-process inference for P1, and when it doesn't

## Introduction

Triton (open-sourced by NVIDIA in 2018 as TensorRT Inference Server, later renamed) separates **what** a model is from **how** it's served. You drop model artifacts into a versioned directory structure; Triton loads each with the right **backend** (ONNX Runtime, TensorRT, PyTorch/LibTorch, Python, FIL for tree ensembles, and others), schedules requests, batches them dynamically, runs multiple instances concurrently on CPUs or GPUs, and exposes everything over HTTP and gRPC using the KServe v2 inference protocol, with Prometheus metrics built in.

This makes Triton the natural choice for **platforms**: a central inference tier that many applications call, where GPU utilization and operational consistency matter more than shaving the last millisecond. For a single latency-critical scorer with a small model, an in-process runtime is usually faster ([[01 - ONNX Runtime for Tabular Models|ONNX Runtime]]). P1's variant B measures that trade-off directly.

---

## 1. The Problem and Why This Solution Exists

### What a model server solves

| Problem | Without a server | With Triton |
|---|---|---|
| Many frameworks | One serving stack per framework | One server, many backends |
| GPU sharing | Each service loads its own copy | Shared GPUs, concurrent model execution |
| Small requests from many clients | Each client batches alone (or not at all) | **Server-side dynamic batching** across clients |
| Model updates | Redeploy every consumer | Swap versions in the repository; clients unchanged |
| Observability | Custom per service | Standard metrics: queue time, compute time, batch sizes |

### What it costs

A model server adds a **network hop** and **serialization**: the client encodes the input tensor (gRPC/protobuf or HTTP/JSON), sends it, Triton decodes it, schedules it, runs it, and returns the result. On localhost gRPC this is typically sub-millisecond to a few milliseconds per request — negligible for a 50 ms GPU model, significant for a 0.1 ms tree model.

---

## 2. Conceptual Deep Dive

### 2.1 Architecture

![Triton architecture](https://raw.githubusercontent.com/triton-inference-server/server/main/docs/user_guide/images/arch.jpg)

*Figure: requests arrive over HTTP/gRPC (or the C API), are routed to per-model schedulers, optionally batched, and executed by framework backends on CPU/GPU instances. Source: Triton Inference Server documentation.*

### 2.2 The model repository

```text
model_repository/
├── fraud_xgb_fil/
│   ├── config.pbtxt
│   └── 7/                      # version directory (integer)
│       └── xgboost.json        # FIL backend expects the XGBoost model file
├── fraud_xgb_onnx/
│   ├── config.pbtxt
│   └── 7/model.onnx
└── fraud_mlp_onnx/
    ├── config.pbtxt
    └── 3/model.onnx
```

**Model control modes:** `none` (load everything at startup), `poll` (rescan the repository periodically — simple hot updates), and `explicit` (load/unload through the API — what an MLOps pipeline drives after promoting a model in MLflow). The **version policy** chooses which versions are live (`latest { num_versions: 1 }`, `specific { versions: [6, 7] }`, or `all`).

### 2.3 Instance groups and concurrent execution

`instance_group [{ count: 2, kind: KIND_CPU }]` creates two execution instances of the model; Triton dispatches requests to free instances in parallel. On GPUs, several models (and several instances of one model) run concurrently on the same device.

![Concurrent model execution](https://raw.githubusercontent.com/triton-inference-server/server/main/docs/user_guide/images/multi_model_exec.png)

*Figure: multiple models and instances executing concurrently. Source: Triton Inference Server documentation.*

### 2.4 Dynamic batching as a queue

With `dynamic_batching { max_queue_delay_microseconds: D, preferred_batch_size: [...] }`, the scheduler holds incoming requests for up to $D$ to form a larger batch, then executes it. With batch cost $o + c\,k$ and arrival rate $\lambda$:

$$
\text{batch size} \approx \min(k_{\max},\; \max(1, \lambda D))
\qquad
L_{\text{request}} \approx \underbrace{\text{wait} \le D}_{\text{queue}} + o + c\,k
$$

Throughput capacity with batching approaches $k / (o + c\,k)$, much higher than $1/(o + c)$ when $o \gg c$ (typical on GPUs). The trade-off is the same as every micro-batch in this vault ([[../43 - Real-time Feature Serving with Redis/04 - Latency Budgets for Online Serving|latency budgets]]): at low traffic, $D$ is pure added latency; at high traffic, batches fill fast and $D$ rarely binds. For latency-sensitive CPU models, start with $D$ around 100–500 µs and **measure**.

### 2.5 Backends for P1's models

| Model | Backend | Why |
|---|---|---|
| XGBoost champion | **FIL** (Forest Inference Library) | Purpose-built tree inference on GPU **and** CPU; loads native XGBoost/LightGBM/Treelite files; `output_class` and `threshold` parameters |
| XGBoost champion (alternative) | **ONNX Runtime** | Same `.onnx` artifact as the in-process scorer → identical numerics |
| PyTorch MLP challenger | **ONNX Runtime** (or LibTorch) | Small dense model, CPU-friendly |
| Pre/post-processing | **Python** backend | Arbitrary Python inside the server (slower; use sparingly) |

### 2.6 Where a request's time goes

Triton's metrics split server-side time precisely:

$$
L_{\text{client}} = L_{\text{net+serde}} + \underbrace{T_{\text{queue}} + T_{\text{compute\_input}} + T_{\text{compute\_infer}} + T_{\text{compute\_output}}}_{\texttt{nv\_inference\_request\_duration\_us}}
$$

exported as cumulative counters (`nv_inference_queue_duration_us`, `nv_inference_compute_infer_duration_us`, …) on `:8002/metrics`. Dividing by request counts gives per-request averages; histograms or perf_analyzer give percentiles ([[03 - Triton Ensembles and perf_analyzer|next note]]). If queue time dominates, add instances or reduce $D$; if compute dominates, optimize the model or move to a GPU.

---

## 3. Production Reality

### Running Triton on an 8 GB laptop

- The standard container (`nvcr.io/nvidia/tritonserver:<YY.MM>-py3`) includes many backends and is **several GB** on disk. Check free space before pulling; for leaner deployments, Triton's build tooling can produce images with only the backends you need.
- CPU-only works: use `KIND_CPU` instance groups and don't request GPUs. Budget ~1–1.5 GB RAM for the server with small models.
- In P1 this is the `serving-triton` Compose profile; with 8 GB it is measured only at low load steps (the full comparison waits for 16 GB).

### In-process vs Triton for P1 — the expected outcome

| Aspect | In-process ONNX (champion path) | Triton (variant B) |
|---|---|---|
| Per-batch overhead | Function call | gRPC + serde + queue |
| Batching | Scorer's micro-batch (Kafka poll) | Server-side dynamic batching across scorers |
| Isolation | Model crash = scorer crash | Separate process; scorer survives |
| Rollout | Scorer reloads the alias | Server swaps versions; clients unchanged |
| GPU sharing | N/A | Natural |
| Expected p95 impact on a laptop | Lowest | +1–3 ms per batch (to be measured in E4) |

The expected conclusion — *in-process wins on latency for a small tree model on CPU; Triton wins on operability and on GPU/multi-model platforms* — is a hypothesis until E4 measures it.

Caso real: large ML platforms commonly run a central Triton tier serving dozens of models (recommendation rankers, embedding encoders, vision models) behind one protocol; teams ship new model versions by writing to the repository, and the platform team tunes instance counts and batching per model from Triton's queue/compute metrics, without touching client services.

Caso real: RAPIDS' FIL backend for Triton was created because GPU tree inference for large XGBoost/LightGBM models at high throughput is dramatically faster than CPU at large batch sizes; for single small requests, the CPU path (or in-process inference) remains competitive — the same crossover as [[01 - ONNX Runtime for Tabular Models|note 01]], at the server level.

---

## 4. Code in Practice

### `config.pbtxt` — ONNX champion on CPU with dynamic batching

```protobuf
name: "fraud_xgb_onnx"
backend: "onnxruntime"
max_batch_size: 512
input  [{ name: "input",         data_type: TYPE_FP32, dims: [ 12 ] }]
output [{ name: "probabilities", data_type: TYPE_FP32, dims: [ 2 ] },
        { name: "label",         data_type: TYPE_INT64, dims: [ 1 ] }]
instance_group [{ count: 2, kind: KIND_CPU }]
dynamic_batching {
  preferred_batch_size: [ 32, 128 ]
  max_queue_delay_microseconds: 300          # start small for latency-critical CPU models; measure
}
version_policy: { latest { num_versions: 1 } }
```

⚠️ **Warning:** output names and shapes must match the ONNX graph exactly (inspect it with Netron or `onnx.load(...).graph.output`). The `label` output's exact dims depend on the converter version — check, don't assume.

### `config.pbtxt` — XGBoost via the FIL backend

```protobuf
name: "fraud_xgb_fil"
backend: "fil"
max_batch_size: 512
input  [{ name: "input__0",  data_type: TYPE_FP32, dims: [ 12 ] }]
output [{ name: "output__0", data_type: TYPE_FP32, dims: [ 2 ] }]
instance_group [{ count: 1, kind: KIND_CPU }]
parameters [
  { key: "model_type",   value: { string_value: "xgboost_json" } },
  { key: "output_class", value: { string_value: "true" } },
  { key: "predict_proba", value: { string_value: "true" } }
]
dynamic_batching { max_queue_delay_microseconds: 300 }
```

💡 **Tip:** FIL parameter names have evolved across releases (e.g., probability output options); copy them from the FIL backend README for the exact Triton tag you pin.

### Running and calling it

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/model_repository:/models" nvcr.io/nvidia/tritonserver:<YY.MM>-py3 \
  tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=30
curl -s localhost:8000/v2/health/ready && curl -s localhost:8002/metrics | grep nv_inference_queue_duration_us
```

```python
import numpy as np
import tritonclient.grpc as grpcclient          # pip install tritonclient[grpc]

client = grpcclient.InferenceServerClient("localhost:8001")
batch = np.random.rand(32, 12).astype(np.float32)
inp = grpcclient.InferInput("input", list(batch.shape), "FP32")
inp.set_data_from_numpy(batch)
res = client.infer("fraud_xgb_onnx", [inp], outputs=[grpcclient.InferRequestedOutput("probabilities")])
p_fraud = res.as_numpy("probabilities")[:, 1]
```

### ❌/✅ Dynamic batching for a latency-critical path

```protobuf
# ❌ GPU-style setting copied to a CPU fraud scorer: every request may wait up to 10 ms at low traffic
dynamic_batching { max_queue_delay_microseconds: 10000 }

# ✅ Small delay; the scorer already sends micro-batches, so server batching only needs to merge them
dynamic_batching { max_queue_delay_microseconds: 300 }
```

### 📦 Compression code: dynamic batching delay vs latency and throughput

```python
# 📦 Compression code: discrete-event simulation of Triton-style dynamic batching
# Covers: max_queue_delay, batch cost o + c·k, p50/p95 vs load, the low-traffic latency tax
import random
import statistics

def simulate(rate: float, max_delay: float, o: float = 0.0008, c: float = 0.00002,
             k_max: int = 512, n: int = 30_000, seed: int = 1) -> tuple[float, float, float]:
    rnd = random.Random(seed)
    t, arrivals = 0.0, []
    for _ in range(n):
        t += rnd.expovariate(rate)
        arrivals.append(t)
    lat, i, server_free, sizes = [], 0, 0.0, []
    while i < n:
        start = max(arrivals[i], server_free)
        deadline = max(start, arrivals[i] + max_delay)           # first request waits ≤ max_delay
        j = i
        while j < n and j - i < k_max and arrivals[j] <= deadline:
            j += 1
        launch = max(start, min(deadline, arrivals[j - 1] if j - i == k_max else deadline))
        done = launch + o + c * (j - i)
        lat += [done - a for a in arrivals[i:j]]
        sizes.append(j - i)
        server_free, i = done, j
    q = statistics.quantiles(lat, n=100)
    return q[49] * 1000, q[94] * 1000, statistics.mean(sizes)

for rate in (200, 2_000, 20_000):
    for d in (0.0, 0.0003, 0.010):
        p50, p95, k = simulate(rate, d)
        print(f"λ={rate:>6}/s  max_delay={d*1e6:>6.0f}µs → p50={p50:6.2f}ms p95={p95:6.2f}ms  avg batch={k:6.1f}")
# ¡Sorpresa! At 200 req/s a 10 ms delay barely grows batches (≈3) but adds ~8 ms to the median request;
# at 20k req/s requests queue anyway, so batches of ~26 form even with zero delay — the queue IS the batcher.
```

---

## 🎯 Key Takeaways
- Triton separates artifacts (versioned **model repository**) from serving (backends, schedulers, protocols, metrics).
- **Instance groups** run models concurrently; **dynamic batching** merges requests across clients.
- Dynamic batching trades up to `max_queue_delay` of waiting for throughput $\approx k/(o + ck)$ — tiny delays for latency-critical CPU models.
- Use **FIL** for tree ensembles (CPU/GPU) or the **ONNX Runtime backend** to reuse the exact in-process artifact.
- Read `nv_inference_queue_duration_us` vs `compute_infer` to know whether to add instances or optimize the model.
- For a small CPU model on one machine, **in-process** usually wins latency; Triton wins **operability, isolation, GPU sharing, and multi-model platforms**.

## References
- NVIDIA Triton Inference Server documentation — model repository, model configuration, dynamic batcher, metrics, model management
- triton-inference-server/fil_backend README — FIL backend parameters
- KServe — Open Inference Protocol (v2)
- Olston et al., *TensorFlow-Serving: Flexible, High-Performance ML Serving* (2017) · Crankshaw et al., *Clipper* (NSDI 2017)
- [[../../10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/03 - NVIDIA Triton|NVIDIA Triton overview]]
- [[01 - ONNX Runtime for Tabular Models|Previous: ONNX Runtime]] · [[03 - Triton Ensembles and perf_analyzer|Next: Ensembles and perf_analyzer]]
