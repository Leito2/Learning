# 🧊 01 - ONNX Runtime for Tabular Models

On an Intel i5-10300H laptop, scoring a single payment with `XGBClassifier.predict_proba` takes about **1.5 ms**; the same 300-tree model exported to ONNX and run with ONNX Runtime takes about **0.09 ms** — roughly 17× faster. At a batch of 500, the ranking **flips** and native XGBoost wins. Both facts come from the same measurement, and both matter: they explain why P1 exports its champion to ONNX, why the scorer batches carefully, and why "ONNX is faster" is only true for the shapes you actually measured.

## 🎯 Learning Objectives
- Explain what ONNX is (graph, operators, opsets, the `ai.onnx.ml` domain) and how tree ensembles are represented
- Understand ONNX Runtime's architecture: graph optimizations, execution providers, session and threading options
- Export **XGBoost** (onnxmltools), **scikit-learn** pipelines (skl2onnx), and **PyTorch** MLPs (`torch.onnx.export`)
- Prove **numerical parity** and understand why tree models can differ at float32 split boundaries
- Benchmark native vs ONNX across **batch sizes and thread counts**, and interpret the crossover
- Avoid the classic traps: ZipMap outputs, feature-order mismatches, opset/version drift

## Introduction

ONNX (Open Neural Network Exchange) is a standard format for ML models as **computation graphs**: nodes are operators from versioned sets (*opsets*), edges are typed tensors, and the whole thing is a protobuf file. It began as a neural-network interchange format (Microsoft and Facebook, 2017) but includes a classical-ML operator domain, `ai.onnx.ml`, with operators such as `TreeEnsembleClassifier`, `LinearClassifier`, `Scaler`, and `OneHotEncoder`. A trained XGBoost model can therefore be expressed as one `TreeEnsembleClassifier` node carrying every tree's splits and leaf values.

ONNX Runtime (ORT) is Microsoft's high-performance engine for executing ONNX graphs. For tabular models in real-time services, its advantages are less about raw arithmetic and more about **overhead**: a lean C++ call path with no per-call data-structure construction, a stable ABI usable from Python, C++, C#, Java, Go, and Rust, and no dependency on the training library at serving time.

For P1 this means the scorer loads a single `.onnx` file from the MLflow registry, runs it in-process with one thread per process, and never imports XGBoost on the hot path — while a **parity test** guarantees the exported model is the model that was evaluated.

---

## 1. The Problem and Why This Solution Exists

### Why native predictors are slow for single rows

Training libraries optimize for throughput over large matrices. Calling `predict_proba` on one row still pays the fixed cost of input validation, building an internal matrix structure (XGBoost's `DMatrix`/inplace predictor setup), dispatching to a thread pool, and allocating outputs. With fixed cost $o$ and per-row cost $c$:

$$
t(k) = o + c\,k
\qquad
\frac{t(k)}{k} = \frac{o}{k} + c
$$

When $k = 1$, the fixed cost dominates. A runtime with a much smaller $o$ wins dramatically at small batches — exactly the regime of real-time scoring — even if its per-row cost $c$ is similar or worse.

### Why not ship the training library?

Serving with the training library couples the scorer to that library's version, Python environment, and threading behavior. An exported graph is **versioned, self-describing, and portable**: the same file runs in a Python scorer, a Go scorer (`onnxruntime-go`), or Triton's ONNX backend.

---

## 2. Conceptual Deep Dive

### 2.1 How a tree ensemble becomes an ONNX graph

Each tree is flattened into arrays: node IDs, feature indices, thresholds, branch modes (`BRANCH_LT`, `BRANCH_LEQ`), true/false child IDs, and leaf weights per class. The `TreeEnsembleClassifier` operator evaluates every tree for each row and aggregates leaf values; a post-transform (`LOGISTIC` for binary XGBoost) maps the margin to a probability:

$$
\hat{p}(x) = \sigma\!\left(b + \sum_{t=1}^{T} w_t\big(\text{leaf}_t(x)\big)\right)
\qquad
\text{cost} \approx O(T \cdot d) \text{ comparisons per row}
$$

For $T = 300$ trees of depth $d = 6$, that is ~1,800 comparisons per row — microseconds. The arithmetic is cheap; the call overhead is what you're paying for.

⚠️ **Warning — float32 split boundaries:** ONNX tree operators store thresholds and compare inputs in **float32**. If a feature value lies within float32 rounding of a split threshold, the export can take the other branch than the native model in rare rows. Parity tests therefore use a small tolerance and also report the **maximum** difference and the count of rows above tolerance — and inputs should be float32 end to end so training and serving see the same values.

### 2.2 ONNX Runtime architecture

![ONNX Runtime execution providers](https://onnxruntime.ai/images/ONNX_Runtime_EP1.png)

*Figure: ORT partitions the graph across execution providers (CPU, CUDA, TensorRT, OpenVINO, DirectML…), falling back to CPU for unsupported nodes. Source: onnxruntime.ai.*

1. **Load and optimize:** the session applies graph optimizations (constant folding, redundant-node elimination, operator fusion) according to `graph_optimization_level`.
2. **Partition:** nodes are assigned to execution providers in priority order.
3. **Run:** `session.run(outputs, {input_name: array})` executes with the configured threading.

Key `SessionOptions` for real-time services:

| Option | Effect | Real-time guidance |
|---|---|---|
| `intra_op_num_threads` | Threads used **inside** an operator | 1 per process when you scale by processes; more only if one process must use several cores |
| `inter_op_num_threads` / `execution_mode` | Parallelism **between** independent nodes | Sequential is fine for single-node tree graphs |
| `graph_optimization_level` | Which optimizations run at load | `ORT_ENABLE_ALL` (default extended/all) |
| `enable_cpu_mem_arena` | Reuse memory buffers | Keep on for steady traffic |

💡 **Tip:** ORT releases the Python GIL during `run`, so a threaded server can overlap inference with other work. For the P1 scorer, simpler is better: one process per core budget, `intra_op_num_threads = 1`, scale with processes and Kafka partitions.

### 2.3 Converters

| Source | Converter | Notes |
|---|---|---|
| XGBoost | `onnxmltools.convert_xgboost` | Feature names must be default (`f0..fn`) or stripped; supports binary/multiclass/regression |
| LightGBM | `onnxmltools.convert_lightgbm` | Same operator family |
| scikit-learn pipelines | `skl2onnx.to_onnx` | Exports preprocessing **and** model as one graph — removes transformation skew |
| Trees → tensor ops | Hummingbird | Compiles trees into GEMM/tree-traversal tensor programs; attractive on GPUs |
| PyTorch | `torch.onnx.export` (dynamo-based exporter in PyTorch 2.x) | Put normalization inside the module so scaling travels with the model |

### 2.4 The ZipMap trap

By default, scikit-learn and some XGBoost classifier exports wrap probabilities in a **ZipMap** node that returns a list of Python dictionaries (`[{0: 0.9, 1: 0.1}, …]`). Building thousands of dicts per batch is slow and awkward. Disable it at export (`options={id(model): {"zipmap": False}}` in skl2onnx) or check that the output is already a tensor, as with the converter versions used below.

### 2.5 Measured: native vs ONNX (this laptop)

Setup: 300 trees, depth 6, 12 float32 features; XGBoost 3.4, ONNX Runtime 1.30, onnxmltools 1.16; Intel i5-10300H; Python 3.13; time per `predict` call.

| Batch size | Native (default threads) | ONNX (default threads) | Native (1 thread) | ONNX (1 thread) |
|---|---|---|---|---|
| 1 | 2.23 ms | **0.13 ms** | 1.53 ms | **0.09 ms** |
| 32 | 6.90 ms | **1.23 ms** | 2.40 ms | **0.82 ms** |
| 500 | **8.39 ms** | 14.27 ms | **7.24 ms** | 9.08 ms |

Max absolute probability difference over 2,000 rows: **4.2 × 10⁻⁷**.

Reading the table with $t(k) = o + ck$: ONNX has a far smaller fixed cost $o$ (≈ 0.09 ms vs ≈ 1.5 ms) — it wins whenever batches are small. Native XGBoost has a lower per-row cost $c$ at large batches in this configuration — it wins when batches are big. P1's scorer batches "up to 500 messages or 5 ms"; at laptop rates of a few thousand events/s, typical batches are **tens of rows**, where ONNX is 3× faster. The point is not the specific numbers (they depend on CPU, versions, and model) but the **method**: benchmark the batch sizes your traffic produces, record them in MLflow, and decide.

⚠️ **Warning:** never quote a speedup without the batch size, thread settings, versions, and hardware. "ONNX is 17× faster" and "native is 1.3× faster" are both true here.

---

## 3. Production Reality

### Contracts that prevent silent errors

- **Feature order:** an ONNX model takes a tensor; it has no idea which column is `amount`. Store the ordered feature list in the model's metadata (`metadata_props`) and in MLflow, and have the scorer **assert** it matches `fraudcore.FEATURE_ORDER` at load time. A swapped column is a silent, catastrophic bug.
- **Dtypes and shapes:** export with a dynamic batch dimension (`[None, n_features]`) and float32 inputs; reject other dtypes at the boundary.
- **Opset pinning:** record `target_opset` and the ORT version; upgrade deliberately and re-run parity.

### Parity as a release gate

The export step in P1 refuses to register a model unless: max |p_native − p_onnx| < 1e-5 on the full test set, the number of rows above 1e-6 is reported, and the benchmark across batch sizes is logged. The scorer loads only registered ONNX artifacts.

Caso real: Microsoft has described ONNX Runtime as the inference engine behind a wide range of its products and services (for example in Office, Bing, and Azure), where standardizing on one runtime across many model types reduced latency and simplified deployment across languages and hardware — the same portability argument that lets a Python-trained model run inside a Go or C# service.

Caso real: in tabular fraud and credit systems, a recurring post-mortem item is "model served with features in the wrong order after a refactor": offline metrics were perfect, online scores were plausible but wrong. Embedding the feature list in the artifact and asserting it at load time turns that into a startup failure.

---

## 4. Code in Practice

### Export, verify, and benchmark XGBoost

```python
import time
import numpy as np
import onnx
import onnxruntime as ort
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

FEATURES = [f"f{i}" for i in range(12)]                       # real project: fraudcore.FEATURE_ORDER

def export(model: xgb.XGBClassifier, n_features: int) -> bytes:
    onx = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                          target_opset=15)
    meta = onx.metadata_props.add()
    meta.key, meta.value = "feature_order", ",".join(FEATURES)     # contract travels with the artifact
    return onx.SerializeToString()

def parity(model, sess, X: np.ndarray, tol: float = 1e-5) -> dict:
    p_native = model.predict_proba(X)[:, 1]
    p_onnx = sess.run(["probabilities"], {"input": X})[0][:, 1]
    diff = np.abs(p_native - p_onnx)
    return {"max_abs_diff": float(diff.max()), "rows_over_1e-6": int((diff > 1e-6).sum()),
            "passed": bool(diff.max() < tol)}

def load_session(blob: bytes, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
    order = sess.get_modelmeta().custom_metadata_map["feature_order"].split(",")
    assert order == FEATURES, "feature order mismatch: refuse to serve"
    return sess
```

### ❌/✅ Probability outputs

```python
# ❌ ZipMap output: a Python dict per row — slow and easy to index wrongly
probs = sess.run(None, {"input": X})[1]          # [{0: 0.97, 1: 0.03}, ...] with some converters
p_fraud = [d[1] for d in probs]

# ✅ Tensor output (zipmap disabled or default tensor), vectorized indexing
p_fraud = sess.run(["probabilities"], {"input": X})[0][:, 1]
```

### PyTorch challenger with normalization inside the graph

```python
import torch

class MLP(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, n: int):
        super().__init__()
        self.register_buffer("mean", mean)            # scaling stats live INSIDE the model
        self.register_buffer("std", std)
        self.net = torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.ReLU(),
                                       torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(self, x):
        return torch.sigmoid(self.net((x - self.mean) / self.std))

m = MLP(torch.zeros(12), torch.ones(12), 12).eval()
torch.onnx.export(m, (torch.randn(1, 12),), "challenger.onnx", input_names=["input"],
                  output_names=["p_fraud"], dynamic_axes={"input": {0: "batch"}})
```

### 📦 Compression code: export, parity, and the batch-size crossover

```python
# 📦 Compression code: XGBoost → ONNX → ORT, parity check, latency vs batch size
# Covers: convert_xgboost, metadata contract, parity tolerance, t(k) = o + c·k crossover
# pip install xgboost onnxruntime onnxmltools onnx numpy
import time
import numpy as np
import onnxruntime as ort
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

rng = np.random.default_rng(0)
X = rng.normal(size=(20_000, 12)).astype(np.float32)
y = ((1.5 * X[:, 0] + X[:, 3] ** 2 - X[:, 7] + rng.normal(0, 0.5, len(X))) > 2.5).astype(int)
model = xgb.XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist", n_jobs=1).fit(X, y)

onx = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, 12]))], target_opset=15)
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])

diff = np.abs(model.predict_proba(X[:2000])[:, 1] - sess.run(["probabilities"], {"input": X[:2000]})[0][:, 1])
print(f"parity: max |Δp| = {diff.max():.2e} → {'PASS' if diff.max() < 1e-5 else 'FAIL'}")

def per_call(fn, k: int) -> float:
    xb, reps = X[:k], max(20, 2000 // k)
    t = time.perf_counter()
    for _ in range(reps):
        fn(xb)
    return (time.perf_counter() - t) / reps * 1000

for k in (1, 32, 500):
    n_ms = per_call(model.predict_proba, k)
    o_ms = per_call(lambda xb: sess.run(["probabilities"], {"input": xb}), k)
    print(f"batch={k:>3}: native {n_ms:6.2f} ms | onnx {o_ms:6.2f} ms | onnx speedup {n_ms / o_ms:4.1f}×")
# ¡Sorpresa! The speedup shrinks — and may invert — as batches grow: ONNX mostly removes per-call overhead.
```

---

## 🎯 Key Takeaways
- ONNX represents tree ensembles as a single `TreeEnsembleClassifier` node; arithmetic is cheap, **call overhead** dominates small batches.
- Model latency as $t(k) = o + ck$: ONNX Runtime's small $o$ wins at real-time batch sizes; native libraries can win at large batches.
- Measured here: **~17× faster at batch 1**, ~3× at 32, slower at 500 — always benchmark **your** batch sizes and threads.
- Use `intra_op_num_threads = 1` and scale by processes for multi-process scorers.
- **Parity** is a release gate: max difference, rows above tolerance, float32 inputs end to end.
- Embed the **feature order** in the artifact and assert it at load; disable **ZipMap** outputs.
- Export preprocessing with the model (skl2onnx pipelines, normalization inside PyTorch modules) to kill transformation skew.

## References
- ONNX specification and operator docs (`ai.onnx.ml` TreeEnsembleClassifier) — https://onnx.ai
- ONNX Runtime documentation — session options, threading, execution providers, graph optimizations
- onnxmltools and skl2onnx documentation · Hummingbird (Nakandala et al., *A Tensor Compiler for Unified Machine Learning Prediction Serving*, OSDI 2020)
- XGBoost documentation — prediction and inplace predict
- [[00 - Welcome to High-Performance Model Serving|Course welcome]] · [[02 - Triton Deep Dive|Next: Triton Deep Dive]]
