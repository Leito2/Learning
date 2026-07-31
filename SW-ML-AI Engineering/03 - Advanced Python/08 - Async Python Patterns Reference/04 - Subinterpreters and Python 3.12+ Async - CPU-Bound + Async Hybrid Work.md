# 🎯 04 - Subinterpreters and Python 3.12+ Async — CPU-Bound + Async Hybrid Work

> **The 2026 frontier. Run CPU-bound code in parallel with async I/O. True parallelism, no GIL contention, on the same Python process.**

## 🎯 Learning Objectives
- Understand what subinterpreters are and why they solve the GIL problem
- Use `concurrent.futures.InterpreterPoolExecutor` (Python 3.12+) for parallel CPU work
- Combine subinterpreters with async for hybrid CPU + I/O workloads
- Decide between subinterpreters, multiprocessing, and asyncio.to_thread
- Benchmark subinterpreters vs alternatives
- Recognize when subinterpreters help and when they don't
- Apply subinterpreters to common AI workloads (model training, batch inference)

## Introduction

The **GIL** (Global Interpreter Lock) has been Python's biggest concurrency limitation for 25 years. For I/O-bound async code, it doesn't matter — the GIL is released during socket calls. But for CPU-bound work in async functions, the GIL serializes all CPU execution. You cannot get true parallelism for CPU work in a single Python process — until now.

Python 3.12 introduced **per-interpreter GIL** (PEP 684) and a new `InterpreterPoolExecutor`. Each subinterpreter has its own GIL. Run multiple subinterpreters in parallel — true CPU parallelism, no GIL contention.

For AI/ML workloads, this is revolutionary:
- Run LLM inference on one subinterpreter while a web server handles requests on the main
- Run feature engineering on multiple subinterpreters (parallel CPU) while async I/O fetches data
- Run model evaluation on 8 subinterpreters in parallel (one per core)

This note shows you how.

---

## 1. The GIL Problem

```python
# ❌ CPU-bound work blocks async I/O
import asyncio
import pandas as pd


async def fetch_and_process():
    data = await fetch_data()  # I/O
    df = pd.DataFrame(data)  # CPU
    result = df.groupby("category").sum()  # CPU — BLOCKS EVENT LOOP
    return result
```

While `df.groupby().sum()` runs:
- The event loop is blocked
- Other async tasks (HTTP requests, DB queries) can't progress
- The web server is unresponsive

For pure CPU work, the workaround has been `multiprocessing` or `to_thread`. But:

- `multiprocessing` has high IPC overhead (pickling)
- `to_thread` is limited by the GIL — threads don't run CPU in parallel
- Both are heavyweight for short-lived CPU tasks

**Subinterpreters solve this**: parallel CPU, low overhead, same process.

---

## 2. What Are Subinterpreters?

A subinterpreter is a **separate Python interpreter** running in the same process. Each has its own:

- GIL (no contention)
- Memory heap
- Module state (with caveats)
- Threading state

The main process can have multiple subinterpreters running concurrently, each doing CPU work in parallel.

```python
# Visual representation:
#
# ┌─────────────────────────┐
# │      Main Process        │
# │  ┌────────┐ ┌────────┐    │
# │  │ Sub-1  │ │ Sub-2  │    │
# │  │ (GIL-1)│ │(GIL-2)│     │
# │  │ CPU-A  │ │ CPU-B  │    │
# │  └────────┘ └────────┘    │
# │  ┌──────────────────┐    │
# │  │  Async Event Loop │    │
# │  │  (I/O, no GIL)    │    │
# │  └──────────────────┘    │
# └─────────────────────────┘
```

Two subinterpreters running CPU work in parallel, while the main event loop handles async I/O.

---

## 3. `InterpreterPoolExecutor` (Python 3.12+)

```python
from concurrent.futures import InterpreterPoolExecutor
import asyncio
import time


def cpu_bound_task(data):
    """Runs in a subinterpreter. CPU-bound; blocks the sub's GIL only."""
    import pandas as pd
    df = pd.DataFrame(data)
    return df.groupby("category").sum().to_dict()


async def hybrid_fetch_and_process(data_sources):
    """Async I/O + parallel CPU in subinterpreters."""
    loop = asyncio.get_running_loop()
    
    # Step 1: Async fetch all data sources (uses event loop)
    data_list = await asyncio.gather(*[fetch_source(src) for src in data_sources])
    
    # Step 2: CPU-bound processing in parallel subinterpreters
    with InterpreterPoolExecutor(max_workers=4) as pool:
        tasks = [
            loop.run_in_executor(pool, cpu_bound_task, data)
            for data in data_list
        ]
        results = await asyncio.gather(*tasks)
    
    return results


async def fetch_source(source):
    await asyncio.sleep(0.5)  # simulate I/O
    return {"category": "A", "value": 1}
```

The key insight: `InterpreterPoolExecutor` runs CPU work in subinterpreters (parallel CPU, no GIL contention) while the main event loop stays free for async I/O.

---

## 4. Subinterpreter Use Cases for AI/ML

### 4.1 Parallel model evaluation

```python
def evaluate_model(model_path, eval_dataset):
    """Evaluate a model on a dataset. CPU-bound."""
    import torch
    model = torch.load(model_path)
    correct = 0
    for batch in eval_dataset:
        pred = model(batch.x)
        correct += (pred == batch.y).sum().item()
    return correct / len(eval_dataset)


async def evaluate_all_models(model_paths, datasets):
    """Evaluate 4 models on 4 datasets in parallel."""
    loop = asyncio.get_running_loop()
    with InterpreterPoolExecutor(max_workers=4) as pool:
        tasks = [
            loop.run_in_executor(pool, evaluate_model, path, dataset)
            for path, dataset in zip(model_paths, datasets)
        ]
        results = await asyncio.gather(*tasks)
    return dict(zip(model_paths, results))
```

4 model evaluations × 4 datasets = 16 evaluations in parallel. Each on a separate subinterpreter. 8-core machine: 8 evaluations in parallel, 2 batches to complete.

### 4.2 Feature engineering

```python
def engineer_features(raw_data):
    """CPU-bound feature engineering."""
    import pandas as pd
    df = pd.DataFrame(raw_data)
    
    # Rolling statistics
    df["rolling_mean_7"] = df["value"].rolling(7).mean()
    df["rolling_std_7"] = df["value"].rolling(7).std()
    
    # Lag features
    df["lag_1"] = df["value"].shift(1)
    df["lag_7"] = df["value"].shift(7)
    
    # Interaction features
    df["interaction"] = df["value"] * df["volume"]
    
    return df.to_dict(orient="records")


async def engineer_all_features(raw_datasets):
    """Process multiple datasets in parallel."""
    loop = asyncio.get_running_loop()
    with InterpreterPoolExecutor(max_workers=os.cpu_count()) as pool:
        tasks = [
            loop.run_in_executor(pool, engineer_features, data)
            for data in raw_datasets
        ]
        return await asyncio.gather(*tasks)
```

### 4.3 Synchronous LLM client (offline batch inference)

```python
def offline_inference(model_path, prompts):
    """Use a sync LLM client (e.g., transformers pipeline)."""
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model_path)
    return [pipe(p)[0]["generated_text"] for p in prompts]


async def batch_inference(prompt_batches):
    """Run offline inference on multiple batches in parallel subinterpreters."""
    loop = asyncio.get_running_loop()
    with InterpreterPoolExecutor(max_workers=2) as pool:
        tasks = [
            loop.run_in_executor(pool, offline_inference, model_path, batch)
            for batch in prompt_batches
        ]
        return await asyncio.gather(*tasks)
```

Use this when you have a synchronous LLM client (transformers, llama.cpp) but want to parallelize.

---

## 5. Subinterpreters vs Multiprocessing vs `to_thread`

| Method | Parallel CPU | Overhead | Memory | When to use |
|--------|:------------:|:--------:|:------:|-------------|
| **`to_thread`** | ❌ (GIL) | Low | Shared | Quick I/O-bound sync calls |
| **`multiprocessing`** | ✅ | High (pickle) | Separate | Long-running CPU, separate memory needed |
| **`InterpreterPoolExecutor`** | ✅ | Low | Shared | CPU work in same process |

### 5.1 Benchmarking

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, InterpreterPoolExecutor


def cpu_bound(n):
    """CPU-bound task: sum of squares."""
    return sum(i * i for i in range(n))


# Benchmark each approach
n = 10_000_000
n_tasks = 4

# ThreadPoolExecutor (GIL — should be slow for CPU)
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(cpu_bound, [n] * n_tasks))
thread_time = time.perf_counter() - start

# ProcessPoolExecutor (separate processes)
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=4) as pool:
    list(pool.map(cpu_bound, [n] * n_tasks))
process_time = time.perf_counter() - start

# InterpreterPoolExecutor (subinterpreters)
start = time.perf_counter()
with InterpreterPoolExecutor(max_workers=4) as pool:
    list(pool.map(cpu_bound, [n] * n_tasks))
interp_time = time.perf_counter() - start

print(f"ThreadPool: {thread_time:.2f}s (GIL-bound, slow)")
print(f"ProcessPool: {process_time:.2f}s (fast but high overhead)")
print(f"InterpreterPool: {interp_time:.2f}s (fast, low overhead)")
```

Typical results on 4-core machine:
- ThreadPool: 4× slower (GIL serialization)
- ProcessPool: 1× time (parallel) + ~50ms IPC overhead
- InterpreterPool: 1× time + ~5ms overhead

---

## 6. Limitations of Subinterpreters

### 6.1 Module state isolation

Each subinterpreter has its own module state. This means:
- Module-level variables are NOT shared
- C extensions must be subinterpreter-safe

```python
# ❌ This won't work — counter is in main process
counter = 0

def increment():
    global counter
    counter += 1  # this counter is the SUB's, not the main's

with InterpreterPoolExecutor() as pool:
    pool.map(increment, range(10))  # each sub has its own counter

print(counter)  # still 0!
```

### 6.2 No shared memory by default

For data sharing, use `multiprocessing.shared_memory` or pass data via `Executor.map` (which pickles).

### 6.3 C extension compatibility

Some C extensions (e.g., NumPy 1.x, certain database drivers) are not subinterpreter-safe. NumPy 2.x added support; older versions may crash.

### 6.4 Python version requirement

`InterpreterPoolExecutor` requires Python 3.12+. For 3.11 and earlier, fall back to `multiprocessing`.

---

## 7. Hybrid Async + Subinterpreter Pattern

The canonical pattern for AI/ML workloads:

```python
async def pipeline():
    # Phase 1: Async I/O (event loop)
    data = await fetch_data_async()
    
    # Phase 2: CPU work in subinterpreters (parallel)
    loop = asyncio.get_running_loop()
    with InterpreterPoolExecutor(max_workers=4) as pool:
        results = await asyncio.gather(*[
            loop.run_in_executor(pool, process_data, d)
            for d in data_chunks
        ])
    
    # Phase 3: Async I/O (save to DB)
    await asyncio.gather(*[save_result(r) for r in results])
```

Three phases:
1. **Async I/O**: fetch data (event loop free)
2. **CPU in subinterpreters**: process data (parallel CPU)
3. **Async I/O**: save results (event loop free)

The event loop is never blocked.

---

## 8. Antipatterns

### 8.1 Antipattern 1: Using `to_thread` for CPU work expecting parallelism

```python
# ❌ to_thread doesn't enable parallel CPU
async def parallel():
    results = await asyncio.gather(*[
        asyncio.to_thread(cpu_task, data) for data in data_list
    ])
    # Threads run sequentially due to GIL!
```

```python
# ✅ Use InterpreterPoolExecutor
async def parallel():
    loop = asyncio.get_running_loop()
    with InterpreterPoolExecutor() as pool:
        results = await asyncio.gather(*[
            loop.run_in_executor(pool, cpu_task, data) for data in data_list
        ])
```

### 8.2 Antipattern 2: Sharing mutable state across subinterpreters

```python
# ❌ State isn't shared; bug
shared_list = []

def add(item):
    shared_list.append(item)  # in subinterpreter; different list!

with InterpreterPoolExecutor() as pool:
    pool.map(add, range(10))

print(shared_list)  # still empty!
```

```python
# ✅ Pass data via args/return values
def add(items, item):
    items.append(item)

shared_list = []
with InterpreterPoolExecutor() as pool:
    pool.map(partial(add, shared_list), range(10))  # still doesn't share!

# Best: return data from subinterpreter
def add_one(item):
    return item * 2

with InterpreterPoolExecutor() as pool:
    results = list(pool.map(add_one, range(10)))
print(results)  # [0, 2, 4, ..., 18]
```

### 8.3 Antipattern 3: Using subinterpreters for short-lived CPU work

```python
# ❌ Subinterpreter startup overhead (~5ms) for trivial CPU work
def trivial(n):
    return n + 1

with InterpreterPoolExecutor() as pool:
    list(pool.map(trivial, range(100)))
# Slower than sequential!

# ✅ Use for work that takes >50ms
def expensive(n):
    return sum(i * i for i in range(n))

with InterpreterPoolExecutor() as pool:
    list(pool.map(expensive, [10_000_000] * 4))
# Much faster than sequential
```

---

## 9. When NOT to Use Subinterpreters

| Scenario | Why not | Right tool |
|----------|---------|-----------|
| **Short-lived CPU work** (<50ms) | Startup overhead > savings | Sequential code or `to_thread` |
| **I/O-bound work** | Subinterpreters don't help I/O | asyncio |
| **Existing shared state** | Subinterpreters don't share | Multiprocessing (with IPC) |
| **Subinterpreter-unsafe C extensions** | Crashes or undefined behavior | `multiprocessing` |
| **Python <3.12** | `InterpreterPoolExecutor` not available | `multiprocessing` or `ProcessPoolExecutor` |

---

## 10. Decision Tree

```
CPU-bound work?
├── No  → Use asyncio
└── Yes
    ├── Python ≥3.12 and no shared state?
    │   └── YES → InterpreterPoolExecutor
    └── Otherwise
        ├── Short-lived (<50ms)?
        │   └── YES → Sequential or to_thread
        └── Long-lived (>50ms)
            ├── Shared state required?
            │   └── YES → Multiprocessing (with IPC)
            └── No shared state
                └── ProcessPoolExecutor
```

---

## 🎯 Key Takeaways

- Subinterpreters give **true CPU parallelism** in a single Python process (Python 3.12+).
- Each subinterpreter has its own GIL → no contention.
- `InterpreterPoolExecutor` is the API; integrates with `asyncio.run_in_executor`.
- Use for: parallel model evaluation, batch feature engineering, offline LLM inference.
- Avoid for: short-lived CPU work, shared mutable state, subinterpreter-unsafe C extensions.
- The hybrid pattern: async I/O → CPU in subinterpreters → async I/O. Event loop never blocked.
- Always benchmark; subinterpreters are not always faster than `multiprocessing`.
- Verify Python 3.12+ before adopting.

## References

- PEP 684 — Per-Interpreter GIL — [peps.python.org/pep-0684](https://peps.python.org/pep-0684/)
- PEP 554 — Multiple Interpreters in Stdlib — [peps.python.org/pep-0554](https://peps.python.org/pep-0554/)
- asyncio.subprocess — [docs.python.org/3/library/asyncio-subprocess.html](https://docs.python.org/3/library/asyncio-subprocess.html)
- [[../01 - Event Loop Internals - uvloop, Selectors, and the GIL Interplay|Note 01 — Event Loop Internals]]
- [[../03 - Async Anti-Patterns Reference Card - 20 Patterns with Stack-Trace Signatures|Note 03 — Anti-Patterns Reference]]
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]
- [[../../09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering|Incident Response Note 04]]