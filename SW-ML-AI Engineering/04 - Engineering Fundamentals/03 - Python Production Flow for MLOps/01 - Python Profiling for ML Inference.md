# Python Profiling for ML Inference

## 1. Why Profile ML Inference?

Inference latency bottlenecks are rarely where you expect them. Common assumptions: "the model is the bottleneck" → reality: data preprocessing, serialization, or network I/O. Profiling identifies the true bottleneck before you optimize.

## 2. cProfile: Statistical Profiling

```python
import cProfile
import pstats

def run_inference(model, inputs):
    preprocessed = preprocess(inputs)     # 1
    predictions = model.predict(preprocessed)  # 2
    return postprocess(predictions)        # 3

profiler = cProfile.Profile()
profiler.enable()
result = run_inference(model, sample_inputs)
profiler.disable()

pstats.Stats(profiler).sort_stats("cumtime").print_stats(10)
```

Key columns: `ncalls` (call count), `tottime` (time in function only), `cumtime` (time including calls).

## 3. py-spy: Sampling Profiler (No Code Changes)

```bash
pip install py-spy
py-spy record -o profile.svg --pid <PID>     # running process
py-spy record -o profile.svg -- python server.py  # from start
py-spy top --pid <PID>                        # live top-like view
```

py-spy is **safe for production**: reads process memory without pausing the process. Outputs flame graphs (SVG).

## 4. Memory Profiling

```python
from memory_profiler import profile

@profile
def load_and_infer(model_path, data):
    model = load_model(model_path)  # Line #, Mem usage, Increment
    result = model.predict(data)
    return result
```

```bash
python -m memory_profiler script.py
```

For production: `tracemalloc` tracks peak memory and leaks:

```python
import tracemalloc

tracemalloc.start()
snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics("lineno")
for stat in stats[:10]:
    print(stat)
```

## 5. GPU Profiling

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

For NVIDIA CUDA kernels:

```python
import torch

# CUDA events for kernel timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(input_tensor)
end.record()
torch.cuda.synchronize()
print(f"Kernel time: {start.elapsed_time(end):.2f} ms")
```

## 6. Flask/ FastAPI Inference Profiling

```python
import time
from functools import wraps

def profile_endpoint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Report to Prometheus / your metrics backend
        INFERENCE_LATENCY.observe(elapsed)
        return result
    return wrapper

@app.post("/predict")
@profile_endpoint
def predict(request: InferenceRequest):
    ...
```

## 7. Flame Graph Interpretation

Flame graphs from py-spy show call stacks as horizontal bars:
- **Width** = time spent
- **Color** = depth in stack
- **Read bottom-up**: base of flame = entry point, top = leaf calls
- **Plateaus**: wide flat bars = likely bottleneck

Look for:
- Wide `model.forward` bars → optimize model (quantize, prune, fuse)
- Wide `preprocess` bars → optimize data pipeline (async, caching)
- Wide `serialization` bars → optimize output format (protobuf, msgpack)

## 8. Optimization Loop Pattern

```
1. Profile → identify bottleneck
2. Optimize → change one thing
3. Profile → verify improvement
4. Repeat until SLO is met
```

```python
def optimize_loop(original_fn, profiler, slo_ms=100):
    baseline = measure_latency(original_fn)
    while baseline > slo_ms:
        bottleneck = profiler.identify_top_caller(original_fn)
        optimized_fn = apply_optimization(original_fn, bottleneck)
        baseline = measure_latency(optimized_fn)
    return optimized_fn
```

## Key Takeaways

- cProfile for call-count stats, py-spy for production-safe flame graphs
- Memory profiling: `memory_profiler` for scripts, `tracemalloc` for production
- GPU profiling: `nvidia-smi` for utilization, CUDA events for kernel timing
- Flame graphs: wide bars = bottlenecks
- Optimization loop: profile one change at a time until SLO is met
- Profile in production with py-spy (zero overhead), not cProfile
