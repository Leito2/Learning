# Capstone: Full Optimization Cycle

## Objective

Take a naive FastAPI model server → profile it → optimize bottlenecks → add async → package it → deploy on Ray Serve → benchmark improvement.

## Phase 1: Naive Server

```python
from fastapi import FastAPI
import numpy as np
import time

app = FastAPI()

class SlowModel:
    def predict(self, features: list[list[float]]) -> list[list[float]]:
        time.sleep(0.5)  # simulate slow preprocessing
        predictions = np.random.rand(len(features), 3).tolist()
        time.sleep(0.3)  # simulate slow postprocessing
        return predictions

model = SlowModel()

@app.post("/predict")
def predict(data: dict):
    features = data["features"]
    result = model.predict(features)
    return {"predictions": result}
```

## Phase 2: Profile

```bash
# Run the server
uvicorn server:app &

# Profile with py-spy
py-spy record -o naive-flame.svg --pid $(pgrep -f uvicorn) --duration 30

# Load test
python -c "
import requests, time
start = time.perf_counter()
for _ in range(100):
    requests.post('http://localhost:8000/predict', json={'features': [[1.0]*10]*4})
print(f'Avg: {(time.perf_counter()-start)/100*1000:.1f} ms')
"
```

**Expected py-spy output:** wide bars for `time.sleep` (simulated I/O), `numpy.random.rand`, and serialization.

## Phase 3: Optimize

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class OptimizedModel:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.precomputed = np.random.rand(1000, 3)  # cache

    async def predict(self, features: list[list[float]]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._predict_sync,
            features
        )
        return result

    def _predict_sync(self, features):
        return np.random.rand(len(features), 3).tolist()
```

## Phase 4: Add Async Batching

```python
import asyncio
from asyncio import Queue, Future

class BatchedModel:
    def __init__(self, model, max_batch=32, timeout_ms=50):
        self.model = model
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000
        self.queue: list[tuple[dict, Future]] = []
        self._task = asyncio.create_task(self._batch_loop())

    async def predict(self, data: dict) -> dict:
        future = asyncio.get_event_loop().create_future()
        self.queue.append((data, future))
        return await future

    async def _batch_loop(self):
        while True:
            await asyncio.sleep(self.timeout)
            if not self.queue:
                continue
            batch_data = self.queue[:self.max_batch]
            self.queue = self.queue[self.max_batch:]
            features = [d["features"] for d, _ in batch_data]
            results = self.model._predict_sync(features)
            for (_, future), result in zip(batch_data, results):
                future.set_result({"prediction": result})
```

## Phase 5: Package

```toml
[project]
name = "optimized-inference"
version = "0.1.0"
dependencies = ["fastapi>=0.115", "numpy>=1.26", "ray[serve]>=2.30"]
```

```bash
mkdir -p src/optimized_inference/
pip install -e .
```

```python
# src/optimized_inference/server.py
from .model import BatchedModel

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 0})
@serve.ingress(app)
class OptimizedServer:
    def __init__(self):
        self.model = BatchedModel(OptimizedModel())

    @app.post("/predict")
    async def predict(self, data: dict):
        return await self.model.predict(data)
```

## Phase 6: Deploy on Ray Serve

```bash
# Local
serve run src.optimized_inference.server:app

# Production (Kubernetes)
serve build src.optimized_inference.server:app -o config.yaml
kubectl apply -f config.yaml
```

## Phase 7: Benchmark

```python
import asyncio
import aiohttp
import time

async def benchmark():
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.post("http://localhost:8000/predict", json={"features": [[1.0]*10]*4})
            for _ in range(100)
        ]
        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        print(f"Async avg: {elapsed/100*1000:.1f} ms")

asyncio.run(benchmark())
```

**Expected improvement:**
| Metric | Naive | Optimized | Improvement |
|--------|-------|-----------|-------------|
| p50 latency | ~800 ms | ~150 ms | 5.3x |
| Throughput | ~1.2 req/s | ~25 req/s | 20x |
| p99 latency | ~1200 ms | ~300 ms | 4x |

## Summary: Optimization Checklist

- [ ] Profile with py-spy → identify bottleneck
- [ ] Move blocking I/O to thread pool
- [ ] Implement async batch inference
- [ ] Package with poetry dep groups
- [ ] Deploy on Ray Serve with autoscaling
- [ ] Compare before/after benchmarks
- [ ] Export latency metrics to Prometheus

## Key Takeaways

- Naive synchronous servers waste GPU capacity while waiting for I/O
- Profiling identifies the true bottleneck (rarely the model itself)
- Async batching increases GPU utilization 5-20x
- Packaging with poetry enables minimal serving dependencies
- Ray Serve provides autoscaling, canary deployments, and K8s integration
- Always benchmark before and after each optimization step
