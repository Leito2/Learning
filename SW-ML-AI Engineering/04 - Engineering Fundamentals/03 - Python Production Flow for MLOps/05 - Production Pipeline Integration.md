# Production Pipeline Integration

## 1. Pipeline Architecture

```
[Request] → Profiling → Async Server → Model (Ray/GPU) → Response
                ↓
          Prometheus / Grafana
```

The four skills (profiling, Ray, async, packaging) compose into a single deployment pipeline.

## 2. Profiling-As-Middleware

```python
from fastapi import FastAPI, Request
import time
from prometheus_client import Histogram

INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency", ["model_version"])

app = FastAPI()

@app.middleware("http")
async def profile_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    model_version = request.headers.get("X-Model-Version", "unknown")
    INFERENCE_LATENCY.labels(model_version=model_version).observe(elapsed)
    response.headers["X-Latency-Ms"] = str(round(elapsed * 1000, 2))
    return response
```

## 3. Async + Ray Serve Integration

```python
from ray import serve
from fastapi import FastAPI
from contextlib import asynccontextmanager

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 5},
)
@serve.ingress(app)
class ProductionModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    @app.post("/predict")
    async def predict(self, data: dict):
        features = data["features"]
        return {"prediction": self.model.predict(features).tolist()}
```

## 4. Packaging + Deployment

```toml
# pyproject.toml
[tool.poetry.group.serve.dependencies]
ray = {extras = ["serve", "default"], version = "^2.30"}
fastapi = "^0.115"
prometheus-client = "^0.21"
```

```bash
poetry install --with serve
serve run production_model:app
```

## 5. CI/CD Pipeline Integration

```yaml
# .github/workflows/deploy.yml
jobs:
  profile:
    runs-on: [self-hosted, GPU]
    steps:
      - run: py-spy record -o profile.svg -- python benchmark.py
      - uses: actions/upload-artifact@v4
        with:
          name: profile-flamegraph

  test:
    runs-on: ubuntu-latest
    steps:
      - run: poetry install --with serve
      - run: pytest tests/ -v --benchmark-compare

  deploy:
    needs: [profile, test]
    runs-on: [self-hosted, GPU]
    steps:
      - run: poetry install --only serve
      - run: serve run production_model:app
```

## 6. Graceful Shutdown in Production

```python
import signal
import asyncio

@serve.deployment
class GracefulModel:
    async def __init__(self):
        self.model = load_model()

    async def reconfigure(self):
        await self.drain()
        await self.load_new_model()

    async def drain(self):
        self.ready = False
        await asyncio.sleep(5)  # allow in-flight requests
```

## 7. Observability Stack

```python
from prometheus_client import Counter, Histogram, Gauge

REQUESTS_TOTAL = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("latency_seconds", "Request latency", ["endpoint"])
GPU_MEMORY = Gauge("gpu_memory_bytes", "GPU memory usage", ["device"])
```

## 8. End-to-End Pipeline Script

```python
class MLOpsPipeline:
    def __init__(self):
        self.metrics = MetricsRegistry()

    @profile(flamegraph="profile.svg")
    @async_batch(max_batch_size=32)
    @package(deployment="model:v1")
    async def predict(self, request: dict) -> dict:
        return await self.model.predict(request["features"])

    async def deploy(self):
        serve.run(self)
```

## Key Takeaways

- Profiling middleware: Histogram + response headers for every request
- Async + Ray Serve: compose async model server with Ray autoscaling
- Poetry + serve: minimal dependency group for deployment
- CI/CD: profile flamegraph as CI artifact, benchmark tests, then deploy
- Graceful shutdown: drain in-flight requests before model swap
- Observability: Prometheus metrics for latency, throughput, GPU utilization
- Pipeline: `@profile` → `@async_batch` → `@package` composes the four skills
