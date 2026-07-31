# 🎯 02 - Online Inference and Event-Driven ML — Sub-50ms Predictions

> **The inference layer. How to make predictions in <50ms, at scale, with proper backpressure, timeouts, and graceful degradation. The architecture that turns a model into a service.**

## 🎯 Learning Objectives
- Distinguish online, batch, and micro-batch inference
- Architect event-driven ML services with sub-50ms latency
- Implement pre-warmed models, request batching, and streaming responses
- Configure HTTP/2, gRPC, and connection pooling for high-throughput inference
- Apply backpressure and load shedding for graceful degradation
- Integrate LLM inference with online feature stores
- Build an inference service that handles 10K RPS

## Introduction

The inference layer is what users actually see. The features (from Note 01) flow into the model; the model's output is the user-facing response. The latency budget is tight: <100ms for traditional ML, <500ms for LLM token streaming.

This note covers the four aspects of online inference:

1. **Latency optimization** — pre-warmed models, request batching, GPU sharing
2. **Throughput patterns** — concurrent requests, async I/O, connection pooling
3. **Backpressure** — load shedding, queue management, graceful degradation
4. **LLM-specific patterns** — token streaming, prompt caching, multi-provider routing

By the end of this note you will have the architecture to serve at 10K RPS with sub-50ms p99 latency.

![Online inference architecture](https://example.com/inference-arch.png)

---

## 1. The Three Inference Modes

| Mode | Latency | Throughput | Cost | Use case |
|------|---------|-----------|------|----------|
| **Online (real-time)** | <100ms | Lower (per-request) | Highest | Fraud detection, recommendations |
| **Micro-batch** | Seconds | Higher | Medium | Session-based predictions |
| **Batch** | Minutes-hours | Highest | Lowest | Daily reports, retraining |

For real-time ML, online inference is the default. The model is loaded in memory; features are read from the online store; predictions are made per request.

---

## 2. The Online Inference Architecture

A typical online inference service:

```python
# inference_service.py
import asyncio
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_client import Histogram, Counter
import joblib
import numpy as np


# Metrics
prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Time to compute prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
predictions_served = Counter(
    "model_predictions_served_total",
    "Total predictions served",
    ["model_version", "outcome"],
)


class ModelServer:
    """Holds the model in memory for fast inference."""
    
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_version = "1.2.3"
    
    async def predict(self, features: dict) -> dict:
        # Convert features to numpy array
        feature_array = np.array([
            features["f1"], features["f2"], features["f3"],
            ...
        ]).reshape(1, -1)
        
        with prediction_latency.time():
            prediction = self.model.predict_proba(feature_array)
        
        return {
            "prediction": float(prediction[0][1]),
            "model_version": self.model_version,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    app.state.model = ModelServer("/models/model.pkl")
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    user_id: str
    features: dict


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        result = await app.state.model.predict(request.features)
        predictions_served.labels(model_version=result["model_version"], outcome="success").inc()
        return result
    except Exception as e:
        predictions_served.labels(model_version="unknown", outcome="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
```

The model is loaded **once** in memory; every request hits it without reloading.

### 2.1 Pre-warming

For GPU-based models, "loading" includes transferring weights to GPU. This takes 30-60 seconds once. Use pre-warming:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model before the service starts accepting requests."""
    app.state.model = await load_model_async(
        model_path="/models/llama-3-70b",
        device="cuda",
        dtype="float16",
    )
    # Warmup: run a dummy inference to trigger CUDA kernel compilation
    await app.state.model.predict_async("Hello world")
    yield
```

Send a dummy request through the model during startup. This triggers CUDA kernel compilation and any first-run overhead. Production traffic hits a pre-warmed model.

---

## 3. Request Batching

Naive serving: one request per inference. Slow.

Batched serving: accumulate requests, batch them, run inference on the batch. Faster.

```python
import asyncio
from collections import deque


class BatchingModelServer:
    """Batch requests for efficient GPU inference."""
    
    def __init__(self, model, max_batch_size: int = 32, max_wait_ms: int = 10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        self.pending_results: dict[int, asyncio.Future] = {}
    
    async def predict(self, features: dict) -> dict:
        """Submit a prediction request; returns when batch is processed."""
        future = asyncio.Future()
        request_id = id(future)
        self.pending_results[request_id] = future
        await self.queue.put((request_id, features))
        return await future
    
    async def batch_processor(self):
        """Background task that batches and processes."""
        while True:
            batch = []
            batch_ids = []
            
            # Wait for first request
            request_id, features = await self.queue.get()
            batch.append(features)
            batch_ids.append(request_id)
            
            # Accumulate more requests within max_wait_ms
            deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    request_id, features = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=timeout,
                    )
                    batch.append(features)
                    batch_ids.append(request_id)
                except asyncio.TimeoutError:
                    break
            
            # Process the batch
            try:
                predictions = self.model.predict_batch(batch)
                for request_id, prediction in zip(batch_ids, predictions):
                    future = self.pending_results.pop(request_id)
                    future.set_result(prediction)
            except Exception as e:
                for request_id in batch_ids:
                    future = self.pending_results.pop(request_id)
                    future.set_exception(e)


# In lifespan
async def lifespan(app: FastAPI):
    model = load_model()
    server = BatchingModelServer(model, max_batch_size=32, max_wait_ms=10)
    asyncio.create_task(server.batch_processor())
    app.state.server = server
    yield
```

This batches up to 32 requests or 10ms, whichever comes first. GPU utilization improves 5-10×.

---

## 4. Concurrency Patterns

### 4.1 Async I/O for feature retrieval

```python
import httpx


async def get_features(user_id: str) -> dict:
    """Fetch features from external services in parallel."""
    async with httpx.AsyncClient() as client:
        # Fetch from multiple sources in parallel
        profile, history, embeddings = await asyncio.gather(
            client.get(f"http://profile-service/users/{user_id}"),
            client.get(f"http://history-service/{user_id}/recent"),
            client.get(f"http://embeddings-service/{user_id}"),
        )
    
    return {
        "profile": profile.json(),
        "history": history.json(),
        "embeddings": embeddings.json(),
    }
```

Sequential fetches would take 3× the latency. Parallel fetches take the max of the three.

### 4.2 Connection pooling

```python
# Singleton client with pool
client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
    ),
    timeout=httpx.Timeout(5.0),
)


async def call_external_service(request: dict) -> dict:
    response = await client.post("http://external-service/predict", json=request)
    return response.json()
```

A single connection pool handles 100 concurrent requests. Without pooling, every request opens a new connection (slow + FD exhaustion).

### 4.3 HTTP/2 multiplexing

```python
client = httpx.AsyncClient(http2=True)  # enables HTTP/2
```

HTTP/2 multiplexes multiple requests over a single connection. Reduces latency by 20-30% for multi-request workloads.

---

## 5. Backpressure and Load Shedding

When the service is overwhelmed, it must **shed load gracefully**:

```python
class BackpressureModelServer:
    """Reject requests when overloaded; never queue indefinitely."""
    
    def __init__(self, max_concurrent: int = 100, queue_max: int = 50):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_max = queue_max
        self.pending = 0
        self.rejected = 0
    
    async def predict(self, features: dict) -> dict:
        # Reject if too many requests are queued
        if self.pending > self.queue_max:
            self.rejected += 1
            raise BackpressureError("Service overloaded; retry with backoff")
        
        async with self.semaphore:
            self.pending += 1
            try:
                return await self._predict_impl(features)
            finally:
                self.pending -= 1
    
    async def _predict_impl(self, features: dict) -> dict:
        # Actual inference
        return await model.predict(features)
```

When the queue is full, return 503 (Service Unavailable) with a `Retry-After` header. Let the client retry.

For graceful degradation, return a **fallback prediction** (a simpler model, a cached response, or a default):

```python
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        return await app.state.model.predict(request.features)
    except BackpressureError:
        # Fall back to the simpler model
        return await app.state.fallback_model.predict(request.features)
```

The fallback model is less accurate but never overwhelmed. The pattern is **graceful degradation**: full quality under load, reduced quality under extreme load.

---

## 6. LLM-Specific Real-time Patterns

For LLM inference (covered in [[06 - Large Language Models/23 - Serverless LLM Platforms|06/23 Serverless LLM]]):

### 6.1 Token streaming

```python
from fastapi.responses import StreamingResponse


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def token_generator():
        async for token in llm_client.stream(
            model="gpt-4o-mini",
            messages=request.messages,
        ):
            yield f"data: {token.content}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

The client receives tokens as they're generated. Time-to-first-token (TTFT) is the key metric — for voice agents, <300ms TTFT is required.

### 6.2 Request coalescing

Multiple identical requests in flight are coalesced into one upstream call:

```python
class RequestCoalescer:
    """Coalesce identical requests to avoid duplicate LLM calls."""
    
    def __init__(self):
        self.in_flight: dict[str, asyncio.Future] = {}
    
    async def chat(self, messages: list[dict]) -> dict:
        # Hash the request
        key = hash(json.dumps(messages, sort_keys=True))
        
        if key in self.in_flight:
            # Coalesce: return the in-flight future
            return await self.in_flight[key]
        
        # New request: create future and start the call
        future = asyncio.Future()
        self.in_flight[key] = future
        
        try:
            result = await llm_client.chat(messages)
            future.set_result(result)
            return result
        finally:
            del self.in_flight[key]
```

If 100 users type "Explain Kubernetes" simultaneously, only 1 LLM call is made. The 99 others wait for the result.

### 6.3 Prompt caching

For prompts with shared prefixes (system prompts, long context), use prompt caching:

```python
# Together AI: 5-10 minute cache on identical prefix
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[
        {"role": "system", "content": long_system_prompt},  # cached
        {"role": "user", "content": user_query},  # not cached
    ],
)
```

Saves 80%+ on input token cost. See [[06 - Large Language Models/23 - Serverless LLM Platforms/04 - Serverless Cost Optimization and Patterns|06/23 Cost Optimization]] for details.

---

## 7. Real-World Example — Real-time Recommendation

```python
# recommendation_service.py
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model (ONNX for fast inference)
    app.state.model = await load_onnx_model("/models/recommender.onnx", device="cuda")
    
    # Feast client for online features
    app.state.feast = FeastClient(repo_path="feature_repo/")
    
    # HTTP clients for external services
    app.state.catalog = httpx.AsyncClient(base_url="http://catalog-service")
    app.state.embeddings = httpx.AsyncClient(base_url="http://embeddings-service")
    
    yield


app = FastAPI(lifespan=lifespan)


class RecommendationRequest(BaseModel):
    user_id: str
    context: list[str]  # recent page views
    top_k: int = 10


@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    start = time.perf_counter()
    
    # 1. Fetch features in parallel (Feast online store + context)
    features_task = app.state.feast.get_online_features(
        feature_view="user_recent_features",
        entity_rows=[{"user_id": request.user_id}],
    )
    embeddings_task = app.state.embeddings.post(
        "/embed", json={"context": request.context}
    )
    
    features, embeddings = await asyncio.gather(features_task, embeddings_task)
    
    # 2. Build feature vector
    feature_vector = np.concatenate([
        features["view_count_5m"],
        features["click_count_5m"],
        embeddings.json()["embedding"],
    ])
    
    # 3. Inference (sub-10ms for ONNX)
    with prediction_latency.time():
        scores = app.state.model.predict(feature_vector.reshape(1, -1))
    
    # 4. Get top-K items by score
    top_k_indices = np.argsort(scores[0])[-request.top_k:][::-1]
    
    # 5. Fetch item details in parallel
    item_ids = [f"item_{i}" for i in top_k_indices]
    items = await asyncio.gather(*[
        app.state.catalog.get(f"/items/{item_id}")
        for item_id in item_ids
    ])
    
    elapsed = time.perf_counter() - start
    return {"items": items, "latency_ms": elapsed * 1000}
```

**Total latency**: typically <50ms (p99), <20ms (p50).

The architecture:
- Feast provides features in <5ms
- Embeddings service provides context in <10ms
- ONNX model inference in <10ms
- Catalog lookup in <20ms
- Total: <50ms p99

---

## 8. Antipatterns

### 8.1 Antipattern 1: Loading model per request

```python
# ❌ Load model every request — slow
@app.post("/predict")
async def predict(request):
    model = joblib.load("/models/model.pkl")  # 2-3 seconds
    return model.predict(request.features)

# ✅ Load once in lifespan
@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load("/models/model.pkl")
    yield
```

### 8.2 Antipattern 2: Synchronous feature fetching

```python
# ❌ Fetch features sequentially
features = await fetch_profile(user_id)
recent = await fetch_history(user_id)
embeddings = await fetch_embeddings(user_id)

# ✅ Fetch in parallel
features, recent, embeddings = await asyncio.gather(
    fetch_profile(user_id),
    fetch_history(user_id),
    fetch_embeddings(user_id),
)
```

### 8.3 Antipattern 3: No backpressure

```python
# ❌ Queue indefinitely — out-of-memory crash
@app.post("/predict")
async def predict(request):
    await app.state.queue.put(request)  # unbounded
    return await wait_for_result(request.id)

# ✅ Use bounded queue + load shedding
@app.post("/predict")
async def predict(request):
    if app.state.queue.qsize() > MAX_QUEUE:
        return {"error": "Service overloaded"}, 503
    await app.state.queue.put(request)
```

### 8.4 Antipattern 4: No graceful degradation

```python
# ❌ Return 500 on timeout
async def predict_with_timeout(request):
    return await asyncio.wait_for(model.predict(request), timeout=0.05)


# ✅ Fall back to simpler model on timeout
async def predict_with_fallback(request):
    try:
        return await asyncio.wait_for(
            app.state.full_model.predict(request),
            timeout=0.05,
        )
    except asyncio.TimeoutError:
        return await app.state.fallback_model.predict(request)
```

### 8.5 Antipattern 5: No metrics

```python
# ❌ No observability — debugging is impossible
@app.post("/predict")
async def predict(request):
    return await model.predict(request.features)

# ✅ Latency + throughput + error metrics
@app.post("/predict")
async def predict(request):
    with prediction_latency.time():
        try:
            result = await model.predict(request.features)
            predictions_served.labels(outcome="success").inc()
            return result
        except Exception:
            predictions_served.labels(outcome="error").inc()
            raise
```

---

## 🎯 Key Takeaways

- Online inference loads model once in memory; pre-warm before serving traffic.
- Request batching: accumulate 5-10ms or 32 requests, then process in parallel for 5-10× throughput.
- Async I/O for feature retrieval: parallel fetches = max(latency), not sum.
- Connection pooling: 100 concurrent connections per pool; HTTP/2 for multi-request workloads.
- Backpressure: bounded queue + 503 with Retry-After; never queue indefinitely.
- Graceful degradation: full model under load, simpler fallback under extreme load.
- LLM-specific: token streaming (TTFT), request coalescing, prompt caching.
- Avoid loading model per request, sync feature fetching, no backpressure, no fallback, no metrics.

## References

- vLLM — [docs.vllm.ai](https://docs.vllm.ai/)
- Triton Inference Server — [github.com/triton-inference-server](https://github.com/triton-inference-server)
- TensorRT — [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)
- ONNX Runtime — [onnxruntime.ai](https://onnxruntime.ai/)
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns|Note 11 — Advanced Async Patterns]] — async patterns
- [[06 - Large Language Models/23 - Serverless LLM Platforms/04 - Serverless Cost Optimization and Patterns|Serverless Cost Optimization]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/01 - Streaming Feature Engineering|Note 01 — Streaming Features]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/03 - Change Data Capture for ML|Note 03 — CDC]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/04 - Drift Detection in Real-time|Note 04 — Drift Detection]]
- [[09 - MLOps y Produccion/40 - Real-time ML Systems/05 - Capstone - Production Real-time ML Pipeline|Note 05 — Capstone]]
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM]] — high-throughput LLM serving