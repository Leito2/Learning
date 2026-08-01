# 🎯 02 - Service Patterns — REST, gRPC, Async, Adaptive Batching

> **The production inference API. Type-safe I/O, multiple protocols, batched inference for low latency. The patterns that make a BentoML service production-grade.**

## 🎯 Learning Objectives
- Use type-safe I/O with Pydantic models (auto OpenAPI generation)
- Implement async endpoints for long-running inference
- Use adaptive batching for low-latency inference at high throughput
- Expose both REST and gRPC endpoints from one service
- Use IO descriptors (JSON, Image, File) for efficient payload handling
- Implement request validation with custom validators
- Build streaming endpoints for real-time inference

## Introduction

The `@bentoml.api` decorator exposes your prediction function as an HTTP (or gRPC) endpoint. The signature (type hints) defines the request/response schema, generates the OpenAPI spec, and validates inputs.

Three patterns matter most for production:

1. **Type-safe I/O** with Pydantic — automatic validation + docs
2. **Adaptive batching** — multiple requests in one model call
3. **Async endpoints** — for long-running inference (LLMs, video)

This note covers the patterns.

---

## 1. Type-Safe I/O with Pydantic

```python
from pydantic import BaseModel, Field
from typing import Annotated
import bentoml


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Feature vector for Iris classifier",
    )
    model_version: str = Field(default="latest")


class PredictResponse(BaseModel):
    species: str
    confidence: float = Field(ge=0, le=1)
    model_version: str


@bentoml.service
class IrisService:
    model_ref = bentoml.models.BentoModel("iris_classifier:latest")
    
    def __init__(self):
        import joblib
        self.model = joblib.load(self.model_ref.path_of("model.pkl"))
    
    @bentoml.api
    def predict(self, req: PredictRequest) -> PredictResponse:
        features = np.array(req.features).reshape(1, -1)
        species_idx = self.model.predict(features)[0]
        confidence = float(self.model.predict_proba(features).max())
        
        return PredictResponse(
            species=["setosa", "versicolor", "virginica"][species_idx],
            confidence=confidence,
            model_version=self.model_ref.tag,
        )
```

The service:
- Validates input: `features` must be exactly 4 floats
- Auto-generates OpenAPI spec
- Returns type-safe response

The OpenAPI spec is auto-generated at `http://localhost:3000/docs.json`.

---

## 2. Batchable Endpoints

For low-latency inference at high throughput, batch multiple requests:

```python
@bentoml.service
class BatchedService:
    model = bentoml.models.BentoModel("model:latest")
    
    def __init__(self):
        self.model_runner = bentoml.pytorch.load_model_runner(self.model)
    
    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=64, max_latency_ms=100)
    def predict(self, input_data: list[np.ndarray]) -> list[np.ndarray]:
        """Batchable endpoint; BentoML batches up to 64 requests or 100ms."""
        # input_data is a list of numpy arrays
        # BentoML groups these into a single batch
        batched = np.stack(input_data)
        return self.model_runner.predict(batched)
```

The `max_batch_size=64, max_latency_ms=100` means:
- BentoML accumulates up to 64 requests
- OR waits up to 100ms
- Whichever comes first triggers the batch

For a model that takes 10ms per inference:
- Single-request: 100 RPS per worker (10ms each)
- Batched (32 reqs): 3,200 RPS per worker (10ms per batch, 32× throughput)

Adaptive batching is the **single biggest latency win** for ML serving.

---

## 3. Async Endpoints

For long-running inference (LLMs, video models):

```python
import asyncio
import bentoml


@bentoml.service
class AsyncService:
    model = bentoml.models.BentoModel("llm:latest")
    
    def __init__(self):
        self.model_runner = bentoml.transformers.load_model_runner(self.model)
    
    @bentoml.api(batchable=False)
    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Async endpoint for LLM inference."""
        # Use the async runner
        result = await self.model_runner.async_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
        )
        return result
```

Async endpoints don't block the worker thread; they release the worker during inference.

---

## 4. Streaming Endpoints

For real-time output (LLM token streaming):

```python
from typing import AsyncIterator
import bentoml


@bentoml.service
class StreamingService:
    model = bentoml.models.BentoModel("llm:latest")
    
    def __init__(self):
        self.model_runner = bentoml.transformers.load_model_runner(self.model)
    
    @bentoml.api
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Streaming endpoint; tokens flow to the client as they're generated."""
        async for token in self.model_runner.async_stream_generate(prompt):
            yield token
```

The client receives tokens as they're generated (SSE protocol).

```bash
curl -N http://localhost:3000/stream_generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Once upon a time"}'
```

`curl -N` disables buffering; tokens stream to the terminal.

---

## 5. Multiple Endpoints Per Service

```python
@bentoml.service
class MultiEndpointService:
    model = bentoml.models.BentoModel("model:latest")
    
    def __init__(self):
        self.model_runner = bentoml.load_model_runner(self.model)
    
    @bentoml.api
    def predict(self, input_data: dict) -> dict:
        """Single prediction."""
        return self.model_runner.predict(input_data).tolist()
    
    @bentoml.api(batchable=True, max_batch_size=64)
    def predict_batch(self, inputs: list[dict]) -> list[dict]:
        """Batch prediction."""
        return self.model_runner.predict_batch(inputs)
    
    @bentoml.api
    def explain(self, input_data: dict) -> dict:
        """SHAP-based explanation."""
        # Compute SHAP values
        import shap
        explainer = shap.Explainer(self.model_runner.predict)
        shap_values = explainer(input_data)
        return {"shap_values": shap_values.values.tolist()}
    
    @bentoml.api
    def health(self) -> dict:
        """Health check."""
        return {"status": "healthy"}
```

Multiple endpoints in one service — single container, single deployment.

---

## 6. IO Descriptors for Different Payloads

For binary data (images, audio, files):

```python
import bentoml
from bentoml.io import Image, File, JSON


@bentoml.service
class ImageService:
    model = bentoml.models.BentoModel("image_classifier:latest")
    
    def __init__(self):
        self.model_runner = bentoml.pytorch.load_model_runner(self.model)
    
    @bentoml.api(batchable=True)
    def predict(self, images: list[Image]) -> list[dict]:
        """Classify a list of images."""
        # Input is automatically decoded from multipart/form-data
        # Each Image is a PIL.Image
        results = []
        for image in images:
            prediction = self.model_runner.predict(image)
            results.append({"class": prediction.argmax(), "confidence": float(prediction.max())})
        return results
```

IO descriptors:
- `JSON` — JSON payloads (default)
- `Image` — image files (multipart or base64)
- `File` — arbitrary binary files
- `Text` — text payloads

---

## 7. gRPC Endpoints

For low-latency, type-safe inter-service communication:

```python
import bentoml
from typing import Annotated


# Define proto-compatible types via Pydantic
class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    species: str
    confidence: float


@bentoml.service
class GrpcService:
    model = bentoml.models.BentoModel("iris_classifier:latest")
    
    @bentoml.api
    def predict(self, req: PredictRequest) -> PredictResponse:
        features = np.array(req.features).reshape(1, -1)
        return {"species": "...", "confidence": 0.95}
```

BentoML auto-generates both REST and gRPC endpoints from the same definition. The same service serves both protocols.

```bash
# REST
curl -X POST http://localhost:3000/predict -d '{"features": [...]}'

# gRPC (requires grpcurl)
grpcurl -plaintext -d '{"features": [5.1, 3.5, 1.4, 0.2]}' localhost:3000 iris_service.PredictService/predict
```

---

## 8. Custom Validators and Pre-processors

```python
from pydantic import BaseModel, Field, field_validator
import bentoml


class PredictRequest(BaseModel):
    features: list[float]
    
    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError("features must have exactly 4 elements")
        if any(x < 0 or x > 10 for x in v):
            raise ValueError("features must be in [0, 10]")
        return v


@bentoml.service
class ValidatedService:
    @bentoml.api
    def predict(self, req: PredictRequest) -> dict:
        # Inputs are guaranteed valid (Pydantic validation)
        return {"prediction": self.model.predict([req.features]).tolist()}
```

Pydantic validators run **before** the endpoint code; invalid inputs return 422 with details.

---

## 9. Background Tasks and Lifecycle

For setup/teardown logic:

```python
@bentoml.service
class LifecycleService:
    model = bentoml.models.BentoModel("model:latest")
    
    def __init__(self):
        """Called at startup."""
        import joblib
        self.model = joblib.load(self.model.path_of("model.pkl"))
        
        # Pre-compute shared state
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Service started; model loaded")
    
    def __del__(self):
        """Called at shutdown."""
        print("Service shutting down")
    
    @bentoml.api
    def predict(self, text: str) -> dict:
        # Use pre-loaded model and embedder
        embedding = self.embedder.encode(text)
        return self.model.predict([embedding]).tolist()
```

---

## 10. Error Handling

```python
@bentoml.service
class RobustService:
    model = bentoml.models.BentoModel("model:latest")
    
    def __init__(self):
        self.model_runner = bentoml.load_model_runner(self.model)
    
    @bentoml.api
    def predict(self, input_data: dict) -> dict:
        try:
            result = self.model_runner.predict(input_data)
            return {"status": "ok", "prediction": result.tolist()}
        except ValueError as e:
            # Custom error message
            return {"status": "error", "message": str(e)}
        except Exception as e:
            # Log and re-raise
            logger.exception("Prediction failed")
            raise
```

BentoML automatically returns 4xx for ValueError and 5xx for other exceptions. Add custom error responses as needed.

---

## 11. Antipatterns

### 11.1 Antipattern 1: Returning raw numpy arrays

```python
# ❌ Returns raw numpy; client can't deserialize
@bentoml.api
def predict(self, input_data) -> np.ndarray:
    return self.model.predict(input_data)

# ✅ Return serializable types (dict, list, Pydantic model)
@bentoml.api
def predict(self, input_data) -> dict:
    prediction = self.model.predict(input_data)
    return {"prediction": prediction.tolist()}
```

### 11.2 Antipattern 2: Not using batchable for high-throughput workloads

```python
# ❌ One request at a time; throughput limited by per-request latency
@bentoml.api
def predict(self, input_data):
    return self.model.predict(input_data)

# ✅ Batchable for low-latency, high-throughput
@bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=100)
def predict(self, inputs: list):
    return self.model.predict_batch(inputs)
```

### 11.3 Antipattern 3: Synchronous endpoint for long inference

```python
# ❌ Blocks worker during 10-second LLM call
@bentoml.api(batchable=False)
def generate(self, prompt: str) -> str:
    return self.model.generate(prompt)  # 10 seconds; blocks worker

# ✅ Async endpoint releases worker
@bentoml.api(batchable=False)
async def generate(self, prompt: str) -> str:
    return await self.model.async_generate(prompt)
```

### 11.4 Antipattern 4: No validation on input

```python
# ❌ Trust input; errors at model layer
@bentoml.api
def predict(self, input_data):
    return self.model.predict(input_data)  # might fail on invalid input

# ✅ Validate at API layer with Pydantic
@bentoml.api
def predict(self, input: PredictRequest) -> PredictResponse:
    return self.model.predict([input.features]).tolist()
```

### 11.5 Antipattern 5: Heavy work in __init__

```python
# ❌ Slow startup; service not ready for 30+ seconds
def __init__(self):
    self.big_model = load_huge_model()  # 30 seconds!
    self.precomputed_index = build_large_index()  # 60 seconds!

# ✅ Lazy initialization or use smaller model
def __init__(self):
    self.model = bentoml.load_model("model:latest")  # fast
    # Lazy load expensive resources on first request
    self._big_index = None
    
    def get_index(self):
        if self._big_index is None:
            self._big_index = build_large_index()
        return self._big_index
```

---

## 🎯 Key Takeaways

- Use Pydantic for type-safe I/O; auto-generates OpenAPI specs.
- `batchable=True` for high-throughput workloads; BentoML handles adaptive batching.
- Async endpoints for long-running inference; releases worker during I/O.
- Streaming endpoints via `async generator` for LLM token streaming.
- Multiple endpoints per service: predict, predict_batch, explain, health.
- IO descriptors: JSON, Image, File, Text.
- gRPC auto-generated from same definitions as REST.
- Pydantic validators run before the endpoint; 422 on invalid input.
- Avoid raw numpy returns, no batching for high throughput, sync endpoints, no validation, heavy init.

## References

- BentoML Service Patterns — [docs.bentoml.com/en/latest/reference/service](https://docs.bentoml.com/en/latest/reference/service.html)
- BentoML IO Descriptors — [docs.bentoml.com/en/latest/reference/io_descriptors](https://docs.bentoml.com/en/latest/reference/io_descriptors/)
- BentoML batching — [docs.bentoml.com/en/latest/guides/batching](https://docs.bentoml.com/en/latest/guides/batching.html)
- BentoML examples — [github.com/bentoml/BentoML/tree/main/examples](https://github.com/bentoml/BentoML/tree/main/examples)
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM]] — LLM serving comparison
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — custom FastAPI alternative
- [[09 - MLOps y Produccion/42 - BentoML Production Model Serving/01 - BentoML Fundamentals|Note 01 — Fundamentals]]
- [[09 - MLOps y Produccion/42 - BentoML Production Model Serving/03 - Distributed Serving - Multi-model, Ensembles, GPU Sharing|Note 03 — Distributed Serving]]