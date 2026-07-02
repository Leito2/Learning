# Performance and Production Patterns

## 1. TypeAdapter Reuse

Creating `TypeAdapter` once and reusing it avoids repeated schema compilation:

```python
from pydantic import TypeAdapter

# BAD: creates schema on every call
def parse_positive(data: str) -> int:
    return TypeAdapter(int, config={"ge": 0}).validate_json(data)

# GOOD: compile once
_positive_int = TypeAdapter(int, config={"ge": 0})
def parse_positive(data: str) -> int:
    return _positive_int.validate_json(data)
```

## 2. Validation Cost Profiling

```python
import time

def benchmark(model_cls, data: dict, iterations: int = 10000):
    start = time.perf_counter()
    for _ in range(iterations):
        model_cls.model_validate(data)
    elapsed = time.perf_counter() - start
    print(f"{model_cls.__name__}: {elapsed/iterations*1e6:.1f} us per call")
```

Pydantic v2's Rust core: ~1-5 microseconds per simple model, ~10-50 us for nested.

## 3. Strict Mode for Hot Paths

```python
class APIRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    user_id: int
    amount: float
```

Strict mode avoids coercion overhead but rejects string inputs. Use for internal service-to-service calls where types are guaranteed.

## 4. model_dump_json vs json.dumps(model_dump())

```python
# Faster: direct to JSON string
json_str = model.model_dump_json()

# Slower: intermediate dict
json_str = json.dumps(model.model_dump())
```

`model_dump_json()` skips the intermediate dict allocation. Benchmark: ~2x faster for large models.

## 5. Validation on Boundaries Only

Validate at API boundaries (FastAPI handlers, message queue consumers), not in internal functions:

```python
# BAD: validate in every function
def process_item(raw: dict):
    item = Item.model_validate(raw)  # expensive
    ...

# GOOD: validate at entry point, pass typed objects
@app.post("/items")
def create_item(item: Item):
    internal_process(item)  # already validated

def internal_process(item: Item):
    ...
```

## 6. Error Handling Patterns

### Structured API Errors

```python
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

app = FastAPI()

@app.post("/items")
def create_item(raw: dict):
    try:
        item = Item.model_validate(raw)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
                    for err in e.errors()]
        )
```

### Graceful Degradation with Partial Validation

```python
class PartialItem(BaseModel):
    model_config = ConfigDict(extra="allow")  # unknown fields pass through

    name: str | None = None
    price: float | None = None
```

## 7. Capstone: Production Config-Driven Inference API

This capstone combines: `BaseModel`, `Field`, validators, `model_config`, serialization, settings, and JSON Schema into a single ML inference API.

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
import numpy as np

# --- Settings ---

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    model_name: str = "default-model"
    max_batch_size: int = Field(default=32, ge=1, le=256)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_feature_store: bool = False

# --- Request Schema ---

class FeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    values: list[float] = Field(min_length=1, max_length=1024)

    @field_validator("values", mode="after")
    @classmethod
    def validate_range(cls, v: list[float]) -> list[float]:
        if any(x < -1e6 or x > 1e6 for x in v):
            raise ValueError("features out of range")
        return v

class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    features: FeatureVector
    model_version: Literal["v1", "v2"] = "v1"

# --- Response Schema ---

class Prediction(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)

class InferenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    predictions: list[Prediction] = Field(min_length=1)
    model_version: str
    latency_ms: float = Field(ge=0.0)

    @computed_field
    @property
    def num_predictions(self) -> int:
        return len(self.predictions)

# --- FastAPI Integration ---

from fastapi import FastAPI, HTTPException

app = FastAPI()
settings = Settings()

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    try:
        scores = np.random.rand(3).tolist()
        predictions = [
            Prediction(score=s, label=f"class_{i}", confidence=s)
            for i, s in enumerate(scores)
        ]
        return InferenceResponse(
            request_id=request.request_id,
            predictions=predictions,
            model_version=request.model_version,
            latency_ms=42.5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Key Takeaways

- Reuse `TypeAdapter` instances to avoid schema compilation overhead
- `model_dump_json()` is faster than `json.dumps(model_dump())`
- Validate at API boundaries, pass typed objects internally
- `strict=True` on internal paths for faster validation
- Profile validation cost with simple time benchmarks
- Capstone pattern: settings → request schema → validators → computed fields → response schema
