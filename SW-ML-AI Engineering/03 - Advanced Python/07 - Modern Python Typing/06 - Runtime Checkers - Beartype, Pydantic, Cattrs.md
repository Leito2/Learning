# 🛡️ Runtime Checkers — Beartype, Pydantic, Cattrs

Static typing catches bugs **before they run**. Runtime checkers catch them **at the moment they matter**. For an ML/AI engineering codebase, the boundary is sharp: untrusted input (HTTP request bodies, LLM JSON outputs, file uploads, environment variables) needs runtime validation; pure internal code paths are statically checked. The question is which runtime checker to pick.

Three libraries dominate: **Pydantic v2** (the FastAPI/LangChain standard for validated data models), **Beartype** (a drop-in `@beartype` decorator that adds runtime checks based on annotations, ~10× faster than Pydantic), and **Cattrs** (a serialization/deserialization library that pairs with attrs and dataclasses). Each fits a different scenario. This note covers the choice matrix, the integration patterns, and the production realities.

## 🎯 Learning Objectives

- Decide between Pydantic, Beartype, and Cattrs based on the use case.
- Use Pydantic v2's `BaseModel` for validated data classes.
- Add Beartype to functions for runtime type assertions.
- Use Cattrs for high-throughput serialization/deserialization.
- Compose runtime checkers with strict static typing ([[05 - Mypy and Pyright Strict Mode in CI|note 05]]).
- Apply runtime checks at the input boundary (request bodies, LLM outputs) only.

## 1. The Input Boundary Problem

```python
# 99% of code is statically checked
def process_user(user: User) -> Response:
    return Response.build(user)

# 1% is the boundary — data comes from outside the trust zone
def endpoint(request_body: object) -> Response:
    # request_body has been JSON-parsed; the type system says object
    # But the runtime could be anything
    ...
```

The 1% boundary is where runtime validation lives. The 99% internal code should be statically checked (mypy/pyright strict) but doesn't need runtime checks — the type system guarantees it.

```mermaid
flowchart LR
    HTTP[HTTP Request Body] --> Parsed[JSON.parse → object]
    Parsed --> Validator[Runtime Validator]
    Validator -->|narrowed to User| Internal[Internal Code (statically typed)]
    Internal --> Response[HTTP Response]
    style Parsed fill:#fbbf24
    style Validator fill:#f87171,color:#fff
    style Internal fill:#34d399
```

## 2. Pydantic v2 — The Default Choice

```python
from pydantic import BaseModel, Field, field_validator
from typing import Annotated

class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str
    age: int = Field(ge=0, le=150)
    tags: list[str] = Field(default_factory=list)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("invalid email")
        return v.lower()
```

Pydantic v2 is written in Rust (`pydantic-core`); a model with 10 fields validates in <100µs. For most ML/AI applications this is fast enough.

### FastAPI Boundary

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict | None = None

@app.post("/search")
async def search(req: QueryRequest) -> dict:
    # req is already validated
    return await search_engine(req.query, req.top_k, req.filters)
```

FastAPI uses Pydantic for request/response models — the entire boundary is type-safe.

### LangChain Tool-Call Arguments

```python
from pydantic import BaseModel
from langchain_core.tools import tool

class SearchArgs(BaseModel):
    query: str
    top_k: int = 5

@tool(args_schema=SearchArgs)
def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the web."""
    return tavily.search(query, top_k=top_k)

# LangChain validates LLM-returned args against SearchArgs before calling
```

> 💡 **Tip:** Pydantic is the production choice for LLM tool-call arguments, FastAPI request bodies, and any data class that crosses a trust boundary. See [[../../../06 - Large Language Models/12 - Production RAG/06 - Pydantic for RAG - FastAPI, LangChain and LlamaIndex.md|the vault's Pydantic for RAG note]] for full coverage.

## 3. Beartype — Drop-In Runtime Checks

```python
from beartype import beartype

@beartype
def add(a: int, b: int) -> int:
    return a + b

add(1, 2)        # OK
add("hi", "world")  # ❌ Beartype raises BeartypeCallHintException
```

Beartype wraps any function and adds runtime checks based on its annotations. No code change; just a decorator.

### Speed: ~10× Faster Than Pydantic for Simple Checks

```python
# Benchmark (rough, machine-dependent)
from beartype import beartype
import timeit

@beartype
def check_int(x: int) -> int:
    return x * 2

# beartype: 0.1µs per call
# isinstance(x, int): 0.05µs per call
# Pydantic model_validate(int): 1.0µs per call
```

For hot-loop code (think inner training loops, per-token LLM preprocessing), Beartype is the right tool.

### Beartype with Generics and Protocols

```python
from beartype import beartype
from typing import Iterable

@beartype
def sum_safe(items: Iterable[int]) -> int:
    return sum(items)

sum_safe([1, 2, 3])    # OK
sum_safe(["a", "b"])   # ❌ Beartype raises on non-int
```

Beartype understands `Iterable`, `list[X]`, `dict[K, V]`, `tuple[X, ...]`, `Callable[[X], Y]`, and PEP 695 generics.

### Beartype as a Decorator for Production Code

```python
@beartype
def critical_pipeline_step(features: list[list[float]], model_id: str) -> dict:
    ...
```

Use Beartype on **public APIs and library boundaries**. Internal private functions can skip Beartype since mypy strict handles them at static check time.

## 4. Cattrs — High-Throughput Serialization

Cattrs pairs with `attrs` (or `dataclasses`) for fast (de)serialization:

```python
from attrs import define
import cattrs

@define
class Point:
    x: float
    y: float

# Structure (untyped dict, e.g., from JSON)
data = {"x": 1.0, "y": 2.0}

# Convert structure to typed instance
converter = cattrs.Converter()
p = converter.structure(data, Point)
print(p)  # Point(x=1.0, y=2.0)

# Convert typed instance to dict (for storage)
d = converter.unstructure(p)
print(d)  # {'x': 1.0, 'y': 2.0}
```

Cattrs is **3-5× faster than Pydantic** for (de)serialization of plain dataclass-style classes. For large-scale ETL (millions of records), Cattrs wins.

### Cattrs vs Pydantic Performance

```python
# Cattrs (with attrs): ~3-5× faster than Pydantic v2
# Pydantic v2 with simple types: slow path (Rust core, but JSON parse)
# Beartype: ~10× faster than Pydantic for primitive checks
```

Choose Cattrs for **ETL pipelines** (millions of records/sec). Choose Pydantic for **LLM tool-call validation** (rich validators). Choose Beartype for **function-level runtime checks**.

## 5. The Decision Matrix

| Use case | Tool | Reason |
|----------|------|--------|
| FastAPI request/response | **Pydantic** | Built-in integration; rich validators; JSON Schema export |
| LangChain tool-call args | **Pydantic** | `@tool(args_schema=BaseModel)` works out of the box |
| LLM JSON output parsing | **Pydantic** | Rich validators + auto-retry on validation errors |
| High-throughput function arg checks | **Beartype** | ~10× faster than Pydantic |
| Dataclass serialization (ETL) | **Cattrs** | Fastest (de)serialization |
| ML training pipeline internals | **mypy strict only** | Static checks; no runtime overhead |
| Untrusted CLI args / env vars | **Pydantic Settings** | Validation + parsing |
| Hot loops with annotations | **Beartype** | Lowest per-call overhead |
| FastAPI + LangChain + Agents | **Pydantic v2 (Rust core)** | Default choice |

## 6. Composition Patterns

### Pydantic at the Boundary + Beartype Internally

```python
from pydantic import BaseModel
from beartype import beartype

# Boundary: Pydantic validates the HTTP body
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Internal: Beartype checks function args
@beartype
def process_search(req: SearchRequest) -> list[float]:
    embedder.embed(req.query)
    ...

# Both work together: Pydantic validates the request, Beartype checks the in-process call
```

### Pydantic Settings for Env Vars

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str
    database_url: str
    debug: bool = False

config = AppConfig()  # validates env vars at construction
```

This is the production way to load environment variables — typed, validated, documented.

### Beartype on Decorator Output

```python
from beartype import beartype
from typing import Callable, ParamSpec, TypeVar
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def beartyped(func: Callable[P, R]) -> Callable[P, R]:
    # Beartype does the runtime check at call time
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)
    return beartype(wrapper)
```

## 7. ❌/✅ Antipatterns

### ❌ Pydantic everywhere, even internal hot loops

```python
# ❌ Pydantic validation on every token generated — slow
class TokenOutput(BaseModel):
    token: str
    logprob: float
    embedding: list[float]

@beartype  # if you must validate per token
def process_token(t: TokenOutput) -> None:
    ...
```

### ✅ Beartype on hot loops; Pydantic on boundaries

```python
# ✅ Pydantic at the boundary, raw types internally
class GenerationRequest(BaseModel): ...

def generate(req: GenerationRequest):
    for token in llm.stream(req.prompt):  # token is dict[str, Any]; no validation
        process(token)
```

### ❌ Beartype on every internal function

```python
# ❌ Adds overhead to 99% of code that's already statically checked
@beartype
def internal_helper(x: int) -> int:
    return x + 1
```

### ✅ Beartype at boundaries only

```python
@beartype  # at the public API
def public_api(input_data: dict) -> dict:
    ...

def internal_helper(x: int) -> int:  # no beartype — statically checked
    return x + 1
```

### ❌ Cattrs for LLM tool calls

```python
# ❌ Cattrs doesn't have validators; you write them manually
@define
class ToolCall:
    name: str
    args: dict

# No rich validators; manual JSON Schema gen; cumbersome
```

### ✅ Pydantic for LLM tool calls (built-in validators + JSON Schema)

```python
class ToolCall(BaseModel): ...
```

## 8. Production Reality

**Caso real — FastAPI + LangChain agent:** Pydantic v2 at the request boundary (FastAPI dependency), Pydantic for LLM tool-call args (LangChain integration), Beartype on internal hot-path functions (`@beartype def process_embedding(...)`). mypy strict in CI. Total runtime validation overhead: <50µs per request.

**Caso real — ETL pipeline processing 10M records:** Switched from Pydantic (`BaseModel.parse_obj(...)`) to Cattrs (`converter.structure(data, RecordCls)`) for 4× speedup. Records per second: 250K → 1M. Validation skipped because records come from a trusted internal source; mypy strict handles the typing.

**Caso real — LLM Evaluation Suite:** All LLM JSON outputs are validated by Pydantic `BaseModel`. Validation errors trigger an automatic retry with a corrective prompt. This is the production-grade approach to LLM output parsing.

## 📦 Compression Code

```python
# 📦 Compression: runtime checkers in one file
# Requires: pip install pydantic beartype attrs cattrs

# === Pydantic: validate at boundary ===
from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)

# Validates dict input → typed model
req = QueryRequest.model_validate({"query": "hello", "top_k": 10})
print(req.query, req.top_k)  # hello 10

# === Beartype: function-level checks ===
from beartype import beartype

@beartype
def add(a: int, b: int) -> int:
    return a + b

print(add(1, 2))  # 3
# add("1", "2")  # raises BeartypeCallHintException

# === Cattrs: high-speed structure ===
from attrs import define
import cattrs

@define
class Point:
    x: float
    y: float

converter = cattrs.Converter()
p = converter.structure({"x": 1.0, "y": 2.0}, Point)
print(p)  # Point(x=1.0, y=2.0)
print(converter.unstructure(p))  # {'x': 1.0, 'y': 2.0}

# === Combined: Pydantic + Beartype ===
class Item(BaseModel):
    name: str
    price: float = Field(gt=0)

@beartype
def process_items(items: list[Item]) -> float:
    return sum(item.price for item in items)

items = [Item(name="apple", price=0.5), Item(name="bread", price=2.5)]
print(process_items(items))  # 3.0
```

## 🎯 Key Takeaways

1. **Pydantic v2 is the default for boundary validation** — FastAPI, LangChain, LLM outputs. Written in Rust, fast, rich validators, JSON Schema export.
2. **Beartype is the drop-in decorator** for function-level runtime checks. ~10× faster than Pydantic for primitive validation.
3. **Cattrs is the high-throughput serializer** for ETL pipelines. 3-5× faster than Pydantic for (de)serialization of attrs/dataclass-style classes.
4. **Validate at the boundary only.** Internal functions are statically checked by mypy strict; runtime checks add overhead for no gain.
5. **Pydantic Settings** for env vars — typed config with validation.
6. **Avoid `Any` to silence checkers**; prefer precise types + `TypeGuard` (note 02) + runtime checkers.
7. **Compose**: Pydantic at boundary, Beartype at hot APIs, Cattrs at ETL, mypy strict everywhere.

## References

- [[02 - Type Narrowing - TypeGuard, isinstance and Asserts|Type Narrowing]] — `TypeGuard` predicates complement runtime checkers.
- [[05 - Mypy and Pyright Strict Mode in CI|Mypy / Pyright]] — strict mode turns annotations into checks; runtime checkers are the complement.
- [[07 - Capstone - Strict-Typed ML Pipeline|Capstone]] — combines Pydantic + Beartype + mypy strict in one ML pipeline.
- [[../../../06 - Large Language Models/12 - Production RAG/06 - Pydantic for RAG - FastAPI, LangChain and LlamaIndex.md|Pydantic for RAG]] — full coverage of Pydantic in production.
- Beartype: https://beartype.readthedocs.io/
- Cattrs: https://cattree.readthedocs.io/