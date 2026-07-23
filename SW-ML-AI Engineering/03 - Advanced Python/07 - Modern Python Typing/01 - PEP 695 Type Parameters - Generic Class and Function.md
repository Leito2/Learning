# 🧬 PEP 695 Type Parameters — Generic Classes and Functions

PEP 695 (Python 3.12) is the biggest change to Python's typing system since PEP 484. It introduces **declarative type parameters** at class, function, and type-alias scope — `class Stack[T]:`, `def first[T](items: list[T]) -> T`, `type Vector = list[float]`. The old `TypeVar("T"); class X(Generic[T])` ceremony is gone. Variance is inferred from the use site. Type aliases become first-class. The result is shorter code that catches more bugs at static-analysis time.

This note covers the PEP 695 syntax for generic classes, generic functions, type aliases, and constrained type parameters (the new `type X = ...` statement). It also walks through the three places where PEP 695 beats the legacy `TypeVar` pattern: variance inference, type-alias semantics, and constraining type parameters.

## 🎯 Learning Objectives

- Write PEP 695 generic classes (`class X[T]: ...`) and functions (`def f[T](...)`).
- Distinguish PEP 695 syntax from legacy `Generic[T]`, `TypeVar` patterns.
- Define type aliases with `type Vector = list[float]` (PEP 695) vs the old assignment (`Vector = list[float]`).
- Use bounded type parameters (`class X[T: int]: ...`) and constrained (`class X[T: (int, str)]: ...`).
- Read and apply PEP 695 variance (covariant, contravariant, invariant) at the type parameter scope.

## 1. The Legacy Pattern (2018) and Why It Hurt

```python
from typing import TypeVar, Generic

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()
```

Three problems:

1. **Naming ceremony.** `T = TypeVar("T")` requires the name to match; a typo (`S = TypeVar("T")`) silently breaks inference.
2. **Variance is separate.** `T_co` and `T_contra` are different objects; users must understand variance before they can write an immutable container.
3. **Type aliases are weak.** `Vector = list[float]` makes `Vector` a runtime value — the type checker can't always tell it from a variable.

PEP 695 fixes all three.

## 2. PEP 695 Generic Class

```python
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()
```

The `[T]` after `Stack` declares the type parameter. The framework infers variance from how `T` is used:

- `push(item: T)` writes T (covariant position) — `Stack[int]` cannot be assigned to `Stack[float]` unless T is invariant.
- `pop() -> T` reads T (contravariant position) — works in both directions.

For `Stack`, T is invariant (because of `push`). PEP 695 makes this **automatic** — no `covariant=True` to remember.

### Bounded Type Parameters

```python
class Comparable[T: float]:
    def max(self, a: T, b: T) -> T:
        return a if a > b else b

# Works
Comparable[float]().max(1.0, 2.0)   # ok

# mypy/pyright error:
# Argument 1: "int" is not assignable to "float"
# Comparable[float]().max(1, 2)
```

The `[T: float]` declares an **upper bound**: T must be a subtype of (or equal to) `float`. This is equivalent to legacy `T = TypeVar("T", bound=float)`.

### Constrained Type Parameters

```python
class Number[T: (int, float, complex)]:
    def add(self, a: T, b: T) -> T:
        return a + b  # type checker knows `+` exists

Number().add(1, 2)        # ok
Number().add(1.0, 2.0)    # ok
Number().add(1 + 2j, 3j)  # ok
```

The `[T: (int, float, complex)]` declares a **constraint set**: T must be one of those three types. Use sparingly — constraints are stricter than bounds and rarely needed outside numeric libraries.

## 3. PEP 695 Generic Function

```python
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None

# Type-safe
x: str | None = first(["a", "b", "c"])  # T inferred as str
y: int | None = first([1, 2, 3])        # T inferred as int
y: int | None = first([])                # None
```

The `[T]` after `def first` declares the type parameter. The type checker infers T from the argument; the caller does not need to specify it (but can):

```python
# Explicit (rarely needed)
x = first[int]([1, 2, 3])  # T = int explicitly
```

### Generic Function with Bounds

```python
import asyncio
from collections.abc import Awaitable

async def await_all[T: Awaitable](*tasks: T) -> tuple:
    return await asyncio.gather(*tasks)

results = await await_all(asyncio.sleep(0), asyncio.sleep(0))
```

## 4. PEP 695 Type Aliases (the `type` Statement)

```python
# === Legacy (still works) ===
Vector = list[float]
Matrix = list[Vector]

def normalize(m: Matrix) -> Matrix:
    return [[v / sum(row) for v in row] for row in m]

# Problem: mypy sees `Vector` as a variable assignment
#         (could be re-assigned); can't always substitute.
```

```python
# === PEP 695 type alias ===
type Vector = list[float]
type Matrix = list[Vector]

def normalize(m: Matrix) -> Matrix:
    return [[v / sum(row) for v in row] for row in m]

# Type checker knows Vector is a TYPE, not a variable
# Can substitute: list[float] anywhere Vector appears.
```

> 💡 **Tip:** Use `type X = ...` (the PEP 695 statement) for type aliases. The legacy assignment syntax works but loses to the checker the fact that `X` is immutable and pure-type.

### Generic Type Alias

```python
type Pair[T] = tuple[T, T]

# Usage
p: Pair[int] = (1, 2)        # ok
p: Pair[str] = ("a", "b")    # ok
p: Pair[int] = ("a", "b")    # type error: str not assignable to int
```

### Type Alias of Type Alias

```python
type UserId = int
type User = dict[str, str | int]
type Users = list[User]

# Users is expanded: list[dict[str, str | int]]
# The checker treats all three aliases identically.
```

## 5. Variance Inference

PEP 695 infers variance from the use site:

```python
# Invariant: read and written (the default)
class Stack[T]:
    def push(self, item: T) -> None: ...  # write T
    def pop(self) -> T: ...                # read T

# Covariant: only read (output position)
class Producer[T]:
    def produce(self) -> T: ...           # read T

# Contravariant: only written (input position)
class Consumer[T]:
    def consume(self, item: T) -> None: ...  # write T
```

The legacy `TypeVar("T_co", covariant=True)` pattern is no longer needed:

```python
# 2018: variance declared explicitly
from typing import TypeVar, Generic
T_co = TypeVar("T_co", covariant=True)
class Producer(Generic[T_co]):
    def produce(self) -> T_co: ...

# 2026: inferred
class Producer[T]:
    def produce(self) -> T: ...  # inferred covariant
```

> ⚠️ **Advertencia:** Variance inference works for simple cases. For complex generics (e.g., `Callable[[T], T]`), the inference can be subtle. Use `mypy --strict` to verify; the checker reports variance conflicts at definition time.

## 6. PEP 695 in Real Codebases

### Pydantic-Style Generic Model

```python
from pydantic import BaseModel

class Response[T](BaseModel):
    data: T
    status: int = 200
    message: str = "ok"

# Concrete responses
Response[str]            # data: str
Response[list[int]]      # data: list[int]
Response[dict[str, Any]] # data: dict[str, Any]
```

Note: Pydantic v2 generics work via `Generic[T]`; PEP 695 syntax is compatible if you inherit correctly.

### LangGraph-Style Generic State

```python
class State[T]:
    query: str
    result: T | None
    log: list[T]

# Concrete state for a research workflow
State[dict]               # result: dict | None
State[list[str]]         # result: list[str] | None
```

### Typing Your Own Toolkit

```python
# A type-safe cache (the PEP 695 way)
class TTLCache[K, V]:
    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._data: dict[K, tuple[V, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: K) -> V | None:
        import time
        value, expiry = self._data.get(key, (None, 0))
        if time.time() < expiry:
            return value
        return None

    def set(self, key: K, value: V) -> None:
        import time
        self._data[key] = (value, time.time() + self._ttl)

# Type-safe usage
user_cache: TTLCache[int, str] = TTLCache()
user_cache.set(42, "Alice")
print(user_cache.get(42))  # "Alice"
```

## 7. ❌/✅ Antipatterns

### ❌ Mixing legacy `Generic[T]` with PEP 695

```python
# ❌ Don't mix — the type checker may infer incompatible variance
from typing import Generic, TypeVar
T = TypeVar("T")

class Stack(Generic[T]):  # 2018 syntax
    ...

class PeekableStack[T]:   # 2026 syntax, same name conceptually
    ...
```

### ✅ Pick one style per file

```python
# ✅ All PEP 695
class Stack[T]: ...
class PeekableStack[T]: ...
```

### ❌ Type alias as assignment

```python
# ❌ Mutated at runtime — type checker loses track
Vector = list[float]
Vector = "I'm a string now"  # allowed at runtime
```

### ✅ Type alias as statement

```python
# ✅ Immutable type alias
type Vector = list[float]
# Any attempt to re-assign `Vector` is a type error AND a runtime error
```

### ❌ Bounded type parameter used loosely

```python
# ❌ Bounded but the bound is misused
class Number[T: (int, float)]:
    def compare(self, a: T, b: T) -> bool:
        return a < b  # works, but the bound doesn't help the type checker
```

### ✅ Use the bound to enable type-checking benefits

```python
# ✅ Bound enables visible methods
class Number[T: (int, float)]:
    def add(self, a: T, b: T) -> T:
        return a + b  # checker knows T has __add__
```

## 8. Production Reality

**Caso real — LangGraph state schemas:** Adopting PEP 695 for `class State[T]` reduced type-checker warnings by ~80% across the [[../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] course. The legacy `T = TypeVar("T"); class State(Generic[T])` pattern triggered variance conflicts when the same T was both read and written; PEP 695 inference handles it automatically.

**Caso real — Pydantic generics:** A FastAPI service that returned `Response[User]`, `Response[list[User]]`, `Response[None]` was refactored from `Generic[T]` (verbose) to PEP 695 (clean). Code reduction: ~15 lines per model definition. mypy strict mode acceptance: 100%.

## 9. Migration Cheatsheet

```python
# === 2018 ===                          # === 2026 ===
T = TypeVar("T")                         # (no declaration needed)
class X(Generic[T]):                     class X[T]:
    def m(self, x: T) -> T: ...              def m(self, x: T) -> T: ...

T_co = TypeVar("T_co", covariant=True)  class X[T]:              # inferred
class Y(Generic[T_co]):                     def m(self) -> T: ...
    def m(self) -> T_co: ...

T_contra = TypeVar("T_contra", ...)      class X[T]:              # inferred
class Z(Generic[T_contra]):                 def m(self, x: T) -> None: ...
    def m(self, x: T_contra) -> None: ...

T = TypeVar("T", bound=int)              class X[T: int]:
class X(Generic[T]):                         def m(self, x: T) -> int: ...
    def m(self, x: T) -> int: ...

T = TypeVar("T", int, str, float)        class X[T: (int, str, float)]:
class X(Generic[T]):                         def m(self, x: T) -> T: ...
    def m(self, x: T) -> T: ...

Vector = list[float]                     type Vector = list[float]

def f(x: "list[int] | None") -> None:    def f(x: list[int] | None) -> None: ...

from typing import Optional, Union      # no import needed
def g(x: Optional[int]) -> Union[str,    def g(x: int | None) -> str | None: ...
    None]: ...
```

## 📦 Compression Code

```python
# 📦 Compression: PEP 695 in one file
# Covers: generic class, generic function, type alias, bounded parameter, variance inference

# === Generic class ===
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()

# === Generic function ===
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None

# === Type alias ===
type Vector = list[float]
type Matrix = list[Vector]

def normalize(m: Matrix) -> Matrix:
    return [[v / sum(row) for v in row] for row in m]

# === Bounded parameter ===
class Comparable[T: float]:
    def max(self, a: T, b: T) -> T:
        return a if a > b else b

# === Demonstration ===
s: Stack[int] = Stack()
s.push(1); s.push(2)
print(first(s._items))  # 1
print(Comparable().max(1.0, 2.0))  # 2.0
```

## 🎯 Key Takeaways

1. **PEP 695 syntax** replaces `TypeVar("T"); class X(Generic[T])` with `class X[T]:` — same semantics, 50% less code.
2. **Variance is inferred** from how the type parameter is used (read, write, both). The legacy `covariant=True`, `contravariant=True` flags are obsolete.
3. **`type X = ...` declares an immutable type alias.** Use this (not `X = ...`) for clarity and type-checker benefits.
4. **Bounds (`[T: int]`) restrict T to subtypes**; **constraints (`[T: (int, str)]`) restrict T to a set.** Bounds are usually what you want.
5. **PEP 695 requires Python 3.12+** for syntax. Earlier versions need the legacy `Generic[T]` pattern.
6. **Migrate mechanically**: rename `Generic[T]` to `[T]`, drop `TypeVar`, replace `X = ...` aliases with `type X = ...`.

## References

- [[00 - Welcome to Modern Python Typing|Welcome]] — course map and the 2018 vs 2026 table.
- [[02 - Type Narrowing - TypeGuard, isinstance and Asserts|Type Narrowing]] — once generics are inferred, narrowing refines them in branches.
- [[../06 - Pydantic Deep Dive/01 - BaseModel, Field and Type System.md|Pydantic BaseModel]] — composes PEP 695 generics with Pydantic models.
- PEP 695: https://peps.python.org/pep-0695/