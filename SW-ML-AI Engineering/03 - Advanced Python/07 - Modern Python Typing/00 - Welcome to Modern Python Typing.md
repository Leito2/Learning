# 🔷 Welcome to Modern Python Typing

You have written Python for years. You use `def f(x: int) -> str` and `from typing import List, Dict`. But the typing system Python ships with in 2026 is **fundamentally different** from the one you learned in 2018. PEP 695 (Python 3.12) introduced **type parameters** — `class Stack[T]`, `def first[T](items: list[T]) -> T` — replacing the awkward `TypeVar` and `Generic[T]` ceremony. PEP 604 (3.10) collapsed `Union[X, Y]` into `X | Y`. PEP 647 (3.10) added `TypeGuard` for runtime narrowing. PEP 673 (3.11) introduced `Self` for fluent APIs. The result is a typing system that catches more bugs, requires less boilerplate, and integrates with modern tools (mypy, pyright, beartype) that turn "hints" into actual correctness guarantees.

Until now, the vault had exactly one dedicated typing note: [[../03 - Python Avanzado/05 - Type Hints y Anotaciones.md|Type Hints y Anotaciones]] in Spanish, covering `TypeVar`, `Protocol`, `Callable`, and `NamedTuple`. That note is correct but stops at the 2018 line. This course closes the gap to 2026.

This is the **English counterpart** to that Spanish note, focused on the modern typing surface: PEP 695 generics, type narrowing, structural subtyping, fluent APIs, strict static analysis, and runtime checkers. Every note pairs a concept with the **AI/ML/Backend context** where it pays off — Pydantic, LangChain, FastAPI, LangGraph, mypy in CI.

## 🎯 Learning Objectives

- Read and write **PEP 695 type parameter syntax** (`class X[T]`, `def f[T](...)`).
- Distinguish **static typing** (mypy, pyright) from **runtime checkers** (beartype, pydantic).
- Use `TypeGuard`, `isinstance`, and `assert` to **narrow** types in conditional branches.
- Apply `typing.Protocol` for structural subtyping (duck typing made explicit).
- Use `Self`, `ParamSpec`, `Concatenate` for fluent APIs, decorators, and partial application.
- Configure mypy `--strict` (or pyright `strict`) in a CI pipeline.
- Pick the right runtime validation layer (Pydantic vs beartype vs cattrs) for your stack.

## Course Map

| # | Note | Core Concept | Portfolio Hook |
|:-:|------|--------------|----------------|
| 00 | [[00 - Welcome to Modern Python Typing\|You are here]] | Why 2026 typing is different from 2018 typing | Course map |
| 01 | [[01 - PEP 695 Type Parameters - Generic Class and Function\|PEP 695]] | `class X[T]`, `def f[T](...)`, type alias syntax | Pydantic-style generics, type-safe containers |
| 02 | [[02 - Type Narrowing - TypeGuard, isinstance and Asserts\|Type Narrowing]] | `TypeGuard`, `isinstance`, smart cast | LangChain output parsers, JSON schema validation |
| 03 | [[03 - Protocols and Structural Subtyping\|Protocols]] | `Protocol`, runtime checkable, default methods | Duck typing without inheritance |
| 04 | [[04 - Self, ParamSpec and Concatenate\|Self, ParamSpec, Concatenate]] | 3.10-3.12 additions for fluent APIs and decorators | Builder pattern, decorator factories |
| 05 | [[05 - Mypy and Pyright Strict Mode in CI\|Strict Static Analysis]] | `mypy --strict`, `pyright strict`, CI integration | Catching bugs before runtime |
| 06 | [[06 - Runtime Checkers - Beartype, Pydantic, Cattrs\|Runtime Checkers]] | `beartype`, `pydantic`, `cattrs` — when to use which | Defending untrusted input |
| 07 | [[07 - Capstone - Strict-Typed ML Pipeline\|Capstone]] | Full ML pipeline: Pydantic + beartype + mypy strict | Job-ready type discipline |

## Why Modern Typing Matters for AI/ML Engineers

Three reasons this is not an academic exercise:

1. **LLM-generated code is untyped.** When GitHub Copilot or Claude Code generates 200 lines of ML pipeline, the type annotations are the difference between "compiles" and "compiles correctly". Strict typing is the cheapest insurance.
2. **Multi-team codebases break at the seams.** Your LangGraph agent calls into HuggingFace Transformers which calls into PyTorch. The boundary contracts are typed; the internals can be loose. Without strict types at the seams, refactoring is impossible.
3. **Hiring signals matter.** The 2026 ML/AI Engineer interview loop increasingly asks about Python type system design. "How would you type a stateful LangGraph agent?" is a real question with a real answer.

## The 2018 vs 2026 Typing Gap

| Concept | 2018 (legacy) | 2026 (modern) | Source PEP |
|---------|---------------|---------------|------------|
| Container types | `List[int]`, `Dict[str, int]` | `list[int]`, `dict[str, int]` | PEP 585 (3.9) |
| Union types | `Union[X, Y]`, `Optional[X]` | `X \| Y`, `X \| None` | PEP 604 (3.10) |
| Generics | `T = TypeVar("T"); class X(Generic[T])` | `class X[T]: ...` | PEP 695 (3.12) |
| Type aliases | `X = List[Tuple[int, str]]` | `type X = list[tuple[int, str]]` | PEP 695 (3.12) |
| Variance | `T = TypeVar("T", covariant=True)` | `class X[T]: ...` (inferred) | PEP 695 (3.12) |
| Self type | `Self = TypeVar("Self", bound="X")` | `Self` (built-in) | PEP 673 (3.11) |
| Narrowing | `cast(X, value)`, isinstance | `TypeGuard`, smart narrowing | PEP 647 (3.10) |
| ParamSpec | `P = ParamSpec("P")` (ceremony) | `def f[**P](*args: P.args, **kwargs: P.kwargs)` | PEP 612 + 646 |

The migration from 2018 to 2026 typing is **mostly mechanical** but **significantly cleaner**. Modern code is shorter, more readable, and easier to refactor.

## Prerequisites

- **Advanced Python (03/03)**: type hints, async, decorators. [[../03 - Python Avanzado/05 - Type Hints y Anotaciones.md|Type Hints y Anotaciones]] is the entry point.
- **Pydantic Deep Dive (03/06)**: notes 01-08 cover Pydantic v2 typing patterns this course references.
- **Python 3.12+** for the PEP 695 syntax. Earlier notes (3-6) work on 3.10+.

## How to Read This Course

1. **Notes 01-03** are the modern core. PEP 695 + narrowing + protocols.
2. **Notes 04-06** are the advanced surface. `Self`/`ParamSpec` + static analysis + runtime checkers.
3. **Note 07** is the capstone — a typed ML pipeline that uses Pydantic, beartype, mypy strict together.

## 📦 Compression Code

```python
# 📦 Welcome - The 2018 vs 2026 gap in one file

from typing import TypeVar, Generic, Iterable  # 2018 imports
from collections.abc import Iterable as IterableABC  # 2018 fallback

T = TypeVar("T")

# === 2018: Generic container (verbose) ===
class Stack2018(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()
    def all(self) -> Iterable[T]:
        return list(self._items)

# === 2026: PEP 695 (clean) ===
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()
    def all(self) -> list[T]:
        return list(self._items)

# === 2018: Union / Optional ===
from typing import Union, Optional
def get_name(user: Optional[dict]) -> Union[str, None]:
    return user["name"] if user else None

# === 2026: PEP 604 ===
def get_name(user: dict | None) -> str | None:
    return user["name"] if user else None

# === 2026: type alias ===
type Vector = list[float]
type Matrix = list[Vector]

def normalize(m: Matrix) -> Matrix:
    return [[v / sum(row) for v in row] for row in m]
```

Three PEPs (PEP 585, PEP 604, PEP 695) collapse ~10 lines of 2018 boilerplate into 5 lines of 2026 code. That is the productivity gain typing buys you in 2026.

## References

- [[../03 - Python Avanzado/05 - Type Hints y Anotaciones.md|Type Hints y Anotaciones]] — the 2018-era Spanish counterpart.
- [[../06 - Pydantic Deep Dive/01 - BaseModel, Field and Type System.md|Pydantic BaseModel]] — the runtime validation layer this course composes with.
- PEP 695: https://peps.python.org/pep-0695/
- PEP 604: https://peps.python.org/pep-0604/
- PEP 647: https://peps.python.org/pep-0647/
- PEP 673: https://peps.python.org/pep-0673/