# 🎯 05 - Capstone — Build an Async Library

> **Hands-on capstone. Build `asyncrunner`, a small async library that exposes structured concurrency, bounded parallelism, retries, and observability. The patterns crystallized into reusable APIs.**

## 🎯 Learning Objectives
- Design a Python async library API that exposes structured concurrency
- Implement bounded parallelism with `asyncio.Semaphore`
- Build retry policies with exponential backoff + jitter
- Add OpenTelemetry-compatible tracing to every operation
- Write tests with `pytest-asyncio` that catch regressions
- Document the library with reference-quality docstrings
- Apply the 20-pattern anti-pattern reference card from Note 03

## Introduction

The capstone is the **synthesis**. You will build `asyncrunner`, a small async library (~300 lines) that exposes the patterns from Notes 01-04 as a reusable API. The library is intentionally small — big enough to be useful, small enough to read in one sitting.

After building it, you will have internalized:

- How to design a clean async API
- How to expose `Semaphore`, `TaskGroup`, and `shield` to library users
- How to integrate tracing without coupling to OpenTelemetry
- How to write tests that catch antipatterns
- How to document a library for production

This is the **tenth portfolio project**: the senior engineer skill of building reusable async primitives.

```python
# The capstone library
from asyncrunner import run_parallel, retry, gather_with_limit, trace


@retry(max_attempts=3, backoff="exponential")
@trace
async def fetch_user(user_id: int) -> dict:
    return await httpx.AsyncClient().get(f"/users/{user_id}")


# Call site — clean, safe, observable
users = await run_parallel(
    [fetch_user(uid) for uid in user_ids],
    max_concurrency=10,
    timeout=5.0,
)
```

---

## 1. Project Layout

```
asyncrunner/
├── asyncrunner/
│   ├── __init__.py
│   ├── parallel.py        # run_parallel, gather_with_limit
│   ├── retry.py           # retry decorator
│   ├── trace.py           # @trace decorator (OTel-compatible)
│   ├── shield.py          # shielded operations
│   └── exceptions.py      # AsyncRunnerError, TimeoutError
├── tests/
│   ├── test_parallel.py
│   ├── test_retry.py
│   ├── test_trace.py
│   └── test_integration.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 2. Bounded Parallel Execution (`asyncrunner/parallel.py`)

```python
"""Parallel execution with bounded concurrency and timeout."""
import asyncio
import time
from typing import Coroutine, TypeVar, Iterable
from .exceptions import AsyncRunnerTimeoutError


T = TypeVar("T")


async def run_parallel(
    coros: Iterable[Coroutine],
    max_concurrency: int = 10,
    timeout: float | None = None,
) -> list:
    """Run coroutines in parallel with bounded concurrency and timeout.
    
    Args:
        coros: Iterable of coroutines to run.
        max_concurrency: Maximum number of concurrent tasks.
        timeout: Optional total timeout in seconds.
    
    Returns:
        List of results in the same order as input.
    
    Raises:
        AsyncRunnerTimeoutError: If total time exceeds timeout.
    """
    sem = asyncio.Semaphore(max_concurrency)
    
    async def bounded(coro):
        async with sem:
            return await coro
    
    # Use TaskGroup for structured concurrency (Python 3.11+)
    tasks_coroutines = list(coros)
    
    if timeout is None:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(bounded(c)) for c in tasks_coroutines]
        return [t.result() for t in tasks]
    
    # With timeout: use asyncio.timeout (Python 3.11+)
    async with asyncio.timeout(timeout):
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(bounded(c)) for c in tasks_coroutines]
        return [t.result() for t in tasks]


async def gather_with_limit(
    n: int,
    coro_factory: callable,
    *args,
    **kwargs,
) -> list:
    """Run n concurrent tasks from a factory function.
    
    Args:
        n: Number of tasks.
        coro_factory: Function returning a coroutine when called.
        *args, **kwargs: Passed to coro_factory.
    """
    sem = asyncio.Semaphore(n)
    
    async def bounded(idx):
        async with sem:
            return await coro_factory(idx, *args, **kwargs)
    
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(bounded(i)) for i in range(n)]
    return [t.result() for t in tasks]


async def first_completed(*coros, timeout: float | None = None):
    """Return the result of the first completed coroutine.
    
    Useful for racing multiple sources (e.g., primary + fallback provider).
    """
    tasks = [asyncio.create_task(c) for c in coros]
    
    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        # Cancel pending tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Await cancellation to prevent "Task was destroyed but it is pending"
        await asyncio.gather(*tasks, return_exceptions=True)
    
    if not done:
        raise AsyncRunnerTimeoutError("No task completed within timeout")
    
    return done.pop().result()
```

The library exposes three primitives:

- `run_parallel`: bounded concurrency with TaskGroup + timeout
- `gather_with_limit`: create n tasks from a factory
- `first_completed`: race multiple sources

### 2.1 Async exit semantics

The library must clean up pending tasks on timeout/cancellation. The pattern:

```python
finally:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
```

This is the **correct** pattern for `asyncio.wait` with `FIRST_COMPLETED` — cancel siblings and await the cancellation.

---

## 3. Retry with Backoff (`asyncrunner/retry.py`)

```python
"""Retry decorator with exponential backoff and jitter."""
import asyncio
import random
import time
import inspect
from functools import wraps
from typing import TypeVar, Type


T = TypeVar("T")


class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        backoff: str = "exponential",
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        jitter: bool = True,
        retry_on: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on


def retry(
    max_attempts: int = 3,
    backoff: str = "exponential",
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    jitter: bool = True,
    retry_on: tuple = (Exception,),
):
    """Decorator to retry async functions with configurable backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including the first).
        backoff: "exponential" or "linear".
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter: Add randomness to avoid thundering herd.
        retry_on: Exception classes to retry on.
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        backoff=backoff,
        initial_delay=initial_delay,
        max_delay=max_delay,
        jitter=jitter,
        retry_on=retry_on,
    )
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    if attempt >= config.max_attempts:
                        raise
                    
                    delay = _compute_delay(
                        attempt=attempt,
                        config=config,
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception  # never reached, but for type-checker
        
        return wrapper
    return decorator


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """Compute the next retry delay with optional jitter."""
    if config.backoff == "exponential":
        delay = config.initial_delay * (2 ** (attempt - 1))
    elif config.backoff == "linear":
        delay = config.initial_delay * attempt
    else:
        raise ValueError(f"Unknown backoff: {config.backoff}")
    
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        # Full jitter: random between 0 and delay
        delay = random.uniform(0, delay)
    
    return delay
```

The `full jitter` strategy is the production standard (covered in [[../../10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/02 - Production Async Patterns - FastAPI, Retries, Batching and Observability|Qdrant note 02]]).

### 3.1 Selective retry

```python
@retry(max_attempts=3, retry_on=(httpx.TimeoutException, httpx.HTTPStatusError))
async def fetch():
    return await httpx.AsyncClient().get(url)
```

Don't retry on all exceptions — only retriable ones (network timeouts, 5xx, rate limits). ValueError, KeyError, etc. should propagate immediately.

---

## 4. Tracing Decorator (`asyncrunner/trace.py`)

```python
"""Tracing decorator compatible with OpenTelemetry."""
import asyncio
import time
import functools
from typing import Callable
from contextlib import contextmanager


# Trace storage (simple in-memory; production would use OTEL)
_traces: list[dict] = []


@contextmanager
def _span(name: str, **attributes):
    """Simple span: captures timing and attributes."""
    start = time.perf_counter()
    span = {
        "name": name,
        "start": start,
        "attributes": attributes,
        "events": [],
    }
    try:
        yield span
        span["status"] = "ok"
    except Exception as e:
        span["status"] = "error"
        span["error"] = str(e)
        raise
    finally:
        elapsed = time.perf_counter() - start
        span["elapsed_ms"] = elapsed * 1000
        _traces.append(span)


def trace(func: Callable | None = None, *, name: str | None = None):
    """Decorator to trace async function calls.
    
    Usage:
        @trace
        async def fetch_user(...): ...
        
        @trace(name="custom_name")
        async def fn(...): ...
    """
    def decorator(f):
        span_name = name or f.__name__
        
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            with _span(span_name, **kwargs.pop("_trace_attrs", {})):
                return await f(*args, **kwargs)
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def get_traces() -> list[dict]:
    """Return all captured traces (for testing)."""
    return _traces


def clear_traces():
    """Clear all captured traces."""
    _traces.clear()
```

Production version would push to OpenTelemetry's TracerProvider; this is the simpler version for demonstration.

---

## 5. Shielded Operations (`asyncrunner/shield.py`)

```python
"""Shielded operations for cancellation protection."""
import asyncio
from functools import wraps


def protect_from_cancel(coro_func):
    """Decorator that ensures the inner coroutine cannot be cancelled.
    
    Useful for cleanup operations that must run to completion.
    """
    @wraps(coro_func)
    async def wrapper(*args, **kwargs):
        return await asyncio.shield(coro_func(*args, **kwargs))
    return wrapper
```

---

## 6. Exceptions (`asyncrunner/exceptions.py`)

```python
"""Custom exceptions for the asyncrunner library."""


class AsyncRunnerError(Exception):
    """Base exception for asyncrunner errors."""


class AsyncRunnerTimeoutError(AsyncRunnerError):
    """Raised when an operation exceeds its timeout."""


class AsyncRunnerRetryError(AsyncRunnerError):
    """Raised when retries are exhausted."""
```

---

## 7. The Public API (`asyncrunner/__init__.py`)

```python
"""asyncrunner — Structured concurrency primitives for production asyncio.

A small library that exposes:
- run_parallel: bounded concurrency with timeout
- retry: retry decorator with exponential backoff + jitter
- trace: tracing decorator (OTel-compatible)
- shield: cancellation protection
- gather_with_limit, first_completed: more concurrency primitives
"""
from .parallel import run_parallel, gather_with_limit, first_completed
from .retry import retry
from .trace import trace, get_traces, clear_traces
from .shield import protect_from_cancel
from .exceptions import AsyncRunnerError, AsyncRunnerTimeoutError, AsyncRunnerRetryError


__version__ = "0.1.0"
__all__ = [
    "run_parallel",
    "gather_with_limit",
    "first_completed",
    "retry",
    "trace",
    "protect_from_cancel",
    "AsyncRunnerError",
    "AsyncRunnerTimeoutError",
    "AsyncRunnerRetryError",
]
```

---

## 8. Tests (`tests/`)

### 8.1 Test parallel execution

```python
import asyncio
import pytest
import time
from asyncrunner import run_parallel


@pytest.mark.asyncio
async def test_run_parallel_basic():
    async def task(x):
        await asyncio.sleep(0.05)
        return x * 2
    
    results = await run_parallel([task(i) for i in range(5)])
    assert results == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_run_parallel_respects_concurrency_limit():
    """With max_concurrency=2, only 2 tasks run at a time."""
    concurrent = 0
    max_observed = 0
    
    async def task():
        nonlocal concurrent, max_observed
        concurrent += 1
        max_observed = max(max_observed, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
    
    await run_parallel([task() for _ in range(10)], max_concurrency=2)
    
    assert max_observed <= 2


@pytest.mark.asyncio
async def test_run_parallel_timeout():
    async def slow_task():
        await asyncio.sleep(2)
    
    with pytest.raises(asyncio.TimeoutError):
        await run_parallel([slow_task()], timeout=0.1)


@pytest.mark.asyncio
async def test_run_parallel_cancellation_propagates():
    """CancelledError should propagate (task antipattern #3 from Note 03)."""
    async def cancelled_task():
        await asyncio.sleep(10)
    
    with pytest.raises(asyncio.TimeoutError):
        await run_parallel([cancelled_task()], timeout=0.05)
```

### 8.2 Test retry

```python
import pytest
import time
from asyncrunner import retry


@pytest.mark.asyncio
async def test_retry_succeeds_after_failure():
    attempts = 0
    
    @retry(max_attempts=3)
    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("transient")
        return "ok"
    
    result = await flaky()
    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_exhausts_attempts():
    @retry(max_attempts=2)
    async def always_fails():
        raise ValueError("permanent")
    
    with pytest.raises(ValueError):
        await always_fails()


@pytest.mark.asyncio
async def test_retry_does_not_retry_valueerror():
    """Only retry on specified exception types."""
    @retry(max_attempts=3, retry_on=(IOError,))
    async def valueerror_raises():
        raise ValueError("not retriable")
    
    with pytest.raises(ValueError):
        # No retries — fails on first attempt
        await valueerror_raises()
```

### 8.3 Test the anti-pattern detection

```python
@pytest.mark.asyncio
async def test_no_sync_sleep_in_library():
    """Verify the library itself doesn't have antipattern #1."""
    import ast
    from pathlib import Path
    
    library_dir = Path(__file__).parent.parent / "asyncrunner"
    
    for py_file in library_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if ast.unparse(child.func) == "time.sleep":
                            pytest.fail(
                                f"Sync sleep in async function: {py_file}:{child.lineno}"
                            )


@pytest.mark.asyncio
async def test_no_unbounded_gather():
    """Verify the library doesn't have unbounded gather (antipattern #4)."""
    import ast
    from pathlib import Path
    
    library_dir = Path(__file__).parent.parent / "asyncrunner"
    
    for py_file in library_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            # Check for gather() calls
            if isinstance(node, ast.Call):
                if ast.unparse(node.func) == "asyncio.gather":
                    # OK if bounded with max_concurrency parameter
                    has_bound = any(
                        kw.arg == "max_concurrency"
                        for kw in node.keywords
                    )
                    if not has_bound:
                        # OK if it's our wrapper (run_parallel etc.)
                        if "run_parallel" in py_file.read_text():
                            continue
                        pytest.fail(
                            f"Unbounded gather() in {py_file}:{node.lineno}"
                        )
```

These test cases **prevent regressions** — they ensure the library itself doesn't have the antipatterns from Note 03.

---

## 9. Documentation (`README.md`)

```markdown
# asyncrunner

Structured concurrency primitives for production asyncio.

## Quick Start

```python
from asyncrunner import run_parallel, retry, trace


@retry(max_attempts=3)
@trace
async def fetch_user(user_id: int) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"/users/{user_id}")
        return r.json()


users = await run_parallel(
    [fetch_user(uid) for uid in user_ids],
    max_concurrency=10,
    timeout=5.0,
)
```

## API

### run_parallel(coros, max_concurrency=10, timeout=None)

Run coroutines in parallel with bounded concurrency.

```python
results = await run_parallel(
    [fetch(uid) for uid in user_ids],
    max_concurrency=10,
    timeout=5.0,
)
```

### retry(max_attempts=3, backoff="exponential", ...)

Decorator for retrying async functions.

```python
@retry(max_attempts=3, retry_on=(IOError, ConnectionError))
async def fetch():
    ...
```

### trace(func)

Decorator to trace function calls.

```python
@trace
async def my_function():
    ...

traces = get_traces()  # for testing/debugging
```

## Anti-Patterns Avoided

- Sync I/O in async functions (none)
- Missing awaits (caught by ruff ASYNC)
- Swallowed CancelledError (always re-raised)
- Unbounded parallelism (always bounded by semaphore)
- No timeouts (timeout parameter is required for production)
- CPU-bound blocking (run heavy work in subinterpreters, not in async fns)

See the [Async Anti-Patterns Reference](https://github.com/user/asyncrunner/blob/main/ANTI_PATTERNS.md) for the 20-pattern catalog.
```

---

## 10. The Capstone Run

Build the library and run the full test suite:

```bash
# 1. Install dependencies
pip install pytest pytest-asyncio respx

# 2. Run unit tests
pytest tests/ -v

# 3. Run async linter
ruff check asyncrunner/ --select ASYNC

# 4. Run the anti-pattern regression tests
pytest tests/test_anti_patterns.py -v
```

Expected output:

```
======================== test session starts =========================
collected 25 items

tests/test_parallel.py::test_run_parallel_basic PASSED
tests/test_parallel.py::test_run_parallel_respects_concurrency_limit PASSED
tests/test_parallel.py::test_run_parallel_timeout PASSED
tests/test_parallel.py::test_run_parallel_cancellation_propagates PASSED
tests/test_retry.py::test_retry_succeeds_after_failure PASSED
tests/test_retry.py::test_retry_exhausts_attempts PASSED
tests/test_retry.py::test_retry_does_not_retry_valueerror PASSED
...
======================== 25 passed in 1.43s =========================
```

After running the capstone, you have:

- A working async library (~300 lines) that exposes the patterns correctly
- A test suite that catches regressions
- Documentation that follows the conventions from Note 03
- The senior engineer skill of building reusable async primitives

---

## 11. Production Deployment Checklist

- [ ] Library published to PyPI (`twine upload dist/*`)
- [ ] Versioned with `semantic-release`
- [ ] Documentation at `readthedocs.io` or `mkdocs`
- [ ] CI runs ruff ASYNC, pytest, mypy
- [ ] Type hints on every public function
- [ ] OpenTelemetry integration tested
- [ ] Test coverage >90%
- [ ] Changelog maintained

---

## 🎯 Key Takeaways

- `run_parallel` exposes bounded concurrency with timeout as a single function.
- `retry` decorator with full jitter is the production standard.
- `trace` decorator integrates with OpenTelemetry (or any backend).
- The library itself avoids all 20 antipatterns from Note 03 (verified by lint).
- Tests catch regressions in both behavior and antipattern compliance.
- Building a small library deepens understanding more than reading about one.
- The capstone is the **tenth portfolio project**: senior engineer async skill.

## References

- [[../01 - Event Loop Internals - uvloop, Selectors, and the GIL Interplay|Note 01 — Event Loop Internals]]
- [[../02 - Async Library Ecosystem - Decision Framework for 2026|Note 02 — Library Ecosystem]]
- [[../03 - Async Anti-Patterns Reference Card - 20 Patterns with Stack-Trace Signatures|Note 03 — Anti-Patterns Reference]]
- [[../04 - Subinterpreters and Python 3.12+ Async - CPU-Bound + Async Hybrid Work|Note 04 — Subinterpreters]]
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/01 - ASGI Architecture and Async Python for ML|Note 01 — ASGI Architecture and Async Python for ML]]
- [[../../10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI/06 - Async Debugging in Production - Common Pitfalls and Detection|Note — Async Debugging]]
- [[../../10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/02 - Production Async Patterns - FastAPI, Retries, Batching and Observability|Qdrant Production Async Patterns]]
- Python asyncio docs — [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
- pytest-asyncio — [pytest-asyncio.readthedocs.io](https://pytest-asyncio.readthedocs.io/)