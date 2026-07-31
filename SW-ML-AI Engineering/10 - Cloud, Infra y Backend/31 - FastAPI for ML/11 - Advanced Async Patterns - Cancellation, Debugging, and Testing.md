# ⚡ 11 - Advanced Async Patterns — Cancellation, Debugging, and Testing

> **The advanced async toolkit that separates senior Python engineers. Shielding, task groups, cancellation propagation, async generators, and the testing patterns you need when async code breaks in production.**

## 🎯 Learning Objectives
- Use `asyncio.TaskGroup` (Python 3.11+) for structured concurrency
- Implement cancellation, timeouts, and `asyncio.shield` correctly
- Detect and debug async hangs, deadlock patterns, and event-loop stalls
- Write robust async tests with `pytest-asyncio`, `freezegun`, and `httpx.AsyncClient`
- Apply advanced patterns: async generators, async context managers, bounded concurrency
- Recognize the seven common async pitfalls and avoid them

## Introduction

The previous notes (especially [[../31 - FastAPI for ML/01 - ASGI Architecture and Async Python for ML|Note 01 — ASGI Architecture and Async Python for ML]]) cover the fundamentals: `async def`, `await`, the event loop, asyncio.gather. This note assumes you have that baseline and goes into the patterns that production code actually needs.

A senior async Python engineer knows how to:

1. **Structure concurrency correctly** — `TaskGroup` over `gather`, structured failure handling
2. **Cancel tasks properly** — `asyncio.shield`, `CancelledError` propagation, cleanup
3. **Bound concurrency** — semaphores, connection pools, backpressure
4. **Set and honor timeouts** — every external call has a deadline
5. **Write tests that don't deadlock** — `pytest-asyncio` configuration, fixtures, mocking
6. **Debug async hangs in production** — `py-spy`, `faulthandler`, the seven common patterns

This note covers all six.

---

## 1. `asyncio.TaskGroup` — Structured Concurrency

`TaskGroup` is the modern Python 3.11+ replacement for `asyncio.gather`. Unlike `gather`, it provides **structured failure**: if any task fails, all sibling tasks are cancelled.

```python
import asyncio


# ✅ Modern pattern (Python 3.11+)
async def fetch_all_prompts(prompts: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_one(p)) for p in prompts]
    return [t.result() for t in tasks]


# ❌ Legacy pattern (still works but less safe)
async def fetch_all_prompts_legacy(prompts: list[str]) -> list[str]:
    tasks = [asyncio.create_task(fetch_one(p)) for p in prompts]
    return await asyncio.gather(*tasks)  # problem: if one fails, others continue
```

The difference:

| Pattern | One task fails | Sibling tasks |
|---------|---------------|---------------|
| `gather(return_exceptions=False)` | Exception propagates | Continue running (potential resource leak) |
| `gather(return_exceptions=True)` | Returns exception object | Continue running |
| `TaskGroup()` | ExceptionGroup | **Cancelled automatically** |

`TaskGroup` is the correct default for production code: when one task fails, siblings are cancelled cleanly.

### 1.1 ExceptionGroup handling

`TaskGroup` raises `ExceptionGroup`, not single exceptions:

```python
async def fetch_with_fallback(prompts: list[str]) -> dict:
    async with asyncio.TaskGroup() as tg:
        tasks = [(p, tg.create_task(fetch_one(p))) for p in prompts]
    
    results = {}
    for prompt, task in tasks:
        try:
            results[prompt] = task.result()
        except* requests.RequestError as eg:  # except* is Python 3.11+ syntax
            for exc in eg.exceptions:
                results[prompt] = f"fallback: {exc}"
    
    return results
```

Use `except*` to extract specific exception types from the group.

---

## 2. Cancellation, Timeouts, and Shielding

### 2.1 The cancellation contract

When you `cancel()` a task (or it times out), `asyncio.CancelledError` is raised inside the task. The task can:

1. **Suppress and continue** — bad practice, hides cancellation
2. **Catch for cleanup** — release resources, then re-raise
3. **Let it propagate** — clean cancellation

```python
import asyncio
import httpx


async def fetch_with_cleanup(url: str) -> str:
    """Correct pattern: catch CancelledError for cleanup, then re-raise."""
    client = httpx.AsyncClient(timeout=10)
    try:
        response = await client.get(url)
        return response.text
    except asyncio.CancelledError:
        # Cleanup: close connection
        await client.aclose()
        raise  # Always re-raise — never swallow
```

❌ **Common antipattern: swallowing cancellation**

```python
# ❌ Suppresses cancellation; the task continues running
async def fetch_bad(url: str) -> str:
    try:
        return await httpx.AsyncClient().get(url)
    except asyncio.CancelledError:
        return ""  # The task CANCELS itself from the outside but continues running internally
```

This causes **silent resource leaks** and **zombie tasks**.

### 2.2 Timeouts

Every external call must have a timeout:

```python
async def with_timeout(coro, timeout: float = 5.0):
    """Wrap any coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise MyTimeoutError(f"Operation timed out after {timeout}s")


# Or use the Python 3.11+ timeout context manager
async def with_timeout_ctx(coro, timeout: float):
    async with asyncio.timeout(timeout):
        return await coro
```

A common pattern in LLM systems:

```python
async def chat_with_timeout(messages: list, model: str) -> str:
    """Chat completion with model-specific timeout."""
    timeout_map = {
        "gpt-4o": 30.0,
        "gpt-4o-mini": 15.0,
        "claude-3-5-sonnet": 30.0,
    }
    async with asyncio.timeout(timeout_map.get(model, 30.0)):
        return await openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
```

### 2.3 `asyncio.shield` — protecting critical work

Sometimes you want to set a timeout on the outer operation but NOT cancel an inner critical cleanup:

```python
async def fetch_with_protected_cleanup(url: str):
    try:
        async with asyncio.timeout(5.0):
            data = await httpx.AsyncClient().get(url)
        return data
    except asyncio.TimeoutError:
        # Outer timeout fires; inner cleanup needs to complete
        # Use shield to ensure cleanup isn't cancelled
        await asyncio.shield(perform_cleanup(url))
        raise
```

`asyncio.shield` prevents the inner task from being cancelled when the outer timeout fires.

---

## 3. Bounded Concurrency — Semaphores and Pools

For high-throughput systems, unbounded concurrency crashes the destination server. Use semaphores:

```python
import asyncio


class BoundedExecutor:
    """Execute tasks with bounded concurrency."""
    
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def submit(self, coro):
        async with self._semaphore:
            return await coro
    
    async def submit_all(self, coros):
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.submit(c)) for c in coros]
        return [t.result() for t in tasks]


# Usage
executor = BoundedExecutor(max_concurrent=20)
results = await executor.submit_all([fetch_one(p) for p in prompts])
```

This pattern is used in:
- **Qdrant** (covered in [[../../../10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/02 - Production Async Patterns - FastAPI, Retries, Batching and Observability|Qdrant Production Async]])
- **OpenAI batch inference** (covered in [[../../../06 - Large Language Models/22 - Instructor and Structured Generation/01 - Instructor - Pydantic-Native Structured Outputs|Instructor note 01]])
- **Background workers** (covered in [[../40 - Background Jobs and Workers for FastAPI/02 - ARQ Modern Async-Native|ARQ note 02]])

### 3.1 Connection pooling for HTTP

For high-throughput HTTP clients, use a connection pool:

```python
import httpx


# ✅ Singleton AsyncClient (recommended)
client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,           # pool size
        max_keepalive_connections=20,  # keep-alive pool
        keepalive_expiry=30.0,         # seconds
    ),
    timeout=httpx.Timeout(10.0),
)


async def fetch_many(urls: list[str]) -> list[str]:
    tasks = [client.get(url) for url in urls]
    responses = await asyncio.gather(*tasks)
    return [r.text for r in responses]
```

Without a pool, every request opens a new connection — slow, leaks file descriptors. With a pool, connections are reused.

---

## 4. Async Generators and Streaming

For streaming responses (e.g., SSE from LLM):

```python
import asyncio
from typing import AsyncIterator


async def stream_chat_completion(
    prompt: str,
) -> AsyncIterator[str]:
    """Async generator for streaming LLM responses."""
    async with openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Consumer (FastAPI endpoint)
from fastapi.responses import StreamingResponse


@app.post("/chat/stream")
async def chat_stream(prompt: str):
    async def generator():
        async for token in stream_chat_completion(prompt):
            yield f"data: {token}\n\n"
    
    return StreamingResponse(generator(), media_type="text/event-stream")
```

Async generators are the bridge between streaming APIs and FastAPI's `StreamingResponse`. Use them for SSE, file uploads, and large LLM responses.

---

## 5. Async Testing with `pytest-asyncio`

Async code is harder to test than sync code. The common pitfalls:

### 5.1 Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # auto-mark all tests as async
```

Or use the decorator:

```python
import pytest


@pytest.mark.asyncio
async def test_chat_completion():
    response = await chat("Hello")
    assert response == "world"
```

### 5.2 Fixtures for async setup

```python
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def async_client():
    """Async client for testing FastAPI endpoints."""
    from app.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_chat_endpoint(async_client):
    response = await async_client.post("/chat", json={"prompt": "Hello"})
    assert response.status_code == 200
```

### 5.3 Mocking async functions

```python
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_with_mock_openai():
    # Mock the OpenAI client
    mock_response = AsyncMock()
    mock_response.choices[0].message.content = "Mocked response"
    
    with patch("app.llm.openai_client.chat.completions.create", return_value=mock_response):
        result = await chat("Hello")
    
    assert result == "Mocked response"
```

### 5.4 Testing timeouts and cancellations

```python
@pytest.mark.asyncio
async def test_timeout_handling():
    """Verify that a slow upstream call is timed out, not hung."""
    async def slow_call():
        await asyncio.sleep(10)  # never completes in test
    
    with pytest.raises(MyTimeoutError):
        await with_timeout(slow_call(), timeout=0.1)
```

### 5.5 Testing `TaskGroup` cancellation

```python
@pytest.mark.asyncio
async def test_taskgroup_cancels_siblings():
    """When one task fails, siblings should be cancelled."""
    
    async def failing():
        await asyncio.sleep(0.1)
        raise ValueError("boom")
    
    async def slow():
        await asyncio.sleep(10)
    
    with pytest.raises(ExceptionGroup) as exc_info:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(failing())
            tg.create_task(slow())
    
    assert any(isinstance(e, ValueError) for e in exc_info.value.exceptions)
```

This test verifies structured concurrency actually cancels siblings.

---

## 6. Debugging Async Hangs in Production

The classic async failure mode: a task hangs forever. The event loop blocks. No logs appear.

### 6.1 The five common async bug patterns

| Pattern | Symptom | Detection |
|---------|---------|-----------|
| **Missing await** | Function returns coroutine, not result | Type checking, `inspect.iscoroutine(result)` |
| **Sync in async function** | `time.sleep(5)` blocks event loop | py-spy dump during hang |
| **Deadlock on semaphore** | All tasks wait on semaphore | py-spy shows sem.acquire in call stack |
| **CancelledError swallowed** | Task continues after cancel | py-spy shows task still running |
| **Event loop blocked** | Web server doesn't respond | `asyncio.get_event_loop()._ready` is empty |

### 6.2 Detection with `py-spy`

`py-spy` attaches to a running Python process and dumps the call stacks:

```bash
# In production (with appropriate permissions)
py-spy dump --pid 12345 --non-blocking | head -100
```

If you see all coroutines blocked on `socket.recv` or `semaphore.acquire`, that's the bug.

### 6.3 Detection with `faulthandler`

```python
import faulthandler
faulthandler.dump_traceback_later(timeout=30, repeat=True)
```

This prints all Python stack traces after 30 seconds — useful for catching hangs in tests.

### 6.4 Detection with custom watchdog

```python
import asyncio
import signal
import sys


def install_watchdog(timeout: float = 30.0):
    """Print stack traces if the event loop is blocked for too long."""
    
    def watchdog():
        import traceback
        print("\n=== EVENT LOOP BLOCKED ===")
        for tid, frame in sys._current_frames().items():
            traceback.print_stack(frame)
        print("=== END ===\n")
    
    def signal_handler(signum, frame):
        watchdog()
    
    signal.signal(signal.SIGUSR1, signal_handler)
```

Send `SIGUSR1` to dump the call stacks on demand.

### 6.5 The seven async pitfalls

| # | Pitfall | Example | Fix |
|---|---------|---------|-----|
| 1 | Sync IO in async function | `time.sleep(5)` | Use `await asyncio.sleep(5)` |
| 2 | Missing await | `result = func()` returns coroutine | `result = await func()` |
| 3 | Swallowed CancelledError | `except CancelledError: pass` | Always re-raise after cleanup |
| 4 | Unbounded concurrency | `gather(*[task() for _ in range(1000)])` | Use semaphore |
| 5 | No timeout | `await external_call()` | Use `asyncio.timeout()` |
| 6 | Sharing event loops across threads | `asyncio.run()` in `asyncio.run()` | Run separate loops in separate threads |
| 7 | Async generator without `await` | `gen = stream(); next(gen)` | Use `async for`, not `next()` |

---

## 7. Antipatterns

### 7.1 Antipattern 1: `time.sleep` in async function

```python
# ❌ Blocks the entire event loop
async def bad_rate_limit(self):
    time.sleep(1)  # the event loop is blocked for 1 second
    return True

# ✅ Cooperative sleep
async def good_rate_limit(self):
    await asyncio.sleep(1)
    return True
```

### 7.2 Antipattern 2: `asyncio.run()` inside `asyncio.run()`

```python
# ❌ RuntimeError: asyncio.run() cannot be called from a running event loop
async def outer():
    result = asyncio.run(inner())  # CRASH

# ✅ Either await or run separately
async def outer():
    result = await inner()  # correct
```

### 7.3 Antipattern 3: `asyncio.gather(*[coro] * 10000)`

```python
# ❌ Unbounded; crashes the upstream
results = await asyncio.gather(*[fetch(url) for _ in range(10000)])

# ✅ Bounded with semaphore
sem = asyncio.Semaphore(50)
async def bounded_fetch(url):
    async with sem:
        return await fetch(url)
results = await asyncio.gather(*[bounded_fetch(url) for url in urls])
```

### 7.4 Antipattern 4: Refactoring sync to async without checking call sites

```python
# ❌ The function is now async; callers still treat it as sync
async def get_user(user_id: int) -> User:
    ...

# Caller doesn't await:
user = get_user(123)  # returns a coroutine, not a User
```

### 7.5 Antipattern 5: `asyncio.create_task` without keeping a reference

```python
# ❌ The task may be garbage-collected before completing
def fire_and_forget():
    asyncio.create_task(do_work())  # reference is dropped!

# ✅ Keep a reference
background_tasks = set()

def fire_and_forget():
    task = asyncio.create_task(do_work())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

---

## 🎯 Key Takeaways

- `asyncio.TaskGroup` (3.11+) is the modern pattern; gives structured failure and cancellation.
- Always handle `CancelledError`: catch for cleanup, then re-raise. Never swallow.
- Every external call needs a timeout via `asyncio.timeout()` or `asyncio.wait_for()`.
- Use `asyncio.shield` to protect critical cleanup from outer cancellation.
- Bounded concurrency via `asyncio.Semaphore`; unbounded `gather` crashes upstreams.
- Async generators are the bridge from streaming APIs to FastAPI's `StreamingResponse`.
- Test with `pytest-asyncio` (auto mode), `AsyncMock`, `AsyncClient` with `ASGITransport`.
- Debug hangs with `py-spy`, `faulthandler`, custom watchdogs. Watch for the seven pitfalls.
- Avoid sync-IO-in-async, nested `asyncio.run`, unbounded concurrency, missing awaits, swallowed `CancelledError`.

## References

- PEP 654 — Exception Groups — [peps.python.org/pep-0654](https://peps.python.org/pep-0654/)
- PEP 678 — ExceptionGroup enhancements — [peps.python.org/pep-0678](https://peps.python.org/pep-0678/)
- PEP 3148 — `asyncio` futures — [peps.python.org/pep-3148](https://peps.python.org/pep-3148/)
- Python asyncio docs — [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
- pytest-asyncio docs — [pytest-asyncio.readthedocs.io](https://pytest-asyncio.readthedocs.io/)
- py-spy — [github.com/benfred/py-spy](https://github.com/benfred/py-spy)
- [[../31 - FastAPI for ML/01 - ASGI Architecture and Async Python for ML|Note 01 — ASGI Architecture and Async Python for ML]]
- [[../31 - FastAPI for ML/03 - Streaming, Background Tasks, and Real-Time Endpoints|Note 03 — Streaming]]
- [[../31 - FastAPI for ML/04 - Dependency Injection, Middleware, and Testing|Note 04 — DI and Testing]]
- [[../38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/01 - Async Engine and Sessions|Note — SQLAlchemy Async]]
- [[../40 - Background Jobs and Workers for FastAPI/02 - ARQ Modern Async-Native|Note — ARQ async-native]]
- [[../../33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/02 - Production Async Patterns - FastAPI, Retries, Batching and Observability|Qdrant Production Async]]
- [[../../33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/03 - Capstone - Production RAG with Qdrant + LiteLLM + Phoenix|Qdrant Capstone]]
- [[05 - Production Deployment and Performance|Note — Production Deployment]]