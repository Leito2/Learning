# 🐛 06 - Async Debugging in Production — Common Pitfalls and Detection

> **The async debugging playbook. When ARQ/Celery/SaaS jobs hang in production, the patterns to recognize, the tools to use, and the fixes to apply.**

## 🎯 Learning Objectives
- Detect the seven common async failure modes in production background workers
- Use `py-spy`, `faulthandler`, and custom watchdogs to diagnose live hangs
- Read async stack traces correctly (the difference between sync and async frames)
- Recognize "zombie tasks" — coroutines that survive cancellation
- Wire up async health endpoints that detect event loop stalls
- Build a CI test suite that catches async antipatterns before deploy
- Apply the seven incident-response patterns from [[../../39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering|Production Incident Response Note 04]] to async systems

## Introduction

The hardest kind of async bug to debug is the one that **only happens in production**. The test suite passes. The Docker image boots. The health check returns 200. But after 3 hours at 2 AM, the worker has stopped processing jobs and no logs show what happened.

This note gives you the playbook for diagnosing and fixing these failures. It draws on:

- [[../../31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]] (foundations)
- [[../02 - ARQ Modern Async-Native|ARQ note]] (production framework)
- [[../../39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering|Incident Response Note 04]] (resilience patterns)

The goal: when an async bug shows up in production, you diagnose it in **<15 minutes** instead of <2 hours.

---

## 1. The Seven Failure Modes

### 1.1 Zombie tasks — the silent killer

A task was cancelled but kept running. Resources leak until the worker process dies.

```python
# ❌ Swallows CancelledError; the task continues until completion
async def bad_cleanup_handler():
    try:
        await expensive_operation()
    except asyncio.CancelledError:
        return  # SUPPRESSED; the task continues running internally
```

**Detection:** `psutil` shows the process holding more memory than expected; `py-spy dump` shows old coroutines still on the stack.

**Fix:**
```python
# ✅ Cleanup then re-raise
async def good_cleanup_handler():
    try:
        await expensive_operation()
    except asyncio.CancelledError:
        await cleanup_resources()
        raise  # always re-raise
```

### 1.2 Event loop starvation

A blocking sync call prevents any other coroutine from running.

```python
# ❌ This blocks the entire event loop for 5 seconds
async def bad_blocking_call():
    time.sleep(5)  # synchronous sleep
    return result
```

**Detection:** `py-spy dump` shows one frame with a `_io.TextIOWrapper.read` or `socket.recv` and zero other coroutines. Web requests and worker jobs all stall.

**Fix:**
```python
# ✅ Cooperative sleep
async def good_blocking_call():
    await asyncio.sleep(5)
    return result
```

### 1.3 Deadlock on semaphore

All 10 worker slots are blocked on a semaphore that's never released.

```python
# ❌ If fetch_one() raises after acquiring the semaphore, we leak
sem = asyncio.Semaphore(10)

async def leak_semaphore():
    async with sem:
        await fetch_one()  # raises → sem is released by async with
        # But what if the cancel happens DURING the await?
```

**Detection:** py-spy shows all workers blocked on `semaphore.acquire`. `asyncio.all_tasks()` shows many tasks waiting.

**Fix:** Use `TaskGroup` for structured concurrency, ensure semaphores are released on all paths.

### 1.4 Async generator with sync consumer

```python
# ❌ Treating an async generator as sync
async def stream_results():
    for chunk in llm_stream:
        yield chunk

stream = stream_results()
first = next(stream)  # TypeError: 'async_generator' object is not iterable
```

**Detection:** `TypeError` in production logs.

**Fix:** Use `async for`:
```python
async for chunk in stream_results():
    print(chunk)
```

### 1.5 Task reference lost

```python
# ❌ The task may be garbage-collected before completion
def fire_and_forget():
    asyncio.create_task(do_work())  # reference is dropped
```

**Detection:** Task doesn't complete; `asyncio.all_tasks()` empty.

**Fix:** Keep a reference:
```python
background_tasks = set()

def fire_and_forget():
    task = asyncio.create_task(do_work())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

### 1.6 N+1 async awaits

```python
# ❌ Sequential awaiting — slow
async def fetch_all_users(user_ids: list[int]) -> list[User]:
    users = []
    for user_id in user_ids:
        users.append(await fetch_user(user_id))  # 100 users × 50ms = 5s
    return users

# ✅ Parallel with bounded concurrency
async def fetch_all_users(user_ids: list[int]) -> list[User]:
    sem = asyncio.Semaphore(20)
    async def bounded_fetch(uid):
        async with sem:
            return await fetch_user(uid)
    return await asyncio.gather(*[bounded_fetch(uid) for uid in user_ids])
```

### 1.7 Unbounded fan-out

```python
# ❌ One user request spawns 1000 coroutines
async def handle_request(request):
    chunks = [fetch_chunk(i) for i in range(1000)]
    await asyncio.gather(*chunks)  # crashes the upstream
```

---

## 2. The Async Health Endpoint

Add a `/health/async` endpoint that detects event loop health:

```python
import asyncio
import time
from fastapi import FastAPI, HTTPException


app = FastAPI()


@app.get("/health/async")
async def health_async():
    """Detect event loop health via ping-pong latency."""
    start = time.perf_counter()
    await asyncio.sleep(0.001)  # 1ms ping
    elapsed = time.perf_counter() - start
    
    # If the event loop took >100ms to ping, it's blocked
    if elapsed > 0.1:
        raise HTTPException(status_code=503, detail=f"Event loop blocked: {elapsed*1000:.0f}ms")
    
    return {
        "status": "ok",
        "event_loop_latency_ms": elapsed * 1000,
        "pending_tasks": len(asyncio.all_tasks()),
    }
```

The Kubernetes liveness probe will catch event-loop stalls.

### 2.1 The async-deep-health pattern

```python
@app.get("/health/deep")
async def health_deep():
    """Verify async subsystems are responsive."""
    checks = {}
    
    # Check Redis
    start = time.perf_counter()
    await redis_client.ping()
    checks["redis_ms"] = (time.perf_counter() - start) * 1000
    
    # Check DB
    start = time.perf_counter()
    await db.execute("SELECT 1")
    checks["db_ms"] = (time.perf_counter() - start) * 1000
    
    # Check LLM provider
    start = time.perf_counter()
    await openai_client.models.list()
    checks["openai_ms"] = (time.perf_counter() - start) * 1000
    
    # Check ARQ worker
    start = time.perf_counter()
    await arq_pool.bulk_size()
    checks["arq_ms"] = (time.perf_counter() - start) * 1000
    
    # If any check is slow, return 503
    max_latency = max(checks.values())
    if max_latency > 5000:
        raise HTTPException(status_code=503, detail=f"Deep health failed: {checks}")
    
    return checks
```

This catches the **zombie task** pattern: if ARQ is hung, this endpoint fails.

---

## 3. The `py-spy` Playbook

`py-spy` is the first tool to reach for when async code hangs.

### 3.1 Dump the call stacks

```bash
# Non-blocking dump (recommended for production)
py-spy dump --pid 12345 --non-blocking | head -200

# Continuous dump (every 5 seconds)
py-spy dump --pid 12345 --non-blocking --duration 30
```

If the worker has hung, you'll see something like:

```
Thread 0x7fff5fbff8c0 (main): asyncio.run
  events.py:81 in _run
    run_forever
      _run_once
        _selector.select (BLOCKED here for 12.3s)
Thread 0x7fff5fbff9c0 (worker-1): Task
  ...
    semaphore.acquire (BLOCKED here for 12.3s)
      workers/arq.py:42 in acquire
```

The selector is blocked → no events are being processed. The semaphore is held → workers are stuck.

### 3.2 The async stack frame signature

Async frames differ from sync frames:

```
Sync frame:
  File "app.py", line 42, in get_user
    result = db.execute(...)

Async frame:
  File "app.py", line 42, in get_user
    coro = db.execute(...)  # returns a coroutine
  File "app.py", line 43, in get_user
    result = await coro  # actual work happens here
```

In async code, you'll see the `coro.send()` line — that's where the current frame is suspended.

### 3.3 High-FPS sampling

```bash
# 100 Hz sampling for 30 seconds
py-spy record --pid 12345 --rate 100 --duration 30 -o profile.svg
```

The flame graph shows which function is consuming the most time. Common findings:

- `socket.recv` 100% of time → network bottleneck
- `semaphore.acquire` 100% of time → pool exhausted
- `time.sleep` 100% of time → sync sleep in async function

---

## 4. The `faulthandler` Playbook

`faulthandler` is a Python builtin that prints all stack traces on signal:

```python
import faulthandler
import signal

# Enable at startup
faulthandler.enable()

# Print tracebacks on SIGUSR1
faulthandler.register(signal.SIGUSR1)

# In production: send SIGUSR1 to dump all stacks
# kill -USR1 <pid>
```

Or automatic periodic dumps:

```python
# Dump every 5 minutes if the loop is still running
import os
faulthandler.dump_traceback_later(timeout=300, repeat=True)
```

This is the **hammer** approach — when you've tried everything else, dump the stacks and look.

---

## 5. The Custom Watchdog Pattern

For ARQ/Celery workers, install a custom watchdog that detects event loop stalls:

```python
import asyncio
import sys
import time
import traceback


async def install_async_watchdog(loop: asyncio.AbstractEventLoop, threshold: float = 30.0):
    """Detect event loop stalls and dump stack traces."""
    
    last_heartbeat = time.time()
    
    async def heartbeat():
        nonlocal last_heartbeat
        while True:
            await asyncio.sleep(5)
            last_heartbeat = time.time()
    
    async def checker():
        while True:
            await asyncio.sleep(10)
            now = time.time()
            stall = now - last_heartbeat - 5  # subtract the sleep
            
            if stall > threshold:
                print(f"\n=== EVENT LOOP STALLED for {stall:.0f}s ===")
                # Dump all task stacks
                for task in asyncio.all_tasks():
                    if not task.done():
                        print(f"Task: {task.get_name()}")
                        if task._coro:
                            frames = task._coro.cr_frame
                            if frames:
                                traceback.print_stack(frames)
                print("=== END ===\n")
    
    # Run both
    asyncio.create_task(heartbeat(), name="heartbeat")
    asyncio.create_task(checker(), name="watchdog")
```

The watchdog detects when the heartbeat task hasn't run for >30 seconds — the event loop is blocked.

---

## 6. The Async Test Patterns for Production

### 6.1 Test that zombie tasks are cancelled

```python
import asyncio
import pytest


@pytest.mark.asyncio
async def test_zombie_task_is_cancelled():
    """When a task is cancelled, it should not leave coroutines running."""
    
    async def long_task():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # ❌ Bug: doesn't re-raise
            return

    task = asyncio.create_task(long_task())
    await asyncio.sleep(0.1)
    task.cancel()
    
    with pytest.raises(asyncio.CancelledError):
        await task  # ❌ This would NOT raise if the test passes
    
    # Verify the task is actually done
    assert task.done()
```

### 6.2 Test that `time.sleep` is not in async code

```python
import ast
import pytest


def test_no_sync_sleep_in_async():
    """Lint-level: ensure no sync I/O in async functions."""
    
    for root, dirs, files in os.walk("app/"):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                tree = ast.parse(fh.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    # Check for sync I/O calls inside async functions
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            name = ast.unparse(child.func)
                            if name in ("time.sleep", "time.time", "requests.get", "requests.post"):
                                pytest.fail(f"Sync call in async function: {path}:{child.lineno} {name}")
```

This catches the most common async bug at CI time.

### 6.3 Test event loop health

```python
@pytest.mark.asyncio
async def test_event_loop_responsive():
    """Ensure the event loop processes coroutines within deadline."""
    start = time.perf_counter()
    await asyncio.sleep(0.001)
    elapsed = time.perf_counter() - start
    
    # If the sleep took >100ms, the event loop is jammed
    assert elapsed < 0.1, f"Event loop blocked: {elapsed*1000:.0f}ms"
```

Run this in the test suite to catch regressions.

---

## 7. The CI Async Linter

Configure `flake8-async` or `ruff` to catch async antipatterns:

```toml
# pyproject.toml
[tool.ruff]
select = ["ASYNC"]  # flake8-async rules

[tool.ruff.lint.flake8-async]
# Ban sync I/O in async functions
banned-modules = ["requests", "time.sleep"]

# Require awaiting on async calls
force-await-async-call = true
```

Ruff's `ASYNC` rules catch:
- `time.sleep` in async functions
- `requests.get` in async functions
- Missing `await` on async calls
- Sync generators in async contexts

This is the **lowest-cost** async bug prevention: catch them at lint time.

---

## 8. The Async Runbook

When an async bug is reported:

```markdown
# Async Debugging Runbook

## 1. Capture the state
- [ ] `py-spy dump --pid <pid> --non-blocking > dump.txt`
- [ ] `curl http://localhost:8000/health/async`
- [ ] `curl http://localhost:8000/health/deep`
- [ ] Check ARQ/Celery worker logs for `asyncio.CancelledError`

## 2. Identify the failure mode
- [ ] All workers blocked on semaphore? → semaphore deadlock
- [ ] Event loop blocked on sync call? → sync-in-async antipattern
- [ ] Tasks not completing? → swallowed CancelledError
- [ ] Web requests slow but workers idle? → asyncio.Queue bottleneck

## 3. Mitigate
- [ ] For semaphore deadlock: increase pool size or restart workers
- [ ] For sync-in-async: check py-spy for `time.sleep` or sync I/O calls
- [ ] For cancelled: add `try/except CancelledError: await cleanup(); raise`
- [ ] For queue: increase queue limits or backpressure

## 4. Recover
- [ ] Restart workers (`pkill -HUP` or `kubectl rollout restart`)
- [ ] Verify health endpoints return 200
- [ ] Push fix
- [ ] Schedule postmortem
```

---

## 9. Antipatterns

### 9.1 Antipattern 1: Ignoring async antipatterns in code review

```python
# ❌ Code review passed because no tests failed
async def fetch():
    time.sleep(5)  # bug discovered in production at 3 AM

# ✅ Async linter catches this at CI time
```

### 9.2 Antipattern 2: No async health endpoint

```python
# ❌ Health endpoint returns 200 but the event loop is stuck
@app.get("/health")
def health():
    return {"status": "ok"}  # sync handler that doesn't probe event loop

# ✅ Async health endpoint that probes event loop
@app.get("/health/async")
async def health_async():
    start = time.perf_counter()
    await asyncio.sleep(0.001)
    if time.perf_counter() - start > 0.1:
        raise HTTPException(503)
    return {"status": "ok"}
```

### 9.3 Antipattern 3: Using `asyncio.gather` for thousands of concurrent tasks

```python
# ❌ 10K concurrent tasks exhaust the file descriptor pool
results = await asyncio.gather(*[fetch(url) for url in urls_10k])

# ✅ Bounded concurrency
sem = asyncio.Semaphore(50)
async def bounded(url):
    async with sem:
        return await fetch(url)
results = await asyncio.gather(*[bounded(url) for url in urls_10k])
```

### 9.4 Antipattern 4: Not handling `asyncio.CancelledError` in long-running tasks

```python
# ❌ Cancel during long task; resources leak
async def long_task():
    conn = await acquire_db()
    await asyncio.sleep(3600)  # cancelled at 30s
    # conn never released!

# ✅ Always cleanup
async def long_task():
    conn = await acquire_db()
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await release_db(conn)
        raise
```

### 9.5 Antipattern 5: Reusing the same `asyncio.Semaphore` for different resources

```python
# ❌ One semaphore for both
db_sem = asyncio.Semaphore(10)
cache_sem = asyncio.Semaphore(10)

# ✅ Separate semaphores
resources = {
    "db": asyncio.Semaphore(10),
    "cache": asyncio.Semaphore(10),
    "llm": asyncio.Semaphore(20),
}
```

---

## 🎯 Key Takeaways

- The seven failure modes: zombie tasks, event loop starvation, semaphore deadlock, async generator misuse, lost task references, N+1 awaits, unbounded fan-out.
- The async health endpoint detects event loop stalls in <30 seconds.
- `py-spy` is the first tool for production async debugging; reads async call stacks correctly.
- `faulthandler` with `SIGUSR1` dumps stacks on demand.
- Custom watchdog detects heartbeats that don't fire — the event loop is blocked.
- CI linter (`ruff ASYNC`) catches sync-in-async at lint time.
- Test zombie tasks are cancelled; run async health in CI.
- The async runbook: capture state → identify mode → mitigate → recover → postmortem.
- Avoid ignoring async antipatterns in code review, no async health, unbounded gather, missing CancelledError cleanup, sharing semaphores.

## References

- py-spy — [github.com/benfred/py-spy](https://github.com/benfred/py-spy)
- faulthandler — [docs.python.org/3/library/faulthandler.html](https://docs.python.org/3/library/faulthandler.html)
- ruff ASYNC rules — [docs.astral.sh/ruff/rules/#flake8-async-async](https://docs.astral.sh/ruff/rules/#flake8-async-async)
- [[../../31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]
- [[../02 - ARQ Modern Async-Native|ARQ note]]
- [[../../39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering|Incident Response Note 04]]
- [[../../39 - Production Incident Response for AI Systems/03 - Triage - Diagnosing AI Incidents|Incident Response Note 03]]
- Google SRE Book — Cascading Failures — [sre.google/sre-book/addressing-cascading-failures](https://sre.google/sre-book/addressing-cascading-failures/)