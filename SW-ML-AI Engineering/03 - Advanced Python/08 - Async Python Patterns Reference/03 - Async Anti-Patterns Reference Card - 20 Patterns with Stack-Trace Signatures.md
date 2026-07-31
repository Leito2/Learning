# 🎯 03 - Async Anti-Patterns Reference Card — 20 Patterns with Stack-Trace Signatures

> **The lookup table. When async code hangs in production, find the matching pattern, apply the fix. Twenty patterns, twenty signatures, twenty fixes.**

## 🎯 Learning Objectives
- Recognize the 20 most common async anti-patterns by stack-trace signature
- Map each pattern to its root cause, detection method, and fix
- Apply the reference card during code review and incident response
- Run the async linter (`ruff ASYNC`) to catch patterns at CI time
- Build a custom `pytest-asyncio` test suite that catches antipatterns pre-deploy
- Recognize when async is the wrong tool (use threads or processes instead)

## Introduction

This note is the **reference card**. When async code breaks, you grep your stack trace for the pattern signature, find the fix, and ship it. The 20 patterns below cover >95% of async bugs I've seen in production.

Each pattern includes:

- **Stack-trace signature** — what your py-spy dump looks like
- **Root cause** — what the bug actually is
- **Detection** — how to find it
- **Fix** — the corrected pattern
- **Test pattern** — how to catch it at CI time

---

## 1. Sync I/O in Async Function

**Stack signature:**
```
File "app.py", line 42, in fetch_user
    time.sleep(5)  # OR requests.get(url)
```

**Root cause:** A blocking sync call inside an async function blocks the event loop.

**Fix:**
```python
async def fetch_user(user_id):
    await asyncio.sleep(5)  # cooperative sleep
    return await asyncio.to_thread(requests.get, url)  # or use httpx
```

**Test:**
```python
def test_no_sync_sleep():
    for f in glob("app/**/*.py"):
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and ast.unparse(child.func) == "time.sleep":
                        pytest.fail(f"Sync sleep in async: {f}:{child.lineno}")
```

---

## 2. Missing `await`

**Stack signature:**
```
TypeError: object coroutine is not iterable
AttributeError: 'coroutine' object has no attribute 'json'
```

**Root cause:** The function returns a coroutine, but the caller forgot to await.

**Fix:**
```python
# ❌ Returns coroutine, not result
async def fetch():
    return await httpx.AsyncClient().get(url)

result = fetch()  # bug: missing await
result = await fetch()  # fix

# Use ruff ASYNC rules to catch this:
# select = ["ASYNC"]
# force-await-async-call = true
```

---

## 3. Swallowed `CancelledError`

**Stack signature:**
```
Task is "done" but resources still held
Process memory grows over time
```

**Root cause:** `except CancelledError: pass` or `except CancelledError: return`.

**Fix:**
```python
async def bad():
    try:
        await expensive_op()
    except asyncio.CancelledError:
        return  # SUPPRESSES cancellation

async def good():
    try:
        await expensive_op()
    except asyncio.CancelledError:
        await cleanup()
        raise  # always re-raise after cleanup
```

**Test:**
```python
@pytest.mark.asyncio
async def test_cancellation_propagates():
    async def cancelled_task():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return  # bug: should re-raise
    
    task = asyncio.create_task(cancelled_task())
    await asyncio.sleep(0.01)
    task.cancel()
    
    with pytest.raises(asyncio.CancelledError):
        await task
```

---

## 4. Unbounded `gather`

**Stack signature:**
```
asyncio.gather(*[task() for _ in range(10000)])  # creates 10K coroutines
File descriptor exhaustion in logs
```

**Root cause:** `asyncio.gather` without a semaphore creates unbounded concurrent tasks.

**Fix:**
```python
sem = asyncio.Semaphore(50)

async def bounded(coro):
    async with sem:
        return await coro

results = await asyncio.gather(*[bounded(fetch_one(p)) for p in prompts])
```

---

## 5. No Timeout on External Calls

**Stack signature:**
```
Selector blocked on socket.recv for 30+ seconds
Event loop never resumes
```

**Root cause:** External calls without timeouts hang forever if the upstream is unresponsive.

**Fix:**
```python
async def with_timeout(coro, timeout=10.0):
    async with asyncio.timeout(timeout):
        return await coro
```

---

## 6. Event Loop Starvation (CPU-bound work)

**Stack signature:**
```
py-spy dump shows one frame with json.loads or pandas operations
All other coroutines blocked
```

**Root cause:** CPU-bound sync work in async function blocks the event loop.

**Fix:**
```python
async def parse_json(text):
    return await asyncio.to_thread(json.loads, text)
```

---

## 7. Event Loop in Event Loop

**Stack signature:**
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Root cause:** `asyncio.run()` called from inside a running loop.

**Fix:**
```python
# ❌ RuntimeError
async def outer():
    result = asyncio.run(inner())

# ✅ Await instead
async def outer():
    result = await inner()

# If you must run a separate loop, use asyncio.Runner
async def outer():
    with asyncio.Runner() as runner:
        runner.run(inner())
```

---

## 8. Lost Task Reference

**Stack signature:**
```
asyncio.all_tasks() shows empty
Work scheduled but never completes
```

**Root cause:** `asyncio.create_task(...)` without keeping a reference; task may be garbage-collected.

**Fix:**
```python
background_tasks = set()

def fire_and_forget():
    task = asyncio.create_task(do_work())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

---

## 9. Async Generator Misuse

**Stack signature:**
```
TypeError: 'async_generator' object is not iterable
```

**Root cause:** Sync iteration of async generator.

**Fix:**
```python
# ❌ next() on async generator
gen = stream_results()
first = next(gen)

# ✅ async for
async for chunk in stream_results():
    process(chunk)
```

---

## 10. N+1 Async Awaits

**Stack signature:**
```
10 users × 50ms = 500ms (sequential)
vs 10 users × 50ms = 50ms (parallel)
```

**Root cause:** Sequential `await` in a loop.

**Fix:**
```python
# ❌ Sequential
for user_id in user_ids:
    users.append(await fetch_user(user_id))

# ✅ Parallel
users = await asyncio.gather(*[fetch_user(uid) for uid in user_ids])
```

---

## 11. Semaphore Deadlock

**Stack signature:**
```
All workers blocked on semaphore.acquire
asyncio.all_tasks() shows many pending tasks
```

**Root cause:** Semaphore acquired but never released (e.g., on exception).

**Fix:**
```python
# ✅ Use `async with` (auto-release)
async def fetch_with_limit(url):
    async with sem:
        return await fetch(url)

# ✅ Or try/finally
async def fetch_with_limit(url):
    await sem.acquire()
    try:
        return await fetch(url)
    finally:
        sem.release()
```

---

## 12. Blocking Sync Subprocess

**Stack signature:**
```
subprocess.run blocks the event loop for 30s
All other coroutines wait
```

**Root cause:** `subprocess.run` is sync; blocks event loop.

**Fix:**
```python
import asyncio

async def run_subprocess(cmd):
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return stdout
```

---

## 13. Sharing Event Loop Across Threads

**Stack signature:**
```
RuntimeError: <Socket> ... is bound to a different event loop
```

**Root cause:** Object created in one thread's loop used in another.

**Fix:**
```python
# ❌ Object created in thread A, used in thread B
# Each thread must have its own event loop
# Or use asyncio.run() per thread

def thread_target():
    asyncio.run(async_main())
```

---

## 14. Race Condition on Shared State

**Stack signature:**
```
Counter shows wrong value after concurrent updates
Two coroutines read/write the same variable
```

**Root cause:** No synchronization on shared state.

**Fix:**
```python
counter_lock = asyncio.Lock()
counter = 0

async def increment():
    global counter
    async with counter_lock:
        counter += 1
```

---

## 15. Memory Leak with Unbounded Queue

**Stack signature:**
```
Memory grows unboundedly
Queue.qsize() is huge but consumer is slow
```

**Root cause:** `asyncio.Queue` with no maxsize accumulates without backpressure.

**Fix:**
```python
# ❌ Unbounded
queue = asyncio.Queue()

# ✅ Bounded (raises QueueFull when full)
queue = asyncio.Queue(maxsize=1000)

# Producer waits if full (backpressure)
await queue.put(item)
```

---

## 16. Blocking DNS Resolution

**Stack signature:**
```
Selector blocked on getaddrinfo for 30s
```

**Root cause:** Sync DNS lookup (e.g., in some libraries).

**Fix:**
```python
import aiodns
import asyncio


async def async_resolve(hostname):
    resolver = aiodns.DNSResolver(loop=asyncio.get_running_loop())
    return await resolver.gethostbyname(hostname, socket.AF_INET)
```

---

## 17. `requests` Library in Async Code

**Stack signature:**
```
httpx.AsyncClient() not used
requests.get() blocks event loop
```

**Root cause:** `requests` is sync; shouldn't be used in async code.

**Fix:**
```python
# ❌
import requests
async def fetch():
    return requests.get(url).json()

# ✅
import httpx
async def fetch():
    async with httpx.AsyncClient() as c:
        return (await c.get(url)).json()
```

---

## 18. Sleep Inside Async Test

**Stack signature:**
```
Tests are flaky; sometimes pass, sometimes timeout
```

**Root cause:** `time.sleep` in tests makes them slow and flaky.

**Fix:**
```python
# ❌ Real time
async def test():
    time.sleep(2)  # slow test
    assert something_happened()

# ✅ Async wait for the actual condition
async def test():
    async for _ in range(20):  # wait up to 2s
        if await something_happened():
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("Condition not met within timeout")
```

---

## 19. Mixing Sync and Async Libraries Without Boundary

**Stack signature:**
```
Module A is async; Module B is sync
Module A awaits Module B (blocking)
```

**Root cause:** Calling a sync function from async without `to_thread`.

**Fix:**
```python
# ✅ Always wrap sync at boundary
async def call_sync_library():
    return await asyncio.to_thread(sync_func, arg1, arg2)
```

---

## 20. Async Test Without Proper Cleanup

**Stack signature:**
```
Tests leak resources (DB connections, Redis pools)
Tests pass individually but fail when run together
```

**Root cause:** Fixtures don't clean up async resources.

**Fix:**
```python
@pytest_asyncio.fixture
async def db_session():
    session = await create_session()
    try:
        yield session
    finally:
        await session.close()  # always close
```

---

## The 20-Pattern Reference Card (Quick Lookup)

| # | Pattern | Signature (one-liner) | Fix |
|---|---------|----------------------|-----|
| 1 | Sync I/O in async | `time.sleep` or `requests.get` in async fn | `await asyncio.sleep`, `asyncio.to_thread` |
| 2 | Missing `await` | `'coroutine' has no attribute` | Add `await`, use ruff ASYNC |
| 3 | Swallowed CancelledError | Task resources leak | `except: ... raise` |
| 4 | Unbounded `gather` | 10K coroutines spawned | `Semaphore(N)` |
| 5 | No timeout | Selector blocked 30s+ | `asyncio.timeout(N)` |
| 6 | CPU-bound in async | `json.loads` blocks loop | `asyncio.to_thread` |
| 7 | Loop-in-loop | `RuntimeError: cannot call run()` | `await inner()` instead |
| 8 | Lost task ref | Task GC'd before complete | Keep reference in set |
| 9 | Async gen misuse | `TypeError: not iterable` | `async for` not `next()` |
| 10 | N+1 awaits | Sequential loop | `asyncio.gather(*tasks)` |
| 11 | Semaphore deadlock | All workers blocked | `async with` or try/finally |
| 12 | Sync subprocess | `subprocess.run` blocks loop | `asyncio.create_subprocess_exec` |
| 13 | Cross-thread loop | "bound to different loop" | One loop per thread |
| 14 | Race condition | Counter wrong after concurrent updates | `asyncio.Lock` |
| 15 | Unbounded queue | Memory grows | `Queue(maxsize=N)` |
| 16 | Sync DNS | Selector blocked on getaddrinfo | `aiodns` |
| 17 | `requests` in async | Event loop blocked | Use `httpx.AsyncClient` |
| 18 | Sync sleep in tests | Tests slow and flaky | `asyncio.sleep` + condition polling |
| 19 | Sync lib in async | Sync call blocks loop | `asyncio.to_thread` at boundary |
| 20 | Test fixture leak | Resources leak between tests | `try/finally` cleanup |

---

## Antipatterns (the meta-antipatterns)

### Meta-Antipattern 1: Not using `ruff ASYNC` in CI

```toml
# pyproject.toml
[tool.ruff.lint]
select = ["ASYNC"]
```

This catches 70% of these patterns at lint time. **Always enable.**

### Meta-Antipattern 2: No async code review checklist

```markdown
## Async Code Review Checklist

- [ ] No `time.sleep` in async functions
- [ ] All async calls are awaited
- [ ] All external calls have timeouts
- [ ] All bounded resources use `async with`
- [ ] All semaphores are released on exceptions
- [ ] No fire-and-forget without reference
- [ ] CPU-bound work uses `to_thread`
- [ ] Tests use `async for` not `next()`
- [ ] Test fixtures use `try/finally` cleanup
- [ ] Concurrency bounded by semaphore
```

### Meta-Antipattern 3: Not running `py-spy` in CI

Run `py-spy dump` periodically in CI to catch event loop stalls in tests:

```bash
py-spy dump --pid $TEST_PID --non-blocking --duration 30
```

---

## 🎯 Key Takeaways

- 20 patterns cover >95% of async production bugs.
- Each pattern has a stack-trace signature, root cause, and fix.
- The single most impactful prevention: `ruff ASYNC` linter in CI.
- The single most impactful detection: py-spy dump in production.
- Code review checklist catches patterns pre-merge.
- Tests with proper cleanup prevent fixture leaks.

## References

- ruff ASYNC rules — [docs.astral.sh/ruff/rules/#flake8-async-async](https://docs.astral.sh/ruff/rules/#flake8-async-async)
- py-spy — [github.com/benfred/py-spy](https://github.com/benfred/py-spy)
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]
- [[../../10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI/06 - Async Debugging in Production - Common Pitfalls and Detection|Note — Async Debugging]]
- [[../01 - Event Loop Internals - uvloop, Selectors, and the GIL Interplay|Note 01 — Event Loop Internals]]
- [[../02 - Async Library Ecosystem - Decision Framework for 2026|Note 02 — Library Ecosystem]]