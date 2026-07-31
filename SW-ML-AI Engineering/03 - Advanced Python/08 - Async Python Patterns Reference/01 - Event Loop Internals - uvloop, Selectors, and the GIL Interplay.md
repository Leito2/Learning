# 🎯 01 - Event Loop Internals — uvloop, Selectors, and the GIL Interplay

> **What actually happens when you `await` something. The selector, the event loop, the OS, the GIL. The internals that explain why async is fast and where it breaks.**

## 🎯 Learning Objectives
- Trace a coroutine from `await` to resumption at the OS-call level
- Understand the difference between `asyncio` (stdlib) and `uvloop` (libuv-based)
- Configure event loop policies for Windows, Linux, macOS, K8s
- Recognize the GIL interaction with CPU-bound async code
- Use `asyncio.Runner` (Python 3.12+) for explicit loop control
- Benchmark and choose the right event loop for production
- Diagnose event loop stalls at the OS level

## Introduction

Most async Python tutorials stop at "the event loop runs coroutines". This note goes deeper: what **is** the event loop? When you `await asyncio.sleep(1)`, what happens between the call and the resumption?

Understanding these internals explains:

- Why `uvloop` is 2-4× faster than `asyncio`
- Why async code is single-threaded despite looking concurrent
- Why `time.sleep(5)` blocks the event loop but `await asyncio.sleep(5)` doesn't
- Why the GIL is mostly irrelevant for I/O-bound async code
- How to debug a stuck event loop at the OS level

This is the deep dive that separates the senior engineer from the mid-level engineer.

---

## 1. The Mental Model — Step by Step

When you write:

```python
async def fetch(url):
    response = await httpx.AsyncClient().get(url)
    return response.json()


async def main():
    data = await fetch("https://example.com")
    print(data)


asyncio.run(main())
```

What happens at the OS level:

```
1. asyncio.run(main()) starts the event loop
2. main() runs until it hits `await fetch(...)`
3. fetch() runs until it hits `await client.get(url)`
4. client.get(url) calls socket.recv (OS-level)
5. The socket is registered with the OS event selector (epoll/kqueue/IOCP)
6. The event loop checks for other ready tasks (none) and waits on selector
7. The OS notifies when the socket has data (or timeout)
8. The event loop resumes fetch() → main() → returns
```

The **key insight**: the event loop does CPU work, then waits for the OS. While waiting, no CPU is used. Other coroutines run during the CPU work.

---

## 2. The Event Loop Implementation

`asyncio` uses `selectors` module — a wrapper around the OS's event notification mechanism:

| OS | Event mechanism | asyncio implementation |
|----|-----------------|------------------------|
| Linux | `epoll` | `EpollSelector` |
| macOS | `kqueue` | `KqueueSelector` |
| Windows | `IOCP` (I/O Completion Ports) | `WindowsProactorEventLoop` (Python 3.8+) |
| All | `select` (fallback) | `SelectSelector` |

Linux's `epoll` is the fastest for most workloads. `uvloop` reimplements the event loop using `libuv` (the same library that powers Node.js), achieving 2-4× speedup over stdlib asyncio.

### 2.1 Using uvloop in production

```python
import asyncio
import uvloop

# Replace stdlib event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Now every asyncio.run() uses uvloop
async def main():
    ...


asyncio.run(main())  # uses uvloop under the hood
```

Or in a FastAPI app:

```python
import uvloop
import asyncio


async def main():
    config = uvicorn.Config("app.main:app", host="0.0.0.0", port=8000, loop="uvloop")
    server = uvicorn.Server(config)
    await server.serve()


asyncio.run(main())
```

uvloop is **drop-in compatible** — no code changes needed, just install + use the policy.

---

## 3. The GIL and Async

The **Global Interpreter Lock** (GIL) is Python's mutex that prevents multiple threads from executing Python bytecode simultaneously. For async code, the GIL is **mostly irrelevant** because:

1. Async code is single-threaded; there's no concurrency at the Python level
2. The GIL is released during I/O operations (socket calls, file reads)
3. While one coroutine is waiting for I/O, another can run (no GIL contention)

```
Time →
Thread 1:  await socket.recv  ←── GIL released ──→  await socket.recv
          CPU work                wait                  CPU work
```

The GIL becomes a problem **only when**:
- CPU-bound code runs on the same thread
- Sync I/O blocks the thread
- CPU-intensive operations block the event loop

For pure I/O-bound async code (HTTP, DB, files), the GIL is invisible.

### 3.1 The CPU-bound problem

```python
# ❌ This CPU-bound work blocks the entire event loop
async def bad_json_parse(text: str) -> dict:
    return json.loads(text)  # CPU-bound; blocks event loop for 100ms+

# ✅ Move CPU-bound work to a thread pool
async def good_json_parse(text: str) -> dict:
    return await asyncio.to_thread(json.loads, text)
```

`asyncio.to_thread` (Python 3.9+) runs the sync function in a thread pool, releasing the event loop during CPU work. The thread runs in parallel with the main event loop.

### 3.2 CPU-bound parallelism: multiprocessing and subinterpreters

For true CPU-bound parallelism (e.g., training a model), use `multiprocessing` or subinterpreters (covered in Note 04):

```python
from concurrent.futures import ProcessPoolExecutor


async def train_model(data):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, train_sync, data)
    return result


def train_sync(data):
    # CPU-bound training
    ...
```

The `ProcessPoolExecutor` runs in a separate process — no GIL contention.

---

## 4. The Selector and the OS

The selector is the bridge between asyncio and the OS:

```python
import selectors
import socket


selector = selectors.DefaultSelector()  # platform-specific


async def read_async(fd: int, n: int) -> bytes:
    loop = asyncio.get_running_loop()
    
    # Register the file descriptor with the selector
    future = loop.create_future()
    
    def on_readable():
        loop.remove_reader(fd)
        try:
            data = os.read(fd, n)
            future.set_result(data)
        except Exception as e:
            future.set_exception(e)
    
    loop.add_reader(fd, on_readable)
    return await future
```

When the OS detects the file descriptor is readable, the callback fires, completing the future, which resumes the awaiting coroutine.

### 4.1 Linux epoll vs macOS kqueue vs Windows IOCP

| OS | Mechanism | Edge-triggered | Performance |
|----|-----------|---------------|-------------|
| Linux | `epoll` | Yes (default) | Best for many connections |
| macOS | `kqueue` | Yes | Comparable to epoll |
| Windows | IOCP | Async I/O natively | Different model (completion-based) |

For Linux production servers, `epoll` (via stdlib asyncio or uvloop) is the standard.

For Windows servers, `ProactorEventLoop` (Python 3.8+) uses IOCP. It's different from the Unix selector model — async I/O is kernel-level, not user-space.

### 4.2 The `selector` module

You can interact with the selector directly:

```python
import selectors


selector = selectors.DefaultSelector()

# Register
key = selector.register(fileobj, selectors.EVENT_READ, data="my_data")

# Wait for events
events = selector.select(timeout=1.0)  # returns [(key, events)]

# Unregister
selector.unregister(fileobj)
```

This is what asyncio does under the hood.

---

## 5. `asyncio.Runner` — Explicit Loop Control (Python 3.12+)

Python 3.12 introduced `asyncio.Runner` for explicit event loop control:

```python
import asyncio


async def main():
    ...


# Standard pattern
asyncio.run(main())

# New explicit pattern (Python 3.12+)
with asyncio.Runner() as runner:
    runner.run(main())
```

The `Runner` API provides:
- Explicit `close()` to clean up the loop
- Better control over loop lifecycle
- Integration with `asyncio.set_event_loop()` for explicit policies

Use `Runner` when:
- You need to run multiple loops in sequence
- You need explicit cleanup of resources
- You're building a library that exposes async APIs

---

## 6. Event Loop Stalls at the OS Level

When the event loop "hangs", it's usually because of an OS-level block:

```bash
# py-spy dump during a stall
py-spy dump --pid 12345 --non-blocking | head -50

# Output might show:
# _selector.select (BLOCKED for 12.3s)
# This means the OS is not notifying the selector
```

The OS is not notifying → the socket isn't ready → the event loop is blocked waiting.

### 6.1 Common causes

| Cause | Symptom | Fix |
|------|---------|-----|
| **DNS lookup hangs** | Selector blocked for many seconds | Use async DNS (`aiodns`) |
| **TCP connect timeout** | Single connection stuck | Set explicit `connect_timeout` |
| **File descriptor exhausted** | Many sockets waiting | Increase `ulimit -n` |
| **Dead socket** | Connection dead but not detected | Enable TCP keepalive |
| **Event loop CPU-blocked** | Selector never gets CPU | Find and remove the blocking call |

### 6.2 TCP keepalive for long-lived connections

```python
import socket


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)  # start probes after 60s idle
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # probe every 10s
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)  # give up after 3 probes
```

This detects dead connections within 90 seconds — critical for LLM API calls that hang.

---

## 7. Benchmarking Event Loops

```python
import asyncio
import time


async def bench(n: int = 100_000):
    async def noop():
        pass
    
    start = time.perf_counter()
    await asyncio.gather(*[noop() for _ in range(n)])
    elapsed = time.perf_counter() - start
    
    print(f"asyncio: {n / elapsed:.0f} coroutines/sec")


asyncio.run(bench())

# vs uvloop:
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
asyncio.run(bench())  # typically 2-3× faster
```

Typical results on a modern Linux server:
- `asyncio`: 1-2M coroutines/sec
- `uvloop`: 3-5M coroutines/sec

The difference comes from `libuv`'s optimized event loop implementation in C, vs stdlib's pure Python.

### 7.1 Choosing the event loop

| Workload | Recommendation |
|----------|----------------|
| Web server (FastAPI) | **uvloop** (2× throughput) |
| LLM inference client | **uvloop** (lower latency) |
| Background workers (ARQ) | **uvloop** |
| DB-heavy workloads | Either (DB call dominates) |
| Windows servers | stdlib asyncio (uvloop doesn't support Windows) |
| macOS development | Either |
| Linux K8s | **uvloop** |

---

## 8. Antipatterns

### 8.1 Antipattern 1: Forgetting `uvloop` policy

```python
# ❌ Default asyncio is slower
async def main():
    ...

asyncio.run(main())

# ✅ Use uvloop for production
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
asyncio.run(main())
```

### 8.2 Antipattern 2: CPU-bound work without `to_thread`

```python
# ❌ CPU-bound json.loads blocks the loop
async def bad(text):
    return json.loads(text)

# ✅ Use to_thread
async def good(text):
    return await asyncio.to_thread(json.loads, text)
```

### 8.3 Antipattern 3: No TCP keepalive

```python
# ❌ Default socket has no keepalive; dead connections hang forever
sock = socket.socket()
sock.connect(("example.com", 80))

# ✅ Enable keepalive
sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
```

### 8.4 Antipattern 4: Re-implementing the event loop

```python
# ❌ Don't re-implement the event loop
def my_event_loop():
    while True:
        for task in tasks:
            if task.ready():
                task.run()
        time.sleep(0.001)

# ✅ Use asyncio or uvloop
asyncio.run(main())
```

### 8.5 Antipattern 5: Confusing threads with async

```python
# ❌ Mixing threads and async poorly
import threading

def background_worker():
    while True:
        time.sleep(1)
        do_work()

t = threading.Thread(target=background_worker)
t.start()  # separate thread; communicates via shared state (bad)

# ✅ Use asyncio tasks for cooperative concurrency
async def background_worker():
    while True:
        await asyncio.sleep(1)
        await do_work()

asyncio.create_task(background_worker())  # cooperative; communicates via asyncio primitives
```

---

## 🎯 Key Takeaways

- The event loop uses the OS event notification mechanism (epoll/kqueue/IOCP) to wait without CPU usage.
- uvloop reimplements the loop with libuv; 2-4× faster on Linux/macOS.
- The GIL is mostly invisible for I/O-bound async code; CPU-bound work needs `to_thread` or processes.
- TCP keepalive detects dead connections; without it, hangs are invisible.
- `asyncio.Runner` (3.12+) gives explicit event loop control for library authors.
- Benchmarking shows asyncio at ~1M coroutines/sec; uvloop at ~3M.
- Choose uvloop for Linux/macOS production; stdlib for Windows.
- Avoid forgetting uvloop policy, CPU-bound without to_thread, no TCP keepalive, re-implementing the loop, mixing threads with async poorly.

## References

- Python asyncio internals — [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)
- PEP 678 — Exception Group enhancements — [peps.python.org/pep-0678](https://peps.python.org/pep-0678/)
- uvloop — [github.com/MagicStack/uvloop](https://github.com/MagicStack/uvloop)
- PEP 684 — Per-Interpreter GIL — [peps.python.org/pep-0684](https://peps.python.org/pep-0684/)
- PEP 554 — Multiple Interpreters in the Stdlib — [peps.python.org/pep-0554](https://peps.python.org/pep-0554/)
- selectors module — [docs.python.org/3/library/selectors.html](https://docs.python.org/3/library/selectors.html)
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]
- [[../../10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI/06 - Async Debugging in Production - Common Pitfalls and Detection|Note — Async Debugging]]
- [[../../03 - Advanced Python/03 - Python Avanzado/06 - Concurrencia - Threading y Asyncio|Python Avanzado Concurrencia]]