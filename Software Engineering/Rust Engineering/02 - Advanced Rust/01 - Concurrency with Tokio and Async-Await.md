# ⚡ Concurrency with Tokio and Async-Await

## Introduction

Rust's concurrency model is unique because it enforces thread safety at compile time through the [[00 - Welcome to Advanced Rust|Send and Sync traits]]. Unlike languages that rely on garbage collection or runtime checks, Rust prevents data races before your program ever runs. This module explores how to harness that power in asynchronous contexts using Tokio, the dominant async runtime in the Rust ecosystem. Understanding Tokio is essential for building network services, high-throughput data pipelines, and responsive applications that handle millions of concurrent operations without drowning in OS threads.

Async/await in Rust is not magic — it is a zero-cost abstraction built on top of futures, polling, and executors. When you write an `async fn`, the compiler transforms it into a state machine that implements the `Future` trait. This state machine is then driven to completion by an executor, typically Tokio's multi-threaded scheduler. The beauty of this design is that you get the ergonomics of sequential-looking code with the performance of event-driven architectures. In this module, we dissect every layer of that stack, from the traits that guarantee safety to the runtime that delivers performance.

## 1. Send, Sync, and Thread Safety at Compile Time

Deep conceptual explanation:

- **`Send`**: A type is `Send` if it is safe to move its value to another thread. Most types in Rust are `Send`, but types like `Rc<T>` (non-atomic reference counting) are not, because moving them across threads could corrupt the reference count.
- **`Sync`**: A type is `Sync` if it is safe to share references to it between threads. `Sync` essentially means `&T` is `Send`. Primitive types, immutable references, and mutex-guarded data are `Sync`.
- The compiler automatically derives these traits for composite types if all their fields implement them. This means a `struct` containing only `Send` fields is itself `Send`.
- These traits form the foundation of Rust's fearless concurrency. You cannot accidentally share a non-thread-safe type across tasks because the code will simply not compile.

⚠️ **Warning:** Manually implementing `unsafe impl Send for MyType {}` is possible but extremely dangerous. Only do this when you have proven thread safety through other synchronization primitives.

💡 **Tip:** Use `tokio::sync::Mutex` instead of `std::sync::Mutex` in async contexts. The standard library mutex blocks the OS thread, which starves the Tokio scheduler. Tokio's async mutex yields control back to the runtime while waiting.

Real case: **Discord** uses Tokio for millions of concurrent connections. Their real-time messaging backend handles voice, video, and text across shards, with each shard running as a Tokio task. By leveraging Rust's `Send` and `Sync` guarantees, they avoid entire classes of concurrency bugs that plague garbage-collected languages at scale.

## 2. Tokio Runtime: Tasks, Schedulers, and I/O Drivers

Tokio is not just a library — it is a full asynchronous runtime. Understanding its architecture is crucial for tuning performance.

| Component | Description | Configuration |
|-----------|-------------|---------------|
| Multi-threaded scheduler | Work-stealing thread pool for CPU-bound and concurrent tasks | `#[tokio::main]` (default) |
| Current-thread scheduler | Single-threaded executor for low-latency or deterministic execution | `#[tokio::main(flavor = "current_thread")]` |
| I/O driver | Epoll/kqueue/IOCP integration for async network and file operations | Enabled by default |
| Timer | Efficient time wheel for `sleep`, `timeout`, and intervals | Enabled by default |
| Blocking pool | Separate thread pool for `spawn_blocking` to avoid blocking the async executor | Auto-scales |

Table: Tokio vs async-std vs smol

| Feature | Tokio | async-std | smol |
|---------|-------|-----------|------|
| Maturity | Production-ready, 5+ years | Stable, smaller ecosystem | Minimal, embeddable |
| Scheduler | Work-stealing, multi-threaded | Similar to async-std runtime | Single-threaded, scalable |
| Ecosystem | Massive (Axum, Tower, Hyper) | Moderate (Tide, Surf) | Small but growing |
| Compilation time | Slower (many features) | Moderate | Fast |
| Use case | Microservices, infrastructure | Applications, prototyping | Embedded, constrained envs |

Formula:

```
Concurrency = Tasks / (Threads × Time)
```

This formula captures the essence of async efficiency. A single Tokio thread can multiplex thousands of tasks because most tasks are I/O-bound and spend their time waiting. Increasing raw thread count (the denominator) without bound actually reduces throughput due to context-switching overhead.

## 3. Async/Await: Futures, Polling, and Executors

When you mark a function `async`, the compiler generates an anonymous type implementing `Future`. This future is lazy — it does nothing until polled. The executor (Tokio's scheduler) repeatedly calls `Future::poll()`, driving the state machine forward until it returns `Ready`.

```mermaid
graph TD
    A[async fn main] --> B[Compiler generates Future state machine]
    B --> C[Future is lazy: nothing happens yet]
    C --> D[Tokio executor calls .poll()]
    D --> E{State?}
    E -->|Pending| F[Await point hit: register waker]
    F --> G[I/O or timer completes]
    G --> H[Waker invoked]
    H --> D
    E -->|Ready| I[Value returned]
```

![Futures execution model](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Futures_and_promises.svg/640px-Futures_and_promises.svg.png)

Spawning tasks decouples the future from the current execution context. `tokio::spawn` submits the task to the runtime's queue, where any worker thread can steal and execute it. This is the key to achieving true parallelism within an async program.

![Tokio runtime architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Thread_pool.svg/640px-Thread_pool.svg.png)

## 4. Practical Tokio Server

Rust code blocks:

```rust
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct AppState {
    counter: Arc<Mutex<u64>>,
}

async fn handle_connection(mut socket: TcpStream, state: AppState) {
    let mut buf = [0u8; 1024];
    loop {
        match socket.read(&mut buf).await {
            Ok(0) => return, // Connection closed
            Ok(n) => {
                let mut counter = state.counter.lock().await;
                *counter += 1;
                let response = format!("Request #{} received\n", counter);
                if socket.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
            }
            Err(e) => {
                eprintln!("Failed to read from socket: {}", e);
                return;
            }
        }
    }
}

#[tokio::main]
async fn main() -> tokio::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    let state = AppState {
        counter: Arc::new(Mutex::new(0)),
    };

    println!("Server listening on 127.0.0.1:8080");

    loop {
        let (socket, _) = listener.accept().await?;
        let state = state.clone();
        tokio::spawn(async move {
            handle_connection(socket, state).await;
        });
    }
}
```

This server demonstrates:
- `Arc<Mutex<T>>` for shared mutable state across tasks
- `tokio::spawn` for handling each connection concurrently
- Async I/O with `read` and `write_all`
- Graceful connection closure handling

## 5. Structured Concurrency and Cancellation

Structured concurrency means that child tasks should not outlive their parent. Tokio provides `tokio::select!` and `JoinSet` to manage task lifecycles. Cancellation tokens allow cooperative shutdown, where tasks periodically check `token.is_cancelled()` and exit cleanly.

⚠️ **Warning:** Dropping a `JoinHandle` does NOT abort the task. The task becomes a "detached" background process. Always `await` your handles or use `AbortHandle` if you need explicit cancellation.

💡 **Tip:** Use `tokio::time::timeout` to bound the execution of any future. This prevents slow handlers from monopolizing worker threads indefinitely.

---

## 📦 Compression Code

Complete Rust script:

```rust
use tokio::time::{sleep, Duration};
use tokio::task::JoinSet;

#[tokio::main]
async fn main() {
    let mut set = JoinSet::new();

    for i in 0..10 {
        set.spawn(async move {
            sleep(Duration::from_millis(100 * i)).await;
            println!("Task {} complete", i);
            i * i
        });
    }

    while let Some(res) = set.join_next().await {
        match res {
            Ok(val) => println!("Got result: {}", val),
            Err(e) => eprintln!("Task panicked: {}", e),
        }
    }
}
```

## 🎯 Documented Project

### Description

Build an async HTTP echo server that counts total requests and supports graceful shutdown. The server must handle at least 10,000 concurrent connections without crashing or leaking memory.

### Functional Requirements

1. Accept TCP connections on a configurable port and echo back received data.
2. Maintain a global atomic request counter visible to all worker tasks.
3. Implement a `/stats` endpoint that returns the current request count as JSON.
4. Support graceful shutdown on SIGINT, waiting up to 5 seconds for open connections to close.
5. Log every connection with a unique task ID and timestamp.

### Main Components

- `main.rs`: Entry point, signal handling, and server lifecycle management.
- `server.rs`: TCP accept loop and connection dispatch using `tokio::spawn`.
- `handler.rs`: Per-connection read/write loop with request counting.
- `state.rs`: Shared `AppState` using `Arc<AtomicU64>` for lock-free counters.

### Success Metrics

- Server handles 10,000 concurrent connections verified via `wrk` or `oha`.
- Memory usage remains stable under load for 5 minutes (no leaks detected via `valgrind` or `dhat`).
- Graceful shutdown completes within 5 seconds with zero dropped requests.
- 100% of code paths covered by unit and integration tests.

### References

- [Tokio Documentation](https://tokio.rs/)
- [Rust Async Book](https://rust-lang.github.io/async-book/)
- [Discord Engineering Blog: Rust at Discord](https://discord.com/blog/why-discord-is-switching-from-go-to-rust)
