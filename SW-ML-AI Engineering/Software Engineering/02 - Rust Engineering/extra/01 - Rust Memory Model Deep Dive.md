# 🧠 Rust Memory Model Deep Dive

## Introduction

Rust's memory model is the foundation of its safety guarantees, ensuring memory safety without a garbage collector. This model is built on three pillars: [[01 - Ownership and Borrowing|ownership]], [[02 - Borrowing Rules|borrowing rules]], and [[03 - Lifetimes|lifetimes]]. By understanding how Rust manages memory at the lowest level, you can write high‑performance, concurrent code that avoids data races and undefined behavior.

Unlike languages like C or C++, Rust enforces memory safety at compile‑time through its ownership system, which statically tracks who owns each piece of data. The compiler then generates efficient machine code with zero runtime overhead for most safety checks. However, in advanced scenarios—such as custom allocators, lock‑free data structures, or interfacing with hardware—you must understand the underlying memory layout and ordering guarantees.

This deep dive explores the stack vs. heap dichotomy, smart pointer internals, atomic operations, and the subtle interactions that make Rust's memory model both powerful and sometimes complex. We’ll see how real‑world systems like [[02 - Async Rust Internals|tokio]] leverage these primitives to handle millions of concurrent tasks safely.

## 1. Memory Layout and Allocation

Rust programs use both stack and heap memory, each with distinct characteristics and performance trade‑offs.

### Stack vs. Heap

- **Stack**: Fast, automatic allocation; limited size; used for values with known, fixed size (primitives, tuples, arrays, structs without `dyn` traits). Allocation is simply moving the stack pointer.
- **Heap**: Flexible, dynamic size; slower allocation; managed by allocators (global `#[global_allocator]` or per‑type). Used for `Box`, `Vec`, `String`, etc.

### Allocators

The default allocator is `alloc::System` (usually `dlmalloc` on Windows, `jemalloc` on Linux). Rust allows custom allocators via the `#[global_allocator]` attribute, enabling specialized memory pools (e.g., for embedded systems or real‑time applications).

**Real case:** The [[04 - Embedded Rust and IoT|embedded]] runtime RTIC uses a custom static allocator to avoid heap fragmentation and guarantee bounded allocation time.

⚠️ **Warning:** Mixing allocators (e.g., freeing memory allocated by one allocator with another) leads to undefined behavior. Always ensure deallocation matches the allocator that performed the allocation.

💡 **Tip:** For maximum performance, prefer stack allocation for small, short‑lived objects. Use `#[repr(transparent)]` to guarantee a struct has the same layout as its single field, enabling zero‑cost abstractions.

### Memory Layout

Rust's memory layout is deterministic for `#[repr(C)]` types, which follow C's layout rules and are essential for FFI. For non‑`repr(C)` types, the compiler may reorder fields for optimal alignment, which can affect pointer casting and unsafe code.

## 2. Smart Pointer Internals

Smart pointers provide heap allocation and runtime polymorphism while enforcing safety through RAII.

| Smart Pointer | Ownership | Thread‑Safe? | Typical Use Case |
|---------------|-----------|--------------|------------------|
| `Box<T>`      | Unique    | Yes (if `T: Send`) | Heap allocation, trait objects |
| `Rc<T>`       | Shared    | No (single‑threaded) | Graph structures, shared ownership |
| `Arc<T>`      | Shared    | Yes (if `T: Send + Sync`) | Concurrent shared ownership |
| `Cell<T>`     | In‑place  | No (single‑threaded) | Interior mutability for `Copy` types |
| `RefCell<T>`  | In‑place  | No (single‑threaded) | Runtime borrow checking |
| `Mutex<T>`    | Exclusive | Yes (if `T: Send`) | Synchronized interior mutability |
| `RwLock<T>`   | Shared read / Exclusive write | Yes (if `T: Send + Sync`) | High‑read‑contention data |

**Real case:** Tokio uses `Arc<Mutex<T>>` and `Arc<RwLock<T>>` extensively for shared state between tasks, but prefers `RwLock` for read‑heavy workloads (e.g., configuration data).

⚠️ **Warning:** `RefCell` and `Cell` are **not** thread‑safe; using them across threads causes data races. Always use `Mutex` or `RwLock` in concurrent contexts.

💡 **Tip:** When you need both shared ownership and interior mutability across threads, combine `Arc` with `Mutex` or `RwLock`. Consider `Arc<AtomicU64>` for simple atomic counters to avoid lock overhead.

## 3. Memory Ordering and Atomics

Atomic operations guarantee that reads and writes to shared memory are indivisible, crucial for lock‑free data structures. Rust's `std::sync::atomic` module provides five ordering variants:

| Ordering | Guarantees | Performance Cost |
|----------|------------|------------------|
| `Ordering::Relaxed` | No ordering constraints; only atomicity. | Lowest |
| `Ordering::Acquire` | Subsequent reads/writes cannot be moved before this load. | Medium |
| `Ordering::Release` | Prior reads/writes cannot be moved after this store. | Medium |
| `Ordering::AcqRel` | Combines Acquire (for loads) and Release (for stores). | High |
| `Ordering::SeqCst` | Total order across all `SeqCst` operations; strongest guarantee. | Highest |

**Real case:** Crossbeam's lock‑free queues use `AcqRel` ordering to synchronize producers and consumers without locks, achieving high throughput.

⚠️ **Warning:** Using `Relaxed` ordering incorrectly can lead to subtle data races; ensure you have proper happens‑before relationships via `Acquire`/`Release` pairs.

💡 **Tip:** Prefer `SeqCst` when correctness is paramount and performance is secondary; profile before switching to weaker orderings.

## 4. Practical Patterns and Pitfalls

This section covers advanced patterns and common mistakes.

### Custom Allocator Example

```rust
use std::alloc::{GlobalAlloc, Layout, System};

struct MyAllocator;

unsafe impl GlobalAlloc for MyAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Custom allocation logic (e.g., bump pointer)
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: MyAllocator = MyAllocator;
```

### Atomic Operations Example

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn main() {
    let handles: Vec<_> = (0..10).map(|_| {
        thread::spawn(|| {
            for _ in 0..1000 {
                COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(COUNTER.load(Ordering::Relaxed), 10000);
}
```

---

## 📦 Compression Code

Complete Rust script that demonstrates a custom bump allocator and atomic reference counting.

```rust
// bump_allocator.rs
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

const MEMORY_SIZE: usize = 1024 * 1024; // 1 MiB

pub struct BumpAllocator {
    memory: [u8; MEMORY_SIZE],
    offset: AtomicUsize,
}

impl BumpAllocator {
    pub fn new() -> Self {
        Self {
            memory: [0; MEMORY_SIZE],
            offset: AtomicUsize::new(0),
        }
    }

    pub fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align();
        let size = layout.size();
        let mut current = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = (current + align - 1) & !(align - 1);
            let next = aligned + size;
            if next > MEMORY_SIZE {
                return None;
            }
            match self.offset.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let ptr = unsafe { self.memory.as_ptr().add(aligned) as *mut u8 };
                    return NonNull::new(ptr);
                }
                Err(new_current) => current = new_current,
            }
        }
    }

    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }
}

use std::alloc::Layout;

fn main() {
    let alloc = BumpAllocator::new();
    let layout = Layout::from_size_align(64, 8).unwrap();
    let ptr1 = alloc.allocate(layout).expect("Allocation failed");
    let ptr2 = alloc.allocate(layout).expect("Allocation failed");
    println!("Allocated two 64‑byte blocks at {:?} and {:?}", ptr1, ptr2);
    alloc.reset();
    let ptr3 = alloc.allocate(layout).expect("Allocation after reset");
    println!("After reset, allocated at {:?}", ptr3);
}
```

## 🎯 Documented Project

### Description

A high‑performance, lock‑free bounded queue for inter‑task communication in async runtimes, using atomic operations and a custom allocator to minimize overhead.

### Functional Requirements

1. Support multiple producers and multiple consumers (MPMC).
2. Bounded capacity (configurable at compile‑time).
3. Non‑blocking `push` and `pop` operations that return `Result`.
4. No allocation after initialization (use a pre‑allocated buffer).
5. Safe for concurrent use across threads (implement `Send + Sync`).

### Main Components

- **Buffer**: Array of `UnsafeCell<Option<T>>` with atomic indices.
- **Producer/Consumer Handles**: `Arc`‑wrapped shared state with `AtomicUsize` head/tail.
- **Allocator**: Optional custom allocator for the buffer (e.g., `BumpAllocator`).
- **Ordering**: Careful use of `Acquire`/`Release` for index updates.

### Success Metrics

- Throughput > 10 million ops/sec on 8‑core machine (benchmark).
- Zero dynamic allocation after queue creation.
- No unsafe data races (verified via Miri and Loom).
- Latency < 100 ns per operation (99th percentile).

### References

- Rustonomicon: [[02 - Unsafe Rust|Unsafe Rust]]
- C++ Concurrency in Action (Rust equivalents)
- Crossbeam‑channel design documents
- Tokio’s MPMC channel implementation