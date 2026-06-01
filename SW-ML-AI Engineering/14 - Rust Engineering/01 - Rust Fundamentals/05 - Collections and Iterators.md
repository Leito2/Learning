# 🦀 5 - Collections and Iterators

## 🎯 Learning Objectives

- Select the appropriate collection type based on algorithmic complexity and memory layout
- Manipulate sequences and associative containers using idiomatic Rust APIs
- Implement the `Iterator` trait to create custom lazy streams over domain-specific data
- Chain adapter methods to build fused, zero-cost data pipelines

## Introduction

Machine learning is data-intensive. Whether storing token embeddings in a vocabulary map, buffering a batch of images in a vector, or streaming sensor readings through a transformation pipeline, the choice of collection directly impacts latency, throughput, and memory safety. Rust's standard library provides collections that are generics-aware, allocator-coupled, and optimized for cache locality.

Iterators are not merely a convenience — they are the canonical abstraction for sequential access. Because the borrow checker proves iterator adapters are side-effect free, libraries like Rayon can safely parallelize chains with a single method call. A data preprocessing pipeline written for a single core can scale to a multi-GPU cluster without rewriting core logic. This connects with [[01 - Ownership, Borrowing, and Lifetimes|safe memory management]] and [[02 - Types, Traits, and Generics|generic abstractions]].

---

## 1. Linear Collections

`Vec<T>` is a dynamic array with geometric reallocation (typically doubling capacity). It provides O(1) amortized push and O(1) indexing. `VecDeque<T>` extends this with a ring buffer for O(1) insertion and removal at both ends.

```rust
// Vec: the default sequential collection
let mut batch: Vec<f32> = Vec::with_capacity(64); // pre-allocate
batch.extend([0.1, 0.2, 0.3]);
batch.push(0.4);

// get returns Option<&T> — safe indexing
let third = batch.get(2); // Some(&0.3)
let tenth = batch.get(9); // None

// VecDeque: FIFO queue
use std::collections::VecDeque;
let mut queue: VecDeque<String> = VecDeque::new();
queue.push_back("request_1".into());
queue.push_back("request_2".into());
let next = queue.pop_front(); // Some("request_1")
```

❌ **Antipattern:** Indexing with `[idx]` panics on out-of-bounds — dangerous when indices come from model predictions.

✅ **Preferred:** Always use `.get()` for fallible indexing:

```rust
fn safe_lookup(weights: &[f32], idx: usize) -> Option<&f32> {
    weights.get(idx)
}
```

❌ **Antipattern:** Using `LinkedList` for general-purpose sequences. Cache thrashing makes it slower than `Vec` even for insertions in the middle.

✅ **Benchmark before choosing `LinkedList`** — in practice, `Vec` wins for almost all workloads due to contiguous memory layout and CPU cache prefetching.

⚠️ **Memory layout:** A `Vec` stores three words on the stack (ptr, len, cap) and the data contiguously on the heap. This contiguity is why iterating a `Vec` is cache-friendly.

💡 Use `shrink_to_fit()` after bulk loading to release excess capacity:

```rust
let mut data: Vec<f32> = Vec::with_capacity(10_000);
// ... fill 5,000 elements
data.shrink_to_fit(); // capacity drops from 10,000 to 5,000
```

**Caso real:** An inference request buffer uses `VecDeque<Request>` for FIFO scheduling. Requests arrive on one thread and are processed by a worker pool. The ring buffer guarantees O(1) push/pop — critical when serving thousands of requests per second with bounded latency.

---

## 2. Associative Containers

`HashMap<K, V>` uses SipHash-1-3 (DoS-resistant) for O(1) average-case lookup. `BTreeMap<K, V>` uses B-trees for O(log n) operations with ordered keys — ideal for range queries.

```rust
use std::collections::HashMap;

// Entry API avoids redundant lookups
let mut counts: HashMap<char, usize> = HashMap::new();
for ch in "abracadabra".chars() {
    *counts.entry(ch).or_insert(0) += 1;
}

use std::collections::BTreeMap;

// BTreeMap: ordered keys for range queries
let mut features: BTreeMap<u64, f32> = BTreeMap::new();
features.insert(1000, 0.5);
features.insert(2000, 0.8);
features.insert(3000, 0.3);

// Range query — O(log n + k) where k is result count
let window: Vec<_> = features.range(1500..2500).collect();
println!("{:?}", window); // [(2000, 0.8)]
```

❌ **Antipattern:** Using `f64` as a `HashMap` key — `NaN != NaN` violates `Eq`, causing panics or incorrect lookups.

✅ **Wrap float keys or use a struct with total ordering:**

```rust
#[derive(PartialEq, Eq, Hash)]
struct FloatKey(u32); // store as bits

impl From<f32> for FloatKey {
    fn from(f: f32) -> Self { FloatKey(f.to_bits()) }
}
```

❌ **Antipattern:** Assuming `HashMap` iteration order is deterministic. It changes between runs and with hash seed.

✅ Use `BTreeMap` when order matters; sort `HashMap` entries after collecting for deterministic output.

⚠️ The **entry API** (`entry(key).or_insert(default)`) performs a single lookup, not two. Prefer it over `contains_key` + `insert`:

```rust
// ❌ Two lookups
if !map.contains_key(&k) { map.insert(k, v); }

// ✅ One lookup
map.entry(k).or_insert(v);
```

💡 Pre-allocate with `HashMap::with_capacity(n)` to reduce rehashing overhead during bulk insertion of large vocabularies.

**Caso real:** A tokenizer vocabulary maps subword strings to integer IDs using `HashMap<String, u32>`. The entry API counts token frequencies in a single pass over the corpus. When the vocabulary exceeds 1M entries, `with_capacity` prevents costly rehashing.

---

## 3. The Iterator Protocol

Rust's `Iterator` trait requires only `next()` returning `Option<Self::Item>`. All adapters (`map`, `filter`, `fold`) are default methods built atop this primitive. Chains **fuse** into a single loop with no intermediate allocations — O(n) with O(1) extra space.

```rust
// Adapters are lazy; nothing happens until a consumer is called
let total: i32 = (0..1_000_000)
    .map(|x| x * 2)          // transform
    .filter(|x| x % 3 == 0)  // predicate
    .take(100)               // bounded window
    .sum();                  // consumer: evaluates the chain

// collect gathers into a concrete collection
let squares: Vec<i32> = (1..=10).map(|x| x * x).collect();

// fold aggregates without intermediate vectors
let product = [1.0, 2.0, 3.0, 4.0]
    .iter()
    .fold(1.0, |a, b| a * b); // 24.0

// try_fold short-circuits on error
let sum: Result<i32, _> = ["1", "2", "x"]
    .iter()
    .map(|s| s.parse::<i32>())
    .try_fold(0, |acc, v| v.map(|x| acc + x));
// Err(ParseIntError)
```

❌ **Antipattern:** Collecting an infinite iterator (`0..`) into a `Vec` — hangs or exhausts memory.

✅ **Always bound infinite sources** with `take` or `take_while` before `collect`.

❌ **Antipattern:** Holding a mutable reference to a collection while iterating:

```rust
let mut v = vec![1, 2, 3];
for x in &v {
    // v.push(4); // ❌ cannot borrow v as mutable
}
```

✅ Use `retain`, `drain`, or restructure to avoid simultaneous aliasing and mutation:

```rust
v.retain(|x| *x > 1); // removes elements in-place
```

⚠️ **Fusion:** `map.filter.take.sum` compiles to a single loop. No `Vec` allocations happen between adapters.

💡 Implement `Iterator` for custom types to integrate with `for` loops and all adapters:

```rust
struct WindowIterator<'a> {
    data: &'a [f32],
    size: usize,
    pos: usize,
}

impl<'a> Iterator for WindowIterator<'a> {
    type Item = &'a [f32];
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.size > self.data.len() { return None; }
        let window = &self.data[self.pos..self.pos + self.size];
        self.pos += 1;
        Some(window)
    }
}
```

**Caso real:** A feature extraction pipeline processes a time-series of 10M points using iterator adapters. The chain `.windows(256).map(compute_features).filter(valid).take(10000)` never allocates more than a single 256-element window at a time — constant memory regardless of input size.

---

## 🎯 Key Takeaways

- `Vec` is the default — contiguous memory, O(1) index, cache-friendly
- `VecDeque` for FIFO queues; `HashMap` for key-value (O(1)); `BTreeMap` for ordered range queries
- Use the entry API for single-lookup insert-or-update patterns
- Iterator adapters are lazy and fuse into O(1) memory loops
- Custom iterators via `Iterator` trait integrate with all adapters and `for` loops
- Pre-allocate with `with_capacity` when sizes are known in advance

## References

- [The Rust Book - Collections](https://doc.rust-lang.org/book/ch08-00-common-collections.html)
- [The Rust Book - Iterators](https://doc.rust-lang.org/book/ch13-02-iterators.html)
- [Rayon Documentation](https://docs.rs/rayon/latest/rayon/)

## 📦 Código de compresión

```rust
use std::collections::HashMap;

fn frequency_analysis(data: &[u8]) -> HashMap<u8, usize> {
    data.iter()
        .fold(HashMap::new(), |mut map, &byte| {
            *map.entry(byte).or_insert(0) += 1;
            map
        })
}

fn compress_rle(data: &[u8]) -> Vec<u8> {
    if data.is_empty() { return Vec::new(); }
    data.iter().skip(1).fold(
        (Vec::new(), data[0], 1u8),
        |(mut result, current, count), &byte| {
            if byte == current && count < 255 {
                (result, current, count + 1)
            } else {
                result.push(current);
                result.push(count);
                (result, byte, 1)
            }
        }
    ).0
}

fn main() {
    let data = b"AAAAABBBBCCCCCDDDDD";
    let freqs = frequency_analysis(data);
    println!("Frequencies: {:?}", freqs);
    let compressed = compress_rle(data);
    println!("Original: {} -> {}", data.len(), compressed.len());
}
```
