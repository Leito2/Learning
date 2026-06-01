# 🦀 6 - Structs, Enums, and Advanced Patterns

## 🎯 Learning Objectives

- Model domain concepts using product types (structs) and sum types (enums) with associated data
- Implement custom methods and constructors that respect Rust's ownership and borrowing rules
- Use `Deref` and `DerefMut` to build smart pointers with ergonomic coercion
- Leverage `Drop`, `Clone`, and `Copy` to manage resources and value semantics precisely

## Introduction

In machine learning, the boundary between research code and production systems is often marked by data modeling rigor. A configuration that silently accepts invalid hyperparameters, a state machine allowing illegal transitions, a tensor handle leaking GPU memory — these are not bugs of algorithms but bugs of representation. Rust's struct and enum system provides the tools to make illegal states unrepresentable.

Structs are **product types**: they combine multiple values into a named bundle — like a feature vector combining scalars. Enums are **sum types**: a value that is one of several alternatives, each carrying distinct data — ideal for model architectures, tokenizer outputs, or protocol states. Combined with traits like `Deref`, `Drop`, and `Clone`, these types integrate with Rust's ownership discipline to manage resources without garbage collection. This connects deeply with [[01 - Ownership, Borrowing, and Lifetimes|ownership]] and [[02 - Types, Traits, and Generics|generics]].

---

## 1. Structs as Product Types

A product type `A × B` contains one value from `A` and one from `B`. Named structs (`struct S { a: A, b: B }`), tuple structs (`struct T(A, B)`), and unit structs (`struct U;`) are Rust's realizations.

```rust
#[derive(Debug, Clone)]
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // Associated function without self = constructor by convention
    fn new(width: f64, height: f64) -> Self {
        Rectangle { width, height }
    }
    // &self borrows immutably — multiple read-only queries
    fn area(&self) -> f64 { self.width * self.height }
    // &mut self enforces exclusive access during mutation
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }
}

// Tuple struct: distinct types for semantically different values
struct Meters(f64);
struct Kilometers(f64);

impl Meters {
    fn to_kilometers(self) -> Kilometers {
        Kilometers(self.0 / 1000.0)
    }
}

fn main() {
    let mut rect = Rectangle::new(10.0, 20.0);
    println!("Area: {}", rect.area());
    rect.scale(2.0);
    println!("Scaled area: {}", rect.area());

    let dist = Meters(1500.0);
    let km = dist.to_kilometers();
    println!("{} km", km.0);
}
```

❌ **Antipattern:** Tuple struct fields accessed by index (`self.0`) are less readable than named fields. Use tuples only for thin wrappers where meaning is obvious.

✅ **Newtype pattern:** A single-field tuple struct creates a distinct type with zero runtime overhead:

```rust
struct UserId(u64);
struct ModelId(u64);
// UserId(1) and ModelId(1) are NOT interchangeable
```

❌ **Antipattern:** Forgetting to derive `Clone` on a struct containing a `Vec` prevents cheap duplication.

⚠️ Implement `Default` for structs with many optional fields:

```rust
#[derive(Default)]
struct TrainingConfig {
    lr: f64,           // default: 0.0
    batch_size: usize, // default: 0
    optimizer: String, // default: ""
}

// Use struct update syntax for partial configs
let base = TrainingConfig::default();
let fine_tune = TrainingConfig { lr: 0.001, ..base };
```

**Caso real:** Hugging Face configs are deeply nested product types. `BertConfig` contains `AttentionConfig`, `LayerConfig`, etc. Each sub-struct is independently testable and reusable across model architectures. A missing field is a compile error, not a runtime `KeyError`.

---

## 2. Enums as Sum Types

Where structs represent conjunction ("this AND that"), enums represent disjunction ("this OR that"). Each variant is a distinct injection into the sum, and the discriminant ensures pattern matching is safe and exhaustive.

```rust
enum ModelFormat {
    Onnx { opset: u32 },
    Safetensors,
    Pickle,
}

impl ModelFormat {
    fn extension(&self) -> &'static str {
        match self {
            ModelFormat::Onnx { .. } => ".onnx",
            ModelFormat::Safetensors => ".safetensors",
            ModelFormat::Pickle => ".pkl",
        }
    }
}

fn maybe_sqrt(x: f64) -> Option<f64> {
    if x >= 0.0 { Some(x.sqrt()) } else { None }
}

fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 { Err("Division by zero".into()) } else { Ok(a / b) }
}

fn main() {
    let fmt = ModelFormat::Onnx { opset: 21 };
    println!("Extension: {}", fmt.extension());
}
```

❌ **Antipattern:** Recursive enums (e.g., expression trees) without indirection cause infinite size:

```rust
// ❌ Compile error: recursive type has infinite size
// enum JsonValue {
//     Null,
//     Object(HashMap<String, JsonValue>), // recursive!
// }

// ✅ Fix: Box indirection
enum JsonValue {
    Null,
    Object(HashMap<String, Box<JsonValue>>),
}
```

❌ **Antipattern:** C-style enums without data cannot carry context:

```rust
// ❌ Limited — need parallel arrays for metadata
enum Status { Active, Inactive }

// ✅ Carry data directly
enum Status {
    Active { since: u64, session_id: String },
    Inactive { reason: String },
}
```

⚠️ **`#[non_exhaustive]`** on public library enums allows adding variants without breaking downstream:

```rust
#[non_exhaustive]
pub enum Event { Click, KeyPress, /* future variants OK */ }
// Downstream must use `_` wildcard
```

💡 `Option<T>` and `Result<T, E>` are the most common enums. Use them instead of sentinel values like `-1` or empty strings.

**Caso real:** A tokenizer outputs `Token::Word(String)` for dictionary hits and `Token::Subword(u32)` for byte-level fallback. The compiler ensures every code path handles both variants. Adding a new `Token::Special` variant triggers compiler errors at every `match` site — guided refactoring.

---

## 3. Smart Pointers and Deref Coercion

`Deref` defines coercion from a smart pointer to a regular reference. `MyBox<T>` can be used where `&T` is expected — transparent access without manual unwrapping.

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> { MyBox(x) }
}

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> std::ops::DerefMut for MyBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

fn greet(name: &str) { println!("Hello, {name}!"); }

fn main() {
    let m = MyBox::new(String::from("Rust"));
    // &MyBox<String> coerces to &String, then &str via Deref chain
    greet(&m);
}
```

❌ **Antipattern:** Implementing `Deref` solely to emulate inheritance — `Deref` should mean "is-a transparent wrapper," not "has-a relationship."

✅ Use `AsRef<str>` or `AsRef<Path>` for polymorphism without implicit coercion:

```rust
fn load_config(path: impl AsRef<std::path::Path>) { /* ... */ }
```

❌ **Antipattern:** `DerefMut` on `Rc` or `Arc` without interior mutability — shared references cannot yield `&mut T`.

✅ Use `RefCell<T>` (single-threaded) or `Mutex<T>` (multi-threaded) for interior mutability:

```rust
use std::cell::RefCell;
use std::rc::Rc;

let shared: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(vec![1, 2, 3]));
shared.borrow_mut().push(4); // mutable access through shared ownership
```

⚠️ Deref coercion happens automatically for function arguments. The compiler inserts as many `*` dereferences as needed.

💡 `Box<T>`, `Rc<T>`, `Arc<T>`, and `RefCell<T>` are the standard smart pointers. `Box` for heap allocation, `Rc`/`Arc` for shared ownership, `RefCell` for interior mutability.

**Caso real:** A shared embedding table uses `Arc<EmbeddingMatrix>` where `Deref` provides transparent read access to the matrix. Multiple model heads reference the same weights without copying. The reference count ensures the matrix is deallocated exactly when all heads are dropped.

---

## 4. Resource Management: Drop, Clone, Copy

**RAII (Resource Acquisition Is Initialization):** resources are acquired on construction and released on destruction. `Drop` provides custom cleanup. `Copy` enables implicit bitwise duplication for stack-only types. `Clone` enables explicit deep copying.

```rust
struct GpuBuffer {
    ptr: *mut f32,
    size: usize,
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        println!("Releasing GPU buffer of size {}", self.size);
        // unsafe { cudaFree(self.ptr); }
    }
}

// Copy for small, trivially duplicable types
#[derive(Clone, Copy)]
struct HyperParams {
    lr: f64,
    batch_size: usize,
}

// Clone for heap-owning types — copies are explicit
#[derive(Clone)]
struct Weights {
    data: Vec<f32>,
}

fn main() {
    let a = HyperParams { lr: 0.01, batch_size: 32 };
    let b = a; // Copy: both a and b remain valid

    let w1 = Weights { data: vec![1.0, 2.0] };
    let w2 = w1.clone(); // Clone: w1 and w2 own separate buffers
    // let w3 = w1; // ❌ move — w1 is no longer valid after clone
}
```

❌ **Antipattern:** Deriving `Copy` on a struct containing `Vec` or `String` won't compile. Manually implementing `Copy` on a pointer-backed type causes double-free undefined behavior.

❌ **Antipattern:** `Drop` is NOT called when a value is leaked via `std::mem::forget` or reference cycles in `Rc`. Use `Weak` references to break cycles:

```rust
use std::rc::{Rc, Weak};

struct Node {
    children: Vec<Rc<Node>>,
    parent: Weak<Node>, // prevents reference cycle
}
```

⚠️ `Copy` is a subtrait of `Clone`. Every `Copy` type must also implement `Clone`.

💡 Derive both `Clone` and `Copy` for small, immutable configuration structs to reduce friction in parallel code where values are moved into closures.

**Caso real:** A GPU memory allocator uses `Drop` to automatically free CUDA buffers when they go out of scope — even if the training loop panics mid-epoch. This prevents GPU memory leaks that would crash subsequent runs. The `HyperParams` struct implements `Copy` so it can be implicitly duplicated across parallel training workers.

---

## 🎯 Key Takeaways

- Structs are product types (AND); enums are sum types (OR) — the compiler checks both exhaustively
- Tuple structs (newtype pattern) create distinct types with zero overhead
- `Deref` enables transparent access for smart pointers; use `AsRef` for explicit polymorphism
- `Drop` guarantees resource cleanup; `Copy` for bitwise duplication; `Clone` for explicit deep copies
- Use `#[non_exhaustive]` on public enums to allow future variants without breaking changes
- Never implement `Copy` for heap-owning types; never implement `Deref` for inheritance-style APIs

## References

- [The Rust Book - Structs](https://doc.rust-lang.org/book/ch05-00-structs.html)
- [The Rust Book - Enums](https://doc.rust-lang.org/book/ch06-00-enums.html)
- [The Rust Book - Smart Pointers](https://doc.rust-lang.org/book/ch15-00-smart-pointers.html)
- [[01 - Ownership, Borrowing, and Lifetimes]]
- [[02 - Types, Traits, and Generics]]

## 📦 Código de compresión

```rust
use std::ops::Deref;

struct CompressedData {
    original_size: usize,
    data: Vec<u8>,
}

impl CompressedData {
    fn new(data: Vec<u8>, original_size: usize) -> Self {
        CompressedData { original_size, data }
    }
    fn compression_ratio(&self) -> f64 {
        self.data.len() as f64 / self.original_size as f64
    }
}

impl Deref for CompressedData {
    type Target = [u8];
    fn deref(&self) -> &[u8] { &self.data }
}

impl Drop for CompressedData {
    fn drop(&mut self) {
        println!("Dropped (ratio: {:.2})", self.compression_ratio());
    }
}

#[derive(Clone, Copy)]
enum Algorithm { Rle, None }

fn compress(data: &[u8], algo: Algorithm) -> CompressedData {
    match algo {
        Algorithm::Rle => {
            if data.is_empty() { return CompressedData::new(Vec::new(), 0); }
            let mut out = Vec::new();
            let mut current = data[0];
            let mut count = 1u8;
            for &byte in &data[1..] {
                if byte == current && count < 255 { count += 1; }
                else { out.push(current); out.push(count); current = byte; count = 1; }
            }
            out.push(current); out.push(count);
            CompressedData::new(out, data.len())
        }
        Algorithm::None => CompressedData::new(data.to_vec(), data.len()),
    }
}

fn main() {
    let data = b"AAAAABBBBCCCCCDDDDD";
    for algo in [Algorithm::Rle, Algorithm::None] {
        let c = compress(data, algo);
        println!("Algo: {} -> bytes, ratio: {:.2}", c.len(), c.compression_ratio());
    } // Drop called here for each CompressedData
}
```
