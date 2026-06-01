# 🦀 2 - Types, Traits, and Generics

## 🎯 Learning Objectives

- Explain how Rust's static type system prevents invalid operations at compile time
- Design reusable abstractions using traits and associated types
- Write generic functions and structs that maintain zero-cost performance
- Choose between monomorphization and trait objects based on API requirements

## Introduction

Rust's type system is one of the most expressive in modern programming, drawing inspiration from ML-family languages while maintaining systems-level performance. The type system prevents invalid operations at compile time — eliminating null pointer dereferences, type confusion, and shape mismatches before the program executes. A confusion matrix that accidentally mixes `u32` counts with `f32` probabilities is a compile error, not a silent wrong result.

Traits are the backbone of Rust's abstraction mechanism. Unlike object-oriented inheritance, traits define shared behavior through interfaces that types can implement. Combined with generics, traits allow you to write code that works over any type satisfying a set of constraints, while generating optimal machine code for each concrete type. This connects deeply with [[01 - Ownership, Borrowing, and Lifetimes|ownership-aware APIs]] and [[03 - Cargo, Crates, and the Module System|project organization]].

---

## 1. The Rust Type System

Rust is static, strong, and nominal with global type inference. Every value has a known type at compile time, and implicit conversions are limited to non-lossy coercions.

```rust
fn main() {
    // Scalars: fixed-size, implement Copy
    let x: i32 = 42;
    let y = x; // bitwise copy

    // Type inference: f64 is the default float
    let pi = 3.14159;

    // Tuples: fixed-size, heterogeneous
    let point: (i32, f64, bool) = (10, 0.5, true);
    let (a, b, c) = point; // destructuring

    // Arrays: compile-time known length
    let counts: [u32; 4] = [1, 2, 3, 4];

    // Slices: fat pointers (data_ptr, length)
    let slice = &counts[1..3];
    println!("{:?}", slice); // [2, 3]

    // Type annotation resolves ambiguity for parse()
    let guess: u32 = "42".parse().expect("Not a number!");
}
```

❌ **Antipattern:** Integer overflow behavior differs between debug (panics) and release (two's complement wrapping).

✅ **Preferred:** Use `checked_add`, `saturating_add`, or `wrapping_add` explicitly when overflow semantics matter:

```rust
let a: u32 = u32::MAX;
let b = a.checked_add(1); // None, not a panic
let c = a.wrapping_add(1); // 0
```

⚠️ Type inference can fail ambiguously when a method is implemented for many types. "Type annotations needed" signals genuine ambiguity — add an explicit annotation.

💡 Prefer explicit types for public API boundaries; let inference handle local variables.

**Caso real:** Burn framework uses const generics for compile-time tensor shape checking. A `Tensor<B, 3, 64, 64>` for a 3×64×64 image prevents shape mismatches — the compiler catches a `(3, 64, 64)` vs `(64, 64, 3)` confusion before training starts.

---

## 2. Traits

Traits are Rust's mechanism for **ad-hoc polymorphism** — closely related to Haskell's typeclasses. A trait defines a contract of methods and associated types that concrete types must provide.

```rust
trait Drawable {
    fn draw(&self);
    fn bounds(&self) -> (f64, f64, f64, f64);
}

struct Circle { x: f64, y: f64, radius: f64 }

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle at ({}, {})", self.x, self.y);
    }
    fn bounds(&self) -> (f64, f64, f64, f64) {
        (self.x - self.radius, self.y - self.radius,
         self.x + self.radius, self.y + self.radius)
    }
}

// Trait bound constrains generic types
fn draw_all<T: Drawable>(shapes: &[T]) {
    for shape in shapes { shape.draw(); }
}

// Multiple bounds with +
fn process<T: Drawable + Clone + Send>(item: T) {}

// Associated types: one output type per implementation
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

fn main() {
    let circle = Circle { x: 0.0, y: 0.0, radius: 5.0 };
    draw_all(&[circle]);
}
```

❌ **Antipattern:** The **orphan rule** — you cannot implement a foreign trait for a foreign type. `impl Display for Vec<MyType>` in a utility crate fails.

✅ **Preferred:** Use the newtype pattern to wrap foreign types:

```rust
struct MyVec(Vec<MyType>);
impl Display for MyVec { /* ... */ }
```

❌ **Antipattern:** Trait objects (`dyn Trait`) require **object safety**. Traits with generic methods or `Self: Sized` bounds can't be trait objects.

⚠️ **Generics vs trait objects:**
- **Generics:** monomorphized, zero-cost, larger binary
- **`dyn Trait`:** vtable dispatch, small overhead, flexible

💡 Use associated types when there should be exactly one mapping from impl type to output type (e.g., `Iterator::Item`). Use generic parameters when multiple implementations for the same type are desired.

**Caso real:** Burn's `Optimizer` trait lets you swap SGD for Adam without changing the training loop. The generic `train<T: Optimizer>` function monomorphizes for each optimizer — zero runtime overhead, full type safety.

---

## 3. Generics and Monomorphization

Generics implement **parametric polymorphism** through **monomorphization**: the compiler generates distinct code for every concrete type instantiation. This enables zero-cost abstractions — each instantiation is optimized independently.

```rust
// Generic function: works for any PartialOrd type
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest { largest = item; }
    }
    largest
}

// Generic struct
struct Point<T> {
    x: T,
    y: T,
}

// Multiple type parameters
struct Pair<T, U> { first: T, second: U }

// Const generics: parameterized by values at compile time
struct Matrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T: Default + Copy, const R: usize, const C: usize> Matrix<T, R, C> {
    fn new() -> Self {
        Matrix { data: [[T::default(); C]; R] }
    }
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row)?.get(col)
    }
}

fn main() {
    let m: Matrix<f64, 3, 3> = Matrix::new();
    println!("{:?}", m.get(1, 1));
}
```

❌ **Antipattern:** Excessive monomorphization bloats binary size. A generic function instantiated for dozens of types produces duplicate code.

✅ **Preferred:** Use trait objects at API boundaries where the concrete type is rarely varied:

```rust
// Instead of: fn process<T: Drawable>(t: T)
// Use when you need heterogeneous collections:
fn process_objects(items: Vec<Box<dyn Drawable>>) { /* ... */ }
```

❌ **Antipattern:** Complex trait bounds become unreadable:

```rust
// Hard to read
fn train<T: Optimizer + Model + Loss + Send + Sync>(...) { }

// Better: where clause
fn train<T>(...) where
    T: Optimizer + Model + Loss + Send + Sync
{ }
```

💡 Start with concrete types and extract generics only after you have at least two use cases. Premature abstraction increases compile times and complexity.

**Caso real:** A quantized LLM uses `Matrix<i8, 4096, 4096>` for INT8 weights. The const generic parameters ensure at compile time that matrix multiplication dimensions match — dimension mismatches are caught before inference starts.

---

## 🎯 Key Takeaways

- Rust's type system is static, strong, and prevents null pointer errors via `Option<T>`
- Traits define shared behavior; generics enable zero-cost, type-safe polymorphism
- Monomorphization generates specialized code per type — zero runtime overhead, larger binaries
- Associated types fit "one output per impl" patterns; generic params fit "many impls per type"
- Use `where` clauses for readable bounds; prefer concrete types before abstracting

## References

- [The Rust Book - Traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Rust By Example - Generics](https://doc.rust-lang.org/rust-by-example/generics.html)
- [[01 - Ownership, Borrowing, and Lifetimes]]
- [[03 - Cargo, Crates, and the Module System]]

## 📦 Código de compresión

```rust
use std::fs;

trait Compressor {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
    fn name(&self) -> &'static str;
}

struct RleCompressor;

impl Compressor for RleCompressor {
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        let mut result = Vec::new();
        if data.is_empty() { return result; }
        let mut current = data[0];
        let mut count = 1u8;
        for &byte in &data[1..] {
            if byte == current && count < 255 { count += 1; }
            else { result.push(current); result.push(count); current = byte; count = 1; }
        }
        result.push(current); result.push(count);
        result
    }
    fn name(&self) -> &'static str { "RLE" }
}

struct IdentityCompressor;

impl Compressor for IdentityCompressor {
    fn compress(&self, data: &[u8]) -> Vec<u8> { data.to_vec() }
    fn name(&self) -> &'static str { "Identity" }
}

fn compress_file<C: Compressor>(path: &str, c: C) -> std::io::Result<Vec<u8>> {
    let data = fs::read(path)?;
    let compressed = c.compress(&data);
    println!("{}: {} -> {} bytes", c.name(), data.len(), compressed.len());
    Ok(compressed)
}

fn main() -> std::io::Result<()> {
    compress_file("Cargo.toml", RleCompressor)?;
    compress_file("Cargo.toml", IdentityCompressor)?;
    Ok(())
}
```
