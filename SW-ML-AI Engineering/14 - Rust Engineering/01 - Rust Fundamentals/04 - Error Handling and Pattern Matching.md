# 🦀 4 - Error Handling and Pattern Matching

## 🎯 Learning Objectives

- Distinguish recoverable errors (`Result<T, E>`) from optional values (`Option<T>`)
- Propagate failures ergonomically with the `?` operator while preserving error context
- Leverage exhaustive pattern matching to enforce total functions over algebraic data types
- Design structured error hierarchies with `thiserror` and `anyhow`

## Introduction

Rust encodes failure directly into the type system. When a tokenizer cannot decode a byte sequence, or a feature store returns an empty slice, the resulting `Result` or `Option` forces the caller to acknowledge the possibility of failure before proceeding. This transforms runtime surprises into compile-time obligations.

Pattern matching is the complement. Rust's `match` is an expression that demands totality: every variant must have a corresponding arm. Adding a new enum variant triggers compiler-guided updates across every site that inspects the value. The combination creates self-documenting, refactor-resistant codebases. This connects with [[01 - Ownership, Borrowing, and Lifetimes|safe resource management]] and [[02 - Types, Traits, and Generics|type-safe abstractions]].

---

## 1. Algebraic Error Types

`Result<T, E>` and `Option<T>` are **sum types** (tagged unions). They force the programmer to handle both variants — eliminating null pointer exceptions and unhandled errors at the language level.

```rust
// Result and Option are enums defined in std:
enum Result<T, E> { Ok(T), Err(E) }
enum Option<T> { Some(T), None }

fn read_model_config(path: &str) -> Result<String, std::io::Error> {
    let text = std::fs::read_to_string(path)?; // Early return on Err
    Ok(text)
}

// map transforms success values without unwrapping
let maybe_batch: Option<usize> = Some(32);
let doubled = maybe_batch.map(|b| b * 2); // Some(64)

// and_then chains fallible steps
let result = Some(10)
    .and_then(|x| Some(x * 2))
    .and_then(|x| Some(x + 1)); // Some(21)
```

❌ **Antipattern:** Using `unwrap()` in data preprocessing — crashes the entire job on the first malformed sample.

✅ **Preferred:** Use `?` or `match` and log the offending record:

```rust
fn process_sample(line: &str) -> Option<f64> {
    line.parse::<f64>().ok() // Converts Result to Option
}

// Or with context
fn load_batch(path: &str) -> Result<Vec<f64>, String> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?
        .lines()
        .map(|l| l.parse::<f64>().map_err(|e| format!("Bad line: {}", e)))
        .collect()
}
```

❌ **Antipattern:** Silently dropping a `Result`:
```rust
// This silently swallows I/O errors!
let _ = file.write_all(&buf);
```

✅ **Preferred:** Handle or propagate explicitly:
```rust
file.write_all(&buf)?; // Propagate
// or
file.write_all(&buf).expect("checkpoint write failed"); // Crash with message
```

⚠️ **Error type conversion:** `?` uses `From<Error>` to convert between error types. If function A returns `io::Error` and function B returns `AppError` with `impl From<io::Error> for AppError`, the `?` operator converts automatically.

💡 Use `Result::map_err` to attach domain-specific context before propagating:

```rust
fn load_shard(path: &str) -> Result<Vec<f32>, ShardError> {
    let data = std::fs::read(path)
        .map_err(|e| ShardError::Io { path: path.into(), source: e })?;
    // ...
}
```

**Caso real:** A batch inference pipeline processes 10,000 shards. Using `?` on each file read means a corrupt shard fails exactly that shard, logging the path. The orchestrator catches the `Result::Err` and retries or skips the shard — rather than crashing the entire job or silently producing partial results.

---

## 2. Error Propagation and `?`

The `?` operator is syntactic sugar for a `match` that returns `Err(From::from(e))` on failure and continues with the unwrapped value on success. It preserves **signature transparency**: the error domain is always visible in the function signature.

```rust
use std::io;

#[derive(Debug)]
enum PipelineError {
    Io(io::Error),
    Parse(String),
}

impl From<io::Error> for PipelineError {
    fn from(e: io::Error) -> Self { PipelineError::Io(e) }
}

fn process_shard(path: &str) -> Result<Vec<f64>, PipelineError> {
    let raw = std::fs::read_to_string(path)?; // io::Error -> PipelineError via From
    let values: Vec<f64> = raw
        .lines()
        .map(|line| line.parse().map_err(|e| PipelineError::Parse(format!("{}", e))))
        .collect::<Result<_, _>>()?; // Collect stops on first Err
    Ok(values)
}

// Using thiserror for automatic Display and From
use thiserror::Error;

#[derive(Error, Debug)]
enum ModelError {
    #[error("IO failed: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid hyperparameter: {name} = {value}")]
    Hyperparameter { name: String, value: String },
}

// anyhow for application-level error handling
use anyhow::{Context, Result};

fn load_model(path: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read from {}", path))?;
    Ok(bytes)
}
```

❌ **Antipattern:** Using `anyhow` in a public library API — forces downstream crates to depend on `anyhow` and prevents programmatic error matching.

✅ **Correct:** Use `thiserror` for libraries (structured enums), `anyhow` for binaries and application code.

❌ **Antipattern:** Using `?` inside a closure passed to `Iterator::map` — closures are separate control-flow boundaries. `?` in a closure returns from the closure, not the outer function.

✅ **Fix:** Use `try_for_each` or collect into `Result<Vec<_>, _>`:

```rust
let results: Result<Vec<_>, _> = items.iter().map(|i| fallible_op(i)).collect();
```

⚠️ `?` in `main()` requires a return type that implements `Process` — typically `Result<(), Box<dyn Error>>` or use `anyhow::Result<()>`.

**Caso real:** An ONNX Runtime binding uses a structured error enum with `thiserror`. Consumers match on `SessionError::ShapeMismatch` to trigger a model reload, `SessionError::Oom` to free GPU memory, and `SessionError::Io` for retry logic. Programmatic error handling is essential for production ML serving.

---

## 3. Exhaustive Pattern Matching

Pattern matching decomposes sum and product types. Rust's compiler enforces **exhaustiveness** — every variant must be covered, making `match` a tool for writing **total functions** defined for all inputs.

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(String), // carries data
}

fn value(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
    }
}

// if let: concise when only one variant matters
let some_value: Option<i32> = Some(3);
if let Some(3) = some_value {
    println!("Three");
}

// while let: consume iterator until exhaustion
let mut stack = vec![1, 2, 3];
while let Some(value) = stack.pop() {
    println!("{}", value);
}

// Guards add runtime predicates
match age {
    0 => println!("Newborn"),
    n if n < 13 => println!("Child"),
    20..=65 => println!("Adult"),
    _ => println!("Senior"),
}
```

❌ **Antipattern:** Adding a variant to a public enum breaks downstream `match` arms:

```rust
// In library crate:
#[non_exhaustive]
pub enum Status { Active, Inactive }

// Downstream crate: must include wildcard because enum is non-exhaustive
match status {
    Status::Active => { /* ... */ }
    _ => { /* future variants handled here */ }
}
```

❌ **Antipattern:** Using `_` wildcard too early — hides new variants from compiler checks.

✅ **Preferred:** Exhaustively list all variants in internal code. Use `#[non_exhaustive]` and `_` only for public library enums that may gain variants.

💡 **`@` bindings** capture a value while pattern matching:

```rust
match value {
    n @ 1..=10 => println!("Small: {}", n),
    n @ 11..=100 => println!("Medium: {}", n),
    _ => println!("Large"),
}
```

**Caso real:** A decision tree inference engine uses `match` on the node type. When a new node variant (e.g., `LeafWithUncertainty`) is added, the compiler points to every `match` in the codebase that must be updated. This prevents silent logic errors where a new node type is not evaluated.

---

## 🎯 Key Takeaways

- `Result<T, E>` and `Option<T>` make failure modes explicit in type signatures
- The `?` operator propagates errors with automatic type conversion via `From`
- Exhaustive `match` guarantees total functions — the compiler catches missing cases
- Use `thiserror` for library error enums, `anyhow` for application error handling
- `#[non_exhaustive]` allows adding enum variants without breaking downstream code

## References

- [The Rust Book - Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [The Rust Book - Pattern Matching](https://doc.rust-lang.org/book/ch18-00-patterns.html)
- [thiserror crate](https://docs.rs/thiserror/latest/thiserror/)
- [anyhow crate](https://docs.rs/anyhow/latest/anyhow/)

## 📦 Código de compresión

```rust
use std::fs::File;
use std::io::{self, Read, Write};
use thiserror::Error;

#[derive(Error, Debug)]
enum CompressError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Empty input")]
    EmptyInput,
    #[error("Output larger than input: {inp} -> {out}")]
    Ineffective { inp: usize, out: usize },
}

fn compress(path: &str) -> Result<usize, CompressError> {
    let mut input = Vec::new();
    File::open(path)?.read_to_end(&mut input)?;
    if input.is_empty() { return Err(CompressError::EmptyInput); }

    let mut out = Vec::new();
    let mut current = input[0];
    let mut count = 1u8;
    for &byte in &input[1..] {
        if byte == current && count < 255 { count += 1; }
        else { out.push(current); out.push(count); current = byte; count = 1; }
    }
    out.push(current); out.push(count);

    if out.len() > input.len() {
        return Err(CompressError::Ineffective { inp: input.len(), out: out.len() });
    }
    let mut file = File::create("output.bin")?;
    file.write_all(&out)?;
    Ok(out.len())
}

fn main() {
    match compress("input.txt") {
        Ok(size) => println!("Compressed to {} bytes", size),
        Err(e) => { eprintln!("Failed: {}", e); std::process::exit(1); }
    }
}
```
