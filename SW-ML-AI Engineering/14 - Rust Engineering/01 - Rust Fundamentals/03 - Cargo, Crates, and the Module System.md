# 🦀 3 - Cargo, Crates, and the Module System

## 🎯 Learning Objectives

- Create, build, test, and publish Rust projects using Cargo
- Distinguish between binary crates, library crates, and packages
- Structure code using modules to control visibility and encapsulation
- Manage multi-crate projects with workspaces and shared dependencies

## Introduction

Cargo is Rust's build system and package manager — handling compilation, dependency resolution, testing, and distribution. Unlike C++ where build systems vary (Make, CMake, Bazel), Cargo provides a unified, batteries-included experience. A model serving binary, a feature store library, and a data preprocessing CLI can all live in the same workspace, sharing dependencies with a single command.

The module system defines how code is organized and encapsulated. It determines visibility, name resolution, and decomposition into manageable units. This connects directly with [[01 - Ownership, Borrowing, and Lifetimes|memory-safe API design]] and [[02 - Types, Traits, and Generics|trait-driven abstractions]].

---

## 1. Cargo and Dependency Management

Reproducible builds matter for ML systems. A model trained with one version of a linear algebra crate and deployed with another may produce different embeddings due to numerical precision changes. Cargo's lock file ensures every build uses identical dependency versions.

```toml
[package]
name = "ml-pipeline"
version = "1.0.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
ndarray = "0.15"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[profile.release]
lto = true
codegen-units = 1
```

```bash
# Create a new project
cargo new my-app

# Build in dev mode (fast compile, debug assertions)
cargo build

# Optimized release build
cargo build --release

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Update dependencies (respects SemVer)
cargo update
```

❌ **Antipattern:** For applications, **not committing `Cargo.lock`** — without it, builds may pull in newer patch versions with behavioral changes.

✅ **Correct:** Commit `Cargo.lock` for binaries; do NOT commit it for libraries (downstream users resolve their own).

⚠️ **Feature unification:** Feature flags are unified across the dependency graph. If crate A enables `serde/derive` and crate B enables `serde/alloc`, the compiled serde includes both features. Audit with `cargo tree -e features`.

💡 Always specify the `edition` in `Cargo.toml`. The 2021 edition introduced disjoint capture in closures and consistent `panic!` macros.

**Caso real:** The Burn framework uses feature flags to select backends — `features = ["cuda"]` compiles CUDA kernels, `features = ["wgpu"]` compiles cross-platform GPU. Feature unification ensures all backends share the same core tensor types.

---

## 2. Crates and the Module System

A crate is the smallest compilation unit: the compiler starts at a root file (`main.rs` or `lib.rs`) and recursively resolves modules. Visibility is opt-in — every item is private by default, preventing accidental API surface expansion.

```rust
// src/lib.rs
pub mod config;
pub mod engine;
pub mod utils;

// Re-export for ergonomic public API
pub use config::Settings;
pub use engine::Processor;

// src/config.rs
pub struct Settings {
    pub timeout: u64,
    pub max_connections: usize,
}

// src/engine/mod.rs
pub mod parser;
pub mod renderer;

use crate::config::Settings;

pub struct Processor { settings: Settings }

impl Processor {
    pub fn new(settings: Settings) -> Self { Processor { settings } }
    pub fn process(&self, input: &str) -> String {
        let parsed = parser::parse(input);
        renderer::render(&parsed, self.settings.timeout)
    }
}

// src/engine/parser.rs
pub struct ParsedData { pub tokens: Vec<String> }
pub fn parse(input: &str) -> ParsedData {
    ParsedData { tokens: input.split_whitespace().map(String::from).collect() }
}

// src/utils.rs — not pub, so invisible outside this crate
fn internal_helper() {}
pub fn sanitize(input: &str) -> String { input.trim().to_lowercase() }
```

**Visibility levels:**
- `pub` — visible everywhere
- `pub(crate)` — visible within current crate
- `pub(super)` — visible to parent module
- (private) — visible only in current module

❌ **Antipattern:** Deep nesting with `mod.rs` files. Mixing `src/engine.rs` with `src/engine/mod.rs` creates confusion.

✅ **Preferred:** Use `src/engine.rs` style (Rust 2018+) for flat hierarchies:

```bash
src/
├── lib.rs
├── config.rs
├── engine.rs         # instead of engine/mod.rs
├── engine/
│   ├── parser.rs
│   └── renderer.rs
└── utils.rs
```

⚠️ A `pub` item inside a non-public module is still invisible outside the parent. Every module in the path must be `pub`.

💡 Use `pub use` to create a clean facade. Users should write `use my_crate::Settings;` not `use my_crate::config::Settings;`.

**Caso real:** Polars separates `crate::series`, `crate::frame`, and `crate::lazy` into distinct modules. The lazy evaluation API is fully public, while internal column chunking is `pub(crate)` — preventing downstream dependence on internal buffer layouts.

---

## 3. Workspaces and Scaling

As projects grow, monolithic crates become slow to compile. Workspaces allow multiple related packages to share a `Cargo.lock` and target directory, enabling incremental compilation and parallel builds.

```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "crates/core",
    "crates/train",
    "crates/serve",
    "crates/evaluate",
]

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = "1"
ndarray = "0.15"
```

```toml
# crates/train/Cargo.toml
[package]
name = "train"
version = "0.1.0"

[dependencies]
serde = { workspace = true }   # references root workspace dep
core = { path = "../core" }    # path dependency for local crate
```

```rust
// crates/train/src/main.rs
use core::ModelConfig;

fn main() {
    let config = ModelConfig::default();
    println!("Training with {:?}", config);
}
```

```bash
# Build entire workspace
cargo build --workspace

# Test all members
cargo test --workspace

# Build one member
cargo build -p train

# Check without producing binaries
cargo check --workspace
```

❌ **Antipattern:** **Cyclic dependencies** between workspace crates are forbidden. If `train` depends on `core` and `core` depends on `train`, Cargo cannot determine build order.

✅ **Fix:** Extract shared types into a third `core` crate that both depend on:

```
my-platform/
├── crates/
│   ├── core/     # shared types, traits
│   ├── train/    # depends on core
│   └── serve/    # depends on core
```

⚠️ Workspace feature unification can leak flags. If `train` enables `serde/alloc`, the unified build includes `alloc` even for `serve`.

💡 Keep workspace members loosely coupled through a small `core` or `domain` crate. Business logic lives in specialized crates depending on `core`, not on each other.

**Caso real:** The Hugging Face Tokenizers repo uses a workspace with the Rust core library and Python bindings crate. The Python bindings depend on `tokenizers-core`, sharing the same lockfile. A change to core triggers recompilation of bindings but not of unrelated workspace members.

---

## 🎯 Key Takeaways

- Cargo provides deterministic, reproducible builds via `Cargo.lock` and semantic versioning
- Modules control visibility with opt-in `pub` — private by default prevents API leaks
- Workspaces enable multi-crate projects with shared dependencies and incremental compilation
- Feature flags are unified across the dependency graph; audit with `cargo tree`
- Avoid cyclic dependencies; extract shared types into a `core` crate

## References

- [The Cargo Book](https://doc.rust-lang.org/cargo/)
- [The Rust Reference - Modules](https://doc.rust-lang.org/reference/items/modules.html)
- [[01 - Ownership, Borrowing, and Lifetimes]]
- [[02 - Types, Traits, and Generics]]

## 📦 Código de compresión

```rust
// Workspace structure:
// crates/compress/src/lib.rs  — library crate
// crates/cli/src/main.rs      — binary crate

// lib.rs
pub trait Encoder {
    fn encode(&self, data: &[u8]) -> Vec<u8>;
    fn name(&self) -> &'static str;
}

pub fn encode_with<E: Encoder>(data: &[u8], encoder: E) -> Vec<u8> {
    encoder.encode(data)
}

pub fn default_encoder() -> impl Encoder {
    RleEncoder
}

struct RleEncoder;

impl Encoder for RleEncoder {
    fn encode(&self, data: &[u8]) -> Vec<u8> {
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

// cli/src/main.rs
// use compress::{default_encoder, encode_with};
// use std::fs;
//
// fn main() {
//     let data = fs::read("input.bin").expect("File not found");
//     let encoder = default_encoder();
//     let compressed = encode_with(&data, encoder);
//     println!("{} -> {} bytes", data.len(), compressed.len());
// }
```
