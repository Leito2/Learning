# 🐧 Rust in the Linux Kernel

## Introduction

The Rust‑for‑Linux project aims to bring Rust's memory safety and expressiveness to kernel development, traditionally dominated by C. By introducing Rust as a second language for kernel modules, the project seeks to eliminate entire classes of bugs: buffer overflows, use‑after‑free, data races, and null pointer dereferences. Rust's ownership system, when combined with carefully designed kernel abstractions, can enforce safety at compile time without runtime overhead.

The effort is not a complete rewrite of the kernel in Rust, but rather a gradual introduction of safe abstractions for kernel primitives (e.g., file operations, ioctls, memory‑mapped I/O). These abstractions allow developers to write kernel modules that are as safe as user‑space Rust code, while still interopping with existing C code. Major players like Google and Huawei are investing in the project, with Android exploring Rust for new kernel drivers.

This deep dive examines the Rust‑for‑Linux project status, kernel abstractions, safety guarantees, and practical examples. We’ll also compare Rust with C for kernel development and look at real‑world adoption.

## 1. Rust‑for‑Linux Project Status

The Rust support for Linux was merged into the mainline kernel in version 6.1 (December 2022), albeit as an experimental feature. The project provides:

- **Rust compiler support**: A custom Rust compiler fork (`rustc`) with kernel‑specific modifications.
- **Kernel‑aware Rust toolchain**: Includes `bindgen` for C bindings and `libclang`.
- **Base abstractions**: `kernel::prelude`, error handling, `pr_info!` macros.
- **Subsystems**: File system, network, device driver, and memory‑management modules.

The project follows a phased approach: first introducing safe wrappers for basic kernel types, then gradually expanding to more complex subsystems.

**Real case:** Android's kernel team is developing a new DRM (Direct Rendering Manager) driver in Rust for future Pixel devices, aiming to reduce vulnerabilities.

⚠️ **Warning:** Rust in the kernel is still experimental; APIs may change, and not all kernel features are wrapped yet. Use `allow(dead_code)` and `allow(unused)` to suppress warnings during development.

💡 **Tip:** Start with simple kernel modules (e.g., a character device) to learn the abstractions before tackling complex drivers.

## 2. Kernel Primitives and Abstractions

Rust‑for‑Linux provides safe abstractions for common kernel concepts:

### Kernel Module

```rust
module! {
    type: MyModule,
    name: "my_module",
    author: "Rust Developer",
    description: "A simple Rust kernel module",
    license: "GPL",
}
```

### File Operations

The `file_operations` trait allows implementing custom file behavior:

```rust
struct MyFile {
    data: Vec<u8>,
}

impl kernel::file::Operations for MyFile {
    fn open(_context: &kernel::file::OpenContext) -> Result<Self> {
        Ok(MyFile { data: Vec::new() })
    }

    fn read(
        &mut self,
        _file: &kernel::file::File,
        buf: &mut kernel::IoBuffer,
        _offset: u64,
    ) -> Result<usize> {
        let to_copy = buf.len().min(self.data.len());
        buf.write(&self.data[..to_copy])?;
        Ok(to_copy)
    }
}
```

### Memory‑Mapped I/O

Safe wrappers for `ioremap` and register access prevent accidental writes to read‑only registers.

**Real case:** Huawei's kernel team is writing a USB driver in Rust for their HarmonyOS devices.

## 3. Safety Guarantees in Kernel Space

Rust's safety guarantees are enforced at compile time, but the kernel environment introduces unique challenges:

| Challenge | Kernel Context | Rust Solution |
|-----------|----------------|---------------|
| No Rust runtime | Kernel has no allocator, no threads | Use `core`, `alloc` with kernel allocator |
| Interrupt handlers | Asynchronous, cannot block | Use `SpinLock` and `_irqsave` macros |
| DMA | Direct memory access bypasses CPU cache | Use `dma_alloc_coherent` with safe wrappers |
| Raw pointers | Hardware registers are raw pointers | Wrap in `Unique<u8>` with lifetime tracking |
| Panics | Kernel panic on panic | Use `panic=abort` or custom panic handler |

**Real case:** The Android Binder driver rewrite in Rust eliminates use‑after‑free bugs that were common in the C version.

Formula: `Safety_Guarantees = Ownership + Borrowing + Lifetimes + Kernel_Abstractions`.

⚠️ **Warning:** `unsafe` blocks are still required for low‑level hardware access; ensure they are minimal and audited.

💡 **Tip:** Use the `#[pinned_init]` macro for safe initialization of structs that will be pinned (e.g., for self‑referential futures in async drivers).

## 4. Rust vs. C for Kernel Development

| Aspect | Rust | C |
|--------|------|---|
| Memory Safety | Compile‑time guarantees | Manual, error‑prone |
| Concurrency | Fearless (ownership) | Data races are UB |
| Error Handling | `Result` type | Return codes, easy to ignore |
| Abstractions | Traits, generics, macros | Function pointers, void* |
| Ecosystem | New, growing | Mature, vast |
| Performance | Comparable | Bare‑metal, no overhead |
| Learning Curve | Steeper (ownership model) | Easier for beginners |

**Real case:** Microsoft's Windows kernel team has explored Rust for driver development, reporting 70% fewer bugs compared to C.

💡 **Tip:** Use C for existing subsystems that rely on complex pointer arithmetic; use Rust for new drivers where safety is paramount.

## 5. Minimal Kernel Module Skeleton

```rust
// my_module.rs
#![no_std]
#![feature(kernel_module)]

use kernel::prelude::*;

module! {
    type: MyModule,
    name: "my_module",
    author: "Rustacean",
    description: "A minimal Rust kernel module",
    license: "GPL",
}

struct MyModule;

impl Module for MyModule {
    fn init(_module: &'static ThisModule) -> Result<Self> {
        pr_info!("MyModule: initialized\n");
        Ok(MyModule)
    }
}

impl Drop for MyModule {
    fn drop(&mut self) {
        pr_info!("MyModule: cleaned up\n");
    }
}
```

---

## 📦 Compression Code

Complete Rust script for a kernel module that creates a character device with read/write operations.

```rust
// char_driver.rs
#![no_std]
#![feature(kernel_module)]

use kernel::prelude::*;
use kernel::file;
use kernel::file::Operations;
use kernel::IoBuffer;

module! {
    type: CharDriver,
    name: "char_driver",
    author: "Rustacean",
    description: "A simple character device driver",
    license: "GPL",
}

struct CharDriver {
    data: Vec<u8>,
    lock: SpinLock<()>,
}

impl kernel::Module for CharDriver {
    fn init(_module: &'static ThisModule) -> Result<Self> {
        pr_info!("CharDriver: loaded\n");
        let driver = CharDriver {
            data: Vec::new(),
            lock: SpinLock::new(()),
        };
        // Register character device
        let _registration = file::Registration::new(
            c_str!("my_char_dev"),
            file::Operations::new::<Self>(),
        )?;
        Ok(driver)
    }
}

impl Operations for CharDriver {
    fn open(_context: &file::OpenContext) -> Result<Self> {
        Ok(CharDriver {
            data: Vec::new(),
            lock: SpinLock::new(()),
        })
    }

    fn read(
        &mut self,
        _file: &file::File,
        buf: &mut IoBuffer,
        _offset: u64,
    ) -> Result<usize> {
        let _guard = self.lock.lock();
        let to_copy = buf.len().min(self.data.len());
        buf.write(&self.data[..to_copy])?;
        // Remove read bytes
        self.data.drain(..to_copy);
        Ok(to_copy)
    }

    fn write(
        &mut self,
        _file: &file::File,
        buf: &IoBuffer,
        _offset: u64,
    ) -> Result<usize> {
        let _guard = self.lock.lock();
        self.data.extend_from_slice(buf);
        Ok(buf.len())
    }
}

impl Drop for CharDriver {
    fn drop(&mut self) {
        pr_info!("CharDriver: unloaded\n");
    }
}
```
