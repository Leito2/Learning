# 👋 Welcome to Go Extra

## What will you learn?

This advanced track covers deep theoretical topics that separate senior Go engineers from mid-level developers. You will master the internals of Go's runtime, learn production-grade concurrency patterns, tune performance for massive scale, navigate enterprise codebases, and extend Go's reach into WebAssembly and kernel observability with eBPF.

By the end of this track, you will be able to reason about Go's memory model with confidence, diagnose sub-millisecond GC issues, design parallel pipelines that survive cancellation, profile and optimize hot paths, organize millions of lines of Go, compile to WASM for browsers and edge runtimes, and instrument the Linux kernel from Go user space.

## Course Structure

- [[01 - Go Memory Model and GC|🧠 01 - Memory Model and GC]]
- [[02 - Advanced Concurrency Patterns|🧩 02 - Concurrency Patterns]]
- [[03 - Go Performance Tuning|⚡ 03 - Performance Tuning]]
- [[04 - Go in Large Codebases|🏢 04 - Large Codebases]]
- [[05 - WebAssembly with TinyGo|🌐 05 - WebAssembly]]
- [[06 - Go and eBPF|🔬 06 - eBPF]]

## Capstone Project

You will build a production-ready observability pipeline: a Go service that collects kernel events via eBPF, processes them through a concurrent fan-out/fan-in pipeline, exports metrics to a WASM dashboard compiled with TinyGo, and runs under strict memory limits tuned with advanced GC settings. The project must include benchmarking, custom linting, and a Bazel-based build to simulate a large-monorepo workflow.

---

💡 **Tip:** These courses separate senior Go engineers from mid-level ones. Master them to lead Go projects.
