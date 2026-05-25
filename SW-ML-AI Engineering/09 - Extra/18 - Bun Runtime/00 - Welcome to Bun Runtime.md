# 🍞 Welcome to Bun Runtime

Bun is a JavaScript/TypeScript runtime, bundler, test runner, and package manager — all in one binary. Originally written in Zig, Bun was completely rewritten in Rust in 2025-2026, eliminating historical memory leaks and segfaults while maintaining or slightly improving its raw throughput. The result is a runtime that starts in ~5ms, handles 150K+ HTTP requests per second, and installs npm packages 10-20x faster than npm — all with ~40% less RAM than Node.js at idle.

For AI/ML engineers building backend APIs, inference gateways, or edge computing services, Bun offers Go-level performance with the JavaScript/TypeScript ecosystem — no need to choose between speed and npm compatibility.

---

## Course Index

1. [[01 - Bun Fundamentals|Bun Fundamentals: Runtime, Performance, and ML Applications]]

---

## Why Bun Matters

| Metric | Node.js | Bun |
|--------|:-------:|:---:|
| **Cold start** | ~50ms | ~5ms |
| **HTTP throughput (JSON)** | 50-65K req/s | 150-180K req/s |
| **RAM at idle** | Baseline | ~40% less |
| **Package install** | Baseline (npm) | 10-20x faster (`bun install`) |
| **Binary size** | N/A (V8 separate) | ~80MB single binary |
| **TypeScript** | Requires ts-node/tsx | Native, zero config |
| **Bundler** | Separate (webpack/esbuild) | Built-in |
| **Test runner** | Separate (jest/vitest) | Built-in |
| **Language** | C++ (V8) + JS | Rust (rewritten from Zig) |

---

💡 **Tip:** Bun is not a "Node.js killer" — it's a Node.js-compatible upgrade. 90% of npm packages work with Bun, and the `bun:sqlite` built-in SQLite driver eliminates the need for `better-sqlite3` native bindings.

⚠️ **Warning:** In real-world applications where the bottleneck is database queries or external API calls (not the runtime), Bun's 3x HTTP advantage shrinks to marginal. The real wins are cold starts (serverless), RAM efficiency (cloud costs), and developer tooling speed.
