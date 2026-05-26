# 🧠 Runtimes, Bun, and Backends — Foundations

## What Is a Runtime?

A **runtime** (runtime environment) is the program that executes your code. When you write JavaScript or TypeScript, you're writing instructions — but something has to read those instructions and turn them into actual machine operations. That "something" is the runtime.

Think of it like a car engine: your JavaScript code is the steering and pedals, the runtime is the entire engine that makes the car move. Without a runtime, your `.js` or `.ts` files are just text — they do nothing.

### What a Runtime Does

| Responsibility | How |
|---|---|
| **Parse code** | Reads your `.js`/`.ts` files and builds an AST (Abstract Syntax Tree) |
| **Compile/JIT** | Converts parsed code into machine instructions the CPU can execute |
| **Memory management** | Allocates and frees RAM for your variables, objects, and functions (garbage collection) |
| **I/O operations** | Reads files, makes HTTP requests, opens sockets — talks to the operating system |
| **Event loop** | Schedules and coordinates async operations (callbacks, promises, timers) |
| **Provide APIs** | Exposes built-in functions: `fetch()`, `setTimeout()`, `console.log()`, file system access |

### The Runtime Spectrum

```
Browser Runtime                    Server Runtime                   Edge/Serverless Runtime
─────────────────────────────────────────────────────────────────────────────────────────
V8 (Chrome)              Node.js (V8 + libuv)              Cloudflare Workers (V8 isolates)
JavaScriptCore (Safari)  Bun (JSC + Rust I/O)              Deno Deploy (V8 sandboxed)
SpiderMonkey (Firefox)   Deno (V8 + Rust)                  AWS Lambda (containerized)
```

---

## What Is Node.js?

**Node.js** (2009, Ryan Dahl) was the first runtime that took JavaScript outside the browser. Before Node, JavaScript only ran in browsers for UI interactions. Node combined:
- **V8 engine** (Google's JavaScript engine from Chrome) — to parse and execute JS code
- **libuv** (C library) — to handle I/O, file system, networking, and the event loop

This let developers use JavaScript for **servers, CLIs, scripts, and backend applications** — the same language on the frontend AND backend.

### What You Can Build with Node.js

| Category | Examples |
|---|---|
| **Web servers** | Express, Fastify, NestJS — REST and GraphQL APIs |
| **CLI tools** | npm, yarn, eslint, prettier, vite |
| **Real-time apps** | Chat servers (Socket.io), collaborative editors, live dashboards |
| **Backend services** | Authentication, payment processing, data pipelines |
| **Dev tools** | Build systems (webpack, esbuild), test runners (jest, vitest) |
| **Scripts** | Automation, data processing, file manipulation |

---

## What Is Bun?

**Bun** (2023, Jarred Sumner) is a runtime that aims to do everything Node.js does — but faster, with fewer dependencies, and with more built-in tools. Originally written in **Zig**, Bun was completely rewritten in **Rust** in 2025-2026 for stability.

Bun replaces:
- **Node.js** (the runtime itself)
- **npm/yarn/pnpm** (package manager — `bun install` is 10-20x faster)
- **webpack/esbuild** (bundler — `bun build` is built-in)
- **jest/vitest** (test runner — `bun test` is built-in)
- **ts-node/tsx** (TypeScript runner — TypeScript works natively, zero config)

All in a **single ~80MB binary**.

### Bun vs Node.js — Architectural Difference

```
Node.js:                          Bun:
┌──────────────┐                  ┌──────────────┐
│ Your JS Code │                  │ Your TS Code │ ← Native TypeScript
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│ V8 Engine    │ (C++)           │ JavaScriptCore│ (C++) — Apple's engine
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│ libuv        │ (C)             │ Rust Core     │ ← Custom I/O, HTTP, FS
│ Event Loop   │                  │ Event Loop    │   Package manager, bundler
│ Thread Pool  │                  │               │   All in one binary
└──────┬───────┘                  └──────────────┘
       │                                 │
       ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│ OS Kernel    │                  │ OS Kernel    │
└──────────────┘                  └──────────────┘
```

---

## What Does the Backend Have to Do with Runtimes?

The **backend** is the server-side of an application — the part users don't see. When you log into a website, your credentials go to a backend server. When you search for a product, a backend queries a database. When an ML model makes a prediction, a backend API receives the request and returns the result.

A **runtime is the engine that powers the backend**. Without a runtime, there is no backend — you need Node.js, Bun, Go, Python, or another runtime to execute server code.

### The Backend Stack

```
┌──────────────────────────────────────────────────────────────┐
│                     THE BACKEND STACK                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Runtime (Node, Bun, Go, Python)                     │   │
│  │  ─ Executes your server code                         │   │
│  │  ─ Handles HTTP requests, WebSockets, gRPC           │   │
│  │  ─ Manages concurrency (thousands of simultaneous    │   │
│  │    connections)                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │  Framework (Express, Fastify, Hono, Gin, FastAPI)    │  │
│  │  ─ Routing: GET /users → handler function            │  │
│  │  ─ Middleware: auth, rate limiting, validation        │  │
│  │  ─ Serialization: JSON, XML, Protocol Buffers         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │  Business Logic                                       │  │
│  │  ─ Application code: authentication, payments,        │  │
│  │    recommendations, ML inference calls                │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐          │
│  │ Database │    │ Cache    │    │ External API │          │
│  │PostgreSQL│    │ Redis    │    │ ML Model     │          │
│  └──────────┘    └──────────┘    └──────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### Why the Runtime Choice Matters for Backends

| Factor | Impact |
|--------|--------|
| **Throughput** | How many requests per second can your backend handle? Bun: 150K, Node: 55K |
| **Cold start** | Serverless functions boot time. Bun: 5ms, Node: 50ms → 10x difference in Lambda costs |
| **RAM usage** | Cloud costs scale with memory. Bun uses 40% less RAM than Node.js at idle |
| **Developer speed** | TypeScript works natively in Bun — no `tsconfig`, no `ts-node` |
| **Ecosystem** | Node has 2M+ packages. Bun is ~90% compatible but still catching up |
| **Stability** | Node: 15 years of production use. Bun (post-Rust rewrite): much improved but newer |

### Concrete Example

```
Scenario: ML Inference API that receives features → calls model → returns prediction

Backend code (same in Node and Bun):
  1. Receive POST /predict with JSON body
  2. Validate input features
  3. Format and send to Triton Inference Server (gRPC)
  4. Return prediction JSON to client

Node.js backend:  55K req/s, 45MB RAM, 50ms cold start
Bun backend:     150K req/s, 28MB RAM, 5ms cold start
Go backend:      200K req/s, 12MB RAM, instant start

All do the same job. Bun gives you Node.js-compatible code with Go-like performance.
```

---

## Key Takeaways

- A **runtime** is the program that executes your code — it handles parsing, memory, I/O, and the event loop.
- **Node.js** = V8 (JS engine) + libuv (I/O). The original server-side JS runtime. Mature, huge ecosystem.
- **Bun** = JavaScriptCore (JS engine) + Rust (I/O + tools). Faster start, lower RAM, built-in bundler/test runner/package manager.
- The **backend** is the server-side — it needs a runtime to execute. The runtime choice determines throughput, latency, RAM cost, and cold start time.
- Bun is not "Node.js but faster" — it's a **unified toolkit** (runtime + package manager + bundler + test runner) in one binary.
