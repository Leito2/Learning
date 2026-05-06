# 🌐 WebAssembly with TinyGo

## Introduction

WebAssembly (WASM) is a binary instruction format designed as a portable compilation target for high-level languages. It runs in browsers at near-native speed and on servers via runtimes like Wasmtime and WasmEdge. Go can compile to WASM, but the standard Go compiler produces large binaries (~2 MB+) due to the included runtime. TinyGo is an alternative compiler that targets a subset of Go and generates dramatically smaller WASM modules, making it ideal for browsers, edge functions, and embedded environments.

This course bridges [[03 - Go Performance Tuning|performance tuning]] (binary size is a performance metric on the web) and [[06 - Go and eBPF|systems programming]] because WASM is increasingly used for sandboxed plugins and edge compute. Understanding TinyGo's limitations—such as restricted reflection and no goroutine scheduler in some targets—is essential for choosing the right tool.

## 1. WASM Fundamentals and TinyGo

Deep conceptual explanation:

- **Linear memory**: WASM modules operate on a single contiguous, resizable array of bytes. All data structures, including the heap and stack, live inside this memory.
- **Exports/Imports**: A WASM module exports functions (callable from the host) and imports functions (provided by the host, e.g., JS `console.log` or WASI `fd_write`).
- **WASI**: WebAssembly System Interface provides portable syscalls (files, clocks, random) for non-browser runtimes.
- **TinyGo**: A Go compiler using LLVM. Supports most of Go but lacks full reflection, complex cgo, and the full `sync` package on some targets.
- ⚠️ **Warning**: TinyGo does not support all standard library packages. Avoid `database/sql`, `net/http`, and heavy reflection when targeting WASM.
- 💡 **Tip**: Use `tinygo build -target wasm -o app.wasm` and pair it with `wasm-opt -Oz` from Binaryen to shrink binaries further.

Real case: Figma uses WASM for performance-critical rendering and computation in the browser. By compiling C++ (and experimenting with Rust/WASM) to WASM, they achieve near-native frame rates for vector graphics manipulation without plugins.

## 2. TinyGo vs Standard Go

TinyGo vs standard Go comparison:

| Feature | TinyGo | Standard Go (wasm/js) |
|---|---|---|
| Binary size | ~10–100 KB | ~2–5 MB |
| GC | Simple conservative / LLVM | Go concurrent tri-color |
| Goroutines | Coroutines / asyncify | Full scheduler |
| Reflection | Limited | Full |
| `net/http` | Not supported | Supported (via fetch polyfill) |
| WASI | Yes | Yes (Go 1.21+) |
| Browser JS interop | Manual via `syscall/js` | `syscall/js` wrapper |

Formula for estimating WASM binary size after compression:

```
WASM_Size = Go_Binary × Compression_Ratio
```

For example, a 50 KB TinyGo WASM binary served with Brotli (`Compression_Ratio ≈ 0.3`) yields a ~15 KB network transfer.

⚠️ **Warning**: Standard Go's WASM output includes the entire runtime and GC. Do not use it for browser widgets where download size matters; choose TinyGo instead.

## 3. WASM Compilation Pipeline

Mermaid diagram of the WASM compilation pipeline:

```mermaid
graph LR
    A[Go Source] -->|tinygo build| B[TinyGo IR]
    B -->|LLVM| C[LLVM IR]
    C -->|backend| D[WASM Binary]
    D -->|wasm-opt| E[Optimized WASM]
    E -->|brotli| F[Deployed Asset]
    style D fill:#ccffcc
    style E fill:#ccffcc
```

Wikimedia Commons reference:

- ![WebAssembly logo concept](https://upload.wikimedia.org/wikipedia/commons/1/1f/WebAssembly_Logo.svg)
- ![LLVM compilation diagram](https://upload.wikimedia.org/wikipedia/commons/1/1e/LLVM_Compiler_Diagram.svg)

## 4. Go Code: TinyGo WASM Module and JS Host

TinyGo WASM module (`main.go`):

```go
package main

//export add
func add(a, b int) int {
	return a + b
}

//export greet
func greet(namePtr, nameLen int32) int32 {
	// In real code, read name from linear memory.
	// This simplified example returns a fixed offset.
	return 42
}

func main() {}
```

Compile:

```bash
tinygo build -o main.wasm -target wasm -no-debug ./main.go
```

JS host (`host.js`):

```javascript
const fs = require('fs');
const source = fs.readFileSync('./main.wasm');

const go = {
  env: {
    "runtime.ticks": () => BigInt(0),
    "runtime.sleepTicks": () => {},
  }
};

WebAssembly.instantiate(source, go).then(result => {
  const exports = result.instance.exports;
  console.log("add(2, 3) =", exports.add(2, 3));
});
```

## 5. Running Go in Serverless WASM Runtimes

Deep conceptual explanation:

- **Edge functions**: Cloudflare Workers, Fastly Compute, and WasmEdge allow WASM modules to respond to HTTP requests with sub-millisecond cold starts.
- **WASI snapshot**: TinyGo's WASI target lets you write CLI tools that run on any WASI runtime without recompilation.
- **Security**: WASM's capability-based security model means a module can only access what the host explicitly imports (no arbitrary filesystem or network access).
- 💡 **Tip**: Use `wasmtime run --dir . module.wasm` to grant directory access to a WASI module during local testing.

---

## 📦 Compression Code

Complete Go script (compiled with TinyGo to WASM or standard Go) that compresses a string with gzip and returns the base64 result via a JS-friendly export:

```go
package main

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
)

//export compressToBase64
func compressToBase64(inputPtr, inputLen int32) int32 {
	// In real WASM linear memory usage, read from memory using syscall/js or unsafe.
	// Here we provide the logic assuming a host bridge.
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	// Write input data...
	gz.Close()
	_ = base64.StdEncoding.EncodeToString(buf.Bytes())
	return int32(buf.Len())
}

func main() {}
```

Native runner script:

```go
package main

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"fmt"
)

func main() {
	data := []byte("hello wasm")
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	gz.Write(data)
	gz.Close()
	fmt.Println(base64.StdEncoding.EncodeToString(buf.Bytes()))
}
```

## 🎯 Documented Project

### Description

Build a browser-based image watermarking tool where the image processing logic is compiled from Go to WASM using TinyGo. The host page uploads an image, passes the bytes into WASM linear memory, and retrieves the watermarked image buffer.

### Functional Requirements

1. A `watermark` Go package that reads a PNG from a byte slice, overlays text using `golang.org/x/image/font`, and returns the encoded PNG bytes.
2. Compile the package to WASM with TinyGo (`-target wasm`) and expose an `applyWatermark` export.
3. A vanilla HTML/JS host that uploads an image, writes it to WASM memory, calls the export, and downloads the result.
4. The WASM binary must be under 500 KB uncompressed (TinyGo should easily achieve this).
5. Include a WASI CLI target so the same Go code can watermark images from the command line via Wasmtime.

### Main Components

- `cmd/wasm`: TinyGo WASM build target with exports.
- `cmd/wasi`: WASI-compatible CLI using `os.Stdin`/`Stdout`.
- `internal/watermark`: Pure Go image processing (no CGO).
- `web/`: HTML host and JS glue code.

### Success Metrics

- WASM binary size < 500 KB before compression.
- Watermarking a 1920x1080 PNG completes in < 2s in a modern browser.
- WASI CLI runs successfully with `wasmtime run wasi.wasm < input.png > output.png`.

### References

- [TinyGo](https://tinygo.org/)
- [WebAssembly.org](https://webassembly.org/)
- [Figma WASM Blog](https://www.figma.com/blog/)
- [WASI](https://wasi.dev/)
