# 🤗 Transformers.js — HuggingFace Models in the Browser

**Transformers.js** (HuggingFace) is the JavaScript port of the `transformers` Python library. It runs **HuggingFace transformer models** in the browser via ONNX Runtime Web + WebGPU acceleration. The HuggingFace Hub has **800,000+ models**; transformers.js makes ~80% of them browser-runnable via pre-converted ONNX files (the `Xenova/` organization).

This is the **fastest path to a HuggingFace model in the browser**. You don't need to export ONNX yourself — HuggingFace's `transformers.js` library handles it. **Use transformers.js for**: text classification, NER, summarization, translation, embeddings (sentence-transformers), image classification, zero-shot classification, fill-mask. **Don't use it for**: large LLMs (use WebLLM, note 03) or production ML pipelines (use ONNX Runtime Web directly, note 02).

This note covers the pipeline API, model selection, the `Xenova/` model namespace, real-time streaming, and the production patterns for caching and WebGPU acceleration.

## 🎯 Learning Objectives

- Use the **`pipeline()` API** for HuggingFace models.
- Pick models from the **`Xenova/` namespace** (pre-converted ONNX).
- Run **text, image, and audio** pipelines in the browser.
- Apply **WebGPU acceleration** via ONNX Runtime Web.
- Cache models in **Cache API** for fast re-load.
- Avoid the three most common transformers.js pitfalls.

## 1. The Pipeline API

The `pipeline()` function is the entry point:

```javascript
import { pipeline } from "@huggingface/transformers";

// 1. Load a pipeline (downloads model on first call)
const classifier = await pipeline(
  "sentiment-analysis",  // task
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english",  // model
  { quantized: true },  // use Q8 quantization for smaller size
);

// 2. Run inference
const result = await classifier("I love this!");
console.log(result);
// [{ label: "POSITIVE", score: 0.9998 }]
```

That's it. **Three lines** for a working text classifier in the browser.

## 2. Available Tasks

| Task | Pipeline name | Example use |
|------|---------------|-------------|
| Text classification | `sentiment-analysis` | Sentiment, toxicity |
| Zero-shot classification | `zero-shot-classification` | Custom labels without training |
| Token classification | `token-classification` | NER, POS tagging |
| Question answering | `question-answering` | Extract answer from context |
| Summarization | `summarization` | Long → short text |
| Translation | `translation` | Language A → B |
| Fill-mask | `fill-mask` | Predict masked words |
| Feature extraction | `feature-extraction` | Embeddings |
| Text generation | `text-generation` | Small LLMs (use WebLLM for larger) |
| Image classification | `image-classification` | Vision models |
| Image-to-text | `image-to-text` | Image captioning |
| Automatic speech recognition | `automatic-speech-recognition` | Speech → text |
| Text-to-speech | `text-to-speech` | Text → audio |

## 3. The `Xenova/` Namespace

Not all HuggingFace models work in the browser. The `Xenova/` namespace contains **pre-converted ONNX models** that work with transformers.js:

```javascript
// ❌ Original PyTorch model — won't run in browser
await pipeline("text-classification", "distilbert-base-uncased-finetuned-sst-2-english");

// ✅ Pre-converted ONNX model — runs in browser
await pipeline("text-classification", "Xenova/distilbert-base-uncased-finetuned-sst-2-english");
```

Browse available models at [huggingface.co/Xenova](https://huggingface.co/Xenova).

## 4. Embeddings with sentence-transformers

```javascript
import { pipeline, env } from "@huggingface/transformers";

// Allow remote models (default is local-only for security)
env.allowRemoteModels = true;
env.allowLocalModels = true;

const embedder = await pipeline(
  "feature-extraction",
  "Xenova/bge-small-en-v1.5",
  { quantized: true },
);

const output = await embedder("What is WebGPU?", { pooling: "mean", normalize: true });
console.log(output);
// 384-dim Float32Array
```

Embeddings for browser-side RAG: embed the user's query, send to a vector DB (server) for retrieval.

## 5. Streaming Tokens

For generation tasks, transformers.js supports streaming:

```javascript
const generator = await pipeline(
  "text-generation",
  "Xenova/distilgpt2",
);

// Get a TextStreamer for live output
import { TextStreamer } from "@huggingface/transformers";
const streamer = new TextStreamer(generator.tokenizer, {
  skip_prompt: true,
  skip_special_tokens: true,
  callback_function: (text) => {
    document.getElementById("output").textContent += text;
  },
});

const output = await generator("Once upon a time", {
  max_new_tokens: 50,
  do_sample: true,
  streamer,  // live output
});
```

The text appears character-by-character as the model generates.

## 6. WebGPU Acceleration

Transformers.js uses ONNX Runtime Web under the hood — `env.backends.onnx.wasm.proxy` controls the backend:

```javascript
import { env } from "@huggingface/transformers";

// Use WebGPU for acceleration
env.backends.onnx.wasm.proxy = false;
// (WebGPU is auto-detected when available)

const classifier = await pipeline(
  "sentiment-analysis",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
);
```

WebGPU vs WASM benchmark on RTX 3060:
- WASM: 200ms per inference
- WebGPU: 30ms per inference (6.5× speedup)

## 7. Image Classification

```javascript
const imageClassifier = await pipeline(
  "image-classification",
  "Xenova/vit-base-patch16-224",
);

// From an <img> element
const img = document.getElementById("cat-photo");
const result = await imageClassifier(img);
console.log(result);
// [{ label: "Egyptian cat", score: 0.92 }, ...]
```

## 8. Question Answering

```javascript
const qa = await pipeline(
  "question-answering",
  "Xenova/distilbert-base-cased-distilled-squad",
);

const answer = await qa({
  question: "What is WebGPU?",
  context: "WebGPU is a W3C standard API that provides GPU access from JavaScript in the browser. It enables ML inference and other compute-heavy workloads to run locally.",
});
console.log(answer);
// { score: 0.95, answer: "a W3C standard API that provides GPU access from JavaScript in the browser", start: 9, end: 75 }
```

## 9. Zero-Shot Classification

```javascript
const classifier = await pipeline(
  "zero-shot-classification",
  "Xenova/distilbert-base-uncased-mnli",
);

const result = await classifier(
  "I love programming in JavaScript",
  ["technology", "sports", "cooking", "politics"],
);
console.log(result);
// { sequence: "...", labels: ["technology", "sports", ...], scores: [0.95, 0.02, ...] }
```

**No training needed** — just provide the labels you care about.

## 10. Model Caching

```javascript
import { env } from "@huggingface/transformers";

// Use Cache API for persistent model storage
env.useFs = false;  // disabled in browser (uses Cache API by default)

// Models are cached in Cache API after first download
// Subsequent loads are instant
```

## 11. Production Patterns

### Lazy Loading

```javascript
let classifier = null;
async function getClassifier() {
  if (!classifier) {
    classifier = await pipeline("sentiment-analysis", "Xenova/distilbert-base-uncased-finetuned-sst-2-english");
  }
  return classifier;
}
```

### Loading Indicator

```javascript
async function loadModelWithProgress() {
  const classifier = await pipeline(
    "feature-extraction",
    "Xenova/bge-small-en-v1.5",
    {
      quantized: true,
      progress_callback: (data) => {
        // data: { status, file, progress, loaded, total }
        console.log(`${data.status}: ${data.file} (${data.progress}%)`);
      },
    },
  );
  return classifier;
}
```

### Web Worker for Non-Blocking

```javascript
// main.js
const worker = new Worker("./transformer-worker.js");
worker.postMessage({ type: "classify", text: "Hello!" });
worker.onmessage = (e) => console.log(e.data);

// transformer-worker.js
import { pipeline } from "@huggingface/transformers";
const classifier = await pipeline("sentiment-analysis", "Xenova/distilbert-base-uncased-finetuned-sst-2-english");
self.onmessage = async (e) => {
  if (e.data.type === "classify") {
    const result = await classifier(e.data.text);
    self.postMessage(result);
  }
};
```

## 12. ❌/✅ Antipatterns

### ❌ Using original HuggingFace model names

```javascript
// ⚠️ Won't work — not converted to ONNX
await pipeline("text-classification", "distilbert-base-uncased-finetuned-sst-2-english");
```

### ✅ Use Xenova/ pre-converted models

```javascript
// ✅ Pre-converted ONNX
await pipeline("text-classification", "Xenova/distilbert-base-uncased-finetuned-sst-2-english");
```

### ❌ Loading model on every call

```javascript
// ⚠️ Re-downloads 250MB model every time
async function classify(text) {
  const classifier = await pipeline("sentiment-analysis", "...");
  return classifier(text);
}
```

### ✅ Load once, reuse

```javascript
const classifier = await pipeline("sentiment-analysis", "...");
async function classify(text) {
  return classifier(text);
}
```

### ❌ Not handling model load errors

```javascript
// ⚠️ Crashes on browser without WebGPU
const classifier = await pipeline(...);
```

### ✅ Fallback handling

```javascript
try {
  classifier = await pipeline("...", "Xenova/...", { device: "webgpu" });
} catch (e) {
  classifier = await pipeline("...", "Xenova/...", { device: "wasm" });
}
```

## 13. Production Reality

**Caso real — Browser-based Chatbot Demo:** Used `Xenova/distilbert-base-uncased-finetuned-sst-2-english` for sentiment classification and `Xenova/bge-small-en-v1.5` for embeddings. Both run in browser via WebGPU. **Total bundle**: 250MB+ (cached after first load).

**Caso real — Real-time Image Tagging:** Image classification pipeline using `Xenova/vit-base-patch16-224`. Webcam frames captured via `getUserMedia()`, classified locally, displayed in 50ms per frame. **Privacy**: photos never leave the device.

## 📦 Compression Code

```javascript
// 📦 Compression: Transformers.js in 30 lines

import { pipeline } from "@huggingface/transformers";

class BrowserPipeline {
  constructor(task, model) {
    this.task = task;
    this.model = model;
    this.pipeline = null;
  }

  async init() {
    if (!this.pipeline) {
      this.pipeline = await pipeline(this.task, this.model, {
        quantized: true,
        progress_callback: (data) => updateUI(data),
      });
    }
    return this.pipeline;
  }

  async run(input) {
    const p = await this.init();
    return p(input);
  }
}

// Usage
const sentiment = new BrowserPipeline(
  "sentiment-analysis",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
);
const result = await sentiment.run("I love this!");
console.log(result);
```

## 🎯 Key Takeaways

1. **`pipeline()` is the entry point** — one function call per task.
2. **Use `Xenova/` namespace** — pre-converted ONNX models.
3. **WebGPU via ONNX Runtime Web** — 5-10× speedup over WASM.
4. **Quantize models** (`quantized: true`) — 4× smaller, slightly lower quality.
5. **Lazy load and cache** — 250MB+ models shouldn't download on page load.
6. **Worker threads** for non-blocking inference.
7. **Progress callbacks** for user feedback during download.

## References

- [[00 - Welcome to WebGPU and On-Device ML|Welcome]] — course map.
- [[02 - ONNX Runtime Web - ML in the Browser|ONNX Runtime]] — the underlying runtime.
- [[03 - WebLLM - Full LLMs in Browser|WebLLM]] — for LLM inference.
- Transformers.js docs: https://huggingface.co/docs/transformers.js
- Xenova models: https://huggingface.co/Xenova
- HuggingFace Tasks: https://huggingface.co/tasks