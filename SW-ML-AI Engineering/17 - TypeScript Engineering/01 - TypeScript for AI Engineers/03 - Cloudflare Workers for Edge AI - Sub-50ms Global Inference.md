# 🎯 03 - Cloudflare Workers for Edge AI — Sub-50ms Global Inference

> **Deploy AI workloads to the edge. Cloudflare Workers + Workers AI = <50ms latency globally. The fastest path to a worldwide AI service.**

## 🎯 Learning Objectives
- Set up Cloudflare Workers with TypeScript
- Use Workers AI for native LLM inference at the edge
- Build a streaming LLM proxy with sub-50ms latency
- Use Durable Objects for stateful AI sessions
- Configure Workers KV for response caching at the edge
- Deploy with `wrangler` to Cloudflare's global network
- Compare Workers to traditional serverless (Vercel, AWS Lambda)

## Introduction

**Cloudflare Workers** runs your code in 300+ cities worldwide. For AI, this means:

- **<50ms latency globally** — your code runs in the city closest to the user
- **No cold starts** — V8 isolates warm in <5ms
- **Workers AI** — native LLM inference on Cloudflare's GPUs (Llama, Mistral, etc.)
- **Durable Objects** — stateful sessions with single-instance consistency
- **Workers KV** — distributed cache, <1ms reads
- **R2** — object storage with zero egress fees

For an AI engineer in 2026, knowing Workers is the difference between "my LLM is fast in us-east-1" and "my LLM is fast everywhere".

---

## 1. Setup

```bash
# Install wrangler
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create a new project
wrangler init my-ai-worker
cd my-ai-worker
```

`wrangler.toml`:
```toml
name = "my-ai-worker"
main = "src/index.ts"
compatibility_date = "2024-12-01"

[ai]
binding = "AI"  # Workers AI binding

# Optional: Durable Objects
[[durable_objects.bindings]]
name = "SESSION_DO"
class_name = "SessionDurableObject"

[[migrations]]
tag = "v1"
new_sqlite_classes = ["SessionDurableObject"]

[vars]
ALLOWED_ORIGINS = "https://my-frontend.com"
```

---

## 2. A Basic LLM Proxy

```typescript
// src/index.ts
export interface Env {
    AI: Ai;
    ALLOWED_ORIGINS: string;
}

export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        // CORS
        if (req.method === "OPTIONS") {
            return new Response(null, {
                headers: {
                    "Access-Control-Allow-Origin": env.ALLOWED_ORIGINS,
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                },
            });
        }
        
        if (req.method !== "POST") {
            return new Response("Method not allowed", { status: 405 });
        }
        
        const { messages } = await req.json<{ messages: Array<{role: string; content: string}> }>();
        
        // Call Workers AI (Llama 3 8B)
        const response = await env.AI.run(
            "@cf/meta/llama-3-8b-instruct",
            {
                messages,
            }
        );
        
        return new Response(JSON.stringify(response), {
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": env.ALLOWED_ORIGINS,
            },
        });
    },
};
```

Deploy:
```bash
wrangler deploy
```

Test:
```bash
curl -X POST https://my-ai-worker.workers.dev \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 3. Streaming LLM Responses

```typescript
export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const { messages } = await req.json();
        
        // Streaming response
        const stream = await env.AI.run(
            "@cf/meta/llama-3-8b-instruct",
            {
                messages,
                stream: true,
            }
        );
        
        return new Response(stream, {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": env.ALLOWED_ORIGINS,
            },
        });
    },
};
```

The client:
```typescript
const response = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ messages }),
});

const reader = response.body!.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    console.log(chunk);
}
```

---

## 4. Workers AI Models

```typescript
// Llama 3 (Meta)
const response = await env.AI.run(
    "@cf/meta/llama-3-8b-instruct",
    { messages }
);

// Mistral
const response = await env.AI.run(
    "@cf/mistral/mistral-7b-instruct-v0.1",
    { messages }
);

// Embeddings
const embeddings = await env.AI.run(
    "@cf/baai/bge-base-en-v1.5",
    { text: ["Hello world"] }
);

// Whisper (audio transcription)
const transcription = await env.AI.run(
    "@cf/openai/whisper",
    { audio: audioArray }
);

// Image generation (Stable Diffusion)
const image = await env.AI.run(
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    { prompt: "a cat" }
);
```

Workers AI runs on Cloudflare GPUs. Pricing is bundled into the Workers plan.

---

## 5. Durable Objects for Stateful AI Sessions

For multi-turn conversations with state:

```typescript
// src/session.ts
export class SessionDurableObject {
    private messages: Array<{role: string; content: string}> = [];
    private createdAt: number = Date.now();
    
    constructor(private state: DurableObjectState, private env: Env) {
        this.state.blockConcurrencyWhile(async () => {
            const stored = await this.state.storage.get<any>("messages");
            this.messages = stored?.messages || [];
        });
    }
    
    async fetch(req: Request): Promise<Response> {
        const url = new URL(req.url);
        const path = url.pathname;
        
        if (path === "/chat" && req.method === "POST") {
            const { message } = await req.json<{message: string}>();
            this.messages.push({ role: "user", content: message });
            
            // Call LLM
            const response = await this.env.AI.run(
                "@cf/meta/llama-3-8b-instruct",
                { messages: this.messages }
            );
            
            this.messages.push({ role: "assistant", content: response.response });
            
            // Persist
            await this.state.storage.put("messages", { messages: this.messages });
            
            return Response.json({ response: response.response, history: this.messages });
        }
        
        if (path === "/history" && req.method === "GET") {
            return Response.json({ messages: this.messages });
        }
        
        return new Response("Not found", { status: 404 });
    }
}
```

Worker routes to the Durable Object:
```typescript
export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const url = new URL(req.url);
        const sessionId = url.searchParams.get("session_id") || "default";
        
        const id = env.SESSION_DO.idFromName(sessionId);
        const stub = env.SESSION_DO.get(id);
        
        return stub.fetch(req);
    },
};
```

Each session has a single Durable Object instance — strong consistency.

---

## 6. KV Cache for LLM Responses

```typescript
export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const { messages } = await req.json();
        
        // Cache key based on messages hash
        const cacheKey = await hashMessages(messages);
        
        // Check cache
        const cached = await env.CACHE.get(cacheKey);
        if (cached) {
            return new Response(cached, {
                headers: { "X-Cache": "HIT" },
            });
        }
        
        // Cache miss: call LLM
        const response = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
            messages,
        });
        
        // Cache for 1 hour
        await env.CACHE.put(cacheKey, response.response, {
            expirationTtl: 3600,
        });
        
        return new Response(response.response, {
            headers: { "X-Cache": "MISS" },
        });
    },
};

async function hashMessages(messages: any[]): Promise<string> {
    const text = JSON.stringify(messages);
    const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(hash))
        .map(b => b.toString(16).padStart(2, "0"))
        .join("");
}
```

KV reads are <1ms globally. Cache hit rate 50% = 50% cost reduction.

---

## 7. Workers AI vs OpenAI API

| Feature | Workers AI | OpenAI API |
|---------|-----------|-----------|
| **Latency** | <50ms (edge) | 200-500ms (us-east-1) |
| **Global** | 300+ cities | 2-3 regions |
| **Models** | Llama, Mistral, etc. | GPT-4o, etc. |
| **Pricing** | Bundled in Workers | Per token |
| **Streaming** | ✅ | ✅ |
| **Fine-tuning** | ❌ | ✅ |
| **Tool calling** | Limited | ✅ |
| **Custom models** | ❌ | ❌ |

For sub-50ms global inference, Workers AI is unmatched. For GPT-4o quality, use OpenAI directly.

The production pattern: **OpenAI for quality, Workers AI for speed**. Use OpenAI for complex reasoning; use Workers AI for simple completion at the edge.

---

## 8. The Hono Framework

For larger Workers apps, use Hono:

```bash
npm install hono
```

```typescript
import { Hono } from "hono";
import { cors } from "hono/cors";

const app = new Hono<{ Bindings: Env }>();

app.use("*", cors({ origin: "*" }));

app.post("/chat", async (c) => {
    const { messages } = await c.req.json();
    const response = await c.env.AI.run(
        "@cf/meta/llama-3-8b-instruct",
        { messages }
    );
    return c.json(response);
});

app.get("/health", (c) => c.json({ status: "ok" }));

app.post("/session/:id/chat", async (c) => {
    const sessionId = c.req.param("id");
    const id = c.env.SESSION_DO.idFromName(sessionId);
    const stub = c.env.SESSION_DO.get(id);
    return stub.fetch(c.req.raw);
});

export default app;
```

Hono provides routing, middleware, and type safety — much cleaner than raw Workers.

---

## 9. Vector Search with Workers AI

```typescript
// Using Vectorize (Cloudflare's vector DB)
export interface Env {
    VECTORIZE: VectorizeIndex;
    AI: Ai;
}

export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const { query } = await req.json<{query: string}>();
        
        // Embed the query
        const { data: embedding } = await env.AI.run(
            "@cf/baai/bge-base-en-v1.5",
            { text: query }
        ) as { data: number[] };
        
        // Search Vectorize
        const matches = await env.VECTORIZE.query(embedding, {
            topK: 5,
            returnMetadata: true,
        });
        
        // Return matches
        return Response.json({
            matches: matches.matches.map(m => ({
                id: m.id,
                score: m.score,
                metadata: m.metadata,
            })),
        });
    },
};
```

Cloudflare Vectorize is a managed vector DB at the edge. <10ms queries.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Not using edge runtime for AI

```typescript
// ❌ Use Vercel (us-east-1 only)
export const runtime = "nodejs";

// ✅ Use Workers (300+ cities)
export const runtime = "edge";  // implicit for Workers
```

### 10.2 Antipattern 2: Caching sensitive data in KV

```typescript
// ❌ PII in cache
await env.CACHE.put(`user_${userId}_history`, messages);

// ✅ Redact PII before caching
const redacted = redactPII(messages);
await env.CACHE.put(`user_${userId}_history`, redacted);
```

### 10.3 Antipattern 3: Synchronous Durable Object operations

```typescript
// ❌ Blocking storage operations
async fetch(req: Request) {
    this.messages = await this.state.storage.get("messages");  // blocks
    // ...
}

// ✅ Use blockConcurrencyWhile
constructor(state, env) {
    this.state.blockConcurrencyWhile(async () => {
        const stored = await this.state.storage.get("messages");
        this.messages = stored?.messages || [];
    });
}
```

### 10.4 Antipattern 4: Workers for large compute

```typescript
// ❌ Workers have 10-50ms CPU time limits
// 30-second inference doesn't fit

// ✅ For long inference, use OpenAI/Anthropic directly
// Workers for short calls (<5s)
```

### 10.5 Antipattern 5: Not handling stream errors

```typescript
// ❌ Stream errors are silent
const stream = await env.AI.run(model, { messages, stream: true });
return new Response(stream);

// ✅ Handle errors in the stream
const stream = await env.AI.run(model, { messages, stream: true });
return new Response(stream, {
    headers: {
        "X-Error-Handling": "enabled",
    },
});
```

---

## 🎯 Key Takeaways

- Cloudflare Workers = <50ms global latency; no cold starts; V8 isolates.
- Workers AI runs Llama, Mistral, etc. on Cloudflare GPUs.
- Streaming with `stream: true` returns a `ReadableStream`.
- Durable Objects for stateful AI sessions with strong consistency.
- Workers KV for sub-1ms cache reads globally.
- Hono framework for clean routing and middleware.
- Vectorize for vector search at the edge.
- Avoid caching sensitive data, blocking DO operations, large compute, no error handling.

## References

- Cloudflare Workers — [developers.cloudflare.com/workers](https://developers.cloudflare.com/workers)
- Workers AI — [developers.cloudflare.com/workers-ai](https://developers.cloudflare.com/workers-ai)
- Durable Objects — [developers.cloudflare.com/durable-objects](https://developers.cloudflare.com/durable-objects)
- Hono framework — [hono.dev](https://hono.dev)
- Cloudflare Vectorize — [developers.cloudflare.com/vectorize](https://developers.cloudflare.com/vectorize)
- [[17 - TypeScript Engineering/01 - TypeScript for AI Engineers/01 - TypeScript Fundamentals|Note 01 — TS Fundamentals]]
- [[17 - TypeScript Engineering/01 - TypeScript for AI Engineers/02 - Next.js + Vercel AI SDK|Note 02 — Next.js]]
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — edge compute
- [[06 - Large Language Models/13 - vLLM and Advanced RAG|vLLM]] — high-throughput LLM serving