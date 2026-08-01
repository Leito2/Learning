# 🎯 02 - Next.js + Vercel AI SDK — Streaming Chat UI for AI Products

> **The most-used 2026 AI frontend stack. Build a streaming chatbot UI in 15 minutes. Next.js + Vercel AI SDK + React Server Components.**

## 🎯 Learning Objectives
- Set up Next.js 15 with the App Router and React Server Components
- Use the Vercel AI SDK (`useChat`, `streamText`, `generateText`)
- Build a streaming chat UI with proper message state management
- Implement tool calling (`useChat` with `tools`)
- Add multi-modal support (image inputs)
- Use Server Actions for backend LLM calls
- Deploy to Vercel with edge runtime

## Introduction

The **Vercel AI SDK** is the canonical way to build AI UIs in 2026. It provides:

- `useChat` — React hook for chat interfaces with streaming
- `streamText` — server-side streaming LLM calls
- `generateText` — non-streaming LLM calls
- `embed` / `embedMany` — embedding utilities
- `useCompletion` — text completion hook
- `useObject` — structured output hook

Combined with **Next.js 15** (App Router, React Server Components), it's the fastest path to a production AI frontend.

This note builds a complete chat UI from scratch.

---

## 1. Setup

```bash
npx create-next-app@latest my-ai-chat --typescript --tailwind --app
cd my-ai-chat
npm install ai @ai-sdk/openai
```

`tsconfig.json` should be set up automatically. The `ai` package is the Vercel AI SDK.

---

## 2. Basic Streaming Chat

```typescript
// app/api/chat/route.ts
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export const runtime = "edge";  // Vercel Edge runtime — <50ms cold start

export async function POST(req: Request) {
    const { messages } = await req.json();
    
    const result = await streamText({
        model: openai("gpt-4o-mini"),
        messages,
    });
    
    return result.toDataStreamResponse();
}
```

The client:
```typescript
// app/page.tsx
"use client";

import { useChat } from "ai/react";

export default function Chat() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({
        api: "/api/chat",
    });
    
    return (
        <div>
            {messages.map(m => (
                <div key={m.id}>
                    <strong>{m.role}:</strong> {m.content}
                </div>
            ))}
            
            <form onSubmit={handleSubmit}>
                <input value={input} onChange={handleInputChange} />
                <button type="submit">Send</button>
            </form>
        </div>
    );
}
```

That's it. Streaming chat UI in 30 lines.

---

## 3. Tool Calling with `useChat`

```typescript
// app/api/chat/route.ts
import { openai } from "@ai-sdk/openai";
import { streamText, tool } from "ai";
import { z } from "zod";

export async function POST(req: Request) {
    const { messages } = await req.json();
    
    const result = await streamText({
        model: openai("gpt-4o-mini"),
        messages,
        tools: {
            getWeather: tool({
                description: "Get the current weather for a location",
                parameters: z.object({
                    location: z.string().describe("City name"),
                }),
                execute: async ({ location }) => {
                    const response = await fetch(
                        `https://api.weather.com/v1/weather?city=${location}`
                    );
                    return response.json();
                },
            }),
            searchDatabase: tool({
                description: "Search the product database",
                parameters: z.object({
                    query: z.string(),
                    limit: z.number().default(5),
                }),
                execute: async ({ query, limit }) => {
                    // Search implementation
                    return results;
                },
            }),
        },
    });
    
    return result.toDataStreamResponse();
}
```

The LLM decides which tool to call; the tool's `execute` runs server-side; the result is fed back to the LLM; the LLM generates a final response. All streamed to the client.

---

## 4. Multi-Modal (Image Inputs)

```typescript
// app/api/vision/route.ts
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export async function POST(req: Request) {
    const { messages, imageBase64 } = await req.json();
    
    const result = await streamText({
        model: openai("gpt-4o"),
        messages: [
            ...messages,
            {
                role: "user",
                content: [
                    { type: "text", text: "What's in this image?" },
                    {
                        type: "image",
                        image: imageBase64,
                    },
                ],
            },
        ],
    });
    
    return result.toDataStreamResponse();
}
```

GPT-4o vision processes the image; the response streams back.

---

## 5. Structured Output with `useObject`

```typescript
// app/api/extract/route.ts
import { openai } from "@ai-sdk/openai";
import { generateObject } from "ai";
import { z } from "zod";

const PersonSchema = z.object({
    name: z.string(),
    age: z.number(),
    email: z.string().email(),
});

export async function POST(req: Request) {
    const { text } = await req.json();
    
    const { object } = await generateObject({
        model: openai("gpt-4o-mini"),
        schema: PersonSchema,
        prompt: `Extract person info: ${text}`,
    });
    
    return Response.json(object);
}
```

Client-side:
```typescript
// app/extract/page.tsx
"use client";

import { useObject } from "ai/react";

export default function ExtractPage() {
    const { object, submit, isLoading } = useObject({
        api: "/api/extract",
        schema: PersonSchema,
    });
    
    return (
        <div>
            <button onClick={() => submit({ text: "Maria is 28..." })}>
                Extract
            </button>
            {object && <pre>{JSON.stringify(object, null, 2)}</pre>}
        </div>
    );
}
```

The object is **incrementally streamed** as it's generated. The UI updates field-by-field as the LLM completes the JSON.

---

## 6. Server Actions (Next.js 15)

```typescript
// app/actions.ts
"use server";

import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

export async function summarizeDocument(document: string) {
    "use server";
    
    const { text } = await generateText({
        model: openai("gpt-4o-mini"),
        prompt: `Summarize this document: ${document}`,
    });
    
    return text;
}
```

Usage in a Server Component:
```typescript
// app/page.tsx
import { summarizeDocument } from "./actions";

export default async function Page() {
    const summary = await summarizeDocument("...");
    
    return <div>{summary}</div>;
}
```

Server Actions let you call LLM from server components without exposing the API key to the client.

---

## 7. The Streaming UI Pattern

For a production chat UI:

```typescript
// components/ChatInterface.tsx
"use client";

import { useChat } from "ai/react";
import { useEffect, useRef } from "react";

export function ChatInterface() {
    const { messages, input, handleInputChange, handleSubmit, isLoading, error } = useChat({
        api: "/api/chat",
        onError: (err) => {
            console.error("Chat error:", err);
        },
        onFinish: (message) => {
            // Log to analytics
            console.log("Finished:", message);
        },
    });
    
    const messagesEndRef = useRef<HTMLDivElement>(null);
    
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);
    
    return (
        <div className="flex flex-col h-screen">
            <div className="flex-1 overflow-y-auto p-4">
                {messages.map(m => (
                    <div
                        key={m.id}
                        className={`mb-4 ${m.role === "user" ? "text-right" : "text-left"}`}
                    >
                        <div className="inline-block p-3 rounded-lg bg-gray-100">
                            {m.content}
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
                
                {isLoading && <div className="text-gray-500">Thinking...</div>}
                {error && <div className="text-red-500">Error: {error.message}</div>}
            </div>
            
            <form onSubmit={handleSubmit} className="border-t p-4">
                <div className="flex">
                    <input
                        value={input}
                        onChange={handleInputChange}
                        placeholder="Ask anything..."
                        className="flex-1 p-2 border rounded-l"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="p-2 bg-blue-500 text-white rounded-r"
                    >
                        Send
                    </button>
                </div>
            </form>
        </div>
    );
}
```

The UI is responsive: scroll to bottom on new messages, show "Thinking..." while loading, show errors gracefully.

---

## 8. Multi-Modal Chat (Images)

```typescript
// components/MultiModalChat.tsx
"use client";

import { useChat } from "ai/react";
import { useState } from "react";

export function MultiModalChat() {
    const { messages, input, handleInputChange, handleSubmit, isLoading, append } = useChat({
        api: "/api/chat",
    });
    const [files, setFiles] = useState<FileList | null>(null);
    
    return (
        <div>
            {messages.map(m => (
                <div key={m.id}>
                    {m.role}: {m.content}
                    {/* Render images if any */}
                    {m.experimental_attachments?.map(att => (
                        <img
                            key={att.url}
                            src={att.url}
                            alt="upload"
                            className="max-w-xs"
                        />
                    ))}
                </div>
            ))}
            
            <form
                onSubmit={async (e) => {
                    e.preventDefault();
                    
                    if (files) {
                        // Convert files to data URLs
                        const attachments = await Promise.all(
                            Array.from(files).map(async (file) => ({
                                name: file.name,
                                contentType: file.type,
                                url: await fileToDataURL(file),
                            }))
                        );
                        
                        append(
                            { role: "user", content: input },
                            { experimental_attachments: attachments }
                        );
                        setFiles(null);
                    } else {
                        handleSubmit(e);
                    }
                }}
            >
                <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(e) => setFiles(e.target.files)}
                />
                <input value={input} onChange={handleInputChange} />
                <button type="submit">Send</button>
            </form>
        </div>
    );
}

async function fileToDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}
```

---

## 9. Streaming with Reasoning Models

```typescript
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export async function POST(req: Request) {
    const { messages } = await req.json();
    
    const result = await streamText({
        model: openai("o1-mini"),  // reasoning model
        messages,
    });
    
    return result.toDataStreamResponse();
}
```

The Vercel AI SDK handles reasoning models transparently. Reasoning tokens are consumed but not surfaced to the UI (unless you customize).

---

## 10. Deployment

```bash
# Push to GitHub
git push origin main

# Deploy to Vercel
vercel deploy --prod
```

Vercel automatically:
- Detects Next.js
- Sets up edge functions for `runtime = "edge"`
- Configures the Vercel AI Gateway (free, with rate limits)
- Sets up preview deployments for PRs

The default `gpt-4o-mini` deployment costs ~$0/month at low traffic; ~$50-500/month at moderate traffic.

---

## 11. Antipatterns

### 11.1 Antipattern 1: Not using edge runtime

```typescript
// ❌ Serverless function — slower cold start
export const runtime = "nodejs";

// ✅ Edge runtime — <50ms cold start
export const runtime = "edge";
```

### 11.2 Antipattern 2: Streaming in server action but not in API route

```typescript
// ❌ Server action with non-streaming
"use server";
export async function chat(message: string) {
    const result = await generateText({ ... });
    return result.text;  // user waits for full response
}

// ✅ API route with streaming
export async function POST(req: Request) {
    const result = await streamText({ ... });
    return result.toDataStreamResponse();
}
```

### 11.3 Antipattern 3: Loading entire message history in every request

```typescript
// ❌ Pass full history every request — expensive
const result = await streamText({
    messages: allMessages,  // grows unboundedly
});

// ✅ Trim history or summarize
const recentMessages = allMessages.slice(-10);
const summary = await generateText({
    model: openai("gpt-4o-mini"),
    prompt: `Summarize this conversation: ${JSON.stringify(allMessages)}`,
});
const result = await streamText({
    messages: [
        { role: "system", content: `Previous conversation summary: ${summary.text}` },
        ...recentMessages,
    ],
});
```

### 11.4 Antipattern 4: Exposing API keys to client

```typescript
// ❌ API key in client bundle
const client = new OpenAI({ apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY });
// This bundle is shipped to the browser — the key is exposed

// ✅ API calls via server route or server action
// Client never sees the key
```

### 11.5 Antipattern 5: No error boundaries

```typescript
// ❌ App crashes on LLM error
const result = await streamText({ ... });
return result.toDataStreamResponse();

// ✅ Wrap in try/catch, return error response
export async function POST(req: Request) {
    try {
        const result = await streamText({ ... });
        return result.toDataStreamResponse();
    } catch (e) {
        console.error("LLM call failed", e);
        return new Response("Internal error", { status: 500 });
    }
}
```

---

## 🎯 Key Takeaways

- Vercel AI SDK + Next.js 15 is the canonical 2026 AI frontend stack.
- `useChat` for chat, `useObject` for structured output, `streamText` server-side.
- Tool calling with Zod schemas; the LLM decides which tool to invoke.
- Multi-modal with `image` content type; works with GPT-4o vision.
- Server Actions for server-side LLM calls without exposing API keys.
- Edge runtime for <50ms cold start.
- Avoid exposing API keys to client, not streaming, no error handling, full history.
- Deploy to Vercel for free preview + edge.

## References

- Vercel AI SDK — [sdk.vercel.ai/docs](https://sdk.vercel.ai/docs)
- Next.js 15 — [nextjs.org/docs](https://nextjs.org/docs)
- React Server Components — [react.dev/reference/rsc/server-components](https://react.dev/reference/rsc/server-components)
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing|Cloud Computing]] — Vercel deployment
- [[06 - Large Language Models/22 - Instructor and Structured Generation|Note — Instructor for Python]]
- [[17 - TypeScript Engineering/01 - TypeScript for AI Engineers/01 - TypeScript Fundamentals|Note 01 — TS Fundamentals]]
- [[17 - TypeScript Engineering/01 - TypeScript for AI Engineers/04 - LangChain.js + AI SDK Integration|Note 04 — LangChain.js]]