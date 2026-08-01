# 🎯 04 - LangChain.js + AI SDK Integration — Type-Safe LLM Orchestration

> **The TypeScript port of LangChain. Chains, agents, RAG, and tools with full type safety. The same patterns as Python LangChain, but compile-time error checking.**

## 🎯 Learning Objectives
- Use LangChain.js with TypeScript types for LLM calls
- Build chains with `RunnableSequence` (the JS equivalent of Python `|`)
- Implement agents with tool calling and type-safe tool definitions
- Build RAG pipelines with LangChain.js + a vector DB
- Use the OpenAI SDK directly with TypeScript types
- Apply the patterns to your portfolio (StayBot, MARS, LLM Edge Gateway)

## Introduction

**LangChain.js** is the 1:1 TypeScript port of Python LangChain. Same primitives, same patterns, but with TypeScript's type safety. The OpenAI Node SDK is the official client with full type coverage.

This note covers both:

1. **OpenAI SDK** — direct calls, structured outputs, function calling
2. **LangChain.js** — chains, agents, RAG, retrievers

By the end, you'll be productive with TypeScript AI development.

---

## 1. The OpenAI Node SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Simple chat
const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is TypeScript?" },
    ],
});

console.log(response.choices[0].message.content);
```

### 1.1 Streaming

```typescript
const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Tell me a story" }],
    stream: true,
});

for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) process.stdout.write(content);
}
```

### 1.2 Function calling

```typescript
import OpenAI from "openai";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const GetWeatherSchema = z.object({
    location: z.string().describe("City name"),
});

const tools = [
    {
        type: "function" as const,
        function: {
            name: "get_weather",
            description: "Get the current weather for a location",
            parameters: zodToJsonSchema(GetWeatherSchema),
        },
    },
];

const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "What's the weather in Paris?" }],
    tools,
});

const toolCall = response.choices[0].message.tool_calls?.[0];
if (toolCall) {
    const args = JSON.parse(toolCall.function.arguments);
    console.log("Tool called:", toolCall.function.name, args);
}
```

### 1.3 Structured outputs

```typescript
const completion = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Maria is 28 and works on LLMs." }],
    response_format: {
        type: "json_schema",
        json_schema: {
            name: "person",
            schema: zodToJsonSchema(
                z.object({
                    name: z.string(),
                    age: z.number().int(),
                    role: z.string(),
                })
            ),
        },
    },
});

const person = JSON.parse(completion.choices[0].message.content!);
```

The OpenAI SDK includes full types for responses, so `person.role` is type-safe.

---

## 2. LangChain.js — The Basics

```bash
npm install langchain @langchain/openai @langchain/core
```

### 2.1 Chat models

```typescript
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
});

const response = await llm.invoke([
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is TypeScript?" },
]);

console.log(response.content);
```

### 2.2 Prompt templates

```typescript
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are an expert in {domain}."],
    ["user", "{question}"],
]);

const formattedPrompt = await prompt.formatMessages({
    domain: "TypeScript",
    question: "What are generics?",
});

const response = await llm.invoke(formattedPrompt);
```

### 2.3 Output parsers

```typescript
import { z } from "zod";
import { StructuredOutputParser } from "langchain/output_parsers";

const parser = StructuredOutputParser.fromZodSchema(
    z.object({
        name: z.string(),
        age: z.number(),
    })
);

const formatInstructions = parser.getFormatInstructions();

const response = await llm.invoke(
    `Extract person info from: Maria is 28.\n${formatInstructions}`
);

const person = await parser.parse(response.content as string);
```

### 2.4 Chains

```typescript
import { RunnableSequence } from "@langchain/core/runnables";

const chain = RunnableSequence.from([
    prompt,
    llm,
    parser,
]);

const result = await chain.invoke({
    domain: "TypeScript",
    question: "What are generics?",
});
```

This is the TypeScript equivalent of Python's `prompt | llm | parser`.

---

## 3. RAG with LangChain.js

### 3.1 Basic RAG pipeline

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough } from "@langchain/core/runnables";

// 1. Create documents
const docs = [
    new Document({
        pageContent: "TypeScript adds static types to JavaScript.",
        metadata: { source: "typescript-intro.md" },
    }),
    new Document({
        pageContent: "Generics enable reusable type-safe code.",
        metadata: { source: "typescript-generics.md" },
    }),
];

// 2. Create embeddings and vector store
const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
});

const vectorStore = await MemoryVectorStore.fromDocuments(
    docs,
    embeddings
);

const retriever = vectorStore.asRetriever({ k: 3 });

// 3. Build RAG chain
const ragPrompt = ChatPromptTemplate.fromMessages([
    ["system", "Answer based on the context:\n\n{context}"],
    ["user", "{question}"],
]);

const formatDocs = (docs: Document[]) => docs.map(d => d.pageContent).join("\n\n");

const ragChain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocs),
        question: new RunnablePassthrough(),
    },
    ragPrompt,
    llm,
    new StringOutputParser(),
]);

// 4. Query
const answer = await ragChain.invoke("What are generics?");
console.log(answer);
```

### 3.2 Using a real vector DB (Qdrant)

```typescript
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantClient } from "@qdrant/js-client-rest";

const client = new QdrantClient({ url: "http://localhost:6333" });
const embeddings = new OpenAIEmbeddings();

const vectorStore = new QdrantVectorStore(embeddings, {
    client,
    collectionName: "documents",
});

const retriever = vectorStore.asRetriever({ k: 5 });
```

---

## 4. Agents with Tool Calling

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import * as fs from "fs/promises";

const readFile = tool(
    async ({ path }: { path: string }) => {
        return await fs.readFile(path, "utf-8");
    },
    {
        name: "read_file",
        description: "Read a file from the filesystem",
        schema: z.object({
            path: z.string().describe("File path to read"),
        }),
    }
);

const searchWeb = tool(
    async ({ query }: { query: string }) => {
        const response = await fetch(`https://api.example.com/search?q=${query}`);
        return response.json();
    },
    {
        name: "search_web",
        description: "Search the web for information",
        schema: z.object({
            query: z.string().describe("Search query"),
        }),
    }
);

const llm = new ChatOpenAI({ model: "gpt-4o" });

const agent = createReactAgent({
    llm,
    tools: [readFile, searchWeb],
});

const result = await agent.invoke({
    messages: [{ role: "user", content: "Read the README.md and search for TypeScript best practices" }],
});

console.log(result.messages[result.messages.length - 1].content);
```

The agent uses LangGraph under the hood (covered in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|07/18 LangGraph]]). The TypeScript port is 1:1 with Python.

---

## 5. Streaming with LangChain.js

```typescript
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

const stream = await llm.stream([
    { role: "user", content: "Tell me a story" },
]);

for await (const chunk of stream) {
    process.stdout.write(chunk.content as string);
}
```

### 5.1 Streaming in FastAPI-style with Vercel AI SDK

```typescript
// app/api/chat/route.ts
import { ChatOpenAI } from "@langchain/openai";
import { LangChainAdapter } from "ai/langchain";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

export async function POST(req: Request) {
    const { messages } = await req.json();
    
    const stream = await llm.stream(messages);
    
    return LangChainAdapter.toDataStreamResponse(stream);
}
```

The LangChain adapter converts LangChain streams to Vercel AI SDK format — drop-in compatible.

---

## 6. The `useChat` Hook with LangChain

```typescript
// app/api/chat/route.ts
import { ChatOpenAI } from "@langchain/openai";
import { LangChainAdapter } from "ai/langchain";

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

export async function POST(req: Request) {
    const { messages } = await req.json();
    
    const stream = await model.stream(messages);
    return LangChainAdapter.toDataStreamResponse(stream);
}
```

Client unchanged — `useChat` from the Vercel AI SDK works with any backend that streams in the AI SDK format.

---

## 7. Type-Safe Tool Definitions

The killer feature of LangChain.js + Zod:

```typescript
import { DynamicStructuredTool } from "@langchain/core/tools";

class CalculatorTool extends DynamicStructuredTool<{ a: number; b: number; op: "+" | "-" | "*" | "/" }> {
    schema = z.object({
        a: z.number(),
        b: z.number(),
        op: z.enum(["+", "-", "*", "/"]),
    });
    
    name = "calculator";
    description = "Perform basic arithmetic";
    
    async call({ a, b, op }: { a: number; b: number; op: string }) {
        switch (op) {
            case "+": return a + b;
            case "-": return a - b;
            case "*": return a * b;
            case "/": return a / b;
        }
    }
}

const tool = new CalculatorTool();

// Type-safe: call.arguments is typed
const result = await tool.call({ a: 5, b: 3, op: "+" });
// result: number
```

The LLM cannot pass invalid arguments because the schema enforces them.

---

## 8. Your Portfolio — TypeScript UI for the LLM Edge Gateway

```typescript
// app/page.tsx
"use client";

import { useChat } from "ai/react";

export default function LLMEdgeGatewayUI() {
    const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
        api: process.env.NEXT_PUBLIC_LLM_GATEWAY_URL!,
        headers: {
            "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
    });
    
    return (
        <div>
            <h1>LLM Edge Gateway</h1>
            {messages.map(m => (
                <div key={m.id}>
                    {m.role}: {m.content}
                </div>
            ))}
            
            <form onSubmit={handleSubmit}>
                <input value={input} onChange={handleInputChange} />
                <button type="submit" disabled={isLoading}>Send</button>
            </form>
        </div>
    );
}
```

Your Python LLM Edge Gateway (covered in [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|06/19 LLM Gateway]]) gets a TypeScript UI.

---

## 9. Antipatterns

### 9.1 Antipattern 1: Using `any` for LLM responses

```typescript
// ❌ Loses type safety
const response: any = await openai.chat.completions.create({ ... });
const name = response.choices[0].message.content;  // no autocomplete

// ✅ Use the SDK's types
const response = await openai.chat.completions.create({ ... });
// response.choices[0] is typed as ChatCompletion.Choice
```

### 9.2 Antipattern 2: Skipping Zod for tool definitions

```typescript
// ❌ JSON Schema in code; no runtime validation
const tool = {
    parameters: {
        type: "object",
        properties: {
            location: { type: "string" },
        },
    },
};

// ✅ Use Zod for type-safe, runtime-validated tools
const tool = tool(
    async (input: { location: string }) => { ... },
    {
        name: "get_weather",
        schema: z.object({ location: z.string() }),
    }
);
```

### 9.3 Antipattern 3: Hardcoding model names

```typescript
// ❌ Hard-coded
const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

// ✅ Configurable
const llm = new ChatOpenAI({ 
    model: process.env.LLM_MODEL || "gpt-4o-mini" 
});
```

### 9.4 Antipattern 4: Not handling rate limits

```typescript
// ❌ Fails on rate limit
const response = await openai.chat.completions.create({ ... });

// ✅ Retry with backoff
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (e: any) {
            if (e?.status === 429 && i < maxRetries - 1) {
                await new Promise(r => setTimeout(r, 2 ** i * 1000));
            } else {
                throw e;
            }
        }
    }
    throw new Error("Max retries exceeded");
}

const response = await withRetry(() => 
    openai.chat.completions.create({ ... })
);
```

### 9.5 Antipattern 5: Synchronous LLM calls in async handlers

```typescript
// ❌ Blocks the event loop
async function handler(req: Request) {
    const response = await openai.chat.completions.create({ ... });
    return Response.json(response);
}

// ✅ Use async client methods
async function handler(req: Request) {
    const client = new OpenAI();  // async-native
    const response = await client.chat.completions.create({ ... });
    return Response.json(response);
}
```

---

## 🎯 Key Takeaways

- The OpenAI Node SDK provides full type coverage for all LLM responses.
- LangChain.js is 1:1 with Python LangChain — same chains, agents, RAG.
- Zod + DynamicStructuredTool = type-safe LLM tool definitions.
- `useChat` + LangChainAdapter = drop-in streaming chat UI.
- Streaming with `llm.stream()` and `LangChainAdapter.toDataStreamResponse()`.
- Build RAG with `MemoryVectorStore` (dev) or `QdrantVectorStore` (prod).
- Configure retry with backoff for rate limits.
- Avoid `any` for LLM responses; skip Zod; hardcode models; no rate limit handling.

## References

- LangChain.js — [js.langchain.com](https://js.langchain.com)
- OpenAI Node SDK — [github.com/openai/openai-node](https://github.com/openai/openai-node)
- LangChain LangGraph.js — [langchain-ai.github.io/langgraphjs](https://langchain-ai.github.io/langgraphjs/)
- Zod — [zod.dev](https://zod.dev)
- Vercel AI SDK + LangChain — [sdk.vercel.ai/docs/ai-sdk-ui/chatbot-with-tool-calling](https://sdk.vercel.ai/docs/ai-sdk-ui/chatbot-with-tool-calling)
- [[17 - TypeScript Engineering/01 - TypeScript for AI Engineers/01 - TypeScript Fundamentals|Note 01 — TS Fundamentals]]
- [[17 - TypeScript Engineering/01 - TypeScript for AI Engineers/02 - Next.js + Vercel AI SDK|Note 02 — Next.js]]
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — Python counterpart
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]]