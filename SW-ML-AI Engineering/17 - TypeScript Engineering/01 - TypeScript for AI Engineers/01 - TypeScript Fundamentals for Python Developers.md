# 🎯 01 - TypeScript Fundamentals for Python Developers

> **The 90-minute TypeScript primer. From `let x = 5` to async generics. The minimum you need to build AI apps.**

## 🎯 Learning Objectives
- Translate Python patterns to TypeScript (variables, functions, classes)
- Use TypeScript's static type system for runtime safety
- Master async/await (same model as Python, different syntax)
- Understand the module system (npm vs pip)
- Use Zod for runtime validation (Python's Pydantic equivalent)
- Build and test a small library with `tsc`, `vitest`, `tsx`

## Introduction

If you know Python, you can learn TypeScript syntax in 90 minutes. The hardest part is **the type system** — Python is dynamically typed, TypeScript is statically typed. The benefit: catching bugs at compile time, not runtime.

This note is the minimum TypeScript you need for AI work. By the end, you'll be productive.

---

## 1. Variables and Types

```typescript
// TypeScript is statically typed
let x: number = 5;
let name: string = "Alice";
let isActive: boolean = true;

// Type inference: TypeScript infers the type
let count = 10;  // inferred as number
let greeting = "Hello";  // inferred as string

// Any and Unknown
let anything: any = "could be anything";  // avoid `any`
let maybeValue: unknown = JSON.parse(data);  // must check before use

// Arrays and tuples
let numbers: number[] = [1, 2, 3];
let tuple: [string, number] = ["Alice", 30];  // fixed length and types

// Objects with type
interface User {
    id: number;
    name: string;
    email?: string;  // optional
}

let user: User = { id: 1, name: "Alice" };

// Literal types
type Role = "admin" | "user" | "guest";
let role: Role = "admin";
```

Compare Python:
```python
# Python
x = 5
name = "Alice"
is_active = True
user = {"id": 1, "name": "Alice"}
```

Same syntax, but TypeScript catches type errors at compile time.

---

## 2. Functions

```typescript
// Function with typed parameters and return
function add(a: number, b: number): number {
    return a + b;
}

// Arrow function (Python lambda equivalent, but more powerful)
const multiply = (a: number, b: number): number => a * b;

// Optional parameters
function greet(name: string, greeting: string = "Hello"): string {
    return `${greeting}, ${name}`;
}

// Async function (returns Promise<T>)
async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`https://api.example.com/users/${id}`);
    return response.json() as Promise<User>;
}

// Generic function (like Python TypeVar)
function identity<T>(value: T): T {
    return value;
}

const num = identity<number>(42);  // T inferred as number
const str = identity<string>("hello");
```

Compare Python:
```python
# Python
def add(a: int, b: int) -> int:
    return a + b

async def fetch_user(id: int) -> User:
    response = await fetch(...)
    return response.json()
```

Identical except `Promise<T>` instead of just `T` for async returns.

---

## 3. Async/Await

```typescript
// Promise-based async (like asyncio)
async function fetchAll(urls: string[]): Promise<string[]> {
    const promises = urls.map(url => fetch(url).then(r => r.text()));
    return Promise.all(promises);
}

// More readable with await in a loop
async function fetchAllSeq(urls: string[]): Promise<string[]> {
    const results: string[] = [];
    for (const url of urls) {
        const response = await fetch(url);
        results.push(await response.text());
    }
    return results;
}

// Parallel with bounded concurrency
async function fetchBounded(urls: string[], max: number): Promise<string[]> {
    const results: string[] = new Array(urls.length);
    let index = 0;
    
    async function worker(): Promise<void> {
        while (index < urls.length) {
            const i = index++;
            const response = await fetch(urls[i]);
            results[i] = await response.text();
        }
    }
    
    const workers = Array(max).fill(null).map(() => worker());
    await Promise.all(workers);
    return results;
}
```

Compare Python:
```python
# Python
async def fetch_all(urls: list[str]) -> list[str]:
    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)

async def fetch_bounded(urls: list[str], max: int) -> list[str]:
    sem = asyncio.Semaphore(max)
    async def worker():
        while True:
            async with sem:
                # ...
    workers = [worker() for _ in range(max)]
    await asyncio.gather(*workers)
```

Same pattern, slightly different syntax.

---

## 4. Generics — TypeScript's Killer Feature

```typescript
// Generic function
function first<T>(arr: T[]): T | undefined {
    return arr[0];
}

const num = first([1, 2, 3]);  // type: number | undefined
const str = first(["a", "b"]);  // type: string | undefined

// Generic interface
interface ApiResponse<T> {
    data: T;
    status: number;
    timestamp: string;
}

const userResponse: ApiResponse<User> = await fetchUser(123);

// Generic class
class State<T> {
    private value: T | undefined;
    
    set(value: T): void {
        this.value = value;
    }
    
    get(): T | undefined {
        return this.value;
    }
}

const stringState = new State<string>();
stringState.set("hello");

// Generic constraints
interface HasId {
    id: number;
}

function getById<T extends HasId>(items: T[], id: number): T | undefined {
    return items.find(item => item.id === id);
}
```

Generics let you write reusable code that's still type-safe.

---

## 5. Modules and Imports

```typescript
// types.ts
export interface User {
    id: number;
    name: string;
}

export type Role = "admin" | "user" | "guest";

// userService.ts
import { User, Role } from "./types";

export async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    return response.json() as Promise<User>;
}

// app.ts
import { fetchUser, type User } from "./userService";

const user: User = await fetchUser(123);
```

Compare Python:
```python
# types.py
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str

# user_service.py
from typing import Literal
import aiohttp

async def fetch_user(user_id: int) -> User:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/users/{user_id}") as response:
            return await response.json()

# app.py
from user_service import fetch_user
```

Same structure.

---

## 6. Zod — Runtime Validation (Pydantic for TS)

```typescript
import { z } from "zod";

// Define schema
const UserSchema = z.object({
    id: z.number().int().positive(),
    name: z.string().min(1).max(100),
    email: z.string().email().optional(),
    age: z.number().int().min(0).max(150),
    role: z.enum(["admin", "user", "guest"]),
});

// Infer type from schema (mirrors Pydantic)
type User = z.infer<typeof UserSchema>;

// Validate data
const result = UserSchema.safeParse({
    id: 1,
    name: "Alice",
    email: "alice@example.com",
    age: 30,
    role: "admin",
});

if (result.success) {
    console.log(result.data);  // type-safe User
} else {
    console.error(result.error.format());
}
```

Zod + `z.infer` is the canonical Pydantic equivalent. Use it for LLM structured outputs.

---

## 7. LLM Structured Output with Zod

```typescript
import { z } from "zod";
import OpenAI from "openai";

const PersonSchema = z.object({
    name: z.string(),
    age: z.number().int().positive(),
    role: z.enum(["ML engineer", "data scientist", "PM"]),
});

type Person = z.infer<typeof PersonSchema>;

const client = new OpenAI();

async function extractPerson(text: string): Promise<Person> {
    const response = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
            { role: "system", content: "Extract person info from the text." },
            { role: "user", content: text },
        ],
        response_format: {
            type: "json_schema",
            json_schema: {
                name: "person",
                schema: zodToJsonSchema(PersonSchema),
            },
        },
    });
    
    const content = response.choices[0].message.content;
    if (!content) throw new Error("No content");
    
    return PersonSchema.parse(JSON.parse(content));
}
```

`zodToJsonSchema` from `zod-to-json-schema` converts Zod to OpenAI's JSON schema format.

---

## 8. The Build System

```bash
# TypeScript compiler
tsc app.ts  # transpiles to JavaScript

# tsx (TypeScript Execute) — runs TS directly
npx tsx app.ts

# Hot reload (like uvicorn --reload for FastAPI)
npx tsx watch app.ts

# Vitest (testing)
npx vitest
```

The typical project structure:
```
my-ai-app/
├── src/
│   ├── app.ts
│   ├── services/
│   ├── types/
│   └── utils/
├── package.json
├── tsconfig.json
└── tests/
```

`package.json` is like `pyproject.toml`. `npm install` is like `pip install -r requirements.txt`.

---

## 9. The tsconfig.json

```json
{
    "compilerOptions": {
        "target": "ES2022",
        "module": "ESNext",
        "moduleResolution": "bundler",
        "strict": true,
        "esModuleInterop": true,
        "skipLibCheck": true,
        "noImplicitAny": true,
        "strictNullChecks": true,
        "noUncheckedIndexedAccess": true,
        "outDir": "./dist",
        "rootDir": "./src"
    },
    "include": ["src/**/*", "tests/**/*"],
    "exclude": ["node_modules"]
}
```

`"strict": true` enables all strict type checks. Use this in every project.

---

## 10. Antipatterns

### 10.1 Antipattern 1: Using `any`

```typescript
// ❌ Defeats the purpose of TypeScript
function process(data: any): any {
    return data.anything;
}

// ❌ Better: use unknown
function process(data: unknown): User {
    if (typeof data === "object" && data !== null) {
        // type-check first
    }
    throw new Error("Invalid data");
}
```

### 10.2 Antipattern 2: Ignoring errors

```typescript
// ❌ Catch and ignore
try {
    await fetch(url);
} catch (e) {
    // silent
}

// ✅ Handle or re-throw
try {
    await fetch(url);
} catch (e) {
    console.error("Fetch failed", e);
    throw e;
}
```

### 10.3 Antipattern 3: `==` instead of `===`

```typescript
// ❌ Type coercion
if (x == "5") { ... }

// ✅ Strict equality
if (x === 5) { ... }
```

### 10.4 Antipattern 4: Using any in third-party types

```typescript
// ❌ Accept any from third-party SDK
import OpenAI from "openai";
const response: any = await openai.chat.completions.create(...);

// ✅ Use the SDK's types
const response: OpenAI.Chat.ChatCompletion = await openai.chat.completions.create(...);
```

### 10.5 Antipattern 5: Not using strict mode

```json
// ❌ tsconfig.json without strict
{
    "compilerOptions": {
        "target": "ES2022"
    }
}

// ✅ Always use strict
{
    "compilerOptions": {
        "target": "ES2022",
        "strict": true
    }
}
```

---

## 🎯 Key Takeaways

- TypeScript is JavaScript with static types. Same syntax + type annotations.
- Use generics for reusable type-safe code.
- `async`/`await` works the same as Python; `Promise<T>` instead of `T`.
- Zod is the canonical runtime validation; mirrors Pydantic.
- `"strict": true` in tsconfig catches bugs at compile time.
- Avoid `any`; use `unknown` and type-narrow.
- Use the SDK's types instead of `any` for third-party libraries.
- Test with vitest; run with tsx.

## References

- TypeScript Handbook — [typescriptlang.org/docs/handbook/intro.html](https://www.typescriptlang.org/docs/handbook/intro.html)
- Zod — [zod.dev](https://zod.dev)
- Vercel AI SDK — [sdk.vercel.ai/docs](https://sdk.vercel.ai/docs)
- LangChain.js — [js.langchain.com](https://js.langchain.com)
- [[03 - Advanced Python/08 - Async Python Patterns Reference|Note — Async Python Patterns]]
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|Note — FastAPI for ML]]