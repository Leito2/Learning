# 🎯 02 - Async Library Ecosystem — Decision Framework for 2026

> **The decision matrix for picking the right async library. HTTP, DB, queues, validation, jobs, locking, pub/sub — the canonical recommendations and the tradeoffs.**

## 🎯 Learning Objectives
- Choose the right async HTTP client (httpx vs aiohttp vs urllib3 async)
- Pick the right async PostgreSQL driver (asyncpg vs psycopg 3 async)
- Select the right Redis client for your workload
- Decide between ARQ, Celery, Dramatiq, and Saq for background jobs
- Choose the right async RabbitMQ/Kafka library for your scale
- Evaluate lock and semaphore libraries (aiocache, aioredlock, asyncio.Lock)
- Recognize when to use stdlib `asyncio` directly vs a third-party library

## Introduction

The async Python ecosystem has matured significantly by 2026. The choice between libraries is no longer about "does it exist?" but "which is the right fit for my workload?". This note gives you the decision framework.

The biggest mistake is choosing based on **popularity** rather than **workload fit**. httpx is more popular than aiohttp, but aiohttp is faster for WebSocket-heavy workloads. asyncpg is the default for PostgreSQL, but psycopg 3's async support is competitive for transaction-heavy workloads. ARQ is the modern default for async-native jobs, but Celery's ecosystem is larger.

This note gives you the **decision matrix** and the **rationale** for each choice.

---

## 1. HTTP Clients

| Library | HTTP/2 | WebSockets | Performance | API style | When to use |
|---------|:------:|:----------:|:-----------:|-----------|-------------|
| **httpx** | ✅ | ✅ | Fast | Sync + Async | Default choice; FastAPI integration |
| **aiohttp** | ❌ | ✅ | Fastest (lowest overhead) | Async only | WebSocket-heavy, max throughput |
| **urllib3 async** | ✅ | ❌ | Comparable | Async only | Existing urllib3 ecosystem |

### 1.1 httpx (the default)

```python
import httpx


# Singleton client (recommended for production)
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=httpx.Timeout(10.0),
    headers={"Authorization": f"Bearer {token}"},
)


async def fetch_users() -> list[dict]:
    response = await client.get("/users")
    response.raise_for_status()
    return response.json()
```

### 1.2 aiohttp (max throughput)

```python
import aiohttp


async def fetch_many(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.text() for r in responses]
```

aiohttp has the **lowest overhead** of any Python HTTP client. For workloads with millions of small requests, it's 30-50% faster than httpx.

### 1.3 Decision

- **httpx**: 95% of cases. Sync + async API, HTTP/2, good defaults.
- **aiohttp**: WebSocket-heavy, max throughput, mature ecosystem.
- **urllib3 async**: Existing urllib3 infrastructure.

---

## 2. PostgreSQL Drivers

| Library | Performance | SQLAlchemy | Type hints | When to use |
|---------|:------------:|:----------:|:----------:|-------------|
| **asyncpg** | Fastest | ✅ (dialect) | Partial | Default for high-throughput RAG/LLM apps |
| **psycopg 3 async** | Comparable | ✅ (dialect) | ✅ Full | Transaction-heavy, full SQL coverage |
| **aiopg** (legacy) | Comparable | ✅ | ❌ | Don't use; psycopg 3 is the replacement |

### 2.1 asyncpg

```python
import asyncpg


async def main():
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="app",
        password="secret",
        database="mydb",
    )
    
    rows = await conn.fetch("SELECT id, name FROM users WHERE active = $1", True)
    async for row in rows:
        print(row["name"])
    
    await conn.close()
```

asyncpg is **2-3× faster** than psycopg 2's psycopg2-binary and competitive with psycopg 3. The protocol is binary (faster than text), the connection pool is battle-tested.

### 2.2 psycopg 3 async

```python
import psycopg


async def main():
    async with await psycopg.AsyncConnection.connect(
        "postgresql://user:pass@localhost/db"
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, name FROM users")
            async for row in cur:
                print(row)
```

psycopg 3 has **full Python type hints** support (e.g., `dict[str, Any]`) and better Postgres-specific feature coverage (e.g., LISTEN/NOTIFY, COPY).

### 2.3 SQLAlchemy 2.0 async

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker


engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)


async def get_users() -> list[User]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        return result.scalars().all()
```

SQLAlchemy 2.0 async is the **right choice** for any non-trivial data model. It supports both asyncpg and psycopg 3 underneath.

### 2.4 Decision

- **asyncpg via SQLAlchemy 2.0 async**: Default for new projects.
- **psycopg 3 via SQLAlchemy 2.0 async**: When you need full SQL type support or psycopg-specific features.
- **Raw asyncpg**: Only if you don't want ORM.
- **Raw psycopg 3 async**: Same — only if no ORM.

---

## 3. Redis Clients

| Library | Performance | Cluster | Pipelining | Type hints | When to use |
|---------|:------------:|:-------:|:----------:|:----------:|-------------|
| **redis.asyncio (redis-py)** | Fast | ✅ | ✅ | ✅ | Default; Redis Cluster support |
| **aioredis (legacy)** | Comparable | ✅ | ✅ | ❌ | Don't use; merged into redis-py |

### 3.1 redis.asyncio (the default)

```python
import redis.asyncio as redis


async def main():
    client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
    )
    
    # Simple operations
    await client.set("key", "value", ex=60)
    value = await client.get("key")
    
    # Pipelining
    async with client.pipeline(transaction=True) as pipe:
        await pipe.set("k1", "v1")
        await pipe.set("k2", "v2")
        await pipe.execute()
    
    # Pub/Sub
    pubsub = client.pubsub()
    await pubsub.subscribe("channel")
    async for message in pubsub.listen():
        if message["type"] == "message":
            print(message["data"])
    
    await client.aclose()
```

### 3.2 Decision

Use `redis.asyncio` (the official redis-py library). aioredis was merged into redis-py in 2021.

---

## 4. Background Job Frameworks

| Library | Broker | Async-native | Performance | Ecosystem | When to use |
|---------|--------|:------------:|:-----------:|-----------|-------------|
| **ARQ** | Redis | ✅ | Fastest | Modern | Default for async-first apps |
| **Dramatiq** | Redis/RabbitMQ | ❌ (threads) | Standard | Medium | Multi-broker needs |
| **Saq** | Redis | Partial | Standard | Small | Simple async jobs |
| **Celery** | Redis/RabbitMQ/SQS | ❌ (threads) | Standard | Largest | Legacy / multi-broker |

### 4.1 ARQ (the default for async-first)

```python
from arq.worker import create_worker
from arq.connections import RedisSettings


async def process_document(ctx, doc_id: str) -> dict:
    """Job handler."""
    doc = await db.get_document(doc_id)
    return {"status": "processed", "doc_id": doc_id}


class WorkerSettings:
    functions = [process_document]
    redis_settings = RedisSettings(host="localhost", port=6379)
```

ARQ is **async-native** (no thread pool). For FastAPI apps, ARQ is the canonical choice.

### 4.2 Decision

- **ARQ**: Default for async-first apps (FastAPI + Redis).
- **Celery**: Legacy apps, multi-broker needs, large ecosystem.
- **Dramatiq**: Multi-broker without Celery complexity.
- **Saq**: Simple async jobs without ARQ's complexity.

---

## 5. Message Queues

### 5.1 RabbitMQ

| Library | Performance | Async | When to use |
|---------|:-----------:|:-----:|-------------|
| **aio-pika** | Standard | ✅ | Default async RabbitMQ client |
| **aiormq** | Standard | ✅ | Lower-level aio-pika alternative |

```python
import aio_pika


async def main():
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
    channel = await connection.channel()
    queue = await channel.declare_queue("events", durable=True)
    
    await channel.default_exchange.publish(
        aio_pika.Message(body=b"hello"),
        routing_key="events",
    )
    
    await connection.close()
```

### 5.2 Kafka

| Library | Performance | Async | When to use |
|---------|:-----------:|:-----:|-------------|
| **aiokafka** | Standard | ✅ | Default async Kafka client |
| **confluent-kafka-python** | Fastest | Partial | Maximum performance; thread-based |

aiokafka is the async-native choice. confluent-kafka-python is faster but uses threads.

---

## 6. Validation and Serialization

### 6.1 Pydantic v2 async

```python
from pydantic import BaseModel, Field


class User(BaseModel):
    id: int
    email: str = Field(pattern=r"^[^@]+@[^@]+$")
    name: str


async def get_user(user_id: int) -> User:
    data = await fetch_from_db(user_id)
    return User.model_validate(data)  # sync, but very fast
```

Pydantic v2 is written in Rust; validation is sync but extremely fast. No need for async validators.

### 6.2 attrs + cattrs

For high-throughput serialization, `attrs` + `cattrs` is faster than Pydantic:

```python
import attrs
import cattrs


@attrs.define
class User:
    id: int
    email: str
    name: str


converter = cattrs.Converter()
data: User = converter.loads(json_str, User)
```

`attrs` + `cattrs` is 3-5× faster than Pydantic for simple cases. Use when validation throughput is critical.

---

## 7. Locking and Distributed Coordination

### 7.1 Single-process: `asyncio.Lock`

```python
lock = asyncio.Lock()


async def critical_section():
    async with lock:
        # Only one coroutine at a time
        ...
```

### 7.2 Distributed: `redis-py` with SET NX EX

```python
async def acquire_lock(key: str, ttl: int = 60) -> bool:
    """Atomic distributed lock."""
    return await redis.set(f"lock:{key}", "1", nx=True, ex=ttl)


async def release_lock(key: str):
    await redis.delete(f"lock:{key}")
```

The SET NX EX pattern is the canonical distributed lock. Use it for:
- Scheduled jobs (only one worker runs)
- Rate limiting
- Idempotency keys

### 7.3 Distributed locks: `python-redis-lock` or `pottery`

For production-grade distributed locks with auto-extension and fencing tokens:

```python
from pottery import Redlock
from redis.asyncio import Redis


redis_client = Redis(host="localhost", port=6379)
redlock = Redlock([redis_client])


async with redlock.lock("my-resource", expire=10):
    # critical section across processes
    ...
```

`pottery.Redlock` is the Python implementation of the Redlock algorithm.

---

## 8. Async Testing Tools

| Tool | Purpose | When to use |
|------|---------|-------------|
| **pytest-asyncio** | Async test support | Every async test suite |
| **httpx.AsyncClient** | Async HTTP testing | FastAPI integration tests |
| **aioresponses** | Mock aiohttp responses | aiohttp-based code |
| **respx** | Mock httpx responses | httpx-based code |
| **pytest-httpx** | pytest plugin for httpx mocking | pytest + httpx |

```python
import pytest
import httpx
from respx import MockRouter


@pytest.mark.asyncio
async def test_with_respx():
    with MockRouter() as mock:
        mock.get("https://api.example.com/users").mock(
            return_value=httpx.Response(200, json={"users": []})
        )
        response = await httpx.AsyncClient().get("https://api.example.com/users")
        assert response.status_code == 200
```

---

## 9. WebSocket Servers

| Library | Performance | FastAPI integration | When to use |
|---------|:-----------:|:-------------------:|-------------|
| **FastAPI native** | Standard | ✅ | Default for FastAPI apps |
| **websockets** | Fast | Manual | High-performance standalone |
| **aiohttp WebSocket** | Standard | Manual | Existing aiohttp code |

```python
from fastapi import FastAPI, WebSocket


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
```

FastAPI's WebSocket support is built on `starlette.websockets`, which uses asyncio.

---

## 10. Async Filesystem Operations

```python
import aiofiles


async def read_file(path: str) -> str:
    async with aiofiles.open(path, "r") as f:
        return await f.read()


async def write_file(path: str, content: str) -> None:
    async with aiofiles.open(path, "w") as f:
        await f.write(content)
```

`aiofiles` provides async file I/O. Most file operations are fast enough to be sync; use `aiofiles` only when blocking matters.

---

## 11. Decision Framework Summary

| Use case | Default | Alternative |
|----------|---------|-------------|
| **HTTP client** | httpx | aiohttp (WebSocket-heavy) |
| **HTTP server** | FastAPI | aiohttp (high-throughput) |
| **PostgreSQL** | asyncpg via SQLAlchemy 2.0 | psycopg 3 via SQLAlchemy 2.0 |
| **Redis** | redis.asyncio | — |
| **Job queue** | ARQ | Celery (legacy) |
| **RabbitMQ** | aio-pika | aiormq |
| **Kafka** | aiokafka | confluent-kafka-python |
| **Validation** | Pydantic v2 | attrs + cattrs |
| **Distributed lock** | Redis SET NX EX | pottery.Redlock |
| **WebSocket server** | FastAPI native | websockets |
| **Files** | aiofiles | stdlib (sync) |
| **Event loop** | uvloop (Linux) | asyncio (Windows) |

---

## 12. Antipatterns

### 12.1 Antipattern 1: Using aiohttp when httpx would do

```python
# ❌ aiohttp for a simple GET
import aiohttp

async def fetch():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as r:
            return await r.json()

# ✅ httpx is simpler for non-WebSocket cases
import httpx

async def fetch():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.example.com")
        return r.json()
```

### 12.2 Antipattern 2: Raw asyncpg without connection pool

```python
# ❌ One connection per request; expensive
async def get_user(user_id):
    conn = await asyncpg.connect(...)
    result = await conn.fetchrow(...)
    await conn.close()
    return result

# ✅ Use connection pool
pool = await asyncpg.create_pool(...)

async def get_user(user_id):
    async with pool.acquire() as conn:
        return await conn.fetchrow(...)
```

### 12.3 Antipattern 3: Redis without connection pool

```python
# ❌ New connection per call
async def get(key):
    r = redis.Redis(...)
    return await r.get(key)

# ✅ Singleton pool
pool = redis.ConnectionPool(host="localhost", port=6379, max_connections=50)

async def get(key):
    async with redis.Redis(connection_pool=pool) as r:
        return await r.get(key)
```

### 12.4 Antipattern 4: Celery when ARQ would do

```python
# ❌ Celery adds significant complexity
from celery import Celery
app = Celery(...)

# ✅ ARQ is simpler for async apps
import arq
```

### 12.5 Antipattern 5: Pydantic in tight loops

```python
# ❌ Pydantic validation in hot path
async def process(data: dict):
    validated = MyModel.model_validate(data)  # 100µs each
    do_work(validated)

# ✅ Validate at boundary, not in hot path
async def process_batch(data_list):
    validated_list = [MyModel.model_validate(d) for d in data_list]
    for v in validated_list:
        do_work(v)
```

---

## 🎯 Key Takeaways

- **httpx** is the default HTTP client; **aiohttp** for max throughput.
- **asyncpg via SQLAlchemy 2.0 async** is the default PostgreSQL stack.
- **redis.asyncio** is the only Redis client; aioredis was merged in.
- **ARQ** is the default for async-first jobs; **Celery** for legacy.
- **Pydantic v2** is the default validation; **attrs + cattrs** for high-throughput.
- Use **uvloop** on Linux/macOS for production; **stdlib asyncio** on Windows.
- Avoid aiohttp when httpx would do, raw asyncpg without pool, Celery when ARQ would do, Pydantic in hot loops.

## References

- httpx — [www.python-httpx.org](https://www.python-httpx.org/)
- aiohttp — [docs.aiohttp.org](https://docs.aiohttp.org/)
- asyncpg — [magicstack.github.io/asyncpg](https://magicstack.github.io/asyncpg/)
- SQLAlchemy 2.0 async — [docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- redis-py — [redis-py.readthedocs.io](https://redis-py.readthedocs.io/)
- ARQ — [arq-docs.helpmanual.io](https://arq-docs.helpmanual.io/)
- aio-pika — [aio-pika.readthedocs.io](https://aio-pika.readthedocs.io/)
- aiokafka — [aiokafka.readthedocs.io](https://aiokafka.readthedocs.io/)
- pottery — [github.com/brainix/pottery](https://github.com/brainix/pottery)
- [[../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/11 - Advanced Async Patterns - Cancellation, Debugging, and Testing|Note 11 — Advanced Async Patterns]]
- [[../../10 - Cloud, Infra y Backend/38 - SQLAlchemy 2.0 Async + Alembic for FastAPI|SQLAlchemy Async]]
- [[../../10 - Cloud, Infra y Backend/40 - Background Jobs and Workers for FastAPI|Background Workers]]
- [[../../10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/12 - Qdrant Python Client Deep Dive/02 - Production Async Patterns - FastAPI, Retries, Batching and Observability|Qdrant Production Async]]