# ⚙️ Async Engine and Sessions

## 🎯 Learning Objectives

- Understand why SQLAlchemy 2.0 introduced a fully async core and what changed at the architectural level
- Use `create_async_engine` correctly with production-grade pooling configuration
- Master the `AsyncSession` lifecycle: open, transaction, commit/rollback, close
- Avoid the seven most common async session pitfalls that break FastAPI services in production

## Introduction

A FastAPI request handler is a coroutine. When that handler needs to query PostgreSQL, the I/O it issues must not block the event loop. SQLAlchemy 1.4 introduced async support as a thin compatibility shim; SQLAlchemy 2.0 rebuilt the async engine from the ground up so that the entire ORM — sessions, queries, transactions, relationship loading — is natively awaitable.

The mental shift is small in code but large in consequences. In sync SQLAlchemy, you call `session.execute(stmt)` and get a `Result` back synchronously. In async SQLAlchemy 2.0, the same call returns a coroutine; the `Result` is the awaited value. The wrong pattern — calling a sync method inside an async handler without `run_in_executor` — silently serializes all DB calls and destroys the throughput of your service. The right pattern is fully awaitable end to end, which is the only way to interleave hundreds of concurrent requests on a single event loop.

This note covers the foundation: the engine, the connection pool, and the session. Models and migrations are covered in later notes.

---

## 1. The Engine: From URL to Connection Pool

### 1.1 What `create_async_engine` actually does

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@host:5432/dbname",
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
)
```

Calling `create_async_engine` does **not** open any database connection. It constructs a configuration object and, lazily, a connection pool. The first connection is opened the first time the pool needs to satisfy a request — typically the first query from the first request handler.

This is intentional. A web server can start, accept health checks, and become ready before the database is reachable. If the engine opened connections eagerly, the service would crash-loop at startup and Kubernetes would mark it as `CrashLoopBackOff`.

### 1.2 The driver is async, the DBAPI is sync

The "async" in `postgresql+asyncpg` means SQLAlchemy uses `asyncpg`'s native coroutine API. The DBAPI protocol (PEP 249) is fundamentally synchronous; SQLAlchemy 2.0 works around this by implementing its own bridge that is awaitable end to end but uses `asyncpg` underneath. The alternative drivers (`aiohttp`-based, `aiopg`) are deprecated or unmaintained. Use `asyncpg` for PostgreSQL.

```python
# ❌ Wrong: blocking the event loop
import psycopg2  # synchronous driver
psycopg2.connect(dsn)  # blocks the entire event loop

# ✅ Right: async-native driver
from sqlalchemy.ext.asyncio import create_async_engine
engine = create_async_engine("postgresql+asyncpg://...")
```

### 1.3 Connection pool configuration

A connection pool holds a fixed number of physical connections to PostgreSQL. Requests borrow a connection for the duration of their transaction and return it. The defaults are conservative; production services usually need tuning.

| Parameter | Default | Recommended | Why it matters |
|-----------|--------:|------------:|----------------|
| `pool_size` | 5 | 10-50 | Number of persistent connections kept warm |
| `max_overflow` | 10 | 10-20 | Extra connections allowed under burst load |
| `pool_pre_ping` | False | **True** | Tests connection before use, avoids "connection reset" errors after DB restart |
| `pool_recycle` | -1 | 1800-3600 | Recycles connections older than N seconds; works around middleboxes and PostgreSQL `idle_in_transaction_session_timeout` |
| `pool_timeout` | 30 | 10-30 | Seconds to wait for a free connection before raising `TimeoutError` |

💡 **Tip**: `pool_size=20` means at most 20 concurrent queries to the database. If you have 4 FastAPI workers and 20 pool size per worker, PostgreSQL can receive up to 80 concurrent connections. PostgreSQL's default `max_connections` is 100, which leaves almost no headroom for migrations, monitoring, and admin. **Size your pool smaller than `max_connections / workers`** or use PgBouncer as a connection multiplexer.

### 1.4 The `NullPool` and `StaticPool` alternatives

Most services want a real connection pool. Two situations call for alternatives:

```python
# Use NullPool for serverless (Cloud Run, Lambda): open + close per request
from sqlalchemy.pool import NullPool
engine = create_async_engine(url, poolclass=NullPool)

# Use StaticPool for SQLite in-memory tests: one shared connection
from sqlalchemy.pool import StaticPool
engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
```

Serverless platforms scale workers to zero between requests; keeping connections warm wastes money and hits cold-start latency. NullPool opens a fresh connection per transaction and disposes of it. The cost is higher per-request latency; the benefit is no idle connection cost and no state to clean up.

---

## 2. The `AsyncSession`: Unit of Work for Async Code

### 2.1 What a session represents

An `AsyncSession` is a **unit of work** — a staging area where you build up a set of changes (inserts, updates, deletes) and commit them atomically. It is **not** a connection. It acquires a connection from the pool only when it needs to talk to the database, and releases it back to the pool when the transaction ends.

```python
from sqlalchemy.ext.asyncio import AsyncSession

async with AsyncSession(engine) as session:
    stmt = select(User).where(User.id == 42)
    result = await session.execute(stmt)
    user = result.scalar_one()
```

The `async with` block guarantees the session is closed when the block exits, even on exception. The connection is acquired lazily, used for the `execute`, and returned to the pool when the context manager exits.

### 2.2 `async_sessionmaker`: the factory pattern

You should not call `AsyncSession(engine)` directly in application code. The factory pattern centralizes session configuration:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)
```

The three flags matter:

| Flag | Default | What it does | Why you usually override |
|------|---------|--------------|--------------------------|
| `expire_on_commit` | True | After commit, all ORM objects become "expired" — accessing any attribute triggers a refresh query | In FastAPI handlers, you often return ORM objects to the response layer after commit. With `expire_on_commit=True`, Pydantic's serialization triggers lazy loads that fail because the session is already closed. **Set to `False`.** |
| `autoflush` | True | Automatically flushes pending changes to the DB before any `execute` | Surprising behavior in tests. Set to `False` and call `session.flush()` explicitly when needed. |
| `autocommit` | False | (Deprecated, but documented) Old-style auto-commit mode | Keep at `False`; always commit explicitly. |

```python
# ❌ Wrong: classic FastAPI bug — Pydantic tries to serialize expired attributes
async def get_user(user_id: int):
    async with SessionLocal() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalar_one()
        await session.commit()  # expires all attributes
    return user  # Pydantic tries to read user.email → triggers a query → session closed → crash

# ✅ Right: expire_on_commit=False
async def get_user(user_id: int):
    async with SessionLocal() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalar_one()
    return user  # session closed cleanly, attributes still accessible
```

### 2.3 The `session.execute` contract

`session.execute(stmt)` returns a coroutine that resolves to a `Result` object. The result is **not** a list of ORM objects — it is a cursor-like object that supports iteration, `scalar()`, `scalars()`, `one()`, `one_or_none()`, `scalar_one()`, and `scalar_one_or_none()`.

```python
result = await session.execute(select(User))
users = result.scalars().all()          # List[User]
user = result.scalar_one()              # exactly one row, raises if 0 or >1
user = result.scalar_one_or_none()      # 0 or 1 row, None if 0
user = result.first()                   # first row or None
```

The 2.0 style strongly prefers `select()` over the legacy `Query` API. The `Query` API is still available but is in maintenance mode and lacks static type checking.

### 2.4 Transactions: `begin`, `commit`, `rollback`

There are two ways to manage transactions in 2.0:

```python
# Style 1: explicit begin / commit
async with engine.begin() as conn:
    await conn.execute(insert(User).values(name="alice"))

# Style 2: implicit transaction inside a session
async with SessionLocal() as session:
    async with session.begin():
        await session.execute(insert(User).values(name="alice"))
    # committed when the begin block exits cleanly
```

The session-managed style is the default in FastAPI handlers. The session opens a transaction lazily on the first query. The transaction commits when you call `await session.commit()` and rolls back automatically if the `async with` block exits with an exception.

```python
async with SessionLocal() as session:
    try:
        user = User(name="alice", email="alice@example.com")
        session.add(user)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise
```

---

## 3. The Seven Async Session Pitfalls

### 3.1 Calling sync code from async context

```python
# ❌ Wrong: blocks the event loop for 200ms during the query
async def get_user(user_id: int):
    user = SessionLocal().query(User).filter(User.id == user_id).first()
    return user
```

The legacy `Query` API is synchronous. Calling it inside an async handler blocks the entire event loop. Use 2.0's `select()` and `await session.execute(stmt)`.

### 3.2 Lazy loading after session close

```python
# ❌ Wrong: triggers a query after the session is closed
async with SessionLocal() as session:
    user = (await session.execute(select(User).where(User.id == 1))).scalar_one()
print(user.profile.bio)  # DetachedInstanceError: lazy load failed
```

Solutions:

```python
# ✅ Option 1: eager-load before the session closes
from sqlalchemy.orm import selectinload
user = (await session.execute(
    select(User).where(User.id == 1).options(selectinload(User.profile))
)).scalar_one()
print(user.profile.bio)  # works, already loaded

# ✅ Option 2: keep the session open until the consumer is done (rare in FastAPI)
```

### 3.3 Sharing a session across requests

```python
# ❌ Wrong: module-level singleton session
session = SessionLocal()  # never do this in a web app
```

Each request gets its own session. The session holds a connection from the pool; sharing a session across requests serializes all of them on one connection. The proper pattern is `Depends(get_session)` (covered in [[05 - FastAPI Integration|the next-but-two note]]).

### 3.4 Forgetting `await session.close()`

```python
# ❌ Wrong: leaks connection back to pool only on GC
session = SessionLocal()
result = await session.execute(stmt)
return result
```

Use `async with` — it guarantees the session is closed and the connection is returned to the pool.

### 3.5 Mixing sync and async drivers

```python
# ❌ Wrong: async session, sync driver
engine = create_async_engine("postgresql+psycopg2://...")  # raises immediately
```

`create_async_engine` requires an async driver (`asyncpg`, `aiosqlite`). Using a sync driver URL raises `InvalidRequestError` at engine creation.

### 3.6 Catching broad exceptions that swallow rollback

```python
# ❌ Wrong: silent rollback
try:
    await session.commit()
except Exception:
    pass  # connection is poisoned, but session still looks healthy

# ✅ Right: re-raise after rollback
try:
    await session.commit()
except Exception:
    await session.rollback()
    raise
```

### 3.7 Long-running transactions

```python
# ❌ Wrong: holding a transaction open for 30 seconds
async with SessionLocal() as session:
    async with session.begin():
        for item in large_list:
            await session.execute(insert(Item).values(**item))
        await asyncio.sleep(30)  # network call, holds the row locks
```

A transaction holds row locks. Long-running transactions block other writers and can deadlock. Break large operations into batches and commit between them.

```python
# ✅ Right: batched commits
async with SessionLocal() as session:
    for batch in chunked(large_list, 1000):
        async with session.begin():
            session.add_all([Item(**item) for item in batch])
        # batch committed, locks released
```

---

## 4. Engine Lifecycle in FastAPI

The engine should outlive any individual request. In FastAPI, the natural place for it is the `lifespan` context manager:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to do, the engine is lazy
    yield
    # Shutdown: dispose of the engine and its pool
    await engine.dispose()

app = FastAPI(lifespan=lifespan)
```

Without `engine.dispose()`, the process hangs for up to 30 seconds during shutdown because the pool is waiting for connections to be released. The `dispose()` call cancels in-flight queries and closes all connections cleanly.

---

## 5. Observability

### 5.1 SQL echo

```python
engine = create_async_engine(url, echo=True)
```

Prints every SQL statement with parameters. Invaluable for debugging; deadly in production (overhead, PII in logs, log volume). Use `echo="debug"` only in dev or temporarily in prod with log sampling.

### 5.2 Prometheus integration

```python
from prometheus_client import Counter, Histogram

query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    labelnames=["model", "operation"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

# Wrap with an event listener
from sqlalchemy import event

@event.listens_for(engine.sync_engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start = time.monotonic()

@event.listens_for(engine.sync_engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    duration = time.monotonic() - context._query_start
    model = statement.split("FROM", 1)[1].split()[0] if "FROM" in statement else "unknown"
    operation = statement.split()[0]
    query_duration.labels(model=model, operation=operation).observe(duration)
```

The `engine.sync_engine` attribute exposes the underlying sync engine so SQLAlchemy event listeners (which are themselves sync) can attach.

---

## 6. Código de Compresión

```python
"""
Compresión: Async Engine and Sessions
Cubre: create_async_engine, async_sessionmaker, AsyncSession lifecycle,
       expire_on_commit, lazy-load pitfalls, lifespan integration.
"""
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str]


# 1) Engine: lazy, pool, pre-ping
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/demo",
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,        # detect dead connections
    pool_recycle=1800,          # recycle every 30 min
    echo=False,                 # set True for dev
)

# 2) Session factory: expire_on_commit=False is the FastAPI default
SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,     # critical for Pydantic response serialization
    autoflush=False,
)

# 3) Handler pattern
async def get_user_by_id(user_id: int) -> User | None:
    async with SessionLocal() as session:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

# 4) Transactional pattern
async def create_user(name: str, email: str) -> User:
    async with SessionLocal() as session:
        async with session.begin():
            user = User(name=name, email=email)
            session.add(user)
        # committed here, expire_on_commit=False keeps attributes accessible
    return user

# 5) Lifespan integration
@asynccontextmanager
async def lifespan(app):
    # startup: nothing to do, engine is lazy
    yield
    # shutdown: dispose to close pool cleanly
    await engine.dispose()

# 6) Common error: lazy load after session close
async def lazy_load_pitfall():
    async with SessionLocal() as session:
        user = (await session.execute(select(User))).scalar_one()
        # ❌ accessing user.email AFTER the async with block raises DetachedInstanceError
    # print(user.email)  # would crash

# 7) Demo
async def main():
    user = await create_user("alice", "alice@example.com")
    print(f"Created: {user.name} <{user.email}> (id={user.id})")
    found = await get_user_by_id(user.id)
    print(f"Found: {found}")
    await engine.dispose()

# asyncio.run(main())
```

---

## Key Takeaways

- `create_async_engine` is lazy; no connection is opened until the first query. This is a feature, not a bug — it lets services start before the DB is reachable.
- Use `asyncpg` for PostgreSQL. Sync drivers (`psycopg2`) block the event loop and silently destroy throughput.
- `expire_on_commit=False` is the FastAPI default. With `True`, Pydantic serialization after commit triggers lazy loads that fail because the session is already closed.
- Pool sizing matters: `pool_size * workers < max_connections` or you will exhaust PostgreSQL's connection limit under load.
- Always use `async with` for sessions. Always call `engine.dispose()` in shutdown. Both prevent connection leaks.
- Eager loading (`selectinload`, `joinedload`) is the only way to safely access relationships after the session closes. Lazy loading from a closed session is the most common async SQLAlchemy bug.

## References

- [SQLAlchemy 2.0 Async I/O (AsyncSession) — official docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [create_async_engine API reference](https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.ext.asyncio.create_async_engine)
- [Pooling documentation](https://docs.sqlalchemy.org/en/20/core/pooling.html)
- [asyncpg — async PostgreSQL driver](https://magicstack.github.io/asyncpg/current/)
- [Connection pool sizing and PgBouncer — Citus Data engineering blog](https://www.citusdata.com/blog/2017/11/30/connection-pooling/)
