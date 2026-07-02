# 🔌 FastAPI Integration: DI and Lifespan

## 🎯 Learning Objectives

- Wire the async SQLAlchemy engine into FastAPI's lifespan context manager
- Provide an `AsyncSession` per request via `Depends(get_session)`
- Connect the `UnitOfWork` from [[04 - Repository Pattern and Unit of Work|note 04]] to a real HTTP handler
- Handle request-scoped concerns: pre-commit validation, post-commit side effects, exception rollback
- Test the full stack with `httpx.AsyncClient` and an in-memory database

## Introduction

The previous notes built the data layer in isolation: engine, session, models, migrations, repositories. None of it knows about HTTP. This note is the bridge: how the data layer plugs into FastAPI's request lifecycle so that every endpoint gets a clean session, every transaction commits at the right moment, and every connection returns to the pool when the response is sent.

The two FastAPI mechanisms that matter are `lifespan` (process-wide setup and teardown) and `Depends` (per-request injection). The engine lives in the lifespan; the session is created by a dependency and tied to a single request. The pattern is small in code but has subtle interactions with background tasks, error handling, and streaming responses. Get it right once and the rest of the application is straightforward.

---

## 1. The Engine in `lifespan`

### 1.1 The lifespan context manager

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings


engine = create_async_engine(
    str(settings.DATABASE_URL),
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=settings.DEBUG,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
```

The engine is created at **module import time**, not in the lifespan. The lifespan only handles `dispose()`. Why?

- Creating the engine is a pure-Python operation (no DB connection); doing it once at import is fine.
- Importing the module triggers the engine creation. If the engine config is wrong, the error happens at import — at process start, when K8s and humans can see it — rather than on the first request.
- `dispose()` is the only stateful cleanup. Putting it in lifespan guarantees it runs even if the process is killed gracefully.

### 1.2 Why `app.state` is wrong here

Some patterns store the engine on `app.state`:

```python
# ❌ Avoid this
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = create_async_engine(...)
    yield
    await app.state.engine.dispose()
```

The problem: `app.state.engine` is a request-time concept. Code that imports the engine for a CLI script or a background worker cannot use `app.state`. A module-level engine is accessible everywhere — tests, scripts, workers, FastAPI handlers — and the lifespan only handles the dispose.

### 1.3 Engine and configuration

Use Pydantic Settings for the URL:

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    DATABASE_URL: str = "postgresql+asyncpg://localhost/dev"
    DEBUG: bool = False
    POOL_SIZE: int = 20


settings = Settings()
```

Pydantic Settings reads from environment variables and `.env` files, validates the types, and provides a typed `settings` object everywhere. The `DATABASE_URL` should be a single string; pool sizes and timeouts are separate settings.

---

## 2. Per-Request Sessions via `Depends`

### 2.1 The `get_session` dependency

```python
from typing import AsyncIterator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

The dependency:

1. Opens a session when the request starts.
2. Yields it to the handler.
3. Closes it when the request ends (success or exception).

The session is **scoped to the request**. No leak between requests. No global state.

### 2.2 The handler signature

```python
@router.post("/users", response_model=UserOut)
async def create_user(
    payload: UserCreate,
    session: AsyncSession = Depends(get_session),
):
    user = User(**payload.model_dump())
    session.add(user)
    await session.commit()
    return user
```

The handler asks for an `AsyncSession`. FastAPI resolves the dependency (calls `get_session`), passes the session, and tears it down after the response is sent.

### 2.3 The auto-commit trap

The default behavior is **no auto-commit**. The handler must call `await session.commit()` explicitly. Forgetting it is the #1 source of "data not saving" bugs in FastAPI + SQLAlchemy apps.

Two ways to make commits explicit and harder to miss:

**Option 1: A `get_session_with_commit` dependency** that commits if no exception:

```python
async def get_session_with_commit() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

Use this for write endpoints. Use the plain `get_session` for read-only endpoints (committing a SELECT does nothing but adds round-trips).

**Option 2: A `get_uow` dependency** that bundles session + repositories + commit:

```python
async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with SessionLocal() as session:
        async with UnitOfWork(session) as uow:
            yield uow
            await uow.commit()
```

The `async with UnitOfWork` opens the session; `await uow.commit()` is the explicit commit. This is the cleanest pattern and pairs with the Unit of Work from [[04 - Repository Pattern and Unit of Work|note 04]].

### 2.4 The exception-rollback guarantee

`async with session` does not auto-rollback on exception. You must call `await session.rollback()` explicitly, or the next user of the session pool will inherit the broken state.

```python
# ❌ The session is left in a failed transaction state
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
        # If the handler raises, the session is not rolled back
        # The connection is returned to the pool with a pending transaction

# ✅ Rollback on exception
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

The `async with UnitOfWork` pattern above handles this in the UoW's `__aexit__`.

---

## 3. Connecting the Unit of Work to Handlers

### 3.1 The UoW dependency

```python
# app/uow.py
from contextlib import asynccontextmanager
from typing import AsyncIterator


class UnitOfWork:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.users = UserRepository(session)
        self.posts = PostRepository(session)

    async def commit(self):
        await self.session.commit()

    async def rollback(self):
        await self.session.rollback()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            await self.rollback()
        await self.session.close()


async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with SessionLocal() as session:
        async with UnitOfWork(session) as uow:
            yield uow
            # If no exception, commit; if there is, __aexit__ rolled back
            if not uow.session.in_transaction():
                return
            # Check if there are uncommitted changes; if so, commit
            # (This is a defensive check; the handler should call uow.commit())
```

A simpler form: have handlers call `uow.commit()` explicitly.

```python
async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with SessionLocal() as session:
        uow = UnitOfWork(session)
        try:
            yield uow
        finally:
            await session.close()


@router.post("/users")
async def create_user(payload: UserCreate, uow: UnitOfWork = Depends(get_uow)):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    return user
```

### 3.2 Subdependencies: repositories via the UoW

Once you have the UoW as a dependency, the repositories come "for free" through it. Don't make the repositories themselves top-level dependencies — that creates a session-management problem (which session does the repo hold?).

```python
# ❌ The session is in two places
async def get_user_repo():
    async with SessionLocal() as session:
        yield UserRepository(session)

async def get_post_repo():
    async with SessionLocal() as session:
        yield PostRepository(session)

# Two sessions, no atomicity
@router.post("/checkout")
async def checkout(users=Depends(get_user_repo), posts=Depends(get_post_repo)):
    # users and posts are on DIFFERENT sessions; the checkout is not atomic
    ...

# ✅ One UoW, one session, atomic
async def get_uow():
    async with SessionLocal() as session:
        uow = UnitOfWork(session)
        try:
            yield uow
        finally:
            await session.close()


@router.post("/checkout")
async def checkout(payload: Cart, uow: UnitOfWork = Depends(get_uow)):
    # uow.users and uow.posts share the same session
    await uow.users.update(...)
    await uow.posts.create(...)
    await uow.commit()  # atomic
```

### 3.3 Per-request validation hooks

A common need: validate the user against the database after the JWT is decoded but before the handler runs. This is a dependency:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")
    user = (await session.execute(
        select(User).where(User.id == payload["sub"])
    )).scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or inactive")
    return user


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    return user
```

The auth check is a sub-dependency of the handler. The session is provided by the parent dependency, and the user is fetched as part of the auth flow. The handler receives an authenticated `User` object.

---

## 4. Streaming and Background Tasks

### 4.1 Streaming responses

When the response is a `StreamingResponse` (e.g., a long-running LLM token stream), the session lifecycle is different:

```python
@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    user = await get_current_user_from_session(session, payload.user_id)
    if not user:
        raise HTTPException(401, "Unauthorized")

    async def generate():
        # The session must be passed to the generator explicitly
        # because the original session is closed when the response starts
        # Actually no — the session lives until the response is fully sent
        async for token in llm_stream(payload.messages):
            yield f"data: {token}\n\n"
        # After all tokens are sent, save the conversation
        await save_conversation(session, user, payload.messages, full_response)

    return StreamingResponse(generate(), media_type="text/event-stream")
```

The session is held until the streaming response is fully sent. The catch: if the client disconnects mid-stream, the cleanup path may not run. Wrap the generator in a try/finally for the commit:

```python
async def generate():
    try:
        async for token in llm_stream(payload.messages):
            yield f"data: {token}\n\n"
    except asyncio.CancelledError:
        # Client disconnected; commit what we have
        await save_conversation(session, user, messages, partial_response)
        await session.commit()
        raise
    else:
        await save_conversation(session, user, messages, full_response)
        await session.commit()
```

### 4.2 `BackgroundTasks`

FastAPI's `BackgroundTasks` runs after the response is sent:

```python
from fastapi import BackgroundTasks


@router.post("/orders")
async def create_order(
    payload: OrderCreate,
    background_tasks: BackgroundTasks,
    uow: UnitOfWork = Depends(get_uow),
):
    order = await uow.orders.create(**payload.model_dump())
    await uow.commit()
    background_tasks.add_task(send_confirmation_email, order.id)
    return order
```

The `BackgroundTasks` runs **in the same process** after the response. The session is **already closed** when the task runs. If the task needs the DB, it must open its own session:

```python
async def send_confirmation_email(order_id: int):
    async with SessionLocal() as session:
        order = await session.get(Order, order_id)
        if not order:
            return
        await email_service.send(order.user.email, ...)
```

For long-running or unreliable background work, prefer an external queue (Celery, ARQ, Dramatiq — see [[../40 - Background Jobs and Workers for FastAPI/00 - Welcome|the dedicated course]]).

---

## 5. Testing the Full Stack

### 5.1 The `TestClient` for sync tests

FastAPI's `TestClient` (based on Starlette's `TestClient` / `httpx`) supports async routes:

```python
from fastapi.testclient import TestClient


def test_create_user(client: TestClient):
    response = client.post("/users", json={"email": "alice@example.com", "name": "Alice"})
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "alice@example.com"
```

`TestClient` is synchronous and runs the event loop internally. For pure handler tests with no real DB, it's the simplest tool.

### 5.2 The `httpx.AsyncClient` for async tests

```python
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient, test_uow):
    response = await async_client.post(
        "/users",
        json={"email": "alice@example.com", "name": "Alice"},
    )
    assert response.status_code == 201
```

### 5.3 Overriding the database dependency

The standard pattern: override `get_uow` to use a test database (in-memory SQLite or testcontainers).

```python
# app/dependencies.py
async def get_uow():
    async with SessionLocal() as session:
        async with UnitOfWork(session) as uow:
            yield uow


# tests/conftest.py
from app.dependencies import get_uow
from app.main import app


@pytest_asyncio.fixture
async def test_uow():
    # Connect to a test database, create tables, etc.
    ...
    yield test_uow_instance


@pytest_asyncio.fixture
async def async_client(test_uow):
    # Override the dependency for the entire test
    async def _get_test_uow():
        yield test_uow
    app.dependency_overrides[get_uow] = _get_test_uow
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
```

The handler code is unchanged; only the dependency is swapped. The test is a real HTTP-level test (route, validation, serialization, exception handling) against a real database.

### 5.4 Test isolation: per-test database or per-test schema

Multiple tests against the same in-memory database share state. Three solutions:

```python
# Option 1: Fresh database per test (slow but simple)
@pytest_asyncio.fixture
async def test_uow():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine)() as session:
        yield UnitOfWork(session)
    await engine.dispose()


# Option 2: Truncate between tests (fast)
@pytest_asyncio.fixture(autouse=True)
async def cleanup_db(test_uow):
    yield
    # After each test, truncate all tables
    for table in reversed(Base.metadata.sorted_tables):
        await test_uow.session.execute(table.delete())


# Option 3: Transactional rollback (fastest, requires careful setup)
@pytest_asyncio.fixture
async def test_uow():
    # Begin a transaction, save the connection, yield the session,
    # then roll back at the end
    ...
```

For most teams, **Option 2** (truncate between tests) is the best balance of speed and isolation.

---

## 6. Connection Pool Sizing Across the App

### 6.1 The math

The total number of connections your service can hold open is the bottleneck:

| Component | Pool size | Notes |
|-----------|----------:|-------|
| FastAPI worker 1 | 20 | Pool per worker process |
| FastAPI worker 2 | 20 | |
| FastAPI worker 3 | 20 | |
| FastAPI worker 4 | 20 | |
| Alembic migrations | 1-5 | Only during `alembic upgrade` |
| Background worker (Celery) | 10-20 | Per worker process |
| Admin / monitoring | 2-5 | Reserved |
| **Total** | **~100-150** | Must be ≤ `max_connections` |

PostgreSQL's default `max_connections` is **100**. A naive setup with `pool_size=20` and 4 workers reaches 80 connections just for FastAPI, leaving 20 for everything else. If you run Celery, more workers, or many replicas of FastAPI, you run out of connections.

### 6.2 Three solutions

**Solution 1: Smaller pool per worker**. `pool_size=10, max_overflow=5` per worker. 4 workers × 15 = 60 connections. Plenty of headroom.

**Solution 2: PgBouncer in front**. A connection multiplexer that holds a small pool of actual PostgreSQL connections and serves many client connections. vLLM and many Python ORMs work with PgBouncer in `transaction` mode; some features (`statement_timeout`, `server-side cursors`) require `session` mode.

**Solution 3: PostgreSQL `max_connections` higher**. Set to 200 or 300. Works, but each connection uses ~10MB of RAM; at 300 connections, that's 3GB just for connections.

For a small-to-medium service, **Solution 1** is simplest. For larger fleets, **Solution 2** (PgBouncer) is the production answer.

---

## 7. The Production Configuration Pattern

A production-ready config might look like:

```python
# app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO: bool = False

    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    @property
    def async_database_url(self) -> str:
        # Convert sync URL to async if needed
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()


# app/db/engine.py
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.core.config import settings

engine = create_async_engine(
    settings.async_database_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,
    echo=settings.DB_ECHO,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

The engine is created at import. Settings come from environment variables. Pool size is configurable per environment. `pool_pre_ping=True` is the only constant.

---

## 8. Common Pitfalls

### 8.1 Forgetting to commit

```python
# ❌ The user is never saved
@router.post("/users")
async def create_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    user = User(**payload.model_dump())
    session.add(user)
    return user  # no commit
```

Always call `await session.commit()` or use the UoW pattern. Make it a code review checklist item.

### 8.2 Returning ORM objects after the session closes

```python
# ❌ Pydantic tries to access expired attributes
@router.get("/users/{user_id}")
async def get_user(user_id: int, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    return user  # session closes, attributes expire, serialization fails
```

The fix: `expire_on_commit=False` (covered in [[01 - Async Engine and Sessions|note 01]]). Or convert to a Pydantic model inside the handler.

### 8.3 Holding a session open across `await` boundaries

```python
# ❌ The session is open while the external API is called
@router.post("/users")
async def create_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    user = User(**payload.model_dump())
    session.add(user)
    await session.commit()
    await external_api.send_welcome(user)  # holds a DB connection during HTTP call
    return user
```

The session holds a connection from the pool while the external API is called. The connection is wasted. Better:

```python
# ✅ Commit first, then call external service
@router.post("/users")
async def create_user(payload: UserCreate, uow: UnitOfWork = Depends(get_uow)):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()  # returns the connection
    await external_api.send_welcome(user)  # no DB connection held
    return user
```

### 8.4 Circular imports

```python
# ❌ app/api/users.py imports app.db.session, which imports app.api.users
# The engine creation fails
```

Solution: put the engine and `SessionLocal` in a separate module (`app/db/session.py`) that depends on neither.

---

## 9. Código de Compresión

```python
"""
Compresión: FastAPI Integration: DI and Lifespan
Cubre: engine in lifespan, get_session, get_uow, BackgroundTasks, async testing.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


# 1) Settings
class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://localhost/dev"
    DEBUG: bool = False
    POOL_SIZE: int = 20

    @property
    def async_database_url(self) -> str:
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL


settings = Settings()


# 2) Engine
engine = create_async_engine(
    settings.async_database_url,
    pool_size=settings.POOL_SIZE,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=settings.DEBUG,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# 3) Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)


# 4) Per-request session
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# 5) Per-request UoW
async def get_uow() -> AsyncIterator["UnitOfWork"]:
    async with SessionLocal() as session:
        uow = UnitOfWork(session)
        try:
            yield uow
        finally:
            await session.close()


# 6) Handler with UoW
@app.post("/orders")
async def create_order(
    payload: dict,
    background_tasks: BackgroundTasks,
    uow: UnitOfWork = Depends(get_uow),
):
    order = await uow.orders.create(**payload)
    await uow.commit()  # release the connection
    background_tasks.add_task(send_email, order.id)  # no DB connection held
    return order


async def send_email(order_id: int):
    async with SessionLocal() as session:
        # own session
        order = await session.get(Order, order_id)
        # ...


# 7) Testing override
from httpx import AsyncClient, ASGITransport

async def override_get_uow():
    # In tests, this points to an in-memory SQLite session
    async with TestSessionLocal() as session:
        yield UnitOfWork(session)


app.dependency_overrides[get_uow] = override_get_uow

# 8) Run a test
async def test_create_order():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/orders", json={"user_id": 1, "total": 99.99})
    assert response.status_code == 200
```

---

## Key Takeaways

- The engine lives at module scope, not in `app.state`. The lifespan only handles `dispose()` on shutdown.
- `expire_on_commit=False` is the FastAPI default. Without it, Pydantic serialization triggers lazy loads that fail because the session is already closed.
- The `UnitOfWork` is the cleanest dependency: one session, multiple repositories, one commit, atomic transactions.
- Always commit before calling external services. Holding a connection during an HTTP call is wasted pool capacity.
- Streaming responses need explicit cleanup if the client disconnects mid-stream. Use `try/except asyncio.CancelledError` to commit partial work.
- `BackgroundTasks` runs in the same process after the response. The session is already closed; the task must open its own session.
- Test the full stack with `httpx.AsyncClient` and a real test database. Override the `Depends(get_uow)` to swap in a test session. The handler code does not change.
- Pool sizing matters: `pool_size × workers + background workers + admin < max_connections`. The default 100 is tight.

## References

- [FastAPI — Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI — Bigger Applications with Multiple Files](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [FastAPI — Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [SQLAlchemy 2.0 — AsyncSession](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [httpx — AsyncClient for testing](https://www.python-httpx.org/async/)
- [PgBouncer — connection pooling for PostgreSQL](https://www.pgbouncer.org/)
