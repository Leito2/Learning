# 🧱 Repository Pattern and Unit of Work

## 🎯 Learning Objectives

- Understand why the Repository pattern matters for FastAPI services and when not to use it
- Build a generic `BaseRepository[T]` that handles CRUD, pagination, and common queries
- Implement the Unit of Work pattern for transactional integrity across multiple repositories
- Use composition (a repository holds a session) instead of inheritance (a service inherits from ORM)
- Test repositories in isolation with a SQLite in-memory database and `pytest-asyncio`

## Introduction

The most common mistake in FastAPI services that use SQLAlchemy is letting the ORM leak into HTTP handlers. A handler that starts with `session.execute(select(...))` and ends with `session.add(...)` is hard to test (it needs a real database), hard to reason about (the handler is now also a data access layer), and hard to evolve (any model change ripples through the API code).

The Repository pattern decouples the HTTP layer from the data layer. A handler asks a `UserRepository` for "users matching this filter" or "create this user"; it never sees a `select()` statement or a `session`. The repository is the only place that knows how the data is stored. Swap SQLAlchemy for MongoDB and the handlers do not change.

The Unit of Work (UoW) complements the Repository by managing the transaction boundary. A unit of work holds a session and exposes multiple repositories; committing the UoW commits all changes atomically. A handler does one `async with uow:` block, makes several repository calls, and commits at the end — or rolls back on exception.

This note shows both patterns in a FastAPI context, with a generic base class for DRY repositories and a tested UoW.

---

## 1. Why a Repository?

### 1.1 The problem: ORM in handlers

```python
# ❌ Anti-pattern: data access inside the handler
@router.post("/users", response_model=UserOut)
async def create_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    # Validation
    if (await session.execute(select(User).where(User.email == payload.email))).scalar_one_or_none():
        raise HTTPException(409, "Email already exists")
    # Business logic
    user = User(**payload.model_dump(), is_active=True)
    session.add(user)
    await session.commit()
    # Side effects
    await send_welcome_email(user.email)
    # Response
    return user
```

Three things are wrong:

1. **Hard to test.** The test needs a real PostgreSQL, the email service mocked, the validation logic exercised. Each test setup is heavy.
2. **No transactional integrity.** If `send_welcome_email` raises, the user is already committed. If it succeeds and the response serialization fails, the email was sent for a user that "doesn't exist" in the response.
3. **No reusability.** The same "email already exists" check is needed in `update_user`, `bulk_import_users`, and `signup_via_oauth`. Duplicating the query in each handler is bug-prone.

### 1.2 The fix: a repository per aggregate

```python
# ✅ Repository: data access isolated
class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: int) -> User | None:
        return (await self.session.execute(
            select(User).where(User.id == user_id)
        )).scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        return (await self.session.execute(
            select(User).where(User.email == email)
        )).scalar_one_or_none()

    async def list(self, *, limit: int = 20, offset: int = 0) -> list[User]:
        return list((await self.session.execute(
            select(User).limit(limit).offset(offset)
        )).scalars().all())

    async def create(self, **fields) -> User:
        user = User(**fields)
        self.session.add(user)
        await self.session.flush()  # populate user.id without committing
        return user
```

```python
# ✅ Handler: clean, testable
@router.post("/users", response_model=UserOut)
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    existing = await uow.users.get_by_email(payload.email)
    if existing:
        raise HTTPException(409, "Email already exists")
    user = await uow.users.create(**payload.model_dump(), is_active=True)
    await uow.commit()  # commits the whole UoW transaction
    return user
```

The handler now contains only **business logic**. The repository contains only **data access**. Both can be tested independently.

### 1.3 When NOT to use a Repository

For tiny services (one or two endpoints, one model), the Repository pattern is overhead. The break-even point is around 3+ endpoints on the same aggregate, or any non-trivial query. For a "Hello World" service, `session.execute(select(User))` is fine.

The other anti-pattern to avoid is **a "god repository"** that holds all data access for the entire app. Each aggregate should have its own repository, and the UoW should hold a coherent set of repositories (e.g., `users`, `posts`, `comments` together; `billing` and `analytics` separately).

---

## 2. The Generic `BaseRepository`

### 2.1 The pattern

A generic base class gives you CRUD for free. Each subclass overrides or extends for custom queries.

```python
from typing import Generic, Type, TypeVar
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

T = TypeVar("T", bound=DeclarativeBase)


class BaseRepository(Generic[T]):
    model: Type[T]  # subclass must set

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id_: int) -> T | None:
        return await self.session.get(self.model, id_)

    async def get_by(self, **filters) -> T | None:
        stmt = select(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        order_by=None,
        **filters,
    ) -> list[T]:
        stmt = select(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        stmt = stmt.limit(limit).offset(offset)
        return list((await self.session.execute(stmt)).scalars().all())

    async def create(self, **fields) -> T:
        instance = self.model(**fields)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def update(self, instance: T, **fields) -> T:
        for field, value in fields.items():
            setattr(instance, field, value)
        await self.session.flush()
        return instance

    async def delete(self, instance: T) -> None:
        await self.session.delete(instance)
        await self.session.flush()
```

### 2.2 A concrete subclass

```python
class UserRepository(BaseRepository[User]):
    model = User

    async def get_active_users(self, limit: int = 20) -> list[User]:
        return await self.list(is_active=True, limit=limit, order_by=User.created_at.desc())

    async def search_by_name(self, query: str, limit: int = 20) -> list[User]:
        stmt = select(User).where(User.name.ilike(f"%{query}%")).limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())
```

The subclass adds domain-specific queries on top of the generic CRUD. The generic base handles `get`, `list`, `create`, `update`, `delete` for any model.

### 2.3 Pagination: cursor vs offset

Offset pagination (`LIMIT 20 OFFSET 100`) is simple but breaks under concurrent inserts. Cursor pagination is more robust:

```python
async def list_cursor(
    self,
    *,
    after_id: int | None = None,
    limit: int = 20,
) -> list[T]:
    stmt = select(self.model).order_by(self.model.id).limit(limit)
    if after_id is not None:
        stmt = stmt.where(self.model.id > after_id)
    return list((await self.session.execute(stmt)).scalars().all())
```

The client receives the last ID from the previous page and passes it as `after_id` for the next. New rows inserted between pages do not cause duplicates or skips.

---

## 3. The Unit of Work

### 3.1 The pattern

A UoW holds a session and the repositories that share it. The UoW is the transactional boundary: committing the UoW commits the session; rolling back rolls back the session.

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class UnitOfWork:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.users: UserRepository = UserRepository(session)
        self.posts: PostRepository = PostRepository(session)
        self.comments: CommentRepository = Comment(session)

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    async def __aenter__(self) -> "UnitOfWork":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            await self.rollback()
        await self.session.close()
```

### 3.2 Wiring in FastAPI

```python
async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with SessionLocal() as session:
        uow = UnitOfWork(session)
        try:
            yield uow
        finally:
            await session.close()


@router.post("/users/{user_id}/posts")
async def create_post(
    user_id: int,
    payload: PostCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    post = await uow.posts.create(
        author_id=user_id,
        title=payload.title,
        body=payload.body,
    )
    # Atomically: insert the post AND update the user's post_count
    user.post_count = (user.post_count or 0) + 1
    await uow.commit()
    return post
```

If the `commit()` fails, both the new post and the post_count increment are rolled back together. The UoW guarantees atomicity.

### 3.3 The dependency injection pattern

The standard FastAPI pattern uses `Depends(get_uow)`. For unit tests, you override the dependency with a test UoW connected to a SQLite in-memory database:

```python
# tests/conftest.py
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.db.base import Base
from app.uow import UnitOfWork


@pytest_asyncio.fixture
async def test_uow():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield UnitOfWork(session)
    await engine.dispose()


# app/main.py — dependency override
def get_uow_override():
    return test_uow  # noqa: not how it works, but illustrates the pattern


app.dependency_overrides[get_uow] = get_uow_override
```

The handler code never knows the difference between a test UoW and a production UoW. The `Depends` system makes the swap transparent.

---

## 4. Composition vs Inheritance

### 4.1 The wrong way: services inheriting from repositories

```python
# ❌ Inheritance creates tight coupling
class UserService(UserRepository):
    def register(self, email: str, password: str):
        # Now UserService IS a UserRepository; it's also a User
        # Adding a new field to UserRepository breaks every subclass
        ...
```

### 4.2 The right way: services compose repositories

```python
# ✅ Composition: service holds a repository
class UserService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
        self.users = uow.users

    async def register(self, email: str, password: str) -> User:
        existing = await self.users.get_by_email(email)
        if existing:
            raise ValueError("Email already exists")
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user = await self.users.create(email=email, password_hash=password_hash)
        await self.uow.commit()
        return user
```

The service does **business logic** (uniqueness check, password hashing). The repository does **data access**. The UoW manages the transaction. Three responsibilities, three classes.

### 4.3 Where does business validation go?

| Concern | Lives in | Example |
|---------|----------|---------|
| Type validation | Pydantic model | `email: EmailStr` |
| Format validation | Pydantic model | `password: constr(min_length=8)` |
| Uniqueness check | Repository | `users.get_by_email()` |
| Business rule | Service | "Cannot deactivate the last admin" |
| Authorization | Service or FastAPI dependency | `current_user.can_edit(post)` |

The repository returns data; it does not decide whether the data is "valid" in a business sense.

---

## 5. Multi-Repository Transactions

### 5.1 The classic case: order + inventory

A checkout flow that decrements inventory and creates an order must be atomic:

```python
@router.post("/checkout")
async def checkout(
    cart: Cart,
    uow: UnitOfWork = Depends(get_uow),
):
    async with uow:
        # 1) Reserve inventory
        for item in cart.items:
            product = await uow.products.get(item.product_id)
            if product.stock < item.quantity:
                raise HTTPException(409, f"{product.name} out of stock")
            product.stock -= item.quantity
        # 2) Create the order
        order = await uow.orders.create(user_id=cart.user_id, total=cart.total)
        # 3) Create the line items
        for item in cart.items:
            await uow.order_items.create(order_id=order.id, **item.model_dump())
        # 4) Commit (single transaction)
        await uow.commit()
    return {"order_id": order.id}
```

If any step fails (e.g., a payment service call), the entire flow is rolled back. The inventory is not decremented without an order; the order has no orphan line items.

### 5.2 The dead-letter pattern

Some failures cannot be retried in-process (e.g., calling an external API that times out). The pattern is to **commit the partial work** and queue a follow-up task:

```python
@router.post("/orders")
async def create_order(payload: OrderCreate, uow: UnitOfWork = Depends(get_uow)):
    # 1) Create the order
    order = await uow.orders.create(**payload.model_dump(), status="pending")
    await uow.commit()  # commit now; the order exists
    # 2) Enqueue async work (background job)
    await uow.background_jobs.enqueue(
        "charge_payment", order_id=order.id, amount=order.total,
    )
    return order
```

The UoW is **explicitly committed** at the boundary where consistency ends. The background job is a separate transaction. The same pattern handles email, PDF generation, webhooks.

---

## 6. Testing Repositories

### 6.1 The test pyramid for data

```python
# tests/repositories/test_user_repository.py
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.user import UserRepository


@pytest.mark.asyncio
async def test_get_by_email(db_session: AsyncSession):
    user = User(email="alice@example.com", name="Alice", is_active=True)
    db_session.add(user)
    await db_session.commit()

    repo = UserRepository(db_session)
    found = await repo.get_by_email("alice@example.com")
    assert found is not None
    assert found.id == user.id


@pytest.mark.asyncio
async def test_list_with_pagination(db_session: AsyncSession):
    for i in range(25):
        db_session.add(User(email=f"u{i}@example.com", name=f"U{i}"))
    await db_session.commit()

    repo = UserRepository(db_session)
    page1 = await repo.list(limit=10, offset=0)
    page2 = await repo.list(limit=10, offset=10)
    assert len(page1) == 10
    assert len(page2) == 10
    assert page1[0].id != page2[0].id


@pytest.mark.asyncio
async def test_list_filters_by_active(db_session: AsyncSession):
    db_session.add(User(email="active@example.com", is_active=True))
    db_session.add(User(email="inactive@example.com", is_active=False))
    await db_session.commit()

    repo = UserRepository(db_session)
    actives = await repo.list(is_active=True)
    assert len(actives) == 1
    assert actives[0].email == "active@example.com"
```

### 6.2 Testing with SQLite in-memory

```python
# tests/conftest.py
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.db.base import Base


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session
    await engine.dispose()
```

SQLite catches most bugs fast, but it differs from PostgreSQL in subtle ways (no JSONB, different SQL dialect for some queries, no concurrent writes in `:memory:`). For deeper integration tests, run against a real PostgreSQL via `testcontainers` or a CI-managed instance.

### 6.3 Mocking repositories in handler tests

```python
# tests/handlers/test_create_user.py
from unittest.mock import AsyncMock
import pytest
from app.main import create_user_handler


@pytest.mark.asyncio
async def test_create_user_409_on_duplicate():
    uow = AsyncMock()
    uow.users.get_by_email.return_value = AsyncMock(id=1, email="alice@example.com")
    with pytest.raises(HTTPException) as exc:
        await create_user_handler(payload=..., uow=uow)
    assert exc.value.status_code == 409
    uow.users.create.assert_not_called()
```

Mocking the UoW lets you test handler logic without a database at all. The repository contract is verified separately with the in-memory SQLite fixture.

---

## 7. Common Pitfalls

### 7.1 Repository holding state across requests

```python
# ❌ The session is bound to one request
class UserRepository:
    def __init__(self):
        self.session = AsyncSession(engine)  # module-level, never do this

    async def get(self, id_): ...
```

The repository is created **per request**, with the session injected. The session is the per-request unit of work.

### 7.2 Repository methods that don't belong

```python
# ❌ A repository shouldn't send emails
class UserRepository:
    async def register(self, email: str):
        user = await self.create(email=email)
        await send_email(email)  # business logic in the data layer
```

The repository knows about persistence, not notifications. Move `send_email` to the service.

### 7.3 Lazy-loaded relationships in the repository

```python
# ❌ Returns a User whose relationships are not loaded
async def get_by_email(self, email: str) -> User | None:
    return (await self.session.execute(
        select(User).where(User.email == email)
    )).scalar_one_or_none()


# Caller: # may trigger N+1
user = await repo.get_by_email("alice@example.com")
for post in user.posts:  # lazy load!
    ...
```

The repository's job is to return **fully-loaded** data for its common use case. If the caller needs posts, return a query with `selectinload(User.posts)`. If different callers need different relationships, expose a parameter:

```python
async def get_by_email(self, email: str, *, with_posts: bool = False) -> User | None:
    stmt = select(User).where(User.email == email)
    if with_posts:
        stmt = stmt.options(selectinload(User.posts))
    return (await self.session.execute(stmt)).scalar_one_or_none()
```

---

## 8. Código de Compresión

```python
"""
Compresión: Repository Pattern and Unit of Work
Cubre: BaseRepository generic CRUD, UnitOfWork, FastAPI dependency injection,
       service composition, test mocking.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator, Generic, Type, TypeVar
from unittest.mock import AsyncMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


T = TypeVar("T", bound=DeclarativeBase)


# 1) Generic base repository
class BaseRepository(Generic[T]):
    model: Type[T]

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id_: int) -> T | None:
        return await self.session.get(self.model, id_)

    async def get_by(self, **filters) -> T | None:
        stmt = select(self.model)
        for f, v in filters.items():
            stmt = stmt.where(getattr(self.model, f) == v)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list(self, *, limit: int = 20, offset: int = 0, **filters) -> list[T]:
        stmt = select(self.model).limit(limit).offset(offset)
        for f, v in filters.items():
            stmt = stmt.where(getattr(self.model, f) == v)
        return list((await self.session.execute(stmt)).scalars().all())

    async def create(self, **fields) -> T:
        instance = self.model(**fields)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def delete(self, instance: T) -> None:
        await self.session.delete(instance)
        await self.session.flush()


# 2) Unit of Work
class UnitOfWork:
    def __init__(self, session: AsyncSession, **repos):
        self.session = session
        for name, repo in repos.items():
            setattr(self, name, repo)

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


# 3) FastAPI dependency
async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with SessionLocal() as session:
        yield UnitOfWork(
            session,
            users=UserRepository(session),
            posts=PostRepository(session),
        )


# 4) Service composition
class UserService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
        self.users = uow.users

    async def register(self, email: str, password: str):
        if await self.users.get_by_email(email):
            raise ValueError("Email already exists")
        user = await self.users.create(email=email, password_hash="...")
        await self.uow.commit()
        return user


# 5) Test mocking
uow = AsyncMock()
uow.users.get_by_email.return_value = None
uow.users.create.return_value = AsyncMock(id=1, email="alice@example.com")
service = UserService(uow)
user = await service.register("alice@example.com", "secret")
assert user.id == 1
```

---

## Key Takeaways

- The Repository pattern isolates data access. Handlers ask a repository for data; they never see `select()` or `session.add()`.
- A generic `BaseRepository[T]` handles 80% of CRUD. Subclasses add domain-specific queries.
- The Unit of Work holds a session and the repositories that share it. It is the transactional boundary.
- Composition over inheritance: services hold a repository, do not extend it.
- Repositories should return **fully-loaded** data for their common use case. Expose a `with_*` parameter for callers that need more.
- Test repositories against an in-memory SQLite (fast feedback). Test handlers by mocking the UoW (no DB at all).
- Multi-repository transactions (e.g., checkout) commit atomically through the UoW. Long-running side effects (email, payment) commit the partial state and enqueue background work.

## References

- [Patterns of Enterprise Application Architecture — Repository, Unit of Work](https://martinfowler.com/eaaCatalog/repository.html) (Fowler, 2002)
- [SQLAlchemy 2.0 — Session Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [Clean Architecture — Uncle Bob](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [12 Factor App — Backing Services](https://12factor.net/backing-services)
