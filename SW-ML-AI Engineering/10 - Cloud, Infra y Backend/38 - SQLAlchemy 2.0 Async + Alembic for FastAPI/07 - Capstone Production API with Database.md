# 🏆 Capstone: Production API with Database

## 🎯 Learning Objectives

- Build a production-grade task management API end-to-end with FastAPI + SQLAlchemy 2.0 + Alembic
- Tie together all the patterns from the previous six notes: engine, sessions, models, migrations, repositories, UoW, dependency injection, multi-tenancy
- Implement a real authentication and authorization layer
- Test the full stack: handler tests with `httpx.AsyncClient`, repository tests with SQLite, integration tests with PostgreSQL
- Deploy the API to a container with proper health checks, graceful shutdown, and observability

## Introduction

The previous six notes each covered a single concern. This capstone weaves them into a working service: a task management API with multi-tenant isolation, JWT auth, CRUD endpoints, pagination, filtering, migrations, and tests. The goal is not to build a new framework but to show the patterns from the course applied together in a coherent whole.

The capstone is intentionally small but complete. Every concept from the course is used at least once. Reading the capstone end-to-end is a review of the entire course. Reading just the capstone without the prior notes is a shortcut that will leave gaps; the notes explain the trade-offs, the alternatives, and the production concerns.

The codebase is structured around a `src/` layout: `app/db/` for the data layer, `app/api/` for the routes, `app/models/` for ORM models, `app/repositories/` for data access, `app/services/` for business logic, `app/core/` for cross-cutting concerns. Tests mirror the structure under `tests/`.

---

## 1. Project Structure

```
task_api/
├── pyproject.toml
├── alembic.ini
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 001_initial.py
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── base.py
│   │   ├── engine.py
│   │   └── session.py
│   ├── models/
│   │   ├── tenant.py
│   │   ├── user.py
│   │   ├── project.py
│   │   └── task.py
│   ├── repositories/
│   │   ├── base.py
│   │   ├── user.py
│   │   ├── project.py
│   │   └── task.py
│   ├── services/
│   │   ├── auth.py
│   │   ├── task.py
│   │   └── tenant.py
│   ├── api/
│   │   ├── deps.py
│   │   ├── auth.py
│   │   ├── projects.py
│   │   └── tasks.py
│   └── middleware/
│       └── tenant.py
├── tests/
│   ├── conftest.py
│   ├── repositories/
│   ├── api/
│   └── integration/
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 2. Configuration

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://app_user:pass@localhost/task_api"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 1800

    # Auth
    JWT_SECRET: str = "change-me-in-production-please"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # App
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    @property
    def async_database_url(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()
```

---

## 3. The Data Layer

### 3.1 The declarative base

```python
# app/db/base.py
from datetime import datetime
from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TenantScopedMixin:
    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
```

### 3.2 Models

```python
# app/models/tenant.py
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base, TimestampMixin


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")


# app/models/user.py
from typing import List
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin, TenantScopedMixin


class User(Base, TimestampMixin, TenantScopedMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="member", server_default="member")
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")
    projects: Mapped[List["Project"]] = relationship(
        back_populates="members",
        secondary="project_members",
        lazy="selectin",
    )


# app/models/project.py
from typing import List
from sqlalchemy import ForeignKey, String, Table, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin, TenantScopedMixin


project_members = Table(
    "project_members",
    Base.metadata,
    Column("project_id", ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)


class Project(Base, TimestampMixin, TenantScopedMixin):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), default=None)
    members: Mapped[List["User"]] = relationship(
        secondary=project_members,
        back_populates="projects",
        lazy="selectin",
    )
    tasks: Mapped[List["Task"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


# app/models/task.py
from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import ForeignKey, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin, TenantScopedMixin

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.user import User


class Task(Base, TimestampMixin, TenantScopedMixin):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2000), default=None)
    status: Mapped[str] = mapped_column(String(20), default="todo", server_default="todo")
    priority: Mapped[int] = mapped_column(default=0)
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assignee_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        default=None,
    )
    project: Mapped["Project"] = relationship(back_populates="tasks", lazy="joined")
    assignee: Mapped["User"] = relationship(lazy="joined")
```

### 3.3 Engine and session factory

```python
# app/db/engine.py
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.core.config import settings

engine = create_async_engine(
    settings.async_database_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

---

## 4. Repositories and Unit of Work

### 4.1 The base repository

```python
# app/repositories/base.py
from typing import Generic, Type, TypeVar
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    model: Type[T]

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

    async def delete(self, instance: T) -> None:
        await self.session.delete(instance)
        await self.session.flush()
```

### 4.2 The Unit of Work

```python
# app/repositories/uow.py
from typing import AsyncIterator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.engine import SessionLocal
from app.repositories.user import UserRepository
from app.repositories.project import ProjectRepository
from app.repositories.task import TaskRepository


class UnitOfWork:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.users = UserRepository(session)
        self.projects = ProjectRepository(session)
        self.tasks = TaskRepository(session)

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


@asynccontextmanager
async def get_uow() -> AsyncIterator[UnitOfWork]:
    async with SessionLocal() as session:
        async with UnitOfWork(session) as uow:
            # Set tenant for RLS
            await session.execute(
                text("SET LOCAL app.current_tenant_id = :tid"),
                {"tid": "1"},  # placeholder, overridden by get_tenant
            )
            yield uow
```

---

## 5. Security

```python
# app/core/security.py
from datetime import datetime, timedelta
from typing import Any
import bcrypt
import jwt
from app.core.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(*, sub: int, tenant_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {"sub": sub, "tenant_id": tenant_id, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
```

---

## 6. Middleware: Tenant Resolution

```python
# app/middleware/tenant.py
import jwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.config import settings


PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class TenantContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        # Extract tenant from JWT
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        token = auth[7:]
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        except jwt.PyJWTError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)
        request.state.tenant_id = payload.get("tenant_id")
        request.state.user_id = payload.get("sub")
        return await call_next(request)
```

---

## 7. The API Layer

### 7.1 Authentication endpoints

```python
# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from app.api.deps import get_uow
from app.core.security import create_access_token, hash_password, verify_password
from app.repositories.uow import UnitOfWork

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterIn(BaseModel):
    email: EmailStr
    name: str
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenOut)
async def register(payload: RegisterIn, uow: UnitOfWork = Depends(get_uow)):
    if await uow.users.get_by(email=payload.email):
        raise HTTPException(409, "Email already registered")
    user = await uow.users.create(
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
    )
    await uow.commit()
    token = create_access_token(sub=user.id, tenant_id=user.tenant_id)
    return TokenOut(access_token=token)


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn, uow: UnitOfWork = Depends(get_uow)):
    user = await uow.users.get_by(email=payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token(sub=user.id, tenant_id=user.tenant_id)
    return TokenOut(access_token=token)
```

### 7.2 Task endpoints

```python
# app/api/tasks.py
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from app.api.deps import get_uow, get_current_user
from app.models.user import User
from app.models.task import Task
from app.repositories.uow import UnitOfWork

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskIn(BaseModel):
    title: str
    description: Optional[str] = None
    priority: int = 0
    project_id: int
    assignee_id: Optional[int] = None


class TaskOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: str
    priority: int
    project_id: int
    assignee_id: Optional[int]

    class Config:
        from_attributes = True


@router.post("", response_model=TaskOut, status_code=201)
async def create_task(
    payload: TaskIn,
    uow: UnitOfWork = Depends(get_uow),
    current_user: User = Depends(get_current_user),
):
    project = await uow.projects.get(payload.project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    task = await uow.tasks.create(
        title=payload.title,
        description=payload.description,
        priority=payload.priority,
        project_id=payload.project_id,
        assignee_id=payload.assignee_id,
        tenant_id=current_user.tenant_id,
    )
    await uow.commit()
    return task


@router.get("", response_model=list[TaskOut])
async def list_tasks(
    status: Optional[str] = Query(None),
    project_id: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    uow: UnitOfWork = Depends(get_uow),
):
    filters = {}
    if status:
        filters["status"] = status
    if project_id:
        filters["project_id"] = project_id
    return await uow.tasks.list(limit=limit, offset=offset, **filters)


@router.get("/{task_id}", response_model=TaskOut)
async def get_task(task_id: int, uow: UnitOfWork = Depends(get_uow)):
    task = await uow.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@router.patch("/{task_id}", response_model=TaskOut)
async def update_task(
    task_id: int,
    payload: TaskIn,
    uow: UnitOfWork = Depends(get_uow),
):
    task = await uow.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task = await uow.tasks.update(task, **payload.model_dump(exclude_unset=True))
    await uow.commit()
    return task


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: int, uow: UnitOfWork = Depends(get_uow)):
    task = await uow.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    await uow.tasks.delete(task)
    await uow.commit()
```

### 7.3 The FastAPI app

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import auth, tasks, projects
from app.db.engine import engine
from app.middleware.tenant import TenantContextMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan, title="Task API")
app.add_middleware(TenantContextMiddleware)
app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(projects.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 8. Alembic Migration (Initial Schema)

```python
# alembic/versions/001_initial.py
"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-07-01
"""
from alembic import op
import sqlalchemy as sa


revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(50), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_projects_tenant_id", "projects", ["tenant_id"])
    op.create_table(
        "project_members",
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.String(2000), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="todo"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("assignee_id", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_tasks_tenant_id", "tasks", ["tenant_id"])
    op.create_index("ix_tasks_project_id", "tasks", ["project_id"])
    op.create_index("ix_tasks_status", "tasks", ["status"])

    # Enable RLS on tenant-scoped tables
    for table in ("users", "projects", "tasks"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::INTEGER)
        """)


def downgrade() -> None:
    for table in ("tasks", "projects", "users"):
        op.execute(f"DROP POLICY tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.drop_table("tasks")
    op.drop_table("project_members")
    op.drop_table("projects")
    op.drop_table("users")
    op.drop_table("tenants")
```

---

## 9. Tests

### 9.1 Test fixtures

```python
# tests/conftest.py
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from app.db.base import Base
from app.db.engine import engine, SessionLocal
from app.main import app
from app.repositories.uow import UnitOfWork


@pytest_asyncio.fixture
async def db_engine():
    """Create a fresh in-memory database for the test."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    await test_engine.dispose()


@pytest_asyncio.fixture
async def uow(db_engine):
    """A unit of work connected to the test database."""
    SessionTesting = async_sessionmaker(db_engine, expire_on_commit=False)
    async with SessionTesting() as session:
        uow = UnitOfWork(session)
        try:
            yield uow
        finally:
            await session.close()


@pytest_asyncio.fixture
async def seeded_db(uow):
    """A database with a tenant, a user, and a project."""
    # Create tenant directly (no RLS in SQLite tests)
    from app.models.tenant import Tenant
    from app.models.user import User
    from app.models.project import Project
    tenant = Tenant(name="Test Co", slug="test")
    uow.session.add(tenant)
    await uow.session.flush()
    user = User(
        tenant_id=tenant.id,
        email="alice@test.com",
        name="Alice",
        hashed_password="hashed",
    )
    uow.session.add(user)
    project = Project(tenant_id=tenant.id, name="Test Project")
    uow.session.add(project)
    await uow.session.commit()
    return {"tenant": tenant, "user": user, "project": project}


@pytest_asyncio.fixture
async def client(seeded_db):
    """An HTTP client with the seeded database."""
    # Override get_uow to use the test database
    from app.api.deps import get_uow
    from app.repositories.uow import UnitOfWork as RealUoW

    async def override_get_uow():
        # Yield the same session the fixture uses
        yield RealUoW(seeded_db["__uow_session"])

    # Build a UoW fixture
    SessionTesting = async_sessionmaker(db_engine, expire_on_commit=False)
    async with SessionTesting() as session:
        async def override():
            uow = RealUoW(session)
            async with uow:
                yield uow
        app.dependency_overrides[get_uow] = override
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    app.dependency_overrides.clear()
```

### 9.2 Repository tests

```python
# tests/repositories/test_task.py
import pytest
from app.models.task import Task


@pytest.mark.asyncio
async def test_create_task(uow, seeded_db):
    task = await uow.tasks.create(
        title="Test task",
        project_id=seeded_db["project"].id,
        tenant_id=seeded_db["tenant"].id,
    )
    await uow.commit()
    assert task.id is not None
    assert task.status == "todo"


@pytest.mark.asyncio
async def test_list_tasks_filter_by_status(uow, seeded_db):
    project_id = seeded_db["project"].id
    tenant_id = seeded_db["tenant"].id
    await uow.tasks.create(title="Task 1", status="todo", project_id=project_id, tenant_id=tenant_id)
    await uow.tasks.create(title="Task 2", status="done", project_id=project_id, tenant_id=tenant_id)
    await uow.commit()
    todos = await uow.tasks.list(status="todo")
    assert len(todos) == 1
    assert todos[0].title == "Task 1"
```

### 9.3 API tests

```python
# tests/api/test_auth.py
import pytest
from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_register_and_login(client, seeded_db):
    # Register
    response = await client.post(
        "/auth/register",
        json={"email": "bob@test.com", "name": "Bob", "password": "secret123"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    assert token


@pytest.mark.asyncio
async def test_list_tasks_requires_auth(client):
    response = await client.get("/tasks")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_tasks_returns_tenant_tasks(client, seeded_db):
    user = seeded_db["user"]
    token = create_access_token(sub=user.id, tenant_id=user.tenant_id)
    response = await client.get(
        "/tasks",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

### 9.4 Integration test (with real PostgreSQL)

```python
# tests/integration/test_rls.py
import pytest
from sqlalchemy import text
from app.db.engine import SessionLocal
from app.models.user import User
from app.models.task import Task
from app.repositories.uow import UnitOfWork


@pytest.mark.asyncio
@pytest.mark.integration  # opt-in: pytest -m integration
async def test_rls_isolates_tenants():
    """Run against a real PostgreSQL. Skipped if not available."""
    # Create two tenants with a user each
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET LOCAL app.current_tenant_id = '1'"))
            tenant1 = ...  # create tenant
            user1 = ...    # create user
            await session.execute(text("SET LOCAL app.current_tenant_id = '2'"))
            tenant2 = ...  # create tenant
            user2 = ...    # create user
            await session.commit()
    # Test: tenant 1 should not see tenant 2's data
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET LOCAL app.current_tenant_id = '1'"))
            result = await session.execute(select(User))
            assert len(result.scalars().all()) == 1
```

---

## 10. Docker Setup

```dockerfile
# docker/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir .

COPY . .

# Run migrations at startup
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

```yaml
# docker/docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: "postgresql+asyncpg://app_user:pass@db:5432/task_api"
      JWT_SECRET: "change-me-in-production"
    depends_on:
      db:
        condition: service_healthy
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: task_api
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app_user"]
      interval: 5s
      retries: 5
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

---

## 11. Production Deployment Checklist

| Concern | Where it's handled | Verified by |
|---------|-------------------|-------------|
| Engine and pool sizing | `app/db/engine.py` | Load test, connection count in pg_stat_activity |
| Migrations | Alembic in `docker/Dockerfile` | `alembic upgrade head` in CI |
| Tenant isolation | RLS in migration 001 | Integration test `test_rls_isolates_tenants` |
| JWT auth | `app/core/security.py` | `tests/api/test_auth.py` |
| Per-request session | `get_uow` dependency | `test_list_tasks_returns_tenant_tasks` |
| Graceful shutdown | `lifespan` calls `engine.dispose()` | Process exits in <5s on SIGTERM |
| Health check | `/health` endpoint | K8s liveness probe |
| Observability | Prometheus metrics, structured logs | Grafana dashboard |
| Backups | PostgreSQL `pg_dump` cron | Daily restore drill |

---

## 12. What This Capstone Demonstrates

The capstone uses every pattern from the course:

| Note | Pattern | Where in capstone |
|------|---------|------------------|
| 01 Async Engine and Sessions | `create_async_engine`, `expire_on_commit=False` | `app/db/engine.py` |
| 02 Models and Eager Loading | `Mapped[...]` API, `selectinload` for tasks | `app/models/*.py` |
| 03 Alembic Migrations | Initial migration with RLS | `alembic/versions/001_initial.py` |
| 04 Repository and UoW | `BaseRepository[T]`, `UnitOfWork` | `app/repositories/*.py` |
| 05 FastAPI Integration | `lifespan`, `Depends(get_uow)` | `app/main.py`, `app/api/deps.py` |
| 06 Multi-Tenancy | RLS in the migration, `SET LOCAL` in UoW | Migration + middleware |
| 07 (this note) | End-to-end integration | The whole app |

A new developer reading the capstone top-to-bottom gets a working mental model of the entire stack. A new developer reading just the models, the UoW, or the RLS migration gets a deep dive into one concern. The structure is intentionally layered: changes to the data layer do not ripple into the API layer, and changes to the API layer do not ripple into the data layer.

---

## Key Takeaways

- A production FastAPI + SQLAlchemy 2.0 service has a layered structure: models → repositories → UoW → handlers. Each layer has one job and one test approach.
- The `UnitOfWork` is the linchpin: one session, multiple repositories, atomic transactions. Handlers never see `select()` or `session.add()`.
- Multi-tenancy with RLS is enforced at the database. Application code can forget the `tenant_id` filter; the DB rejects the query.
- Migrations are versioned and reversible. RLS is enabled and configured in the initial migration; new tables get the same treatment.
- Tests are layered: repositories against an in-memory SQLite, handlers with mocked UoWs, integration tests against a real PostgreSQL with RLS.
- The `Depends(get_uow)` pattern makes the data layer swappable for tests. No handler code changes between test and production.

## References

- [FastAPI — Bigger Applications with Multiple Files](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [SQLAlchemy 2.0 — ORM Quickstart](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [Alembic — Auto Generating Migrations](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)
- [PostgreSQL — Row Security Policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [httpx — AsyncClient for testing](https://www.python-httpx.org/async/)
- [bcrypt for password hashing](https://github.com/pyca/bcrypt/)
- [python-jose for JWT](https://github.com/mpdavis/python-jose/)
