# 🏢 Multi-Tenancy Patterns

## 🎯 Learning Objectives

- Distinguish the three core multi-tenancy strategies and choose the right one for a given service
- Implement schema-per-tenant with SQLAlchemy + Alembic
- Apply row-level security in PostgreSQL to enforce tenant isolation at the database
- Build a tenant context middleware that propagates the active tenant through every request
- Avoid the cross-tenant query trap: ensure the tenant filter is always applied

## Introduction

A multi-tenant application serves many customers ("tenants") from a single codebase. The data for tenant A must be invisible to tenant B; the users of tenant A must not be able to mutate tenant B's records; a bug that forgets the tenant filter is a security incident. SQLAlchemy 2.0 + PostgreSQL give you three fundamentally different ways to enforce isolation; the right choice depends on the workload, the compliance requirements, and the operational cost.

This note walks through the three strategies with a focus on what SQLAlchemy makes easy and what requires explicit care. The strategies are **schema-per-tenant** (each tenant has its own PostgreSQL schema), **row-level security** (one schema, a `tenant_id` column on every table, RLS policies in the DB), and **shared-schema with discriminator** (one schema, a `tenant_id` column, the application is the only enforcement). The trade-offs are real: schema-per-tenant gives the strongest isolation but is the most operationally heavy; shared-schema-with-discriminator is the cheapest but relies on developer discipline.

---

## 1. The Three Strategies

### 1.1 Strategy comparison

| Strategy | Isolation | Cost | Best for |
|----------|-----------|------|----------|
| Database-per-tenant | Strongest (separate PostgreSQL cluster) | Very high (provisioning, migrations) | Highly regulated, very large tenants |
| Schema-per-tenant | Strong (separate PG schema, one DB) | Medium (schema routing, migration orchestration) | Mid-size B2B SaaS |
| Shared schema with `tenant_id` | Application-enforced (one DB, one schema, one row per tenant) | Low (single migration, no routing) | Small to mid B2B SaaS |
| Shared schema with RLS | DB-enforced (`tenant_id` + PostgreSQL Row-Level Security) | Low (same as above, but the DB enforces) | Recommended default for most B2B SaaS |

The **shared schema with RLS** strategy is the modern default. It gives the operational simplicity of a single schema with the security guarantee of database-enforced isolation. A bug in application code that forgets the `tenant_id` filter still fails the query, because the database itself refuses to return rows that don't match the active tenant.

### 1.2 How tenant identity flows

Every request has a tenant. The tenant identity is derived from a subdomain (`acme.example.com`), a JWT claim (`tenant_id`), a header (`X-Tenant-ID`), or a path segment (`/api/v1/tenants/{tenant_id}/...`). The tenant is resolved early in the request lifecycle and propagated through:

- The database session (so the `WHERE tenant_id = ?` filter is automatic).
- The logging context (so logs are grouped by tenant).
- The metrics (so dashboards split by tenant).
- The background jobs (so the worker knows which tenant to act as).

---

## 2. Shared Schema with Discriminator

### 2.1 The naive model

```python
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(255))
    name: Mapped[str]
```

A `tenant_id` on every tenant-scoped table. Every query must include `WHERE tenant_id = ?`. The danger: forgetting the filter returns or mutates data for the wrong tenant.

### 2.2 The repository pattern mistake

The naïve repository has the filter in every method:

```python
# ❌ The filter is repeated everywhere
class UserRepository:
    async def get(self, tenant_id: int, user_id: int) -> User | None:
        return (await self.session.execute(
            select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        )).scalar_one_or_none()

    async def list(self, tenant_id: int) -> list[User]:
        return list((await self.session.execute(
            select(User).where(User.tenant_id == tenant_id)
        )).scalars().all())
```

A bug: a developer adds a new method and forgets the `tenant_id` filter. The unit test passes. Production sees cross-tenant data.

### 2.3 The SQLAlchemy event listener approach

A more robust pattern: a SQLAlchemy event listener that auto-injects the `tenant_id` filter on every query. The session has a "current tenant" attribute; the listener reads it and adds the filter.

```python
from sqlalchemy import event
from sqlalchemy.orm import Query


class TenantSession(AsyncSession):
    """An AsyncSession that knows about the current tenant."""
    _current_tenant_id: int | None = None

    def set_tenant(self, tenant_id: int) -> None:
        self._current_tenant_id = tenant_id


@event.listens_for(TenantSession, "do_orm_execute")
def _add_tenant_filter(orm_execute_state):
    if (
        orm_execute_state.is_select
        and orm_execute_state.session._current_tenant_id is not None
    ):
        # Add tenant_id = :tenant_id to every SELECT
        tenant_id = orm_execute_state.session._current_tenant_id
        # Apply to all entities with a tenant_id column
        for entity in orm_execute_state.bind_arguments.get("mapper", []):
            if hasattr(entity.class_, "tenant_id"):
                ...
```

This is complex. It also has edge cases: aggregations (`COUNT(*)` without a `mapper`), joins that span tenants, ad-hoc `text()` queries. The event-listener approach is fragile.

The cleaner alternative: **PostgreSQL Row-Level Security**. The database itself rejects cross-tenant queries. Application code can forget the filter; the DB catches it.

---

## 3. PostgreSQL Row-Level Security (RLS)

### 3.1 The RLS pattern

RLS uses policies that filter rows based on a session-level setting:

```sql
-- Enable RLS on the users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Define a policy: only rows where tenant_id matches the session variable are visible
CREATE POLICY tenant_isolation ON users
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);
```

The application sets the session variable per request:

```sql
SET app.current_tenant_id = 42;
```

A query without a `tenant_id` filter:

```sql
SELECT * FROM users;  -- returns only users where tenant_id = 42
```

If `app.current_tenant_id` is not set, the query returns zero rows. A bug in application code that forgets the filter is now caught at the database level.

### 3.2 SQLAlchemy integration: the `SET LOCAL` pattern

The pattern is: open a transaction, set the session variable, run the query, commit/rollback.

```python
class TenantAwareSession(AsyncSession):
    _current_tenant_id: int | None = None

    def set_tenant(self, tenant_id: int) -> None:
        self._current_tenant_id = tenant_id

    async def execute(self, statement, parameters=None, **kwargs):
        if self._current_tenant_id is None:
            raise RuntimeError("Tenant not set on session")
        # Begin a transaction (if not already in one)
        if not self.in_transaction():
            await self.begin()
        # Set the session variable
        await super().execute(
            text("SET LOCAL app.current_tenant_id = :tid"),
            {"tid": self._current_tenant_id},
        )
        return await super().execute(statement, parameters, **kwargs)
```

Every `execute()` is wrapped: it begins a transaction (if needed), sets the tenant variable via `SET LOCAL` (scoped to the transaction), then runs the user's query. The `SET LOCAL` is rolled back when the transaction ends, so a returned-to-pool connection does not leak the tenant.

### 3.3 The migration to enable RLS

In an Alembic migration:

```python
def upgrade() -> None:
    # Enable RLS on each tenant-scoped table
    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE posts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE comments ENABLE ROW LEVEL SECURITY")
    # Define the policies
    op.execute("""
        CREATE POLICY tenant_isolation_users ON users
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::INTEGER)
    """)
    op.execute("""
        CREATE POLICY tenant_isolation_posts ON posts
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::INTEGER)
    """)
    # ...


def downgrade() -> None:
    op.execute("DROP POLICY tenant_isolation_users ON users")
    op.execute("DROP POLICY tenant_isolation_posts ON posts")
    op.execute("ALTER TABLE users DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE posts DISABLE ROW LEVEL SECURITY")
    # ...
```

The `TRUE` as the second argument to `current_setting` means "return NULL if not set, don't raise". With this, queries against a session without the variable return zero rows instead of erroring.

### 3.4 Bypassing RLS for admin operations

Some operations need to span tenants (reporting, migrations, superadmin). Use a separate DB role with `BYPASSRLS`:

```sql
-- Create an admin role
CREATE ROLE app_admin BYPASSRLS;

-- Grant the role to the migration user
GRANT app_admin TO migration_user;
```

The application's normal user does not have `BYPASSRLS`. Migrations and admin scripts connect with the admin role. This is enforced at the database — no way to forget it from the application.

### 3.5 Testing RLS

```python
# tests/test_rls.py
import pytest
from sqlalchemy import text
from app.db.base import Base
from app.db.session import SessionLocal


@pytest.mark.asyncio
async def test_rls_isolates_tenants():
    # Setup: create two tenants with users
    async with SessionLocal() as session:
        await session.execute(text("SET LOCAL app.current_tenant_id = '1'"))
        await session.execute(text("INSERT INTO users (tenant_id, email) VALUES (1, 'a@x.com')"))
        # ... commit and switch to tenant 2
        await session.commit()

        await session.execute(text("SET LOCAL app.current_tenant_id = '2'"))
        result = await session.execute(text("SELECT * FROM users"))
        rows = result.fetchall()
        assert len(rows) == 0  # tenant 2 cannot see tenant 1's user
```

The test must run against a real PostgreSQL, not SQLite. SQLite does not support `SET LOCAL` or RLS.

---

## 4. Schema-per-Tenant

### 4.1 The pattern

Each tenant has its own PostgreSQL schema (a namespace within one database). The application sets the `search_path` to the tenant's schema before running queries.

```sql
CREATE SCHEMA tenant_1;
CREATE SCHEMA tenant_2;
CREATE TABLE tenant_1.users (...);  -- separate physical tables
CREATE TABLE tenant_2.users (...);
```

```sql
SET search_path = tenant_1;
SELECT * FROM users;  -- reads from tenant_1.users
```

The advantage: cross-tenant queries are physically impossible (the table doesn't exist in the wrong schema). The disadvantage: every migration runs once per tenant; new tenants need a schema bootstrap.

### 4.2 The SQLAlchemy pattern

```python
class TenantSession(AsyncSession):
    _current_schema: str | None = None

    async def set_schema(self, schema: str) -> None:
        self._current_schema = schema
        await self.execute(text(f'SET search_path TO "{schema}"'))
```

The `search_path` is set per session. The session holds the schema for its lifetime.

### 4.3 Alembic and per-schema migrations

The standard Alembic approach: a single `Base.metadata`; the migration creates tables in the active schema. Running migrations for a new tenant is a script that switches `search_path` first:

```python
def run_migration_for_tenant(tenant_id: int, schema_name: str):
    engine = create_engine(SYNC_URL)  # Alembic is sync
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {schema_name}'))
        conn.execute(text(f'SET search_path TO {schema_name}'))
        # Now run Alembic
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
```

This is more complex than the shared-schema approach. The operational cost is real: every new tenant requires a schema bootstrap, every migration must be run N times (one per tenant).

### 4.4 When to use schema-per-tenant

- The tenants have very different data shapes (one tenant has a custom field; the schema is genuinely different).
- A tenant has a hard compliance requirement (e.g., data must be physically separable).
- A tenant outgrows the shared database and needs to be migrated to its own cluster (schema-per-tenant is the halfway step).

For most B2B SaaS, **shared schema with RLS** is the better choice. Schema-per-tenant is the option you reach for when the simpler approach can't satisfy a real requirement.

---

## 5. Resolving the Tenant Identity

### 5.1 The resolution hierarchy

A request arrives. The tenant is resolved from one of:

| Source | Example | When to use |
|--------|---------|-------------|
| JWT claim | `{"tenant_id": 42}` | Authenticated multi-tenant apps |
| Subdomain | `acme.example.com` | Marketing-friendly URLs |
| Header | `X-Tenant-ID: 42` | Internal services |
| Path | `/tenants/42/...` | Admin tools |
| Database (default) | User → Tenant | SSO / user-pool-of-tenants |

The order of preference: **JWT claim first** (already authenticated), then subdomain, then header. Path and DB lookups are for admin tools.

### 5.2 A tenant resolution dependency

```python
from fastapi import Depends, Header, Request
from app.db.session import SessionLocal
from app.models.tenant import Tenant


async def get_tenant(
    request: Request,
    x_tenant_id: int | None = Header(default=None, alias="X-Tenant-ID"),
) -> int:
    """Resolve the active tenant for this request."""
    # 1) Try the request state (set by auth middleware)
    if hasattr(request.state, "tenant_id") and request.state.tenant_id is not None:
        return request.state.tenant_id
    # 2) Fall back to the header
    if x_tenant_id is not None:
        return x_tenant_id
    raise HTTPException(400, "Tenant not specified")


async def get_uow(
    tenant_id: int = Depends(get_tenant),
) -> AsyncIterator["TenantScopedUnitOfWork"]:
    async with SessionLocal() as session:
        uow = TenantScopedUnitOfWork(session, tenant_id=tenant_id)
        try:
            await uow.set_tenant_scope(tenant_id)
            yield uow
        finally:
            await session.close()
```

The dependency chain:

1. `get_tenant` resolves the tenant ID from JWT or header.
2. `get_uow` opens a session and applies the tenant scope.
3. The handler uses the UoW; the UoW's session has the RLS variable set.

### 5.3 Setting the tenant at session start

```python
class TenantScopedUnitOfWork(UnitOfWork):
    def __init__(self, session: AsyncSession, tenant_id: int):
        super().__init__(session)
        self.tenant_id = tenant_id

    async def set_tenant_scope(self, tenant_id: int) -> None:
        """Set the PostgreSQL session variable for RLS."""
        # Begin a transaction (SET LOCAL is transaction-scoped)
        if not self.session.in_transaction():
            await self.session.begin()
        await self.session.execute(
            text("SET LOCAL app.current_tenant_id = :tid"),
            {"tid": str(tenant_id)},
        )
```

The `SET LOCAL` is the right tool here: it lives for the duration of the transaction, then disappears. A connection returned to the pool has no tenant set.

---

## 6. Cross-Tenant Operations (Superadmin)

### 6.1 The pattern

A superadmin endpoint needs to query all tenants. The role has `BYPASSRLS`; the application uses a different DB user for superadmin operations.

```python
# app/db/admin.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings

# Separate engine for admin operations, with a user that has BYPASSRLS
admin_engine = create_async_engine(settings.ADMIN_DATABASE_URL, ...)
AdminSessionLocal = async_sessionmaker(admin_engine, expire_on_commit=False)


async def get_admin_uow() -> AsyncIterator["UnitOfWork"]:
    async with AdminSessionLocal() as session:
        uow = UnitOfWork(session)  # no tenant scope
        try:
            yield uow
        finally:
            await session.close()


@router.get("/admin/all-users")
async def list_all_users(admin_uow: UnitOfWork = Depends(get_admin_uow)):
    # No tenant filter needed; the admin user has BYPASSRLS
    return await admin_uow.users.list(limit=1000)
```

### 6.2 Why a separate engine

The temptation is to use a flag on the regular session: `session._is_admin = True` and skip the `SET LOCAL`. The problem: the `AsyncSession` is checked out from a pool; the next user of the connection has the same connection state. A separate engine ensures the admin connection is never used for tenant-scoped work.

### 6.3 The deployment reality

| Component | DB user | RLS | Purpose |
|-----------|---------|-----|---------|
| FastAPI workers | `app_user` | Enforced | All tenant-scoped work |
| Alembic migrations | `app_admin` | Bypassed | Schema changes |
| Superadmin endpoints | `app_admin` | Bypassed | Cross-tenant operations |
| Monitoring / BI | `app_readonly` | Enforced | Reports limited to current tenant context |

---

## 7. Tenant Bootstrapping and Migrations

### 7.1 Creating a new tenant

When a new customer signs up:

```python
# app/services/tenant.py
async def create_tenant(name: str, slug: str) -> Tenant:
    """Create a new tenant with its own schema or RLS scope."""
    async with AdminSessionLocal() as session:
        async with session.begin():
            # 1) Create the tenant row
            tenant = Tenant(name=name, slug=slug, is_active=True)
            session.add(tenant)
            await session.flush()
            # 2) Bootstrap the tenant's data
            await session.execute(
                text("SET LOCAL app.current_tenant_id = :tid"),
                {"tid": str(tenant.id)},
            )
            # 3) Insert the default admin user
            admin = User(
                tenant_id=tenant.id,
                email=f"admin@{slug}.com",
                role="admin",
                is_active=True,
            )
            session.add(admin)
    return tenant
```

For shared-schema RLS, the new tenant is a row in `tenants` plus a row in `users` (the admin). For schema-per-tenant, the script creates the schema and runs migrations:

```python
async def create_tenant_with_schema(name: str, slug: str) -> Tenant:
    async with AdminSessionLocal() as session:
        async with session.begin():
            tenant = Tenant(name=name, slug=slug)
            session.add(tenant)
            await session.flush()
            schema_name = f"tenant_{tenant.id}"
            await session.execute(text(f'CREATE SCHEMA "{schema_name}"'))
    # Run migrations for the new schema
    run_migration_for_tenant(tenant.id, schema_name)
    return tenant
```

### 7.2 Migrations for shared schema

Migrations run once, against the shared schema. The RLS policies are part of the migration:

```python
# alembic/versions/abc_add_users_table.py
def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    # Enable RLS
    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON users
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::INTEGER)
    """)
```

### 7.3 Migrations for schema-per-tenant

Migrations must run once per tenant. A wrapper script:

```python
# scripts/migrate_all_tenants.py
import asyncio
from sqlalchemy import select, text
from app.db.admin import AdminSessionLocal
from app.models.tenant import Tenant


async def migrate_all():
    async with AdminSessionLocal() as session:
        tenants = (await session.execute(select(Tenant))).scalars().all()
    for tenant in tenants:
        schema = f"tenant_{tenant.id}"
        print(f"Migrating {schema}...")
        # Run Alembic with the schema set
        engine = create_engine(settings.async_database_url)
        with engine.begin() as conn:
            conn.execute(text(f'SET search_path TO "{schema}"'))
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
        print(f"  Done.")


asyncio.run(migrate_all())
```

---

## 8. The Tenant Context Middleware

A middleware that resolves the tenant early and sets the session variable:

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class TenantContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1) Extract tenant ID from the request
        tenant_id = await self._resolve_tenant(request)
        if tenant_id is None:
            return JSONResponse({"detail": "Tenant not found"}, status_code=400)
        # 2) Validate the tenant
        if not await self._is_active_tenant(tenant_id):
            return JSONResponse({"detail": "Tenant inactive"}, status_code=403)
        # 3) Store on request state
        request.state.tenant_id = tenant_id
        # 4) Call the next handler
        response = await call_next(request)
        return response

    async def _resolve_tenant(self, request: Request) -> int | None:
        # Try JWT
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            try:
                payload = jwt.decode(token, options={"verify_aud": False})
                return payload.get("tenant_id")
            except jwt.PyJWTError:
                pass
        # Try header
        tid = request.headers.get("X-Tenant-ID")
        if tid:
            try:
                return int(tid)
            except ValueError:
                pass
        return None
```

The middleware runs before the FastAPI dependency resolution. The tenant is on `request.state.tenant_id` when the handler runs.

---

## 9. Common Pitfalls

### 9.1 Forgetting the tenant filter in a JOIN

```python
# ❌ A JOIN that crosses tenants
stmt = select(User, Post).join(Post)  # returns users and posts from ALL tenants
```

The RLS policy applies to each table individually. A JOIN reads from both; the policy filters both. So this is safe with RLS. But it returns the cartesian product, not the natural one. Always specify the JOIN condition.

### 9.2 Caching across tenants

```python
# ❌ A response cache that doesn't include the tenant
@cache(expire=60)
async def get_user_profile(user_id: int):
    return await db.get_user(user_id)  # tenant 1's user might be returned to tenant 2
```

The cache key must include the tenant ID:

```python
@cache(key=lambda tenant_id, user_id: f"profile:{tenant_id}:{user_id}", expire=60)
async def get_user_profile(tenant_id: int, user_id: int):
    return await db.get_user(user_id)
```

### 9.3 Background jobs without tenant scope

A Celery worker that calls `db.get_user(42)` without setting the tenant gets nothing (RLS rejects). Or worse, with the wrong tenant gets the wrong user. Background jobs must set their tenant context explicitly:

```python
# celery worker
@celery_app.task
def send_welcome_email(user_id: int, tenant_id: int):
    # Use a context manager that sets the tenant for the duration of the task
    with tenant_scope(tenant_id):
        user = db.get_user(user_id)
        send_email(user.email, ...)
```

### 9.4 Testing with the wrong tenant

```python
# ❌ A test that doesn't set the tenant
async def test_user_list():
    users = await uow.users.list()
    # Returns nothing if RLS is enabled and tenant is not set
```

Always set the tenant in test setup:

```python
@pytest_asyncio.fixture
async def uow_for_tenant():
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET LOCAL app.current_tenant_id = '1'"))
        yield UnitOfWork(session)
```

---

## 10. Código de Compresión

```python
"""
Compresión: Multi-Tenancy Patterns
Cubre: shared schema with RLS, schema-per-tenant, tenant resolution,
       cross-tenant admin, migrations.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# 1) Engine and session
engine = create_async_engine("postgresql+asyncpg://app_user:pass@localhost/db", pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# 2) Tenant resolution
async def get_tenant(
    request: Request,
    x_tenant_id: int | None = Header(default=None),
) -> int:
    if hasattr(request.state, "tenant_id") and request.state.tenant_id is not None:
        return request.state.tenant_id
    if x_tenant_id is not None:
        return x_tenant_id
    raise HTTPException(400, "Tenant not specified")


# 3) Tenant-scoped UoW
class TenantScopedUoW:
    def __init__(self, session: AsyncSession, tenant_id: int):
        self.session = session
        self.tenant_id = tenant_id
        self.users = UserRepository(session)

    async def __aenter__(self):
        # Begin transaction
        if not self.session.in_transaction():
            await self.session.begin()
        # Set tenant for RLS
        await self.session.execute(
            text("SET LOCAL app.current_tenant_id = :tid"),
            {"tid": str(self.tenant_id)},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            await self.session.rollback()
        await self.session.close()

    async def commit(self):
        await self.session.commit()


async def get_uow(tenant_id: int = Depends(get_tenant)) -> AsyncIterator[TenantScopedUoW]:
    async with SessionLocal() as session:
        async with TenantScopedUoW(session, tenant_id) as uow:
            yield uow


# 4) Handler
@app.get("/users")
async def list_users(uow: TenantScopedUoW = Depends(get_uow)):
    return await uow.users.list()  # RLS ensures only the tenant's users


# 5) RLS migration
def upgrade() -> None:
    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON users
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::INTEGER)
    """)
```

---

## Key Takeaways

- The default choice for B2B SaaS is **shared schema with PostgreSQL Row-Level Security**. RLS makes the database itself enforce tenant isolation, so a bug in application code that forgets the `tenant_id` filter is caught at the DB.
- The `SET LOCAL` pattern with `app.current_tenant_id` is the standard. The setting is transaction-scoped, so a connection returned to the pool has no tenant set.
- Cross-tenant operations (admin, reporting) need a **separate DB user with `BYPASSRLS`**. The application's normal user does not have it; the privilege is granted to the admin user only.
- Migrations for RLS are simple: enable RLS on each tenant-scoped table, define the policy, done.
- Migrations for schema-per-tenant are complex: run Alembic once per tenant. New tenant provisioning includes a schema bootstrap.
- Background jobs and caches need the tenant in the cache key and the tenant context set in the worker. A `tenant_scope(tenant_id)` context manager is the right pattern.
- The cross-tenant query trap is real. Code review must check that every query path includes the tenant filter or relies on RLS.
- Tests must set the tenant explicitly. SQLite does not support RLS, so integration tests run against PostgreSQL (testcontainers, GitHub Actions service containers, or a CI-managed instance).

## References

- [PostgreSQL — Row Security Policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [PostgreSQL — SET LOCAL](https://www.postgresql.org/docs/current/sql-set.html)
- [AWS — SaaS Tenant Isolation Strategies](https://docs.aws.amazon.com/whitepapers/latest/saas-tenant-isolation-strategies/saas-tenant-isolation-strategies.html)
- [Microsoft — Multitenant SaaS Database Tenancy Patterns](https://learn.microsoft.com/en-us/azure/azure-sql/database/saas-tenancy-app-design-patterns)
- [Citus — Row-Level Security in Multi-Tenant Apps](https://www.citusdata.com/blog/2018/12/27/postgres-row-level-security-by-example/)
