# 🏢 Multi-Tenant Authentication

## 🎯 Learning Objectives

- Identify the four patterns of multi-tenant authentication
- Implement a tenant context that flows from login through every request
- Handle users who belong to multiple tenants with different roles per tenant
- Build superadmin flows that span tenants safely
- Avoid the six most common multi-tenant auth mistakes

## Introduction

A multi-tenant service serves many customers from one codebase. The previous notes in this course covered auth in the single-tenant case: one user, one role, one permission set. Multi-tenancy adds a layer: the same user might have an account in tenant A and a different role in tenant B. The user might log in once and have a session that includes both tenants, or might log in to a specific tenant at a time.

This note covers the four patterns of multi-tenant auth, the data model for users-tenant membership, and the FastAPI patterns for tenant-scoped authentication and authorization. The multi-tenancy data layer (RLS, schema-per-tenant) is covered in [[../38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/06 - Multi-Tenancy Patterns|the SQLAlchemy course]]; this note focuses on the auth layer.

The patterns assume PostgreSQL with RLS for tenant isolation, the standard choice for B2B SaaS. SQLite or MySQL would require application-level enforcement (covered at the end).

---

## 1. The Four Patterns

### 1.1 Pattern A: Single-tenant user, single tenant at a time

The simplest case. A user belongs to exactly one tenant. The login establishes the tenant context; subsequent requests are scoped to that tenant.

```
User logs in → tenant_id stored in the JWT → all queries scoped to that tenant
```

Best for: small businesses, single-organization users (most B2B SaaS).

### 1.2 Pattern B: Multi-tenant user, single tenant per session

A user belongs to multiple tenants but logs in to one at a time. The login UI shows "log in to Acme Corp" vs "log in to Globex"; the JWT carries one tenant_id.

```
User "alice" belongs to tenants 1 and 2.
Login UI: "log in to Acme Corp" → JWT for tenant 1.
All subsequent requests are scoped to tenant 1.
```

Best for: consultants, freelancers with multiple clients, employees at multi-org conglomerates.

### 1.3 Pattern C: Multi-tenant user, multiple tenants per session

A user has a "home tenant" and can switch to other tenants they belong to. The session is one identity; the active tenant changes.

```
User "alice" has a session with sub=alice_id.
She can switch the active tenant via /auth/switch-tenant.
The JWT (or a separate tenant claim) carries the current active_tenant_id.
```

Best for: power users, enterprise admins, support staff.

### 1.4 Pattern D: Federated identity, no "tenant" per se

The user is identified by an external IdP (Google Workspace, Okta, Azure AD). The "tenant" is the IdP tenant. The service trusts the IdP's tenant claim and provisions local accounts on demand.

```
User logs in with Google → Google returns a tenant claim → service provisions a user
record (or matches an existing one) and scopes the session to that tenant.
```

Best for: B2B SaaS that integrates with enterprise IdPs.

### 1.5 Choosing a pattern

| Pattern | When | Complexity | Examples |
|---------|------|-----------|----------|
| A: Single-tenant user | Most B2B SaaS | Low | Slack, Notion |
| B: Single session, multiple tenants | Freelancers, consultants | Medium | Upwork, Fiverr |
| C: Switchable active tenant | Enterprise power users | High | GitHub, GitLab |
| D: Federated identity | Enterprise SSO | High | Salesforce, Workday |

For a new B2B SaaS, **Pattern A** is the right starting point. Pattern B is added when users need to belong to multiple organizations. Pattern C is a power-user feature for enterprise tiers. Pattern D is for enterprise SSO.

---

## 2. The Data Model

### 2.1 User-tenant membership

```python
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    # Note: no `tenant_id` here. The user exists independently of any tenant.


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    slug: Mapped[str] = mapped_column(String(50), unique=True)


class Membership(Base):
    __tablename__ = "memberships"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20), default="member")
    is_active: Mapped[bool] = mapped_column(default=True)
    invited_at: Mapped[datetime] = mapped_column(server_default=func.now())
    joined_at: Mapped[datetime | None] = mapped_column(default=None)

    __table_args__ = (
        UniqueConstraint("user_id", "tenant_id", name="uq_membership_user_tenant"),
    )
```

A user can have many memberships. Each membership has a role (specific to that tenant). The user's "global" role is replaced by per-tenant roles.

### 2.2 Why a separate `Membership` table

A naive design puts `tenant_id` on the `User` table. This forces one user = one tenant. The membership table allows many-to-many: a user belongs to many tenants, with a different role per tenant.

```python
# ❌ Naive: one user, one tenant
class User(Base):
    tenant_id: Mapped[int]  # a user can be in only one tenant

# ✅ Membership table: one user, many tenants
class Membership(Base):
    user_id: Mapped[int]
    tenant_id: Mapped[int]
    role: Mapped[str]
```

### 2.3 The role-per-tenant

Each membership has its own role. A user can be an admin in tenant A and a viewer in tenant B.

```python
@dataclass
class TenantContext:
    user: User
    tenant: Tenant
    membership: Membership  # has the role

    @property
    def role(self) -> str:
        return self.membership.role

    @property
    def permissions(self) -> set[str]:
        return ROLE_PERMISSIONS.get(self.role, set())
```

A handler receives a `TenantContext` instead of a `User`. The role is the per-tenant role, not a global one.

---

## 3. The Tenant Resolution Flow

### 3.1 The middleware (Pattern A)

The middleware reads the JWT, extracts the `tenant_id`, validates the membership, and attaches the `TenantContext` to the request.

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


PUBLIC_PATHS = {"/health", "/auth/login", "/auth/register", "/docs", "/openapi.json"}


class TenantAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        # 1) Extract the JWT
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        token = auth[7:]
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        except jwt.PyJWTError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)
        user_id = int(payload["sub"])
        tenant_id = payload["tenant_id"]
        # 2) Validate the membership
        async with SessionLocal() as session:
            membership = (await session.execute(
                select(Membership).where(
                    Membership.user_id == user_id,
                    Membership.tenant_id == tenant_id,
                    Membership.is_active == True,
                )
            )).scalar_one_or_none()
            if not membership:
                return JSONResponse({"detail": "No active membership"}, status_code=403)
            user = await session.get(User, user_id)
            tenant = await session.get(Tenant, tenant_id)
        # 3) Attach to request state
        request.state.tenant_context = TenantContext(
            user=user, tenant=tenant, membership=membership,
        )
        return await call_next(request)
```

The middleware opens its own session (separate from the request's session) to validate. The `TenantContext` is on `request.state.tenant_context` for the duration of the request.

### 3.2 The dependency (Pattern B)

For Pattern B (multi-tenant, single session), the tenant is selected at login time:

```python
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_slug: str  # which tenant the user wants to log into


@router.post("/auth/login")
async def login(payload: LoginRequest):
    user = await authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    # Find the membership for the requested tenant
    membership = await uow.memberships.get_by(user_id=user.id, tenant_slug=payload.tenant_slug)
    if not membership or not membership.is_active:
        raise HTTPException(403, "You do not have access to this tenant")
    # Issue a JWT scoped to that tenant
    token, jti = create_access_token(
        sub=user.id,
        tenant_id=membership.tenant_id,
        role=membership.role,
    )
    return {"access_token": token, "tenant_id": membership.tenant_id}
```

The user explicitly says "log in to tenant X". The middleware (above) validates that they have a membership.

### 3.3 The switcher (Pattern C)

For Pattern C (switchable active tenant), the user logs in to a "home" tenant and can switch to any other tenant they belong to.

```python
@router.post("/auth/switch-tenant")
async def switch_tenant(
    tenant_id: int,
    ctx: TenantContext = Depends(get_tenant_context),
):
    # Verify the user has a membership in the new tenant
    new_membership = await uow.memberships.get_by(
        user_id=ctx.user.id, tenant_id=tenant_id,
    )
    if not new_membership or not new_membership.is_active:
        raise HTTPException(403, "No access to this tenant")
    # Issue a new JWT with the new tenant
    token, _ = create_access_token(
        sub=ctx.user.id,
        tenant_id=tenant_id,
        role=new_membership.role,
    )
    return {"access_token": token, "tenant_id": tenant_id}
```

The endpoint requires a current `TenantContext` (i.e., the user is already logged in), verifies the new membership, and issues a fresh JWT.

### 3.4 The dependency

```python
from fastapi import Request


async def get_tenant_context(request: Request) -> TenantContext:
    ctx = getattr(request.state, "tenant_context", None)
    if ctx is None:
        raise HTTPException(401, "Not authenticated or no tenant")
    return ctx


TenantCtx = Annotated[TenantContext, Depends(get_tenant_context)]


@router.get("/projects")
async def list_projects(ctx: TenantCtx, uow: UnitOfWork = Depends(get_uow)):
    return await uow.projects.list(tenant_id=ctx.tenant.id)
```

The handler doesn't see a `User`; it sees a `TenantContext`. The role is the per-tenant role.

---

## 4. The RLS Integration

### 4.1 Setting the tenant at session start

The data layer (covered in [[../38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/06 - Multi-Tenancy Patterns|note 06 of the SQLAlchemy course]]) uses PostgreSQL Row-Level Security. The `SET LOCAL` pattern sets the active tenant for the session:

```python
class TenantAwareUoW:
    def __init__(self, session: AsyncSession, tenant_id: int):
        self.session = session
        self.tenant_id = tenant_id

    async def __aenter__(self):
        # Begin a transaction
        if not self.session.in_transaction():
            await self.session.begin()
        # Set the RLS variable
        await self.session.execute(
            text("SET LOCAL app.current_tenant_id = :tid"),
            {"tid": str(self.tenant_id)},
        )
        return self
```

### 4.2 The wiring

The middleware sets the tenant on `request.state.tenant_id`. The `get_uow` dependency reads it and uses it for the RLS variable:

```python
async def get_uow(request: Request) -> AsyncIterator["TenantAwareUoW"]:
    tenant_id = request.state.tenant_context.tenant.id
    async with SessionLocal() as session:
        async with TenantAwareUoW(session, tenant_id) as uow:
            yield uow
```

The handler never sees the session or the RLS variable. It just uses the UoW; the UoW handles the tenant scoping.

---

## 5. Superadmin Patterns

### 5.1 The superadmin role

A superadmin (often called `system_admin` or `global_admin`) is a user that can act across tenants. There are two designs:

| Design | Implementation | Trade-off |
|--------|----------------|-----------|
| Special role on the `User` | `user.is_superadmin = True` | Simple, but no per-tenant audit |
| Membership in a "system" tenant | `membership(tenant_id=0, role="superadmin")` | Consistent with the model, can use the same authz |

The "system tenant" pattern is more consistent. The superadmin has a membership in tenant 0 (a special tenant) with role `superadmin`. The RLS policy bypasses tenant filtering for the system tenant.

### 5.2 The superadmin DB user

The application uses a regular DB user (`app_user`) for tenant-scoped work. Admin operations (cross-tenant reports, schema migrations) use a separate user (`app_admin`) with `BYPASSRLS`:

```python
# app/db/admin.py
admin_engine = create_async_engine(settings.ADMIN_DATABASE_URL, ...)
AdminSessionLocal = async_sessionmaker(admin_engine, expire_on_commit=False)


@router.get("/admin/all-tenants")
async def list_all_tenants(
    admin_uow: UnitOfWork = Depends(get_admin_uow),
    user: User = Depends(require_superadmin),
):
    return await admin_uow.tenants.list()
```

The handler requires both `user.is_superadmin` AND uses the `admin_uow` (with the `BYPASSRLS` user). Two layers of protection.

### 5.3 The superadmin audit log

Every superadmin action is logged with the user, the action, the affected tenant, and the timestamp. The log is append-only and stored in a table that even superadmins cannot modify:

```python
class AdminAuditLog(Base):
    __tablename__ = "admin_audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor_user_id: Mapped[int]
    action: Mapped[str]
    affected_tenant_id: Mapped[int | None]
    details: Mapped[dict] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())


async def log_admin_action(user_id: int, action: str, tenant_id: int | None, details: dict):
    async with AdminSessionLocal() as session:
        async with session.begin():
            session.add(AdminAuditLog(
                actor_user_id=user_id,
                action=action,
                affected_tenant_id=tenant_id,
                details=details,
            ))
```

The audit log is the trail of accountability. When a superadmin does something they shouldn't have, the log is the evidence.

---

## 6. Invitations and Onboarding

### 6.1 The invitation flow

A tenant admin invites a new user. The user receives an email with a magic link; clicking it creates a new user (or adds a membership to an existing one).

```python
class Invitation(Base):
    __tablename__ = "invitations"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    role: Mapped[str] = mapped_column(String(20))
    token: Mapped[str] = mapped_column(String(64), unique=True)
    invited_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime]
    accepted_at: Mapped[datetime | None] = mapped_column(default=None)


@router.post("/invitations")
async def create_invitation(
    payload: InvitationCreate,
    ctx: TenantContext = Depends(get_tenant_context),
):
    # Only admins can invite
    if ctx.role != "admin":
        raise HTTPException(403, "Only admins can invite users")
    # Generate a secure token
    token = secrets.token_urlsafe(32)
    invitation = Invitation(
        email=payload.email,
        tenant_id=ctx.tenant.id,
        role=payload.role,
        token=token,
        invited_by_user_id=ctx.user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    await uow.invitations.create(invitation)
    await send_invitation_email(payload.email, token)
    await uow.commit()
    return {"status": "sent"}


@router.post("/invitations/accept")
async def accept_invitation(
    payload: InvitationAccept,
    password: str,
):
    invitation = await uow.invitations.get_by(token=payload.token)
    if not invitation or invitation.expires_at < datetime.now(timezone.utc):
        raise HTTPException(400, "Invalid or expired invitation")
    if invitation.accepted_at is not None:
        raise HTTPException(400, "Invitation already accepted")
    # Find or create the user
    user = await uow.users.get_by(email=invitation.email)
    if not user:
        user = await uow.users.create(
            email=invitation.email,
            hashed_password=hash_password(password),
        )
    # Add the membership
    await uow.memberships.create(
        user_id=user.id,
        tenant_id=invitation.tenant_id,
        role=invitation.role,
        joined_at=datetime.now(timezone.utc),
    )
    invitation.accepted_at = datetime.now(timezone.utc)
    await uow.commit()
    # Issue a token
    token, _ = create_access_token(sub=user.id, tenant_id=invitation.tenant_id, role=invitation.role)
    return {"access_token": token}
```

The invitation token is a one-time use; the email link expires; the user sets a password at acceptance.

### 6.2 IdP-driven onboarding (Pattern D)

For federated identity, the IdP is the source of truth. The service trusts the IdP's user info and provisions a local user on first login.

```python
@router.post("/auth/sso/google/callback")
async def google_sso_callback(code: str, request: Request):
    # Exchange code for tokens with Google
    google_user_info = await exchange_google_code(code)
    email = google_user_info["email"]
    # Google Workspace provides the tenant domain
    domain = email.split("@")[1]
    # Find the tenant by domain
    tenant = await uow.tenants.get_by(google_workspace_domain=domain)
    if not tenant:
        raise HTTPException(403, "Tenant not registered for this Google Workspace")
    # Find or create the user
    user = await uow.users.get_by(email=email)
    if not user:
        user = await uow.users.create(email=email, name=google_user_info["name"])
    # Add a membership if not present
    membership = await uow.memberships.get_by(user_id=user.id, tenant_id=tenant.id)
    if not membership:
        await uow.memberships.create(user_id=user.id, tenant_id=tenant.id, role="member")
    # Issue a JWT
    token, _ = create_access_token(
        sub=user.id, tenant_id=tenant.id, role=membership.role,
    )
    return {"access_token": token}
```

The IdP is the source of identity; the service is the source of permissions. The local user record is provisioned on demand.

---

## 7. The Six Multi-Tenant Auth Mistakes

### 7.1 Storing the tenant_id in the User, not the Membership

```python
# ❌ One user, one tenant — no way to support multi-tenant users
class User(Base):
    tenant_id: Mapped[int]

# ✅ Use Membership
class User(Base):
    pass
class Membership(Base):
    user_id: Mapped[int]
    tenant_id: Mapped[int]
    role: Mapped[str]
```

### 7.2 Trusting the tenant_id from the request body

```python
# ❌ The client picks the tenant
@router.get("/projects")
async def list_projects(
    tenant_id: int = Query(...),  # from the client
    user: User = Depends(get_current_user),
):
    return await uow.projects.list(tenant_id=tenant_id)

# ✅ The server picks the tenant from the JWT
@router.get("/projects")
async def list_projects(ctx: TenantContext = Depends(get_tenant_context)):
    return await uow.projects.list(tenant_id=ctx.tenant.id)
```

### 7.3 Leaking the active tenant in the token to other tenants

If the user is logged in to tenant A and the token also contains a list of "other tenants they belong to", the client could use the other tenants' tenant_id to access data. The fix: the token carries the **active** tenant only. Switching tenants requires a fresh login or a switcher call that issues a new token.

### 7.4 Storing the role globally, not per tenant

```python
# ❌ A user is an admin everywhere because they're an admin in one tenant
if user.role == "admin":
    # user can do admin things in any tenant

# ✅ The role is per membership
if ctx.role == "admin":  # admin in THIS tenant
    ...
```

### 7.5 Letting a tenant admin promote themselves to system admin

```python
# ❌ A tenant admin sets is_superadmin = True
user.is_superadmin = True  # now they can act across tenants
```

The `is_superadmin` flag should be set only by another superadmin, not by the user themselves or by a tenant admin. The `User` model should not even expose this column to the tenant-scoped UoW.

### 7.6 Not invalidating tokens when a user is removed from a tenant

```python
# ❌ A user is removed from a tenant but their token is still valid
# They can continue to act in the tenant until the token expires
```

When a membership is deleted, invalidate the user's refresh tokens for that tenant. The user's tokens for other tenants are unaffected.

```python
async def remove_membership(user_id: int, tenant_id: int):
    # 1) Delete the membership
    async with session.begin():
        await session.execute(
            delete(Membership).where(
                Membership.user_id == user_id,
                Membership.tenant_id == tenant_id,
            )
        )
    # 2) Invalidate refresh tokens for this user+tenant
    # (The access token will expire on its own; refresh tokens are the long-lived risk)
    await invalidate_refresh_tokens_for_user_tenant(user_id, tenant_id)
```

---

## 8. Código de Compresión

```python
"""
Compresión: Multi-Tenant Authentication
Covers: membership data model, tenant context flow, RLS integration,
       superadmin, invitations, SSO.
"""
from datetime import datetime, timedelta, timezone
import secrets
import jwt
from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import ForeignKey, String, select, delete
from sqlalchemy.orm import Mapped, mapped_column, relationship


# 1) Data model
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str]
    hashed_password: Mapped[str]
    memberships: Mapped[list["Membership"]] = relationship(back_populates="user")


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    slug: Mapped[str] = mapped_column(String(50), unique=True)
    google_workspace_domain: Mapped[str | None] = mapped_column(String(255), default=None)
    memberships: Mapped[list["Membership"]] = relationship(back_populates="tenant")


class Membership(Base):
    __tablename__ = "memberships"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20), default="member")
    is_active: Mapped[bool] = mapped_column(default=True)
    user: Mapped["User"] = relationship(back_populates="memberships")
    tenant: Mapped["Tenant"] = relationship(back_populates="memberships")


# 2) The TenantContext (passed to handlers)
from dataclasses import dataclass


@dataclass
class TenantContext:
    user: User
    tenant: Tenant
    membership: Membership

    @property
    def role(self) -> str:
        return self.membership.role

    @property
    def permissions(self) -> set[str]:
        return ROLE_PERMISSIONS.get(self.role, set())


# 3) Middleware
PUBLIC_PATHS = {"/health", "/auth/login", "/auth/register", "/docs"}


class TenantAuthMiddleware:
    async def __call__(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        try:
            payload = jwt.decode(auth[7:], "secret", algorithms=["HS256"])
        except jwt.PyJWTError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)
        async with SessionLocal() as session:
            membership = (await session.execute(
                select(Membership).where(
                    Membership.user_id == int(payload["sub"]),
                    Membership.tenant_id == payload["tenant_id"],
                    Membership.is_active == True,
                )
            )).scalar_one_or_none()
            if not membership:
                return JSONResponse({"detail": "No active membership"}, status_code=403)
            user = await session.get(User, membership.user_id)
            tenant = await session.get(Tenant, membership.tenant_id)
        request.state.tenant_context = TenantContext(user=user, tenant=tenant, membership=membership)
        return await call_next(request)


# 4) The dependency
async def get_tenant_context(request: Request) -> TenantContext:
    ctx = getattr(request.state, "tenant_context", None)
    if ctx is None:
        raise HTTPException(401, "Not authenticated or no tenant")
    return ctx


TenantCtx = Annotated[TenantContext, Depends(get_tenant_context)]


# 5) Handler
@router.get("/projects")
async def list_projects(ctx: TenantCtx, uow: UnitOfWork = Depends(get_uow)):
    # RLS already scopes the query; no explicit tenant filter needed
    return await uow.projects.list(tenant_id=ctx.tenant.id)
```

---

## Key Takeaways

- A multi-tenant user has a separate `Membership` row for each tenant they belong to. The role is per-tenant, not global.
- The active tenant is established at login and carried in the JWT. Subsequent requests are scoped to that tenant.
- The `TenantContext` (user + tenant + membership) flows from the middleware through the dependency to the handler. Handlers never see a raw `User`; they see a `TenantContext`.
- PostgreSQL RLS enforces tenant isolation at the database. The application sets `app.current_tenant_id` per session; queries cannot return rows from other tenants.
- Superadmins are a special case. They use a separate DB user with `BYPASSRLS`, and every action is logged.
- Trust the JWT, not the request body, for the tenant identity. The client picks the active tenant at login; the server enforces it on every request.
- The role is per-tenant. A user can be an admin in tenant A and a viewer in tenant B. The token carries the active tenant's role; the membership table is the source of truth.
- When a user is removed from a tenant, invalidate their refresh tokens for that tenant. The access token will expire on its own; the refresh token is the long-lived risk.
- Federated identity (Google Workspace, Okta) is the modern way to onboard enterprise tenants. The IdP is the source of identity; the service is the source of permissions.

## References

- [Auth0 — Multi-Tenant Applications](https://auth0.com/docs/get-started/architecture-scenarios/multiple-architecture)
- [Okta — Multi-Tenant Identity](https://developer.okta.com/docs/concepts/multi-tenancy/)
- [NIST SP 800-162 — ABAC](https://nvlpubs.nist.gov/nistpubs/specialpublications/NIST.sp.800-162.pdf)
- [PostgreSQL — Row Security Policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [Google Workspace — SSO with OIDC](https://developers.google.com/identity/openid-connect/openid-connect)
- [Auth0 — Tenant Member Roles](https://auth0.com/docs/manage-users/organizations/configure-organizations/use-org-name-authentication-api)
