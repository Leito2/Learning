# 🏆 Capstone: SaaS Authentication System

## 🎯 Learning Objectives

- Build a production-grade multi-tenant authentication system end-to-end
- Tie together every pattern from the previous five notes: JWT, OAuth2, RBAC, multi-tenancy, MFA
- Implement login, registration, MFA enrollment, password reset, recovery, and admin reset flows
- Test the full stack with httpx.AsyncClient and Pytest
- Deploy with proper observability, rate limiting, and security headers

## Introduction

This capstone is the integration of every concept in the course. The system is a multi-tenant B2B SaaS with:
- Email + password authentication
- Optional TOTP-based MFA
- Optional WebAuthn passkeys
- OAuth2 login (Google)
- JWT access tokens with refresh-token rotation
- Per-tenant roles with RBAC
- Per-tenant data isolation via PostgreSQL RLS
- Recovery codes, admin reset, audit log

The codebase is intentionally small but complete. Reading it end-to-end is a review of the course. The patterns from the previous five notes are all used; if any are missing, that's a gap to fix.

---

## 1. The Architecture

```mermaid
flowchart LR
    A[User] --> B[Login UI]
    B --> C{Auth method}
    C -->|Email+Password| D[/auth/login]
    C -->|Google| E[/auth/google/login]
    C -->|Passkey| F[/auth/webauthn/login]
    D --> G{2FA enabled?}
    G -->|No| H[Access + Refresh]
    G -->|Yes| I[/auth/mfa/check]
    E --> H
    F --> H
    I --> J[Verify TOTP]
    J --> H
    H --> K[FastAPI handlers]
    K --> L[TenantContext]
    L --> M[PostgreSQL + RLS]
```

---

## 2. The Data Layer

### 2.1 Models

```python
# app/models/user.py
from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    hashed_password: Mapped[str | None] = mapped_column(String(255), default=None)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")
    is_superadmin: Mapped[bool] = mapped_column(default=False)
    mfa_enabled: Mapped[bool] = mapped_column(default=False)
    mfa_secret_encrypted: Mapped[bytes | None] = mapped_column(default=None)
    mfa_recovery_codes_hashed: Mapped[list[str]] = mapped_column(default=list)
    mfa_failed_attempts: Mapped[int] = mapped_column(default=0)
    mfa_lockout_until: Mapped[datetime | None] = mapped_column(default=None)
    memberships: Mapped[list["Membership"]] = relationship(back_populates="user", lazy="selectin")


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")
    memberships: Mapped[list["Membership"]] = relationship(back_populates="tenant")


class Membership(Base, TimestampMixin):
    __tablename__ = "memberships"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20), default="member")
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")

    user: Mapped["User"] = relationship(back_populates="memberships")
    tenant: Mapped["Tenant"] = relationship(back_populates="memberships")

    __table_args__ = (
        UniqueConstraint("user_id", "tenant_id", name="uq_membership_user_tenant"),
    )


class WebAuthnCredential(Base):
    __tablename__ = "webauthn_credentials"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    credential_id: Mapped[bytes] = mapped_column(unique=True)
    public_key: Mapped[bytes]
    sign_count: Mapped[int] = mapped_column(default=0)
    name: Mapped[str] = mapped_column(String(100))  # user-friendly name
    last_used_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AdminAuditLog(Base):
    __tablename__ = "admin_audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor_user_id: Mapped[int]
    action: Mapped[str]
    affected_tenant_id: Mapped[int | None] = mapped_column(default=None)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())


class WebAuthnChallenge(Base):
    __tablename__ = "webauthn_challenges"
    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(index=True)
    challenge: Mapped[str]
    purpose: Mapped[str]  # "register" or "authenticate"
    expires_at: Mapped[datetime]
```

### 2.2 Initial migration with RLS

```python
# alembic/versions/001_initial.py
def upgrade() -> None:
    # Create tables
    op.create_table("tenants", ...)
    op.create_table("users", ...)
    op.create_table("memberships", ...)
    op.create_table("webauthn_credentials", ...)
    op.create_table("admin_audit_log", ...)
    op.create_table("webauthn_challenges", ...)

    # Enable RLS on tenant-scoped tables
    for table in ("memberships", "webauthn_credentials"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                USING (
                    tenant_id = current_setting('app.current_tenant_id', TRUE)::INTEGER
                    OR current_setting('app.is_superadmin', TRUE) = 'true'
                )
        """)
```

The `OR is_superadmin` clause lets the system bypass RLS for superadmin operations by setting a different session variable.

---

## 3. The Security Utilities

```python
# app/core/security.py
from datetime import datetime, timedelta, timezone
from typing import Any
import bcrypt
import jwt
from cryptography.fernet import Fernet
from app.core.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(
    *,
    sub: int,
    tenant_id: int,
    role: str,
    expires_delta: timedelta = timedelta(minutes=15),
) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    jti = secrets.token_urlsafe(16)
    payload = {
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "sub": str(sub),
        "iat": now,
        "exp": now + expires_delta,
        "jti": jti,
        "tenant_id": tenant_id,
        "role": role,
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    return token, jti


def create_refresh_token(*, sub: int, family_id: str | None = None) -> tuple[str, str, str]:
    family_id = family_id or secrets.token_urlsafe(16)
    jti = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc)
    payload = {
        "iss": settings.JWT_ISSUER,
        "sub": str(sub),
        "iat": now,
        "exp": now + timedelta(days=7),
        "jti": jti,
        "family_id": family_id,
        "type": "refresh",
    }
    token = jwt.encode(payload, settings.JWT_REFRESH_SECRET, algorithm="HS256")
    return token, jti, family_id


# Encryption for TOTP secrets at rest
fernet = Fernet(settings.ENCRYPTION_KEY.encode())


def encrypt(data: str) -> bytes:
    return fernet.encrypt(data.encode())


def decrypt(data: bytes) -> str:
    return fernet.decrypt(data).decode()
```

---

## 4. The Middleware

```python
# app/middleware/tenant_auth.py
import jwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from sqlalchemy import select
from app.core.security import decrypt
from app.core.config import settings
from app.db.engine import SessionLocal
from app.models import User, Tenant, Membership


PUBLIC_PATHS = {"/health", "/auth/login", "/auth/register", "/auth/refresh", "/docs", "/openapi.json", "/redoc"}


class TenantAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        # Extract the JWT
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        try:
            payload = jwt.decode(
                auth[7:],
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
                audience=settings.JWT_AUDIENCE,
                issuer=settings.JWT_ISSUER,
            )
        except jwt.PyJWTError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)
        user_id = int(payload["sub"])
        tenant_id = payload["tenant_id"]
        # Validate the membership
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
            user = await session.get(User, membership.user_id)
            tenant = await session.get(Tenant, membership.tenant_id)
        # Attach the context
        request.state.tenant_context = TenantContext(
            user=user, tenant=tenant, membership=membership,
        )
        return await call_next(request)
```

---

## 5. The Auth API

### 5.1 Login

```python
# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from app.api.deps import get_uow
from app.core.security import (
    create_access_token, create_refresh_token,
    hash_password, verify_password,
)
from app.repositories.uow import UnitOfWork


router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    email: EmailStr
    password: str
    tenant_slug: str | None = None  # optional; default to user's primary tenant


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    mfa_required: bool = False
    mfa_token: str | None = None


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn, uow: UnitOfWork = Depends(get_uow)):
    # 1) Find the user
    user = await uow.users.get_by(email=payload.email)
    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(403, "Account disabled")
    # 2) Find the membership
    if payload.tenant_slug:
        membership = await uow.memberships.get_by(
            user_id=user.id, tenant_slug=payload.tenant_slug, is_active=True,
        )
    else:
        membership = await uow.memberships.get_primary_for_user(user.id)
    if not membership:
        raise HTTPException(403, "No active tenant membership")
    # 3) If MFA is enabled, return a challenge
    if user.mfa_enabled:
        mfa_token = create_mfa_challenge_token(user.id)
        return TokenOut(
            access_token="",
            refresh_token="",
            mfa_required=True,
            mfa_token=mfa_token,
        )
    # 4) No MFA: issue tokens
    access, _ = create_access_token(sub=user.id, tenant_id=membership.tenant_id, role=membership.role)
    refresh, _, _ = create_refresh_token(sub=user.id)
    return TokenOut(access_token=access, refresh_token=refresh)
```

### 5.2 MFA check

```python
class MFACheckIn(BaseModel):
    mfa_token: str
    code: str


@router.post("/mfa/check", response_model=TokenOut)
async def mfa_check(payload: MFACheckIn, uow: UnitOfWork = Depends(get_uow)):
    user_id = verify_mfa_challenge_token(payload.mfa_token)
    user = await uow.users.get(user_id)
    if not user:
        raise HTTPException(401, "Invalid challenge")
    # Lockout check
    if user.mfa_lockout_until and user.mfa_lockout_until > datetime.now(timezone.utc):
        raise HTTPException(429, "Account temporarily locked")
    # Verify
    secret = decrypt(user.mfa_secret_encrypted)
    if not pyotp.TOTP(secret).verify(payload.code, valid_window=1):
        user.mfa_failed_attempts += 1
        if user.mfa_failed_attempts >= 5:
            user.mfa_lockout_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        await uow.commit()
        raise HTTPException(401, "Invalid code")
    # Success
    user.mfa_failed_attempts = 0
    user.mfa_lockout_until = None
    await uow.commit()
    membership = await uow.memberships.get_primary_for_user(user.id)
    access, _ = create_access_token(sub=user.id, tenant_id=membership.tenant_id, role=membership.role)
    refresh, _, _ = create_refresh_token(sub=user.id)
    return TokenOut(access_token=access, refresh_token=refresh)
```

### 5.3 Refresh with rotation

```python
class RefreshIn(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=TokenOut)
async def refresh(payload: RefreshIn, uow: UnitOfWork = Depends(get_uow)):
    try:
        decoded = jwt.decode(
            payload.refresh_token,
            settings.JWT_REFRESH_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid refresh token")
    # Check that the refresh has not been used
    jti = decoded["jti"]
    if await uow.used_refresh_tokens.exists(jti=jti):
        # Reuse detected: revoke the entire family
        await revoke_token_family(decoded["family_id"])
        raise HTTPException(401, "Refresh token reuse detected")
    # Mark as used
    await uow.used_refresh_tokens.create(jti=jti, family_id=decoded["family_id"])
    # Issue new tokens
    user = await uow.users.get(int(decoded["sub"]))
    membership = await uow.memberships.get_by(
        user_id=user.id, tenant_id=decoded["tenant_id"], is_active=True,
    )
    access, _ = create_access_token(sub=user.id, tenant_id=membership.tenant_id, role=membership.role)
    new_refresh, _, _ = create_refresh_token(sub=user.id, family_id=decoded["family_id"])
    await uow.commit()
    return TokenOut(access_token=access, refresh_token=new_refresh)
```

### 5.4 MFA setup

```python
class MFASetupOut(BaseModel):
    secret: str
    qr_code_url: str
    recovery_codes: list[str]


@router.post("/mfa/setup", response_model=MFASetupOut)
async def setup_mfa(
    ctx: TenantContext = Depends(get_tenant_context),
    uow: UnitOfWork = Depends(get_uow),
):
    secret = pyotp.random_base32()
    recovery_codes = generate_recovery_codes()
    ctx.user.mfa_secret_encrypted = encrypt(secret)
    ctx.user.mfa_recovery_codes_hashed = [hash_recovery_code(c) for c in recovery_codes]
    ctx.user.mfa_enabled = False  # not enabled until verified
    await uow.commit()
    uri = pyotp.TOTP(secret).provisioning_uri(name=ctx.user.email, issuer_name="MyApp")
    return MFASetupOut(
        secret=secret,
        qr_code_url=uri,  # frontend renders this as a QR code
        recovery_codes=recovery_codes,
    )


@router.post("/mfa/verify-setup")
async def verify_mfa_setup(
    code: str,
    ctx: TenantContext = Depends(get_tenant_context),
    uow: UnitOfWork = Depends(get_uow),
):
    secret = decrypt(ctx.user.mfa_secret_encrypted)
    if not pyotp.TOTP(secret).verify(code, valid_window=1):
        raise HTTPException(401, "Invalid code")
    ctx.user.mfa_enabled = True
    await uow.commit()
    return {"status": "enabled"}
```

### 5.5 Password reset

```python
class PasswordResetRequest(BaseModel):
    email: EmailStr


@router.post("/password/reset-request")
async def password_reset_request(payload: PasswordResetRequest, uow: UnitOfWork = Depends(get_uow)):
    user = await uow.users.get_by(email=payload.email)
    if not user:
        # Don't reveal whether the email exists
        return {"status": "If the email is registered, a reset link has been sent"}
    token = create_password_reset_token(user.id)
    await send_password_reset_email(user.email, token)
    return {"status": "If the email is registered, a reset link has been sent"}


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


@router.post("/password/reset-confirm")
async def password_reset_confirm(payload: PasswordResetConfirm, uow: UnitOfWork = Depends(get_uow)):
    user_id = verify_password_reset_token(payload.token)
    user = await uow.users.get(user_id)
    if not user:
        raise HTTPException(400, "Invalid token")
    user.hashed_password = hash_password(payload.new_password)
    # Invalidate all refresh tokens for this user
    await uow.used_refresh_tokens.invalidate_all_for_user(user.id)
    await uow.commit()
    return {"status": "password reset"}
```

The reset token is a short-lived JWT (15 minutes) with a single purpose. The endpoint requires no authentication (the token is the proof of identity).

### 5.6 Admin MFA reset

```python
@router.post("/admin/users/{user_id}/mfa/reset")
async def admin_reset_mfa(
    user_id: int,
    admin_ctx: AdminContext = Depends(get_admin_context),
    uow: UnitOfWork = Depends(get_uow),
):
    if not admin_ctx.user.is_superadmin:
        raise HTTPException(403, "Superadmin only")
    # Verify the admin's own MFA (or use a separate audit-only flow)
    user = await uow.users.get(user_id)
    user.mfa_enabled = False
    user.mfa_secret_encrypted = None
    user.mfa_recovery_codes_hashed = []
    # Audit log
    await uow.admin_audit_log.create(
        actor_user_id=admin_ctx.user.id,
        action="mfa_reset",
        affected_tenant_id=user.primary_tenant_id,
        details={"target_user_id": user_id},
    )
    # Notify
    await send_mfa_reset_notification(user.email)
    await uow.commit()
    return {"status": "reset"}
```

---

## 6. The Tenant Context Dependency

```python
# app/api/deps.py
from fastapi import Request, HTTPException
from app.models.membership import Membership, TenantContext


async def get_tenant_context(request: Request) -> TenantContext:
    ctx = getattr(request.state, "tenant_context", None)
    if ctx is None:
        raise HTTPException(401, "Not authenticated or no tenant")
    return ctx


TenantCtx = Annotated[TenantContext, Depends(get_tenant_context)]


def require_role(*roles: str):
    async def checker(ctx: TenantContext = Depends(get_tenant_context)) -> TenantContext:
        if ctx.membership.role not in roles:
            raise HTTPException(403, f"Role {ctx.membership.role} not in {roles}")
        return ctx
    return checker


def require_permission(*permissions: str):
    async def checker(ctx: TenantContext = Depends(get_tenant_context)) -> TenantContext:
        perms = ROLE_PERMISSIONS.get(ctx.membership.role, set())
        if "*" in perms:
            return ctx
        for p in permissions:
            if p not in perms:
                raise HTTPException(403, f"Permission {p} required")
        return ctx
    return checker
```

---

## 7. Tests

### 7.1 Test fixtures

```python
# tests/conftest.py
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import text
from app.db.base import Base
from app.db.engine import engine, SessionLocal
from app.main import app
from app.repositories.uow import UnitOfWork


@pytest_asyncio.fixture
async def db_engine():
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    await test_engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine):
    SessionTesting = async_sessionmaker(db_engine, expire_on_commit=False)
    from app.api.deps import get_uow

    async def override_get_uow():
        async with SessionTesting() as session:
            uow = UnitOfWork(session)
            async with uow:
                yield uow
    app.dependency_overrides[get_uow] = override_get_uow
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
```

### 7.2 Auth flow tests

```python
@pytest.mark.asyncio
async def test_register_login_logout(client):
    # Register
    response = await client.post(
        "/auth/register",
        json={
            "email": "alice@example.com",
            "name": "Alice",
            "password": "secret123",
            "tenant_name": "Test Co",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"]


@pytest.mark.asyncio
async def test_login_with_mfa(client, user_with_mfa):
    # First step: password
    response = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "secret123"},
    )
    assert response.status_code == 200
    assert response.json()["mfa_required"] is True
    mfa_token = response.json()["mfa_token"]
    # Generate a valid TOTP code
    import pyotp
    code = pyotp.TOTP(user_with_mfa.mfa_secret).now()
    # Second step: TOTP
    response = await client.post(
        "/auth/mfa/check",
        json={"mfa_token": mfa_token, "code": code},
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


@pytest.mark.asyncio
async def test_refresh_token_reuse_detected(client, user):
    # Login
    response = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "secret123"},
    )
    refresh = response.json()["refresh_token"]
    # First refresh: success
    response = await client.post("/auth/refresh", json={"refresh_token": refresh})
    assert response.status_code == 200
    # Second refresh with the same token: detected
    response = await client.post("/auth/refresh", json={"refresh_token": refresh})
    assert response.status_code == 401
    assert "reuse" in response.json()["detail"]
```

### 7.3 RBAC tests

```python
@pytest.mark.asyncio
async def test_member_cannot_access_admin_endpoint(client, member_token):
    response = await client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {member_token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_access_admin_endpoint(client, admin_token):
    response = await client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
```

### 7.4 Integration test (real PostgreSQL)

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_rls_isolates_tenants():
    """Run against a real PostgreSQL with RLS enabled."""
    # Create two tenants
    async with engine.begin() as conn:
        await conn.execute(text("INSERT INTO tenants (name, slug) VALUES ('T1', 't1'), ('T2', 't2')"))
        # Create users in each tenant
        # Create memberships
    # Verify that tenant 1 cannot see tenant 2's data
    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET LOCAL app.current_tenant_id = '1'"))
            result = await session.execute(text("SELECT COUNT(*) FROM memberships"))
            assert result.scalar() == 1
```

---

## 8. The FastAPI App

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.api import auth, mfa, password, webauthn
from app.db.engine import engine
from app.middleware.tenant_auth import TenantAuthMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan, title="SaaS Auth API")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TenantAuthMiddleware)

app.include_router(auth.router)
app.include_router(mfa.router)
app.include_router(password.router)
app.include_router(webauthn.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 9. Deployment Checklist

| Concern | Where | Verified by |
|---------|-------|-------------|
| HTTPS only | Reverse proxy (Traefik) | TLS cert test |
| Rate limiting on /auth/login | Nginx/Traefik or slowapi | Load test |
| Bcrypt for passwords | `app/core/security.py` | Static analysis |
| TOTP secret encrypted at rest | `app/core/security.py` | DB dump inspection |
| WebAuthn origin pinned | `verify_registration_response` | Static analysis |
| JWT short-lived | `create_access_token` | Static analysis |
| Refresh token rotation | `refresh` endpoint | Test `test_refresh_token_reuse_detected` |
| Recovery codes hashed | `hash_recovery_code` | DB dump inspection |
| MFA lockout | `mfa_check` | Test the threshold |
| Audit log on admin actions | `admin_reset_mfa` | Test the log entry |
| Tenant isolation | RLS migration | Test `test_rls_isolates_tenants` |
| Trusted hosts | `TrustedHostMiddleware` | Test with bad host header |
| CORS | `CORSMiddleware` | Test from allowed/blocked origins |
| Security headers | Middleware | Curl with `-I` |

---

## 10. What This Capstone Demonstrates

The capstone uses every pattern from the course:

| Note | Pattern | Where in capstone |
|------|---------|------------------|
| 01 JWT | Access + refresh with rotation, lockout, recovery codes | `app/api/auth.py`, `password.py` |
| 02 OAuth2 | Google login flow (authlib) | `app/api/auth.py` |
| 03 RBAC | Per-tenant roles, scope checks, policy class | `app/api/deps.py` |
| 04 Multi-Tenant | Membership table, tenant context middleware | `app/middleware/tenant_auth.py` |
| 05 MFA | TOTP, recovery codes, lockout, admin reset | `app/api/mfa.py` |
| 06 (this note) | End-to-end integration | The whole app |

A new developer reading the capstone top-to-bottom gets a working mental model of the entire stack. A new developer reading just the auth handler, the middleware, or the MFA endpoint gets a deep dive into one concern.

---

## Key Takeaways

- A production auth system is many small systems wired together: JWT, OAuth2, MFA, RBAC, multi-tenancy. Each is a separate concern with its own patterns.
- The middleware is the chokepoint. Every request passes through it; the tenant context is established once and propagated.
- Token rotation and reuse detection catch stolen refresh tokens. A token that never expires is a credential waiting to be stolen.
- MFA is mandatory for any production B2B service. The 99.9% reduction in account takeover is the cheapest security win available.
- Recovery codes are the second factor of the second factor. Users lose devices; the recovery codes are the safety net.
- The audit log is the trail of accountability. Every admin action is logged with the actor, the action, and the affected entity.
- Tests are layered: handler tests with mocked UoW, repository tests with SQLite, integration tests with PostgreSQL and RLS. Each layer catches a different class of bug.
- The production checklist is a security gate. Every item must be verified before the service handles real users.

## References

- [OWASP ASVS 4.0](https://owasp.org/www-project-application-security-verification-standard/)
- [RFC 6749 — OAuth 2.0](https://www.rfc-editor.org/rfc/rfc6749)
- [RFC 6238 — TOTP](https://www.rfc-editor.org/rfc/rfc6238)
- [W3C WebAuthn Level 2](https://www.w3.org/TR/webauthn-2/)
- [FastAPI Security — OAuth2 with JWT](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
- [Authlib Documentation](https://authlib.org/)
- [pyotp Documentation](https://pyauth.github.io/pyotp/)
- [py_webauthn](https://github.com/duo-labs/py_webauthn)
- [NIST SP 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [PostgreSQL Row Security Policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
