# 🔑 JWT Deep Dive

## 🎯 Learning Objectives

- Understand the JWT structure: header, payload, signature, and what each part does
- Choose the right signing algorithm: HS256 vs RS256 vs EdDSA
- Implement access tokens with proper expiration and refresh-token rotation
- Build a session-management layer: revocation, sliding sessions, refresh token reuse detection
- Avoid the seven most common JWT pitfalls in production

## Introduction

JSON Web Tokens (JWTs) are the dominant token format for stateless authentication. A JWT is a self-contained, signed, base64url-encoded JSON object. The server issues a token to a client; the client presents the token on subsequent requests; the server validates the signature and reads the claims. The server holds no session state; the token itself carries identity.

The simplicity of JWTs is also their danger. A JWT signed with `alg: none` accepts any token as valid. A JWT with a long expiration is a stolen-credential goldmine. A JWT that includes sensitive data in the payload leaks that data to anyone who can read the token. The format is a tool; using it well requires understanding the trade-offs.

This note covers the full lifecycle: signing, validating, refreshing, and revoking JWTs. It also covers the seven pitfalls that turn a "secure" JWT implementation into a vulnerability.

---

## 1. The JWT Structure

### 1.1 Three parts, two dots

A JWT has three parts separated by dots:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

Decoded:

```json
// Header
{
  "alg": "HS256",
  "typ": "JWT"
}

// Payload
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022
}

// Signature
HMACSHA256(
  base64url(header) + "." + base64url(payload),
  secret
)
```

The **signature** is computed over the header and payload. The receiver recomputes the signature with the same secret; if it matches, the token was not tampered with.

### 1.2 Standard claims (RFC 7519)

| Claim | Name | Meaning |
|-------|------|---------|
| `iss` | Issuer | Who issued the token (e.g., your auth service) |
| `sub` | Subject | The user ID (the token's owner) |
| `aud` | Audience | Who the token is for (e.g., your API) |
| `exp` | Expiration | When the token stops being valid |
| `nbf` | Not Before | When the token becomes valid |
| `iat` | Issued At | When the token was issued |
| `jti` | JWT ID | Unique identifier for this token (for revocation) |

```python
payload = {
    "iss": "https://auth.example.com",
    "sub": str(user.id),
    "aud": "https://api.example.com",
    "iat": now,
    "exp": now + 900,  # 15 minutes
    "jti": str(uuid4()),
    "tenant_id": user.tenant_id,
    "role": user.role,
}
```

### 1.3 What JWTs are NOT

A common misconception: JWTs are encrypted. They are not. The header and payload are **base64url-encoded**, not encrypted. Anyone with the token can read every claim. Sensitive data (passwords, PII, financial data) must not be in the payload.

The signature is for **integrity** (no one modified the token), not **confidentiality** (no one read the token). For confidentiality, encrypt the token (JWE) or use a session ID and store the data server-side.

---

## 2. Signing Algorithms

### 2.1 The three families

| Family | Examples | Use case |
|--------|----------|----------|
| Symmetric (HMAC) | HS256, HS384, HS512 | Single trusted party signs and verifies |
| Asymmetric (RSA) | RS256, RS384, RS512 | Multiple verifiers, only one signer |
| Asymmetric (ECDSA / EdDSA) | ES256, ES384, EdDSA | Same as RSA but smaller, faster |

### 2.2 When to use which

**HS256 (HMAC-SHA256)**: the secret is one shared string. Use it when the issuer and verifier are the same service (most apps). Simple, fast, no key distribution.

**RS256 (RSA-SHA256)**: the issuer has a private key; verifiers have the public key. Use it when:
- Multiple services verify the token (microservices).
- The verifier is a third party (an external client).
- You need to verify without holding the signing key.

**ES256 (ECDSA-P256)**: like RS256 but with elliptic-curve keys. Smaller signatures, faster verification. The modern default for new systems that need asymmetric signing.

**EdDSA (Ed25519)**: the cutting-edge choice. Even smaller, even faster, well-audited libraries. Some legacy libraries don't support it yet.

### 2.3 The `alg: none` vulnerability

Early JWT libraries honored an `alg: none` header, accepting unsigned tokens. A token like

```json
{"alg": "none", "typ": "JWT"}
{"sub": "admin"}
```

was accepted as valid by some libraries. Modern libraries (PyJWT 2.x, `python-jose` 3.x) reject this by default. **Always set the algorithm explicitly when verifying:**

```python
# ❌ Accepts any algorithm the token claims
jwt.decode(token, secret, options={"verify_signature": True})

# ✅ Only the algorithm you trust
jwt.decode(token, secret, algorithms=["HS256"])
```

### 2.4 The algorithm confusion attack

A library that accepts multiple algorithms and uses the same secret for symmetric and asymmetric is vulnerable. An attacker creates a token with `alg: HS256` signed with the public key (which is known) and the verifier treats it as a valid HMAC token.

The fix: **use different secrets for different algorithms** or **explicitly allow-list algorithms** in your decode call. Always do the latter.

---

## 3. Issuing Tokens

### 3.1 The basic flow

```python
from datetime import datetime, timedelta, timezone
import jwt
import secrets
from app.core.config import settings


def create_access_token(
    *,
    sub: int,
    tenant_id: int,
    role: str,
    expires_delta: timedelta = timedelta(minutes=15),
) -> tuple[str, str]:
    """Create a JWT access token. Returns (token, jti)."""
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
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, jti
```

The `jti` is the unique token ID. It enables revocation: store the `jti` in a database; check it on every request; if the row is gone, the token is revoked.

### 3.2 Storing the JTI for revocation

```python
class RevokedToken(Base):
    __tablename__ = "revoked_tokens"
    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    revoked_at: Mapped[datetime] = mapped_column(server_default=func.now())
    expires_at: Mapped[datetime]  # when the underlying token would have expired


async def revoke_token(jti: str, user_id: int, expires_at: datetime):
    async with SessionLocal() as session:
        async with session.begin():
            session.add(RevokedToken(
                jti=jti,
                user_id=user_id,
                expires_at=expires_at,
            ))
```

The `expires_at` lets a background job clean up revocation rows after the token would have expired anyway. Without this, the revocation table grows unboundedly.

### 3.3 The check on every request

```python
async def is_revoked(jti: str) -> bool:
    async with SessionLocal() as session:
        result = await session.execute(
            select(RevokedToken).where(RevokedToken.jti == jti)
        )
        return result.scalar_one_or_none() is not None
```

The check adds a DB query to every request. For high-throughput services, cache the result: a Redis cache of "revoked jtis" with TTL matching the longest token expiration. The cache is checked first; the DB is the source of truth.

---

## 4. Refresh Tokens

### 4.1 The access + refresh pattern

JWT access tokens should be short-lived (5-15 minutes). When they expire, the user must log in again — but that's a bad UX. The solution: a longer-lived **refresh token** that the client can exchange for a new access token.

```
Login: server returns access_token (15 min) + refresh_token (7 days)
Access token expires: client sends refresh_token to /auth/refresh
Server validates refresh, issues new access_token + new refresh_token
Refresh expires: client must log in again
```

### 4.2 Refresh token rotation

The critical security pattern: every refresh **invalidates the old refresh token** and issues a new one. This catches stolen refresh tokens: if an attacker uses a stolen refresh token, the legitimate user's next refresh fails, alerting the system.

```python
@router.post("/auth/refresh")
async def refresh_token(refresh: str, uow: UnitOfWork = Depends(get_uow)):
    try:
        payload = jwt.decode(refresh, settings.JWT_REFRESH_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid refresh token")
    jti = payload["jti"]
    # Check if this refresh token has been used
    if await uow.used_refresh_tokens.exists(jti=jti):
        # Token reuse detected; revoke the entire family
        await revoke_token_family(payload["family_id"])
        raise HTTPException(401, "Refresh token reuse detected")
    # Mark as used
    await uow.used_refresh_tokens.create(jti=jti, family_id=payload["family_id"])
    # Issue new access + refresh
    user = await uow.users.get(int(payload["sub"]))
    new_access = create_access_token(sub=user.id, tenant_id=user.tenant_id, role=user.role)
    new_refresh = create_refresh_token(
        sub=user.id, family_id=payload["family_id"]  # same family for rotation tracking
    )
    await uow.commit()
    return {"access_token": new_access, "refresh_token": new_refresh}
```

### 4.3 The "refresh token family"

A family is a chain of refresh tokens issued to the same user session. When a refresh token is used, the entire family can be revoked if reuse is detected (indicating a stolen token).

```python
def create_refresh_token(*, sub: int, family_id: str | None = None) -> tuple[str, str]:
    if family_id is None:
        family_id = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc)
    jti = secrets.token_urlsafe(16)
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
    return token, family_id
```

The `family_id` is created at login and passed through every refresh. A detected reuse revokes all tokens in the family (forcing a re-login from the legitimate user).

### 4.4 Refresh token storage

Refresh tokens are long-lived credentials. They must be stored securely:

- **Client side**: in an HTTP-only, secure, same-site cookie. Never in `localStorage` (XSS-readable).
- **Server side**: hashed (like passwords). If the database is compromised, the attacker cannot use the refresh tokens.

```python
import hashlib


def hash_refresh(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
```

Store the hash, not the token. The token is only ever seen in the request body; the server compares hashes.

---

## 5. The Seven JWT Pitfalls

### 5.1 `alg: none`

Already covered (§2.3). Always set `algorithms=["HS256"]` in `decode`.

### 5.2 Algorithm confusion

Already covered (§2.4). Use a single algorithm and pin it.

### 5.3 Long-lived tokens

```python
# ❌ A token that outlives the user
expires_delta=timedelta(days=365)
```

A leaked token is valid for a year. The 15-minute access token + 7-day refresh pattern is the standard. For sensitive operations, shorter is better.

### 5.4 Storing sensitive data in the payload

```python
# ❌ PII in the token
payload = {"sub": user.id, "email": user.email, "ssn": user.ssn}

# ✅ Minimal claims; fetch the rest from the DB
payload = {"sub": user.id, "tenant_id": user.tenant_id, "role": user.role}
```

The token is signed, not encrypted. Anyone with the token reads it. Keep the payload small and reference-free.

### 5.5 No signature verification

```python
# ❌ Decode without verification
payload = jwt.decode(token, options={"verify_signature": False})

# ✅ Always verify
payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
```

### 5.6 Clock skew

The server's clock and the client's clock may differ. JWTs use the server's clock; a token issued at 14:59:59 with a 15-minute expiration is expired at 15:00:00 even if the request arrived at 14:59:58. A 30-second clock skew window avoids this:

```python
payload = jwt.decode(
    token,
    settings.JWT_SECRET,
    algorithms=["HS256"],
    leeway=30,  # 30 seconds
)
```

### 5.7 Logging the secret

```python
# ❌ The secret ends up in logs
logger.info(f"Verifying token with secret={settings.JWT_SECRET}")
```

Treat the secret like a password. Never log it, never print it, never include it in stack traces. The `[REDACTED]` filter in your logging config should catch it as a safety net.

---

## 6. Session vs JWT: When to Use Each

### 6.1 The fundamental difference

| | Sessions | JWT |
|--|----------|-----|
| Storage | Server-side (Redis, DB) | Client-side (token) |
| Revocation | Trivial (delete the row) | Requires DB lookup or short expiry |
| Scalability | Requires shared state (Redis) | Stateless |
| Cookie size | Small (just the session ID) | Larger (the full token) |
| Mobile clients | Awkward (cookies) | Natural (Authorization header) |
| Cross-domain | CORS, CSRF concerns | Natural |

### 6.2 The modern hybrid

Most production systems use both:
- **Session cookies** for browser clients (HTTP-only, secure, same-site).
- **JWT in the Authorization header** for API clients (mobile, third-party, server-to-server).

The session backend can be Redis (with the session ID as the cookie). The JWT can be issued for API access (mobile SDKs). Both validate the same user record in the same database.

### 6.3 When JWT is the wrong choice

If you have:
- Web-only clients.
- Sensitive operations that need immediate revocation (e.g., admin actions).
- High session churn with low expiry needs.

Use server-side sessions. JWTs add complexity without benefit. For most browser apps, a Redis-backed session is simpler and more secure than a JWT.

---

## 7. JWT with FastAPI

### 7.1 The dependency

```python
from datetime import datetime, timezone
from typing import Annotated
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    uow: UnitOfWork = Depends(get_uow),
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")
    # Check revocation
    jti = payload.get("jti")
    if jti and await uow.revoked_tokens.is_revoked(jti):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has been revoked")
    # Check expiration (already in decode, but explicit for clarity)
    if payload.get("exp", 0) < datetime.now(timezone.utc).timestamp():
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has expired")
    # Load the user
    user = await uow.users.get(int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
```

The dependency is the single source of truth for "is this request authenticated?" Handlers declare `user: CurrentUser` and the framework injects the authenticated user.

### 7.2 The optional auth dependency

Some endpoints accept both authenticated and anonymous users (e.g., a public profile that shows extra info for logged-in users):

```python
async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
) -> User | None:
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials, ...)  # reuse the strict dep
    except HTTPException:
        return None
```

The `auto_error=False` makes the bearer optional. The function catches the auth failure and returns `None` instead of raising.

---

## 8. Sliding Sessions

### 8.1 The pattern

In a sliding session, every authenticated request extends the session's lifetime. The user is "remembered" as long as they keep using the service. Long-idle users are logged out.

```python
async def get_current_user_with_sliding_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    uow: UnitOfWork = Depends(get_uow),
) -> User:
    user = await get_current_user(credentials, uow)
    # Extend the session on activity
    await extend_session(credentials.credentials, uow)
    return user


async def extend_session(token: str, uow: UnitOfWork):
    payload = jwt.decode(token, options={"verify_signature": False})
    if payload.get("sliding"):
        new_payload = {**payload, "iat": int(time.time())}
        # The original token's expiration is unchanged; we just bump the iat
        # The handler issues a new token in the response header
        # (set by a middleware that reads from request.state.new_token)
        ...
```

The implementation is non-trivial: a sliding session needs to issue a new token on every request, which means setting a response header from a dependency. The cleanest way is a middleware that checks `request.state.token_renewed` and writes the new token in the `Authorization` header.

### 8.2 The trade-off

Sliding sessions are convenient for users (no re-login) but risk: a stolen token is "renewed" by the attacker's activity, extending its life indefinitely. The mitigation is a maximum session lifetime (e.g., 30 days) regardless of activity. The token is valid for 30 days from issue, not 30 days from last use.

---

## 9. Testing JWT

### 9.1 Unit tests for token issuance

```python
def test_create_access_token_has_required_claims():
    token, jti = create_access_token(sub=42, tenant_id=1, role="admin")
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    assert payload["sub"] == "42"
    assert payload["tenant_id"] == 1
    assert payload["role"] == "admin"
    assert payload["jti"] == jti
    assert payload["exp"] > time.time()


def test_rejects_token_with_wrong_algorithm():
    # Create a token with a different algorithm
    payload = {"sub": "1", "exp": int(time.time()) + 100}
    token = jwt.encode(payload, "secret", algorithm="HS512")
    with pytest.raises(jwt.InvalidAlgorithmError):
        jwt.decode(token, "secret", algorithms=["HS256"])


def test_rejects_expired_token():
    payload = {"sub": "1", "exp": int(time.time()) - 100}
    token = jwt.encode(payload, "secret", algorithm="HS256")
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, "secret", algorithms=["HS256"])
```

### 9.2 Integration tests

```python
@pytest.mark.asyncio
async def test_protected_endpoint_requires_token(client):
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_returns_user_with_valid_token(client, user):
    token, _ = create_access_token(sub=user.id, tenant_id=user.tenant_id, role=user.role)
    response = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == user.id
```

---

## 10. Código de Compresión

```python
"""
Compresión: JWT Deep Dive
Cubre: token structure, signing, refresh rotation, revocation, FastAPI integration.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


# 1) Settings (illustrative)
class Settings:
    JWT_SECRET = "your-256-bit-secret"  # 32+ bytes
    JWT_REFRESH_SECRET = "your-refresh-secret-32-bytes"
    JWT_ALGORITHM = "HS256"
    JWT_ISSUER = "https://auth.example.com"
    JWT_AUDIENCE = "https://api.example.com"
    ACCESS_TOKEN_EXPIRE = timedelta(minutes=15)
    REFRESH_TOKEN_EXPIRE = timedelta(days=7)


settings = Settings()
security = HTTPBearer()


# 2) Token creation
def create_access_token(*, sub: int, tenant_id: int, role: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    jti = secrets.token_urlsafe(16)
    payload = {
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "sub": str(sub),
        "iat": now,
        "exp": now + settings.ACCESS_TOKEN_EXPIRE,
        "jti": jti,
        "tenant_id": tenant_id,
        "role": role,
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM), jti


def create_refresh_token(*, sub: int, family_id: str | None = None) -> tuple[str, str, str]:
    family_id = family_id or secrets.token_urlsafe(16)
    jti = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc)
    payload = {
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "sub": str(sub),
        "iat": now,
        "exp": now + settings.REFRESH_TOKEN_EXPIRE,
        "jti": jti,
        "family_id": family_id,
        "type": "refresh",
    }
    token = jwt.encode(payload, settings.JWT_REFRESH_SECRET, algorithm="HS256")
    return token, jti, family_id


# 3) Hash for storage
def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# 4) Validation
class AuthError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(401, detail)


def validate_token(token: str, *, secret: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            leeway=30,
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}")
    if payload.get("type") != expected_type:
        raise AuthError(f"Expected {expected_type} token, got {payload.get('type')}")
    return payload


# 5) FastAPI dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    payload = validate_token(
        credentials.credentials,
        secret=settings.JWT_SECRET,
        expected_type="access",
    )
    # In production: load from DB, check revocation
    return {"id": int(payload["sub"]), "tenant_id": payload["tenant_id"], "role": payload["role"]}


# 6) Demo
app = FastAPI()


@app.post("/auth/login")
async def login(email: str, password: str):
    # In production: verify password against DB hash
    user_id = 42  # placeholder
    tenant_id = 1
    role = "admin"
    access, _ = create_access_token(sub=user_id, tenant_id=tenant_id, role=role)
    refresh, _, _ = create_refresh_token(sub=user_id)
    return {"access_token": access, "refresh_token": refresh}


@app.get("/me")
async def me(user=Depends(get_current_user)):
    return user
```

---

## Key Takeaways

- JWTs are signed, not encrypted. Anyone with the token reads every claim. Keep the payload minimal.
- Always pin the algorithm in `jwt.decode(..., algorithms=[...])`. The `alg: none` and algorithm-confusion attacks are still exploited in the wild.
- Access tokens should be short (5-15 min); refresh tokens longer (7 days), with rotation on every use. Refresh-token reuse detection catches stolen tokens.
- The `jti` claim is the lever for revocation. Store revoked `jti`s in a DB or cache; check on every request.
- For browser clients, server-side sessions (Redis-backed) are often simpler and more secure than JWTs. Use JWTs for API and mobile clients.
- Sliding sessions are convenient but risky. Combine with a maximum session lifetime to cap the damage from a stolen token.
- Clock skew between server and client is real. Use a 30-second leeway in `jwt.decode`.
- Never log JWT secrets. Treat them like passwords: in environment variables, never in code, never in stack traces.

## References

- [RFC 7519 — JSON Web Token](https://www.rfc-editor.org/rfc/rfc7519)
- [RFC 8725 — JWT Best Current Practices](https://www.rfc-editor.org/rfc/rfc8725)
- [OWASP JSON Web Token Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
- [Auth0 — JWT Handbook](https://auth0.com/resources/ebooks/jwt-handbook)
- [PyJWT Documentation](https://pyjwt.readthedocs.io/)
- [Stripe's blog — API Keys](https://stripe.com/docs/keys)
- [GitHub's blog — Token Authentication in Production](https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/)
