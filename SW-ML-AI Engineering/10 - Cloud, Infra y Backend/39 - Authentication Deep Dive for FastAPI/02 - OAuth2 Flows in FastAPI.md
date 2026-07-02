# 🔄 OAuth2 Flows in FastAPI

## 🎯 Learning Objectives

- Understand the OAuth2 family of flows: Authorization Code, Client Credentials, PKCE, Device Code, and Refresh Token
- Choose the right flow for your client type (web app, SPA, mobile, server-to-server, IoT)
- Implement OAuth2 in FastAPI with `authlib` and `httpx-oauth`
- Avoid the eight most common OAuth2 implementation mistakes
- Integrate third-party providers (Google, GitHub, Auth0) into your FastAPI service

## Introduction

OAuth2 is the dominant standard for delegated authorization. It defines how a client application can act on behalf of a user to access resources on a third-party server — without ever seeing the user's password. The standard is flexible: there are several flows optimized for different client types, and a few well-known extensions (PKCE, DPoP) that close real-world security gaps.

The challenge is that OAuth2 is a **standard for authorization**, not authentication. People commonly conflate them: "log in with Google" is OAuth2 + OpenID Connect (OIDC), where OIDC adds an identity layer on top. This note covers the authorization flows and the OIDC layer on top, with a focus on what FastAPI services need to implement.

The OAuth 2.1 draft is the consolidation of OAuth 2.0 + the security BCP (best current practice) recommendations. Most modern libraries implement OAuth 2.1 by default. Where the older RFC 6749 says one thing and the BCP says another, this note follows the BCP.

---

## 1. The OAuth2 Cast of Characters

| Role | Description | Example |
|------|-------------|---------|
| Resource Owner | The user | You |
| Client | The application requesting access | Your web app, mobile app |
| Authorization Server | Issues tokens | Google, GitHub, your auth service |
| Resource Server | Hosts the protected resources | Your API |
| User Agent | Browser or app that mediates the redirect | Chrome, Safari, native app |
| Redirect URI | Where the auth server sends the user back | `https://app.com/callback` |

The flow:

1. User clicks "Log in with Google" on your app.
2. App redirects user to Google with `client_id`, `redirect_uri`, `scope`, `state`, `code_challenge`.
3. User authenticates with Google and approves the requested scopes.
4. Google redirects user back to `redirect_uri` with an authorization `code`.
5. App exchanges the `code` (and `code_verifier`) for an `access_token`.
6. App uses the `access_token` to call Google's APIs (e.g., fetch user profile).

The user never shares their Google password with your app.

---

## 2. The Flows

### 2.1 Authorization Code Flow (with PKCE)

The recommended flow for **all** public clients (web apps, SPAs, mobile apps, native apps). The 2022 OAuth 2.0 Security BCP made PKCE **mandatory** for all clients.

```
┌──────────┐                            ┌──────────────┐
│  Browser │                            │ Auth Server  │
│ (User)   │                            │  (Google)    │
└────┬─────┘                            └──────┬───────┘
     │  1. Click "Log in"                    │
     │ ───────────────────────────────────────>
     │  2. Redirect to /authorize             │
     │     ?client_id=...                    │
     │     &redirect_uri=https://app/cb     │
     │     &scope=openid+email              │
     │     &state=abc123                    │
     │     &code_challenge=BASE64(SHA256(verifier))
     │     &code_challenge_method=S256
     │ <───────────────────────────────────────
     │  3. User authenticates, approves scopes │
     │  4. Redirect to https://app/cb        │
     │     ?code=xyz789                      │
     │     &state=abc123                    │
     │ <───────────────────────────────────────
     │  5. App exchanges code + verifier      │
     │     POST /token                       │
     │     code=xyz789                       │
     │     code_verifier=verifier            │
     │ <───────────────────────────────────────
     │  6. Returns access_token + id_token   │
     │ ───────────────────────────────────────>
     │  7. App uses access_token to call API  │
     │ ───────────────────────────────────────>
```

**Why PKCE**: a public client (SPA, mobile) cannot keep a `client_secret` truly secret — the code is shipped to the user. PKCE replaces the secret with a one-time `code_verifier` / `code_challenge` pair. The verifier is generated client-side and never sent until step 5; the challenge is a hash of the verifier sent in step 2. The auth server verifies the hash matches; if it does, the client is genuine. An attacker who intercepts the `code` in step 4 cannot exchange it without the `code_verifier`.

### 2.2 Client Credentials Flow

The right flow for **server-to-server** communication. No user involved.

```
App Server                         Auth Server
    │                                  │
    │  POST /token                      │
    │  grant_type=client_credentials   │
    │  client_id=...                    │
    │  client_secret=...                │
    │  scope=read:users                 │
    │ <─────────────────────────────────
    │  Returns access_token             │
    │ ─────────────────────────────────>
```

The `client_id` and `client_secret` are credentials for the app itself, not for a user. The token represents the app's identity. Use it for:
- Microservices calling each other.
- Backend jobs calling APIs.
- Cron tasks that need to read but not act as a user.

### 2.3 Resource Owner Password Credentials (DEPRECATED)

The "username + password" flow. **Deprecated in OAuth 2.1.** Use it only for migration from legacy systems; never for new applications. The password is shared with the client, which is exactly what OAuth was designed to avoid.

### 2.4 Device Code Flow

For **input-limited devices** (smart TVs, IoT, CLI tools). The device polls an auth server; the user authenticates on a separate device (phone or laptop).

```
Device                              Auth Server         User
  │  POST /device/code                  │                  │
  │ <── device_code, user_code ─────── │                  │
  │  Show user_code + verification URL │                  │
  │                                     │  User opens URL  │
  │                                     │  enters user_code│
  │  POST /token (poll)                 │                  │
  │ <── access_token (when approved)── │                  │
```

### 2.5 Implicit Flow (DEPRECATED)

Deprecated. The flow returned the access token directly in the URL fragment. It was simple but insecure. Use Authorization Code with PKCE instead.

---

## 3. Choosing the Right Flow

| Client type | Recommended flow |
|-------------|-----------------|
| Web server app (server-rendered) | Authorization Code + PKCE |
| SPA (browser) | Authorization Code + PKCE (with BFF pattern) |
| Mobile app | Authorization Code + PKCE |
| Server-to-server | Client Credentials |
| CLI / IoT | Device Code |
| First-party (your own frontend + backend) | Session cookies, not OAuth2 |

The 2022 BCP says: **use Authorization Code with PKCE for everything user-facing.** Even SPAs, which historically used Implicit, now use Auth Code with PKCE and a Backend-for-Frontend (BFF) that stores the token in an HTTP-only cookie.

---

## 4. Implementing OAuth2 in FastAPI

### 4.1 With `authlib`

`authlib` is the standard Python library for OAuth2 / OIDC. It implements the client side, the server side, and the resource protection.

```bash
pip install authlib httpx
```

### 4.2 The auth server (you're issuing tokens)

```python
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.responses import Redirect
from fastapi import FastAPI, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

config = Config(".env")
oauth = OAuth(config)

oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=config("GOOGLE_CLIENT_ID"),
    client_secret=config("GOOGLE_CLIENT_SECRET"),
    client_kwargs={"scope": "openid email profile"},
)

app = FastAPI()


@app.get("/auth/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    # In production: create or update a user record, issue a JWT
    return {"user": user_info}
```

The OAuth client is registered once with the metadata URL. `authlib` fetches the auth server's endpoints, JWKS, and supported features automatically.

### 4.3 The auth server (you're protecting a resource)

```python
from authlib.integrations.starlette_oauth2 import ResourceProtector
from authlib.oauth2.rfc7662 import IntrospectTokenValidator
import httpx

require_oauth = ResourceProtector()

# For external providers (Google, GitHub), the access token is opaque;
# validate it by calling the provider's introspection endpoint
class GoogleTokenValidator(IntrospectTokenValidator):
    def get_introspection_url(self):
        return "https://oauth2.googleapis.com/tokeninfo"

    def introspect_token(self, token):
        resp = httpx.get(self.get_introspection_url(), params={"access_token": token})
        if resp.status_code != 200:
            return None
        return resp.json()


validator = GoogleTokenValidator()
require_oauth.register_token_validator(validator)


@app.get("/api/v1/me")
@require_oauth()
async def get_me(request: Request):
    user = require_oauth.context["user"]  # populated from the token's claims
    return user
```

For a self-issued JWT (your own auth service), use `authlib`'s JWT bearer validator:

```python
from authlib.oauth2.rfc7523 import JWTBearerTokenValidator
from authlib.jose import JsonWebKey


class MyJWTValidator(JWTBearerTokenValidator):
    def __init__(self, public_key, issuer, audience):
        super().__init__(public_key)
        self.claims_options = {
            "iss": {"essential": True, "value": issuer},
            "aud": {"essential": True, "value": audience},
        }


jwks = JsonWebKey.import_key_set(your_jwks_dict)
validator = MyJWTValidator(jwks, issuer="https://auth.example.com", audience="https://api.example.com")
require_oauth.register_token_validator(validator)
```

### 4.4 FastAPI Users

`fastapi-users` is a higher-level library that handles registration, login, password reset, OAuth, and more:

```bash
pip install "fastapi-users[sqlalchemy,oauth]"
```

```python
from fastapi import FastAPI
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication
from fastapi_users.db import SQLAlchemyUserDatabase
from app.models.user import User
from app.db.engine import SessionLocal


async def get_user_db():
    async with SessionLocal() as session:
        yield SQLAlchemyUserDatabase(session, User)


auth_backends = [
    JWTAuthentication(secret=settings.JWT_SECRET, lifetime_seconds=3600),
]

fastapi_users = FastAPIUsers(
    get_user_db,
    auth_backends,
    User,
    UserCreate,
    UserUpdate,
    UserDB,
)

app = FastAPI()
app.include_router(fastapi_users.router, prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_register_router(), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
```

`fastapi-users` is opinionated and handles 80% of common cases. For unusual requirements, fall back to `authlib` or build your own on top of `pyjwt`.

---

## 5. The OIDC Layer

### 5.1 OpenID Connect vs OAuth2

| | OAuth2 | OIDC |
|--|-------|------|
| Purpose | Authorization (access resources) | Authentication (who is the user) |
| Token | access_token | id_token (in addition to access_token) |
| Claims | Optional | Standardized: `sub`, `email`, `name`, `picture` |
| Endpoint | `/authorize`, `/token` | Adds `/.well-known/openid-configuration` and `/userinfo` |

OIDC is built on OAuth2. A client that requests the `openid` scope gets an `id_token` in addition to the `access_token`. The `id_token` is a JWT signed by the auth server; the client verifies it and reads the user's identity.

### 5.2 Verifying the ID token

```python
from authlib.jose import jwt


def verify_id_token(id_token: str, client_id: str, issuer: str, jwks: dict) -> dict:
    claims = jwt.decode(id_token, jwks)
    claims.options = {
        "iss": {"essential": True, "value": issuer},
        "aud": {"essential": True, "value": client_id},
        "sub": {"essential": True},
    }
    claims.validate()
    return claims
```

Critical checks: `iss` matches the expected auth server, `aud` matches your `client_id`, `exp` is in the future, and the signature is valid against the JWKS published by the auth server.

### 5.3 Mapping identity to your user record

The `id_token` has a `sub` claim — a stable identifier for the user at the auth server. Your user record should have a column for `auth_provider` (e.g., `"google"`) and `external_id` (the `sub`). On first login, create a user record with these values. On subsequent logins, look up by `(auth_provider, external_id)`.

```python
async def upsert_user_from_id_token(payload: dict, db) -> User:
    sub = payload["sub"]
    provider = "google"
    user = await db.users.get_by(auth_provider=provider, external_id=sub)
    if not user:
        user = await db.users.create(
            auth_provider=provider,
            external_id=sub,
            email=payload.get("email"),
            name=payload.get("name"),
        )
    return user
```

---

## 6. The Eight OAuth2 Pitfalls

### 6.1 Missing `state` parameter

```python
# ❌ No state; vulnerable to CSRF
return await oauth.google.authorize_redirect(request, redirect_uri)

# ✅ State is generated automatically by authlib
# Verify it matches in the callback
```

The `state` parameter binds the auth request to the user's session. Without it, an attacker can trick the user into completing a malicious auth flow that links the attacker's account to the victim's profile.

### 6.2 Missing `nonce` in OIDC

```python
# ❌ No nonce in OIDC
return await oauth.google.authorize_redirect(request, redirect_uri, scope="openid email")

# ✅ Generate a nonce, store it in the session, verify in the id_token
```

`nonce` prevents replay attacks on the `id_token`.

### 6.3 Missing PKCE

```python
# ❌ Authorization Code without PKCE
await oauth.google.authorize_redirect(request, redirect_uri)

# ✅ PKCE is enabled by default in modern libraries
```

`authlib` enables PKCE by default for public clients. Verify it's not disabled in your config.

### 6.4 Open redirect in the callback

```python
# ❌ Trusts any redirect_uri
if request.query_params.get("next"):
    return Redirect(request.query_params["next"])

# ✅ Whitelist the redirect target
allowed = {"/dashboard", "/profile", "/"}
next_url = request.query_params.get("next", "/")
if next_url not in allowed:
    next_url = "/"
return Redirect(next_url)
```

### 6.5 Storing the access token in localStorage

```javascript
// ❌ XSS-readable
localStorage.setItem("access_token", token);
```

Store tokens in HTTP-only, secure, same-site cookies. JavaScript cannot read them; XSS cannot exfiltrate them.

### 6.6 Not validating the `aud` claim

```python
# ❌ Accepts any JWT from the auth server
payload = jwt.decode(id_token, options={"verify_aud": False})

# ✅ Verify the audience
payload = jwt.decode(
    id_token,
    key,
    audience=settings.OIDC_CLIENT_ID,
    options={"verify_aud": True},
)
```

A token issued for another client of the same auth server must not be accepted by your client.

### 6.7 Token leakage in logs

```python
# ❌ Logs include the token
logger.info(f"OAuth callback: token={token}, user={user}")

# ✅ Log a fingerprint, not the token
logger.info(f"OAuth callback: token_fp={hash_token(token)[:8]}, user={user.id}")
```

### 6.8 Refresh token in the SPA

```javascript
// ❌ SPA holds the refresh token
const refresh = localStorage.getItem("refresh_token");
const newAccess = await fetch("/token", { body: { grant_type: "refresh_token", refresh_token: refresh } });
```

The refresh token is in the browser, exposed to XSS. The BFF (Backend-for-Frontend) pattern keeps the refresh token on the server: the SPA talks to the BFF, the BFF talks to the auth server, and the refresh token never leaves the backend.

---

## 7. The BFF Pattern

### 7.1 What BFF means

A **Backend-for-Frontend** is a small server-side component that:
- Holds the access and refresh tokens.
- Receives requests from the SPA.
- Adds the access token to upstream calls.
- Refreshes the token when needed.

The SPA never sees the tokens. The BFF acts as a trusted intermediary.

### 7.2 BFF in FastAPI

```python
from fastapi import FastAPI, Request, Response, HTTPException
import httpx

app = FastAPI()
OAUTH_BASE = "https://auth.example.com"
CLIENT_ID = "my-spa"
CLIENT_SECRET = "..."  # server-side, never exposed to the browser


@app.post("/bff/login")
async def login(request: Request, response: Response):
    """Exchange authorization code for tokens; set session cookie."""
    code = (await request.json()).get("code")
    if not code:
        raise HTTPException(400, "Missing code")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OAUTH_BASE}/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri": "/callback",  # the SPA's callback, not the BFF's
            },
        )
    if resp.status_code != 200:
        raise HTTPException(401, "Token exchange failed")
    tokens = resp.json()
    # Set HTTP-only cookie with the session ID
    # The actual tokens are stored in a server-side session keyed by the cookie
    response.set_cookie(
        "bff_session",
        create_session(tokens),  # server-side session, tokens in Redis
        httponly=True,
        secure=True,
        samesite="strict",
    )
    return {"ok": True}


@app.get("/bff/api/{path:path}")
async def proxy(path: str, request: Request):
    """Proxy to the upstream API, adding the access token."""
    session_id = request.cookies.get("bff_session")
    if not session_id:
        raise HTTPException(401, "Not authenticated")
    tokens = get_session(session_id)  # from Redis
    if tokens_expired(tokens):
        tokens = await refresh_tokens(tokens)
        update_session(session_id, tokens)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.example.com/{path}",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
```

The SPA sees a normal API surface (`/bff/api/...`); the tokens are invisible.

---

## 8. Third-Party Providers

### 8.1 Google

```python
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    client_kwargs={"scope": "openid email profile"},
)
```

### 8.2 GitHub

```python
oauth.register(
    name="github",
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_id=settings.GITHUB_CLIENT_ID,
    client_secret=settings.GITHUB_CLIENT_SECRET,
    client_kwargs={"scope": "read:user user:email"},  # for OIDC-like identity
)
```

GitHub does not support OIDC natively. To get the user's email, you need the `user:email` scope and an extra API call to `/user/emails`.

### 8.3 Auth0

```python
oauth.register(
    name="auth0",
    server_metadata_url=f"https://{settings.AUTH0_DOMAIN}/.well-known/openid-configuration",
    client_id=settings.AUTH0_CLIENT_ID,
    client_secret=settings.AUTH0_CLIENT_SECRET,
    client_kwargs={"scope": "openid email profile"},
)
```

### 8.4 The standard onboarding flow

```python
@app.get("/auth/{provider}/login")
async def login_with_provider(provider: str, request: Request):
    redirect_uri = request.url_for("auth_callback", provider=provider)
    client = oauth.create_client(provider)
    return await client.authorize_redirect(request, redirect_uri)


@app.get("/auth/{provider}/callback")
async def auth_callback(provider: str, request: Request):
    client = oauth.create_client(provider)
    token = await client.authorize_access_token(request)
    user_info = token.get("userinfo") or await client.parse_id_token(request, token)
    # Upsert the user, issue a JWT
    user = await upsert_user_from_id_token(user_info, ...)
    access, _ = create_access_token(sub=user.id, tenant_id=user.tenant_id, role=user.role)
    return {"access_token": access}
```

The same callback handler works for any provider because the `id_token` claims are standardized.

---

## 9. Testing OAuth2

### 9.1 Mock the auth server

In tests, you don't want to call real Google or GitHub. Mock the OAuth flow:

```python
@pytest.fixture
def mock_oauth():
    """Mock the OAuth client to bypass real provider calls."""
    with patch("app.api.auth.oauth.google") as mock:
        mock.authorize_redirect.return_value = Redirect("/auth/callback?code=test_code&state=test_state")
        mock.authorize_access_token.return_value = {
            "access_token": "test_access",
            "id_token": "test_id_token",
            "userinfo": {
                "sub": "google_123",
                "email": "alice@example.com",
                "name": "Alice",
            },
        }
        yield mock


@pytest.mark.asyncio
async def test_google_login_callback_creates_user(mock_oauth, client):
    response = await client.get("/auth/google/callback?code=test_code&state=test_state")
    assert response.status_code == 200
    assert "access_token" in response.json()
```

### 9.2 Test the state validation

```python
@pytest.mark.asyncio
async def test_callback_rejects_invalid_state(client):
    response = await client.get("/auth/google/callback?code=test&state=wrong_state")
    assert response.status_code == 400
```

### 9.3 Test PKCE

```python
@pytest.mark.asyncio
async def test_pkce_verifier_matches_challenge(client):
    # Generate a verifier and challenge
    verifier = base64url(secrets.token_bytes(32))
    challenge = base64url(hashlib.sha256(verifier).digest())
    state = "test_state"
    # 1) Authorize
    response = await client.get(f"/auth/google/login?state={state}")
    # Capture the cookie
    # 2) Callback with the verifier
    response = await client.get(
        f"/auth/google/callback?code=test&state={state}&code_verifier={verifier}"
    )
    assert response.status_code == 200
```

---

## 10. Código de Compresión

```python
"""
Compresión: OAuth2 Flows in FastAPI
Cubre: Authorization Code + PKCE, Client Credentials, BFF pattern, third-party providers.
"""
import secrets
import hashlib
import base64
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Redirect
import httpx
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

config = Config(".env")
oauth = OAuth(config)
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=config("GOOGLE_CLIENT_ID"),
    client_secret=config("GOOGLE_CLIENT_SECRET"),
    client_kwargs={"scope": "openid email profile"},
)

app = FastAPI()


# 1) PKCE helpers
def generate_pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


# 2) Login (Auth Code + PKCE flow)
@app.get("/auth/{provider}/login")
async def login(request: Request, provider: str):
    verifier, challenge = generate_pkce_pair()
    # Store the verifier in a session or signed cookie
    request.session["pkce_verifier"] = verifier
    redirect_uri = request.url_for("auth_callback", provider=provider)
    client = oauth.create_client(provider)
    return await client.authorize_redirect(
        request,
        redirect_uri,
        code_challenge=challenge,
        code_challenge_method="S256",
    )


# 3) Callback
@app.get("/auth/{provider}/callback")
async def auth_callback(request: Request, provider: str):
    verifier = request.session.pop("pkce_verifier", None)
    if not verifier:
        raise HTTPException(400, "Missing PKCE verifier")
    client = oauth.create_client(provider)
    token = await client.authorize_access_token(request, code_verifier=verifier)
    user_info = token.get("userinfo")
    # In production: upsert user, issue a JWT
    return {
        "user": user_info,
        "access_token": token["access_token"],
    }


# 4) Client Credentials (server-to-server)
@app.post("/auth/token")
async def client_credentials_token(grant_type: str, client_id: str, client_secret: str, scope: str = ""):
    # In production: validate client_id + client_secret against a registered client table
    if grant_type != "client_credentials":
        raise HTTPException(400, "Unsupported grant type")
    if not validate_client(client_id, client_secret):
        raise HTTPException(401, "Invalid client credentials")
    # Issue a token for the app itself (no user)
    return {"access_token": issue_app_token(client_id, scope), "token_type": "bearer"}


# 5) Device Code flow
@app.post("/auth/device/code")
async def device_code(client_id: str):
    device_code = secrets.token_urlsafe(32)
    user_code = f"{secrets.token_hex(4)}".upper()  # 8-char code for the user to enter
    verification_uri = "https://app.com/device"
    # Store device_code -> user_code mapping
    store_device_code(device_code, user_code, client_id)
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "expires_in": 600,
        "interval": 5,
    }
```

---

## Key Takeaways

- The 2022 OAuth 2.0 Security BCP mandates **PKCE for all clients**, including confidential ones. Use it everywhere.
- The Authorization Code flow is the right choice for user-facing apps (web, SPA, mobile). Client Credentials is for server-to-server. Device Code is for input-limited devices. Avoid Implicit (deprecated) and Password (deprecated).
- OIDC is OAuth2 + identity. When you "log in with Google", you're using OIDC. The `id_token` carries the user's identity claims.
- The BFF pattern keeps tokens on the server. The SPA talks to the BFF; the BFF holds the access and refresh tokens. XSS cannot exfiltrate what JavaScript cannot read.
- `state` and `nonce` are non-optional security parameters. Generate them, verify them, log when they fail.
- Never store tokens in `localStorage`. Use HTTP-only cookies via the BFF.
- `authlib` is the standard library for OAuth2 / OIDC in Python. `fastapi-users` is a higher-level wrapper that handles common cases.
- For a self-issued JWT (your own auth service), use `authlib.oauth2.rfc7523.JWTBearerTokenValidator` and pin the algorithm, audience, and issuer.

## References

- [OAuth 2.0 Security BCP (RFC 9700)](https://www.rfc-editor.org/rfc/rfc9700)
- [OAuth 2.1 Draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-v2-1/)
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- [PKCE RFC 7636](https://www.rfc-editor.org/rfc/rfc7636)
- [Authlib Documentation](https://authlib.org/)
- [FastAPI Users](https://fastapi-users.github.io/fastapi-users/)
- [OWASP OAuth2 Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/OAuth2_Cheat_Sheet.html)
- [Auth0 — OAuth 2.0 Security Best Practices](https://auth0.com/blog/oauth-2-best-practices/)
