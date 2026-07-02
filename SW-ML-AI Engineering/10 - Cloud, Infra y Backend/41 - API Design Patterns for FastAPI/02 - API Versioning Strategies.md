# 🏷️ API Versioning Strategies

## 🎯 Learning Objectives

- Choose the right versioning strategy for your API: path, header, content negotiation
- Implement each strategy in FastAPI with the trade-offs spelled out
- Deprecate an API version gracefully without breaking existing clients
- Avoid the four most common versioning mistakes

## Introduction

Every API needs a versioning story from day one. The choice is irreversible in practice: clients write code against the version you publish, and changing the version means breaking that code. The decision is between **path-based** (`/v1/users`), **header-based** (`Accept: application/vnd.myapi.v2+json`), or **content negotiation**. The industry has converged on path-based as the default; this note explains why and shows the FastAPI patterns.

The harder question is deprecation. The day you ship v2, you have v1 clients who haven't migrated. You can't break them; you can't maintain two codebases forever. The deprecation story — sunset headers, migration windows, parallel routing — is what separates a managed API from a chaos.

---

## 1. The Three Strategies

### 1.1 Path-based versioning

```python
@router.get("/v1/users/{user_id}")
async def get_user_v1(user_id: int): ...


@router.get("/v2/users/{user_id}")
async def get_user_v2(user_id: int): ...
```

The version is in the URL. Pros: easy to read, easy to route, easy to document. Cons: the URL is a resource, not a version; mixing the two confuses semantics.

### 1.2 Header-based versioning

```http
GET /users/12345 HTTP/1.1
Accept: application/vnd.myapi.v2+json
```

The version is in the `Accept` header. Pros: clean URLs. Cons: harder to test in a browser; harder to route; some HTTP clients don't easily set custom headers.

### 1.3 Content negotiation

```http
GET /users/12345 HTTP/1.1
Accept: application/vnd.myapi+json; version=2
```

Like header-based but with structured format parameters. Pros: explicit. Cons: most complicated; least common.

### 1.4 The industry default: path-based

| API | Versioning | Notes |
|-----|-----------|-------|
| Stripe | URL (`/v1/...`) | Standard |
| GitHub | Header (`X-GitHub-Api-Version`) | Also accepts `?version=` query param |
| Twilio | URL | Standard |
| Google Cloud | URL | Standard |
| AWS | URL + header | Both work |
| Microsoft Azure | URL + header | Both work |

The path is the most discoverable. A developer can paste a URL into a browser or curl and see what version they're hitting. Header-based requires the developer to know the convention.

The trade-off: path-based exposes the version in the URL, which can lead to "version proliferation" in the docs. Header-based keeps the URL clean but hides the version from logs and analytics.

**For a new FastAPI service, use path-based.** It's the standard, the easiest to implement, and the easiest for clients to consume.

---

## 2. Path-Based Versioning in FastAPI

### 2.1 The router-per-version pattern

```python
# app/api/v1/users.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}")
async def get_user(user_id: int): ...


# app/api/v2/users.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}")
async def get_user(user_id: int): ...  # different shape
```

```python
# app/main.py
from app.api.v1 import users as users_v1
from app.api.v2 import users as users_v2

app = FastAPI()
app.include_router(users_v1.router, prefix="/v1")
app.include_router(users_v2.router, prefix="/v2")
```

The prefix is set at the include level, not the router level. This keeps the router reusable across versions.

### 2.2 Shared logic with a base router

```python
# app/api/v1/users.py
from fastapi import APIRouter
from app.api.base import BaseUserRouter


class V1UserRouter(BaseUserRouter):
    version = "v1"
    user_response_model = UserV1Out


router = V1UserRouter().build()


# app/api/v2/users.py
from app.api.base import BaseUserRouter
from app.api.v2.schemas import UserV2Out


class V2UserRouter(BaseUserRouter):
    version = "v2"
    user_response_model = UserV2Out


router = V2UserRouter().build()
```

```python
# app/api/base.py
from fastapi import APIRouter, Depends
from app.db.session import UnitOfWork
from app.api.deps import get_uow


class BaseUserRouter:
    version: str
    user_response_model: type

    def build(self) -> APIRouter:
        router = APIRouter(prefix="/users", tags=[f"users-{self.version}"])
        router.add_api_route(
            "/{user_id}",
            self.get_user,
            methods=["GET"],
            response_model=self.user_response_model,
        )
        return router

    async def get_user(
        self,
        user_id: int,
        uow: UnitOfWork = Depends(get_uow),
    ):
        user = await uow.users.get(user_id)
        if not user:
            raise NotFoundProblem(...)
        return user
```

The base class encapsulates the shared logic; subclasses override the version-specific bits (response model, perhaps validation).

### 2.3 The lifespan dependency

The version is a stable string; it does not need a request-time dependency. The version is in the URL; the router handles the rest.

### 2.4 Version detection for shared logic

For code that needs to know the version (e.g., a custom middleware), use `request.url.path`:

```python
from fastapi import Request


async def log_request(request: Request, call_next):
    path = request.url.path
    if path.startswith("/v1/"):
        version = "v1"
    elif path.startswith("/v2/"):
        version = "v2"
    else:
        version = "unknown"
    logger.info("request", path=path, version=version)
    response = await call_next(request)
    return response
```

A cleaner pattern is to use the `version` kwarg on the APIRouter, but FastAPI does not natively route by version in this way.

---

## 3. The Deprecation Story

### 3.1 The deprecation headers

The standard headers for deprecation (RFC 8594):

```http
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 01 Mar 2026 00:00:00 GMT
Link: <https://docs.example.com/migrate-to-v2>; rel="successor-version"
```

- `Deprecation: true` tells the client the endpoint is deprecated.
- `Sunset: <date>` is the date the endpoint will be removed.
- `Link: <url>; rel="successor-version"` points to the new endpoint.

### 3.2 Implementing in FastAPI

```python
from datetime import datetime
from fastapi import APIRouter, Response

router = APIRouter(prefix="/v1", deprecated=True)


@router.get("/users/{user_id}")
async def get_user_v1(user_id: int, response: Response):
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Sat, 01 Mar 2026 00:00:00 GMT"
    response.headers["Link"] = '</v2/users/' + str(user_id) + '>; rel="successor-version"'
    # ... existing logic
```

`deprecated=True` (FastAPI's built-in) marks the route as deprecated in the OpenAPI schema. The manual headers add the runtime deprecation signals.

### 3.3 The deprecation window

The recommended window is **6-12 months** for non-breaking changes, **12-18 months** for breaking changes. During the window:

- v1 still works (with deprecation headers).
- v2 is the recommended version (in docs, in error messages, in onboarding).
- New endpoints ship in v2 only.

After the window:

- v1 returns `410 Gone`.
- v1 docs are archived.
- v1 endpoints are removed.

### 3.4 The 410 Gone response

```python
@router.get("/v1/{path:path}", include_in_schema=False)
async def v1_gone(request: Request, path: str):
    problem = ProblemDetails(
        type="https://example.com/probs/gone",
        title="Gone",
        status=410,
        detail=f"v1 is no longer available. Use v2: /v2/{path}",
        instance=str(request.url.path),
    )
    return JSONResponse(
        status_code=410,
        content=problem.model_dump(mode="json", exclude_none=True),
        media_type="application/problem+json",
        headers={
            "Link": '</v2/' + path + '>; rel="successor-version"',
        },
    )
```

A single catch-all route at `/v1/...` returns 410 for every v1 request after sunset. Clients see the error and the migration target.

---

## 4. Versioning the Schema, Not Just the URL

### 4.1 The problem

Two endpoints in the same version can return different shapes if they were added at different times. The "version" is a moving target.

### 4.2 The fix: explicit response models

```python
# v1
class UserV1Out(BaseModel):
    id: int
    email: str
    name: str


# v2
class UserV2Out(BaseModel):
    id: int
    email: str
    name: str
    avatar_url: str | None = None  # added in v2
    created_at: datetime           # added in v2
```

The response model is the contract. A handler that returns the v1 model is contractually v1. Adding `avatar_url` requires a new model and a new version.

### 4.3 Pydantic v2 patterns

```python
from pydantic import BaseModel, ConfigDict


class UserOutV2(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        # Strict: reject extra fields
        extra="forbid",
    )
    id: int
    email: str
    name: str
    avatar_url: str | None = None
    created_at: datetime
```

`extra="forbid"` makes the model strict: a request with an unknown field is rejected. The contract is enforced at the schema level.

### 4.4 Schema evolution within a version

Adding a field to a response is non-breaking (clients ignore unknown fields). Removing a field is breaking (clients that depended on it fail). Renaming a field is breaking.

For backwards-compatible changes within a version, add fields with sensible defaults. For breaking changes, ship a new version.

---

## 5. The OpenAPI Schema Per Version

### 5.1 Why separate schemas

A single OpenAPI schema for both v1 and v2 would mix deprecated and current endpoints, making it hard for clients to discover. FastAPI generates the schema from the routers; separate routers = separate schemas.

### 5.2 Multi-schema setup

```python
# app/main.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()
app.include_router(users_v1.router, prefix="/v1")
app.include_router(users_v2.router, prefix="/v2")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="MyApp API",
        version="2.0.0",
        description="MyApp v2 is the current version. v1 is deprecated.",
        routes=app.routes,
    )
    # Mark v1 as deprecated in the OpenAPI doc
    for path, methods in schema["paths"].items():
        if path.startswith("/v1/"):
            for method in methods.values():
                method["deprecated"] = True
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
```

Clients can fetch `/openapi.json` to get the schema. The deprecation flag is preserved per-endpoint.

### 5.3 Documentation per version

The docs site (e.g., Mintlify, ReadMe, custom) should:
- Default to v2 in the navigation.
- Have a v1 archive section.
- Show the deprecation date on v1 endpoints.
- Provide a migration guide from v1 to v2.

---

## 6. The Four Common Versioning Mistakes

### 6.1 No versioning

```python
# ❌ One endpoint that evolves in place
@router.get("/users/{user_id}")
async def get_user(user_id: int): ...
# Today returns {id, email, name}; tomorrow returns {id, email, name, avatar_url};
# breaking every client that depended on the old shape
```

Without versioning, every change is a breaking change. The first time you remove a field or rename a key, clients break. Versioning is the safety net.

### 6.2 Versioning in the wrong place

```python
# ❌ Version in the body
{"v": 2, "user": {...}}

# ✅ Version in the URL
/v2/users
```

The body is the data; the URL is the API. Mixing the two makes routing and caching harder.

### 6.3 Unversioned deprecation

```python
# ❌ Removed v1 silently
# Today: v1 works
# Tomorrow: 404 on v1 (clients break)
```

A deprecated endpoint must continue to work until the sunset date. Silent removal breaks clients.

### 6.4 Versioned docs but unversioned runtime

```python
# ❌ Docs say v2; runtime serves v1
```

If the OpenAPI schema is for v2 but the runtime is v1, clients can't trust the docs. The runtime and the schema must match.

---

## 7. The Migration Window

### 7.1 The standard timeline

| Day | Action |
|-----|--------|
| 0 | Ship v2 alongside v1 |
| 0 | Document v2 as the recommended version |
| 0 | Add deprecation headers to v1 |
| 0 | Show a deprecation warning in the v1 docs |
| 90 | Review the metrics: how many clients still on v1? |
| 180 | Email v1 clients with the migration guide |
| 365 (1 year) | v1 sunset date; v1 returns 410 |
| 365+ | v1 is removed from the runtime |

### 7.2 The metrics to track

- **v1 requests per day**: how many clients are still on v1?
- **v1 error rate**: are v1 clients seeing errors? (signal of broken integrations)
- **Migration funnel**: how many clients have moved to v2?

A dashboard that shows v1 vs v2 traffic over time is the single most useful operational view for deprecation.

### 7.3 The deprecation email

When v1 is deprecated, send an email to v1 clients:

```
Subject: Action required — v1 of MyApp API will sunset on 2026-03-01

Hi [Client],

You're currently using v1 of the MyApp API. v1 will be retired on 2026-03-01.
After that date, v1 requests will return 410 Gone.

To migrate to v2:
1. Update your base URL from /v1/ to /v2/
2. Review the migration guide: https://docs.example.com/migrate-v1-to-v2
3. Update your response parsing (v2 includes avatar_url and created_at on User)
4. Test in your staging environment

Need help? Reply to this email or open a ticket at https://support.example.com.

Thanks,
The MyApp team
```

The email is the canonical communication; the headers and 410 are the runtime enforcement.

---

## 8. URL Design: Resources vs Actions

### 8.1 Resources (Level 2 REST)

```
GET    /v2/users              # list
POST   /v2/users              # create
GET    /v2/users/{id}         # read
PUT    /v2/users/{id}         # update
DELETE /v2/users/{id}         # delete
GET    /v2/users/{id}/posts   # nested resource
```

### 8.2 Actions (Level 1)

```
POST   /v2/users/{id}/activate
POST   /v2/users/{id}/deactivate
POST   /v2/users/{id}/reset-password
```

Actions are operations that don't fit CRUD. They are nested under the resource; the action name is in the URL path.

### 8.3 RPC-style (avoid)

```
POST   /v2/execute
{"action": "create_user", "params": {...}}
```

This is the "Swamp of POX". Avoid it. Even for non-CRUD actions, prefer a resource-oriented URL with a verb in the path.

---

## 9. Código de Compresión

```python
"""
Compresión: API Versioning Strategies
Covers: path-based versioning, deprecation headers, 410 Gone, version detection.
"""
from datetime import datetime
from fastapi import APIRouter, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# 1) Versioned routers
v1_router = APIRouter(prefix="/v1", deprecated=True)
v2_router = APIRouter(prefix="/v2")


@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int, response: Response):
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Sat, 01 Mar 2026 00:00:00 GMT"
    response.headers["Link"] = f'</v2/users/{user_id}>; rel="successor-version"'
    return {"id": user_id, "email": "alice@example.com", "name": "Alice"}


@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {
        "id": user_id, "email": "alice@example.com", "name": "Alice",
        "avatar_url": "https://cdn.example.com/avatars/1.jpg",
        "created_at": "2024-01-15T12:00:00Z",
    }


# 2) The 410 Gone catch-all
async def v1_gone(request: Request, path: str):
    problem = {
        "type": "https://example.com/probs/gone",
        "title": "Gone",
        "status": 410,
        "detail": f"v1 is no longer available. Use v2: /v2/{path}",
        "instance": str(request.url.path),
    }
    return JSONResponse(
        status_code=410,
        content=problem,
        media_type="application/problem+json",
        headers={"Link": f'</v2/{path}>; rel="successor-version"'},
    )


# 3) Version detection middleware
async def log_request(request: Request, call_next):
    path = request.url.path
    version = "v1" if path.startswith("/v1/") else "v2" if path.startswith("/v2/") else "unknown"
    print(f"request: {request.method} {path} (version={version})")
    response = await call_next(request)
    return response


# 4) OpenAPI customization
def custom_openapi():
    from fastapi.openapi.utils import get_openapi
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="MyApp API", version="2.0.0",
        description="v2 is current. v1 is deprecated.",
        routes=app.routes,
    )
    for path, methods in schema["paths"].items():
        if path.startswith("/v1/"):
            for method in methods.values():
                method["deprecated"] = True
    app.openapi_schema = schema
    return schema


app = FastAPI()
app.include_router(v1_router)
app.include_router(v2_router)
app.openapi = custom_openapi
```

---

## Key Takeaways

- **Path-based versioning** (`/v1/...`) is the industry default. Easy to read, easy to route, easy to test in a browser.
- **Header-based versioning** is more elegant but harder to consume. Use only if you have a strong reason.
- **Deprecated endpoints must continue to work** until the sunset date. Use `Deprecation: true`, `Sunset: <date>`, and `Link: <rel="successor-version">` headers.
- **The 410 Gone catch-all** is the cleanest way to retire a version. A single route returns 410 for every deprecated path.
- **Response models are the contract**. Adding fields is non-breaking; removing or renaming is breaking. Track schema evolution with Pydantic models.
- **OpenAPI per version**: separate schemas for v1 and v2; mark v1 as deprecated in the schema.
- **The migration window** is 6-12 months for non-breaking changes, 12-18 months for breaking. Track v1 vs v2 traffic to know when to sunset.
- **The 410 Gone response** includes the migration target. Clients see where to go next.
- **Resource-oriented URLs** (`/users/{id}/posts`) are Level 2 REST. Action-oriented URLs (`/users/{id}/activate`) are also Level 2. Avoid RPC-style (`/execute` with `action` in body).

## References

- [RFC 8594 — The Sunset HTTP Header Field](https://www.rfc-editor.org/rfc/rfc8594)
- [Microsoft REST API Guidelines — Versioning](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#123-versioning)
- [Google API Design Guide — Versioning](https://cloud.google.com/apis/design/versioning)
- [Stripe API Versioning](https://stripe.com/docs/api/versioning)
- [GitHub API Versioning](https://docs.github.com/en/rest/about-the-rest-api/api-versions)
- [FastAPI — Path Parameters and Routing](https://fastapi.tiangolo.com/tutorial/path-params/)
- [OpenAPI 3.1 Specification](https://spec.openapis.org/oas/v3.1.0)
