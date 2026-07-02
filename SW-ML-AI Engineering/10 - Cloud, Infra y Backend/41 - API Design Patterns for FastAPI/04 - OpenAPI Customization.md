# 📋 OpenAPI Customization

## 🎯 Learning Objectives

- Customize the OpenAPI schema with descriptions, examples, and tags that help clients
- Configure security schemes (API key, OAuth2, OpenID Connect) so clients can authenticate from the docs
- Add multiple OpenAPI specs (one per version) for versioned APIs
- Generate client SDKs from the OpenAPI schema
- Avoid the four most common OpenAPI mistakes

## Introduction

A FastAPI service's default OpenAPI schema is functional but generic. It has the endpoint names, the request/response shapes, the status codes. What it lacks is the context that turns a schema into documentation: descriptions that explain intent, examples that show usage, security schemes that let clients authenticate, tags that group related endpoints, and server URLs that match the environment.

The OpenAPI 3.1 spec is the standard format; FastAPI generates it automatically. The work is in the customization: making the schema useful for a developer who's never seen your API. The investment pays off in faster client integration, fewer support questions, and the ability to generate client SDKs.

This note covers the customization patterns: descriptions, examples, security schemes, tags, servers, and the multi-spec pattern for versioned APIs. The capstone shows them in a real service.

---

## 1. The Default OpenAPI Schema

### 1.1 What FastAPI generates

A FastAPI app with handlers generates a `/openapi.json` automatically. The schema has:

- The endpoint paths and methods.
- The Pydantic request/response models (with their fields).
- The status codes (from the `responses` parameter).
- The tags (from `APIRouter(tags=[...])`).

What it does **not** generate by default:

- Detailed descriptions of what the endpoint does.
- Examples of requests and responses.
- Security schemes (you must add them).
- Server URLs for different environments.
- Custom metadata (logo, contact, license).

### 1.2 Why customize

A client developer opens the Swagger UI and sees:

```yaml
/users/{user_id}:
  get:
    summary: Get User
    parameters:
      - name: user_id
        in: path
        required: true
        schema:
          type: integer
    responses:
      '200':
        description: Successful Response
```

That's enough for the developer to make a request. But they don't know:
- What does a User look like exactly?
- What does an error response look like?
- How do I authenticate?
- What does "200 with no body" mean here?

Customization answers these questions in the schema, so the client doesn't have to read separate docs.

---

## 2. The Customization Layers

### 2.1 Layer 1: App-level metadata

```python
# app/main.py
from fastapi import FastAPI

app = FastAPI(
    title="MyApp API",
    description="""
    The MyApp API provides programmatic access to users, projects, and tasks.

    ## Authentication
    All endpoints require a Bearer token. Get one via `POST /v1/auth/login`.

    ## Versioning
    v1 is current. v2 is the recommended version.

    ## Support
    - Documentation: https://docs.example.com
    - Support: https://support.example.com
    """,
    version="2.0.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "api-support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "users", "description": "Operations on user accounts."},
        {"name": "projects", "description": "Operations on projects."},
        {"name": "tasks", "description": "Operations on tasks."},
        {"name": "auth", "description": "Authentication and authorization."},
    ],
)
```

The `description` is rendered as Markdown in the docs UI. The `contact` and `license_info` are visible in the header. The `openapi_tags` add descriptions to tag groups.

### 2.2 Layer 2: Endpoint descriptions and examples

```python
from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/users", tags=["users"])


class UserCreate(BaseModel):
    email: str = Field(
        ...,
        description="The user's email address. Must be unique.",
        examples=["alice@example.com"],
    )
    name: str = Field(
        ...,
        description="The user's full name.",
        examples=["Alice Smith"],
        min_length=1,
        max_length=100,
    )


@router.post(
    "",
    response_model=UserOut,
    status_code=201,
    summary="Create a user",
    description="""
    Create a new user account. The email must be unique across the system.

    Returns the created user with a generated id.

    Errors:
    - 409 Conflict: Email already exists.
    - 422 Unprocessable Entity: Validation failed.
    """,
    responses={
        409: {"description": "Email already exists"},
        422: {"description": "Validation failed"},
    },
)
async def create_user(payload: UserCreate, uow: UnitOfWork = Depends(get_uow)):
    """Create a new user account."""
    if await uow.users.get_by(email=payload.email):
        raise ConflictProblem(...)
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    return user
```

The `summary` is a one-liner; the `description` is the detailed explanation (Markdown). The `responses` adds documentation for non-200 status codes. The `examples` on the Pydantic fields show up in the request body schema.

### 2.3 Layer 3: Response examples

```python
class UserOut(BaseModel):
    id: int
    email: str
    name: str
    avatar_url: str | None = None
    created_at: datetime
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "email": "alice@example.com",
                    "name": "Alice Smith",
                    "avatar_url": "https://cdn.example.com/avatars/1.jpg",
                    "created_at": "2024-01-15T12:00:00Z",
                }
            ]
        }
    }
```

The `examples` in the model config shows up in the response schema. Clients see a real-looking example in the docs.

### 2.4 Layer 4: Path and query documentation

```python
@router.get(
    "/{user_id}",
    response_model=UserOut,
    summary="Get a user",
)
async def get_user(
    user_id: int = Path(
        ...,
        description="The unique identifier of the user.",
        examples=[1],
    ),
    include_posts: bool = Query(
        False,
        description="Whether to include the user's posts in the response.",
    ),
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.get(user_id)
    if not user:
        raise NotFoundProblem(...)
    return user
```

`Path` and `Query` are like `Body` but for path and query parameters. The `description` and `examples` show up in the schema.

---

## 3. Security Schemes

### 3.1 Why document security

A client developer needs to know:
- How do I authenticate? (Bearer token, API key, OAuth2?)
- Where do I put the credentials? (Header, query, cookie?)
- How do I get a credential in the first place? (Login endpoint, OAuth flow?)

The OpenAPI `securitySchemes` block answers all three. The docs UI shows an "Authorize" button that prompts for credentials; subsequent requests include them automatically.

### 3.2 The HTTPBearer scheme

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()


@router.get(
    "/me",
    response_model=UserOut,
    summary="Get the current user",
)
async def get_me(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = jwt.decode(token, ...)
    user = await uow.users.get(int(payload["sub"]))
    return user
```

FastAPI's `HTTPBearer` automatically documents the security scheme in the OpenAPI schema. The docs UI shows a "Bearer" auth box.

### 3.3 The OAuth2 scheme

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="v1/auth/login",
    scopes={
        "read:users": "Read user information",
        "write:users": "Modify user information",
    },
)


@router.get("/me", dependencies=[Security(oauth2_scheme, scopes=["read:users"])])
async def get_me(): ...
```

The `OAuth2PasswordBearer` documents the OAuth2 password flow with scopes. The docs UI shows a "Password" auth box with username and password fields; on submit, it requests a token from `tokenUrl`.

### 3.4 The API key scheme

```python
from fastapi.security import APIKeyHeader

api_key = APIKeyHeader(name="X-API-Key")


@router.get("/external/data", dependencies=[Depends(api_key)])
async def get_external_data(): ...
```

For machine-to-machine APIs that use API keys (not user authentication).

### 3.5 Multiple schemes

```python
from fastapi import Security
from fastapi.security import HTTPBearer, APIKeyHeader

user_auth = HTTPBearer()
api_key_auth = APIKeyHeader(name="X-API-Key")


@router.get("/user/data", dependencies=[Security(user_auth)])
async def get_user_data(): ...


@router.get("/external/data", dependencies=[Security(api_key_auth)])
async def get_external_data(): ...
```

Different endpoints can require different security schemes. The OpenAPI schema documents each.

---

## 4. Servers

### 4.1 The default

A FastAPI app with no `servers` config has one server: the URL the request is made to. The docs show this as "the server".

### 4.2 Multiple environments

```python
app = FastAPI(
    servers=[
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
        {"url": "http://localhost:8000", "description": "Local development"},
    ],
)
```

The docs UI lets the user switch between servers. The "Try it out" button uses the selected server.

### 4.3 Per-environment overrides

For different deployments:

```python
# Production: only production server
app = FastAPI(servers=[{"url": "https://api.example.com", "description": "Production"}])

# Staging: staging + production
app = FastAPI(
    servers=[
        {"url": "https://staging-api.example.com", "description": "Staging"},
        {"url": "https://api.example.com", "description": "Production"},
    ]
)

# Local: local + staging
app = FastAPI(
    servers=[
        {"url": "http://localhost:8000", "description": "Local"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
    ]
)
```

The deployment environment determines which servers are documented.

---

## 5. The Multi-Spec Pattern for Versioned APIs

### 5.1 The need

A versioned API has two distinct APIs (v1, v2). The OpenAPI schema for v1 is deprecated; the schema for v2 is current. A single schema that mixes both is confusing.

### 5.2 The pattern

```python
# app/main.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

# App 1: v1 (deprecated)
v1_app = FastAPI(
    title="MyApp API (v1, deprecated)",
    version="1.0.0",
    openapi_url="/openapi-v1.json",
    docs_url="/docs-v1",
    redoc_url="/redoc-v1",
)
v1_app.include_router(users_v1.router, prefix="/v1")
v1_app.openapi = lambda: mark_deprecated(v1_app.openapi_schema or build_v1_schema(v1_app))

# App 2: v2 (current)
v2_app = FastAPI(
    title="MyApp API (v2, current)",
    version="2.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)
v2_app.include_router(users_v2.router, prefix="/v2")

# Mount both in the main app
app = FastAPI()
app.mount("/v1", v1_app)
app.mount("/v2", v2_app)
```

A single FastAPI app, two mounted sub-apps, two distinct OpenAPI specs.

### 5.3 Marking deprecated endpoints

```python
def mark_deprecated(schema: dict) -> dict:
    """Add deprecation markers to all paths in a v1 schema."""
    if not schema.get("paths"):
        return schema
    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            op["deprecated"] = True
            op.setdefault("description", "")
            op["description"] = (
                f"**DEPRECATED** — v1 is sunset on 2026-03-01. Use v2: "
                f"`{path.replace('/v1/', '/v2/')}`\n\n" + op["description"]
            )
    return schema
```

Clients see the deprecation directly in the schema. The migration path is shown alongside each endpoint.

---

## 6. Code Generation

### 6.1 Generating client SDKs

The OpenAPI schema is the input to client SDK generators. The most popular are:

- **openapi-generator**: the canonical tool, supports 50+ languages.
- **openapi-typescript-codegen**: TypeScript with full type safety.
- **ferdium-openapi**: a thin wrapper around openapi-generator.
- **speakeasy**: a managed service for SDK generation.

```bash
# Install openapi-generator
npm install @openapitools/openapi-generator-cli -g

# Generate a Python client
openapi-generator-cli generate \
    -i http://localhost:8000/openapi.json \
    -g python \
    -o ./client-python \
    --additional-properties=package_name=myapp_client

# Generate a TypeScript client
openapi-generator-cli generate \
    -i http://localhost:8000/openapi.json \
    -g typescript-fetch \
    -o ./client-ts
```

The generated client has typed methods for every endpoint, request/response models, and the security scheme. A client developer runs `pip install myapp-client` and has the entire API in their IDE.

### 6.2 What makes a good client SDK

A generated client is only as good as the schema. The schema needs:
- **Types** (Pydantic does this).
- **Examples** (so generated docs are useful).
- **Security scheme** (so the client can authenticate).
- **Tags** (so the generated client groups methods logically).
- **Server URLs** (so the client knows the base URL).

If your schema has all of these, the generated client is excellent. If it's the default FastAPI output, the generated client is functional but untyped and undocumented.

---

## 7. The Documentation Site

### 7.1 Swagger UI

FastAPI's default `/docs` page is Swagger UI. The default is functional; the customizations above make it useful. For a public API, host Swagger UI at a public URL with a clear "v1 is deprecated" banner.

### 7.2 ReDoc

FastAPI's `/redoc` page is ReDoc, which is more readable for humans. For a public API, ReDoc is the right default for the marketing site; Swagger UI is the right default for developer tools.

### 7.3 Custom portals

For a polished developer experience, use a custom portal:
- [Mintlify](https://mintlify.com/)
- [ReadMe](https://readme.com/)
- [Stoplight](https://stoplight.io/)

These services host your OpenAPI schema, add tutorials, code examples, and a changelog. Most can import an OpenAPI schema directly.

### 7.4 The versioned docs site

```yaml
# Mintlify navigation example
- group: v2 (current)
  pages:
    - users
    - projects
    - tasks
- group: v1 (deprecated, sunset 2026-03-01)
  pages:
    - users-v1
    - projects-v1
    - migration-guide
```

The v1 section is visible but marked as deprecated. The migration guide is in the v1 section (clients find it when looking at v1) and in the v2 section (clients find it before migrating).

---

## 8. The Four Common OpenAPI Mistakes

### 8.1 The default schema is good enough

The default schema has the basics. For a small internal service, that's fine. For a public API, the default is missing descriptions, examples, security schemes, and tags. The investment in customization pays off in client developer time.

### 8.2 Pydantic field names as user-facing names

```python
# ❌ Internal name exposed
class User(BaseModel):
    hashed_pw: str  # client sees "hashed_pw" in the docs


# ✅ User-friendly names with aliases
class User(BaseModel):
    hashed_password: str = Field(..., alias="hashedPw", serialization_alias="password_hash")
```

The schema field name is part of the public API. Use the public name in the schema, even if the internal name is different.

### 8.3 Forgetting error responses

```python
# ❌ Only the happy path is documented
@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int): ...

# ✅ Document the error paths too
@router.get(
    "/users/{user_id}",
    response_model=UserOut,
    responses={
        404: {"description": "User not found", "model": ProblemDetails},
        401: {"description": "Authentication required", "model": ProblemDetails},
    },
)
```

Clients need to know what the errors look like. Use the `ProblemDetails` model from [[01 - REST Maturity and RFC 7807|note 01]].

### 8.4 One schema for all environments

```python
# ❌ The schema is for production; staging is a mystery
app = FastAPI()  # no servers config

# ✅ Multiple environments documented
app = FastAPI(
    servers=[
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
    ]
)
```

The docs UI lets the client switch servers. The "Try it out" button works against the selected server.

---

## 9. The OpenAPI Validation

### 9.1 The schema as a contract

The OpenAPI schema is the contract between the API and its consumers. A change to the schema is a change to the contract. Use tools to validate that:
- The deployed API matches the schema.
- The schema doesn't change without notice.
- Client SDKs can be regenerated from the latest schema.

```bash
# Spectral: lint the OpenAPI schema
npx @stoplight/spectral lint openapi.json

# Diff two versions
npx @stoplight/spectral diff openapi-v1.json openapi-v2.json
```

Spectral catches:
- Unused schemas.
- Inconsistent naming.
- Missing descriptions.
- Deprecated patterns.

### 9.2 Contract testing

```python
# Schemathesis: fuzz the API against the schema
import schemathesis

schema = schemathesis.openapi.from_uri("http://localhost:8000/openapi.json")

@schema.parametrize()
def test_api(case):
    # Schemathesis generates requests that match the schema
    response = case.call()
    case.validate_response(response)
```

Schemathesis generates random requests that conform to the schema, calls the API, and validates the response. It catches:
- Endpoints that crash on valid input.
- Endpoints that return invalid responses.
- Type mismatches between schema and runtime.

---

## 10. The Operations Dashboard

### 10.1 What to track

A useful OpenAPI-driven dashboard shows:

- **Most-used endpoints**: which endpoints are hit most?
- **Error rates by endpoint**: which endpoints fail most?
- **Latency by endpoint**: which endpoints are slow?
- **Auth failures by endpoint**: which endpoints have auth issues?

The OpenAPI schema is the join key. Each API call has a path and method; the schema tells you what each path is supposed to do.

### 10.2 Per-endpoint SLOs

```python
# OpenAPI tag → SLO
SLO = {
    "users": {"latency_p99_ms": 100, "error_rate": 0.001},
    "projects": {"latency_p99_ms": 200, "error_rate": 0.001},
    "tasks": {"latency_p99_ms": 200, "error_rate": 0.001},
}
```

The OpenAPI tags let you group endpoints into SLOs. A dashboard shows each group's metrics.

---

## 11. Código de Compresión

```python
"""
Compresión: OpenAPI Customization
Covers: descriptions, examples, security schemes, multi-version schemas,
       deprecation markers.
"""
from fastapi import FastAPI, Path, Query, Security
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from pydantic import BaseModel, Field
from datetime import datetime


# 1) App-level metadata
app = FastAPI(
    title="MyApp API",
    description="""
    The MyApp API provides programmatic access to users, projects, and tasks.

    ## Authentication
    All endpoints require a Bearer token. Get one via `POST /v1/auth/login`.
    """,
    version="2.0.0",
    contact={"name": "API Support", "email": "api-support@example.com"},
    license_info={"name": "MIT"},
    servers=[
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
    ],
    openapi_tags=[
        {"name": "users", "description": "Operations on user accounts."},
        {"name": "projects", "description": "Operations on projects."},
        {"name": "auth", "description": "Authentication and authorization."},
    ],
)


# 2) Security schemes
bearer = HTTPBearer()
oauth2 = OAuth2PasswordBearer(
    tokenUrl="v1/auth/login",
    scopes={
        "read:users": "Read user information",
        "write:users": "Modify user information",
    },
)


# 3) Pydantic with examples
class UserOut(BaseModel):
    id: int
    email: str
    name: str
    avatar_url: str | None = None
    created_at: datetime

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": 1, "email": "alice@example.com", "name": "Alice",
                "avatar_url": "https://cdn.example.com/avatars/1.jpg",
                "created_at": "2024-01-15T12:00:00Z",
            }]
        }
    }


# 4) Endpoint with full documentation
@router.get(
    "/{user_id}",
    response_model=UserOut,
    summary="Get a user",
    description="""
    Get a single user by their unique identifier.

    Returns the user object, or a 404 if the user does not exist.
    """,
    responses={
        404: {"description": "User not found", "model": ProblemDetails},
        401: {"description": "Authentication required"},
    },
)
async def get_user(
    user_id: int = Path(..., description="The unique user ID", examples=[1]),
    bearer_creds: HTTPAuthorizationCredentials = Security(bearer),
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.get(user_id)
    if not user:
        raise NotFoundProblem(...)
    return user


# 5) Multi-version OpenAPI
def mark_deprecated(schema: dict) -> dict:
    for path, methods in schema.get("paths", {}).items():
        for method, op in methods.items():
            if isinstance(op, dict) and "deprecated" not in op:
                op["deprecated"] = True
    return schema


def custom_openapi_v1():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title="MyApp API (v1, deprecated)",
        version="1.0.0",
        routes=v1_app.routes,
    )
    return mark_deprecated(schema)


v1_app.openapi = custom_openapi_v1
```

---

## Key Takeaways

- The **default FastAPI OpenAPI schema** is functional but generic. Customization adds descriptions, examples, security schemes, and tags.
- **Descriptions and examples** are the difference between a schema that documents and one that just exists. The investment is small; the payoff is faster client integration.
- **Security schemes** (HTTPBearer, OAuth2PasswordBearer, APIKeyHeader) make the docs interactive. The "Authorize" button in Swagger UI lets clients test authenticated endpoints.
- **Multiple OpenAPI specs** for versioned APIs. Each version has its own docs page, its own schema, and its own deprecation markers.
- **Deprecated endpoints** in the schema should be marked with the `deprecated: true` flag and a clear migration path in the description.
- **Server URLs** in the schema let the docs UI show the available environments. The "Try it out" button uses the selected server.
- **Generated client SDKs** depend on a good schema. The investment in customization pays off in client developer time.
- **Contract testing** (Schemathesis, Spectral) validates that the deployed API matches the schema. It catches regressions before clients do.
- **Spectral** lints the schema for common mistakes: unused schemas, missing descriptions, inconsistent naming.

## References

- [OpenAPI 3.1 Specification](https://spec.openapis.org/oas/v3.1.0)
- [FastAPI — OpenAPI Documentation](https://fastapi.tiangolo.com/tutorial/metadata/)
- [Spectral — OpenAPI Linter](https://stoplight.io/open-source/spectral)
- [Schemathesis — API Fuzzing](https://schemathesis.readthedocs.io/)
- [openapi-generator](https://openapi-generator.tech/)
- [Stoplight Elements](https://stoplight.io/open-source/elements)
- [Mintlify](https://mintlify.com/)
- [ReadMe](https://readme.com/)
