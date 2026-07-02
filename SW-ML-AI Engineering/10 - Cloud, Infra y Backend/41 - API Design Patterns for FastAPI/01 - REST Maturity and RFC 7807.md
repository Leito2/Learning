# 🏛️ REST Maturity and RFC 7807 Error Responses

## 🎯 Learning Objectives

- Understand the Richardson Maturity Model and choose the right level for your API
- Build error responses that conform to RFC 7807 (Problem Details for HTTP APIs)
- Implement FastAPI exception handlers that produce consistent errors
- Avoid the six most common error response mistakes

## Introduction

Two patterns define the user experience of any API: how successful responses are structured and how errors are communicated. Successful responses are easy; every framework gives you JSON. Errors are hard: there are a dozen common shapes (plain text, custom JSON, wrapped envelopes, problem-details), each with subtle semantic differences. The industry has converged on **RFC 7807** as the standard. FastAPI does not implement it out of the box; this note shows how to add it.

The Richardson Maturity Model is a complementary concern: it describes the four levels of REST (resources, HTTP verbs, hypermedia controls, and the mythical Level 3 "true REST"). Most production APIs are Level 2; knowing where you are on the model helps you decide which patterns to apply.

---

## 1. The Richardson Maturity Model

### 1.1 The four levels

| Level | What | Example |
|-------|------|---------|
| 0 | The Swamp of POX | One URI, one HTTP verb (POST), all operations |
| 1 | Resources | Multiple URIs, one verb each, body has the action |
| 2 | HTTP Verbs | Multiple URIs, multiple verbs, no action in body |
| 3 | Hypermedia Controls | Links in responses tell the client what to do next |

The vast majority of public APIs are Level 2: Stripe, GitHub, Twilio. They use HTTP semantics (GET, POST, PUT, DELETE) and have well-defined resources. Level 3 (HATEOAS, hypermedia) is rare in practice; the added complexity rarely justifies the benefit.

### 1.2 Where your API should target

| API type | Target level | Why |
|----------|-------------|-----|
| Public REST API (Stripe, GitHub) | Level 2 | Industry standard; tooling exists; clients expect it |
| Internal microservice | Level 1 or 2 | Less formal; team conventions suffice |
| Hypermedia-driven (e.g., banking) | Level 3 | Compliance or domain requirements |
| gRPC / async messaging | N/A | Not REST; use Protocol Buffers, not HTTP verbs |

For a FastAPI service, **Level 2 is the right target**. GET to read, POST to create, PUT/PATCH to update, DELETE to remove. Use HTTP status codes semantically. Don't put actions in the body.

---

## 2. RFC 7807: Problem Details

### 2.1 What the RFC says

RFC 7807 defines a standard JSON shape for error responses:

```json
{
  "type": "https://example.com/probs/out-of-credit",
  "title": "You do not have enough credit.",
  "detail": "Your current balance is 30, but that costs 50.",
  "instance": "/account/12345/msgs/abc",
  "balance": 30,
  "accounts": ["/account/12345", "/account/67890"]
}
```

The required fields:

| Field | Type | Meaning |
|-------|------|---------|
| `type` | URI | A URI identifying the problem type (a doc URL) |
| `title` | string | A short, human-readable summary |
| `status` | int | The HTTP status code (optional but recommended) |
| `detail` | string | A human-readable explanation specific to this occurrence |
| `instance` | URI | A URI identifying the specific occurrence |

Extension fields are allowed: any field not in the standard is preserved. The example above extends with `balance` and `accounts`. Clients ignore unknown fields.

### 2.2 The content type

RFC 7807 responses use the `application/problem+json` content type. The `+json` suffix indicates a structured suffix per RFC 6839. Clients can detect the format and parse accordingly.

### 2.3 Why use RFC 7807

- **Tooling support**: many HTTP libraries have built-in handling.
- **Self-documenting**: the `type` URI points to a doc page that describes the problem.
- **Machine-readable**: extension fields are first-class.
- **Standard**: clients don't need to learn your custom error format.

---

## 3. Implementing RFC 7807 in FastAPI

### 3.1 The ProblemDetails class

```python
# app/errors.py
from typing import Any
from pydantic import BaseModel, Field, HttpUrl


class ProblemDetails(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""
    type: HttpUrl = Field(default="about:blank")
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    
    # Extension fields are allowed
    model_config = {"extra": "allow"}
    
    def to_json(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)
```

The `extra="allow"` config is critical: it lets you add custom fields like `balance` or `retry_after`.

### 3.2 Custom exception classes

```python
# app/errors.py
class ProblemException(Exception):
    """Base exception for RFC 7807 problems."""
    def __init__(
        self,
        *,
        title: str,
        status: int,
        type: str = "about:blank",
        detail: str | None = None,
        instance: str | None = None,
        **extensions: Any,
    ):
        self.problem = ProblemDetails(
            type=type,
            title=title,
            status=status,
            detail=detail,
            instance=instance,
            **extensions,
        )
        super().__init__(title)


class NotFoundProblem(ProblemException):
    def __init__(self, detail: str, instance: str | None = None):
        super().__init__(
            type="https://example.com/probs/not-found",
            title="Not Found",
            status=404,
            detail=detail,
            instance=instance,
        )


class ValidationProblem(ProblemException):
    def __init__(self, errors: list[dict], instance: str | None = None):
        super().__init__(
            type="https://example.com/probs/validation",
            title="Validation Failed",
            status=422,
            detail="One or more fields failed validation.",
            instance=instance,
            errors=errors,
        )


class ConflictProblem(ProblemException):
    def __init__(self, detail: str, instance: str | None = None):
        super().__init__(
            type="https://example.com/probs/conflict",
            title="Conflict",
            status=409,
            detail=detail,
            instance=instance,
        )


class RateLimitProblem(ProblemException):
    def __init__(self, retry_after: int, instance: str | None = None):
        super().__init__(
            type="https://example.com/probs/rate-limit",
            title="Too Many Requests",
            status=429,
            detail="You have exceeded the rate limit.",
            instance=instance,
            retry_after=retry_after,
        )
```

### 3.3 The exception handlers

```python
# app/errors.py
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def problem_response(problem: ProblemDetails, request: Request) -> JSONResponse:
    """Build an RFC 7807 JSON response from a ProblemDetails object."""
    headers = {}
    if problem.status == 429 and "retry_after" in problem.model_extra:
        headers["Retry-After"] = str(problem.model_extra["retry_after"])
    return JSONResponse(
        status_code=problem.status,
        content=problem.to_json(),
        media_type="application/problem+json",
        headers=headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Install all RFC 7807 error handlers on the FastAPI app."""
    
    @app.exception_handler(ProblemException)
    async def handle_problem(request: Request, exc: ProblemException):
        return problem_response(exc.problem, request)
    
    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        # Convert Pydantic errors to a list of {field, message, type}
        errors = [
            {
                "field": ".".join(str(loc) for loc in e["loc"]),
                "message": e["msg"],
                "type": e["type"],
            }
            for e in exc.errors()
        ]
        problem = ProblemDetails(
            type="https://example.com/probs/validation",
            title="Validation Failed",
            status=422,
            detail="One or more fields failed validation.",
            instance=str(request.url.path),
            errors=errors,
        )
        return problem_response(problem, request)
    
    @app.exception_handler(404)
    async def handle_404(request: Request, exc):
        problem = ProblemDetails(
            type="https://example.com/probs/not-found",
            title="Not Found",
            status=404,
            detail=f"No route matches {request.url.path}",
            instance=str(request.url.path),
        )
        return problem_response(problem, request)
    
    @app.exception_handler(405)
    async def handle_405(request: Request, exc):
        problem = ProblemDetails(
            type="https://example.com/probs/method-not-allowed",
            title="Method Not Allowed",
            status=405,
            detail=f"Method {request.method} not allowed for {request.url.path}",
            instance=str(request.url.path),
        )
        return problem_response(problem, request)
    
    @app.exception_handler(500)
    async def handle_500(request: Request, exc):
        # In production, log the exception details
        # logger.exception(f"Unhandled error on {request.url.path}")
        problem = ProblemDetails(
            type="https://example.com/probs/internal",
            title="Internal Server Error",
            status=500,
            detail="An unexpected error occurred.",
            instance=str(request.url.path),
        )
        return problem_response(problem, request)
```

The `install_error_handlers` function is called in the FastAPI app factory:

```python
# app/main.py
from app.errors import install_error_handlers


def create_app() -> FastAPI:
    app = FastAPI(...)
    install_error_handlers(app)
    return app
```

### 3.4 Using the exceptions in handlers

```python
@router.get("/users/{user_id}")
async def get_user(user_id: int, uow: UnitOfWork = Depends(get_uow)):
    user = await uow.users.get(user_id)
    if not user:
        raise NotFoundProblem(
            detail=f"User {user_id} does not exist",
            instance=f"/users/{user_id}",
        )
    return user


@router.post("/users", status_code=201)
async def create_user(payload: UserCreate, uow: UnitOfWork = Depends(get_uow)):
    if await uow.users.get_by(email=payload.email):
        raise ConflictProblem(
            detail=f"User with email {payload.email} already exists",
        )
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    return user
```

The handler is clean. The exception class carries the status code, type, and any extensions.

### 3.5 The client side

A client receiving an RFC 7807 response can:

```python
import httpx

response = httpx.get("https://api.example.com/users/12345")
if response.headers["content-type"] == "application/problem+json":
    problem = response.json()
    print(f"Error type: {problem['type']}")
    print(f"Title: {problem['title']}")
    print(f"Status: {problem['status']}")
    print(f"Detail: {problem['detail']}")
    # Custom extension fields
    if "retry_after" in problem:
        print(f"Retry after: {problem['retry_after']} seconds")
```

The structured format makes error handling predictable.

---

## 4. The Six Common Error Response Mistakes

### 4.1 Returning a wrapped envelope

```json
// ❌ Custom shape
{
  "success": false,
  "error": {
    "code": "user_not_found",
    "message": "User 12345 does not exist"
  }
}

// ✅ RFC 7807
{
  "type": "https://example.com/probs/not-found",
  "title": "Not Found",
  "status": 404,
  "detail": "User 12345 does not exist"
}
```

The envelope is fine for successful responses (where it differentiates from errors); for errors, the standard is RFC 7807.

### 4.2 Including the stack trace in production

```python
# ❌ Stack trace in the response
return JSONResponse({
    "detail": str(exc),
    "traceback": traceback.format_exc(),
}, status_code=500)
```

The stack trace is a security risk (it reveals internal structure). Log it server-side; return a generic message to the client.

### 4.3 Using 200 for errors

```python
# ❌ 200 OK with an error in the body
return JSONResponse({
    "status": "error",
    "message": "User not found",
})
```

HTTP status codes are designed for this. Use them.

### 4.4 Using 500 for client errors

```python
# ❌ 500 for a missing user
if not user:
    raise Exception("User not found")
```

A missing user is a 404, not a 500. Reserve 500 for unexpected server errors.

### 4.5 Inconsistent error shapes

```python
# ❌ Different shapes for different errors
return {"error": "Not found"}  # 404
return {"detail": "Bad request"}  # 400
return {"message": "Internal error"}  # 500
```

Use one shape for all errors. RFC 7807 is the standard.

### 4.6 Returning English-only errors

```python
# ❌ English-only
detail="User not found"

# ✅ Localized
detail=user_facing_message("user_not_found", locale=locale)
```

For public APIs, support localization via the `Accept-Language` header and translate the `title` and `detail` accordingly.

---

## 5. RFC 7807 + Authentication

### 5.1 The 401 challenge

A 401 response must include a `WWW-Authenticate` header:

```python
@app.exception_handler(401)
async def handle_401(request: Request, exc):
    problem = ProblemDetails(
        type="https://example.com/probs/unauthorized",
        title="Unauthorized",
        status=401,
        detail="Authentication required",
        instance=str(request.url.path),
    )
    response = problem_response(problem, request)
    response.headers["WWW-Authenticate"] = 'Bearer realm="api", error="invalid_token"'
    return response
```

The `WWW-Authenticate` header tells the client how to authenticate. Without it, RFC 7235 requires the response to be discarded.

### 5.2 The 403 problem

```python
@app.exception_handler(403)
async def handle_403(request: Request, exc):
    problem = ProblemDetails(
        type="https://example.com/probs/forbidden",
        title="Forbidden",
        status=403,
        detail="You do not have permission to access this resource",
        instance=str(request.url.path),
    )
    return problem_response(problem, request)
```

403 means the user is authenticated but lacks permission. The `type` URI can point to docs explaining the permission model.

---

## 6. Testing Error Responses

### 6.1 Test each exception class

```python
@pytest.mark.asyncio
async def test_404_returns_problem_details(client):
    response = await client.get("/users/99999")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"] == "https://example.com/probs/not-found"
    assert body["title"] == "Not Found"
    assert body["status"] == 404
    assert "99999" in body["detail"]


@pytest.mark.asyncio
async def test_422_validation_returns_field_errors(client):
    response = await client.post("/users", json={"email": "not-an-email", "name": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "https://example.com/probs/validation"
    assert "errors" in body
    assert len(body["errors"]) > 0
```

### 6.2 Test the content type

```python
def test_all_errors_use_problem_json():
    """Every 4xx/5xx response must be application/problem+json."""
    # ... test all endpoints with bad input
    for response in error_responses:
        assert response.headers["content-type"] == "application/problem+json"
```

A linter or contract test ensures consistency.

### 6.3 Test extension fields

```python
@pytest.mark.asyncio
async def test_429_includes_retry_after_header(client):
    # Trigger a rate limit
    for _ in range(100):
        await client.get("/users")
    response = await client.get("/users")
    assert response.status_code == 429
    assert "retry_after" in response.json()
    assert response.headers["Retry-After"] == str(response.json()["retry_after"])
```

The `Retry-After` header is a HTTP standard; the `retry_after` body field is the RFC 7807 extension.

---

## 7. Código de Compresión

```python
"""
Compresión: REST Maturity and RFC 7807
Covers: problem details, exception handlers, custom exception classes.
"""
from typing import Any
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl


# 1) ProblemDetails
class ProblemDetails(BaseModel):
    type: HttpUrl = Field(default="about:blank")
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None

    model_config = {"extra": "allow"}


# 2) Custom exceptions
class ProblemException(Exception):
    def __init__(self, *, title, status, type="about:blank", detail=None, instance=None, **extensions):
        self.problem = ProblemDetails(
            type=type, title=title, status=status, detail=detail, instance=instance, **extensions
        )
        super().__init__(title)


class NotFoundProblem(ProblemException):
    def __init__(self, detail, instance=None):
        super().__init__(
            type="https://example.com/probs/not-found",
            title="Not Found", status=404, detail=detail, instance=instance,
        )


class ValidationProblem(ProblemException):
    def __init__(self, errors, instance=None):
        super().__init__(
            type="https://example.com/probs/validation",
            title="Validation Failed", status=422,
            detail="One or more fields failed validation.", instance=instance,
            errors=errors,
        )


class ConflictProblem(ProblemException):
    def __init__(self, detail, instance=None):
        super().__init__(
            type="https://example.com/probs/conflict",
            title="Conflict", status=409, detail=detail, instance=instance,
        )


class RateLimitProblem(ProblemException):
    def __init__(self, retry_after, instance=None):
        super().__init__(
            type="https://example.com/probs/rate-limit",
            title="Too Many Requests", status=429,
            detail="You have exceeded the rate limit.", instance=instance,
            retry_after=retry_after,
        )


# 3) Response builder
def problem_response(problem: ProblemDetails, request: Request) -> JSONResponse:
    headers = {}
    if problem.status == 429 and "retry_after" in (problem.model_extra or {}):
        headers["Retry-After"] = str(problem.model_extra["retry_after"])
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(mode="json", exclude_none=True),
        media_type="application/problem+json",
        headers=headers,
    )


# 4) Install handlers
def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemException)
    async def handle_problem(request, exc):
        return problem_response(exc.problem, request)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request, exc):
        errors = [
            {"field": ".".join(str(l) for l in e["loc"]), "message": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return problem_response(
            ProblemDetails(
                type="https://example.com/probs/validation",
                title="Validation Failed", status=422,
                detail="One or more fields failed validation.",
                instance=str(request.url.path), errors=errors,
            ),
            request,
        )

    @app.exception_handler(404)
    async def handle_404(request, exc):
        return problem_response(
            ProblemDetails(
                type="https://example.com/probs/not-found",
                title="Not Found", status=404,
                detail=f"No route matches {request.url.path}",
                instance=str(request.url.path),
            ),
            request,
        )

    @app.exception_handler(500)
    async def handle_500(request, exc):
        return problem_response(
            ProblemDetails(
                type="https://example.com/probs/internal",
                title="Internal Server Error", status=500,
                detail="An unexpected error occurred.",
                instance=str(request.url.path),
            ),
            request,
        )
```

---

## Key Takeaways

- **RFC 7807** is the standard for HTTP API error responses. The required fields are `type`, `title`, `status`; `detail` and `instance` are optional. Extension fields are allowed.
- The response must use the `application/problem+json` content type.
- FastAPI does not implement RFC 7807 out of the box; build it with custom exception classes, exception handlers, and a `ProblemDetails` Pydantic model.
- The **Richardson Maturity Model** describes four levels of REST. Most public APIs are Level 2; target Level 2 unless you have a reason for Level 3.
- **Use HTTP status codes semantically**: 404 for missing, 409 for conflict, 422 for validation, 429 for rate limit, 500 for unexpected.
- **Always include a `WWW-Authenticate` header** on 401 responses.
- **Never include stack traces** in production responses. Log them server-side.
- **Be consistent**: every 4xx/5xx response should be the same shape. A linter or contract test enforces this.

## References

- [RFC 7807 — Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc7807)
- [RFC 6839 — Additional Media Type Structured Syntax Suffixes](https://www.rfc-editor.org/rfc/rfc6839)
- [RFC 7235 — HTTP/1.1 Authentication](https://www.rfc-editor.org/rfc/rfc7235)
- [Microsoft REST API Guidelines — Error Handling](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#710-error-condition-responses)
- [Google API Design Guide — Errors](https://cloud.google.com/apis/design/errors)
- [Stripe API Design — Errors](https://stripe.com/docs/api/errors)
- [Richardson Maturity Model](https://martinfowler.com/articles/richardsonMaturityModel.html)
- [FastAPI Exception Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
