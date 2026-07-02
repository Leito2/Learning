# 🌐 HTTP Caching: ETag, Cache-Control

## 🎯 Learning Objectives

- Understand the HTTP caching stack: how `Cache-Control` and `ETag` work
- Implement ETag-based conditional requests in FastAPI
- Build a custom middleware that adds the right cache headers to every response
- Distinguish between public, private, no-cache, and no-store caches
- Avoid the four most common HTTP caching mistakes

## Introduction

HTTP caching is the first and most effective layer in the caching stack. A single `Cache-Control: public, max-age=3600` header turns a 200ms request into a 1ms browser-cache hit — a 200x speedup with one line of code. The cost is correctness: the client may serve stale data if the server doesn't tell it when the data has changed.

The HTTP caching model has three players: the **origin server** (your FastAPI app), the **cache** (browser, CDN, gateway), and the **client** (which may or may not honor the cache). The headers they exchange — `Cache-Control`, `ETag`, `If-None-Match`, `Last-Modified`, `If-Modified-Since` — form a contract. The server promises the cache will be valid for some time; the cache promises to revalidate when that time expires.

This note covers the headers, the conditional request flow, and the FastAPI middleware that implements them. The patterns apply to any HTTP API; the FastAPI specifics are minimal.

---

## 1. The HTTP Caching Headers

### 1.1 `Cache-Control`

`Cache-Control` is the primary header. It has multiple directives:

| Directive | Meaning |
|-----------|---------|
| `public` | Any cache (browser, CDN, proxy) may store the response |
| `private` | Only the user's browser may store the response; no shared cache |
| `no-cache` | Cache may store, but must revalidate before serving |
| `no-store` | Cache must not store at all (sensitive data) |
| `max-age=N` | The response is fresh for N seconds |
| `s-maxage=N` | Like max-age, but for shared caches (CDN, proxy) |
| `must-revalidate` | Cache must revalidate when stale; do not serve stale |
| `immutable` | The response will never change; cache forever |

The default in HTTP/1.1 is `no-cache max-age=0`: every response must be revalidated. To enable caching, the server must explicitly opt in.

### 1.2 `ETag`

`ETag` is a unique identifier for the response. The client sends it back in `If-None-Match`; if the ETag matches, the server returns 304 Not Modified (no body).

```http
GET /users/12345 HTTP/1.1

HTTP/1.1 200 OK
ETag: "abc123"
Cache-Control: private, max-age=300
{"id": 12345, ...}

# Subsequent request
GET /users/12345 HTTP/1.1
If-None-Match: "abc123"

HTTP/1.1 304 Not Modified
ETag: "abc123"
```

The 304 response has no body; the client uses the cached body. This is fast and saves bandwidth.

### 1.3 `Last-Modified` and `If-Modified-Since`

A weaker form of ETag. The server returns `Last-Modified: <date>`; the client sends it back in `If-Modified-Since`. If the resource hasn't changed since that date, the server returns 304.

`Last-Modified` is a date (second-level precision), so it's not as precise as an ETag. Use ETag for any non-trivial resource.

### 1.4 `Vary`

`Vary: <header>` tells the cache: "this response varies based on this request header". A response with `Vary: Accept-Encoding` is cached separately for each `Accept-Encoding` value. Without `Vary`, the cache might serve a gzipped response to a client that doesn't support gzip.

Common `Vary` values:

| Header | When |
|--------|------|
| `Vary: Accept-Encoding` | Response depends on compression |
| `Vary: Accept-Language` | Response depends on language |
| `Vary: Authorization` | Response depends on the auth token (don't cache!) |
| `Vary: Origin` | CORS response depends on origin |

For multi-tenant APIs, **`Vary: Authorization` is critical** — without it, the cache might serve one tenant's data to another.

---

## 2. The Caching Strategy by Endpoint Type

Different endpoints have different caching requirements:

| Endpoint type | Cache-Control | ETag | Vary |
|---------------|--------------|------|------|
| Public static content | `public, max-age=31536000, immutable` | Optional | None |
| Public dynamic content (e.g., a blog post) | `public, max-age=300, must-revalidate` | Yes | `Accept-Encoding` |
| User-specific (e.g., profile) | `private, max-age=60, must-revalidate` | Yes | `Authorization` |
| Real-time (e.g., a chat message) | `no-store` | No | None |
| Sensitive (e.g., a credit card form) | `no-store, private` | No | None |
| Idempotent POST result | `private, no-cache` | No | None |

The patterns:

- **Public static content**: cache forever. CDN handles it.
- **Public dynamic content**: cache briefly; revalidate with ETag.
- **User-specific**: cache in the user's browser only; revalidate often.
- **Real-time or sensitive**: no cache at all.

### 2.1 The decision tree

```
Is the response user-specific?
├── No: Is it static or rarely-changing?
│   ├── Yes: public, max-age=long (immutable if truly static)
│   └── No: public, max-age=short, must-revalidate, ETag
└── Yes: Is it real-time or sensitive?
    ├── Yes: no-store (or no-cache + private)
    └── No: private, max-age=short, must-revalidate, ETag, Vary: Authorization
```

---

## 3. ETag in FastAPI

### 3.1 The middleware

```python
# app/middleware/etag.py
import hashlib
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class ETagMiddleware(BaseHTTPMiddleware):
    """Add ETag to JSON responses, handle If-None-Match."""
    
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Only add ETags to GETs that succeeded
        if request.method != "GET" or response.status_code != 200:
            return response
        # Don't add ETags to responses that explicitly opt out
        if response.headers.get("Cache-Control") == "no-store":
            return response
        # Read the body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        # Compute the ETag (weak: same algorithm, no semantic guarantee)
        etag = f'W/"{hashlib.md5(body).hexdigest()}"'
        # If the client sends the same ETag, return 304
        if request.headers.get("If-None-Match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        # Add the ETag to the response
        response.headers["ETag"] = etag
        # Re-wrap the body (consumed by reading)
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
```

The middleware:
1. Reads the response body.
2. Computes the ETag (a hash of the body).
3. If the client's `If-None-Match` matches, returns 304 Not Modified.
4. Otherwise, returns the body with the ETag header.

The "weak" prefix (`W/"..."`) indicates that the body is semantically equivalent but not byte-identical (e.g., JSON key order might differ).

### 3.2 The `Cache-Control` middleware

```python
# app/middleware/cache_control.py
from starlette.middleware.base import BaseHTTPMiddleware


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Add Cache-Control headers to responses based on the route."""
    
    DEFAULT_RULES = {
        # Path pattern -> Cache-Control value
        "/v2/users/me": "private, no-cache",
        "/v2/users/{id}": "private, max-age=60, must-revalidate",
        "/v2/projects": "private, max-age=30, must-revalidate",
        "/v2/health": "no-store",
    }
    
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Determine the cache control from the route
        path = request.url.path
        cache_control = self._match_cache_control(path)
        if cache_control:
            response.headers["Cache-Control"] = cache_control
            # Add Vary for user-specific responses
            if "private" in cache_control:
                response.headers["Vary"] = "Authorization"
        return response
    
    def _match_cache_control(self, path: str) -> str | None:
        # ... match path against DEFAULT_RULES
        # (use a router or simple pattern matching)
        for pattern, value in self.DEFAULT_RULES.items():
            if self._match(pattern, path):
                return value
        return None
```

A simpler pattern: use a decorator that handlers can apply to themselves:

```python
# app/core/cache_headers.py
from functools import wraps
from fastapi import Response


def cache_control(value: str, vary: list[str] | None = None):
    """Decorator to add Cache-Control headers to a handler."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the Response object in the kwargs
            response: Response = kwargs.get("response")
            if response is None:
                # Create one if the handler didn't
                response = Response()
                kwargs["response"] = response
            response.headers["Cache-Control"] = value
            if vary:
                response.headers["Vary"] = ", ".join(vary)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@router.get("/me")
@cache_control("private, no-cache", vary=["Authorization"])
async def get_me(...): ...
```

The decorator pattern is more explicit but requires every handler to opt in. The middleware is automatic but less flexible.

---

## 4. Conditional Requests

### 4.1 The flow

A conditional request asks the server: "is this resource still the same?"

```http
GET /users/12345 HTTP/1.1
If-None-Match: "abc123"

HTTP/1.1 304 Not Modified
ETag: "abc123"
```

The server responds 304 if the resource hasn't changed (no body); 200 with the new body if it has.

### 4.2 The cost saving

For a 1KB JSON response, the body is ~99% of the bytes sent. A 304 response is just the headers (~200 bytes). The bandwidth saving is 5-10x; the latency saving is 2-5x (the body doesn't travel).

### 4.3 The implementation

The middleware above handles `If-None-Match`. The client (browser, `curl`, httpx) handles `If-None-Match` automatically if you've seen the ETag.

```python
import httpx

# First request: server returns ETag
response1 = httpx.get("https://api.example.com/users/12345")
etag = response1.headers["ETag"]

# Second request: client sends If-None-Match
response2 = httpx.get(
    "https://api.example.com/users/12345",
    headers={"If-None-Match": etag},
)
# response2.status_code is 304 if the data hasn't changed
```

The `httpx` library doesn't automatically send `If-None-Match` from cached responses. The client needs to implement this. The `requests-cache` library adds the behavior on top of `requests`.

---

## 5. The Cache-Control Heuristics

### 5.1 How long should you cache?

The right TTL depends on:

| Factor | Rule of thumb |
|--------|--------------|
| Static assets (JS, CSS, images) | `max-age=31536000, immutable` (1 year) |
| Web fonts | `max-age=31536000, immutable` |
| Public API responses | `max-age=300` (5 min) with ETag revalidation |
| User-specific responses | `max-age=60` (1 min) with ETag revalidation |
| Real-time data | `no-store` |

The shorter the TTL, the more often clients hit the server. The longer the TTL, the more stale data. The right answer is "as long as the data is useful".

### 5.2 The stale-while-revalidate pattern

`Cache-Control: max-age=60, stale-while-revalidate=300` says: "the response is fresh for 60 seconds; after that, serve the stale version for up to 300 seconds while a background request refreshes it".

```http
# Client receives a 200 with:
Cache-Control: max-age=60, stale-while-revalidate=300
# 70 seconds later, the cache serves the stale version
# AND issues a background request to refresh
```

The user sees the stale version immediately; the next request gets the fresh version. This is the standard pattern for "fast even when the server is slow".

FastAPI's response headers support this directly:

```python
response.headers["Cache-Control"] = "max-age=60, stale-while-revalidate=300"
```

---

## 6. The Four Common HTTP Caching Mistakes

### 6.1 Caching user-specific responses as public

```python
# ❌ User A's profile cached, served to User B
@router.get("/users/me")
async def get_me(...):
    return {"id": user.id, "email": user.email}

# Middleware adds: Cache-Control: public, max-age=300
# User B's request hits the cache and gets User A's data!
```

The fix: user-specific responses are `private`, never `public`. And add `Vary: Authorization` to ensure the cache separates by user.

### 6.2 Forgetting `Vary` for variable responses

```python
# ❌ A response that depends on Accept-Language
@router.get("/welcome")
async def welcome(request):
    if "es" in request.headers.get("Accept-Language", ""):
        return {"message": "Bienvenido"}
    return {"message": "Welcome"}

# The cache stores the English version and serves it to Spanish clients
```

The fix: add `Vary: Accept-Language` to the response. The cache then stores a separate version per language.

### 6.3 Returning `max-age=0, must-revalidate` without ETag

```python
# ❌ Tells the client to revalidate, but gives no way to do it efficiently
response.headers["Cache-Control"] = "max-age=0, must-revalidate"
# No ETag header; the client must fetch the full body every time
```

The fix: either give a longer TTL (with ETag for revalidation) or explicitly say `no-store` (don't cache at all).

### 6.4 Caching POST responses

```python
# ❌ A POST that has side effects; caching it is dangerous
@router.post("/charge", cache_control("public, max-age=3600"))
async def charge(request):
    # Idempotent? Maybe. But the default is wrong.
    ...
```

POST responses are typically not cached. If you have an idempotent POST (e.g., a payment with an idempotency key), the cache key must include the request body, not just the URL.

---

## 7. The CDN Layer

### 7.1 What a CDN adds

A CDN (Cloudflare, CloudFront, Fastly) sits between the client and your server. It caches responses closer to the user. The `Cache-Control` headers you set on your server propagate to the CDN.

For a CDN to cache, the response must be `public` (or `s-maxage` for the CDN specifically). The `private` directive tells the CDN not to cache.

### 7.2 The `s-maxage` directive

```python
response.headers["Cache-Control"] = "public, max-age=60, s-maxage=600"
```

- `max-age=60`: the browser can use the cached response for 60 seconds.
- `s-maxage=600`: the CDN can use the cached response for 600 seconds.

The CDN caches longer than the browser because the CDN serves many users; the browser caches per-user.

### 7.3 The `Vary: Accept-Encoding` and CDN

A CDN strips `Accept-Encoding` from the request when caching (and the cache stores separate versions for gzip, br, etc.). The `Vary: Accept-Encoding` header is automatic; the CDN handles it.

### 7.4 Cache invalidation on the CDN

When you deploy a new version, you need to invalidate the CDN cache. Most CDNs have an API:

```python
import httpx


async def invalidate_cdn(paths: list[str]):
    """Invalidate Cloudflare cache for the given paths."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.cloudflare.com/client/v4/zones/{ZONE_ID}/purge_cache",
            headers={"Authorization": f"Bearer {CF_TOKEN}"},
            json={"files": [f"https://api.example.com{p}" for p in paths]},
        )
        return response.json()
```

The CDN purges the cached responses; the next request goes to the origin.

---

## 8. The Cache-Control Cheatsheet

```python
def cache_headers_for(endpoint_type: str) -> dict:
    """Returns the right Cache-Control and Vary headers for an endpoint type."""
    rules = {
        "public_static": {
            "Cache-Control": "public, max-age=31536000, immutable",
        },
        "public_dynamic": {
            "Cache-Control": "public, max-age=300, must-revalidate",
        },
        "user_specific": {
            "Cache-Control": "private, max-age=60, must-revalidate",
            "Vary": "Authorization",
        },
        "real_time": {
            "Cache-Control": "no-store",
        },
        "sensitive": {
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
        },
    }
    return rules.get(endpoint_type, {})
```

Pick the rule that matches the endpoint; apply via middleware or decorator.

---

## 9. The Test Cases

```python
# Test that the ETag is set
def test_etag_is_set_on_get(client):
    response = await client.get("/v2/users/12345")
    assert response.status_code == 200
    assert "ETag" in response.headers


# Test that 304 is returned when the ETag matches
def test_304_when_etag_matches(client):
    response1 = await client.get("/v2/users/12345")
    etag = response1.headers["ETag"]
    response2 = await client.get(
        "/v2/users/12345",
        headers={"If-None-Match": etag},
    )
    assert response2.status_code == 304
    assert response2.headers["ETag"] == etag


# Test that Cache-Control is correct for user-specific endpoints
def test_user_specific_has_private_and_vary(client):
    response = await client.get("/v2/users/me", headers={"Authorization": "Bearer ..."})
    assert "private" in response.headers["Cache-Control"]
    assert "Authorization" in response.headers.get("Vary", "")


# Test that public endpoints have the right Cache-Control
def test_public_endpoint_cacheable(client):
    response = await client.get("/v2/projects")
    assert "public" in response.headers.get("Cache-Control", "") or "max-age" in response.headers.get("Cache-Control", "")
```

These tests catch the most common mistakes at the unit-test level.

---

## 10. Código de Compresión

```python
"""
Compresión: HTTP Caching
Covers: ETag middleware, Cache-Control middleware, conditional requests,
       304 Not Modified, Vary header.
"""
import hashlib
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# 1) ETag middleware
class ETagMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.method != "GET" or response.status_code != 200:
            return response
        if response.headers.get("Cache-Control") == "no-store":
            return response

        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        etag = f'W/"{hashlib.md5(body).hexdigest()}"'

        if request.headers.get("If-None-Match") == etag:
            return Response(status_code=304, headers={"ETag": etag})

        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


# 2) Cache-Control decorator
def cache_control(value: str, vary: list[str] | None = None):
    from functools import wraps
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            response = kwargs.get("response")
            if response is None:
                response = Response()
                kwargs["response"] = response
            response.headers["Cache-Control"] = value
            if vary:
                response.headers["Vary"] = ", ".join(vary)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 3) The decision helper
def cache_headers_for(endpoint_type: str) -> dict:
    rules = {
        "public_static": {"Cache-Control": "public, max-age=31536000, immutable"},
        "public_dynamic": {"Cache-Control": "public, max-age=300, must-revalidate"},
        "user_specific": {
            "Cache-Control": "private, max-age=60, must-revalidate",
            "Vary": "Authorization",
        },
        "real_time": {"Cache-Control": "no-store"},
        "sensitive": {"Cache-Control": "no-store, no-cache, must-revalidate, private"},
    }
    return rules.get(endpoint_type, {})


# 4) Usage
@router.get("/me")
async def get_me(response: Response):
    response.headers.update(cache_headers_for("user_specific"))
    return {"id": 1, "email": "alice@example.com"}


@router.get("/projects")
async def list_projects(response: Response):
    response.headers.update(cache_headers_for("public_dynamic"))
    return [...]
```

---

## Key Takeaways

- **HTTP caching** is the first and most effective layer. A single `Cache-Control` header can turn a 200ms request into a 1ms browser-cache hit.
- The **right Cache-Control** depends on the endpoint type: public static (`public, max-age=31536000, immutable`), public dynamic (`public, max-age=300, must-revalidate`), user-specific (`private, max-age=60, must-revalidate, Vary: Authorization`), real-time (`no-store`).
- **ETags** enable conditional requests: the client sends `If-None-Match`, the server returns 304 if the data hasn't changed. The bandwidth and latency saving is significant.
- **`Vary: Authorization`** is critical for multi-tenant APIs. Without it, the cache might serve one tenant's data to another.
- **`stale-while-revalidate`** serves a stale response immediately while a background request refreshes the cache. The user never waits.
- **Never cache user-specific responses as `public`**. Always `private` and `Vary: Authorization`.
- **POST responses are not cached by default**. If you have an idempotent POST, the cache key must include the request body.
- **The CDN** propagates the Cache-Control headers you set. `s-maxage` is for the CDN specifically; `max-age` is for the browser.
- **The middleware pattern** (adds ETag + Cache-Control to all responses) is the simplest. The decorator pattern (per-handler) is more explicit.

## References

- [RFC 9111 — HTTP Caching](https://www.rfc-editor.org/rfc/rfc9111)
- [RFC 7232 — HTTP/1.1 Conditional Requests](https://www.rfc-editor.org/rfc/rfc7232)
- [MDN — HTTP Caching](https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching)
- [Web Fundamentals — HTTP Caching](https://developers.google.com/web/fundamentals/performance/optimizing-content-efficiency/http-caching)
- [FastAPI — Response Headers](https://fastapi.tiangolo.com/advanced/response-headers/)
- [Starlette — Middleware](https://www.starlette.io/middleware/)
- [Cloudflare — Cache-Control](https://developers.cloudflare.com/cache/concepts/cache-control/)
