# 🎯 Response Caching Decorators

## 🎯 Learning Objectives

- Build a custom response-caching decorator with key derivation
- Implement per-user caching that respects authorization
- Use Pydantic models for typed cache values
- Compose multiple decorators (rate limit, cache, auth)
- Avoid the three most common decorator mistakes

## Introduction

The previous notes covered HTTP caching and Redis as a cache backend. This note is about a third pattern: **decorator-based response caching**. The handler returns a Pydantic model; the decorator checks the cache, returns the cached value on a hit, or runs the handler and caches the result on a miss. The handler code is unaware of the cache.

The pattern is powerful for two reasons: it isolates the caching concern (the handler is pure business logic), and it lets you compose caching with other decorators (rate limiting, auth, logging). The cost is a custom dependency that other handlers must opt into.

This note covers the pattern: key derivation, per-user caching, Pydantic-friendly values, and composition with other decorators. The capstone ties it all together.

---

## 1. The Basic Decorator

### 1.1 The shape

```python
from functools import wraps
from typing import Callable, TypeVar
import json
import hashlib
import redis.asyncio as redis

T = TypeVar("T")


def cached_response(
    ttl: int = 300,
    key_prefix: str = "",
    vary_on_user: bool = True,
):
    """Decorator that caches the response of an endpoint."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Derive the cache key
            key = _derive_key(func, args, kwargs, key_prefix, vary_on_user)
            # Check the cache
            cached = await cache.get(key)
            if cached:
                return _deserialize(cached, func.__annotations__.get("return"))
            # Cache miss: run the handler
            result = await func(*args, **kwargs)
            # Populate the cache
            await cache.set(key, _serialize(result), ttl=ttl)
            return result
        return wrapper
    return decorator
```

The decorator:
1. Derives a cache key from the function name + arguments.
2. Checks the cache; returns the cached value on a hit.
3. Runs the handler on a miss; populates the cache.

The handler is pure business logic; it doesn't know about the cache.

### 1.2 The key derivation

```python
def _derive_key(func, args, kwargs, key_prefix: str, vary_on_user: bool) -> str:
    # Use the function name as part of the key
    parts = [func.__name__]
    if key_prefix:
        parts.insert(0, key_prefix)
    # Add the arguments (positional and keyword)
    for arg in args:
        parts.append(str(arg))
    for k, v in sorted(kwargs.items()):
        if k in ("user", "current_user", "uow", "redis"):  # Skip non-cacheable deps
            continue
        if vary_on_user and k in ("user_id", "tenant_id"):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    # Hash to keep the key short
    raw = ":".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()
```

The key is a hash of the function name + the relevant arguments. Non-cacheable dependencies (database sessions, Redis clients) are skipped. User/tenant IDs are included for multi-tenant safety.

### 1.3 The serialization

```python
import json


def _serialize(result) -> str:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result)


def _deserialize(cached: str, return_type: type) -> any:
    if cached is None:
        return None
    if return_type and issubclass(return_type, BaseModel):
        return return_type.model_validate_json(cached)
    return json.loads(cached)
```

Pydantic models serialize to JSON; on the way out, they deserialize back to the model. The cache value is always a string.

---

## 2. Per-User Caching

### 2.1 The problem

A handler returns user-specific data. The cache must not serve one user's data to another. The key must include the user identity.

### 2.2 The implementation

```python
def cached_response(ttl: int = 300, key_prefix: str = ""):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user=None, **kwargs):
            # Get the user from the FastAPI dependency
            user_id = getattr(current_user, "id", None) if current_user else None
            if user_id is None:
                # No user (e.g., public endpoint); don't cache
                return await func(*args, current_user=current_user, **kwargs)
            # Key includes the user
            key = f"{key_prefix}:{func.__name__}:user={user_id}:{_hash_args(args, kwargs)}"
            cached = await cache.get(key)
            if cached:
                return _deserialize(cached, func.__annotations__.get("return"))
            result = await func(*args, current_user=current_user, **kwargs)
            await cache.set(key, _serialize(result), ttl=ttl)
            return result
        return wrapper
    return decorator
```

The pattern: extract the user from the FastAPI dependency; include the user ID in the key. Without a user (e.g., a public endpoint), don't cache.

### 2.3 The dependency-aware decorator

```python
import inspect
from fastapi import Depends


def cached_response(
    ttl: int = 300,
    key_prefix: str = "",
    include_user: bool = True,
):
    """Decorator that caches a FastAPI response, optionally per-user."""
    def decorator(func):
        sig = inspect.signature(func)
        # Identify the user parameter
        user_param = None
        for name, param in sig.parameters.items():
            if name in ("current_user", "user", "ctx"):
                user_param = name
                break

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the user
            user_id = None
            if user_param and user_param in kwargs:
                user = kwargs[user_param]
                user_id = getattr(user, "id", None) if user else None
            # Build the key
            key_parts = [key_prefix, func.__name__]
            if include_user and user_id:
                key_parts.append(f"u={user_id}")
            for k, v in sorted(kwargs.items()):
                if k in (user_param, "uow", "redis", "db"):
                    continue
                key_parts.append(f"{k}={v}")
            key = ":".join(p for p in key_parts if p)
            # Cache lookup
            cached = await cache.get(key)
            if cached:
                return _deserialize(cached, sig.return_annotation)
            # Run the handler
            result = await func(*args, **kwargs)
            # Populate the cache
            await cache.set(key, _serialize(result), ttl=ttl)
            return result
        return wrapper
    return decorator
```

The decorator introspects the function signature to find the user parameter. It supports multiple naming conventions (`current_user`, `user`, `ctx`).

---

## 3. Pydantic-Friendly Caching

### 3.1 The challenge

The handler returns a Pydantic model. The cache stores a string. The deserializer must reconstruct the model.

```python
@router.get("/users/{user_id}", response_model=UserOut)
@cached_response(ttl=300, key_prefix="user")
async def get_user(user_id: int, current_user: User = Depends(get_current_user)) -> UserOut:
    # ... fetch the user
    return UserOut(id=user.id, email=user.email, name=user.name)
```

The `response_model=UserOut` is the FastAPI-level declaration; the `-> UserOut` is the return-type annotation. The decorator uses the annotation to deserialize the cached value.

### 3.2 The annotated return type

```python
@cached_response(ttl=300, key_prefix="user")
async def get_user(user_id: int) -> UserOut:
    # ...

# At runtime
sig = inspect.signature(get_user)
return_type = sig.return_annotation  # UserOut
```

The decorator reads the annotation at decoration time. The handler's return type is the deserializer target.

### 3.3 The list response

```python
class UserListOut(BaseModel):
    items: list[UserOut]
    next_cursor: str | None = None


@cached_response(ttl=60, key_prefix="user-list")
async def list_users(
    limit: int = 20,
    cursor: str | None = None,
    current_user: User = Depends(get_current_user),
) -> UserListOut:
    # ... fetch the list
    return UserListOut(items=[...], next_cursor=...)
```

The decorator handles the Pydantic model. The list of users is serialized as JSON; on retrieval, it's deserialized back to `UserListOut` with the nested `UserOut` items.

---

## 4. Composing Decorators

### 4.1 The order matters

```python
@cached_response(ttl=300)        # 4. Inner: runs the handler
@rate_limit("100/min")           # 3. Checks the rate limit
@auth_required                   # 2. Checks the auth
@router.get("/users/{user_id}")  # 1. Outer: matches the route
async def get_user(user_id: int, current_user: User = Depends(get_current_user)):
    # ...
```

The order:
1. The route matches.
2. The auth decorator runs (no user, no handler).
3. The rate limit decorator runs (no user, no handler).
4. The cache decorator runs (with the user, decides whether to cache or run).
5. The handler runs on a cache miss.

The cache must come **after** auth (so the user is in scope) and **after** rate limit (so we don't count cache hits against the rate limit).

### 4.2 The rate-limit-aware cache

```python
def rate_limit_aware_cache(ttl: int = 300, rate_limit: str = "100/min"):
    def decorator(func):
        @cached_response(ttl=ttl)
        @rate_limit(rate_limit)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@router.get("/users/{user_id}")
@rate_limit_aware_cache(ttl=300, rate_limit="100/min")
async def get_user(user_id: int, current_user: User = Depends(get_current_user)):
    # ...
```

A single decorator that combines caching and rate limiting. The order is correct: rate limit first, then cache.

### 4.3 The auth-aware cache

```python
def auth_aware_cache(ttl: int = 300, key_prefix: str = ""):
    def decorator(func):
        @cached_response(ttl=ttl, key_prefix=key_prefix, include_user=True)
        @auth_required
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

Same pattern. The auth decorator is below the cache (so the user is in the cache key), but above the handler.

---

## 5. The Invalidation Pattern

### 5.1 The problem

A user updates their profile. The cached `/users/12345` is now stale. The TTL handles it eventually, but we want immediate invalidation.

### 5.2 The pattern: invalidation by tag

```python
# Set with a tag
async def set_with_tag(key: str, value: str, tag: str, ttl: int = 300):
    await cache.set(key, value, ttl=ttl)
    # Add the key to a tag set
    await redis.sadd(f"tag:{tag}", key)
    # Set a TTL on the tag set too
    await redis.expire(f"tag:{tag}", ttl + 60)


# Invalidate by tag
async def invalidate_tag(tag: str):
    members = await redis.smembers(f"tag:{tag}")
    if members:
        await redis.delete(*members)
    await redis.delete(f"tag:{tag}")
```

The pattern: every cache write also records the key in a tag set. To invalidate, delete all keys in the tag set.

### 5.3 The user-update case

```python
@cached_response(ttl=300, key_prefix="user")
async def get_user(user_id: int): ...


async def update_user(user_id: int, data: UserUpdate) -> User:
    # 1) Update the database
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # 2) Invalidate the cache
    await invalidate_tag(f"user:{user_id}")
    return user
```

The update handler invalidates the user's cache. The next read repopulates it.

### 5.4 The list-update case

```python
# A list endpoint that includes a user might also be cached
@cached_response(ttl=60, key_prefix="user-list")
async def list_users(...): ...


async def update_user(user_id: int, data: UserUpdate) -> User:
    # ... update the user
    # Invalidate the user AND any lists
    await invalidate_tag(f"user:{user_id}")
    await invalidate_tag("user-list")  # all user lists
    return user
```

A user update invalidates the user cache and all list caches. Lists are repopulated on the next read.

---

## 6. The Three Common Decorator Mistakes

### 6.1 Caching side-effecting handlers

```python
# ❌ The handler has side effects; caching it is dangerous
@cached_response(ttl=300)
@router.post("/users", status_code=201)
async def create_user(payload: UserCreate) -> UserOut:
    user = await db.create(**payload.model_dump())
    await send_welcome_email(user.email)  # side effect!
    return user
```

A POST that creates a user and sends an email must not be cached. The cache returns the cached value; the side effects never run.

The fix: only cache GET (and sometimes PUT/PATCH) endpoints. Use a decorator name that makes this explicit: `@cached_get_response(ttl=300)`.

### 6.2 Forgetting to handle cache failures

```python
# ❌ The cache is down; the handler fails
@cached_response(ttl=300)
async def get_user(user_id: int):
    user = await db.get(User, user_id)
    return UserOut.from_orm(user)
```

If Redis is down, the cache lookup fails, the cache write fails, and the handler returns an error. The user-facing experience is broken.

The fix: the cache is a performance optimization, not a correctness requirement. Cache failures should be logged and ignored:

```python
async def wrapper(*args, **kwargs):
    try:
        cached = await cache.get(key)
        if cached:
            return _deserialize(cached, ...)
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
    # Cache miss or cache failure; run the handler
    result = await func(*args, **kwargs)
    try:
        await cache.set(key, _serialize(result), ttl=ttl)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
    return result
```

The handler always runs. The cache is best-effort.

### 6.3 The unbounded key

```python
# ❌ The key includes a user-supplied string
@cached_response(ttl=300)
async def search_users(q: str): ...
# Key: "search_users:q=very long string with many special characters"
```

The key includes the user's search string. If the string is long or has special characters, the key is too long or has unexpected characters. Redis has a key length limit (512 MB, but practically <1 KB); a 1 MB key wastes memory.

The fix: hash the arguments to a fixed-length key:

```python
import hashlib

def _hash_args(args, kwargs) -> str:
    raw = json.dumps({"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items() if k not in ("uow", "redis", "current_user")}}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

A SHA-256 truncated to 16 hex characters is 64 bits of entropy and 16 characters long. Fixed length, no special characters, no length issues.

---

## 7. The Test Patterns

### 7.1 Test the cache hit

```python
@pytest.mark.asyncio
async def test_cache_hit(client, db_session, redis):
    # First request: cache miss
    response1 = await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    assert response1.status_code == 200
    # The cache is populated
    assert await redis.get("user:u=1:...") is not None
    # Second request: cache hit (verify by changing the DB and ensuring the response is the cached value)
    user = await db_session.get(User, 1)
    user.name = "Modified"
    await db_session.commit()
    response2 = await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    assert response2.json()["name"] == "Alice"  # cached value
```

The test changes the database after the first request; the second request should return the cached (old) value. If it returns the new value, the cache is not working.

### 7.2 Test the invalidation

```python
@pytest.mark.asyncio
async def test_invalidation_on_update(client, redis):
    # Populate the cache
    await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    assert await redis.get("user:u=1:...") is not None
    # Update the user
    await client.put(
        "/v2/users/1",
        json={"name": "Alice Updated"},
        headers={"Authorization": "Bearer ..."},
    )
    # The cache is invalidated
    assert await redis.get("user:u=1:...") is None
```

### 7.3 Test the cache failure

```python
@pytest.mark.asyncio
async def test_cache_failure_falls_through(client, redis, monkeypatch):
    # Make the cache fail
    async def broken_get(*args, **kwargs):
        raise ConnectionError("Redis is down")
    monkeypatch.setattr(cache, "get", broken_get)
    # The handler still works
    response = await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    assert response.status_code == 200
```

The cache is best-effort. A Redis outage should not break the service.

---

## 8. The Production Decorator

The full decorator with all the production patterns:

```python
import inspect
import json
import hashlib
import logging
from functools import wraps
from typing import Callable, TypeVar

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T")


def cached_response(
    ttl: int = 300,
    key_prefix: str = "",
    include_user: bool = True,
    skip_deps: tuple = ("uow", "redis", "db", "current_user", "ctx"),
):
    """Decorator that caches a FastAPI response, with key derivation, per-user safety, and graceful failure."""
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        # Identify the user parameter
        user_param = None
        for name in ("current_user", "user", "ctx"):
            if name in sig.parameters:
                user_param = name
                break

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build the cache key
            key = _build_key(func, sig, args, kwargs, key_prefix, include_user, user_param, skip_deps)
            # Try the cache
            try:
                cached = await _get_cache().get(key)
                if cached is not None:
                    return _deserialize(cached, sig.return_annotation)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
            # Cache miss; run the handler
            result = await func(*args, **kwargs)
            # Populate the cache
            try:
                await _get_cache().set(key, _serialize(result), ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
            return result
        return wrapper
    return decorator


def _build_key(func, sig, args, kwargs, key_prefix, include_user, user_param, skip_deps) -> str:
    parts = [key_prefix, func.__name__]
    if include_user and user_param and user_param in kwargs:
        user = kwargs[user_param]
        user_id = getattr(user, "id", None) if user else None
        if user_id:
            parts.append(f"u={user_id}")
    # Hash the arguments (excluding non-cacheable deps)
    cacheable = {k: v for k, v in kwargs.items() if k not in skip_deps and k != user_param}
    cacheable.update({i: v for i, v in enumerate(args)})
    if cacheable:
        raw = json.dumps(cacheable, sort_keys=True, default=str)
        parts.append(hashlib.sha256(raw.encode()).hexdigest()[:16])
    return ":".join(p for p in parts if p)


def _serialize(result) -> str:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result)


def _deserialize(cached: str, return_type: type) -> any:
    if return_type and isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type.model_validate_json(cached)
    return json.loads(cached)


# Global cache (set in lifespan)
_cache = None


def _get_cache():
    global _cache
    if _cache is None:
        _cache = Cache(Cache.REDIS, endpoint="redis://localhost:6379")
    return _cache
```

---

## 9. Código de Compresión

```python
"""
Compresión: Response Caching Decorators
Covers: key derivation, per-user safety, Pydantic-friendly serialization,
       graceful cache failure, tag-based invalidation.
"""
import hashlib
import inspect
import json
import logging
from functools import wraps
from typing import Callable

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# 1) The decorator
def cached_response(ttl: int = 300, key_prefix: str = "", include_user: bool = True):
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        user_param = next(
            (name for name in ("current_user", "user", "ctx") if name in sig.parameters),
            None,
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = _build_key(func, sig, args, kwargs, key_prefix, include_user, user_param)
            try:
                cached = await _cache.get(key)
                if cached is not None:
                    return _deserialize(cached, sig.return_annotation)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
            result = await func(*args, **kwargs)
            try:
                await _cache.set(key, _serialize(result), ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
            return result
        return wrapper
    return decorator


# 2) Key builder
def _build_key(func, sig, args, kwargs, key_prefix, include_user, user_param) -> str:
    parts = [key_prefix or "", func.__name__]
    if include_user and user_param and user_param in kwargs:
        user = kwargs[user_param]
        user_id = getattr(user, "id", None) if user else None
        if user_id:
            parts.append(f"u={user_id}")
    # Hash arguments
    cacheable = {
        k: v for k, v in kwargs.items()
        if k not in ("uow", "redis", "db", user_param)
    }
    for i, v in enumerate(args):
        cacheable[f"_{i}"] = v
    if cacheable:
        raw = json.dumps(cacheable, sort_keys=True, default=str)
        parts.append(hashlib.sha256(raw.encode()).hexdigest()[:16])
    return ":".join(p for p in parts if p)


# 3) Serialization
def _serialize(result) -> str:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result)


def _deserialize(cached, return_type):
    if return_type and isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type.model_validate_json(cached)
    return json.loads(cached)


# 4) Tag-based invalidation
async def invalidate_tag(tag: str):
    members = await redis.smembers(f"tag:{tag}")
    if members:
        await redis.delete(*members)
    await redis.delete(f"tag:{tag}")


# 5) Usage
@router.get("/users/{user_id}", response_model=UserOut)
@cached_response(ttl=300, key_prefix="user")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
) -> UserOut:
    user = await uow.users.get(user_id)
    return UserOut.from_orm(user)
```

---

## Key Takeaways

- **Decorator-based response caching** isolates the cache concern from the handler. The handler is pure business logic.
- **Key derivation** is the hard part. The key must be unique to the request (function + args + user) and short (use a hash for unbounded strings).
- **Per-user caching** is mandatory for user-specific responses. Include the user ID in the key.
- **Pydantic models** serialize to JSON and deserialize back. The decorator uses the return type annotation as the deserializer target.
- **Compose decorators** in the right order: route → auth → rate limit → cache → handler. The cache comes after auth (so the user is in scope) and after rate limit (so cache hits don't count).
- **Cache failures must be best-effort**. A Redis outage should not break the service. Log the failure and fall through to the handler.
- **Tag-based invalidation** is the right pattern for invalidating groups of related cache keys (a user and all the lists they're in).
- **Don't cache side-effecting handlers**. POST that creates a user, sends an email, etc. must not be cached. Use the cache only for read endpoints (and idempotent PUT/PATCH).

## References

- [aiocache Documentation](https://aiocache.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [FastAPI Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [Python functools — wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
- [Redis — Tag-based Invalidation](https://redis.io/docs/latest/operate/oss_and_stack/management/config-file/)
- [Effective Python Caching Patterns](https://realpython.com/lru-cache-python/)
