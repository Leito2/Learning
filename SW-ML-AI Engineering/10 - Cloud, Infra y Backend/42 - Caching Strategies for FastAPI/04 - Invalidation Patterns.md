# ♻️ Invalidation Patterns

## 🎯 Learning Objectives

- Choose the right invalidation strategy: TTL, event-based, or dependency-aware
- Implement event-based invalidation on writes
- Handle the cache stampede: thundering herd, expired-while-fetching, and write-write races
- Avoid the four most common invalidation mistakes in production

## Introduction

The previous notes covered how to read from and write to the cache. This note covers the harder problem: **when to delete the cache**. A wrong cache value is a bug; a stale cache value is a slow-burning bug that customers discover weeks after deploy. The patterns in this note are what separate a working cache from a reliable one.

The four strategies, in order of complexity:
- **TTL only**: the cache expires after a fixed time. The simplest; data may be stale within the TTL window.
- **Event-based invalidation**: writes delete the cache. The most common; data is fresh immediately.
- **Dependency-aware invalidation**: writes invalidate related cache keys (a user update invalidates user lists). More complex; data is consistent.
- **Cache stampede prevention**: the cache expires; the first request refreshes it; concurrent requests don't all hit the database.

This note covers each, with the production patterns that make them reliable.

---

## 1. The Four Strategies

### 1.1 TTL only

```python
await cache.set(f"user:{user_id}", user_data, ex=300)
# 5 minutes later, the cache expires
# The next request is a cache miss
# It hits the database, populates the cache
```

Pros: trivial to implement; the data is at most TTL seconds old. Cons: the data is always at least 0 seconds old; if the TTL is 5 minutes, a user might see stale data for up to 5 minutes.

### 1.2 Event-based invalidation

```python
async def update_user(user_id: int, data: UserUpdate) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # Invalidate the cache
    await cache.delete(f"user:{user_id}")
    return user
```

Pros: the data is fresh immediately after the write. Cons: every write needs to know which cache keys to invalidate; missing one is a stale-data bug.

### 1.3 Dependency-aware invalidation

```python
async def update_user(user_id: int, data: UserUpdate) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # Invalidate the user AND any lists that contain this user
    await cache.delete(f"user:{user_id}")
    await cache.delete_pattern("user-list:*")  # all user lists
    return user
```

Pros: comprehensive; the cache is always consistent with the DB. Cons: `delete_pattern` is expensive (Redis scans all keys); large keyspaces need a tagging system.

### 1.4 Cache stampede prevention

```python
# When the cache expires, the first request to miss acquires a lock
# Subsequent requests wait for the lock holder to populate the cache
# (covered in the previous note)
```

Pros: the database is not overwhelmed by concurrent cache misses. Cons: more code; lock holder must be careful about failure modes.

---

## 2. The Pattern: TTL with Event-Based Invalidation

### 2.1 The hybrid

The right default for most services: TTL on every cache entry, plus invalidation on writes. The TTL is the safety net; the invalidation is the immediate update.

```python
# Read
async def get_user(user_id: int) -> User | None:
    cache_key = f"user:{user_id}"
    cached = await cache.get(cache_key)
    if cached:
        return UserOut.model_validate_json(cached)
    user = await db.get(User, user_id)
    if user is None:
        return None
    # Set with a 5-minute TTL
    await cache.set(cache_key, user.model_dump_json(), ex=300)
    return user


# Write
async def update_user(user_id: int, data: UserUpdate) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # Invalidate the cache immediately
    await cache.delete(f"user:{user_id}")
    return user
```

The TTL handles:
- Cases where invalidation is forgotten.
- Cases where the invalidation runs but the cache is unreachable (e.g., a network blip).
- Cases where the write is read by another process that doesn't know to invalidate.

### 2.2 The TTL selection

The right TTL balances freshness and cache hit rate:

| Data type | TTL | Rationale |
|-----------|-----|-----------|
| User profile | 5-15 min | Most fields change rarely; the user might notice a 5-min delay on email change |
| Project metadata | 5-15 min | Similar |
| Search results | 1-5 min | The result set is volatile |
| Aggregations (counts, stats) | 1-2 min | Computed from the DB; frequent updates |
| Hot data (top 10 trending) | 1 min | Updates frequently |
| Session/auth | No cache | Always read from DB or Redis directly |
| Rarely-changing (countries) | 1 hour+ | Stable data |

The "right" TTL is "as long as the data is useful, as short as necessary for freshness". For a user profile, 5 minutes is the common choice.

---

## 3. The Pattern: Tag-Based Invalidation

### 3.1 The problem

A user update might invalidate:
- `user:42` (the user themselves)
- `users:list` (any list containing this user)
- `users:42:posts` (if posts are included)
- `users:42:followers` (if this is a social graph)

Invalidating all of these on every write is complex. The simpler approach: tag the cache entries with a tag, and invalidate by tag.

### 3.2 The implementation

```python
import redis.asyncio as redis


class TaggedCache:
    """A cache that supports tag-based invalidation."""
    def __init__(self, redis_client: redis.Redis, namespace: str = "myapp"):
        self.redis = redis_client
        self.namespace = namespace

    async def set(self, key: str, value: str, ttl: int, tags: list[str] = None):
        """Set a value with optional tags."""
        namespaced_key = f"{self.namespace}:{key}"
        await self.redis.set(namespaced_key, value, ex=ttl)
        if tags:
            for tag in tags:
                tag_key = f"{self.namespace}:tag:{tag}"
                await self.redis.sadd(tag_key, namespaced_key)
                # Tag set expires after TTL + 60 seconds
                await self.redis.expire(tag_key, ttl + 60)

    async def get(self, key: str) -> str | None:
        return await self.redis.get(f"{self.namespace}:{key}")

    async def invalidate_tag(self, tag: str):
        """Invalidate all cache entries with the given tag."""
        tag_key = f"{self.namespace}:tag:{tag}"
        members = await self.redis.smembers(tag_key)
        if members:
            await self.redis.delete(*members)
        await self.redis.delete(tag_key)
```

### 3.3 The user-update case

```python
cache = TaggedCache(redis)


async def get_user_with_lists(user_id: int) -> dict:
    cache_key = f"user:{user_id}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)
    user = await db.get(User, user_id)
    posts = await db.execute(select(Post).where(Post.author_id == user_id))
    data = {
        "user": {"id": user.id, "name": user.name, "email": user.email},
        "posts": [{"id": p.id, "title": p.title} for p in posts.scalars()],
    }
    # Tag the cache with user-specific tags
    await cache.set(
        cache_key,
        json.dumps(data),
        ttl=300,
        tags=[f"user:{user_id}", "user-list"],  # invalidated on user update or any list update
    )
    return data


async def update_user(user_id: int, data: UserUpdate) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # Invalidate by tag
    await cache.invalidate_tag(f"user:{user_id}")
    return user


async def create_user(user_data: UserCreate) -> User:
    user = await db.create(**user_data.model_dump())
    await db.commit()
    # All user lists are now stale
    await cache.invalidate_tag("user-list")
    return user
```

The pattern: every cache write tags the entry with one or more tags. Every write invalidates by tag. The list tag is invalidated on any user create/update/delete; the user tag is invalidated on any user-specific update.

### 3.4 The Redis Streams alternative

For very high-throughput services, Redis Streams can replace the tagged-cache pattern:

```python
# Set a value; emit an event
await cache.set(key, value, ttl=300)
await redis.xadd("cache-events", {"event": "set", "key": key, "tags": "user:42"})

# A consumer subscribes to the stream and... actually, this is complex
```

The Stream-based pattern is overkill for most services. The simple tagged-cache is enough.

---

## 4. The Pattern: Cache Stampede Prevention

### 4.1 The thundering herd

The cache expires at time `T`. At `T+1`, 1000 requests hit the cache, all miss, and all 1000 hit the database at once. The database is overwhelmed.

The solution: a **lock** that the first request acquires; subsequent requests wait for the lock holder to populate the cache.

### 4.2 The lock pattern (Redis SETNX)

```python
import secrets
import asyncio


async def get_user_with_lock(
    user_id: int, redis: redis.Redis, db: AsyncSession, lock_ttl: int = 10,
) -> User | None:
    cache_key = f"user:{user_id}"
    cached = await redis.get(cache_key)
    if cached:
        return UserOut.model_validate_json(cached)
    # Try to acquire the lock
    lock_key = f"lock:{cache_key}"
    token = secrets.token_urlsafe(16)
    acquired = await redis.set(lock_key, token, ex=lock_ttl, nx=True)
    if acquired:
        try:
            # We're the first; do the work
            user = await db.get(User, user_id)
            if user is None:
                return None
            data = UserOut.model_dump_json(user)
            await redis.set(cache_key, data, ex=300)
            return data
        finally:
            # Release the lock if we still own it
            if await redis.get(lock_key) == token:
                await redis.delete(lock_key)
    else:
        # Someone else has the lock; wait for them
        for _ in range(50):  # 5 seconds max
            await asyncio.sleep(0.1)
            cached = await redis.get(cache_key)
            if cached:
                return UserOut.model_validate_json(cached)
        # Fall through; do the work anyway
        user = await db.get(User, user_id)
        return UserOut.model_dump_json(user) if user else None
```

The pattern:
1. The first request acquires the lock and does the work.
2. Subsequent requests see the lock is held; they wait.
3. The lock holder populates the cache; the waiting requests read from the cache.
4. The lock has a TTL (10s) so a crashed lock holder doesn't block forever.

### 4.3 Probabilistic Early Expiration (XFetch)

A simpler alternative: each request has a small chance of refreshing the cache early. The probability increases as the TTL approaches.

```python
import math
import random


async def get_user_with_xfetch(
    user_id: int, redis: redis.Redis, db: AsyncSession, beta: float = 1.0,
) -> User | None:
    cache_key = f"user:{user_id}"
    meta_key = f"user:{user_id}:meta"
    cached = await redis.get(cache_key)
    meta_raw = await redis.get(meta_key) if cached else None
    if cached and meta_raw:
        meta = json.loads(meta_raw)
        # XFetch probability
        age = time.time() - meta.get("ts", 0)
        ttl = meta.get("ttl", 300)
        delta = age / ttl
        if delta < 1.0 and random.random() < delta * beta * 0.1:
            # Trigger an early refresh in the background
            asyncio.create_task(_refresh_user_cache(user_id, redis, db))
        return UserOut.model_validate_json(cached)
    # Cache miss
    return await _refresh_user_cache(user_id, redis, db)


async def _refresh_user_cache(user_id: int, redis: redis.Redis, db: AsyncSession) -> User | None:
    user = await db.get(User, user_id)
    if user is None:
        return None
    data = UserOut.model_dump_json(user)
    await redis.set(f"user:{user_id}", data, ex=300)
    await redis.set(f"user:{user_id}:meta", json.dumps({"ts": time.time(), "ttl": 300}), ex=300)
    return data
```

The XFetch pattern: as the cache gets older, more requests have a chance of triggering a refresh. By the time the cache expires, someone has already refreshed it.

### 4.4 Background refresh with job queue

```python
async def get_user_with_background_refresh(
    user_id: int, redis: redis.Redis, arq: ArqRedis, db: AsyncSession,
) -> User | None:
    cache_key = f"user:{user_id}"
    cached = await redis.get(cache_key)
    if cached:
        # Check if the cache is close to expiring
        ttl = await redis.ttl(cache_key)
        if ttl < 60:  # Less than a minute left
            # Trigger a background refresh
            await arq.enqueue_job("refresh_user_cache", user_id=user_id)
        return UserOut.model_validate_json(cached)
    # Cache miss
    user = await db.get(User, user_id)
    if user is None:
        return None
    data = UserOut.model_dump_json(user)
    await redis.set(cache_key, data, ex=300)
    return data
```

The pattern uses [[../40 - Background Jobs and Workers for FastAPI/00 - Welcome|ARQ (or Celery)]] for the background job. The user gets the cached value immediately; the next request gets a fresh value.

---

## 5. The Pattern: Write-Write Race

### 5.1 The race

Two clients update the same user at the same time. The cache might be in an inconsistent state:

```
T0: Client A reads user (version 1, email="alice@old.com")
T1: Client B reads user (version 1, email="alice@old.com")
T2: Client A updates user (version 2, email="alice@new.com") and cache → cache has version 2
T3: Client B updates user (version 2, email="bob@old.com") and cache → cache has version 1 of bob's email with version 2 timestamp
```

The race condition: Client B's update was based on stale data. The cache reflects a state that never existed in the database.

### 5.2 The fix: optimistic locking

```python
async def update_user_with_version(
    user_id: int, data: UserUpdate, expected_version: int, redis: redis.Redis, db: AsyncSession,
) -> User:
    # Optimistic locking: only update if the version matches
    result = await db.execute(
        update(User)
        .where(User.id == user_id, User.version == expected_version)
        .values(**data.model_dump(exclude_unset=True), version=User.version + 1)
    )
    if result.rowcount == 0:
        raise ConflictProblem(detail="Concurrent update detected; please refresh and retry")
    await db.commit()
    # Invalidate the cache
    await cache.delete(f"user:{user_id}")
    return await db.get(User, user_id)
```

The client must send the expected version (e.g., as a `If-Match` header). The server only updates if the version matches. A mismatch returns 409 Conflict; the client refreshes and retries.

### 5.3 The HTTP pattern

The HTTP standard for optimistic concurrency control is `If-Match`:

```http
GET /v2/users/12345 HTTP/1.1
ETag: "v1"

HTTP/1.1 200 OK
ETag: "v2"
{"id": 12345, "email": "alice@example.com", ...}

# Client updates
PUT /v2/users/12345 HTTP/1.1
If-Match: "v2"
Content-Type: application/json
{"email": "alice@new.com"}

HTTP/1.1 200 OK
ETag: "v3"
{"id": 12345, "email": "alice@new.com", ...}

# Conflict: another client updated in the meantime
PUT /v2/users/12345 HTTP/1.1
If-Match: "v2"
{"email": "alice@another.com"}

HTTP/1.1 412 Precondition Failed
```

The `If-Match` header carries the expected version. A 412 means "your version is stale". The client refreshes and retries.

### 5.4 The implementation in FastAPI

```python
from fastapi import Header, HTTPException


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    data: UserUpdate,
    if_match: str | None = Header(None, alias="If-Match"),
    uow: UnitOfWork = Depends(get_uow),
):
    expected_version = if_match.strip('"') if if_match else None
    try:
        user = await uow.users.update_with_version(
            user_id=user_id, data=data, expected_version=expected_version,
        )
    except ConcurrentUpdateError:
        raise ConflictProblem("Concurrent update detected")
    await uow.commit()
    return user
```

The handler extracts the `If-Match` header, strips the quotes, and passes it to the repository. The repository uses it in the WHERE clause of the UPDATE.

---

## 6. The Four Common Invalidation Mistakes

### 6.1 Invalidation forgotten

```python
# ❌ Update the user but forget to invalidate the cache
async def update_user(user_id: int, data: UserUpdate) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    return user
    # The cache still has the old user data!
```

The fix: a code-review checklist, or a centralized write path that always invalidates.

### 6.2 Invalidation that doesn't catch all the keys

```python
# ❌ Update the user, invalidate the user cache, but forget the user's list cache
await cache.delete(f"user:{user_id}")
# The next "list users" request returns a list with the old user data
```

The fix: tag-based invalidation (covered above) catches all related keys.

### 6.3 The expired-while-fetching race

```
T0: Request A reads cache; cache is stale (or expired); A starts DB query
T1: The cache is updated by another process
T2: A finishes the DB query and overwrites the cache with the stale value
```

The fix: write-through with version checking, or a short TTL on the cache (so the stale write expires soon).

### 6.4 The double-fetch race

```
T0: Request A reads cache; cache expires; A starts DB query
T1: Request B reads cache; cache miss; B starts DB query
T2: A and B both write to the cache
```

The fix: the lock pattern (covered above). The first request acquires the lock; subsequent requests wait.

---

## 7. The Production Invalidation Strategy

For most services, the right strategy is:

1. **TTL on every cache entry**: 5-15 minutes for user data, 1-5 minutes for hot data, 1 hour+ for static data.
2. **Invalidation on writes**: delete the cache key after every successful write.
3. **Tag-based invalidation for related keys**: invalidate the user and any list the user is in.
4. **Lock-based stampede prevention**: prevent the thundering herd when the cache expires.
5. **Optimistic locking for high-stakes updates**: use `If-Match` for any update that has business consequences (billing, profile, etc.).

```python
# The default pattern
async def update_user(user_id: int, data: UserUpdate) -> User:
    # 1) Optimistic locking (high-stakes)
    expected_version = data.version
    user = await db.execute(
        update(User)
        .where(User.id == user_id, User.version == expected_version)
        .values(**data.model_dump(exclude={"version"}), version=User.version + 1)
    )
    if user.rowcount == 0:
        raise ConflictProblem("Concurrent update")
    await db.commit()
    # 2) Tag-based invalidation
    await cache.invalidate_tag(f"user:{user_id}")
    # 3) The TTL is the safety net
    return await db.get(User, user_id)
```

The combination: optimistic locking for correctness, tag-based invalidation for consistency, TTL for safety. The lock pattern is for read-heavy endpoints with high cache miss rates.

---

## 8. Monitoring the Invalidation

### 8.1 The metrics

```python
cache_invalidations = Counter(
    "cache_invalidations_total",
    "Cache invalidations by tag",
    ["tag"],
)
cache_stale_hits = Counter(
    "cache_stale_hits_total",
    "Cache hits that returned stale data",
    ["key_prefix"],
)
```

`cache_stale_hits` is the most important: it tracks how often a cache hit returned a value that was older than the TTL. A high number suggests the TTL is too long or the invalidation is broken.

### 8.2 The dashboard

- **Cache hit rate**: should be > 90% for hot data.
- **Stale hit rate**: should be < 1% (the TTL should catch most staleness).
- **Invalidation rate**: the rate of `cache.delete` calls. A spike might indicate a buggy write path.
- **Lock contention**: how often requests wait for a cache lock. A high rate suggests the lock holder is slow.

### 8.3 The alerts

```yaml
# Cache hit rate below 50% for 5 minutes
- alert: CacheHitRateLow
  expr: rate(cache_hits_total[5m]) / rate(cache_requests_total[5m]) < 0.5
  for: 5m
  annotations:
    summary: "Cache hit rate below 50%"

# Stale hit rate above 5%
- alert: StaleHitsHigh
  expr: rate(cache_stale_hits_total[5m]) > 0.05
  for: 5m
  annotations:
    summary: "Stale cache hits are high"
```

Alerts catch problems before customers do. A high stale hit rate is a "your data is wrong" alert; a high invalidation rate might indicate a buggy write path.

---

## 9. The Test Patterns

### 9.1 Test the TTL

```python
@pytest.mark.asyncio
async def test_ttl(client, redis):
    # Populate the cache
    await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    # Verify the TTL is set
    ttl = await redis.ttl("user:1:...")
    assert 0 < ttl <= 300
```

### 9.2 Test the invalidation

```python
@pytest.mark.asyncio
async def test_invalidation_on_update(client, redis):
    # Populate the cache
    await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    assert await redis.get("user:1:...") is not None
    # Update the user
    await client.put(
        "/v2/users/1",
        json={"name": "Alice Updated"},
        headers={"Authorization": "Bearer ..."},
    )
    # The cache is invalidated
    assert await redis.get("user:1:...") is None
```

### 9.3 Test the lock-based stampede prevention

```python
@pytest.mark.asyncio
async def test_lock_prevents_stampede(client, redis, db_with_user):
    # Simulate 100 concurrent requests when the cache is empty
    tasks = [
        client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
        for _ in range(100)
    ]
    responses = await asyncio.gather(*tasks)
    # All return 200
    assert all(r.status_code == 200 for r in responses)
    # The DB was hit only once (or a small number of times)
    # Verify by checking the DB query counter
    assert db_with_user.query_count < 5  # or whatever threshold
```

The exact assertion depends on the implementation. The point: concurrent requests don't all hit the database.

### 9.4 Test the optimistic locking

```python
@pytest.mark.asyncio
async def test_concurrent_update_returns_409(client):
    # Two clients read the user with the same version
    response1 = await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    etag1 = response1.headers["ETag"]
    response2 = await client.get("/v2/users/1", headers={"Authorization": "Bearer ..."})
    etag2 = response2.headers["ETag"]
    assert etag1 == etag2  # Same version
    # Client 1 updates
    response = await client.put(
        "/v2/users/1",
        json={"name": "Alice 1"},
        headers={"Authorization": "Bearer ...", "If-Match": etag1},
    )
    assert response.status_code == 200
    # Client 2 tries to update with the same version
    response = await client.put(
        "/v2/users/1",
        json={"name": "Alice 2"},
        headers={"Authorization": "Bearer ...", "If-Match": etag2},
    )
    assert response.status_code == 409  # Conflict
```

The test exercises the optimistic locking flow. The second client sees a 409 because the version is stale.

---

## 10. The Production Checklist

| Concern | Where | Verified by |
|---------|-------|-------------|
| TTL on every entry | `cache.set(..., ex=ttl)` | Test `redis.ttl()` |
| Invalidation on writes | `cache.delete(...)` after commit | Test invalidation |
| Tag-based invalidation | `cache.invalidate_tag(...)` | Test list invalidation |
| Lock-based stampede prevention | Lock pattern in `get_*` | Test concurrent requests |
| Optimistic locking | `If-Match` header | Test 409 on stale version |
| Cache failure fallback | try/except around cache | Test Redis down |
| Monitoring | Prometheus metrics | Dashboard + alerts |
| Stale hit detection | TTL check in deserializer | Track in metrics |

---

## 11. Código de Compresión

```python
"""
Compresión: Invalidation Patterns
Covers: TTL + event-based invalidation, tag-based invalidation,
       lock-based stampede prevention, optimistic locking.
"""
import asyncio
import json
import secrets
import time
from typing import Any


# 1) Tagged cache
class TaggedCache:
    def __init__(self, redis, namespace: str = "myapp"):
        self.redis = redis
        self.namespace = namespace

    async def set(self, key: str, value: str, ttl: int, tags: list[str] = None):
        namespaced_key = f"{self.namespace}:{key}"
        await self.redis.set(namespaced_key, value, ex=ttl)
        if tags:
            for tag in tags:
                tag_key = f"{self.namespace}:tag:{tag}"
                await self.redis.sadd(tag_key, namespaced_key)
                await self.redis.expire(tag_key, ttl + 60)

    async def get(self, key: str) -> str | None:
        return await self.redis.get(f"{self.namespace}:{key}")

    async def invalidate_tag(self, tag: str):
        tag_key = f"{self.namespace}:tag:{tag}"
        members = await self.redis.smembers(tag_key)
        if members:
            await self.redis.delete(*members)
        await self.redis.delete(tag_key)


# 2) Lock-based stampede prevention
async def get_user_with_lock(cache_key, redis, db_session, ttl: int = 300, lock_ttl: int = 10):
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    lock_key = f"lock:{cache_key}"
    token = secrets.token_urlsafe(16)
    acquired = await redis.set(lock_key, token, ex=lock_ttl, nx=True)
    if acquired:
        try:
            user = await db_session.execute(select(User).where(User.id == 1))
            user_data = user.scalar_one_or_none()
            if user_data is None:
                return None
            data = {"id": user_data.id, "name": user_data.name, "email": user_data.email}
            await redis.set(cache_key, json.dumps(data), ex=ttl)
            return data
        finally:
            if await redis.get(lock_key) == token:
                await redis.delete(lock_key)
    else:
        for _ in range(50):
            await asyncio.sleep(0.1)
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        return None  # Fallback; in production, hit the DB


# 3) XFetch (probabilistic early refresh)
async def get_user_with_xfetch(cache_key, meta_key, redis, db_session, beta: float = 1.0, ttl: int = 300):
    import math
    import random
    cached = await redis.get(cache_key)
    meta_raw = await redis.get(meta_key) if cached else None
    if cached and meta_raw:
        meta = json.loads(meta_raw)
        age = time.time() - meta.get("ts", 0)
        delta = age / meta.get("ttl", ttl)
        if random.random() < delta * beta * 0.1:
            asyncio.create_task(_refresh(cache_key, meta_key, redis, db_session, ttl))
        return json.loads(cached)
    return await _refresh(cache_key, meta_key, redis, db_session, ttl)


async def _refresh(cache_key, meta_key, redis, db_session, ttl):
    user = await db_session.execute(select(User).where(User.id == 1))
    user_data = user.scalar_one_or_none()
    if user_data is None:
        return None
    data = {"id": user_data.id, "name": user_data.name, "email": user_data.email}
    await redis.set(cache_key, json.dumps(data), ex=ttl)
    await redis.set(meta_key, json.dumps({"ts": time.time(), "ttl": ttl}), ex=ttl)
    return data


# 4) Optimistic locking
async def update_user_with_version(user_id, data, redis, db_session):
    from sqlalchemy import update
    result = await db_session.execute(
        update(User)
        .where(User.id == user_id, User.version == data.version)
        .values(**{k: v for k, v in data.model_dump().items() if k != "version"},
                version=User.version + 1)
    )
    if result.rowcount == 0:
        raise Exception("Concurrent update detected")
    await db_session.commit()
    # Invalidate
    await redis.delete(f"user:{user_id}")
    return await db_session.get(User, user_id)
```

---

## Key Takeaways

- **TTL + event-based invalidation** is the right default for most services. The TTL is the safety net; the invalidation is the immediate update.
- **Tag-based invalidation** catches related keys (a user and the lists they're in). The pattern: every cache write tags the entry; every write invalidates by tag.
- **The thundering herd** is the biggest production issue. Fix it with locks (the first request acquires; subsequent requests wait), XFetch (probabilistic early refresh), or background refreshes (when the cache is close to expiring).
- **Optimistic locking** prevents the write-write race. Use `If-Match` headers; a stale version returns 409 Conflict.
- **The cache failure fallback** is critical. A Redis outage should not break the service. Log the failure and fall through to the handler.
- **Monitor cache hit rate, stale hit rate, invalidation rate, and lock contention**. A high stale hit rate is a "your data is wrong" alert.
- **The expired-while-fetching race** is a subtle bug: a cache miss, a slow DB query, and a cache write that overwrites a fresher value. Mitigate with short TTLs or version checking.
- **The double-fetch race** is a different subtle bug: two concurrent requests, both miss, both query, both write. Mitigate with locks.

## References

- [Optimal Probabilistic Cache Stampede Prevention (Vattani et al., 2015)](https://www.vldb.org/pvldb/vol8/p886-vattani.pdf)
- [Redis — Tag-based Invalidation](https://redis.io/docs/latest/operate/oss_and_stack/management/config-file/)
- [RFC 9110 — HTTP Semantics (If-Match, ETag)](https://www.rfc-editor.org/rfc/rfc9110)
- [Caching at Reddit](https://reddit.engineering/caching-at-reddit-the-software/)
- [Caching Best Practices (Dropbox)](https://dropbox.tech/infrastructure/caching-in-theory-and-practice)
- [Optimistic Concurrency Control (Microsoft)](https://learn.microsoft.com/en-us/ef/core/saving/concurrency)
- [Cache Invalidation Patterns (AWS)](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)
