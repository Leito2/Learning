# 🔴 Redis as Cache Backend

## 🎯 Learning Objectives

- Use `aiocache` and Redis as a server-side cache in FastAPI
- Implement the cache-aside pattern with proper invalidation
- Handle the thundering herd problem with locks and stale-while-revalidate
- Avoid the four most common Redis cache mistakes

## Introduction

HTTP caching handles the "client re-fetches the same data" case. For the more common case — your application re-queries the same data from the database on every request — you need a server-side cache. Redis is the standard: it's fast (microsecond latency), supports rich data structures, and is well-understood.

The cache-aside pattern is the simplest and most common: read from the cache, fall through to the database on miss, write to the cache after a hit. The hard parts are invalidation (when to delete the cache) and the thundering herd (what happens when the cache expires and 1000 requests hit the database at once).

This note covers the patterns: cache-aside, write-through, write-behind, and the two anti-patterns to avoid (stale data, cache stampede). The capstone ties them together in a real FastAPI service.

---

## 1. The Cache-Aside Pattern

### 1.1 The flow

```python
async def get_user(user_id: int, redis: Redis, db: AsyncSession) -> User | None:
    cache_key = f"user:{user_id}"
    # 1) Try the cache
    cached = await redis.get(cache_key)
    if cached:
        return User.model_validate_json(cached)
    # 2) Cache miss: hit the database
    user = await db.get(User, user_id)
    if not user:
        return None
    # 3) Populate the cache
    await redis.set(cache_key, user.model_dump_json(), ex=300)  # 5 min TTL
    return user
```

The pattern is simple: read cache, fall through to DB, populate cache. The TTL prevents stale data from living forever.

### 1.2 The TTL

The TTL (time-to-live) is the maximum age of a cached item. Common values:

| Data type | TTL |
|-----------|-----|
| User profile | 5-15 minutes |
| Project metadata | 5-15 minutes |
| Search results | 1-5 minutes |
| Hot data (trending posts) | 1 minute |
| Rarely-changing data (countries, categories) | 1 hour+ |
| Critical (auth tokens) | Never (no cache) |

The right TTL balances freshness vs. cache hit rate. A 5-minute TTL on a user profile means: if a user's email changes, it takes up to 5 minutes to propagate. That's usually acceptable; users don't notice 5-minute delays on email changes.

### 1.3 When to use cache-aside

- The data is read more than written.
- The data is expensive to compute or fetch.
- Stale data is acceptable (within the TTL window).
- The data is not safety-critical (auth, billing).

For write-heavy data or zero-staleness requirements, use a different pattern.

---

## 2. `aiocache`: The Standard Library

### 2.1 Setup

```bash
pip install aiocache[redis]
```

### 2.2 Configuration

```python
# app/core/cache.py
from aiocache import Cache
from aiocache.backends.redis import RedisCache

cache = Cache(
    Cache.REDIS,
    endpoint="redis://localhost:6379",
    namespace="myapp",
    serializer=JsonSerializer(),  # Pydantic-friendly
)
```

The `namespace` is a prefix for all keys: `myapp:user:123` instead of just `user:123`. Useful for multi-environment (dev, staging, prod) sharing a Redis instance.

### 2.3 Basic usage

```python
from aiocache import cached

# As a decorator
@cached(ttl=300, key="user:{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    return user


# As a context manager
async with cache:
    await cache.set("user:123", user_data, ttl=300)
    value = await cache.get("user:123")


# Directly
await cache.set("user:123", user_data, ttl=300)
value = await cache.get("user:123")
```

`aiocache` provides both a decorator (clean, hides the cache logic) and direct methods (more control).

### 2.4 The key function

The `key` argument supports placeholders from the function arguments:

```python
@cached(ttl=300, key="user:{user_id}")
async def get_user(user_id: int): ...

@cached(ttl=60, key="project:{project_id}:users")
async def get_project_users(project_id: int): ...
```

The `{}` placeholders are filled with the function's argument values. The function name is implicitly part of the key (so two functions with the same `key=` string don't collide).

### 2.5 The serializer

`aiocache` ships with `JsonSerializer` and `PickleSerializer`. Use `JsonSerializer` for Pydantic models; the serialization is portable and debuggable:

```python
from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    email: str


await cache.set("user:1", UserOut(id=1, email="alice@example.com").model_dump_json(), ttl=300)
# Stored as a string. Retrieve as a string; parse with UserOut.model_validate_json.
```

Avoid `PickleSerializer` for untrusted data — pickle allows arbitrary code execution.

---

## 3. The Pattern: Cache-Aside with Invalidation

### 3.1 Read-through with TTL

```python
async def get_user(user_id: int, redis: Redis, db: AsyncSession) -> User | None:
    cache_key = f"user:{user_id}"
    cached = await redis.get(cache_key)
    if cached:
        return UserOut.model_validate_json(cached)
    user = await db.get(User, user_id)
    if user is None:
        return None
    await redis.set(cache_key, user.model_dump_json(), ex=300)
    return user
```

The TTL is the safety net: if the cache is never invalidated, the data is at most TTL seconds old.

### 3.2 Write-through

```python
async def update_user(user_id: int, data: UserUpdate, redis: Redis, db: AsyncSession) -> User:
    # 1) Update the database
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # 2) Update the cache
    cache_key = f"user:{user_id}"
    await redis.set(cache_key, user.model_dump_json(), ex=300)
    return user
```

Write-through updates the cache on every write. The cache and DB are always consistent (modulo the write failing — see below).

### 3.3 Invalidation on write

```python
async def update_user(user_id: int, data: UserUpdate, redis: Redis, db: AsyncSession) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # Invalidate the cache
    await redis.delete(f"user:{user_id}")
    return user
```

The "lazy" approach: just delete the cache. The next read repopulates it. Simpler than write-through; the next reader pays the cost.

### 3.4 When to use each

| Pattern | When |
|---------|------|
| Cache-aside + TTL | Read-heavy, freshness not critical |
| Write-through | Read-heavy, freshness critical |
| Invalidation (delete) | Write-heavy, freshness critical |

Most production services use a mix: cache-aside for reads, write-through or invalidation for writes. The choice per endpoint is a trade-off between consistency and performance.

---

## 4. The Thundering Herd

### 4.1 The problem

The cache expires at time `T`. At `T+1`, 1000 requests hit the cache, all miss, and all 1000 hit the database at once. The database is overwhelmed.

```
Time 0:   1 request, cache miss, DB query, cache populated
Time 5:   1 request, cache hit
Time 300: cache expires
Time 301: 1000 requests, all miss, all hit DB
```

The "thundering herd" or "cache stampede". The database is now doing 1000x the work it was doing at time 0.

### 4.2 The fix: locks

The first request to miss the cache acquires a lock; subsequent requests wait or return the stale value:

```python
import asyncio
from contextlib import asynccontextmanager


class CacheLock:
    def __init__(self, redis: Redis, ttl: int = 10):
        self.redis = redis
        self.ttl = ttl

    @asynccontextmanager
    async def acquire(self, key: str):
        """Acquire a lock; release on exit."""
        token = secrets.token_urlsafe(16)
        lock_key = f"lock:{key}"
        acquired = await self.redis.set(lock_key, token, ex=self.ttl, nx=True)
        try:
            yield acquired
        finally:
            # Release only if we still own the lock
            current = await self.redis.get(lock_key)
            if current == token:
                await self.redis.delete(lock_key)


async def get_user_with_lock(user_id: int, redis: Redis, db: AsyncSession, lock: CacheLock) -> User:
    cache_key = f"user:{user_id}"
    cached = await redis.get(cache_key)
    if cached:
        return UserOut.model_validate_json(cached)
    # Cache miss; try to acquire the lock
    async with lock.acquire(cache_key) as acquired:
        if acquired:
            # We're the first; do the work
            user = await db.get(User, user_id)
            if user is None:
                return None
            await redis.set(cache_key, user.model_dump_json(), ex=300)
            return user
        else:
            # Someone else has the lock; wait for them
            for _ in range(50):  # max 5 seconds
                await asyncio.sleep(0.1)
                cached = await redis.get(cache_key)
                if cached:
                    return UserOut.model_validate_json(cached)
            # Fall through; do the work anyway
            user = await db.get(User, user_id)
            return user
```

The pattern: the first request acquires the lock and does the work; subsequent requests wait for the cache to be populated. After the lock TTL (10 seconds), the lock is released even if the first request is still working (defense against the lock holder crashing).

### 4.3 The fix: stale-while-revalidate

```python
async def get_user_with_swr(user_id: int, redis: Redis, db: AsyncSession) -> User:
    cache_key = f"user:{user_id}"
    meta_key = f"user:{user_id}:meta"
    cached = await redis.get(cache_key)
    if cached:
        # Check if the cache is fresh or stale
        meta = await redis.get(meta_key)
        if meta and meta.get("state") == "stale":
            # Trigger a background refresh (don't block this request)
            asyncio.create_task(refresh_user_cache(user_id, redis, db))
        return UserOut.model_validate_json(cached)
    # Cache miss; do the work synchronously
    return await refresh_user_cache(user_id, redis, db)


async def refresh_user_cache(user_id: int, redis: Redis, db: AsyncSession) -> User:
    user = await db.get(User, user_id)
    if user is None:
        return None
    await redis.set(f"user:{user_id}", user.model_dump_json(), ex=300)
    await redis.set(f"user:{user_id}:meta", json.dumps({"state": "fresh", "ts": time.time()}), ex=300)
    return user
```

The pattern: the cache has a `state` field (`fresh` or `stale`). When stale, the request returns the cached value immediately and triggers a background refresh. The next request gets the fresh value.

The complexity is the same as the lock approach, but the user never waits.

### 4.4 The fix: background refresh

```python
async def get_user_with_background_refresh(user_id: int, redis: Redis, db: AsyncSession, arq: ArqRedis) -> User:
    cache_key = f"user:{user_id}"
    cached = await redis.get(cache_key)
    if cached:
        # Check the timestamp; if close to expiry, refresh in background
        ttl = await redis.ttl(cache_key)
        if ttl < 60:  # Less than a minute left
            await arq.enqueue_job("refresh_user_cache", user_id=user_id)
        return UserOut.model_validate_json(cached)
    # Cache miss
    return await refresh_user_cache(user_id, redis, db)
```

When the cache is close to expiring, trigger a background job to refresh it. By the time the cache actually expires, the background job has repopulated it. The user never sees a cache miss.

The background refresh pattern uses [[../40 - Background Jobs and Workers for FastAPI/00 - Welcome|ARQ (or Celery)]] for the job queue. The pattern combines caching with background jobs.

---

## 5. The Pattern: Multi-Tenant Cache Keys

### 5.1 The naming convention

A multi-tenant cache key includes the tenant:

```python
def cache_key(tenant_id: int, resource: str, resource_id: int | None = None) -> str:
    if resource_id is None:
        return f"t:{tenant_id}:{resource}"
    return f"t:{tenant_id}:{resource}:{resource_id}"


# Examples
cache_key(1, "users")  # "t:1:users"
cache_key(1, "users", 42)  # "t:1:users:42"
```

The `t:<tenant_id>:` prefix ensures tenant isolation at the cache layer. Tenant 1's `users:42` is a different key from tenant 2's `users:42`.

### 5.2 The cross-tenant cache hit risk

Without tenant-prefixed keys:

```python
# ❌ Tenant 1 and tenant 2 share the same key
await redis.set("users:42", user_1_data)
# Later
user_2 = await redis.get("users:42")  # gets tenant 1's data!
```

The fix: always include the tenant ID in the key. The pattern above is mandatory for any multi-tenant service.

### 5.3 The cache for tenant list

The list of tenants is itself a cache key. Use a constant for the global cache:

```python
GLOBAL_TENANTS_KEY = "t:0:tenants"  # tenant 0 = global
```

---

## 6. The Pattern: Cache Stampede Prevention with Probabilistic Early Expiration

### 6.1 The algorithm

A probabilistic approach: each request, when reading the cache, has a small chance of triggering an early refresh. The probability increases as the TTL approaches.

```python
import math
import random


async def get_user_with_xfetch(
    user_id: int, redis: Redis, db: AsyncSession, arq: ArqRedis,
    beta: float = 1.0,  # tuning parameter
) -> User:
    cache_key = f"user:{user_id}"
    meta_key = f"user:{user_id}:meta"
    cached = await redis.get(cache_key)
    meta = json.loads(await redis.get(meta_key) or "{}") if cached else {}
    
    if cached and meta:
        # Compute the probability of early refresh
        delta = (time.time() - meta.get("ts", 0)) / meta.get("ttl", 300)
        # XFetch algorithm: probability increases as delta approaches 1
        xfetch_prob = delta * beta * math.log(random.random())
        if xfetch_prob > 0:  # or some threshold
            # Trigger a background refresh
            await arq.enqueue_job("refresh_user_cache", user_id=user_id)
        return UserOut.model_validate_json(cached)
    # Cache miss
    return await refresh_user_cache(user_id, redis, db)
```

The XFetch algorithm is described in the paper "Optimal Probabilistic Cache Stampede Prevention" by Vattani, Chierichetti, and Lowenstein (2015). The probability of an early refresh is `delta * beta * log(rand())`, where `delta` is the fraction of TTL elapsed and `beta` is a tuning parameter.

The intuition: as the cache gets older, more and more requests have a chance of refreshing it. By the time the cache expires, the probability is high enough that someone has already refreshed it.

### 6.2 When to use XFetch

XFetch is the right tool when:
- The cache has a fixed TTL.
- The DB load is the bottleneck.
- The data is read frequently.
- Stale data is acceptable within the TTL.

For most services, the simple lock or stale-while-revalidate is enough. XFetch is the production-grade version that scales to thousands of requests per second.

---

## 7. The Four Common Redis Cache Mistakes

### 7.1 No TTL

```python
# ❌ The cache entry lives forever
await redis.set(f"user:{user_id}", user_data)
# 6 months later, the data is stale and still in the cache
```

Every cache entry must have a TTL. Even a long one (24 hours) is better than infinite.

### 7.2 Cache key collisions

```python
# ❌ Two different resources with the same key
await redis.set("user:1", user_1_data)  # tenant 1's user 1
await redis.set("user:1", user_2_data)  # overwrites with tenant 2's user 1

# ✅ Tenant-prefixed key
await redis.set(f"t:1:user:1", user_1_data)
```

The fix is always a unique prefix per resource, per tenant, per environment.

### 7.3 Storing too much

```python
# ❌ Storing the entire database row
user = await db.execute(select(User).options(selectinload(User.posts, User.comments)))
await redis.set(f"user:{user_id}", user.model_dump_json(), ex=300)
# A user with 1000 posts is now a 5MB JSON blob in Redis
```

Cache small, focused objects. A user with their posts is a different cache key than a user without:

```python
# Cache the basic user info
await redis.set(f"user:{user_id}", user_basic_data, ex=300)
# Cache the user's posts separately
await redis.set(f"user:{user_id}:posts", posts_data, ex=60)
```

### 7.4 No monitoring

```python
# ❌ No way to know if the cache is helping
await redis.set(f"user:{user_id}", user_data, ex=300)
```

You need metrics: cache hits, cache misses, evictions, average TTL. Most Redis clients expose these; `redis-cli info stats` shows them at the Redis level. Export to Prometheus for dashboarding.

```python
cache_hits = Counter("cache_hits_total", "Cache hits", ["key_prefix"])
cache_misses = Counter("cache_misses_total", "Cache misses", ["key_prefix"])


async def get_user_with_metrics(user_id: int, ...):
    cached = await redis.get(cache_key)
    if cached:
        cache_hits.labels(key_prefix="user").inc()
        return UserOut.model_validate_json(cached)
    cache_misses.labels(key_prefix="user").inc()
    # ...
```

A hit rate below 50% suggests the TTL is too short or the cache key is wrong. A hit rate above 95% is the goal for hot data.

---

## 8. The Pattern: Cache Invalidation on Events

### 8.1 The problem

A user updates their profile. The cached `user:42` is now stale. The TTL handles this within 5 minutes, but the user expects the change to be immediate.

### 8.2 The fix: invalidate on write

```python
async def update_user_profile(user_id: int, data: UserUpdate, redis: Redis, db: AsyncSession) -> User:
    user = await db.get(User, user_id)
    user.update(**data.model_dump(exclude_unset=True))
    await db.commit()
    # Invalidate the cache
    await redis.delete(f"user:{user_id}")
    # Also invalidate any related cache keys
    # e.g., a list of users might include this user
    # A more sophisticated approach: tag-based invalidation (Redis 7.4+)
    return user
```

The pattern: every write deletes the corresponding cache key. The next read repopulates.

### 8.3 The cascade problem

A user update might invalidate:
- `user:42` (the user themselves)
- `users:list` (any list containing this user)
- `users:42:posts` (if posts are included)
- `users:42:followers` (if this is a social graph)

Invalidating all of these on every write is complex. The simpler approach: invalidate the user, let the lists expire naturally via TTL.

### 8.4 The tag-based invalidation (Redis 7.4+)

```python
# Set with a tag
await redis.set("user:42", user_data, ex=300)
await redis.sadd("tag:user:42", "user:42")

# Invalidate by tag
await redis.delete("tag:user:42")  # The set is gone; the user key is orphaned
# A background job cleans up orphaned keys
```

Redis 7.4 added a feature called "client-side caching invalidation" that uses keyspace notifications. For most services, the simple "delete the key" approach is enough.

---

## 9. The Production Cache Configuration

### 9.1 The Redis instance

For a FastAPI service, Redis is the standard cache. The right instance type:

| Workload | Instance type | Memory |
|----------|---------------|--------|
| Light (small SaaS) | `cache.t3.micro` (AWS) | 1 GB |
| Medium (B2B SaaS) | `cache.t3.medium` or `cache.r6g.large` | 6-32 GB |
| Heavy (large B2B or consumer) | Cluster mode, 3+ nodes | 100+ GB |

The memory is sized for the working set: 10x the typical cached data is a safe margin.

### 9.2 The eviction policy

```bash
# maxmemory-policy: how Redis evicts keys when memory is full
# allkeys-lru: evict the least-recently-used key (good for caching)
# allkeys-lfu: evict the least-frequently-used key (better for many small items)
# volatile-lru: only evict keys with a TTL set
# noeviction: never evict; return an error (default, not good for caching)
```

For a cache, `allkeys-lru` or `allkeys-lfu` is the right choice. The cache will drop the least-used items first when memory is full.

### 9.3 The connection pool

```python
import redis.asyncio as redis


# Create a connection pool
pool = redis.ConnectionPool.from_url(
    "redis://localhost:6379",
    max_connections=50,
    decode_responses=True,
)
client = redis.Redis(connection_pool=pool)
```

The pool's size is the maximum concurrent Redis operations per process. With 50 connections and 4 FastAPI workers, the total Redis connections is 200. Make sure Redis's `maxclients` is at least 200 (default is 10000).

---

## 10. The Test Patterns

### 10.1 Test the cache hit

```python
@pytest.mark.asyncio
async def test_cache_hit_on_second_request(client, db_session, redis):
    # First request: cache miss, populates the cache
    response1 = await client.get("/v2/users/1")
    assert response1.status_code == 200
    # Second request: cache hit
    response2 = await client.get("/v2/users/1")
    assert response2.status_code == 200
    # The second response might be faster, but the test is that the cache was populated
    assert await redis.get("t:1:user:1") is not None
```

### 10.2 Test the invalidation

```python
@pytest.mark.asyncio
async def test_cache_invalidation_on_update(client, db_session, redis):
    # Populate the cache
    await client.get("/v2/users/1")
    assert await redis.get("t:1:user:1") is not None
    # Update the user
    response = await client.put(
        "/v2/users/1",
        json={"name": "Alice Updated"},
        headers={"Authorization": "Bearer ..."},
    )
    assert response.status_code == 200
    # The cache is invalidated
    assert await redis.get("t:1:user:1") is None
```

### 10.3 Test the TTL

```python
@pytest.mark.asyncio
async def test_cache_ttl(redis):
    await redis.set("test:key", "value", ex=10)
    assert await redis.get("test:key") == "value"
    # Wait 11 seconds; the cache is empty
    await asyncio.sleep(11)
    assert await redis.get("test:key") is None
```

These tests catch the most common mistakes at the unit level.

---

## 11. Código de Compresión

```python
"""
Compresión: Redis as Cache Backend
Covers: cache-aside, aiocache, lock-based stampede prevention,
       multi-tenant keys, invalidation.
"""
import asyncio
import json
import secrets
from contextlib import asynccontextmanager

import redis.asyncio as redis
from aiocache import Cache
from aiocache.serializers import JsonSerializer
from pydantic import BaseModel


# 1) Cache setup
cache = Cache(
    Cache.REDIS,
    endpoint="redis://localhost:6379",
    namespace="myapp",
    serializer=JsonSerializer(),
)


# 2) Cache-aside pattern
async def get_user_cached(user_id: int, tenant_id: int, db_session) -> dict | None:
    cache_key = f"t:{tenant_id}:user:{user_id}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached) if isinstance(cached, str) else cached
    user = await db_session.get(User, user_id)
    if user is None:
        return None
    data = {"id": user.id, "email": user.email, "name": user.name}
    await cache.set(cache_key, json.dumps(data), ttl=300)
    return data


# 3) Invalidation on write
async def update_user(user_id: int, data: dict, tenant_id: int, db_session) -> dict:
    user = await db_session.get(User, user_id)
    user.update(**data)
    await db_session.commit()
    await cache.delete(f"t:{tenant_id}:user:{user_id}")
    return {"id": user.id, ...}


# 4) Cache lock for stampede prevention
@asynccontextmanager
async def cache_lock(redis_client, key: str, ttl: int = 10):
    token = secrets.token_urlsafe(16)
    lock_key = f"lock:{key}"
    acquired = await redis_client.set(lock_key, token, ex=ttl, nx=True)
    try:
        yield acquired
    finally:
        if await redis_client.get(lock_key) == token:
            await redis_client.delete(lock_key)


async def get_user_with_lock(user_id: int, tenant_id: int, db_session, redis_client) -> dict:
    cache_key = f"t:{tenant_id}:user:{user_id}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached) if isinstance(cached, str) else cached
    async with cache_lock(redis_client, cache_key) as acquired:
        if not acquired:
            # Wait for the lock holder to populate the cache
            for _ in range(50):
                await asyncio.sleep(0.1)
                cached = await cache.get(cache_key)
                if cached:
                    return json.loads(cached) if isinstance(cached, str) else cached
        # We have the lock; do the work
        user = await db_session.get(User, user_id)
        if user is None:
            return None
        data = {"id": user.id, "email": user.email, "name": user.name}
        await cache.set(cache_key, json.dumps(data), ttl=300)
        return data
```

---

## Key Takeaways

- **Redis is the standard cache** for FastAPI services. It's fast, supports rich data structures, and is well-understood.
- **Cache-aside** is the most common pattern: read from cache, fall through to DB on miss, populate the cache.
- **Every cache entry must have a TTL**. Even a long one (24 hours) is better than infinite.
- **The thundering herd** is the biggest production issue. Fix it with locks, stale-while-revalidate, or background refreshes.
- **Multi-tenant keys are mandatory**. Always include the tenant ID in the key to prevent cross-tenant data leaks.
- **Invalidate on write** (delete the cache key). The next read repopulates.
- **Monitor cache hit rate**. Below 50% suggests the TTL is wrong; above 95% is the goal for hot data.
- **`allkeys-lru` or `allkeys-lfu`** is the right Redis eviction policy for caching.
- **Cache small, focused objects**. Don't store 5MB JSON blobs in Redis.
- **XFetch** (probabilistic early expiration) is the production-grade stampede prevention for high-traffic services.

## References

- [Redis Documentation](https://redis.io/docs/)
- [aiocache Documentation](https://aiocache.readthedocs.io/)
- [Optimal Probabilistic Cache Stampede Prevention (Vattani et al., 2015)](https://www.vldb.org/pvldb/vol8/p886-vattani.pdf)
- [Caching at Reddit](https://reddit.engineering/caching-at-reddit-the-software/)
- [Caching Best Practices (Dropbox)](https://dropbox.tech/infrastructure/caching-in-theory-and-practice)
- [FastAPI Caching Patterns](https://github.com/long2ice/fastapi-cache)
