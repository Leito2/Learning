# 🎭 Dramatiq and Saq

## 🎯 Learning Objectives

- Use Dramatiq for Celery-like features with a modern, simpler API
- Use Saq as an async-native alternative to RQ
- Compare the two against ARQ and Celery
- Choose the right framework for your team's operational maturity
- Implement the production patterns: retries, rate limiting, dead-letter queues

## Introduction

ARQ and Celery cover most FastAPI services. Dramatiq and Saq fill the remaining niches: Dramatiq is the modern Celery alternative for sync workers; Saq is the async-native alternative to RQ. This note covers both, with a focus on when each is the right choice.

The two frameworks share a lot of design philosophy: a single broker dependency, a small API surface, modern Python features, no chains or canvas. The main difference is sync (Dramatiq) vs async (Saq).

---

## 1. Dramatiq

### 1.1 What Dramatiq is

Dramatiq is a task queue for Python, designed to be a modern, simpler alternative to Celery. It uses RabbitMQ or Redis as a broker, supports actors (Dramatiq's term for tasks), retries with policies, rate limiting, and dead-letter queues.

The design philosophy: tasks should be easy to define, hard to misuse, and observable by default.

### 1.2 Installation

```bash
pip install dramatiq[redis]  # or dramatiq[rabbitmq]
```

### 1.3 An actor

```python
import dramatiq


@dramatiq.actor(
    max_retries=3,
    min_backoff=1000,   # 1 second
    max_backoff=60000,  # 60 seconds
    retry_when=lambda retries, exc: retries < 3,
)
def send_welcome_email(to: str, user_id: int):
    """Send a welcome email. Retries on failure with exponential backoff."""
    # ... send the email
    pass
```

The `@dramatiq.actor` decorator registers the function as a task. The arguments control retries, backoff, and other behavior.

### 1.4 Calling an actor

```python
@router.post("/users", status_code=201)
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    send_welcome_email.send(user.email, user.id)  # .send() enqueues
    return user
```

The `.send()` method is the equivalent of Celery's `.delay()`.

### 1.5 The broker

```python
import dramatiq
from dramatiq.brokers.redis import RedisBroker

broker = RedisBroker(host="redis", port=6379)
dramatiq.set_broker(broker)
```

The broker is the queue. The actor's `send()` enqueues to the broker; the worker consumes from it.

### 1.6 The worker

```bash
dramatiq app.actors --processes 4 --threads 8
```

The worker is a separate process. `--processes` is the number of worker processes; `--threads` is the number of threads per process. Total concurrency = `processes × threads`.

### 1.7 Retries with policies

Dramatiq's retry policy is a callable that decides whether to retry:

```python
import dramatiq
from dramatiq import Retry


@dramatiq.actor(
    retry_when=lambda retries, exc: isinstance(exc, ConnectionError) and retries < 5,
)
def call_external_api(url: str):
    """Call an external API. Retries only on ConnectionError, max 5 times."""
    response = httpx.get(url)
    response.raise_for_status()
```

The `retry_when` callable receives `(retries, exc)` and returns `True` to retry, `False` to fail. The default is "retry up to `max_retries` on any exception".

### 1.8 Rate limiting

```python
@dramatiq.actor(
    max_retries=3,
    rate_limit="100/min",  # max 100 messages per minute per actor
)
def call_external_api(url: str):
    # ...
```

Dramatiq's rate limiter is per-actor and per-key. Useful for not overwhelming an external API.

### 1.9 Dead-letter queue

```python
@dramatiq.actor(
    max_retries=3,
    on_failure=log_to_dlq,
)
def process_payment(payment_id: int):
    # ...


def log_to_dlq(message_data, exception):
    """Called when an actor's retries are exhausted."""
    db.add(DeadLetter(
        actor=message_data.actor,
        args=message_data.args,
        error=str(exception),
    ))
    db.commit()
```

The `on_failure` callable receives the message data and the exception. The standard pattern is to log the failure to a database for human inspection.

### 1.10 Scheduled jobs

Dramatiq has a separate scheduler package:

```bash
pip install dramatiq[redis] dramatiq-scheduler
```

```python
from dramatiq_scheduler import CronJob


@dramatiq.actor
def cleanup_old_uploads():
    # ... clean up
    pass


scheduler = CronJob()
cron_cleanup = scheduler.daily(at="03:00", actor=cleanup_old_uploads.message())
```

The scheduler enqueues the actor at the configured time. The scheduler itself is a separate process:

```bash
dramatiq_scheduler app.scheduler
```

### 1.11 Monitoring

Dramatiq has built-in support for Prometheus:

```python
import dramatiq
from dramatiq.middleware import Prometheus

dramatiq.add_middleware(PrometheusMiddleware())
```

The middleware exposes metrics on the worker's HTTP server (configurable port). Scrape with Prometheus and visualize in Grafana.

### 1.12 Dramatiq vs Celery

| Feature | Dramatiq | Celery |
|---------|----------|--------|
| Chains / groups / chords | ❌ | ✅ |
| Canvas | ❌ | ✅ |
| Scheduled jobs | `dramatiq-scheduler` (separate) | `beat` (built-in) |
| Broker | Redis, RabbitMQ | Many |
| Result backend | Optional | Built-in |
| API surface | Small | Large |
| Sync workers | ✅ | ✅ (default) |
| Async workers | ❌ (use `asgiref` for FastAPI) | ❌ (same) |

For complex workflows, Celery is the better choice. For everything else, Dramatiq is a sweet spot.

### 1.13 The FastAPI integration

Dramatiq is sync, but FastAPI handlers are async. The integration is straightforward: the handler calls `.send()` (sync, but just enqueues to Redis), and the worker processes the task in a separate process.

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dramatiq broker is initialized at import time; no setup here
    yield


app = FastAPI(lifespan=lifespan)
```

The FastAPI app enqueues; the Dramatiq worker runs in a separate process. No async/sync bridge needed because `.send()` is a quick Redis call.

---

## 2. Saq

### 2.1 What Saq is

Saq (Simple Async Queue) is an async-native job queue for Python, designed to be the async equivalent of RQ. It uses Redis as the broker, has a small API surface, and integrates naturally with FastAPI.

### 2.2 Installation

```bash
pip install saq
```

### 2.3 A job

```python
from saq import Job


async def send_welcome_email(ctx, to: str, user_id: int):
    """Send a welcome email. `ctx` is the worker context."""
    # ... send the email
    pass
```

A Saq job is an async function with a `ctx` parameter. The first argument is the worker context; subsequent arguments are the job's arguments.

### 2.4 The worker

```python
# app/worker.py
from saq import Queue

queue = Queue.from_url("redis://localhost:6379")


async def startup(ctx):
    # Called once when the worker starts
    ctx["db"] = async_session_maker()


async def shutdown(ctx):
    # Called when the worker stops
    await ctx["db"].close()


settings = {
    "queue": queue,
    "functions": [send_welcome_email, generate_report, process_upload],
    "concurrency": 5,
    "cron_jobs": [
        # Cron-style jobs
    ],
    "startup": startup,
    "shutdown": shutdown,
}
```

Run the worker:

```bash
saq app.worker.settings
```

### 2.5 Enqueueing jobs

```python
from saq import Queue

queue = Queue.from_url("redis://localhost:6379")


@router.post("/users", status_code=201)
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    await queue.enqueue("send_welcome_email", to=user.email, user_id=user.id)
    return user
```

`queue.enqueue` takes the function name (first argument) and the args.

### 2.6 Cron jobs

```python
settings = {
    "queue": queue,
    "functions": [send_welcome_email, cleanup_old_uploads],
    "cron_jobs": [
        # Run cleanup_old_uploads daily at 3am
        {
            "function": "cleanup_old_uploads",
            "cron": "0 3 * * *",
        },
        # Run weekly_report on Mondays at 9am
        {
            "function": "weekly_report",
            "cron": "0 9 * * 1",
        },
    ],
}
```

The cron syntax is the standard 5-field format. Saq uses `croniter` for parsing.

### 2.7 Retries

```python
from saq import Job


async def call_external_api(ctx, url: str, retries: int = 0):
    try:
        response = httpx.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        if retries < 3:
            # Re-enqueue with incremented retries
            await ctx["queue"].enqueue("call_external_api", url=url, retries=retries + 1)
        else:
            # Send to dead-letter
            await send_to_dlq("call_external_api", url, str(exc))
        raise
```

Saq's retry pattern is manual: the job re-enqueues itself with an incremented counter. This is more code than ARQ's built-in retries, but gives you full control.

### 2.8 Unique jobs (idempotency)

```python
await queue.enqueue("send_welcome_email", to=user.email, user_id=user.id, unique=True)
```

The `unique=True` argument prevents duplicate enqueues. Saq uses the function name and args as the unique key.

### 2.9 Saq vs ARQ

| Feature | Saq | ARQ |
|---------|-----|-----|
| Async-native | ✅ | ✅ |
| Cron jobs | ✅ (built-in) | ✅ (built-in) |
| Built-in retries | ❌ (manual) | ✅ |
| Unique jobs | ✅ | ❌ |
| Broker | Redis | Redis |
| API surface | Small | Small |

Saq and ARQ are very similar; the choice is mostly stylistic. Saq has cron jobs and unique jobs as built-in features; ARQ has built-in retries.

### 2.10 The FastAPI integration

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from saq import Queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.saqs = []
    for _ in range(4):
        q = Queue.from_url("redis://localhost:6379")
        app.state.saqs.append(q)
    yield
    for q in app.state.saqs:
        await q.close()


app = FastAPI(lifespan=lifespan)


async def get_saq(request: Request) -> Queue:
    # Round-robin across the pools
    return request.app.state.saqs.pop(0)


SaqDep = Annotated[Queue, Depends(get_saq)]


@router.post("/users")
async def create_user(payload: UserCreate, saq: Queue = Depends(get_saq)):
    # ... create user
    await saq.enqueue("send_welcome_email", to=user.email, user_id=user.id)
    return user
```

The pool of Saq queues enables connection pooling. Each handler gets a queue from the pool.

---

## 3. The Decision: Dramatiq, Saq, ARQ, or Celery?

| Need | Framework | Why |
|------|-----------|-----|
| FastAPI, async-native, simple, no chains | **ARQ** | Modern default for FastAPI |
| FastAPI, async-native, need unique jobs | Saq | Built-in `unique=True` |
| Celery-like features, modern API, sync workers | **Dramatiq** | The Celery alternative |
| Complex workflows (chains, groups, chords) | **Celery** | The only choice |
| Battle-tested, many features, mature | **Celery** | 15+ years in production |

For a **new FastAPI service in 2026**:
- **ARQ** is the default (covered in [[02 - ARQ Modern Async-Native|the previous note]]).
- Switch to **Celery** if you need chains, canvas, or complex routing.
- Consider **Dramatiq** if you need Celery-like features but with a smaller API surface.
- Consider **Saq** if you need unique jobs or async-native cron jobs that ARQ doesn't have.

---

## 4. The Production Patterns

### 4.1 Idempotency

All four frameworks expect idempotent jobs. The pattern is the same: pass a unique key, check before doing the work.

```python
# ARQ
async def send_email_with_idempotency(ctx, to: str, idempotency_key: str):
    if await ctx["redis"].get(f"email:sent:{idempotency_key}"):
        return
    # ... send
    await ctx["redis"].set(f"email:sent:{idempotency_key}", "1", ex=86400)


# Saq
await queue.enqueue("send_email", to=to, idempotency_key=key, unique=True)


# Dramatiq
@dramatiq.actor(unique=True)
def send_email(to: str, idempotency_key: str):
    # ...


# Celery
@celery_app.task(unique=True)
def send_email(to: str, idempotency_key: str):
    # ...
```

### 4.2 Retries

| Framework | Default | Configuration |
|-----------|---------|----------------|
| ARQ | 5 attempts, exponential backoff | `max_tries`, `retry_backoff` in `WorkerSettings` |
| Saq | Manual | Re-enqueue with incremented counter |
| Dramatiq | Configurable per actor | `max_retries`, `retry_when`, `min_backoff`, `max_backoff` |
| Celery | Configurable per task | `max_retries`, `default_retry_delay`, `autoretry_for` |

### 4.3 Dead-letter queue

| Framework | Built-in DLQ | Inspection |
|-----------|--------------|------------|
| ARQ | ✅ (Redis stream) | `arq.zrange("arq:dead", ...)` |
| Saq | Manual | Custom implementation |
| Dramatiq | ✅ (`on_failure` callback) | Configure a callback that logs |
| Celery | Manual (catch the exception) | Custom implementation |

### 4.4 Monitoring

| Framework | Built-in | Tools |
|-----------|----------|-------|
| ARQ | Logs only | Prometheus middleware (custom) |
| Saq | Logs only | External (Prometheus) |
| Dramatiq | `PrometheusMiddleware` | Built-in |
| Celery | Flower, celery-exporter | Rich ecosystem |

---

## 5. The Health Check Pattern

```python
@router.get("/health/workers")
async def worker_health(saq: Queue = Depends(get_saq)):
    info = await saq.info()
    return {
        "queued": info.queued,
        "active": info.active,
        "scheduled": info.scheduled,
        "workers": info.workers,
    }
```

Most frameworks expose a `.info()` method that returns queue stats. Use this in a `/health/workers` endpoint that Kubernetes (or your load balancer) pings.

---

## 6. The Cross-Framework Comparison

| Feature | ARQ | Saq | Dramatiq | Celery |
|---------|-----|-----|----------|--------|
| Async-native | ✅ | ✅ | ❌ (sync) | ❌ (sync) |
| Sync workers | ❌ | ❌ | ✅ | ✅ |
| Built-in cron | ✅ | ✅ | Via `dramatiq-scheduler` | ✅ (beat) |
| Built-in retries | ✅ | Manual | ✅ | ✅ |
| Idempotency key | Custom | ✅ `unique=True` | Custom | Custom |
| Dead-letter queue | ✅ | Manual | ✅ (callback) | Manual |
| Result backend | Optional | Optional | Optional | ✅ |
| Chains / canvas | ❌ | ❌ | ❌ | ✅ |
| Task routing | ❌ | ❌ | ✅ | ✅ |
| Multiple brokers | ❌ (Redis only) | ❌ (Redis only) | ✅ (Redis, RabbitMQ) | ✅ (many) |
| Prometheus | Custom | External | ✅ Built-in | Via celery-exporter |
| Active maintenance | ✅ | ✅ | ✅ | ✅ |
| Lines of code | ~2K | ~3K | ~5K | ~30K |
| Learning curve | Low | Low | Medium | High |

---

## 7. The Migration Path

### 7.1 From FastAPI BackgroundTasks to ARQ

```python
# Before: FastAPI BackgroundTasks
@router.post("/users")
async def create_user(payload: UserCreate, background_tasks: BackgroundTasks):
    user = await save_user(payload)
    background_tasks.add_task(send_email, user.email)
    return user


# After: ARQ
@router.post("/users")
async def create_user(payload: UserCreate, arq: ArqRedis = Depends(get_arq)):
    user = await save_user(payload)
    await arq.enqueue_job("send_welcome_email", to=user.email)
    return user
```

The change is small. The win: jobs survive process restarts, retries are configured, observability is added.

### 7.2 From ARQ to Celery

```python
# Before: ARQ
@router.post("/users")
async def create_user(payload: UserCreate, arq: ArqRedis = Depends(get_arq)):
    user = await save_user(payload)
    await arq.enqueue_job("send_welcome_email", to=user.email)
    return user


# After: Celery
@router.post("/users")
async def create_user(payload: UserCreate):
    user = await save_user(payload)
    send_welcome_email.delay(user.email, user.id)
    return user
```

The change is to drop the dependency; Celery is a global. The win: chains, canvas, beat, routing, mature ecosystem.

---

## 8. Código de Compresión

```python
"""
Compresión: Dramatiq and Saq
Covers: actor/job definitions, broker, worker, cron, retries, idempotency.
"""
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from saq import Queue
from saq import Job as SaqJob


# === Dramatiq ===
broker = RedisBroker(host="redis", port=6379)
dramatiq.set_broker(broker)


@dramatiq.actor(
    max_retries=3,
    min_backoff=1000,
    max_backoff=60000,
    retry_when=lambda retries, exc: isinstance(exc, ConnectionError) and retries < 3,
)
def send_welcome_email_dramatiq(to: str, user_id: int, idempotency_key: str):
    """Idempotent: check before sending."""
    if db.query(EmailLog).filter_by(idempotency_key=idempotency_key).first():
        return
    # ... send
    db.add(EmailLog(idempotency_key=idempotency_key))
    db.commit()


# === Saq ===
queue = Queue.from_url("redis://localhost:6379")


async def send_welcome_email_saq(ctx, to: str, user_id: int):
    """Saq job: async function with ctx parameter."""
    # ... send
    pass


async def saq_startup(ctx):
    ctx["db"] = async_session_maker()


async def saq_shutdown(ctx):
    await ctx["db"].close()


saq_settings = {
    "queue": queue,
    "functions": [send_welcome_email_saq],
    "concurrency": 5,
    "cron_jobs": [
        {"function": "cleanup_old_uploads", "cron": "0 3 * * *"},
    ],
    "startup": saq_startup,
    "shutdown": saq_shutdown,
}


# Enqueue from a FastAPI handler
@router.post("/users")
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    # Dramatiq: sync .send()
    send_welcome_email_dramatiq.send(user.email, user.id, idempotency_key=f"welcome:{user.id}")
    # Saq: async .enqueue() (also supports unique=True)
    await queue.enqueue("send_welcome_email_saq", to=user.email, user_id=user.id, unique=True)
    return user
```

---

## Key Takeaways

- **Dramatiq** is the modern Celery alternative for sync workers. Smaller API, better defaults, but no chains/canvas.
- **Saq** is the async-native RQ alternative. Similar to ARQ but with built-in cron jobs and unique-job support.
- For a **new FastAPI service**: ARQ by default, Celery if you need chains, Dramatiq if you want Celery-like with smaller API, Saq if you need unique jobs.
- **Idempotency** is mandatory for all four frameworks. The same job must be able to run twice without harm.
- **Retries** with exponential backoff and jitter prevent thundering herd. Every framework supports it, with different APIs.
- **Dead-letter queue** is the safety net. After max retries, the job goes to a DLQ for human inspection.
- **Cron jobs** are supported in all four (Celery beat, ARQ cron_jobs, Saq cron_jobs, dramatiq-scheduler). Celery's beat is the most mature.
- **Monitoring** is essential. Queue depth, job duration, failure rate, DLQ size.
- **Migration between frameworks** is straightforward. The handler code changes little; the worker setup changes more.

## References

- [Dramatiq Documentation](https://dramatiq.io/)
- [Dramatiq Scheduler](https://github.com/Bogdanp/dramatiq-scheduler)
- [Saq Documentation](https://github.com/tobymao/saq)
- [Celery vs Dramatiq comparison (2024)](https://dramatiq.io/comparison.html)
- [Redis Streams — the broker primitive](https://redis.io/docs/latest/develop/data-types/streams/)
- [Cron syntax reference](https://crontab.guru/)
- [Stripe's blog — Idempotency](https://stripe.com/blog/idempotency)
