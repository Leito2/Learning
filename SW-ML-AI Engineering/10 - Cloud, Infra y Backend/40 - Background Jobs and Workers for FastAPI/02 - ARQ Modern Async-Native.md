# 🚀 ARQ: Modern Async-Native

## 🎯 Learning Objectives

- Set up ARQ as the default job queue for a FastAPI service
- Define jobs (functions) and workers (processes that run them)
- Implement the production patterns: retries, idempotency, scheduled jobs, graceful shutdown
- Integrate ARQ with FastAPI's lifespan and the existing SQLAlchemy stack
- Avoid the four most common ARQ mistakes

## Introduction

ARQ is the modern default for async-native background jobs in a FastAPI service. It uses Redis Streams (the modern Redis data structure for queues), speaks the same `asyncio` dialect as FastAPI, and has a tiny API surface. There is no Celery-style canvas (no chains, groups, or chords); ARQ is opinionated about simplicity. For 80% of FastAPI services, that simplicity is exactly the right trade-off.

This note walks through ARQ from "what is it" to "running in production". The patterns assume the [[../38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/00 - Welcome|SQLAlchemy 2.0 stack]] and Redis as the broker. The capstone at the end of the course shows ARQ in a real workflow.

---

## 1. Why ARQ

### 1.1 The design philosophy

ARQ is the opinion that a job queue should be:

- **Async-native**: no sync/async impedance mismatch with FastAPI.
- **Single-purpose**: jobs are async functions; no canvas, no DSL.
- **Redis-only**: one broker, one dependency.
- **Lightweight**: ~2K lines of code, easy to read and debug.
- **Modern**: Redis Streams (the modern queue primitive), not the older list-based queues.

The trade-off: ARQ does not have Celery's chains, groups, or beat scheduler. If you need those, use Celery. If you don't, ARQ is simpler to operate.

### 1.2 When ARQ is the right choice

- FastAPI service (async-native, same event loop).
- Job types are independent (no chains).
- Redis is already in the stack (or you can add it).
- Team prefers simple, readable code over powerful frameworks.
- Workload is moderate (thousands of jobs per second, not millions).

---

## 2. Installation

```bash
pip install arq
```

That's it. ARQ is a single small package with one broker dependency (Redis).

---

## 3. The Anatomy of an ARQ Job

### 3.1 A job is an async function

```python
async def send_welcome_email(ctx, to: str):
    """Send a welcome email. `ctx` is the worker context."""
    async with aiosmtplib.SMTP(...) as smtp:
        await smtp.send_message(...)
```

The first parameter is the **worker context** (`ctx`). It carries the worker's runtime state: the Redis connection, the logger, custom configuration. Subsequent parameters are the job's arguments.

### 3.2 A worker is a process

```python
# worker.py
from arq.worker import create_worker
from arq.connections import RedisSettings

from app.jobs import send_welcome_email, generate_report, process_upload

redis_settings = RedisSettings(host="redis", port=6379)

class WorkerSettings:
    functions = [send_welcome_email, generate_report, process_upload]
    redis_settings = redis_settings
    max_jobs = 10  # max concurrent jobs per worker
    job_timeout = 300  # max seconds per job
    keep_result = 60  # keep results in Redis for 60 seconds
    health_check_interval = 30  # how often to log worker health
```

Run the worker as a separate process:

```bash
arq app.worker.WorkerSettings
```

### 3.3 Enqueueing jobs

```python
from arq import create_pool
from arq.connections import RedisSettings

redis = await create_pool(RedisSettings())

await redis.enqueue_job("send_welcome_email", to="alice@example.com")
```

The first argument is the **function name** (must match a function in `WorkerSettings.functions`). The remaining arguments are passed to the function.

---

## 4. FastAPI Integration

### 4.1 The Redis pool as a lifespan dependency

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from arq import create_pool
from arq.connections import RedisSettings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create the Redis pool
    app.state.arq = await create_pool(RedisSettings())
    yield
    # Shutdown: close the pool
    await app.state.arq.close()


app = FastAPI(lifespan=lifespan)
```

### 4.2 Enqueueing from a handler

```python
from fastapi import Request, Depends
from arq import ArqRedis


async def get_arq(request: Request) -> ArqRedis:
    return request.app.state.arq


ArqDep = Annotated[ArqRedis, Depends(get_arq)]


@router.post("/users", status_code=201)
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
    arq: ArqRedis = Depends(get_arq),
):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    # Enqueue the welcome email
    await arq.enqueue_job("send_welcome_email", to=user.email, user_id=user.id)
    return user
```

The handler returns immediately (201 Created) with the user. The email is sent in the background.

### 4.3 The full integration

```python
# app/jobs/__init__.py
from arq.connections import RedisSettings


class WorkerSettings:
    functions = [
        send_welcome_email,
        generate_report,
        process_upload,
    ]
    redis_settings = RedisSettings()
    max_jobs = 10
    job_timeout = 300
    keep_result = 60
```

```python
# app/jobs/send_welcome_email.py
from arq.connections import ArqRedis
import aiosmtplib
from email.message import EmailMessage
from app.core.config import settings


async def send_welcome_email(ctx, to: str, user_id: int):
    """Send a welcome email. Called by the ARQ worker."""
    msg = EmailMessage()
    msg["From"] = settings.EMAIL_FROM
    msg["To"] = to
    msg["Subject"] = "Welcome!"
    msg.set_content("Welcome to MyApp!")
    async with aiosmtplib.SMTP(hostname=settings.SMTP_HOST, port=587) as smtp:
        await smtp.login(settings.SMTP_USER, settings.SMTP_PASS)
        await smtp.send_message(msg)
    # Log the success
    ctx["logger"].info(f"Welcome email sent to {to}")
```

### 4.4 Idempotency

```python
async def send_welcome_email(ctx, to: str, user_id: int, idempotency_key: str):
    # Check if already sent
    if await ctx["redis"].get(f"email:sent:{idempotency_key}"):
        ctx["logger"].info(f"Email already sent: {idempotency_key}")
        return
    # Send
    await _do_send(to)
    # Mark as sent (24 hours)
    await ctx["redis"].set(f"email:sent:{idempotency_key}", "1", ex=86400)
```

The idempotency key is unique per logical operation (e.g., `welcome-email:{user_id}`). The Redis SET with TTL is the marker.

---

## 5. Retries with Exponential Backoff

### 5.1 Per-job retry configuration

```python
class WorkerSettings:
    functions = [...]
    redis_settings = ...
    # Global defaults
    max_tries = 5  # total attempts (initial + 4 retries)
    retry_backoff = lambda ctx: 2 ** ctx["job_try"]  # exponential: 2, 4, 8, 16, 32 seconds
```

The `retry_backoff` callable is called with the worker context; it returns the delay in seconds before the next retry.

### 5.2 Per-job overrides

```python
# A job that should not retry
async def fire_and_forget(ctx, to: str):
    """Send a notification; do not retry on failure."""
    await _send_notification(to)
```

The function's name in `WorkerSettings.functions` controls the default behavior. For per-job control, raise specific exceptions:

```python
async def process_payment(ctx, payment_id: int):
    try:
        await _charge(payment_id)
    except PaymentDeclinedError:
        # Permanent failure; do not retry
        raise
    except NetworkError:
        # Transient failure; ARQ will retry
        raise
```

By default, ARQ retries on any exception. To mark an exception as non-retryable, raise `arq.worker.JobError` with `retry=False`:

```python
from arq.worker import JobError


async def process_payment(ctx, payment_id: int):
    try:
        await _charge(payment_id)
    except PaymentDeclinedError as e:
        raise JobError(f"Payment declined: {e}", retry=False)
    except NetworkError:
        raise  # ARQ will retry
```

### 5.3 Dead-letter queue

After `max_tries`, the job is moved to a "dead" stream. ARQ exposes this as a separate Redis stream; you can inspect it or set up an alert:

```python
# admin/inspect_dlq.py
async def inspect_dead_letter(arq: ArqRedis):
    """List jobs that have exhausted their retries."""
    jobs = await arq.zrange("arq:dead", 0, -1, withscores=True)
    for job_id, score in jobs:
        job = await arq.job_info(job_id)
        print(f"Job {job_id}: {job.function} (failed at {score})")
```

For automated handling, set up a separate worker that consumes the dead-letter queue and:
- Logs the job details.
- Sends an alert (Slack, PagerDuty).
- Optionally re-enqueues with a fix (after human review).

---

## 6. Scheduled Jobs (Cron)

ARQ has a built-in scheduler for recurring jobs:

```python
from arq.cron import cron

class WorkerSettings:
    functions = [...]
    cron_jobs = [
        # Run every 5 minutes
        cron(generate_recommendations, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}),
        # Run daily at 3am
        cron(cleanup_old_uploads, hour=3, minute=0),
        # Run every Monday at 9am
        cron(weekly_report, weekday=0, hour=9, minute=0),
    ]
```

The arguments to `cron` are the same as the standard crontab syntax. The function is the job to run.

### 6.1 A scheduled job example

```python
async def cleanup_old_uploads(ctx):
    """Delete uploads older than 30 days."""
    cutoff = datetime.now() - timedelta(days=30)
    async with async_session_maker() as session:
        async with session.begin():
            await session.execute(
                delete(Upload).where(Upload.created_at < cutoff)
            )
    ctx["logger"].info("Old uploads cleaned up")
```

---

## 7. Job Results

### 7.1 Return values

A job can return a value. The result is stored in Redis for `keep_result` seconds:

```python
async def generate_report(ctx, report_id: int) -> dict:
    """Generate a report and return the result."""
    report = await _compute_report(report_id)
    return {"report_id": report_id, "size": len(report)}
```

The caller can fetch the result:

```python
job = await arq.enqueue_job("generate_report", report_id=42)
# Wait for the result (with timeout)
result = await job.result(timeout=60)
print(result)
```

The result is a JSON-serializable value. The default is 60s retention; set `keep_result` on the worker to adjust.

### 7.2 Polling from a handler

For long-running jobs, the user polls a status endpoint:

```python
@router.post("/reports", status_code=202)
async def start_report(payload: ReportRequest, arq: ArqRedis = Depends(get_arq)):
    job = await arq.enqueue_job("generate_report", report_id=...)
    return {"job_id": job.job_id, "status": "queued"}


@router.get("/reports/{job_id}")
async def get_report_status(job_id: str, arq: ArqRedis = Depends(get_arq)):
    info = await arq.job_info(job_id)
    if info is None:
        raise HTTPException(404, "Job not found")
    return {
        "status": info.status,  # "queued", "in_progress", "complete", "failed"
        "result": info.result if info.status == "complete" else None,
    }
```

---

## 8. The Worker Context

### 8.1 What `ctx` carries

The worker context is a dict with:

- `redis`: the Redis connection.
- `logger`: a structured logger.
- `job_id`: the unique ID of the current job.
- `job_try`: the attempt number (1-based).
- `max_tries`: the configured max retries.
- `enqueue_time`: when the job was enqueued.
- `score`: the job's priority (for delayed jobs).

You can also add custom keys via `ctx['custom'] = ...` in the worker's startup.

### 8.2 Custom startup

```python
class WorkerSettings:
    functions = [...]
    redis_settings = ...
    
    async def on_startup(ctx):
        # Called once when the worker starts
        ctx["db_session_maker"] = async_session_maker()
        ctx["logger"].info("Worker started")
    
    async def on_shutdown(ctx):
        # Called when the worker shuts down
        await ctx["db_session_maker"].close()
        ctx["logger"].info("Worker stopped")
```

The `on_startup` callback is the right place to set up database connections, HTTP clients, or any other long-lived resources.

### 8.3 Using the context in jobs

```python
async def process_upload(ctx, upload_id: int):
    """Process an upload. Uses the shared DB session from the worker context."""
    session = ctx["db_session_maker"]()
    try:
        # ... process the upload
    finally:
        await session.close()
```

The `ctx["db_session_maker"]` is created once in `on_startup`; the job uses it for each invocation.

---

## 9. Production Patterns

### 9.1 Monitoring

```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

jobs_total = Counter("arq_jobs_total", "Total jobs processed", ["function", "status"])
job_duration = Histogram("arq_job_duration_seconds", "Job duration", ["function"])


async def instrumented_send_welcome_email(ctx, to: str, user_id: int):
    start = time.monotonic()
    try:
        await send_welcome_email(ctx, to, user_id)
        jobs_total.labels(function="send_welcome_email", status="success").inc()
    except Exception as e:
        jobs_total.labels(function="send_welcome_email", status="fail").inc()
        raise
    finally:
        job_duration.labels(function="send_welcome_email").observe(time.monotonic() - start)
```

Alternatively, the ARQ worker logs metrics via structured logs. The metrics are scraped by Prometheus or a similar tool.

### 9.2 Health check endpoint

```python
@router.get("/health/workers")
async def worker_health(arq: ArqRedis = Depends(get_arq)):
    info = await arq.info()
    return {
        "jobs_complete": info.jobs_complete,
        "jobs_failed": info.jobs_failed,
        "jobs_retried": info.jobs_retried,
        "jobs_in_progress": info.jobs_in_progress,
        "queue_size": info.queue_size,
    }
```

The `arq.info()` returns a snapshot of worker stats.

### 9.3 Graceful shutdown

The worker handles SIGTERM by:
1. Stopping the poll loop.
2. Waiting for in-flight jobs to complete.
3. Exiting.

Kubernetes' `terminationGracePeriodSeconds` is the timeout for the worker to finish. Set it to longer than the longest job (e.g., 600s for a 5-minute job timeout).

```yaml
# k8s deployment
spec:
  terminationGracePeriodSeconds: 600
  containers:
  - name: worker
    command: ["arq", "app.jobs.WorkerSettings"]
```

### 9.4 Scaling workers

ARQ scales horizontally. Run more worker processes to handle more jobs:

```bash
# Run 4 workers in parallel
arq app.jobs.WorkerSettings &
arq app.jobs.WorkerSettings &
arq app.jobs.WorkerSettings &
arq app.jobs.WorkerSettings &
```

In Kubernetes, the worker deployment has `replicas: 4`. The Redis broker distributes jobs across all four.

### 9.5 Per-job concurrency

By default, ARQ runs `max_jobs` jobs concurrently per worker. For mixed workloads (some fast, some slow), separate the workers:

```bash
# Fast worker for email
arq app.jobs.EmailWorkerSettings --max-jobs 20

# Slow worker for reports
arq app.jobs.ReportWorkerSettings --max-jobs 2
```

The workers share the same Redis but use different `functions` lists and different `max_jobs` settings.

---

## 10. Testing

### 10.1 In-process testing with the `arq` mock

```python
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from app.jobs import send_welcome_email


@pytest.fixture
async def arq_pool():
    pool = await create_pool(RedisSettings())
    yield pool
    await pool.close()


@pytest.mark.asyncio
async def test_send_welcome_email_runs(arq_pool):
    job = await arq_pool.enqueue_job("send_welcome_email", to="test@example.com", user_id=1)
    # Wait for the result
    result = await job.result(timeout=10)
    # Verify the side effect (e.g., a database row was created)
    ...
```

For pure unit tests, you can call the job function directly:

```python
async def test_send_welcome_email_unit():
    ctx = {"redis": mock_redis, "logger": mock_logger}
    await send_welcome_email(ctx, to="test@example.com", user_id=1)
    # Verify the side effect
```

### 10.2 End-to-end test with TestClient

```python
@pytest.mark.asyncio
async def test_signup_triggers_welcome_email(client, arq_pool):
    response = await client.post(
        "/users",
        json={"email": "alice@example.com", "name": "Alice", "password": "secret123"},
    )
    assert response.status_code == 201
    # The job was enqueued; verify it's in the queue
    job_info = await arq_pool.job_info(response.json()["welcome_job_id"])
    assert job_info.function == "send_welcome_email"
```

---

## 11. Código de Compresión

```python
"""
Compresión: ARQ for FastAPI
Covers: job definition, worker setup, FastAPI integration, retries,
       idempotency, cron, monitoring, graceful shutdown.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, Request
from arq import create_pool, ArqRedis
from arq.connections import RedisSettings
from arq.cron import cron
from arq.worker import JobError
import aiosmtplib
from email.message import EmailMessage


# 1) Jobs
async def send_welcome_email(ctx, to: str, user_id: int, idempotency_key: str):
    """Idempotent: check Redis before sending."""
    if await ctx["redis"].get(f"email:sent:{idempotency_key}"):
        return
    msg = EmailMessage()
    msg["From"] = "noreply@example.com"
    msg["To"] = to
    msg["Subject"] = "Welcome!"
    msg.set_content("Welcome to MyApp!")
    async with aiosmtplib.SMTP(hostname="smtp.example.com", port=587) as smtp:
        await smtp.login("user", "pass")
        await smtp.send_message(msg)
    await ctx["redis"].set(f"email:sent:{idempotency_key}", "1", ex=86400)


async def cleanup_old_uploads(ctx):
    """Scheduled job: delete uploads older than 30 days."""
    cutoff = datetime.now() - timedelta(days=30)
    async with ctx["db_session_maker"]() as session:
        async with session.begin():
            await session.execute(
                Upload.__table__.delete().where(Upload.created_at < cutoff)
            )


async def process_payment(ctx, payment_id: int):
    try:
        await _charge(payment_id)
    except PaymentDeclinedError as e:
        raise JobError(f"Payment declined: {e}", retry=False)
    except NetworkError:
        raise  # ARQ will retry


# 2) Worker settings
class WorkerSettings:
    functions = [send_welcome_email, process_payment]
    redis_settings = RedisSettings()
    max_jobs = 10
    job_timeout = 300
    max_tries = 5
    retry_backoff = lambda ctx: 2 ** ctx["job_try"]
    keep_result = 60
    cron_jobs = [
        cron(cleanup_old_uploads, hour=3, minute=0),
    ]
    
    async def on_startup(ctx):
        ctx["db_session_maker"] = async_session_maker()
    
    async def on_shutdown(ctx):
        await ctx["db_session_maker"].close()


# 3) FastAPI integration
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.arq = await create_pool(RedisSettings())
    yield
    await app.state.arq.close()


app = FastAPI(lifespan=lifespan)


async def get_arq(request: Request) -> ArqRedis:
    return request.state.arq


ArqDep = Annotated[ArqRedis, Depends(get_arq)]


# 4) Enqueueing from a handler
@router.post("/users", status_code=201)
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
    arq: ArqRedis = Depends(get_arq),
):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    await arq.enqueue_job(
        "send_welcome_email",
        to=user.email,
        user_id=user.id,
        idempotency_key=f"welcome:{user.id}",
    )
    return user
```

---

## Key Takeaways

- ARQ is the modern default for FastAPI background jobs: async-native, lightweight, Redis-based, integrates with FastAPI's lifespan.
- A job is an **async function** with a `ctx` parameter. The function is registered in `WorkerSettings.functions`.
- The Redis pool is created in **lifespan** and shared across requests. The worker is a **separate process**; the API does not run jobs.
- **Idempotency** is mandatory. Use a unique key per logical operation; check the database or Redis before doing the work.
- **Retries** are bounded (5 is common); after that, the job goes to a **dead-letter queue** for human inspection.
- **Scheduled jobs** (cron) are part of the worker config. No separate scheduler process is needed.
- The worker context (`ctx`) carries the Redis connection, logger, and shared resources created in `on_startup`.
- **Graceful shutdown** is the worker's responsibility: finish in-flight jobs before exiting. Kubernetes' `terminationGracePeriodSeconds` is the timeout.
- **Scaling** is horizontal: run more worker processes. Mixed workloads (fast + slow) can use different worker pools.
- **Monitoring** is essential: queue depth, job duration, failure rate, dead-letter size.

## References

- [ARQ Documentation](https://arq-docs.helpmanual.io/)
- [ARQ GitHub Repository](https://github.com/python-arq/arq)
- [Redis Streams — the broker primitive](https://redis.io/docs/latest/develop/data-types/streams/)
- [Sentry's blog — Background Job Processing](https://sentry.io/blog/background-jobs/)
- [Stripe's blog — Idempotency](https://stripe.com/blog/idempotency)
- [AWS — Backpressure Pattern](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)
