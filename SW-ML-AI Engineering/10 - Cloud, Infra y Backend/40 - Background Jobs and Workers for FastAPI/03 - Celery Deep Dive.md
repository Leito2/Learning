# 🏗️ Celery Deep Dive

## 🎯 Learning Objectives

- Set up Celery for a FastAPI service: broker, backend, worker, beat scheduler
- Define tasks with the full Celery feature set: retries, ETA, expires, canvas (chains, groups, chords)
- Use the beat scheduler for cron-style jobs
- Monitor Celery with Flower and Prometheus
- Avoid the six most common Celery mistakes in production

## Introduction

Celery is the most feature-complete job queue in the Python ecosystem. Born in 2009, it has weathered every Python version change, every broker evolution, and every production scale-out. Its API is verbose and its learning curve is real, but for complex workflows — chains of tasks, parallel groups, scheduled jobs, results that need to be queried — Celery is unmatched.

This note covers Celery from "what is it" to "running in production". The patterns assume the [[../38 - SQLAlchemy 2.0 Async + Alembic for FastAPI/00 - Welcome|SQLAlchemy 2.0 stack]] and Redis as the broker. The note also covers the trade-off between Celery and ARQ: when to choose the simpler ARQ (covered in [[02 - ARQ Modern Async-Native|the previous note]]) and when Celery's complexity is worth it.

---

## 1. Why Celery (or Not)

### 1.1 When Celery is the right choice

- **Complex workflows**: chains (A → B → C), groups (A + B + C in parallel), chords (group + callback).
- **Scheduled jobs**: the `beat` scheduler is mature and battle-tested.
- **Multiple result backends**: Redis, RabbitMQ, Amazon SQS, databases.
- **Distributed task routing**: send tasks to specific queues based on the task name.
- **Team familiarity**: many Python developers already know Celery.

### 1.2 When ARQ is the better choice

- **FastAPI service**: ARQ is async-native; Celery's worker is sync (using `asgiref` to bridge to async adds complexity).
- **Simplicity preferred**: ARQ is ~2K LoC; Celery is ~30K LoC.
- **Moderate scale**: thousands of jobs per second is fine for ARQ; Celery's overhead is wasted if you don't need its features.

For a new FastAPI service with no chains or canvas, the default is **ARQ**. Switch to Celery when you hit a feature that ARQ doesn't have.

---

## 2. Installation

```bash
pip install celery[redis]
```

The `redis` extra pulls in the Redis client and message format.

---

## 3. The Anatomy of a Celery App

### 3.1 The Celery app instance

```python
# app/celery_app.py
from celery import Celery

celery_app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=[
        "app.tasks.email",
        "app.tasks.reports",
        "app.tasks.payments",
    ],
)
```

The `include` argument tells Celery which modules to import when the worker starts. The broker is the queue; the backend is where task results are stored.

### 3.2 The configuration

```python
# app/celery_config.py
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    broker_connection_retry_on_startup=True,
)
```

Key settings:

| Setting | What it does |
|---------|--------------|
| `task_serializer="json"` | Serialize tasks as JSON (more portable than pickle) |
| `task_acks_late=True` | Acknowledge the task **after** it completes; safer for crashes |
| `worker_prefetch_multiplier=1` | Workers fetch one task at a time (better for long tasks) |
| `task_default_queue="default"` | The default queue name |
| `broker_connection_retry_on_startup=True` | Retry the broker connection at startup |

### 3.3 A task

```python
# app/tasks/email.py
from app.celery_app import celery_app


@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,  # 60 seconds before retry
    autoretry_for=(ConnectionError, TimeoutError),
    acks_late=True,
)
def send_welcome_email(self, to: str, user_id: int):
    """Send a welcome email."""
    try:
        # ... send the email
        pass
    except ConnectionError as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
```

The `bind=True` argument gives the task access to its instance (`self`), which provides `self.request.retries` and `self.retry()`. The `autoretry_for` is a shortcut for retrying on specific exceptions.

### 3.4 Calling a task

```python
# From a FastAPI handler
from app.tasks.email import send_welcome_email


@router.post("/users", status_code=201)
async def create_user(
    payload: UserCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    # Enqueue the task
    send_welcome_email.delay(user.email, user.id)
    return user
```

The `.delay()` is the shortcut for `.apply_async()`. It enqueues the task and returns immediately.

---

## 4. Canvas: Chains, Groups, and Chords

### 4.1 When to use canvas

Canvas is for workflows where one task's output is the next task's input. Without canvas, you enqueue tasks one by one and manually chain them. With canvas, you describe the chain declaratively.

### 4.2 Chain (A → B → C)

```python
from celery import chain
from app.tasks.processing import step_one, step_two, step_three

# Three tasks; each takes the previous task's result
workflow = chain(step_one.s(input_data), step_two.s(), step_three.s())
result = workflow.apply_async()
# result.id is the workflow ID; result.parent is the previous task's ID
```

The `.s()` creates a task signature. `step_one.s(input_data)` is "step_one with input_data as its argument". `step_two.s()` takes whatever `step_one` returned.

### 4.3 Group (A + B + C in parallel)

```python
from celery import group
from app.tasks.processing import process_chunk

# Process 10 chunks in parallel
workflow = group(process_chunk.s(chunk) for chunk in chunks)
result = workflow.apply_async()
# result.join() waits for all to complete; result.results is the list of AsyncResult
```

A group runs tasks in parallel and returns a `GroupResult`.

### 4.4 Chord (group + callback)

```python
from celery import chord
from app.tasks.processing import process_chunk, aggregate_results

# 10 chunks in parallel, then aggregate the results
workflow = chord(
    group(process_chunk.s(chunk) for chunk in chunks),
    aggregate_results.s(),
)
result = workflow.apply_async()
```

A chord is a group with a callback that runs after the group completes. The callback receives the group's results.

### 4.5 The canvas in practice

```python
@router.post("/orders", status_code=202)
async def create_order(
    payload: OrderCreate,
    uow: UnitOfWork = Depends(get_uow),
):
    order = await uow.orders.create(**payload.model_dump())
    await uow.commit()
    # Pipeline: charge payment → reserve inventory → send confirmation
    workflow = chain(
        charge_payment.s(order.id),
        reserve_inventory.s(order.id),
        send_order_confirmation.s(order.id),
    )
    workflow.apply_async()
    return {"order_id": order.id, "status": "processing"}
```

The three tasks run sequentially; the user gets a 202 Accepted with the order ID and can poll the status.

---

## 5. The Beat Scheduler

### 5.1 What beat is

`celery beat` is a separate process that schedules recurring tasks. It runs alongside the workers and enqueues tasks at the configured times.

```python
# app/celery_config.py
celery_app.conf.beat_schedule = {
    "cleanup-old-uploads": {
        "task": "app.tasks.maintenance.cleanup_old_uploads",
        "schedule": 86400.0,  # every 24 hours
    },
    "send-daily-report": {
        "task": "app.tasks.reports.send_daily_report",
        "schedule": crontab(hour=8, minute=0),
    },
    "process-pending-webhooks": {
        "task": "app.tasks.webhooks.process_pending",
        "schedule": crontab(minute="*/5"),  # every 5 minutes
    },
}
```

### 5.2 Running beat

```bash
celery -A app.celery_app.celery_app beat --loglevel=info
```

Beat is a separate process from the worker. Both connect to the same broker; beat enqueues tasks, workers consume them.

### 5.3 The beat + worker pattern

In production:
- One or more **beat** processes (usually 1; multiple beats cause duplicate enqueues).
- One or more **worker** processes (scale horizontally based on throughput).

```yaml
# docker-compose.yml
services:
  beat:
    image: myapp
    command: celery -A app.celery_app.celery_app beat
  worker:
    image: myapp
    command: celery -A app.celery_app.celery_app worker -l info
    deploy:
      replicas: 4
```

### 5.4 The redbeat alternative

`celery beat` uses in-process scheduling, which can lose scheduled tasks if the process crashes. `redbeat` is an alternative that stores the schedule in Redis (durable):

```bash
pip install celery-redbeat
```

```python
celery_app.conf.update(
    redbeat_redis_url="redis://localhost:6379/2",
    redbeat_lock_timeout=25,
    beat_scheduler="redbeat.RedBeatScheduler",
)
```

`redbeat` is the right choice for production where beat uptime matters.

---

## 6. Task Routing

### 6.1 Why route tasks

Some tasks are CPU-bound; some are I/O-bound. Some are critical; some are not. Routing lets you put different task types on different queues with different worker pools:

```python
# app/celery_config.py
celery_app.conf.task_routes = {
    "app.tasks.reports.*": {"queue": "reports"},
    "app.tasks.payments.*": {"queue": "payments"},
    "app.tasks.email.*": {"queue": "email"},
    "app.tasks.maintenance.*": {"queue": "low_priority"},
}
```

Now you can run separate workers for each queue:

```bash
# Email worker
celery -A app.celery_app.celery_app worker -Q email -l info

# Report worker (CPU-bound, fewer concurrent)
celery -A app.celery_app.celery_app worker -Q reports --concurrency=2 -l info
```

This isolation is one of Celery's most powerful features.

### 6.2 Priority queues

```python
celery_app.conf.task_routes = {
    "app.tasks.payments.charge": {"queue": "high_priority"},
    "app.tasks.reports.generate": {"queue": "low_priority"},
}
```

Workers can be configured to consume high-priority queues first:

```bash
celery -A app.celery_app.celery_app worker -Q high_priority,low_priority
```

---

## 7. Retries and Idempotency

### 7.1 Retry configuration

```python
@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,  # exponential backoff
    retry_backoff_max=600,  # max 10 minutes
    retry_jitter=True,  # add randomness to spread retries
)
def send_welcome_email(self, to: str, user_id: int):
    # ... send the email
```

| Setting | Effect |
|---------|--------|
| `max_retries=5` | After 5 attempts, the task fails permanently |
| `default_retry_delay=60` | Base delay between retries |
| `retry_backoff=True` | Multiplies the delay by 2 each retry (1, 2, 4, 8, 16, 32, ...) |
| `retry_backoff_max=600` | Caps the delay at 10 minutes |
| `retry_jitter=True` | Adds random jitter (0-1s) to prevent thundering herd |

### 7.2 Idempotency

```python
@celery_app.task(bind=True, max_retries=3)
def charge_credit_card(self, payment_id: int, idempotency_key: str):
    # Check if already processed
    payment = db.query(Payment).filter_by(id=payment_id).first()
    if payment.status == "completed":
        return  # already charged
    # Charge
    stripe.Charge.create(
        amount=payment.amount,
        idempotency_key=idempotency_key,  # Stripe-side idempotency
    )
    payment.status = "completed"
    db.commit()
```

The idempotency key is unique per logical operation. The check on the database prevents double-charging.

---

## 8. The Result Backend

### 8.1 What it stores

Celery's result backend stores the return value of a task. The handler that enqueued the task can fetch the result:

```python
job = process_image.delay(image_id)
# Wait for the result
result = job.get(timeout=30)
# result is the return value of the task
```

The result is stored in the backend (Redis, database, etc.) for `result_expires` seconds (default 1 day).

### 8.2 When to use the result backend

Use it when:
- The task is **synchronous** from the user's perspective (the user polls for the result).
- You need the **output** of the task for the next step in a workflow.

Don't use it when:
- The task is **fire-and-forget** (just enqueue and forget).
- The result is large (the backend is for short values, not blobs).

---

## 9. Monitoring with Flower

### 9.1 What Flower is

[Flower](https://flower.readthedocs.io/) is a web-based dashboard for Celery. It shows active tasks, worker status, task history, and graphs.

```bash
pip install flower
celery -A app.celery_app.celery_app flower --port=5555
```

Visit `http://localhost:5555` to see the dashboard.

### 9.2 What Flower shows

- Active workers and their pool sizes.
- Active tasks (with arguments).
- Task history (success, failure, duration).
- Per-task-type statistics.
- Broker stats (queue depth, throughput).

### 9.3 Production monitoring

For production, pair Flower with Prometheus and Grafana:

```python
# app/celery_config.py
celery_app.conf.update(
    worker_send_task_events=True,
    task_send_sent_event=True,
)
```

The `celery-exporter` translates Celery events to Prometheus metrics. Grafana dashboards for Celery are well-documented.

---

## 10. The FastAPI Integration

### 10.1 The lifespan

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.celery_app import celery_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI does NOT start the Celery worker or beat;
    # those run as separate processes.
    yield


app = FastAPI(lifespan=lifespan)
```

The FastAPI app enqueues tasks; the Celery workers and beat run as separate processes. FastAPI does not run Celery.

### 10.2 Inspecting tasks in handlers

```python
from celery.result import AsyncResult
from app.celery_app import celery_app


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    return {
        "status": result.status,  # "PENDING", "STARTED", "SUCCESS", "FAILURE"
        "result": result.result if result.status == "SUCCESS" else None,
    }
```

The user can poll the status of a long-running job.

---

## 11. Production Patterns

### 11.1 The worker process

```bash
# Production: 4 workers, 8 threads each
celery -A app.celery_app.celery_app worker \
    --concurrency=8 \
    --queues=email,reports,payments \
    --loglevel=info \
    --max-tasks-per-child=1000  # restart worker after 1000 tasks to prevent memory leaks
```

`--max-tasks-per-child` is critical: it tells the worker to recycle after N tasks. This prevents the worker process from accumulating memory leaks.

### 11.2 Graceful shutdown

```bash
# SIGTERM (or K8s sends SIGTERM on pod termination)
# The worker:
# 1) Stops fetching new tasks
# 2) Waits for in-flight tasks to complete
# 3) Exits
```

Set K8s `terminationGracePeriodSeconds` to longer than the longest task (e.g., 600s for a 5-minute task).

### 11.3 Task priorities

```python
# Higher priority = processed first
send_payment_notification.apply_async(
    args=[payment_id],
    priority=9,  # 0 (lowest) to 9 (highest)
)
```

Priority requires a broker that supports it (Redis with `redis_queue_priority_max=10`).

### 11.4 Monitoring

```python
# Add a custom event
from celery.signals import task_postrun

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    duration = time.monotonic() - kwargs.get("start_time", time.monotonic())
    metrics.task_duration.labels(function=task.name).observe(duration)
```

Pair with `celery-exporter` for Prometheus metrics.

---

## 12. Testing

### 12.1 In-process testing with `CELERY_TASK_ALWAYS_EAGER`

```python
import pytest
from app.celery_app import celery_app


@pytest.fixture(autouse=True)
def eager_celery():
    """Run tasks synchronously in tests."""
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def test_signup_triggers_welcome_email(client, db_session):
    response = client.post("/users", json={"email": "alice@example.com", ...})
    # The task ran synchronously; verify the side effect
    assert db_session.query(EmailLog).filter_by(user_id=alice.id).count() == 1
```

`task_always_eager=True` makes tasks run synchronously. The test's HTTP request triggers the task in the same process.

### 12.2 Mocking specific tasks

```python
from unittest.mock import patch


def test_signup_doesnt_send_email(mocker):
    mocker.patch("app.tasks.email.send_welcome_email.delay")
    response = client.post("/users", json={...})
    send_welcome_email.delay.assert_called_once()
```

`mocker.patch` replaces the `.delay()` method; the task is never enqueued.

### 12.3 End-to-end with a real broker

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    # Use a real Redis instance
    result = chain(step_one.s(...), step_two.s(...), step_three.s(...)).apply_async()
    final = result.get(timeout=30)
    assert final == "expected"
```

This is slow but catches integration issues that the eager mode hides.

---

## 13. Common Pitfalls

### 13.1 Using pickle as the serializer

```python
# ❌ Pickle is dangerous
celery_app.conf.task_serializer = "pickle"

# ✅ JSON is portable and safe
celery_app.conf.task_serializer = "json"
```

Pickle allows arbitrary code execution; a malicious task payload can run code on the worker. Use JSON unless you have a specific reason.

### 13.2 Forgetting `acks_late`

```python
# ❌ Default: ack when the task is dequeued
# If the worker crashes mid-task, the task is lost

# ✅ Ack when the task completes
celery_app.conf.task_acks_late = True
```

`acks_late=True` ensures that if the worker crashes, the task is re-enqueued for another worker.

### 13.3 Not recycling workers

```bash
# ❌ Run forever
celery -A app.celery_app.celery_app worker

# ✅ Recycle after N tasks
celery -A app.celery_app.celery_app worker --max-tasks-per-child=1000
```

Without recycling, memory leaks accumulate. The worker eventually OOMs.

### 13.4 No concurrency limit

```python
# ❌ 1000 concurrent tasks overwhelm the worker
celery_app.conf.worker_concurrency = 1000

# ✅ A reasonable default
celery_app.conf.worker_concurrency = 8  # 8 tasks per worker
```

The default is the number of CPU cores. For I/O-bound tasks, more is OK; for CPU-bound, fewer.

### 13.5 Hardcoded broker URL

```python
# ❌ Hardcoded
celery_app = Celery("myapp", broker="redis://localhost:6379/0")

# ✅ Configurable
celery_app = Celery("myapp", broker=settings.CELERY_BROKER_URL)
```

Use environment variables for the broker URL (different in dev, staging, prod).

### 13.6 Missing `task_acks_late` and `acks_on_failure`

Celery's default behavior is to ack a task when it's dequeued. If the worker crashes mid-task, the task is lost. With `task_acks_late=True`, the ack happens after the task completes; a crash means the task is re-enqueued.

---

## 14. The Production Checklist

| Concern | Where | Verified by |
|---------|-------|-------------|
| Broker | Redis with persistence | `redis-cli ping` |
| Worker recycling | `--max-tasks-per-child` | Restart count |
| Graceful shutdown | `terminationGracePeriodSeconds` | Test SIGTERM |
| Idempotency | Per-task check | Test double-enqueue |
| Retry limits | `max_retries` | Test failure |
| Dead-letter handling | `celery-exporter` + alert | Alert on `dlq_size > 0` |
| Monitoring | Flower + Prometheus + Grafana | Dashboards |
| JSON serializer | `task_serializer="json"` | Inspect task payload |
| Task routing | `task_routes` config | Inspect queue |
| Acks late | `task_acks_late=True` | Crash test |

---

## 15. Código de Compresión

```python
"""
Compresión: Celery Deep Dive
Covers: setup, tasks, canvas, beat, routing, monitoring.
"""
from celery import Celery, chain, group, chord, crontab
from celery.signals import task_postrun
import time


# 1) App setup
celery_app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["app.tasks.email", "app.tasks.reports", "app.tasks.payments"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


# 2) Tasks
@celery_app.task(
    bind=True, max_retries=5, default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True, retry_backoff_max=600, retry_jitter=True,
)
def send_welcome_email(self, to: str, user_id: int, idempotency_key: str):
    # Idempotency
    if db.query(EmailLog).filter_by(idempotency_key=idempotency_key).first():
        return
    # ... send the email
    db.add(EmailLog(idempotency_key=idempotency_key, sent_at=...))
    db.commit()


@celery_app.task(bind=True, max_retries=3)
def generate_report(self, report_id: int):
    return _compute_report(report_id)


@celery_app.task
def charge_payment(payment_id: int):
    # ... charge via Stripe
    pass


# 3) Beat schedule
celery_app.conf.beat_schedule = {
    "cleanup-old-uploads": {
        "task": "app.tasks.maintenance.cleanup_old_uploads",
        "schedule": 86400.0,
    },
    "send-daily-report": {
        "task": "app.tasks.reports.send_daily_report",
        "schedule": crontab(hour=8, minute=0),
    },
}


# 4) Task routing
celery_app.conf.task_routes = {
    "app.tasks.payments.*": {"queue": "payments"},
    "app.tasks.reports.*": {"queue": "reports"},
    "app.tasks.email.*": {"queue": "email"},
}


# 5) Canvas example
@router.post("/orders", status_code=202)
async def create_order(payload: OrderCreate, uow: UnitOfWork = Depends(get_uow)):
    order = await uow.orders.create(**payload.model_dump())
    await uow.commit()
    workflow = chain(
        charge_payment.s(order.id),
        reserve_inventory.s(order.id),
        send_order_confirmation.s(order.id),
    )
    workflow.apply_async()
    return {"order_id": order.id, "status": "processing"}


# 6) Monitoring
@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    duration = kwargs.get("runtime", 0)
    metrics.task_duration.labels(function=task.name).observe(duration)
```

---

## Key Takeaways

- Celery is the most feature-complete Python job queue: chains, groups, chords, beat, routing, mature ecosystem. It is the right choice for complex workflows.
- For a **FastAPI service with no canvas**, ARQ is simpler and async-native. Switch to Celery when you need chains, scheduled jobs, or task routing.
- Celery workers and beat run as **separate processes** from the FastAPI app. FastAPI enqueues tasks; workers consume them.
- **Task routing** lets you put different task types on different queues with different worker pools. Email workers can be lightweight; report workers can be CPU-optimized.
- **Beat** is a separate process that schedules recurring tasks. Use `redbeat` for durable schedule storage in Redis.
- **Idempotency** is mandatory. The same task must be able to run twice without harm; check a unique key before doing the work.
- **Retries with exponential backoff and jitter** prevent thundering herd. After `max_retries`, the task fails permanently; monitor the dead-letter queue.
- **Task `acks_late=True`** is the default for production. A worker crash mid-task re-enqueues the task; without it, the task is lost.
- **`--max-tasks-per-child`** recycles workers periodically, preventing memory leaks. The default of "run forever" is a recipe for OOMs.
- **JSON serialization** is safer than pickle. Pickle allows arbitrary code execution; only use it for trusted internal payloads.
- **Flower** is the web dashboard; **celery-exporter** is the Prometheus bridge. Both are needed for production visibility.

## References

- [Celery Documentation](https://docs.celeryq.dev/)
- [Celery Canvas — Chains, Groups, Chords](https://docs.celeryq.dev/en/stable/userguide/canvas.html)
- [Celery Beat](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html)
- [Celery Routing](https://docs.celeryq.dev/en/stable/userguide/routing.html)
- [Flower](https://flower.readthedocs.io/)
- [celery-exporter](https://github.com/danihodovic/celery-exporter)
- [redbeat](https://github.com/sibson/redbeat)
- [Designing Data-Intensive Applications — Chapter 11 (Stream Processing)](https://dataintensive.net/)
