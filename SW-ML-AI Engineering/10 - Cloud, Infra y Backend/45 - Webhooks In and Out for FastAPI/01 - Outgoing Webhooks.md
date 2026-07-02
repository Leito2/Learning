# 📤 Outgoing Webhooks

## 🎯 Learning Objectives

- Build an outgoing webhook delivery system with HMAC signatures
- Implement retry with exponential backoff and jitter
- Set up a dead-letter queue for permanently failed deliveries
- Verify webhook signatures on the receiving end
- Avoid the four most common outgoing webhook mistakes

## Introduction

Outgoing webhooks are how your service notifies other systems about events. The user signs up, your service POSTs to the customer's CRM; the payment succeeds, your service POSTs to the accounting system; the build completes, your service POSTs to Slack. The pattern is simple, but production-grade delivery is a real engineering challenge: the receiver can be down, the network can drop packets, the receiver can be slow. A single retry isn't enough; the retry needs backoff, jitter, and a dead-letter queue for permanent failures.

This note covers the full outgoing webhook flow: signing, retry, monitoring, and failure handling. The patterns are based on the [Standard Webhooks](https://www.standardwebhooks.com/) specification and production experience at Stripe, GitHub, Shopify, and others.

---

## 1. The Anatomy of an Outgoing Webhook

### 1.1 The HTTP request

```
POST /webhook HTTP/1.1
Host: customer-crm.example.com
Content-Type: application/json
User-Agent: MyApp-Webhook/1.0
X-MyApp-Signature: sha256=7d38c...
X-MyApp-Timestamp: 1700000000
X-MyApp-Id: evt_abc123
X-MyApp-Event: user.signed_up

{
  "event": "user.signed_up",
  "data": {
    "user_id": 42,
    "email": "alice@example.com",
    "name": "Alice"
  }
}
```

The body is the event payload. The headers carry metadata: signature (for verification), timestamp (for replay protection), event ID (for idempotency), and event type (for routing).

### 1.2 The signature

The signature is an HMAC-SHA256 of the timestamp + body, using a shared secret:

```
signature = HMAC-SHA256(secret, f"{timestamp}.{body}")
```

The receiver:
1. Reads the `X-MyApp-Timestamp` and `X-MyApp-Signature` headers.
2. Computes the HMAC of `timestamp.body` with their copy of the secret.
3. Compares (constant-time) with the provided signature.
4. Verifies the timestamp is within a tolerance window (e.g., 5 minutes).

The timestamp is part of the signed payload to prevent replay attacks: an attacker who captures a valid request can't replay it indefinitely because the timestamp check will fail.

### 1.3 The Standard Webhooks spec

The [Standard Webhooks](https://www.standardwebhooks.com/) initiative (by Svix, the company behind the popular webhook platform) defines a vendor-neutral format:

```
Webhook payload format:
{
  "id": "evt_...",
  "object": "event",
  "type": "user.signed_up",
  "created_at": "2024-01-15T12:00:00Z",
  "data": { ... }
}

Signature header: webhook-id, webhook-timestamp, webhook-signature
Signature scheme: v1,sha256=base64(HMAC-SHA256(secret, "id.timestamp.body"))
```

Following the spec means the receiver can use a standard library to verify, regardless of which service sent the webhook. It's the right default for new implementations.

---

## 2. The Implementation

### 2.1 The payload

```python
# app/webhooks/outgoing.py
import secrets
import time
import json
from typing import Any


def make_webhook_payload(
    event_type: str,
    data: dict,
    *,
    event_id: str | None = None,
) -> dict:
    """Build a webhook payload in the Standard Webhooks format."""
    return {
        "id": event_id or f"evt_{secrets.token_urlsafe(16)}",
        "object": "event",
        "type": event_type,
        "created_at": int(time.time()),
        "data": data,
    }
```

The payload has a unique `id` (for idempotency), a `type` (for routing), and the event data.

### 2.2 The signature

```python
import hmac
import hashlib
import base64


def sign_payload(secret: str, payload: dict, timestamp: int | None = None) -> dict:
    """Sign a webhook payload. Returns the headers to include."""
    timestamp = timestamp or int(time.time())
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signed = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode()
    return {
        "webhook-id": payload["id"],
        "webhook-timestamp": str(timestamp),
        "webhook-signature": f"v1,{signature}",
    }
```

The signature is `v1,<base64-digest>`. The `v1` prefix is the version; future versions (e.g., `v2`) can rotate without breaking v1 receivers.

### 2.3 The send function

```python
import httpx


async def send_webhook(
    url: str,
    secret: str,
    payload: dict,
    *,
    timeout: float = 30.0,
) -> httpx.Response:
    """Send a webhook with the Standard Webhooks signature."""
    headers = sign_payload(secret, payload)
    headers["Content-Type"] = "application/json"
    headers["User-Agent"] = "MyApp-Webhook/1.0"
    
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, content=body, headers=headers)
    return response
```

The `httpx` call is async; the response includes the status code and the receiver's response body. The caller checks the status code for success (2xx) or failure.

### 2.4 The retry policy

```python
import random


RETRY_DELAYS = [
    (1, 30),      # 1st retry: 1-30 seconds
    (30, 300),    # 2nd retry: 30s - 5 min
    (300, 1800),  # 3rd retry: 5 min - 30 min
    (1800, 7200), # 4th retry: 30 min - 2 hours
]


def get_retry_delay(attempt: int) -> int:
    """Compute the retry delay for the given attempt (1-indexed)."""
    if attempt > len(RETRY_DELAYS):
        return None  # No more retries
    min_delay, max_delay = RETRY_DELAYS[attempt - 1]
    return random.randint(min_delay, max_delay)  # Jitter
```

The delay is **exponential with jitter**. The jitter prevents thundering herd when the receiver comes back online after an outage.

---

## 3. The Delivery System

### 3.1 The data model

```python
class WebhookSubscription(Base):
    """A customer's webhook subscription."""
    __tablename__ = "webhook_subscriptions"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    url: Mapped[str] = mapped_column(String(500))
    secret: Mapped[str] = mapped_column(String(64))  # The shared secret for signing
    event_types: Mapped[list[str]] = mapped_column(JSON)  # e.g., ["user.signed_up", "payment.succeeded"]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class WebhookDelivery(Base):
    """A record of each webhook delivery attempt."""
    __tablename__ = "webhook_deliveries"
    id: Mapped[int] = mapped_column(primary_key=True)
    subscription_id: Mapped[int] = mapped_column(ForeignKey("webhook_subscriptions.id"), index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)  # The webhook-id
    event_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # "pending", "delivered", "failed", "dead"
    attempts: Mapped[int] = mapped_column(default=0)
    last_attempt_at: Mapped[datetime | None] = mapped_column(default=None)
    last_response_status: Mapped[int | None] = mapped_column(default=None)
    last_response_body: Mapped[str | None] = mapped_column(default=None)
    next_retry_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    delivered_at: Mapped[datetime | None] = mapped_column(default=None)
```

The `WebhookSubscription` is the customer's configuration. The `WebhookDelivery` is the per-attempt record. The `attempts` counter and `next_retry_at` field drive the retry logic.

### 3.2 The publish function

```python
async def publish_event(
    event_type: str,
    data: dict,
    *,
    event_id: str | None = None,
):
    """Publish an event to all matching subscriptions."""
    payload = make_webhook_payload(event_type, data, event_id=event_id)
    event_id = payload["id"]
    # Find all subscriptions that match
    async with async_session_maker()() as session:
        subscriptions = (await session.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.is_active == True,
                WebhookSubscription.event_types.contains([event_type]),
            )
        )).scalars().all()
        # Create a delivery record for each subscription
        for sub in subscriptions:
            delivery = WebhookDelivery(
                subscription_id=sub.id,
                event_id=event_id,
                event_type=event_type,
                payload=payload,
            )
            session.add(delivery)
        await session.commit()


# In the handler
@app.post("/users", status_code=201)
async def create_user(payload: UserCreate, uow: UnitOfWork = Depends(get_uow)):
    user = await uow.users.create(**payload.model_dump())
    await uow.commit()
    # Publish the event (does not block the request)
    await publish_event("user.signed_up", {"user_id": user.id, "email": user.email})
    return user
```

The publish function is fast (just creates delivery records). The actual HTTP send happens in the background.

### 3.3 The delivery worker

```python
async def deliver_webhook_job(ctx, delivery_id: int):
    """Background job: deliver a webhook, with retries."""
    async with async_session_maker()() as session:
        delivery = (await session.execute(
            select(WebhookDelivery).where(WebhookDelivery.id == delivery_id)
        )).scalar_one_or_none()
        if not delivery or delivery.status in ("delivered", "dead"):
            return
        
        subscription = await session.get(WebhookSubscription, delivery.subscription_id)
        if not subscription or not subscription.is_active:
            delivery.status = "dead"
            await session.commit()
            return
        
        # Attempt the delivery
        try:
            response = await send_webhook(subscription.url, subscription.secret, delivery.payload)
            delivery.attempts += 1
            delivery.last_attempt_at = func.now()
            delivery.last_response_status = response.status_code
            delivery.last_response_body = response.text[:1000]
            
            if 200 <= response.status_code < 300:
                delivery.status = "delivered"
                delivery.delivered_at = func.now()
                ctx["logger"].info(f"Webhook {delivery_id} delivered")
            else:
                # Failed; schedule retry
                delay = get_retry_delay(delivery.attempts)
                if delay is None:
                    delivery.status = "dead"
                    ctx["logger"].warning(f"Webhook {delivery_id} permanently failed")
                else:
                    delivery.next_retry_at = func.now() + timedelta(seconds=delay)
                    ctx["logger"].warning(f"Webhook {delivery_id} failed; retry in {delay}s")
        except Exception as e:
            delivery.attempts += 1
            delivery.last_attempt_at = func.now()
            delivery.last_response_status = None
            delivery.last_response_body = str(e)[:1000]
            delay = get_retry_delay(delivery.attempts)
            if delay is None:
                delivery.status = "dead"
            else:
                delivery.next_retry_at = func.now() + timedelta(seconds=delay)
        await session.commit()
```

The worker: load the delivery record, attempt the send, record the result, schedule a retry or mark as dead. The retry is implemented by a separate scheduled job that picks up due deliveries:

```python
async def retry_due_webhooks_job(ctx):
    """Find due webhooks and enqueue them for delivery."""
    now = func.now()
    async with async_session_maker()() as session:
        due = (await session.execute(
            select(WebhookDelivery).where(
                WebhookDelivery.status == "pending",
                WebhookDelivery.next_retry_at <= now,
            ).limit(100)
        )).scalars().all()
        for delivery in due:
            await arq.enqueue_job("deliver_webhook_job", delivery_id=delivery.id)
```

### 3.4 The worker settings

```python
class WorkerSettings:
    functions = [deliver_webhook_job, retry_due_webhooks_job, ...]
    redis_settings = RedisSettings()
    # The deliver_webhook_job has a long timeout (30s)
    job_timeout = 60
    # The retry_due_webhooks_job runs every minute
    cron_jobs = [
        cron(retry_due_webhooks_job, minute="*"),  # Every minute
    ]
```

The delivery job is triggered on event publish. The retry sweep is scheduled every minute.

---

## 4. The Receiver-Side Verification

### 4.1 The signature verification

```python
import hmac
import hashlib
import base64
import time


def verify_signature(
    payload: bytes,
    headers: dict,
    secret: str,
    tolerance_seconds: int = 300,
) -> bool:
    """Verify a Standard Webhooks signature."""
    msg_id = headers.get("webhook-id")
    timestamp = headers.get("webhook-timestamp")
    signature_header = headers.get("webhook-signature", "")
    
    if not all([msg_id, timestamp, signature_header]):
        return False
    
    # Check the timestamp
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > tolerance_seconds:
        return False  # Too old or too far in the future
    
    # Compute the expected signature
    signed = f"{timestamp}.".encode() + payload
    expected = base64.b64encode(
        hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    ).decode()
    
    # Compare with each signature in the header (comma-separated)
    received_signatures = signature_header.split(",")
    for sig in received_signatures:
        if sig.startswith("v1,"):
            received = sig[3:]
            if hmac.compare_digest(expected, received):
                return True
    return False
```

The verification: check the timestamp (replay protection), compute the expected signature, compare with the received signature. The `hmac.compare_digest` is constant-time to prevent timing attacks.

### 4.2 The endpoint

```python
from fastapi import APIRouter, Header, HTTPException, Request


router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/receive")
async def receive_webhook(
    request: Request,
    webhook_id: str = Header(..., alias="webhook-id"),
    webhook_timestamp: str = Header(..., alias="webhook-timestamp"),
    webhook_signature: str = Header(..., alias="webhook-signature"),
    uow: UnitOfWork = Depends(get_uow),
):
    """Receive a webhook."""
    # Read the raw body
    body = await request.body()
    # Look up the secret by the webhook-id
    async with async_session_maker()() as session:
        subscription = (await session.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.id == webhook_id,
                WebhookSubscription.is_active == True,
            )
        )).scalar_one_or_none()
        if not subscription:
            raise HTTPException(401, "Unknown webhook")
    
    # Verify the signature
    headers = {
        "webhook-id": webhook_id,
        "webhook-timestamp": webhook_timestamp,
        "webhook-signature": webhook_signature,
    }
    if not verify_signature(body, headers, subscription.secret):
        raise HTTPException(401, "Invalid signature")
    
    # Parse the payload
    payload = json.loads(body)
    # Process the event
    await process_webhook_event(payload, uow)
    
    # Return 200 quickly
    return {"status": "ok"}


async def process_webhook_event(payload: dict, uow: UnitOfWork):
    """Process the received webhook event."""
    event_id = payload.get("id")
    event_type = payload.get("type")
    data = payload.get("data")
    
    # Idempotency: check if we've already processed this event
    async with async_session_maker()() as session:
        existing = (await session.execute(
            select(ProcessedEvent).where(ProcessedEvent.event_id == event_id)
        )).scalar_one_or_none()
        if existing:
            return  # Already processed
    
    # Process the event
    if event_type == "user.signed_up":
        await handle_user_signed_up(data, uow)
    elif event_type == "payment.succeeded":
        await handle_payment_succeeded(data, uow)
    
    # Record the processed event
    async with async_session_maker()() as session:
        session.add(ProcessedEvent(event_id=event_id, event_type=event_type))
        await session.commit()
```

The receiver: verify the signature, check idempotency, process the event, return 200 quickly. Long processing should be enqueued as a job (not done inline).

### 4.3 The replay protection

The timestamp check is the replay protection: a request older than `tolerance_seconds` (default 5 minutes) is rejected. The `webhook-id` provides additional idempotency: even if the same event is sent multiple times (e.g., the sender retries), the receiver processes it only once.

The `ProcessedEvent` table records every event ID the receiver has seen. The `event_id` is the idempotency key.

---

## 5. The Production Patterns

### 5.1 The worker isolation

The webhook delivery worker should be separate from the main API worker:
- Different scaling (more workers during high webhook volume).
- Different failure isolation (a slow webhook receiver doesn't slow the API).
- Different rate limits (the API has user-facing rate limits; the worker has provider rate limits).

```yaml
# k8s deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webhook-worker
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: worker
        command: ["arq", "app.jobs.webhook_worker.WorkerSettings"]
```

### 5.2 The rate limiting

A single receiver can handle only N webhooks per second. The sender should respect the receiver's limits:

```python
# Per-subscription rate limit
async def deliver_with_rate_limit(subscription: WebhookSubscription, payload: dict):
    rate_key = f"webhook_rate:{subscription.id}:{int(time.time())}"
    count = await redis.incr(rate_key)
    await redis.expire(rate_key, 1)
    if count > subscription.max_per_second:  # e.g., 10
        # Delay the delivery by 1 second
        await asyncio.sleep(1)
    # ... send
```

Or, more elegantly, the receiver returns a `429 Too Many Requests` with a `Retry-After` header; the sender respects the header.

### 5.3 The circuit breaker

If a receiver is down, the sender should stop trying for a while (avoid hammering a dead endpoint):

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 60):
        self.failures = 0
        self.last_failure = 0
        self.threshold = failure_threshold
        self.recovery = recovery_time
        self.is_open = False
    
    async def call(self, func, *args, **kwargs):
        if self.is_open:
            if time.time() - self.last_failure > self.recovery:
                self.is_open = False  # Try again
            else:
                raise Exception("Circuit open")
        try:
            result = await func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.is_open = True
            raise
```

The circuit breaker is a per-receiver mechanism: 5 failures in a row → open the circuit → no more attempts for 60 seconds → try again.

### 5.4 The monitoring

```python
webhook_deliveries = Counter("webhook_deliveries_total", "Webhook delivery attempts", ["status"])
webhook_duration = Histogram("webhook_delivery_duration_seconds", "Webhook delivery duration")
webhook_dlq = Gauge("webhook_dead_letter_count", "Webhooks in dead-letter state")


# In the delivery worker
status = "delivered" if success else "failed"
webhook_deliveries.labels(status=status).inc()
webhook_duration.observe(time.time() - start)
```

The dashboard shows:
- Delivery success rate.
- Average delivery time.
- Dead-letter queue size.
- Per-customer delivery stats.

A growing dead-letter queue is a "your webhooks are failing" alert.

---

## 6. The Four Common Outgoing Webhook Mistakes

### 6.1 No signature

```python
# ❌ Anyone who knows the URL can send fake webhooks
await httpx.post(url, json=payload)
```

The fix: always sign with HMAC-SHA256. The receiver verifies the signature.

### 6.2 No retry

```python
# ❌ A single failure means the event is lost
await httpx.post(url, json=payload)
if response.status_code != 200:
    # log and move on
```

The fix: retry with exponential backoff and jitter. Up to 4 retries; then dead-letter.

### 6.3 No idempotency

```python
# ❌ A retry duplicates the event
# The receiver processes the event twice
```

The fix: every event has a unique `id` (the `webhook-id` header). The receiver records the ID; duplicates are ignored.

### 6.4 Synchronous delivery

```python
# ❌ The user waits 30 seconds for the webhook
@app.post("/users")
async def create_user(...):
    user = await create_user(...)
    await send_webhook(...)  # blocks the request
    return user
```

The fix: enqueue the webhook; return immediately. The webhook is delivered in the background.

---

## 7. The Test Patterns

### 7.1 Test the signature

```python
def test_sign_and_verify():
    secret = "test-secret"
    payload = {"event": "test", "data": {}}
    headers = sign_payload(secret, payload)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    assert verify_signature(body, headers, secret) is True


def test_verify_rejects_wrong_secret():
    secret = "test-secret"
    payload = {"event": "test", "data": {}}
    headers = sign_payload(secret, payload)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    assert verify_signature(body, headers, "wrong-secret") is False


def test_verify_rejects_old_timestamp():
    secret = "test-secret"
    payload = {"event": "test", "data": {}}
    old_ts = int(time.time()) - 600  # 10 minutes ago
    headers = sign_payload(secret, payload, timestamp=old_ts)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    assert verify_signature(body, headers, secret) is False
```

### 7.2 Test the retry

```python
@pytest.mark.asyncio
async def test_deliver_webhook_with_retry(arq_worker, mock_httpx):
    # First call: 500
    # Second call: 200
    mock_httpx.post.side_effect = [
        httpx.Response(500),
        httpx.Response(200),
    ]
    # Enqueue the delivery
    await arq_worker.enqueue_job("deliver_webhook_job", delivery_id=1)
    # Wait for the job to complete
    await arq_worker.run_until_empty()
    # The delivery was retried
    assert mock_httpx.post.call_count == 2
```

### 7.3 Test the receiver

```python
@pytest.mark.asyncio
async def test_receive_webhook_validates_signature(client, db_with_subscription):
    sub = db_with_subscription["subscription"]
    payload = {"event": "user.signed_up", "data": {}}
    headers = sign_payload(sub.secret, payload)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    
    response = await client.post(
        "/webhooks/receive",
        content=body,
        headers={
            "webhook-id": str(sub.id),
            **headers,
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_receive_rejects_invalid_signature(client, db_with_subscription):
    sub = db_with_subscription["subscription"]
    payload = {"event": "user.signed_up", "data": {}}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    
    response = await client.post(
        "/webhooks/receive",
        content=body,
        headers={
            "webhook-id": str(sub.id),
            "webhook-timestamp": "1700000000",
            "webhook-signature": "v1,invalid",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_receive_idempotency(client, db_with_subscription):
    sub = db_with_subscription["subscription"]
    event_id = "evt_abc123"
    payload = {"id": event_id, "event": "user.signed_up", "data": {}}
    headers = sign_payload(sub.secret, payload)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    
    # First call: 200
    response1 = await client.post("/webhooks/receive", content=body, headers={"webhook-id": str(sub.id), **headers})
    assert response1.status_code == 200
    # Second call: also 200 (idempotent; the side effect ran only once)
    response2 = await client.post("/webhooks/receive", content=body, headers={"webhook-id": str(sub.id), **headers})
    assert response2.status_code == 200
    # But the side effect (e.g., user created) ran only once
    # ... verify in the test
```

### 7.4 Test the dead-letter

```python
@pytest.mark.asyncio
async def test_dead_letter_after_max_retries(arq_worker, mock_httpx):
    mock_httpx.post.return_value = httpx.Response(500)  # Always fails
    await arq_worker.enqueue_job("deliver_webhook_job", delivery_id=1)
    # Run all retries
    for _ in range(5):
        await arq_worker.run_until_empty()
        # The retry sweep picks up the next attempt
        await asyncio.sleep(1)
    # The delivery is dead
    async with SessionLocal()() as session:
        delivery = (await session.execute(
            select(WebhookDelivery).where(WebhookDelivery.id == 1)
        )).scalar_one()
        assert delivery.status == "dead"
        assert delivery.attempts == 4
```

---

## 8. Código de Compresión

```python
"""
Compresión: Outgoing Webhooks
Covers: signing (Standard Webhooks), retry with backoff+jitter,
       dead-letter queue, circuit breaker, receiver verification.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx


# 1) The payload
def make_payload(event_type: str, data: dict, event_id: str = None) -> dict:
    import secrets
    return {
        "id": event_id or f"evt_{secrets.token_urlsafe(16)}",
        "object": "event",
        "type": event_type,
        "created_at": int(time.time()),
        "data": data,
    }


# 2) The signature
def sign_payload(secret: str, payload: dict, timestamp: int = None) -> dict:
    timestamp = timestamp or int(time.time())
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signed = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode()
    return {
        "webhook-id": payload["id"],
        "webhook-timestamp": str(timestamp),
        "webhook-signature": f"v1,{signature}",
    }


# 3) Retry delays (exponential with jitter)
RETRY_DELAYS = [(1, 30), (30, 300), (300, 1800), (1800, 7200)]


def get_retry_delay(attempt: int) -> int | None:
    if attempt > len(RETRY_DELAYS):
        return None
    min_d, max_d = RETRY_DELAYS[attempt - 1]
    return random.randint(min_d, max_d)


# 4) The send function
async def send_webhook(url: str, secret: str, payload: dict, timeout: float = 30.0) -> httpx.Response:
    headers = sign_payload(secret, payload)
    headers["Content-Type"] = "application/json"
    headers["User-Agent"] = "MyApp-Webhook/1.0"
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(url, content=body, headers=headers)


# 5) The receiver-side verification
def verify_signature(payload: bytes, headers: dict, secret: str, tolerance: int = 300) -> bool:
    msg_id = headers.get("webhook-id")
    timestamp = headers.get("webhook-timestamp")
    sig_header = headers.get("webhook-signature", "")
    if not all([msg_id, timestamp, sig_header]):
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > tolerance:
        return False
    signed = f"{timestamp}.".encode() + payload
    expected = base64.b64encode(
        hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    ).decode()
    for sig in sig_header.split(","):
        if sig.startswith("v1,"):
            received = sig[3:]
            if hmac.compare_digest(expected, received):
                return True
    return False


# 6) Circuit breaker
class CircuitBreaker:
    def __init__(self, threshold: int = 5, recovery: int = 60):
        self.failures = 0
        self.threshold = threshold
        self.recovery = recovery
        self.is_open = False
        self.opened_at = 0

    async def call(self, func, *args, **kwargs):
        if self.is_open:
            if time.time() - self.opened_at > self.recovery:
                self.is_open = False
            else:
                raise Exception("Circuit open")
        try:
            result = await func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.is_open = True
                self.opened_at = time.time()
            raise
```

---

## Key Takeaways

- **Always sign with HMAC-SHA256.** The receiver verifies the signature; the timestamp prevents replay.
- **Follow the Standard Webhooks spec** (Svix). Vendor-neutral format; standard libraries on the receiver side.
- **Retry with exponential backoff and jitter.** Up to 4 retries; then dead-letter. The jitter prevents thundering herd.
- **Idempotency is mandatory.** Every event has a unique `id` (the `webhook-id` header). The receiver processes each event only once.
- **Outgoing webhook delivery is a job.** The handler publishes; the worker delivers. The user never waits.
- **A circuit breaker per receiver** prevents hammering a dead endpoint.
- **Monitor delivery rate, success rate, and dead-letter size.** A growing dead-letter is a "your webhooks are failing" alert.
- **The receiver returns 200 quickly.** Long processing should be enqueued; the response is just "I got it".
- **The `webhook-id` is the idempotency key.** Even if the sender retries, the receiver processes only once.

## References

- [Standard Webhooks Specification](https://www.standardwebhooks.com/)
- [Stripe Webhooks Documentation](https://stripe.com/docs/webhooks)
- [GitHub Webhook Events](https://docs.github.com/en/webhooks/webhook-events-and-payload)
- [RFC 2104 — HMAC](https://www.rfc-editor.org/rfc/rfc2104)
- [Shopify Webhook Best Practices](https://shopify.dev/docs/apps/build/webhooks/best-practices)
- [Svix — Webhook Infrastructure](https://www.svix.com/)
- [Hookdeck — Webhook Delivery Platform](https://hookdeck.com/)
- [OWASP Webhook Security](https://cheatsheetseries.owasp.org/cheatsheets/Webhook_Security_Cheat_Sheet.html)
- [CloudEvents Specification](https://cloudevents.io/)
- [Webhook Reliability: At Least Once Delivery (2019)](https://www.braintreepayments.com/blog/webhook-reliability-at-least-once-delivery/)
- [Designing Robust and Predictable Webhooks (Brandur)](https://brandur.org/webhooks)
