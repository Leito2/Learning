# 🔔 Push Notifications and Production Patterns

## 🎯 Learning Objectives

- Implement FCM (Firebase Cloud Messaging) for Android and web
- Implement APNs (Apple Push Notification service) for iOS
- Build a device-token registry with refresh and removal
- Combine push and email for time-sensitive alerts
- Implement rate limiting, throttling, and quiet hours
- Avoid the four most common push notification mistakes

## Introduction

Email is the asynchronous channel; push notifications are the synchronous one. A security alert, a chat message, a live event update — these need to arrive in seconds, not minutes. Push notifications are the right tool.

This note covers FCM (Firebase Cloud Messaging) and APNs (Apple Push Notification service), the device token lifecycle, the rate limiting and quiet hours patterns, and the combined push + email flow for production alerts. The patterns apply to any FastAPI service that needs to reach users in real time.

---

## 1. The Push Notification Landscape

### 1.1 The platforms

| Platform | Service | Use case |
|----------|---------|----------|
| **Android** | FCM (Firebase Cloud Messaging) | Android phones, Chrome (web push) |
| **iOS** | APNs (Apple Push Notification service) | iPhones, iPads, Macs |
| **Web** | FCM (web push) or APNs (Safari) | Browser notifications |
| **Desktop** | FCM or APNs | Electron apps, native apps |

For most services, you need both FCM and APNs. The good news: the SDKs are similar; the device token management is the same pattern.

### 1.2 The device token

A device token is a unique identifier for a (device, app) pair. The OS generates it; your app sends it to your server; your server uses it to send push notifications.

```
1. The app launches
2. The OS generates a device token
3. The app sends the token to your server
4. The server stores the token
5. When you want to send a push, the server uses the token
```

The device token changes:
- When the user reinstalls the app.
- When the user clears the app data.
- When the user restores from a backup.
- (Rarely) for no apparent reason.

The server must handle token changes and invalid tokens.

### 1.3 The data model

```python
class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    platform: Mapped[str] = mapped_column(String(20))  # "ios", "android", "web"
    app_version: Mapped[str] = mapped_column(String(50))
    os_version: Mapped[str] = mapped_column(String(50))
    locale: Mapped[str] = mapped_column(String(10), default="en")
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    is_active: Mapped[bool] = mapped_column(default=True)
    last_seen_at: Mapped[datetime] = mapped_column(default=func.now())
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

The `token` is unique (one device, one token). The `is_active` flag is set to `False` when the token is invalid (a 410 response from FCM or APNs). The `last_seen_at` is updated on every push to detect stale devices.

### 1.4 The registration endpoint

```python
class DeviceRegister(BaseModel):
    token: str
    platform: str
    app_version: str
    os_version: str
    locale: str = "en"
    timezone: str = "UTC"


@router.post("/devices/register")
async def register_device(
    payload: DeviceRegister,
    current_user: User = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    """Register or update a device for push notifications."""
    # Upsert
    async with async_session_maker()() as session:
        device = (await session.execute(
            select(Device).where(Device.token == payload.token)
        )).scalar_one_or_none()
        if device:
            device.user_id = current_user.id
            device.app_version = payload.app_version
            device.os_version = payload.os_version
            device.locale = payload.locale
            device.timezone = payload.timezone
            device.is_active = True
            device.last_seen_at = func.now()
        else:
            device = Device(
                user_id=current_user.id,
                token=payload.token,
                platform=payload.platform,
                **payload.model_dump(exclude={"token", "platform"}),
            )
            session.add(device)
        await session.commit()
    return {"status": "registered"}
```

The endpoint is idempotent: re-registering the same token is a no-op. The `user_id` is updated to the current user (a token can be reassigned to a different user if the user logs out and a new user logs in on the same device).

---

## 2. FCM (Firebase Cloud Messaging)

### 2.1 The setup

```bash
pip install firebase-admin
```

The `firebase-admin` SDK is the official Python SDK. It requires a service account JSON file from the Firebase console.

```python
# app/notifications/fcm.py
import firebase_admin
from firebase_admin import credentials, messaging


# Initialize once at app startup
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
```

### 2.2 Sending a notification

```python
async def send_fcm_notification(
    device_token: str,
    title: str,
    body: str,
    data: dict | None = None,
    priority: str = "high",
) -> dict:
    """Send a push notification via FCM."""
    message = messaging.Message(
        token=device_token,
        notification=messaging.Notification(title=title, body=body),
        data={k: str(v) for k, v in (data or {}).items()},  # FCM requires string values
        android=messaging.AndroidConfig(priority=priority),
        apns=messaging.APNSConfig(  # Also for iOS via FCM
            headers={"apns-priority": "10" if priority == "high" else "5"},
        ),
    )
    try:
        message_id = messaging.send(message)
        return {"success": True, "message_id": message_id}
    except firebase_admin.exceptions.NotFoundError:
        # The token is invalid; mark the device as inactive
        return {"success": False, "error": "invalid_token"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

FCM returns a `message_id` on success. The `NotFoundError` indicates the token is invalid; the server should mark the device as inactive and remove the token.

### 2.3 The data payload

The `data` field carries custom data. FCM requires all values to be strings:

```python
data={
    "alert_id": "123",
    "user_id": "42",
    "action": "open_chat",
    "conversation_id": "abc",
}
```

The client receives this data in the push handler and navigates accordingly. The data is not displayed to the user; it's for the app's logic.

### 2.4 The high-priority vs normal-priority

FCM has two priority levels:
- `high`: delivered immediately. Uses battery. For time-sensitive (security, chat).
- `normal`: delivered when the device is awake. For non-urgent (news, social).

```python
message = messaging.Message(
    ...,
    android=messaging.AndroidConfig(priority="normal"),
)
```

Use `normal` for everything that's not time-sensitive. Reserve `high` for true emergencies.

---

## 3. APNs (Apple Push Notification Service)

### 3.1 The setup

```bash
pip install aioapns
```

`aioapns` is an async APNs client. It requires:
- An APNs auth key (`.p8` file from Apple Developer).
- Your team ID and key ID.

```python
# app/notifications/apns.py
from aioapns import APNs, NotificationRequest


# Initialize once at app startup
apns = APNs(
    key="path/to/AuthKey_XXXXXX.p8",
    key_id="XXXXXX",
    team_id="YYYYYY",
    topic="com.example.myapp",  # Your app's bundle ID
)
```

### 3.2 Sending a notification

```python
async def send_apns_notification(
    device_token: str,
    title: str,
    body: str,
    data: dict | None = None,
    priority: int = 10,  # 10 = immediate, 5 = power-conserving
) -> dict:
    """Send a push notification via APNs."""
    payload = {
        "aps": {
            "alert": {"title": title, "body": body},
            "sound": "default",
            "badge": 1,  # Increment the badge
        },
        **(data or {}),
    }
    request = NotificationRequest(
        device_token=device_token,
        message=payload,
        priority=priority,
    )
    try:
        response = await apns.send_notification(request)
        if response.is_successful:
            return {"success": True, "message_id": response.notification_id}
        else:
            # BadDeviceToken, Unregistered, etc.
            return {"success": False, "error": response.reason}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

APNs returns a `NotificationResponse` with success/failure status. The `reason` field contains the APNs error code (`BadDeviceToken`, `Unregistered`, etc.).

### 3.3 The token invalidation

When APNs returns `Unregistered` or `BadDeviceToken`, the device token is invalid:

```python
async def send_apns_notification(device_token: str, ...):
    ...
    if response.reason in ("Unregistered", "BadDeviceToken"):
        # Mark the device as inactive
        async with async_session_maker()() as session:
            device = (await session.execute(
                select(Device).where(Device.token == device_token)
            )).scalar_one_or_none()
            if device:
                device.is_active = False
                await session.commit()
        return {"success": False, "error": "invalid_token"}
    ...
```

The next push to this device is skipped (the `is_active` flag is checked first).

---

## 4. The Unified Push Service

### 4.1 The provider interface

```python
# app/notifications/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PushMessage:
    device_token: str
    title: str
    body: str
    data: dict | None = None
    priority: str = "high"  # "high" or "normal"


@dataclass
class PushResult:
    success: bool
    message_id: str | None = None
    error: str | None = None
    token_invalid: bool = False  # The token should be removed


class PushProvider(ABC):
    @abstractmethod
    async def send(self, message: PushMessage) -> PushResult: ...


# app/notifications/fcm_provider.py
class FCMPushProvider(PushProvider):
    async def send(self, message: PushMessage) -> PushResult:
        # FCM implementation
        ...


# app/notifications/apns_provider.py
class APNsPushProvider(PushProvider):
    async def send(self, message: PushMessage) -> PushResult:
        # APNs implementation
        ...
```

The handler is unaware of which provider is used; the device's `platform` field determines the provider.

### 4.2 The router

```python
# app/notifications/router.py
from app.notifications.base import PushProvider, PushMessage, PushResult
from app.notifications.fcm_provider import FCMPushProvider
from app.notifications.apns_provider import APNsPushProvider


PROVIDERS = {
    "ios": APNsPushProvider(),
    "android": FCMPushProvider(),
    "web": FCMPushProvider(),  # Web push uses FCM
}


def get_provider(platform: str) -> PushProvider:
    return PROVIDERS.get(platform, FCMPushProvider())  # default
```

### 4.3 The send function

```python
async def send_push(user_id: int, title: str, body: str, data: dict = None, priority: str = "high"):
    """Send a push notification to all of a user's active devices."""
    async with async_session_maker()() as session:
        devices = (await session.execute(
            select(Device).where(
                Device.user_id == user_id,
                Device.is_active == True,
            )
        )).scalars().all()
    
    for device in devices:
        provider = get_provider(device.platform)
        result = await provider.send(PushMessage(
            device_token=device.token,
            title=title,
            body=body,
            data=data,
            priority=priority,
        ))
        if result.token_invalid:
            device.is_active = False
        device.last_seen_at = func.now()
        await session.commit()
        # If the token was invalid, mark and move on
```

The send function iterates over all active devices for the user, picks the right provider per device, and handles invalid tokens.

---

## 5. Rate Limiting and Quiet Hours

### 5.1 The per-user rate limit

A user can have multiple devices. A push to all of them can be 5-10 notifications in a second. To prevent abuse:

```python
from collections import defaultdict
import time


class PushRateLimiter:
    """Per-user rate limiter for push notifications."""
    
    def __init__(self, max_per_hour: int = 10):
        self.max_per_hour = max_per_hour
        self.sent: dict[int, list[float]] = defaultdict(list)  # user_id -> [timestamps]
    
    def can_send(self, user_id: int) -> bool:
        now = time.time()
        hour_ago = now - 3600
        # Drop old timestamps
        self.sent[user_id] = [t for t in self.sent[user_id] if t > hour_ago]
        return len(self.sent[user_id]) < self.max_per_hour
    
    def record_send(self, user_id: int) -> None:
        self.sent[user_id].append(time.time())


rate_limiter = PushRateLimiter(max_per_hour=10)


async def send_push(user_id: int, ...):
    if not rate_limiter.can_send(user_id):
        logger.info(f"Rate limit hit for user {user_id}; skipping push")
        return
    # ... send
    rate_limiter.record_send(user_id)
```

The limiter is in-memory; for production, use Redis:

```python
class RedisPushRateLimiter:
    def __init__(self, redis, max_per_hour: int = 10):
        self.redis = redis
        self.max_per_hour = max_per_hour
    
    async def can_send(self, user_id: int) -> bool:
        key = f"push_rate:{user_id}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 3600)
        return count <= self.max_per_hour
```

The Redis version works across multiple FastAPI workers.

### 5.2 The quiet hours

A push at 3 AM is a notification that gets the user banned from the app. Quiet hours let the user opt out of non-urgent notifications during sleep:

```python
async def in_quiet_hours(device: Device) -> bool:
    """Check if the current time is in the device's quiet hours."""
    from datetime import datetime
    import pytz
    
    tz = pytz.timezone(device.timezone)
    now = datetime.now(tz)
    hour = now.hour
    # Quiet hours: 10 PM to 8 AM
    return hour >= 22 or hour < 8


async def send_push(user_id: int, title: str, body: str, data: dict = None, priority: str = "high"):
    # High-priority notifications bypass quiet hours (security alerts)
    if priority != "high":
        async with async_session_maker()() as session:
            devices = (await session.execute(
                select(Device).where(Device.user_id == user_id, Device.is_active == True)
            )).scalars().all()
        # Filter out devices in quiet hours
        devices = [d for d in devices if not await in_quiet_hours(d)]
    else:
        # All devices
        devices = ...
    # ... send
```

The high-priority notifications (security alerts) ignore quiet hours. Everything else respects them.

### 5.3 The user preferences

```python
class NotificationPreference(Base):
    __tablename__ = "notification_preferences"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    push_enabled: Mapped[bool] = mapped_column(default=True)
    email_enabled: Mapped[bool] = mapped_column(default=True)
    # Per-type preferences
    security_alerts: Mapped[bool] = mapped_column(default=True)
    marketing: Mapped[bool] = mapped_column(default=False)
    product_updates: Mapped[bool] = mapped_column(default=True)
    quiet_hours_start: Mapped[int] = mapped_column(default=22)  # 10 PM
    quiet_hours_end: Mapped[int] = mapped_column(default=8)  # 8 AM
```

The user can disable specific notification types via the API. The send function checks the preference before sending.

---

## 6. The Combined Push + Email Flow

```python
async def send_security_alert(user: User, alert: SecurityAlert):
    """Send a security alert via push (immediate) and email (with details)."""
    
    # Get the user's preferences
    prefs = await get_notification_preferences(user.id)
    if not prefs.security_alerts:
        return  # User has disabled this category
    
    # 1) Push notification (immediate, fire-and-forget)
    if prefs.push_enabled:
        await send_push_to_user(
            user_id=user.id,
            title="Security Alert",
            body=alert.short_message,
            data={"alert_id": alert.id, "type": "security"},
            priority="high",  # Bypass quiet hours
        )
    
    # 2) Email (asynchronous, with full details)
    if prefs.email_enabled:
        await arq.enqueue_job(
            "send_email_job",
            template="security_alert",
            to=user.email,
            locale=user.locale,
            idempotency_key=f"security_alert:{alert.id}:{user.id}",
            user_name=user.name,
            alert=alert.model_dump(),
        )
```

The push arrives in seconds; the email arrives a moment later with the full details. The user sees the alert immediately; they have the details when they open the email.

---

## 7. The Four Common Push Notification Mistakes

### 7.1 No token invalidation handling

```python
# ❌ Sending to invalid tokens wastes resources and pollutes metrics
async def send_push(device_token, ...):
    response = await fcm.send(...)
    # No handling of NotFoundError
```

The fix: catch the `NotFoundError`; mark the device as inactive; remove the token.

### 7.2 No quiet hours

```python
# ❌ A push at 3 AM wakes the user
async def send_push(user_id, ...):
    # No quiet hours check
```

The fix: check the device's timezone; respect quiet hours for non-urgent notifications.

### 7.3 High-priority for everything

```python
# ❌ Marketing push with high priority
async def send_marketing_push(user_id, ...):
    message = messaging.Message(..., android=messaging.AndroidConfig(priority="high"))
```

High-priority pushes wake the device. Use `normal` for non-urgent; reserve `high` for security.

### 7.4 Sending to inactive devices

```python
# ❌ Sending to a device the user logged out of 6 months ago
async def send_push(user_id, ...):
    devices = await get_devices(user_id)  # includes inactive
    for device in devices:
        await send(device.token, ...)
```

The fix: filter `is_active=True` first.

---

## 8. The Production Patterns

### 8.1 The device cleanup

A device that hasn't received a push in 30 days is probably abandoned. Mark it as inactive:

```python
async def cleanup_stale_devices():
    """Mark devices that haven't been seen in 30 days as inactive."""
    cutoff = datetime.utcnow() - timedelta(days=30)
    async with async_session_maker()() as session:
        await session.execute(
            update(Device)
            .where(Device.last_seen_at < cutoff, Device.is_active == True)
            .values(is_active=False)
        )
        await session.commit()
```

Run this daily as a background job. The `last_seen_at` is updated on every successful push.

### 8.2 The notification log

```python
class NotificationLog(Base):
    __tablename__ = "notification_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id"), nullable=True)
    channel: Mapped[str] = mapped_column(String(20))  # "push", "email"
    template: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))  # "sent", "failed", "suppressed"
    provider_message_id: Mapped[str | None] = mapped_column(default=None)
    error: Mapped[str | None] = mapped_column(default=None)
    sent_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

The log lets you see what was sent, when, and why. Critical for debugging user complaints ("I didn't get the alert").

### 8.3 The metrics

```python
pushes_sent = Counter("pushes_sent_total", "Push notifications sent", ["platform", "status"])
pushes_invalid = Counter("pushes_invalid_total", "Push notifications with invalid tokens", ["platform"])
```

The dashboard shows:
- Push send rate per platform.
- Failure rate (provider outages, network issues).
- Invalid token rate (a high rate suggests a token refresh bug).

---

## 9. The Test Patterns

### 9.1 Test the FCM send

```python
@pytest.mark.asyncio
async def test_fcm_send(mock_fcm):
    mock_fcm.send.return_value = "msg-123"
    result = await send_fcm_notification("device-token", "Title", "Body", {"alert_id": "1"})
    assert result["success"] is True
    assert result["message_id"] == "msg-123"


@pytest.mark.asyncio
async def test_fcm_invalid_token(mock_fcm):
    mock_fcm.send.side_effect = firebase_admin.exceptions.NotFoundError("not found")
    result = await send_fcm_notification("invalid-token", "Title", "Body")
    assert result["success"] is False
    assert result["token_invalid"] is True
```

### 9.2 Test the combined flow

```python
@pytest.mark.asyncio
async def test_security_alert_sends_push_and_email(arq_worker, mock_fcm, db_with_user):
    user = db_with_user["user"]
    alert = SecurityAlert(id=1, message="Suspicious login from new device", short_message="Suspicious login")
    
    await send_security_alert(user, alert)
    
    # The push was sent
    mock_fcm.send.assert_called()
    # The email was enqueued
    assert arq_worker.enqueue_job.call_count == 1
```

### 9.3 Test quiet hours

```python
@pytest.mark.asyncio
async def test_quiet_hours_block_non_urgent(device_at_3am, arq_worker):
    # Device timezone is America/New_York; current time is 3 AM
    await send_push(device_at_3am.user_id, "Marketing", "Sale!", priority="normal")
    # The push was NOT sent (quiet hours)
    assert arq_worker.enqueue_job.call_count == 0


@pytest.mark.asyncio
async def test_quiet_hours_bypass_urgent(device_at_3am, arq_worker):
    await send_push(device_at_3am.user_id, "Security Alert", "Suspicious login", priority="high")
    # The push WAS sent (security alert)
    assert arq_worker.enqueue_job.call_count == 1
```

### 9.4 Test rate limiting

```python
@pytest.mark.asyncio
async def test_rate_limit(arq_worker):
    user_id = 1
    # Send 10 pushes (the limit)
    for i in range(10):
        await send_push(user_id, "Test", "Body", priority="normal")
    # The 11th is rate-limited
    await send_push(user_id, "Test", "Body", priority="normal")
    # Only 10 were enqueued
    assert arq_worker.enqueue_job.call_count == 10
```

---

## 10. Código de Compresión

```python
"""
Compresión: Push Notifications and Production Patterns
Covers: FCM, APNs, device tokens, rate limiting, quiet hours,
       combined push+email flow.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta


# 1) The provider interface
@dataclass
class PushMsg:
    device_token: str
    title: str
    body: str
    data: dict | None = None
    priority: str = "high"


@dataclass
class PushRes:
    success: bool
    message_id: str | None = None
    error: str | None = None
    token_invalid: bool = False


class PushProvider(ABC):
    @abstractmethod
    async def send(self, message: PushMsg) -> PushRes: ...


# 2) FCM provider
class FCMPushProvider(PushProvider):
    async def send(self, message: PushMsg) -> PushRes:
        try:
            from firebase_admin import messaging
            m = messaging.Message(
                token=message.device_token,
                notification=messaging.Notification(title=message.title, body=message.body),
                data={k: str(v) for k, v in (message.data or {}).items()},
                android=messaging.AndroidConfig(priority=message.priority),
            )
            message_id = messaging.send(m)
            return PushRes(success=True, message_id=message_id)
        except Exception as e:
            token_invalid = "not found" in str(e).lower()
            return PushRes(success=False, error=str(e), token_invalid=token_invalid)


# 3) APNs provider
class APNsPushProvider(PushProvider):
    def __init__(self, apns_client):
        self.apns = apns_client

    async def send(self, message: PushMsg) -> PushRes:
        payload = {
            "aps": {
                "alert": {"title": message.title, "body": message.body},
                "sound": "default",
            },
            **(message.data or {}),
        }
        try:
            from aioapns import NotificationRequest
            request = NotificationRequest(
                device_token=message.device_token,
                message=payload,
                priority=10 if message.priority == "high" else 5,
            )
            response = await self.apns.send_notification(request)
            if response.is_successful:
                return PushRes(success=True, message_id=response.notification_id)
            token_invalid = response.reason in ("Unregistered", "BadDeviceToken")
            return PushRes(success=False, error=response.reason, token_invalid=token_invalid)
        except Exception as e:
            return PushRes(success=False, error=str(e))


# 4) The send function with rate limiting and quiet hours
class PushRateLimiter:
    def __init__(self, max_per_hour: int = 10):
        self.max_per_hour = max_per_hour
        self.sent: dict[int, list[float]] = defaultdict(list)

    def can_send(self, user_id: int) -> bool:
        now = time.time()
        hour_ago = now - 3600
        self.sent[user_id] = [t for t in self.sent[user_id] if t > hour_ago]
        return len(self.sent[user_id]) < self.max_per_hour

    def record(self, user_id: int) -> None:
        self.sent[user_id].append(time.time())


rate_limiter = PushRateLimiter()


async def send_push(user_id: int, title: str, body: str, data: dict = None, priority: str = "high"):
    if priority != "high" and not rate_limiter.can_send(user_id):
        return
    
    # Get devices
    async with async_session_maker()() as session:
        devices = (await session.execute(
            select(Device).where(Device.user_id == user_id, Device.is_active == True)
        )).scalars().all()
    
    # Filter by quiet hours (skip for high-priority)
    if priority != "high":
        devices = [d for d in devices if not _in_quiet_hours(d)]
    
    for device in devices:
        provider = get_provider(device.platform)
        result = await provider.send(PushMsg(
            device_token=device.token, title=title, body=body, data=data, priority=priority,
        ))
        if result.token_invalid:
            device.is_active = False
        device.last_seen_at = datetime.utcnow()
        await session.commit()
    
    if priority != "high":
        rate_limiter.record(user_id)


def _in_quiet_hours(device) -> bool:
    import pytz
    tz = pytz.timezone(device.timezone)
    hour = datetime.now(tz).hour
    return hour >= 22 or hour < 8


# 5) The combined push + email flow
async def send_security_alert(user, alert):
    """Security alert: push immediately, email with details."""
    await send_push(
        user.id, "Security Alert", alert.short_message,
        data={"alert_id": alert.id, "type": "security"},
        priority="high",  # Bypass quiet hours
    )
    # Email with full details (background job)
    await arq.enqueue_job(
        "send_email_job",
        template="security_alert",
        to=user.email,
        locale=user.locale,
        idempotency_key=f"security_alert:{alert.id}:{user.id}",
        user_name=user.name,
        alert=alert.model_dump(),
    )
```

---

## Key Takeaways

- **Push notifications** are the synchronous channel. Email is asynchronous; use both.
- **The device token** is the bridge between your server and the device. The server stores it; the provider uses it to deliver.
- **Token invalidation is critical.** Catch `NotFoundError` (FCM) and `Unregistered` (APNs); mark the device as inactive.
- **High-priority** is for emergencies. `normal` is for everything else. Reserve `high` for security.
- **Quiet hours** respect the user's timezone. High-priority notifications bypass quiet hours.
- **Rate limiting** prevents abuse. 10 pushes per user per hour is a reasonable default.
- **The combined flow** for time-sensitive alerts: push immediately, email with details in the background. The user sees the alert now and has the details later.
- **Stale devices** should be marked inactive. A device that hasn't received a push in 30 days is probably abandoned.
- **The notification log** is critical for debugging user complaints. Every push should be logged with the user, template, status, and provider message ID.

## References

- [Firebase Cloud Messaging](https://firebase.google.com/docs/cloud-messaging)
- [Apple Push Notification Service](https://developer.apple.com/documentation/usernotifications/)
- [aioapns — Async APNs client](https://github.com/Fatal1ty/aioapns)
- [firebase-admin — Python Firebase Admin SDK](https://firebase.google.com/docs/cloud-messaging/server)
- [APNs Provider API](https://developer.apple.com/documentation/usernotifications/sending-notification-requests-to-apns)
- [FCM HTTP v1 API](https://firebase.google.com/docs/reference/fcm/rest/v1/projects.messages)
- [Push Notification Best Practices (Android Developers)](https://developer.android.com/develop/ui/views/notifications/best-practices)
- [Push Notification Design Guidelines (Apple HIG)](https://developer.apple.com/design/human-interface-guidelines/notifications)
- [Quiet Hours UX Patterns](https://www.smashingmagazine.com/2018/03/respecting-users-time-notifications/)
- [Notification Patterns (Material Design)](https://material.io/design/communication/notifications.html)
