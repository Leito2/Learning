# 📤 Transactional Email & Templates

## 🎯 Learning Objectives

- Use SES, SendGrid, and Postmark from FastAPI
- Build a provider-agnostic email interface
- Render HTML and plain-text emails with Jinja2 templates
- Handle bounces and complaints with webhooks
- Implement i18n and unsubscribe
- Avoid the four most common email mistakes in production

## Introduction

The email layer is one of the most failure-prone parts of any web service. The naïve approach — `smtplib` to a local SMTP server — works in development but is a disaster in production: no idempotency, no rate limiting, no bounce handling, and a high chance of being marked as spam. The right approach is a transactional email provider: SES, SendGrid, Postmark, or Mailgun. They handle the deliverability, the bounces, the complaints, and the rate limits; you write the content and the API.

This note covers the provider SDKs, the template rendering with Jinja2, and the production patterns that make email reliable. The two notes (provider + templates) are combined here into a single comprehensive reference; the patterns apply to any provider.

---

## 1. The Email Provider Landscape

### 1.1 The major providers

| Provider | Pricing (per 1k emails) | Strengths | Best for |
|----------|------------------------|-----------|----------|
| **AWS SES** | $0.10 | Cheap, scales, deep AWS integration | AWS-hosted services, high volume |
| **SendGrid** | $15-20 (Essentials+) | Easy setup, marketing + transactional, analytics | Mid-size, marketing email needs |
| **Postmark** | $15 (Pro) | Best-in-class deliverability, fast support | Transactional email only |
| **Mailgun** | $15-35 | EU data residency, predictable routing | EU compliance, hybrid |
| **Resend** | $20 (Pro) | Developer-friendly, modern SDK | Startups, dev-focused teams |

For most FastAPI services, **AWS SES** is the right default: it's cheap, scales, and integrates with other AWS services. **Postmark** is the right choice if email deliverability is critical (e.g., security alerts).

### 1.2 The provider-agnostic interface

```python
# app/email/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmailMessage:
    """A provider-agnostic email message."""
    to: str
    subject: str
    html: str
    text: str  # Plain text alternative
    from_email: str | None = None
    from_name: str | None = None
    reply_to: str | None = None
    headers: dict[str, str] = None  # Custom headers
    metadata: dict[str, str] = None  # Provider-specific metadata for tracking
    idempotency_key: str | None = None  # Prevent duplicate sends
    tags: list[str] = None  # For filtering in the provider dashboard


@dataclass
class EmailResult:
    """The result of sending an email."""
    message_id: str  # Provider's ID
    accepted: bool  # Whether the provider accepted the message
    error: str | None = None


class EmailProvider(ABC):
    """The provider-agnostic interface."""
    
    @abstractmethod
    async def send(self, message: EmailMessage) -> EmailResult: ...
    
    @abstractmethod
    async def send_batch(self, messages: list[EmailMessage]) -> list[EmailResult]: ...
```

The handler is unaware of the provider. Switching from SES to Postmark is a configuration change.

---

## 2. The SES Implementation

### 2.1 The setup

```bash
pip install boto3 aiosmtplib jinja2
```

```python
# app/email/ses.py
import boto3
import asyncio
from app.email.base import EmailProvider, EmailMessage, EmailResult


class SESProvider(EmailProvider):
    def __init__(self, region: str = "us-east-1", from_email: str = "noreply@example.com"):
        self.client = boto3.client("ses", region_name=region)
        self.default_from = from_email
    
    async def send(self, message: EmailMessage) -> EmailResult:
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: self._send_sync(message)
            )
            return EmailResult(
                message_id=response["MessageId"],
                accepted=True,
            )
        except Exception as e:
            return EmailResult(message_id="", accepted=False, error=str(e))
    
    def _send_sync(self, message: EmailMessage) -> dict:
        body = {"Html": {"Charset": "UTF-8", "Data": message.html},
                "Text": {"Charset": "UTF-8", "Data": message.text}}
        return self.client.send_email(
            Source=message.from_email or self.default_from,
            Destination={"ToAddresses": [message.to]},
            Message={
                "Subject": {"Charset": "UTF-8", "Data": message.subject},
                "Body": body,
            },
            ReplyToAddresses=[message.reply_to] if message.reply_to else [],
            ConfigurationSetName=message.headers.get("ConfigurationSet") if message.headers else None,
        )
```

The `boto3` client is sync; the call runs in a thread executor. SES returns a `MessageId` for tracking.

### 2.2 The idempotency

```python
async def send(self, message: EmailMessage) -> EmailResult:
    # Use a deterministic MessageDeduplicationId
    dedup_id = message.idempotency_key or hashlib.sha256(
        f"{message.to}:{message.subject}:{message.html[:100]}".encode()
    ).hexdigest()
    ...
    return self.client.send_email(
        ...,
        MessageDeduplicationId=dedup_id,  # SES dedupes within 5 minutes
    )
```

SES deduplicates emails with the same `MessageDeduplicationId` within a 5-minute window. A retry doesn't send a duplicate.

---

## 3. The SendGrid Implementation

### 3.1 The setup

```bash
pip install sendgrid
```

```python
# app/email/sendgrid.py
import asyncio
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, From, To, Content, ReplyTo, Header
from app.email.base import EmailProvider, EmailMessage, EmailResult


class SendGridProvider(EmailProvider):
    def __init__(self, api_key: str, from_email: str = "noreply@example.com"):
        self.client = SendGridAPIClient(api_key=api_key)
        self.default_from = from_email
    
    async def send(self, message: EmailMessage) -> EmailResult:
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: self._send_sync(message))
            return EmailResult(
                message_id=response.headers.get("X-Message-Id", ""),
                accepted=response.status_code in (200, 201, 202),
            )
        except Exception as e:
            return EmailResult(message_id="", accepted=False, error=str(e))
    
    def _send_sync(self, message: EmailMessage) -> dict:
        from_email = message.from_email or self.default_from
        from_obj = From(from_email, message.from_name or "") if message.from_name else From(from_email)
        to = To(message.to)
        content_html = Content("text/html", message.html)
        content_text = Content("text/plain", message.text)
        
        mail = Mail(
            from_email=from_obj,
            to=to,
            subject=message.subject,
            content=[content_text, content_html],
        )
        if message.reply_to:
            mail.reply_to = ReplyTo(message.reply_to)
        if message.headers:
            for k, v in message.headers.items():
                mail.add_header(Header(k, v))
        if message.metadata:
            for k, v in message.metadata.items():
                mail.custom_arg = {k: v}
        
        return self.client.send(mail)
```

SendGrid's SDK is sync; the call runs in a thread executor. The response includes a `X-Message-Id` header for tracking.

---

## 4. The Template Engine: Jinja2

### 4.1 The basic template

```html
<!-- app/email/templates/welcome.html -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to {{ app_name }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .button { background: #0066cc; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Welcome, {{ user.name }}!</h1>
    <p>Thanks for signing up for {{ app_name }}. Get started by clicking the button below.</p>
    <a href="{{ action_url }}" class="button">Verify your email</a>
    <p>If the button doesn't work, copy and paste this link:</p>
    <p><a href="{{ action_url }}">{{ action_url }}</a></p>
    <hr>
    <p style="font-size: 12px; color: #999;">
        You're receiving this email because you signed up for {{ app_name }}.
        <a href="{{ unsubscribe_url }}">Unsubscribe</a>
    </p>
</body>
</html>
```

A simple HTML email. Inline styles (no external CSS); the `viewport` meta tag for mobile.

### 4.2 The plain-text alternative

```text
<!-- app/email/templates/welcome.txt -->
Welcome, {{ user.name }}!

Thanks for signing up for {{ app_name }}. Get started by clicking the link below.

{{ action_url }}

If the button doesn't work in the HTML email, copy and paste this link.

---
You're receiving this email because you signed up for {{ app_name }}.
Unsubscribe: {{ unsubscribe_url }}
```

Every email should have a plain-text alternative. Some clients (corporate, accessibility) prefer text.

### 4.3 The Jinja2 environment

```python
# app/email/templates.py
from jinja2 import Environment, FileSystemLoader, select_autoescape


env = Environment(
    loader=FileSystemLoader("app/email/templates"),
    autoescape=select_autoescape(["html", "xml"]),  # Escape HTML, not plain text
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **context) -> tuple[str, str]:
    """Render a template; return (html, text)."""
    html_template = f"{template_name}.html"
    text_template = f"{template_name}.txt"
    html = env.get_template(html_template).render(**context)
    text = env.get_template(text_template).render(**context)
    return html, text
```

The `select_autoescape` escapes HTML but not plain text. The `trim_blocks` and `lstrip_blocks` clean up whitespace.

### 4.4 The i18n pattern

For multi-language emails, the template is per locale:

```python
# app/email/templates/welcome.en.html, welcome.es.html, welcome.fr.html
def render_email(template_name: str, locale: str, **context) -> tuple[str, str]:
    """Render a template for a specific locale."""
    base = f"app/email/templates/{locale}/{template_name}"
    html = open(f"{base}.html").read()
    text = open(f"{base}.txt").read()
    return (
        env.from_string(html).render(**context),
        env.from_string(text).render(**context),
    )
```

The template files are organized by locale: `welcome.en.html`, `welcome.es.html`, etc. The `locale` is determined by the user's profile or the request's `Accept-Language` header.

### 4.5 The MJML alternative

MJML is a responsive email framework. You write the email in MJML; it compiles to HTML that works across email clients.

```html
<!-- app/email/templates/welcome.mjml -->
<mjml>
  <mj-body>
    <mj-section>
      <mj-column>
        <mj-text>Welcome, {{ user.name }}!</mj-text>
        <mj-button href="{{ action_url }}">Verify your email</mj-button>
      </mj-column>
    </mj-section>
  </mj-body>
</mjml>
```

The MJML compiles to HTML at send time:

```python
import subprocess


def render_mjml(template_path: str) -> str:
    result = subprocess.run(
        ["mjml", template_path, "-o", "/dev/stdout"],
        capture_output=True, text=True,
    )
    return result.stdout
```

MJML is the right choice for complex emails with multiple columns, buttons, and images. For simple emails, plain HTML is fine.

### 4.6 The dark mode

```html
<style>
    @media (prefers-color-scheme: dark) {
        body { background: #1a1a1a; color: #fff; }
        .button { background: #4d94ff; }
    }
</style>
```

The `@media (prefers-color-scheme: dark)` query styles the email for dark mode. Most modern email clients (Apple Mail, Gmail iOS) respect it.

---

## 5. The Email Job

```python
# app/jobs/email.py
from app.email.templates import render_template
from app.email.base import EmailMessage
from app.core.email import get_email_provider
from app.db.session import async_session_maker
from app.models.email_log import EmailLog
from sqlalchemy import select
import hashlib


async def send_email_job(
    ctx,
    template: str,
    to: str,
    locale: str = "en",
    idempotency_key: str | None = None,
    **context,
):
    """Background job: render and send an email."""
    # Render the template
    html, text = render_email(template, locale, **context)
    # Check the email log (idempotency)
    if idempotency_key:
        async with async_session_maker()() as session:
            existing = (await session.execute(
                select(EmailLog).where(EmailLog.idempotency_key == idempotency_key)
            )).scalar_one_or_none()
            if existing and existing.status == "sent":
                ctx["logger"].info(f"Email already sent: {idempotency_key}")
                return
    
    # Build the message
    message = EmailMessage(
        to=to,
        subject=context.get("subject", ""),
        html=html,
        text=text,
        idempotency_key=idempotency_key,
    )
    # Send
    provider = get_email_provider()
    result = await provider.send(message)
    
    # Log
    async with async_session_maker()() as session:
        log = EmailLog(
            to=to,
            template=template,
            subject=message.subject,
            idempotency_key=idempotency_key or hashlib.sha256(
                f"{template}:{to}:{message.subject}".encode()
            ).hexdigest(),
            provider_message_id=result.message_id,
            status="sent" if result.accepted else "failed",
            error=result.error,
        )
        session.add(log)
        await session.commit()
    
    if not result.accepted:
        ctx["logger"].error(f"Email send failed: {result.error}")
        raise  # ARQ will retry
```

The job: render the template, check idempotency, send, log, retry on failure. The `idempotency_key` prevents duplicate sends.

---

## 6. Bounce and Complaint Handling

### 6.1 The webhooks

SES, SendGrid, and Postmark all send webhooks for bounces and complaints. The webhook is an HTTP POST to your server with the bounce details.

```python
# app/api/email_webhooks.py
from fastapi import APIRouter, Request
from app.db.session import async_session_maker
from app.models.suppression_list import SuppressionList
from sqlalchemy import select


router = APIRouter(prefix="/webhooks/email", tags=["email-webhooks"])


@router.post("/ses")
async def ses_webhook(request: Request):
    """SES notification webhook for bounces and complaints."""
    # Verify the SNS signature (SES sends via SNS)
    # ... (covered in the Webhooks course)
    
    body = await request.json()
    notification_type = body.get("notificationType")
    
    if notification_type == "Bounce":
        bounce = body.get("bounce", {})
        bounced_recipients = bounce.get("bouncedRecipients", [])
        for recipient in bounced_recipients:
            email = recipient["emailAddress"]
            bounce_type = bounce.get("bounceType")  # "Permanent" or "Transient"
            # Add to suppression list
            if bounce_type == "Permanent":
                async with async_session_maker()() as session:
                    sup = SuppressionList(
                        email=email,
                        reason="bounce",
                        provider="ses",
                    )
                    session.add(sup)
                    await session.commit()
    
    elif notification_type == "Complaint":
        complaint = body.get("complaint", {})
        complained_recipients = complaint.get("complainedRecipients", [])
        for recipient in complained_recipients:
            email = recipient["emailAddress"]
            # Add to suppression list
            async with async_session_maker()() as session:
                sup = SuppressionList(email=email, reason="complaint", provider="ses")
                session.add(sup)
                await session.commit()
    
    return {"status": "ok"}


@router.post("/sendgrid")
async def sendgrid_webhook(request: Request):
    """SendGrid Event Webhook for bounces and complaints."""
    events = await request.json()
    for event in events:
        event_type = event.get("event")
        email = event.get("email")
        if event_type in ("bounce", "dropped"):
            await add_to_suppression_list(email, reason=event_type)
        elif event_type == "spamreport":
            await add_to_suppression_list(email, reason="complaint")
    return {"status": "ok"}
```

The webhook handler adds bounced or complained recipients to the suppression list. Future emails to these addresses are blocked.

### 6.2 The suppression list

```python
class SuppressionList(Base):
    __tablename__ = "suppression_list"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    reason: Mapped[str] = mapped_column(String(50))  # "bounce", "complaint", "manual"
    provider: Mapped[str] = mapped_column(String(50))  # "ses", "sendgrid"
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

Before sending an email, check the suppression list:

```python
async def is_suppressed(email: str) -> bool:
    async with async_session_maker()() as session:
        result = await session.execute(
            select(SuppressionList).where(SuppressionList.email == email)
        )
        return result.scalar_one_or_none() is not None


async def send_email_job(ctx, ...):
    if await is_suppressed(to):
        ctx["logger"].info(f"Recipient {to} is suppressed; skipping")
        return
    # ... continue
```

Sending to a suppressed address is a deliverability incident; the provider may flag the account.

---

## 7. Unsubscribe (RFC 8058)

### 7.1 The one-click unsubscribe header

```python
import base64


def one_click_unsubscribe(unsubscribe_url: str) -> str:
    """Generate the List-Unsubscribe header for one-click unsubscribe."""
    # Base64-encode the URL
    encoded = base64.urlsafe_b64encode(unsubscribe_url.encode()).decode()
    return f"<{unsubscribe_url}>, <mailto:unsubscribe@example.com?subject=unsubscribe>"


message.headers["List-Unsubscribe"] = one_click_unsubscribe(unsubscribe_url)
message.headers["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
```

The `List-Unsubscribe` and `List-Unsubscribe-Post` headers are required by RFC 8058. Email clients (Gmail, Apple Mail) show an "unsubscribe" button.

### 7.2 The unsubscribe endpoint

```python
@router.post("/unsubscribe")
async def unsubscribe(
    email: str,
    uow: UnitOfWork = Depends(get_uow),
):
    """One-click unsubscribe endpoint."""
    # Add to suppression list
    user = await uow.users.get_by(email=email)
    if user:
        user.subscribed = False
    await uow.commit()
    return {"status": "unsubscribed"}
```

The endpoint is idempotent: a user can click unsubscribe multiple times. The action is reversible: a "resubscribe" endpoint sets the flag back.

---

## 8. The Four Common Email Mistakes

### 8.1 Sending inline in the request

```python
# ❌ The user waits 5 seconds for the welcome email
@app.post("/signup")
async def signup(payload: UserCreate):
    user = await create_user(payload)
    await send_welcome_email(user.email)  # blocks the request
    return user
```

The fix: enqueue the email; return immediately. The email is sent in the background.

### 8.2 No idempotency

```python
# ❌ A retry sends the email twice
async def send_welcome_email(user: User):
    # ... send the email
    # The network fails; the client retries; the email is sent twice
```

The fix: an `idempotency_key` based on `(template, recipient, subject)`. The provider deduplicates.

### 8.3 No bounce handling

```python
# ❌ A bouncing address is emailed forever
async def send_email_job(ctx, ...):
    # Send the email
    # The address bounces; we never know
    # We send the next email; it bounces again
```

The fix: webhook handlers for bounces and complaints; add to the suppression list.

### 8.4 HTML-only emails

```python
# ❌ A plain-text-only client sees garbage
async def send_email(...):
    # ... build html
    message.text = ""  # no plain-text alternative
```

The fix: always provide a plain-text alternative. Some clients (corporate, accessibility tools) prefer text.

---

## 9. Push Notifications

### 9.1 The FCM (Android) and APNs (iOS) basics

Push notifications are the "real-time" channel: a security alert, a chat message, a live update. Email is for asynchronous communication; push is for synchronous.

```python
# app/notifications/push.py
import httpx


async def send_fcm_notification(
    device_token: str,
    title: str,
    body: str,
    data: dict = None,
):
    """Send a push notification via FCM (Firebase Cloud Messaging)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://fcm.googleapis.com/fcm/send",
            headers={
                "Authorization": f"key={FCM_SERVER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "to": device_token,
                "notification": {"title": title, "body": body},
                "data": data or {},
                "priority": "high",
            },
        )
    return response.json()


async def send_apns_notification(
    device_token: str,
    title: str,
    body: str,
    data: dict = None,
):
    """Send a push notification via APNs (Apple Push Notification service)."""
    # APNs requires JWT-based auth and HTTP/2; use a library
    from aioapns import APNs
    apns = APNs(...)
    return await apns.send(device_token, {
        "aps": {"alert": {"title": title, "body": body}, "sound": "default"},
        **data or {},
    })
```

The SDKs are simple. The hard part is the device token lifecycle: registering, refreshing, removing invalid tokens.

### 9.2 Push vs email

| Channel | Latency | Best for |
|---------|---------|----------|
| Push | Seconds | Time-sensitive alerts (security, chat) |
| Email | Minutes | Asynchronous (welcome, receipts, digests) |

Use push for things that must arrive now. Use email for things that can wait a few minutes.

### 9.3 The combined flow

```python
async def send_security_alert(user: User, alert: SecurityAlert):
    """Send a security alert via push (immediate) and email (asynchronous)."""
    # Push (immediate, fire-and-forget)
    if user.device_token:
        await send_fcm_notification(
            device_token=user.device_token,
            title="Security Alert",
            body=alert.message,
            data={"alert_id": alert.id},
        )
    # Email (asynchronous, with details)
    await arq.enqueue_job(
        "send_email_job",
        template="security_alert",
        to=user.email,
        locale=user.locale,
        alert=alert,
    )
```

The push arrives in seconds; the email arrives a moment later with the full details. The user sees the alert immediately; they have the details when they open the email.

---

## 10. The Production Configuration

### 10.1 The provider selection

```python
def get_email_provider():
    if settings.EMAIL_PROVIDER == "ses":
        return SESProvider(region=settings.AWS_REGION, from_email=settings.EMAIL_FROM)
    elif settings.EMAIL_PROVIDER == "sendgrid":
        return SendGridProvider(api_key=settings.SENDGRID_API_KEY, from_email=settings.EMAIL_FROM)
    elif settings.EMAIL_PROVIDER == "postmark":
        return PostmarkProvider(token=settings.POSTMARK_TOKEN, from_email=settings.EMAIL_FROM)
    else:
        raise ValueError(f"Unknown provider: {settings.EMAIL_PROVIDER}")
```

The provider is a configuration choice. The handler code is the same.

### 10.2 The deliverability setup

For the emails to land in the inbox (not the spam folder):

1. **SPF record** (RFC 7208): the DNS TXT record that lists the IPs authorized to send email for the domain.
2. **DKIM signing** (RFC 6376): a cryptographic signature on every email, verified by the receiver.
3. **DMARC policy** (RFC 7489): the receiver's policy for handling failures (none, quarantine, reject).

Most email providers (SES, SendGrid, Postmark) handle DKIM signing automatically. SPF and DMARC require DNS changes.

### 10.3 The monitoring

```python
emails_sent = Counter("emails_sent_total", "Total emails sent", ["template", "status"])
bounces = Counter("email_bounces_total", "Email bounces", ["type"])
complaints = Counter("email_complaints_total", "Email complaints")


# In the job
emails_sent.labels(template=template, status="success").inc()
emails_sent.labels(template=template, status="failure").inc()


# In the webhook handler
bounces.labels(type=bounce_type).inc()
```

The dashboard shows:
- Email send rate (per template).
- Failure rate (provider outages, network issues).
- Bounce rate (should be <2%).
- Complaint rate (should be <0.1%).

A high bounce rate or complaint rate is a deliverability incident; the account may be flagged by the provider.

---

## 11. The Test Patterns

### 11.1 Test the email send

```python
@pytest.mark.asyncio
async def test_send_email_job(arq_worker, db_with_user, mock_email_provider):
    """Run the job in eager mode and verify the email was sent."""
    user = db_with_user["user"]
    
    # Enqueue the job
    await arq_worker.enqueue_job(
        "send_email_job",
        template="welcome",
        to=user.email,
        locale="en",
        idempotency_key="welcome:1",
        user_name=user.name,
    )
    # Wait for the job to complete
    await arq_worker.run_until_empty()
    
    # Verify the email was sent
    assert mock_email_provider.sent_count == 1
    last_email = mock_email_provider.last_email
    assert last_email.to == user.email
    assert user.name in last_email.html
```

### 11.2 Test the bounce handling

```python
@pytest.mark.asyncio
async def test_ses_bounce_webhook_adds_to_suppression(client):
    # Send a bounce notification
    body = {
        "notificationType": "Bounce",
        "bounce": {
            "bounceType": "Permanent",
            "bouncedRecipients": [{"emailAddress": "bounced@example.com"}],
        },
    }
    response = await client.post("/webhooks/email/ses", json=body)
    assert response.status_code == 200
    
    # The email is in the suppression list
    async with SessionLocal()() as session:
        sup = (await session.execute(
            select(SuppressionList).where(SuppressionList.email == "bounced@example.com")
        )).scalar_one()
        assert sup.reason == "bounce"
```

### 11.3 Test the idempotency

```python
@pytest.mark.asyncio
async def test_email_idempotency(arq_worker, mock_email_provider):
    # Enqueue the same email twice with the same idempotency key
    await arq_worker.enqueue_job(
        "send_email_job",
        template="welcome",
        to="alice@example.com",
        idempotency_key="welcome:1:alice",
        user_name="Alice",
    )
    await arq_worker.enqueue_job(
        "send_email_job",
        template="welcome",
        to="alice@example.com",
        idempotency_key="welcome:1:alice",
        user_name="Alice",
    )
    await arq_worker.run_until_empty()
    
    # Only one email was sent
    assert mock_email_provider.sent_count == 1
```

### 11.4 Test the template rendering

```python
def test_template_renders():
    html, text = render_email("welcome", "en", user_name="Alice", app_name="MyApp", action_url="https://...")
    assert "Alice" in html
    assert "MyApp" in html
    assert "https://" in html
    # Plain text alternative
    assert "Alice" in text
    assert "https://" in text
    # No HTML in plain text
    assert "<html>" not in text
```

### 11.5 Test the unsubscribe

```python
@pytest.mark.asyncio
async def test_one_click_unsubscribe(client, db_with_user):
    user = db_with_user["user"]
    response = await client.post(f"/unsubscribe?email={user.email}")
    assert response.status_code == 200
    
    # The user is unsubscribed
    async with SessionLocal()() as session:
        user = (await session.execute(
            select(User).where(User.id == db_with_user["user"].id)
        )).scalar_one()
        assert user.subscribed is False
```

---

## 12. Código de Compresión

```python
"""
Compresión: Email & Templates
Covers: SES/SendGrid providers, Jinja2 templates, bounce handling,
       unsubscribe, push notifications.
"""
import asyncio
import base64
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.message import EmailMessage as PyEmailMessage
from typing import Any

import boto3
from jinja2 import Environment, FileSystemLoader, select_autoescape


# 1) Provider-agnostic interface
@dataclass
class EmailMsg:
    to: str
    subject: str
    html: str
    text: str
    from_email: str | None = None
    from_name: str | None = None
    reply_to: str | None = None
    headers: dict[str, str] = None
    metadata: dict[str, str] = None
    idempotency_key: str | None = None
    tags: list[str] = None


@dataclass
class EmailResult:
    message_id: str
    accepted: bool
    error: str | None = None


class EmailProvider(ABC):
    @abstractmethod
    async def send(self, message: EmailMsg) -> EmailResult: ...


# 2) SES implementation
class SESProvider(EmailProvider):
    def __init__(self, region="us-east-1", from_email="noreply@example.com"):
        self.client = boto3.client("ses", region_name=region)
        self.default_from = from_email

    async def send(self, message: EmailMsg) -> EmailResult:
        loop = asyncio.get_event_loop()
        try:
            dedup = message.idempotency_key or hashlib.sha256(
                f"{message.to}:{message.subject}".encode()
            ).hexdigest()
            response = await loop.run_in_executor(
                None, lambda: self.client.send_email(
                    Source=message.from_email or self.default_from,
                    Destination={"ToAddresses": [message.to]},
                    Message={
                        "Subject": {"Charset": "UTF-8", "Data": message.subject},
                        "Body": {
                            "Html": {"Charset": "UTF-8", "Data": message.html},
                            "Text": {"Charset": "UTF-8", "Data": message.text},
                        },
                    },
                    MessageDeduplicationId=dedup,
                )
            )
            return EmailResult(message_id=response["MessageId"], accepted=True)
        except Exception as e:
            return EmailResult(message_id="", accepted=False, error=str(e))


# 3) Template environment
env = Environment(
    loader=FileSystemLoader("app/email/templates"),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_email(template: str, locale: str, **context) -> tuple[str, str]:
    """Render HTML and text versions."""
    base = f"{locale}/{template}"
    return env.get_template(f"{base}.html").render(**context), env.get_template(f"{base}.txt").render(**context)


# 4) Unsubscribe header (RFC 8058)
def list_unsubscribe_header(url: str) -> str:
    return f"<{url}>, <mailto:unsubscribe@example.com?subject=unsubscribe>"


# 5) The email job
async def send_email_job(ctx, template: str, to: str, locale: str = "en", idempotency_key: str = None, **context):
    from app.db.session import async_session_maker
    from app.models.email_log import EmailLog
    from sqlalchemy import select

    html, text = render_email(template, locale, **context)
    msg = EmailMsg(
        to=to, subject=context.get("subject", ""), html=html, text=text,
        idempotency_key=idempotency_key,
        headers={"List-Unsubscribe": list_unsubscribe_header(f"https://example.com/unsubscribe?email={to}")},
    )
    provider = SESProvider()
    result = await provider.send(msg)

    # Log
    async with async_session_maker()() as session:
        log = EmailLog(
            to=to, template=template, subject=msg.subject,
            idempotency_key=idempotency_key or hashlib.sha256(
                f"{template}:{to}:{msg.subject}".encode()
            ).hexdigest(),
            status="sent" if result.accepted else "failed",
            error=result.error,
        )
        session.add(log)
        await session.commit()
    if not result.accepted:
        raise RuntimeError(f"Email send failed: {result.error}")


# 6) Bounce webhook
async def handle_ses_bounce(body: dict):
    from app.db.session import async_session_maker
    from app.models.suppression_list import SuppressionList
    if body.get("notificationType") != "Bounce":
        return
    bounce = body.get("bounce", {})
    if bounce.get("bounceType") == "Permanent":
        for recipient in bounce.get("bouncedRecipients", []):
            email = recipient["emailAddress"]
            async with async_session_maker()() as session:
                sup = SuppressionList(email=email, reason="bounce", provider="ses")
                session.add(sup)
                await session.commit()
```

---

## Key Takeaways

- **Use a transactional email provider** (SES, SendGrid, Postmark), not raw SMTP. They handle deliverability, bounces, and rate limits.
- **The provider-agnostic interface** isolates the handler from the provider. Switching is a configuration change.
- **Email is background work.** The handler enqueues; the job renders and sends. The user sees the upload as instant.
- **Idempotency is mandatory.** A retry must not send a duplicate. Use `idempotency_key` based on `(template, recipient, subject)`.
- **Bounce and complaint handling** is the production pattern. Webhook handlers add to the suppression list; the sender checks the list before sending.
- **One-click unsubscribe** (RFC 8058) is required for compliance. Add the `List-Unsubscribe` header; provide a POST endpoint.
- **Plain-text alternative** is required. Some clients (corporate, accessibility) prefer text.
- **Templates with Jinja2** for HTML and text. MJML for complex responsive emails. i18n with per-locale templates.
- **Push notifications** for time-sensitive alerts; email for asynchronous. The combined flow: push immediately, email with details.
- **DKIM + SPF + DMARC** for deliverability. The provider handles DKIM; you configure SPF and DMARC in DNS.

## References

- [AWS SES Documentation](https://docs.aws.amazon.com/ses/)
- [SendGrid Python SDK](https://github.com/sendgrid/sendgrid-python)
- [Postmark Python Library](https://postmarkapp.com/developer/integration/official-libraries#python)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [MJML — Responsive email framework](https://mjml.io/)
- [Firebase Cloud Messaging](https://firebase.google.com/docs/cloud-messaging)
- [Apple Push Notification Service](https://developer.apple.com/documentation/usernotifications/)
- [RFC 8058 — One-Click Unsubscribe](https://www.rfc-editor.org/rfc/rfc8058)
- [RFC 6376 — DKIM Signing](https://www.rfc-editor.org/rfc/rfc6376)
- [RFC 7208 — SPF](https://www.rfc-editor.org/rfc/rfc7208)
- [RFC 7489 — DMARC](https://www.rfc-editor.org/rfc/rfc7489)
- [Spamhaus — Anti-spam standards](https://www.spamhaus.org/)
- [Litmus — Email previews and testing](https://www.litmus.com/)
- [Really Good Emails — Email design inspiration](https://reallygoodemails.com/)
- [HTML Email Boilerplate](https://www.htmlemailboilerplate.com/)
