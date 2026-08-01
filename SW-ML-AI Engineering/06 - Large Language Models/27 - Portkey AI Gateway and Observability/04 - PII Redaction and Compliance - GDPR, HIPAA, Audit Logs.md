# 🎯 04 - PII Redaction and Compliance — GDPR, HIPAA, Audit Logs

> **The compliance layer for LLM services. Automatic PII redaction before sending to providers. Audit logs for SOC 2 / GDPR / HIPAA. The features that make Portkey the enterprise gateway.**

## 🎯 Learning Objectives
- Use Portkey's built-in PII redaction for emails, SSNs, credit cards, etc.
- Configure custom redaction patterns for domain-specific data
- Set up audit logs for SOC 2 / GDPR compliance
- Build a request-anonymization pipeline for HIPAA
- Implement data residency controls (EU vs US processing)
- Distinguish redaction at gateway vs redaction in application
- Apply to portfolio projects (LLM Edge Gateway, healthcare)

## Introduction

Sending PII to OpenAI/Anthropic is a **regulatory violation** under GDPR, HIPAA, and most data protection laws. Portkey's PII redaction scrubs sensitive data **before** the request leaves the gateway.

The features covered in this note:

1. **Built-in PII redaction** — emails, SSNs, credit cards, phone numbers, IPs
2. **Custom patterns** — for domain-specific PII (patient IDs, account numbers, etc.)
3. **Audit logs** — every request logged for compliance
4. **Data residency** — route to EU vs US providers based on data subject location
5. **Compliance reporting** — generate audit reports for SOC 2 / GDPR

This is the feature that makes Portkey the **enterprise default** for regulated industries.

---

## 1. The Compliance Problem

When you send a chat completion to OpenAI, the following data leaves your infrastructure:

```python
{
    "messages": [
        {"role": "user", "content": "Hi, my email is alice@example.com and my SSN is 123-45-6789. Can you help with my account?"}
    ]
}
```

This message contains:
- PII (email, SSN)
- Possibly confidential business logic

The data goes to:
- OpenAI's servers (US for OpenAI; may differ for Azure OpenAI)
- Backed up to OpenAI's disaster recovery
- Used for abuse monitoring (opt-out available but tricky)
- Subject to OpenAI's data retention policies

For GDPR, HIPAA, SOC 2 compliant systems, this is **not acceptable** without controls.

---

## 2. Built-in PII Redaction

Portkey detects and redacts common PII types automatically:

```python
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",
    Authorization="vk-***",
    config={
        "redact_pii": True,
        "pii_config": {
            "entities": [
                "email",
                "phone",
                "ssn",
                "credit_card",
                "ip_address",
                "name",
                "address",
            ],
            "method": "replace",  # or "mask", "remove"
            "replacement": "[REDACTED]",
        },
    },
)

# The original message:
# "My email is alice@example.com and my SSN is 123-45-6789"
# 
# Becomes (sent to provider):
# "My email is [REDACTED] and my SSN is [REDACTED]"
```

The LLM never sees the actual PII. The user sees the redacted response.

### 2.1 Supported entities

| Entity | Example | Format |
|--------|---------|--------|
| Email | `alice@example.com` | `[\w.]+@[\w.]+` |
| Phone | `+1-415-555-1234` | Various |
| SSN | `123-45-6789` | `\d{3}-\d{2}-\d{4}` |
| Credit card | `4111-1111-1111-1111` | Luhn-validated |
| IP address | `192.168.1.1` | IPv4 and IPv6 |
| Name | `John Smith` | NER-based |
| Address | `123 Main St, SF` | NER-based |
| Email | `john@example.com` | Regex |
| URL | `https://example.com` | Regex |
| Date | `2026-07-23` | Various |
| SSN | `123-45-6789` | Regex |

### 2.2 Redaction methods

```python
# "replace" — replace with placeholder
"My email is [REDACTED]"

# "mask" — replace with asterisks
"My email is ****************"

# "remove" — delete the entire match
"My email is "  # empty

# "hash" — replace with hash (allows downstream correlation)
"My email is 5d41402abc4b2a76b9719d911017c592"
```

For most use cases, **replace** is best. For analyses where you need to track which user said what, **hash** is useful.

---

## 3. Custom PII Patterns

For domain-specific PII (e.g., patient IDs, account numbers):

```python
config = {
    "redact_pii": True,
    "pii_config": {
        "entities": [
            "email", "phone", "ssn",  # built-in
        ],
        "custom_patterns": [
            {
                "name": "patient_id",
                "regex": r"PT-\d{8}",
                "method": "replace",
                "replacement": "[PATIENT_ID]",
            },
            {
                "name": "account_number",
                "regex": r"ACC-\d{10}",
                "method": "mask",
            },
            {
                "name": "internal_secret",
                "type": "keyword",  # exact match
                "values": ["COMPANY_SECRET_1", "INTERNAL_TOKEN"],
                "method": "remove",
            },
        ],
    },
}
```

Custom patterns support:
- **Regex** — for structured PII (SSN, phone, account)
- **Keyword** — for specific names/secrets to redact
- **NER-based** — for free-form PII (names, addresses)

---

## 4. The Compliance Difference: Redact vs Not Redact

```python
# Without redaction
config = {}  # no PII config
# OpenAI receives: "alice@example.com"

# With redaction
config = {"redact_pii": True, "pii_config": {...}}
# OpenAI receives: "[REDACTED]"
```

The data that hits OpenAI is fully anonymized. The user sees the original; the LLM doesn't.

For LLM responses, you can also redact:

```python
config = {
    "redact_pii": True,
    "pii_config": {
        "input": True,    # redact before sending
        "output": True,   # redact response before returning to user
    },
}
```

Use case: the LLM might invent PII (a fake email, a hallucinated SSN). Redact output to be safe.

---

## 5. Audit Logs

Every request is logged for compliance:

```json
{
    "request_id": "req-abc-123",
    "timestamp": "2026-07-23T12:34:56Z",
    "tenant_id": "alice",
    "user_id": "user-123",
    "source_ip": "203.0.113.42",
    "model": "gpt-4o",
    "provider": "openai",
    "request_hash": "sha256:abc123...",  # hash of request content
    "response_hash": "sha256:def456...",
    "request_size_bytes": 1240,
    "response_size_bytes": 320,
    "tokens_input": 156,
    "tokens_output": 89,
    "latency_ms": 1234,
    "status": "success",
    "redacted": true,
    "metadata": {"feature": "chat"}
}
```

The audit log includes:
- **Who**: tenant_id, user_id, source_ip
- **What**: model, request/response hashes (not contents), tokens
- **When**: timestamp
- **Where**: provider, region
- **How**: latency, status

Important: Portkey stores **request_hash** (not content) for compliance — you can verify a request was made but not reproduce its content (privacy + GDPR).

### 5.1 Export audit logs

For SOC 2 / GDPR audits, export logs to your SIEM:

```python
# Portkey can stream logs to S3, BigQuery, or webhook
# Configured in the Portkey dashboard

# Or fetch programmatically
client = Portkey(api_key="pk-***")
audit_logs = client.audit_logs.list(
    tenant_id="alice",
    start_date="2026-07-01",
    end_date="2026-07-31",
)
```

Export to Splunk, Datadog, or your data warehouse for long-term retention.

---

## 6. Data Residency

For GDPR, data subjects in the EU must have their data processed in the EU (or with adequate safeguards).

```python
config = {
    "data_residency": {
        "region": "eu",  # restrict to EU providers
        "allowed_providers": [
            "azure-openai-eu",  # Azure OpenAI in EU
            "anthropic-eu",     # Anthropic EU
            "mistral-eu",       # Mistral (EU)
        ],
    },
}
```

For US data:
```python
config = {
    "data_residency": {
        "region": "us",
        "allowed_providers": ["openai", "anthropic", "google-vertex-us"],
    },
}
```

Choose routing based on user's region.

```python
def get_config(user_region: str) -> dict:
    if user_region == "eu":
        return {
            "data_residency": {"region": "eu", "allowed_providers": ["azure-openai-eu", "mistral-eu"]},
        }
    else:
        return {
            "data_residency": {"region": "us", "allowed_providers": ["openai", "anthropic"]},
        }
```

### 6.1 SOC 2 + HIPAA

For healthcare, the configuration is:

```python
config = {
    "redact_pii": True,
    "pii_config": {
        "entities": ["email", "phone", "ssn", "name", "address", "credit_card"],
        "custom_patterns": [
            {"name": "patient_id", "regex": r"PT-\d{8}"},
            {"name": "mrn", "regex": r"MRN-\d{9}"},  # medical record number
        ],
    },
    "data_residency": {
        "region": "us",
        "allowed_providers": ["azure-openai-us-gov"],  # Government cloud
    },
    "audit_logging": {
        "retention_days": 2555,  # 7 years (HIPAA)
        "export_to": "s3://my-bucket/audit-logs/",
    },
    "encryption": {
        "at_rest": "AES-256",
        "in_transit": "TLS-1.3",
    },
}
```

This config satisfies HIPAA's technical safeguards.

---

## 7. The Application-Level vs Gateway-Level Choice

**Where to redact PII:**

| Option | Where | Pros | Cons |
|--------|-------|------|------|
| **Application** | Your code | Full control, custom logic | Every dev must implement; easy to forget |
| **Gateway (Portkey)** | Centralized | Consistent, automatic, auditable | Less flexible per-request |

**Recommendation:** Use **both**:
- Application: validates that expected PII is removed before sending
- Gateway (Portkey): catches any PII the application missed

This is defense in depth.

```python
# Application-level (your code)
def prepare_messages(user_input: str) -> list[dict]:
    # Custom domain-specific redaction
    cleaned = remove_company_secrets(user_input)
    cleaned = anonymize_internal_references(cleaned)
    return [{"role": "user", "content": cleaned}]


# Gateway-level (Portkey)
config = {
    "redact_pii": True,  # catches anything your app missed
}

# Defense in depth
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=prepare_messages(user_input),
    config=config,
)
```

---

## 8. Real-World Example — Healthcare Chatbot

A healthcare chatbot that needs HIPAA compliance:

```python
from portkey_ai import Portkey

client = Portkey(
    api_key="pk-***",
    Authorization="vk-***",
    config={
        # Redact PII before sending to LLM
        "redact_pii": True,
        "pii_config": {
            "entities": ["email", "phone", "ssn", "name", "address"],
            "custom_patterns": [
                {"name": "patient_id", "regex": r"PT-\d{8}"},
                {"name": "mrn", "regex": r"MRN-\d{9}"},
                {"name": "diagnosis_code", "regex": r"ICD-10-[A-Z]\d{2}(\.\d+)?"},
            ],
            "method": "replace",
            "replacement": "[REDACTED]",
        },
        
        # Data residency — US only
        "data_residency": {
            "region": "us",
            "allowed_providers": ["azure-openai-us-gov"],
        },
        
        # Audit logging — 7-year retention
        "audit_logging": {
            "retention_days": 2555,
            "export_to": "s3://my-bucket/audit-logs/",
        },
        
        # Metadata for compliance
        "metadata": {
            "tenant_id": "patient-123",
            "user_id": "doctor-456",
            "phi_audit": True,
        },
    },
)

# The patient_id "PT-12345678" in the user's message becomes "[REDACTED]"
# before being sent to OpenAI. The LLM never sees the actual patient ID.
```

---

## 9. Antipatterns

### 9.1 Antipattern 1: Redacting at the application only

```python
# ❌ App-level only
def chat(messages):
    cleaned = my_redact(messages)
    response = openai_client.chat.completions.create(messages=cleaned)

# ✅ App + gateway
def chat(messages):
    cleaned = my_redact(messages)
    response = portkey_client.chat.completions.create(
        messages=cleaned,
        config={"redact_pii": True},  # defense in depth
    )
```

### 9.2 Antipattern 2: Hashing PII when you need to re-identify

```python
# ❌ Hashing PII when you need to know who it was
config = {"redact_pii": True, "method": "hash"}
# User asks "What's my account balance for alice@example.com?"
# LLM sees "[REDACTED: 5d41402abc4b2a76b9719d911017c592]"
# LLM can't provide account-specific answer

# ✅ Replace when re-identification matters
config = {"redact_pii": True, "method": "replace"}
# LLM sees "What's my account balance for [USER_EMAIL]?"
# LLM can answer user-specifically
```

### 9.3 Antipattern 3: Not auditing PII redaction

```python
# ❌ Trust that redaction works
config = {"redact_pii": True}

# ✅ Regularly audit: send test PII and verify
# Test: "My email is alice@example.com"
# Expected: "My email is [REDACTED]"
```

### 9.4 Antipattern 4: Storing raw request bodies in audit logs

```python
# ❌ Log full request content
"audit_log": {"messages": [...]}

# ✅ Hash + metadata only
"audit_log": {
    "request_hash": "sha256:...",
    "metadata": {"tenant_id": "...", "feature": "..."},
}
```

### 9.5 Antipattern 5: Ignoring data residency

```python
# ❌ Send EU user data to US provider
config = {"provider": "openai"}  # openai.com (US)

# ✅ Route to EU provider for EU users
config = {
    "data_residency": {"region": "eu", "allowed_providers": ["azure-openai-eu"]},
}
```

---

## 🎯 Key Takeaways

- Portkey's PII redaction scrubs sensitive data before sending to providers.
- Built-in entities: email, phone, SSN, credit card, IP, name, address.
- Custom patterns for domain-specific PII (patient IDs, account numbers).
- Audit logs capture who/what/when/where/how; export to SIEM for SOC 2 / GDPR.
- Data residency: route to EU providers for EU users; US for US users.
- Defense in depth: app-level + gateway-level redaction.
- For HIPAA: 7-year retention, US GovCloud, encryption at rest + in transit.
- Avoid redacting at app only, hashing when re-identification matters, not auditing, storing raw content, ignoring residency.

## References

- Portkey PII Redaction — [portkey.ai/docs/product/ai-gateway/redact-pii](https://portkey.ai/docs/product/ai-gateway/redact-pii)
- Portkey Audit Logs — [portkey.ai/docs/product/observability/audit-logs](https://portkey.ai/docs/product/observability/audit-logs)
- Portkey Data Residency — [portkey.ai/docs/product/ai-gateway/data-residency](https://portkey.ai/docs/product/ai-gateway/data-residency)
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/01 - Portkey Core - Gateway Fundamentals|Note 01 — Portkey Core]]
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/02 - Observability and Cost Tracking|Note 02 — Observability]]
- [[06 - Large Language Models/25 - AI Compliance and Governance|AI Compliance and Governance]] — EU AI Act
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML/02 - Cost Visibility - Per-Tenant Attribution, Chargeback, and Showback|Cost Visibility]]