# 🎯 05 - Capstone — Production Portkey Stack

> **The sixteenth portfolio project. End-to-end Portkey deployment for a multi-tenant LLM service: gateways + observability + PII redaction + audit logs + fallbacks. Boot the whole stack with one docker-compose.**

## 🎯 Learning Objectives
- Build a complete multi-tenant LLM gateway with Portkey self-hosted
- Configure virtual keys for tenant isolation and budgets
- Implement PII redaction and audit logs for compliance
- Set up cost-aware routing across multiple providers
- Deploy with Docker Compose for one-command local dev
- Integrate observability via Portkey's native dashboards

## Introduction

The capstone is the **synthesis** of all four notes. You will build a complete production-shaped Portkey stack:

1. **Self-hosted Portkey gateway** — OpenAI-compatible API
2. **Multi-provider routing** — OpenAI, Anthropic, Together AI, with fallbacks
3. **Per-tenant virtual keys** — budgets, scopes, audit logs
4. **PII redaction** — GDPR/HIPAA compliance out of the box
5. **Audit logs** — every request logged for SOC 2
6. **Observability dashboard** — Portkey's native UI

The architecture demonstrates Portkey replacing both LiteLLM (gateway) and LangFuse (observability). The capstone is the **sixteenth portfolio project**: enterprise-grade LLM infrastructure.

```mermaid
graph TB
    subgraph "Clients"
        WEB[Web App]
        API[API Server]
        BATCH[Batch Jobs]
    end
    
    subgraph "Portkey Gateway (self-hosted)"
        LB[Load Balancer]
        RETRY[Retry Logic]
        CB[Circuit Breaker]
        PII[PII Redaction]
        AUDIT[Audit Log]
    end
    
    subgraph "Providers"
        OA[OpenAI]
        ANT[Anthropic]
        TOG[Together AI]
    end
    
    subgraph "Storage"
        LOG[Audit Logs DB]
        DASH[Portkey UI]
    end
    
    WEB --> LB
    API --> LB
    BATCH --> LB
    
    LB --> PII
    PII --> RETRY
    RETRY --> CB
    CB --> OA
    CB --> ANT
    CB --> TOG
    
    PII --> AUDIT
    AUDIT --> LOG
    AUDIT --> DASH
```

---

## 1. Project Layout

```
portkey-stack/
├── docker-compose.yml        # Portkey gateway + Postgres + Redis + UI
├── config/
│   ├── providers.json       # Provider API keys
│   ├── virtual-keys.json    # Per-tenant virtual keys
│   └── routing.json         # Default routing config
├── app/                      # Example client app (FastAPI)
│   └── main.py
├── eval/
│   └── benchmark.py          # Compare Portkey vs direct provider calls
└── README.md
```

---

## 2. The Self-Hosted Portkey Gateway

```bash
git clone https://github.com/Portkey-AI/gateway.git
cd gateway
cp .env.example .env
```

`.env`:
```bash
# Provider API keys
OPENAI_API_KEY=sk-***
ANTHROPIC_API_KEY=sk-ant-***
TOGETHER_API_KEY=tog-***

# Portkey config
PORTKEY_LOG_LEVEL=info
PORTKEY_CACHE_TYPE=redis
REDIS_URL=redis://redis:6379

# Database for audit logs
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/portkey
```

Deploy with Docker Compose:

```yaml
# docker-compose.yml
version: "3.9"

services:
  portkey-gateway:
    image: portkeyai/gateway:latest
    ports:
      - "8787:8787"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/portkey
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=portkey
    volumes:
      - postgres_data:/var/lib/postgresql/data

  portkey-ui:
    image: portkeyai/dashboard:latest
    ports:
      - "3000:3000"
    environment:
      - PORTKEY_API_URL=http://portkey-gateway:8787

volumes:
  postgres_data:
```

```bash
docker compose up -d
```

- Gateway: http://localhost:8787 (OpenAI-compatible API)
- Dashboard: http://localhost:3000 (observability UI)
- Postgres: audit logs

---

## 3. Virtual Keys for Tenants

Create virtual keys via the dashboard or programmatically:

```python
# create_virtual_keys.py
import requests

PORTKEY_URL = "http://localhost:8787"
ADMIN_KEY = "pk-admin-***"


def create_tenant_key(tenant_id: str, tier: str, monthly_budget_usd: float):
    """Create a virtual key for a tenant."""
    
    providers_by_tier = {
        "free": ["openai"],
        "pro": ["openai", "anthropic", "together-ai"],
        "enterprise": ["openai", "anthropic", "together-ai", "azure-openai", "bedrock"],
    }
    
    response = requests.post(
        f"{PORTKEY_URL}/v1/virtual-keys",
        headers={"Authorization": ADMIN_KEY, "Content-Type": "application/json"},
        json={
            "name": f"tenant-{tenant_id}",
            "provider": providers_by_tier[tier],
            "budget": monthly_budget_usd,
            "budget_duration": "monthly",
            "metadata": {
                "tenant_id": tenant_id,
                "tier": tier,
            },
        },
    )
    return response.json()


# Create virtual keys for tenants
alice_key = create_tenant_key("alice", "pro", monthly_budget_usd=1000)
bob_key = create_tenant_key("bob", "enterprise", monthly_budget_usd=10000)
charlie_key = create_tenant_key("charlie", "free", monthly_budget_usd=100)

# Store keys in your DB / Vault
keys = {
    "alice": alice_key["virtual_key"],
    "bob": bob_key["virtual_key"],
    "charlie": charlie_key["virtual_key"],
}
```

Each virtual key has:
- A specific list of allowed providers (tier-based)
- A monthly budget
- Metadata (tenant_id, tier) for observability

---

## 4. The Default Routing Config

Configure the gateway to apply fallbacks and PII redaction by default:

```yaml
# config/routing.yaml (loaded by Portkey gateway at startup)
defaults:
  retry:
    attempts: 3
    on_status_codes: [429, 500, 502, 503, 504]
    backoff: exponential
  
  circuit_breaker:
    failure_threshold: 0.5
    window_size: 60
    cooldown_period: 300
  
  redact_pii:
    enabled: true
    entities: [email, phone, ssn, credit_card, name, address]
    method: replace
  
  audit_logging:
    enabled: true
    retention_days: 90
  
  cache:
    mode: semantic
    max_age: 3600
```

The gateway applies this config to every request unless overridden.

---

## 5. The Example Client App (FastAPI)

```python
# app/main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from portkey_ai import Portkey
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.portkey = Portkey(api_key=os.getenv("PORTKEY_ADMIN_KEY"))
    yield


app = FastAPI(title="Multi-Tenant LLM Service", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 500


class ChatResponse(BaseModel):
    response: str
    cost_usd: float
    tenant_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_tenant_id: str = Header(...),
):
    """Chat endpoint with per-tenant virtual key."""
    
    # Get the tenant's virtual key (from your DB / Vault)
    virtual_key = await get_tenant_virtual_key(x_tenant_id)
    if not virtual_key:
        raise HTTPException(403, "No virtual key for tenant")
    
    # Create a Portkey client with the tenant's virtual key
    client = Portkey(
        api_key=os.getenv("PORTKEY_ADMIN_KEY"),
        Authorization=virtual_key,
    )
    
    response = client.chat.completions.create(
        model=req.model,
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=req.max_tokens,
        config={
            "metadata": {
                "tenant_id": x_tenant_id,
                "feature": "chat",
            },
        },
    )
    
    return ChatResponse(
        response=response.choices[0].message.content,
        cost_usd=0.0,  # computed by Portkey; query dashboard for exact
        tenant_id=x_tenant_id,
    )


async def get_tenant_virtual_key(tenant_id: str) -> str | None:
    """Get the tenant's virtual key from Vault or DB."""
    # In production: from Vault / AWS Secrets Manager / your DB
    keys_db = {
        "alice": "vk-alice-***",
        "bob": "vk-bob-***",
    }
    return keys_db.get(tenant_id)
```

---

## 6. The Single Docker Compose Stack

```yaml
# docker-compose.yml (final)
version: "3.9"

services:
  portkey-gateway:
    image: portkeyai/gateway:latest
    ports:
      - "8787:8787"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/portkey
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=portkey
    volumes:
      - postgres_data:/var/lib/postgresql/data

  portkey-ui:
    image: portkeyai/dashboard:latest
    ports:
      - "3000:3000"
    environment:
      - PORTKEY_API_URL=http://portkey-gateway:8787

  app:
    build: ./app
    ports:
      - "8000:8000"
    environment:
      - PORTKEY_ADMIN_KEY=${PORTKEY_ADMIN_KEY}
    depends_on:
      - portkey-gateway

volumes:
  redis_data:
  postgres_data:
```

```bash
docker compose up -d
```

- Gateway: http://localhost:8787
- Dashboard: http://localhost:3000
- App: http://localhost:8000

---

## 7. Production Deployment Checklist

- [ ] Portkey self-hosted gateway deployed (Docker or Kubernetes)
- [ ] Postgres + Redis for audit logs and caching
- [ ] Virtual keys for each tenant with appropriate tier + budget
- [ ] PII redaction enabled (default for all requests)
- [ ] Fallbacks configured (OpenAI → Anthropic → Together)
- [ ] Retries with exponential backoff
- [ ] Circuit breaker for each provider
- [ ] Audit log export to S3 / BigQuery for long-term retention
- [ ] Budget alerts via webhook → Slack
- [ ] Data residency configured (EU providers for EU users)
- [ ] Monitoring on the gateway itself (latency, error rate)
- [ ] Quarterly compliance review of access logs
- [ ] Disaster recovery: backup virtual keys + provider credentials

---

## 🎯 Key Takeaways

- The capstone composes all four notes into a self-hosted Portkey stack.
- One `docker-compose.yml` boots gateway + Postgres + Redis + dashboard + app.
- Virtual keys enforce per-tenant budgets and provider scoping.
- PII redaction is default for all requests.
- Fallbacks + retries + circuit breakers = reliability.
- Audit logs to S3 for SOC 2 / GDPR compliance.
- The capstone is the **sixteenth portfolio project**: enterprise LLM infrastructure.

## References

- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/01 - Portkey Core - Gateway Fundamentals|Note 01 — Portkey Core]]
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/02 - Observability and Cost Tracking|Note 02 — Observability]]
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/03 - Fallbacks, Load Balancing, and Conditional Routing|Note 03 — Reliability]]
- [[06 - Large Language Models/27 - Portkey AI Gateway and Observability/04 - PII Redaction and Compliance|Note 04 — PII Compliance]]
- Portkey Gateway — [github.com/Portkey-AI/gateway](https://github.com/Portkey-AI/gateway)
- Portkey Docs — [portkey.ai/docs](https://portkey.ai/docs)
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — LiteLLM comparison
- [[06 - Large Language Models/25 - AI Compliance and Governance|AI Compliance and Governance]]
- [[09 - MLOps y Produccion/41 - Cost Engineering as Discipline - FinOps for ML|Cost Engineering / FinOps]]