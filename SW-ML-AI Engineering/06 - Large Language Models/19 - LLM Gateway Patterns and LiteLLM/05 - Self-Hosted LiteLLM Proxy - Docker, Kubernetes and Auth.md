# 🏷️ Self-Hosted LiteLLM Proxy: Docker, Kubernetes and Auth

## 🎯 Learning Objectives
- Understand the `litellm-proxy` server architecture (FastAPI + Postgres + Redis)
- Write a production-grade `config.yaml` covering model list, SSO, spend limits, and tracing
- Deploy the proxy with Docker Compose for local development
- Deploy with the official Helm chart on Kubernetes for production
- Integrate SSO (Google, Azure AD, Okta) and JWT-based authentication
- Operate the Postgres logging database and the Redis distributed rate limiter

## Introduction

The library-level surface from [[02 - LiteLLM Core - Unified Multi-Provider Interface]] through [[04 - Observability, Cost Tracking and Rate Limiting]] is a single-process abstraction. The moment you have multiple applications, multiple teams, and a need for centralized control — virtual keys, spend limits, SSO, audit logs, a UI — the right answer is the **proxy server**: a standalone process that applications point at, and that internally routes to 100+ providers. The `litellm-proxy` ships in the same package (`pip install 'litellm[proxy]'`) and is the dominant production pattern for teams serving more than one internal customer.

This note covers the architecture, the `config.yaml` schema, the SSO integrations, the Postgres logging database, the Redis rate limiter, the Docker Compose local-dev setup, and the Helm chart for Kubernetes. By the end, you should be able to take a 10-engineer team running on the library and migrate them to a self-hosted proxy with SSO, virtual keys, and a spend dashboard in a single afternoon.

---

## 1. The Proxy Architecture

The proxy is a FastAPI server that exposes an OpenAI-compatible HTTP API on the front and a private admin API on the back. Requests enter at `/v1/chat/completions`, are validated against the virtual key in the `Authorization` header, are routed through the `litellm.Router` from [[03 - Routing, Fallback and Retry Strategies]], and exit to the appropriate provider. Side effects — callbacks, spend tracking, audit logs — fire on the way out.

```mermaid
flowchart TB
    Client[Application / SDK] -->|HTTPS + Bearer key| Proxy[litellm-proxy :4000]
    Proxy --> Auth[Auth + Key Validation]
    Auth --> RL[Redis Rate Limiter]
    RL --> Router[litellm.Router]
    Router --> OAI[OpenAI]
    Router --> Ant[Anthropic]
    Router --> Bed[Bedrock]
    Router --> VLLM[vLLM / SGLang]
    Router --> Groq[Groq]
    Router --> Cost[Cost Engine]
    Cost --> PG[(Postgres: spend + audit)]
    Cost --> Langf[Langfuse / Phoenix]
    Cost --> S3[S3 / GCS archive]
    Admin[Admin UI / API] -->|Master key| Proxy
    Admin --> SSO[SSO Providers]
    SSO --> Google[Google]
    SSO --> Azure[Azure AD]
    SSO --> Okta[Okta]
```

Three roles are explicit:

- **End users** (applications) hit the OpenAI-compatible endpoint with a virtual key
- **Platform admins** hit the admin API with the master key to create keys, set budgets, view spend
- **SSO providers** (Google, Azure AD, Okta) issue identity tokens that the proxy validates before minting a virtual key

The Postgres database stores: virtual keys, teams, users, audit logs, per-call spend records, and the LiteLLM internal state. The Redis instance stores: rate-limit counters, the cooldown state from [[03 - Routing, Fallback and Retry Strategies]], and the semantic cache (when enabled).

---

## 2. The `config.yaml` Schema

The single most important file. It declares the model list, the routing/fallback config, the callbacks, the SSO setup, and the rate limits. The schema is large; a production example covers most of the surface:

```yaml
# === MODEL LIST ===
model_list:
  # Premium tier — GPT-4o
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 20000
      tpm: 10000000

  # Standard tier — Claude 3.5 Sonnet
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      rpm: 15000
      tpm: 8000000

  # Budget tier — Llama 3.3 70B on Groq + Fireworks for redundancy
  - model_name: llama-3.3-70b
    litellm_params:
      model: groq/llama-3.3-70b-versatile
      api_key: os.environ/GROQ_API_KEY
      rpm: 30000
      tpm: 15000000
  - model_name: llama-3.3-70b
    litellm_params:
      model: fireworks/llama-3.3-70b-instruct
      api_key: os.environ/FIREWORKS_API_KEY
      rpm: 30000
      tpm: 15000000

  # Self-hosted vLLM
  - model_name: llama-3.3-70b-self-hosted
    litellm_params:
      model: openai/llama-3.3-70b-instruct
      api_base: http://vllm-server:8000/v1
      api_key: "not-needed"
      rpm: 1000
      tpm: 2000000

  # Embeddings
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY

  # Gemini 2.5 Pro for long-context
  - model_name: gemini-2.5-pro
    litellm_params:
      model: gemini/gemini-2.5-pro
      api_key: os.environ/GOOGLE_API_KEY

# === ROUTING + FALLBACKS ===
router_settings:
  routing_strategy: usage-based-routing-v2
  num_retries: 2
  cooldown_time: 60
  timeout: 30
  redis_host: os.environ/REDIS_HOST
  redis_port: 6379

fallbacks:
  - gpt-4o: [claude-3-5-sonnet, llama-3.3-70b]
  - claude-3-5-sonnet: [llama-3.3-70b]

context_window_fallbacks:
  - gpt-4o: [gemini-2.5-pro]
  - claude-3-5-sonnet: [gemini-2.5-pro]

# === LITELLM SETTINGS ===
litellm_settings:
  drop_params: true
  set_verbose: false
  success_callback: ["langfuse", "s3"]
  failure_callback: ["langfuse", "s3"]
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
    ttl: 600
    supported_call_types: ["acompletion", "aembedding"]

# === GENERAL SETTINGS ===
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  disable_spend_logs: false
  proxy_budget_rescheduler_min_time: 60
  proxy_budget_rescheduler_max_time: 120
  alert_to_webhook_url: os.environ/SLACK_WEBHOOK_URL

# === SSO + AUTH ===
general_settings:
  ui_username: admin
  ui_password: os.environ/LITELLM_UI_PASSWORD
  proxy_admin_email: os.environ/ADMIN_EMAIL

# SSO via Google
general_settings:
  google_client_id: os.environ/GOOGLE_CLIENT_ID
  google_client_secret: os.environ/GOOGLE_CLIENT_SECRET

# JWT (for programmatic use)
general_settings:
  jwt_secret: os.environ/JWT_SECRET

# === RATE LIMITS ===
# Applied per virtual key, see section 4

# === ALERTS ===
general_settings:
  alerting: [slack]
  alerting_threshold: 0.9    # alert at 90% of budget
```

The `os.environ/...` syntax tells the proxy to resolve the value from the environment at startup. This is the right pattern for secrets — never commit real keys.

---

## 3. Docker Compose for Local Development

The fastest path from zero to a working proxy:

```yaml
# docker-compose.yml
version: "3.9"

services:
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    volumes:
      - ./config.yaml:/app/config.yaml
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - DATABASE_URL=postgresql://litellm:litellm@postgres:5432/litellm
      - REDIS_HOST=redis
      - LITELLM_MASTER_KEY=sk-1234
      - LITELLM_UI_PASSWORD=admin123
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=litellm
      - POSTGRES_PASSWORD=litellm
      - POSTGRES_DB=litellm
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U litellm"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  pgdata:
```

Bring it up:

```bash
docker compose up -d
# Wait 10 seconds for migrations
docker compose logs -f litellm
```

Test it:

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

The response is a normal OpenAI chat-completion JSON. The admin UI is at `http://localhost:4000/ui` with the master key as the password.

---

## 4. Virtual Keys, Teams, and Per-Key Policies

The admin API is the interface to the multi-tenant control plane. The most common operations:

```bash
# Create a team
curl -X POST http://localhost:4000/team/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "Data Science",
    "max_budget": 50000,
    "budget_duration": "30d",
    "models": ["gpt-4o", "claude-3-5-sonnet", "llama-3.3-70b"]
  }'

# Issue a key for that team
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team_data_science_abc",
    "user_id": "alice@company.com",
    "models": ["gpt-4o", "claude-3-5-sonnet", "llama-3.3-70b"],
    "max_budget": 5000,
    "budget_duration": "30d",
    "rpm": 200,
    "tpm": 100000
  }'
# Returns: {"key": "sk-litellm-virtual-XYZ...", ...}
```

The applications now use `sk-litellm-virtual-XYZ...` as their API key. The proxy enforces:

- **Model access**: this key can only call `gpt-4o`, `claude-3-5-sonnet`, `llama-3.3-70b`. Any other model returns 403.
- **Rate limits**: 200 RPM, 100K TPM. Exceeded → 429.
- **Spend cap**: $5K in 30 days. Exceeded → 403.
- **Audit trail**: every call is logged with `user_id`, `team_id`, `model`, tokens, cost.

The full set of admin endpoints (`/key/generate`, `/key/update`, `/key/delete`, `/key/info`, `/team/new`, `/team/update`, `/spend/logs`, `/spend/users`, `/spend/keys`) gives the platform team a complete API for managing the multi-tenant surface.

---

## 5. SSO: Google, Azure AD, Okta, JWT

For human users, the proxy supports browser-based SSO. The flow:

1. User visits `https://proxy.company.com/ui`
2. Proxy redirects to `https://accounts.google.com/o/oauth2/...`
3. User authenticates; Google redirects back to the proxy with an `id_token`
4. Proxy validates the `id_token` against the configured `google_client_id`
5. Proxy creates (or updates) a user record and a virtual key bound to that user
6. User is logged in

Configuration is the three fields in `config.yaml` from section 2: `google_client_id`, `google_client_secret`, and the proxy's authorized redirect URI (e.g., `https://proxy.company.com/sso/callback`). The same pattern applies to Azure AD and Okta with provider-specific field names.

For service-to-service auth (typical for backend applications, not human users), the standard is **JWT**:

```yaml
general_settings:
  jwt_secret: os.environ/JWT_SECRET
```

The application generates a JWT signed with `JWT_SECRET`, sends it as `Authorization: Bearer <jwt>`, and the proxy validates the signature and extracts `user_id` and `team_id` from the claims. This is the right answer for platform engineering: no shared secrets, fine-grained per-service identities, and tokens that expire automatically.

---

## 6. The Postgres Logging Database

The proxy writes to Postgres on every call. The schema is owned by LiteLLM (migrations are auto-run on startup) and includes these tables:

| Table | Purpose |
|-------|---------|
| `LiteLLM_SpendLogs` | Per-call: timestamp, user, team, model, tokens, cost, latency |
| `LiteLLM_VerificationToken` | Virtual keys: hash, user, team, budget, models, rpm, tpm |
| `LiteLLM_TeamTable` | Teams: id, alias, max_budget, models |
| `LiteLLM_UserTable` | Users: id, email, role, spend |
| `LiteLLM_Config` | Audit log of config.yaml changes |
| `LiteLLM_SlackAlerting` | Alert history |

Querying spend per team per day:

```sql
SELECT
    DATE(start_time) AS day,
    SUM(spend) AS total_usd,
    COUNT(*) AS calls
FROM "LiteLLM_SpendLogs"
WHERE team_id = 'team_data_science_abc'
  AND start_time >= NOW() - INTERVAL '30 days'
GROUP BY DATE(start_time)
ORDER BY day;
```

The `spend` column is the pre-computed cost, populated by the cost engine from [[04 - Observability, Cost Tracking and Rate Limiting]]. The proxy's own `/spend/logs` endpoint exposes the same data through HTTP.

⚠️ The Postgres writes are the *critical path* for the budget enforcement. If the DB is down, the proxy will *fail closed* (refuse requests) rather than allow unbounded spend. This is the right default but means DB availability is now an LLM-availability concern. A managed Postgres (RDS, Cloud SQL, Neon) with multi-AZ is the standard answer.

---

## 7. Redis for Distributed Rate Limiting

The library-level rate limits from [[04 - Observability, Cost Tracking and Rate Limiting]] are per-process. The proxy's limits are cluster-wide because they are coordinated through Redis.

The implementation is a sliding-window counter: each key has a `rpm` field and a Redis hash keyed by `rl:<key>:<minute_bucket>`. On every call, the proxy increments the counter and checks the threshold. Redis is the right tool because the operations are O(1) and the network latency is sub-millisecond in the same region.

The proxy also uses Redis for the **semantic cache** (when enabled) and the **cooldown state** from [[03 - Routing, Fallback and Retry Strategies]]. A single Redis instance is the standard deployment; high-availability setups use Redis Sentinel or a managed Redis (ElastiCache, MemoryStore, Upstash).

---

## 8. Kubernetes Deployment with the Helm Chart

For production, the official Helm chart is the path of least resistance:

```bash
helm repo add litellm https://BerriAI.github.io/litellm-helm
helm install litellm litellm/litellm \
  --namespace litellm --create-namespace \
  --set image.tag=main-latest \
  --set secrets.create=true \
  --set secrets.data.OPENAI_API_KEY=$OPENAI_API_KEY \
  --set secrets.data.ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --set configMap.create=true \
  --set configMap.data.configYaml="$(cat config.yaml)"
```

The chart deploys:

- A `Deployment` of 3+ LiteLLM proxy replicas behind a `Service`
- An `Ingress` (configurable for nginx, traefik, or ALB) with TLS termination
- A `ConfigMap` holding `config.yaml`
- A `Secret` holding the provider API keys
- Optional: a Postgres and Redis sub-chart for self-contained dev clusters
- `HorizontalPodAutoscaler` based on CPU or custom Prometheus metrics

The HPA is the right way to scale: each proxy replica is stateless, so you can scale to N replicas with no coordination beyond Redis (for rate limits) and Postgres (for spend tracking). The proxy is also pod-disruption-budget friendly: graceful shutdown completes in-flight requests before the SIGTERM.

For most production deployments, the Postgres and Redis are **external** (managed services) and only the proxy replicas are in the cluster. This isolates the stateful components from Kubernetes churn.

---

## 9. Real Case: Fortune 500 Retailer, 10M Requests/Day

A US Fortune 500 retailer deployed LiteLLM proxy in late 2024 to serve 2,000+ internal developers across 40 product teams. The architecture:

- **Proxy**: 6 replicas on EKS, behind an ALB
- **Postgres**: RDS Multi-AZ, db.r6g.4xlarge
- **Redis**: ElastiCache, cache.r6g.large, 3 nodes
- **SSO**: Okta with auto-provisioning
- **Observability**: Langfuse for traces, Datadog for metrics
- **CI/CD**: Helm + ArgoCD, config changes via Git

The outcomes (from the public postmortem):

- **$2.4M annual savings** vs. a hypothetical per-team OpenAI Enterprise contract, achieved by routing 40% of traffic to open-weights on Groq and self-hosted Llama
- **Single audit trail** for SOC 2 — every LLM call tagged with `user_id`, `team_id`, `model`, and `cost`
- **Zero-downtime model rollouts** — adding Claude 3.5 Sonnet was a one-line config change and a Helm upgrade
- **30% reduction in average latency** vs. direct-to-provider, from the proxy's Redis cache absorbing 18% of traffic as cache hits

The architecture is essentially the Helm chart from this note plus a Datadog sidecar and an ArgoCD sync. The scale (10M requests/day) is well within the proxy's design envelope; the official BerriAI blog has benchmarks of single-replica proxy handling 1K req/s.

---

## 10. Production Reality

- **Config reload**: the proxy supports `SIGHUP` to reload `config.yaml` without a restart. Use this for emergency model disables and budget changes. Helm chart exposes a `reloadStrategy: SIGHUP` toggle.
- **Network egress**: the proxy makes outbound HTTPS calls to provider APIs. In a locked-down VPC, ensure the egress allowlist includes the provider domains (or use VPC endpoints for Bedrock).
- **Postgres failover**: RDS Multi-AZ has a 60–120s failover window. The proxy's connection pool re-establishes quickly, but in-flight requests during failover may fail. The client should retry; the cost ceiling will not be exceeded.
- **Redis failure**: if Redis is unreachable, the proxy falls back to per-process rate limits (less strict) and disables the semantic cache. The system continues to serve, just with reduced protection.
- **Log volume**: at 10M requests/day, `LiteLLM_SpendLogs` accumulates ~10M rows/day. Partition by month and archive to S3 after 90 days is the standard pattern.

---

## 🎯 Key Takeaways

- The `litellm-proxy` is a FastAPI server with three roles: end-user OpenAI-compatible API, admin API for keys/teams, and SSO for human login
- A production `config.yaml` covers model list, routing/fallbacks, callbacks, SSO, and rate limits in a single declarative file
- Docker Compose is the local-dev path; the official Helm chart is the production path on Kubernetes
- Virtual keys bind users to teams, models, budgets, and rate limits; the proxy enforces centrally with Redis-backed sliding windows
- SSO is configurable for Google, Azure AD, Okta; JWT is the right answer for service-to-service auth
- The Postgres logging database is on the critical path for budget enforcement — managed Postgres with Multi-AZ is the standard answer
- The Fortune 500 case study shows the architecture scales to 10M requests/day with 6 proxy replicas, RDS, and ElastiCache
- The proxy is stateless at the pod level; scaling is the HPA's job, coordination is Redis's job, durability is Postgres's job

## References

- LiteLLM Proxy documentation, [docs.litellm.ai/docs/proxy](https://docs.litellm.ai/docs/proxy)
- LiteLLM config.yaml reference, [docs.litellm.ai/docs/proxy/configs](https://docs.litellm.ai/docs/proxy/configs)
- LiteLLM Helm chart, [github.com/BerriAI/litellm-helm](https://github.com/BerriAI/litellm-helm)
- SSO setup guides, [docs.litellm.ai/docs/proxy/sso](https://docs.litellm.ai/docs/proxy/sso)
- Vault cross-links: [[02 - LiteLLM Core - Unified Multi-Provider Interface]], [[03 - Routing, Fallback and Retry Strategies]], [[04 - Observability, Cost Tracking and Rate Limiting]], [[06 - Capstone - Multi-Provider RAG Gateway with LiteLLM]]
