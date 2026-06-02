# 🏷️ Observability, Cost Tracking and Rate Limiting

## 🎯 Learning Objectives
- Implement LiteLLM callbacks for success/failure observability
- Integrate with Langfuse, Phoenix, LangSmith, Helicone, and S3 for trace storage
- Build a `CustomLogger` subclass for bespoke observability pipelines
- Use virtual keys and per-team spend limits for multi-tenant control
- Configure per-user rate limits (RPM, TPM) at the proxy level
- Compute and store token costs in a relational database for finance reporting

## Introduction

Once a gateway sits in front of every LLM call, the natural next question is: *what is happening inside it?* Which model is being hit, by whom, at what cost, with what latency, succeeding or failing for what reason, and what is the cumulative effect on the monthly bill? The answers to these questions live in the observability layer; the financial answers live in the cost-tracking layer; the policy answers live in the rate-limiting layer. This note covers all three, with the unifying idea that they are not three separate systems but three views on the same event stream.

The library-level callbacks from [[02 - LiteLLM Core - Unified Multi-Provider Interface]] cover the single-process case. The proxy server from [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]] extends the same primitives across many workers, with Redis-backed coordination and Postgres-backed durability. This note covers the library-level surface in depth and previews the proxy-level extensions.

---

## 1. The Callback Interface

LiteLLM fires two kinds of events:

- **Success events**: fired after a successful completion, with the full `ModelResponse`, the original `kwargs`, and timing
- **Failure events**: fired after an exception, with the exception object and the original kwargs

The hooks are settable on the module level or per call:

```python
import litellm

def my_success_callback(kwargs, completion_response, start_time, end_time):
    # kwargs contains model, messages, user_id, etc.
    # completion_response is the ModelResponse
    print(f"[OK] {kwargs['model']} in {(end_time - start_time).total_seconds():.2f}s")

def my_failure_callback(kwargs, exception, start_time, end_time):
    print(f"[ERR] {kwargs.get('model')}: {exception}")

litellm.success_callback = [my_success_callback]
litellm.failure_callback = [my_failure_callback]
```

This is the lowest-level hook. Every shipped integration — Langfuse, Phoenix, Helicone, S3, LangSmith — is just a success/failure callback that knows the wire format of the destination system.

### 1.1 Async Callbacks

For high-throughput services, sync callbacks block the worker. LiteLLM supports async callbacks for non-blocking writes:

```python
async def my_async_callback(kwargs, completion_response, start_time, end_time):
    # network call, but doesn't block the worker
    await some_async_store(kwargs, completion_response)

litellm.success_callback = [my_async_callback]
```

The library detects coroutine return values and awaits them on the event loop. For CPU-bound work (e.g., hashing, compression) use sync callbacks; for network I/O (e.g., pushing to a trace store) use async.

---

## 2. The First-Party Integrations

LiteLLM ships first-party integrations with the major observability backends. Configuration is usually a single line plus an environment variable:

| Integration | Setup | Best For |
|-------------|-------|----------|
| **Langfuse** | `litellm.success_callback = ["langfuse"]` + `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | OSS traces, prompt versioning |
| **Phoenix (Arize)** | `litellm.success_callback = ["arize_phoenix"]` + `PHOENIX_API_KEY` or local | OpenTelemetry-native, free OSS |
| **LangSmith** | `litellm.success_callback = ["langsmith"]` + `LANGCHAIN_API_KEY` | Already in the LangChain ecosystem |
| **Helicone** | `litellm.success_callback = ["helicone"]` + `HELICONE_API_KEY` | Quick SaaS proxy, per-user cost |
| **S3 / GCS / Azure Blob** | `litellm.success_callback = ["s3"]` + `AWS_*` envs | Long-term log archival, audit |
| **OpenTelemetry** | `litellm.success_callback = ["otel"]` + OTLP endpoint | Vendor-neutral, integrate with existing APM |
| **Prometheus** | `litellm --detailed_debug` exposes metrics endpoint | Self-hosted, Grafana dashboards |

A minimal Langfuse setup:

```python
import litellm
import os

litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUE_SECRET_KEY"] = "sk-..."  # typo intentional, do not copy-paste

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"user_id": "user_42", "trace_name": "chat-eval"},
)
```

The Langfuse callback automatically extracts `user_id` and `trace_name` from `metadata` and creates a trace with the full prompt/response/tokens/cost. The Langfuse UI then shows the trace in the timeline.

### 2.1 The Cost Field

Every trace, regardless of integration, includes a `cost` field calculated by `litellm.completion_cost()`. The per-model cost dictionary is what powers the financial reporting downstream.

```python
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"trace_id": "abc-123"},
)
print(litellm.completion_cost(completion_response=response))
# 0.000015  (USD)
```

The cost field is the basis for everything in section 3.

---

## 3. Custom Loggers: Bespoke Observability

When the first-party integrations are not enough — usually because you need to push to an internal system, or to compute a custom metric, or to enforce a policy at the callback layer — the `CustomLogger` subclass is the extension point:

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger


class PhoenixPlusPostgresLogger(CustomLogger):
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        # 1. Push a trace to Phoenix
        self._push_to_phoenix(kwargs, response_obj, start_time, end_time)
        # 2. Write a row to Postgres for finance
        self._write_to_postgres(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, exception, start_time, end_time):
        # 1. Push an error trace to Phoenix
        self._push_error_to_phoenix(kwargs, exception, start_time, end_time)
        # 2. Write a row to Postgres marked as failure
        self._write_error_to_postgres(kwargs, exception, start_time, end_time)
```

The two methods receive the `kwargs` dictionary (which includes `model`, `messages`, `user_id`, `metadata`, etc.) and either the `ModelResponse` or the exception. The implementation has access to anything in the original `acompletion()` call, plus LiteLLM-computed fields like the cost.

### 3.1 A Full Implementation: Phoenix + Postgres

```python
import os
import json
import asyncio
import httpx
import asyncpg
from datetime import datetime
from litellm.integrations.custom_logger import CustomLogger
import litellm

PHOENIX_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")
POSTGRES_DSN = os.environ["POSTGRES_DSN"]


class PhoenixPlusPostgresLogger(CustomLogger):
    def __init__(self):
        self.pg = None
        self.http = None

    async def _ensure_connections(self):
        if self.pg is None:
            self.pg = await asyncpg.connect(POSTGRES_DSN)
        if self.http is None:
            self.http = httpx.AsyncClient(timeout=5.0)

    async def log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._ensure_connections()
        model = kwargs.get("model", "unknown")
        cost = litellm.completion_cost(completion_response=response_obj) or 0.0
        prompt_tokens = response_obj.usage.prompt_tokens
        completion_tokens = response_obj.usage.completion_tokens
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        user_id = kwargs.get("user_id") or kwargs.get("metadata", {}).get("user_id")
        trace_id = kwargs.get("metadata", {}).get("trace_id")

        await asyncio.gather(
            self._push_to_phoenix(trace_id, model, kwargs, response_obj, latency_ms, cost),
            self._write_to_postgres(
                user_id, trace_id, model, prompt_tokens, completion_tokens,
                cost, latency_ms, "success", "",
            ),
        )

    async def log_failure_event(self, kwargs, exception, start_time, end_time):
        await self._ensure_connections()
        model = kwargs.get("model", "unknown")
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        user_id = kwargs.get("user_id") or kwargs.get("metadata", {}).get("user_id")
        trace_id = kwargs.get("metadata", {}).get("trace_id")

        await self._push_error_to_phoenix(trace_id, model, kwargs, exception, latency_ms)
        await self._write_to_postgres(
            user_id, trace_id, model, 0, 0, 0.0, latency_ms, "failure", str(exception),
        )

    async def _push_to_phoenix(self, trace_id, model, kwargs, response_obj, latency_ms, cost):
        # Phoenix accepts OpenTelemetry OTLP; we just POST a JSON span
        span = {
            "name": f"litellm.{model}",
            "span_id": trace_id or "auto",
            "start_time": kwargs.get("start_time"),
            "end_time": kwargs.get("end_time"),
            "attributes": {
                "model": model,
                "prompt_tokens": response_obj.usage.prompt_tokens,
                "completion_tokens": response_obj.usage.completion_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            },
        }
        await self.http.post(f"{PHOENIX_URL}/v1/spans", json=span)

    async def _push_error_to_phoenix(self, trace_id, model, kwargs, exception, latency_ms):
        span = {
            "name": f"litellm.{model}.error",
            "span_id": trace_id or "auto",
            "status": {"code": "ERROR", "message": str(exception)},
            "attributes": {"model": model, "latency_ms": latency_ms},
        }
        await self.http.post(f"{PHOENIX_URL}/v1/spans", json=span)

    async def _write_to_postgres(
        self, user_id, trace_id, model, prompt_tokens, completion_tokens,
        cost, latency_ms, status, error,
    ):
        await self.pg.execute(
            """
            INSERT INTO llm_calls (
                user_id, trace_id, model, prompt_tokens, completion_tokens,
                cost_usd, latency_ms, status, error, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
            user_id, trace_id, model, prompt_tokens, completion_tokens,
            cost, latency_ms, status, error,
        )


# Wire it up
litellm.callbacks = [PhoenixPlusPostgresLogger()]
```

The Postgres schema implied:

```sql
CREATE TABLE llm_calls (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT,
    trace_id        TEXT,
    model           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL,
    status          TEXT NOT NULL,    -- 'success' | 'failure'
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_llm_calls_user_created ON llm_calls (user_id, created_at);
CREATE INDEX idx_llm_calls_created ON llm_calls (created_at);
```

Finance can now run:

```sql
SELECT user_id, SUM(cost_usd) AS total_spend
FROM llm_calls
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY user_id
ORDER BY total_spend DESC;
```

---

## 4. Virtual Keys and Team-Based Spend Limits

The library-level `CustomLogger` writes *records* of who spent what. The proxy-level virtual keys *prevent* overspend in the first place.

A virtual key is a string that the proxy issues to a team, scoped to:

- **Models allowed**: which model names this key can call
- **RPM/TPM rate limits**: requests/min, tokens/min
- **Spend ceiling**: max USD per day/week/month
- **Team association**: which team the key belongs to

The key is sent to the proxy as `Authorization: Bearer sk-litellm-virtual-...`. The proxy validates the key, applies the limits, calls the real provider, and decrements the spend counter.

```yaml
# litellm proxy config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params: { model: gpt-4o, api_key: os.environ/OPENAI_API_KEY }
  - model_name: claude-3-5-sonnet
    litellm_params: { model: claude-3-5-sonnet-20241022, api_key: os.environ/ANTHROPIC_API_KEY }

litellm_settings:
  success_callback: ["langfuse"]

general_settings:
  database_url: "postgresql://litellm:litellm@postgres:5432/litellm"
  master_key: os.environ/LITELLM_MASTER_KEY

# Per-team keys (created via the proxy admin API)
# This is the schema for keys stored in Postgres
# team_id, key, models, max_budget, rpm, tpm, duration
```

Creating a key via the admin API:

```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team_data_science",
    "models": ["gpt-4o", "claude-3-5-sonnet"],
    "max_budget": 50000,
    "budget_duration": "30d",
    "rpm": 1000,
    "tpm": 500000
  }'
```

The response is the new virtual key. Once `team_data_science` has spent $50K in 30 days, the proxy starts returning `403` for any call using that key.

---

## 5. Rate Limiting

The proxy enforces RPM (requests/min) and TPM (tokens/min) per key, per team, or per user. The implementation is sliding-window with Redis-backed counters:

```yaml
# In config.yaml — global limits apply to the team
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 10000        # requests per minute globally
      tpm: 5000000      # tokens per minute globally
```

Per-key limits override the deployment limits. The proxy also supports:

- **Burst allowance**: `burst_limit` lets a key exceed RPM for a short window
- **Per-route limits**: limit by URL path (`/v1/chat/completions` vs `/v1/embeddings`)
- **Per-model limits**: a key can have different RPMs for different models

The library-level analog is the `rpm` and `tpm` fields on the Router deployment configuration from [[03 - Routing, Fallback and Retry Strategies]]. At the library level, limits are advisory (the library throws a `RateLimitError` when the limit is exceeded); at the proxy level, they are enforced centrally.

---

## 6. The Token-Counting and Cost-Calculation Engine

A subtle but important point: the cost calculation is **not** `tokens * price`. Some providers charge differently for input vs output. Some providers charge per image, per second of audio, or per character. LiteLLM's per-model cost dictionary encodes all of this:

```python
litellm.model_cost["gpt-4o"]
# {
#   "input_cost_per_token": 0.0000025,    # $2.50 per 1M tokens
#   "output_cost_per_token": 0.00001,     # $10.00 per 1M tokens
#   "max_tokens": 16384,
#   "max_input_tokens": 128000,
#   "mode": "chat",
# }
```

For models with a long-context surcharge:

```python
litellm.model_cost["claude-3-5-sonnet-20241022"]
# {
#   "input_cost_per_token": 0.000003,
#   "output_cost_per_token": 0.000015,
#   "input_cost_per_token_above_200k_tokens": 0.000006,  # 2x
#   "output_cost_per_token_above_200k_tokens": 0.0000225,
#   ...
# }
```

The library applies the tiered pricing automatically. The full cost formula is:

$$
\text{cost} = \sum_{i=0}^{T} p_i \cdot n_i
$$

where $p_i$ is the per-token price for tier $i$ and $n_i$ is the number of tokens in that tier. For models with caching discounts (Anthropic prompt caching, OpenAI cached input), the same shape applies with a separate `cached_input_cost_per_token` field.

For audio and image models, the fields are different:

```python
litellm.model_cost["whisper-1"]
# {
#   "input_cost_per_second": 0.0001,  # $0.0001 per second of audio
#   "mode": "audio_transcription",
# }
litellm.model_cost["dall-e-3"]
# {
#   "input_cost_per_image": 0.04,  # $0.04 per 1024x1024 image
#   "output_cost_per_image": 0.04,
#   "mode": "image_generation",
# }
```

The user-extensibility (the `litellm.model_cost` dictionary from [[02 - LiteLLM Core - Unified Multi-Provider Interface]]) is what makes this work for fine-tuned or self-hosted models where the price is a custom negotiated rate.

---

## 7. Real Case: A Fintech Enforces $50K/Month per Team

A US-based fintech with 200 internal developers, 12 product teams, and $600K/month of LLM spend deployed LiteLLM proxy in early 2025. Their configuration:

- 12 teams, each with a virtual key
- Per-team budget: $50K/month hard cap
- Per-team RPM: 500 (raised to 2000 for the data science team)
- All calls logged to Postgres with `team_id` and `user_id`
- Langfuse for prompt-versioning and quality tracking
- Daily cron job emails team leads a "yesterday's spend" report

The outcome after 6 months:

- **$0 in budget overruns**: every team's spend was visible in real time, and the $50K cap is enforced at the proxy level. Previously, a single agent-loop bug in a research notebook had burned $40K in one weekend.
- **2x faster model rollouts**: when Anthropic released Claude 3.5 Sonnet, the platform team added it to the model list and made it available to all 12 teams in a single config change.
- **Audit trail for compliance**: the SOC 2 audit, which used to take 3 weeks of pulling logs from various systems, was reduced to a single Postgres query.

The architecture is the same one in [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]] — the difference is the policy layer (virtual keys, budgets) layered on top.

---

## 8. Production Reality

A few operational caveats that matter at scale:

- **Callback latency**: an async callback that takes 100ms adds 100ms to the *eventual* consistency window of your traces. Phoenix/S3 pushes are usually <50ms; Postgres writes in the same region are <10ms; cross-region Postgres writes can be 100ms+ and are a real source of trace lag.
- **Callback failures**: if a callback throws, the library logs the error and continues. The original LLM call still succeeds. This is the right behavior, but it means a callback that silently dies (e.g., wrong API key) becomes invisible. A separate heartbeat job that calls each observability sink every minute is the standard mitigation.
- **Spend ceiling enforcement**: the proxy uses a best-effort check. In a bursty scenario, a team can exceed their RPM by 10–20% before the rate limiter catches up. The spend ceiling is harder; it depends on cost arriving in the DB before the next call. With Postgres latency of <10ms, the cap is usually enforced within one call, but a long-running streamed call can be the exception.
- **Cost dictionary freshness**: provider prices change. A `litellm.completion_cost()` call against a model that was re-priced last week will return the old price until you upgrade LiteLLM. Pin a recent version in `requirements.txt` and watch the [release notes](https://github.com/BerriAI/litellm/releases).

---

## 🎯 Key Takeaways

- LiteLLM callbacks fire on success and failure; the shipped integrations (Langfuse, Phoenix, LangSmith, Helicone) are themselves callbacks
- `CustomLogger` subclasses give full access to `kwargs`, `ModelResponse`, timing, and computed cost for bespoke observability
- A typical setup pushes traces to Phoenix (UI) and writes cost rows to Postgres (finance) from the same callback
- Virtual keys in the proxy layer enforce per-team spend caps; library-level callbacks only *record* spend
- RPM/TPM rate limits are sliding-window with Redis-backed counters in the proxy; library-level limits are advisory
- The cost engine is per-model with tiered pricing, long-context surcharges, and image/audio modes; user-extensible for fine-tuned or self-hosted models
- The fintech case study shows the ROI: $0 budget overruns, 2x faster model rollouts, audit trail in one query
- Observability correctness is bounded by callback latency; a separate heartbeat is the right way to detect silent callback failures

## References

- LiteLLM Callbacks, [docs.litellm.ai/docs/observability](https://docs.litellm.ai/docs/observability)
- Langfuse integration, [docs.litellm.ai/docs/observability/langfuse_integration](https://docs.litellm.ai/docs/observability/langfuse_integration)
- Phoenix (Arize) integration, [docs.litellm.ai/docs/observability/arize_phoenix_integration](https://docs.litellm.ai/docs/observability/arize_phoenix_integration)
- CustomLogger reference, [docs.litellm.ai/docs/observability/custom_callback](https://docs.litellm.ai/docs/observability/custom_callback)
- Vault cross-links: [[02 - LiteLLM Core - Unified Multi-Provider Interface]], [[03 - Routing, Fallback and Retry Strategies]], [[05 - Self-Hosted LiteLLM Proxy - Docker, Kubernetes and Auth]], [[06 - Capstone - Multi-Provider RAG Gateway with LiteLLM]]
