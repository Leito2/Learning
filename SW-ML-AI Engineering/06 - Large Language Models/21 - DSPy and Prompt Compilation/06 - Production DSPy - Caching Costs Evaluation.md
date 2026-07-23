# 🏭 Production DSPy — Caching, Costs, and Evaluation

A compiled DSPy program is a **production asset**: it must serve thousands of users, cost less than $0.01 per request, never go down, and improve over time. The optimizer that produced your program is offline; the runtime is online. This note teaches the production patterns that bridge compiled DSPy to deployable services: **LM caching** to avoid redundant calls, **cost attribution** to track spend per request, **deployable servers** for FastAPI integration, **monitoring** for production drift, and **re-compilation triggers** when quality drops.

By the end of this note you can take a compiled DSPy program and deploy it as a production service with the same operational discipline as any other backend.

## 🎯 Learning Objectives

- Use **DSPy's built-in LM cache** to avoid redundant calls.
- Configure **per-stage LM selection** for cost control.
- Deploy DSPy programs as **FastAPI services**.
- Set up **monitoring** via OpenTelemetry traces.
- Trigger **re-compilation** when quality drifts.
- Avoid the four most common production DSPy pitfalls.

## 1. LM Caching

DSPy caches LM calls by signature + inputs (deterministic caching):

```python
import dspy

# Configure disk cache
dspy.configure_cache(
    enable_disk_cache=True,
    disk_cache_dir="./dspy_cache",
)

# Or in-memory cache (per-process)
dspy.configure_cache(enable_memory_cache=True)
```

When the same `(signature, inputs)` is called again, DSPy returns the cached output without calling the LM. **Critical for**: (1) testing (run the same eval 10× without re-paying for LM calls), (2) production retry loops, (3) dev iteration.

### Production Cache Strategy

```python
import os

# Disk cache for cross-process / cross-restart persistence
dspy.configure_cache(
    enable_disk_cache=True,
    disk_cache_dir=os.environ.get("DSPY_CACHE_DIR", "/var/cache/dspy"),
    enable_memory_cache=True,  # in-process LRU for hot keys
)

# Tune for production
dspy.configure_cache(
    enable_disk_cache=True,
    disk_cache_dir="/var/cache/dspy",
    enable_memory_cache=True,
    memory_cache_size=10_000,  # 10K entries (~100MB)
)
```

### Cache Invalidation

```python
# Invalidate cache for a specific signature after a model upgrade
import shutil
shutil.rmtree("/var/cache/dspy/openai/gpt-4o-mini")  # force re-cache
```

Or version the cache directory per compiled-program version:

```python
import os
import hashlib

compiled_hash = hashlib.md5(compiled.to_json().encode()).hexdigest()[:8]
cache_dir = f"/var/cache/dspy/{compiled_hash}"
dspy.configure_cache(enable_disk_cache=True, disk_cache_dir=cache_dir)
```

## 2. Per-Stage LM Selection for Cost Control

```python
class CostOptimizedRAG(dspy.Module):
    """Use expensive LM only for generation; cheap LM for rewriting."""

    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought(QueryRewrite)  # cheap
        self.generate = dspy.ChainOfThought(Generate)      # expensive

    def forward(self, query, context):
        # Cheap LM for rewriting
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", max_tokens=200)):
            rewritten = self.rewrite(original_query=query).rewritten_query

        # Expensive LM for generation (default)
        return self.generate(query=rewritten, context=context)
```

**Cost breakdown for a typical RAG query:**
- Rewrite (gpt-4o-mini, 100 tokens out): $0.00006
- Generate (gpt-4o, 800 tokens out): $0.008
- Total: $0.00806

vs all-gpt-4o: $0.008 (just the generation). The rewrite cost is negligible compared to the quality gain from using a more powerful LM where it matters.

### Conditional LM

```python
class AdaptiveRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)
        self.simple_qa = dspy.ChainOfThought(SimpleQA)
        self.complex_qa = dspy.ChainOfThought(ComplexQA)

    def forward(self, query, context):
        # Cheap LM for intent classification
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
            intent = self.classify(query=query)

        # Route to expensive LM only for complex queries
        if intent.complexity == "high":
            with dspy.context(lm=dspy.LM("openai/gpt-4o")):
                return self.complex_qa(query=query, context=context)
        return self.simple_qa(query=query, context=context)
```

## 3. Deployable FastAPI Service

```python
# server.py
import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

# Configure once at startup
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
)
dspy.configure_cache(enable_disk_cache=True, disk_cache_dir="/var/cache/dspy")

# Load compiled program
compiled = dspy.Module()
compiled.load("/path/to/compiled_program.json")

app = FastAPI(title="DSPy RAG Service")


class QueryRequest(BaseModel):
    query: str
    context: list[str]


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, x_thread_id: str = Header(default="default")):
    try:
        result = compiled(query=req.query, context=req.context)
        return QueryResponse(answer=result.answer, citations=result.citations)
    except dspy.AssertionError as e:
        raise HTTPException(422, f"Output validation failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "compiled_version": "v2.4.1"}
```

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 4. OpenTelemetry Instrumentation

DSPy auto-emits spans through the configured LM. To instrument with OTel:

```python
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

# At app startup
OpenAIInstrumentor().instrument(capture_prompts=False, capture_completions=False)

# DSPy calls flow through the configured LM, which is wrapped by OTel
result = compiled(query=..., context=[...])
# Span emitted: openai.chat, with gen_ai.system, gen_ai.request.model attributes
```

For thread_id propagation, use the same [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/04 - OTel for LangGraph and Agent Frameworks.md|OTel baggage pattern]] from the LangGraph integration note.

## 5. Cost Tracking and Attribution

```python
class CostTrackingModule(dspy.Module):
    """Wraps any DSPy module to attribute costs to thread/user."""

    def __init__(self, wrapped_module):
        super().__init__()
        self.wrapped = wrapped_module

    def forward(self, **kwargs):
        from opentelemetry import baggage, metrics

        # Get token counts from the LM
        result = self.wrapped(**kwargs)

        # Get the latest LM usage
        lm_usage = dspy.settings.lm.history[-1].get("usage", {})
        input_tokens = lm_usage.get("prompt_tokens", 0)
        output_tokens = lm_usage.get("completion_tokens", 0)

        # Calculate cost
        cost_usd = (
            input_tokens * MODEL_PRICING["gpt-4o-mini"]["input"]
            + output_tokens * MODEL_PRICING["gpt-4o-mini"]["output"]
        )

        # Set as span attributes
        from opentelemetry import trace
        span = trace.get_current_span()
        span.set_attribute("cost.usd", cost_usd)
        span.set_attribute("cost.input_tokens", input_tokens)
        span.set_attribute("cost.output_tokens", output_tokens)
        span.set_attribute("cost.model", "gpt-4o-mini")

        return result
```

## 6. Monitoring Production Quality

```python
# Run a sample of production requests through RAGAS daily
import random
from ragas import evaluate
from ragas.metrics import faithfulness

# Sample 5% of today's traffic
sampled = random.sample(today_traces, k=int(0.05 * len(today_traces)))

# Evaluate
results = evaluate(
    [trace_to_ragas_sample(t) for t in sampled],
    metrics=[faithfulness],
)

# Alert if faithfulness drops below threshold
if results["faithfulness"].mean() < 0.85:
    send_alert(f"Faithfulness dropped to {results['faithfulness'].mean():.3f}")
```

### Drift Detection

```python
def detect_drift(recent_scores: list[float], historical_scores: list[float], threshold: float = 0.05):
    """Compare last week's scores to historical baseline."""
    from scipy import stats
    t, p = stats.ttest_ind(recent_scores, historical_scores)
    return {
        "drift_detected": p < 0.05 and abs(recent_scores.mean() - historical_scores.mean()) > threshold,
        "p_value": p,
        "delta": recent_scores.mean() - historical_scores.mean(),
    }
```

## 7. Re-Compilation Triggers

```python
def maybe_recompile(current_quality: float, threshold: float = 0.85):
    """Trigger re-compilation if quality drops below threshold."""
    if current_quality < threshold:
        # Re-compile with the same optimizer settings
        new_compiled = MIPRO(metric=metric, auto="medium").compile(
            RAG(),
            trainset=trainset,
            valset=valset,
        )
        # Save and deploy via CI/CD
        new_compiled.save("compiled_v2.json")
        # Trigger deployment
        trigger_deployment("compiled_v2.json")
        return new_compiled
    return current_compiled
```

The trigger logic lives in your CI/CD: when the daily RAGAS eval drops below threshold, run the optimizer and deploy the new compiled version.

## 8. ❌/✅ Antipatterns

### ❌ Recompiling on every drift signal

```python
# ⚠️ Compiles are expensive; drift signals are noisy
if quality < threshold:
    recompile()  # every hour
```

### ✅ Threshold + cooldown for re-compilation

```python
# ✅ Re-compile only when sustained drift is detected
if quality < threshold and last_compile > 7_days_ago:
    recompile()
```

### ❌ No disk cache in production

```python
# ⚠️ Same query repeated = same LM call = wasted money
result = compiled(query="Capital of France?", context=[...])
result2 = compiled(query="Capital of France?", context=[...])  # re-charged!
```

### ✅ Disk cache enabled

```python
dspy.configure_cache(enable_disk_cache=True, disk_cache_dir="/var/cache/dspy")
# result2 returns from cache, no LM call
```

### ❌ All GPT-4o for all stages

```python
# ⚠️ Wasteful
self.generate = dspy.ChainOfThought(Generate, lm=dspy.LM("openai/gpt-4o"))
```

### ✅ Tiered LMs per stage

```python
# ✅ Cheap LM for rewriting, expensive LM for generation
self.rewrite = dspy.ChainOfThought(Rewrite, lm=dspy.LM("openai/gpt-4o-mini"))
self.generate = dspy.ChainOfThought(Generate, lm=dspy.LM("openai/gpt-4o"))
```

### ❌ No cost attribution

```python
# ⚠️ Don't know which user / thread costs the most
result = compiled(...)  # cost invisible
```

### ✅ Span attributes for cost

```python
span.set_attribute("cost.usd", cost_usd)
span.set_attribute("cost.user_id", user_id)
# Phoenix dashboard: per-user cost
```

## 9. Production Reality

**Caso real — Production RAG Project:** Compiled RAG deployed as FastAPI service. Disk cache enabled (`/var/cache/dspy`); cache hit rate 35% in production (many repeated queries). Re-compile triggers when 7-day faithfulness drops below 0.85 — fires ~once per quarter. Cost per request: $0.004 average, $0.012 p99 (long-context queries).

**Caso real — Multi-Agent Research System:** Three compiled agents, each with its own compiled version. CI/CD pipeline re-runs RAGAS on each compile and gates deploys. The compiled prompts are versioned in git (`compiled_research_v3.json`, `compiled_audit_v2.json`); rollbacks are trivial.

## 📦 Compression Code

```python
# 📦 Compression: Production DSPy in 50 lines

import dspy
import os
from fastapi import FastAPI

# 1. Configure LM + cache at startup
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
)
dspy.configure_cache(
    enable_disk_cache=True,
    disk_cache_dir="/var/cache/dspy",
    enable_memory_cache=True,
)

# 2. Load compiled program
compiled = RAG()  # your DSPy module class
compiled.load("/models/compiled_rag_v3.json")

# 3. Cost tracking per call
from opentelemetry import trace

class CostTracked(dspy.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, **kwargs):
        result = self.wrapped(**kwargs)
        usage = dspy.settings.lm.history[-1].get("usage", {})
        cost = usage.get("prompt_tokens", 0) * 0.00015 / 1000 + usage.get("completion_tokens", 0) * 0.0006 / 1000
        trace.get_current_span().set_attribute("cost.usd", cost)
        return result

tracked = CostTracked(compiled)

# 4. Deploy
app = FastAPI()

@app.post("/query")
async def query(req):
    return tracked(query=req.query, context=req.context)
```

## 🎯 Key Takeaways

1. **Disk cache** — `enable_disk_cache=True` saves 30%+ in production.
2. **Per-stage LM** — cheap LM for rewriting, expensive for generation.
3. **Conditional LM** — route to expensive LM only for complex queries.
4. **FastAPI + compiled** — load `.json` once, serve as endpoint.
5. **OTel instrumentation** — every LM call is a span with `gen_ai.*` attributes.
6. **Cost attribution via span attributes** — per-user cost dashboards in Phoenix.
7. **Re-compilation triggers** — CI/CD re-runs optimizer when quality drifts.

## References

- [[00 - Welcome to DSPy and Prompt Compilation|Welcome]] — course map.
- [[02 - Optimizers|Optimizers]] — the compiler.
- [[04 - DSPy + LangGraph Integration|LangGraph]] — agent integration.
- [[05 - DSPy Assertions|Assertions]] — quality constraints in production.
- [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers.md|OTel]] — observability.
- DSPy caching: https://dspy.ai/deploy-production/
- DSPy FastAPI: https://dspy.ai/tutorials/observability/