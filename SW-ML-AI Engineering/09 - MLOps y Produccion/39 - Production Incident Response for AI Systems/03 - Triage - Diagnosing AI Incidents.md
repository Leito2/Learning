# 🎯 03 - Triage — Diagnosing AI Incidents in Under 5 Minutes

> **The OODA loop applied to AI. Where to look first, what to check second, when to rollback. The 5-minute diagnostic checklist that separates senior engineers from everyone else.**

## 🎯 Learning Objectives
- Apply the OODA loop (Observe, Orient, Decide, Act) to AI incidents
- Use LangFuse + Phoenix traces to diagnose the dominant category in < 5 minutes
- Execute the 5-minute diagnostic checklist for each of the 7 AI incident categories
- Identify the right rollback procedure for each category (prompt, model, version, kill switch)
- Communicate status to stakeholders during the incident
- Document the diagnosis in a structured incident note

## Introduction

The first 5 minutes of an incident determine the next 5 hours. A senior on-call engineer diagnoses the dominant category in < 5 minutes, applies the right mitigation, and pages the team. A junior engineer spends 2 hours reading logs, asking the team chat, and trying random fixes. The difference is not talent — it is **pattern recognition from a runbook**.

This note covers the **OODA loop for AI incidents**:

1. **Observe**: what does the alert tell you? Which category?
2. **Orient**: what does the trace data show? Which tenant? Which prompt?
3. **Decide**: which rollback or mitigation? Can you auto-mitigate?
4. **Act**: apply the fix, page the team, communicate status.

The patterns here draw from Note 01 (Taxonomy), Note 02 (Detection), and build toward Note 04 (Resolution Patterns). Mastering triage is the difference between **5-minute incidents** and **5-hour incidents**.

![OODA loop for AI incidents](https://example.com/ooda-loop.png)

---

## 1. The 5-Minute Diagnostic Checklist

When an alert fires (or a user reports an issue), execute this checklist:

```markdown
## 5-Minute AI Incident Checklist

### Minute 1: Acknowledge + Scope
- [ ] Acknowledge the alert in PagerDuty / Slack
- [ ] Note the alert name and severity
- [ ] Open the LangFuse dashboard for the relevant time window

### Minute 2: Identify the category
- [ ] Check the alert type:
  - Quality score alert → hallucination cascade / quality drift
  - Cost alert → cost explosion / runaway agent
  - Latency alert → provider outage / cold start
  - Error rate alert → generic outage
  - Security alert → prompt injection / data leakage

### Minute 3: Scope the impact
- [ ] Run query in LangFuse: filter traces by tenant, model, prompt
- [ ] Identify how many users / how much traffic is affected
- [ ] Check if the issue is tenant-specific or global

### Minute 4: Find the trigger
- [ ] Look at the most recent traces with the worst scores / highest latency
- [ ] Identify the common pattern: prompt version? model version? new feature flag?
- [ ] Check the deploy log for the last 24 hours

### Minute 5: Apply mitigation
- [ ] Auto-mitigate if possible (rate limit, kill switch)
- [ ] Page the team if SEV-1/SEV-2
- [ ] Communicate status to stakeholders
- [ ] Begin postmortem tracking
```

If you can't diagnose in 5 minutes, **page the team** — it's SEV-2 by default and a senior engineer should take over.

---

## 2. The OODA Loop for Each Category

### 2.1 Category 1: Hallucination Cascade

**Observe:** Quality score alert fired. LangFuse shows `correctness` < 0.7.

**Orient:**
```python
# In LangFuse: filter traces by low quality score
traces = langfuse.fetch_traces(
    scores=[{"name": "correctness", "operator": "<", "value": 0.7}],
    hours_back=1,
    limit=50,
)
```

Common patterns to look for:
- All traces use the same prompt version → prompt issue
- All traces use the same model → model issue (provider change)
- All traces are about a specific topic → data corpus issue (stale docs)
- All traces are from one tenant → tenant-specific issue (data privacy, custom prompt)

**Decide:**
| Pattern | Rollback |
|---------|----------|
| Prompt version | Rollback to last known-good prompt version |
| Model version | Pin to last known-good model (e.g., `gpt-4o-2024-08-06`) |
| Stale corpus | Re-ingest documents |
| Tenant-specific | Disable tenant's custom prompt |

**Act:** Apply the rollback, monitor quality score for 15 minutes, page team if not recovered.

### 2.2 Category 2: Cost Explosion

**Observe:** Cost alert fired. Per-tenant or global cost spike.

**Orient:**
```python
# Find the top tenants by cost
top_tenants = langfuse.fetch_scores(
    name="cost_usd",
    hours_back=1,
    group_by="tenant_id",
    order_by="value",
    limit=10,
)
```

Common patterns:
- One tenant → rate limit that tenant immediately
- All tenants on one model → check provider pricing changes
- All tenants on one prompt → runaway prompt template (loop)
- All tenants → DDoS or abuse pattern

**Decide:**
| Pattern | Mitigation |
|---------|------------|
| One tenant runaway | Per-tenant rate limit + audit |
| All tenants on one model | Switch to fallback provider |
| Prompt loop | Rollback prompt template |
| DDoS | Enable CloudFlare rate limiting |

**Act:** Apply rate limit, kill switch if SEV-1, page team.

### 2.3 Category 3: Latency Cliff

**Observe:** Latency alert. p99 > 2x baseline.

**Orient:**
```python
# Check provider health
import requests

for provider in ["openai", "anthropic", "together", "fireworks"]:
    status = requests.get(f"https://status.{provider}.com/api/v2/status.json").json()
    print(f"{provider}: {status['status']['indicator']}")
```

Common patterns:
- Provider outage → failover to secondary
- Cold start → pre-warm containers
- Token explosion → check recent prompt changes
- Queue backlog → check rate limit changes

**Decide:**
| Pattern | Mitigation |
|---------|------------|
| Provider outage | Failover via LiteLLM Router |
| Cold start | Pre-warm containers |
| Token explosion | Rollback prompt template |
| Queue backlog | Apply backpressure / 503 responses |

**Act:** LiteLLM Router automatic failover, page team if manual intervention needed.

### 2.4 Category 4: Quality Drift

**Observe:** Quality score drift alert. 7-day trend is downward.

**Orient:**
```python
# Compare last 7 days vs prior 30 days
recent = langfuse.fetch_scores(name="correctness", days_back=7)
historical = langfuse.fetch_scores(name="correctness", days_back=30, exclude_last=7)

# Per-metric drift
for metric in ["correctness", "relevance", "faithfulness"]:
    recent_avg = mean([s.value for s in recent if s.name == metric])
    historical_avg = mean([s.value for s in historical if s.name == metric])
    print(f"{metric}: {recent_avg:.2f} (was {historical_avg:.2f}, drift={(recent_avg/historical_avg):.1%})")
```

Common patterns:
- All metrics drift together → model or prompt issue
- Specific metric drifts → data corpus or retrieval issue
- Gradual drift → distribution shift (user behavior)
- Sudden drift → provider change or deploy

**Decide:**
| Pattern | Mitigation |
|---------|------------|
| All metrics, gradual | Rollback to last known-good model/prompt; investigate distribution shift |
| Specific metric, sudden | Re-ingest corpus; check retrieval |
| After provider change | Pin model version |

**Act:** Rollback, re-evaluate, monitor.

### 2.5 Category 5: Prompt Injection

**Observe:** Security alert. Pattern detector triggered.

**Orient:**
```python
# Find the malicious trace
trace = langfuse.get_trace(trace_id)
print("Input:", trace.input)
print("Output:", trace.output)
print("User:", trace.metadata.get("user_id"))
```

Common patterns:
- Direct injection: "ignore previous instructions..."
- Indirect: malicious content in retrieved document
- Token smuggling: Unicode tricks, base64

**Decide:**
| Pattern | Mitigation |
|---------|------------|
| Direct injection | Block user; add to detector patterns |
| Indirect injection | Sanitize retrieved documents; restrict corpus |
| Token smuggling | Add normalization layer |

**Act:** Block user, update detector, page team.

### 2.6 Category 6: Data Leakage

**Observe:** PII detector found sensitive data in traces.

**Orient:**
```python
# Find traces with detected PII
leaked_traces = langfuse.fetch_traces(
    filter={"metadata.contains_pii": True},
    hours_back=24,
)
```

**Decide:**
| Severity | Action |
|----------|--------|
| Single user, self-reported | Redact trace; no further action |
| Multi-user, accidental | Disable tracing; notify compliance |
| External breach | Data breach protocol; 72-hour notification (GDPR) |

**Act:** Follow the data breach protocol. Page compliance team immediately.

### 2.7 Category 7: Cascading Agent Failures

**Observe:** Downstream agent quality drops. LangFuse traces show the upstream failure.

**Orient:**
```python
# Find the failing upstream agent
trace = langfuse.get_trace(trace_id)
for span in trace.spans:
    print(f"{span.name}: quality={span.metadata.get('quality_score')}")
```

**Decide:** Rollback the failing agent; disable if irrecoverable.

**Act:** Apply rollback, page team.

---

## 3. Rollback Procedures

The four levels of rollback:

### 3.1 Level 1: Rollback the prompt

```python
# In LangFuse: switch active prompt to last known-good version
langfuse.update_prompt_labels(
    name="qa_system_prompt",
    version=PREVIOUS_PROD_VERSION,
    labels=["production"],
)
# Takes effect within 1-2 minutes for new requests
```

### 3.2 Level 2: Rollback the model

```python
# Pin to last known-good model version
client = OpenAI(model="gpt-4o-2024-08-06")  # was "gpt-4o" (latest)
# Or via LiteLLM
router = litellm.Router(
    model_list=[{
        "model_name": "production-llm",
        "litellm_params": {"model": "gpt-4o-2024-08-06"},
    }],
)
```

### 3.3 Level 3: Rollback the deploy

```bash
# Kubernetes rollback
kubectl rollout undo deployment/llm-service
# Returns to previous Deployment spec within 30 seconds
```

### 3.4 Level 4: Disable the feature (kill switch)

```python
# Feature flag system
@feature_flag("chat_completion", default=False)
async def chat_completion(...):
    # ... code ...

# During incident:
feature_flags.set("chat_completion", False)  # disables all traffic
# All requests get a 503 fallback
```

The kill switch is the last resort. Use it when:
- Cost is exploding with no time to diagnose
- Security incident (prompt injection breach)
- Quality degradation is severe and rollback didn't work

---

## 4. Communicating Status

During an incident, communicate at 4 levels:

### 4.1 Status page (external users)

```
[12:34] Investigating: We're seeing elevated error rates on chat completions.
[12:45] Identified: Provider issue; failing over to secondary.
[12:50] Resolved: Service restored. Quality scores normal.
```

### 4.2 Slack #incidents channel

```
@here SEV-1 — Cost anomaly detected.
Top tenant: acme-corp ($5,000 in 10 minutes).
Investigating runaway agent pattern.
Updates every 15 min.
```

### 4.3 Email to stakeholders

```
Subject: [SEV-2] Quality degradation on chat completions

Team,

We're investigating a quality score drop on chat completions detected at 12:34 UTC.
Initial data suggests a prompt regression; investigating with on-call engineer.
Expected resolution: 30 min.

Customer impact: ~5% of recent responses may have lower quality.
Mitigation: rolling back to previous prompt version.

Will follow up at 13:30 with resolution details.

— AI Platform Team
```

### 4.4 Postmortem kickoff (after resolution)

After resolution, the postmortem is the most important artifact. See Note 05 for the template.

---

## 5. Status Communication Checklist

```markdown
- [ ] First status update within 5 minutes of incident start
- [ ] Updates every 15 minutes (or at every material change)
- [ ] Final status update with timestamp
- [ ] All updates posted to status page
- [ ] Slack #incidents channel notified
- [ ] Email to stakeholders (SEV-1/SEV-2 only)
- [ ] Customer support team briefed (if customer-facing)
- [ ] Legal/compliance notified (if data-related)
```

---

## 6. The Incident Note

Every incident gets a structured note (in Notion, Confluence, or GitHub):

```markdown
## Incident Note

**ID:** INC-2026-0723-001
**Severity:** SEV-1
**Status:** Resolved
**Start:** 2026-07-23 12:34 UTC
**End:** 2026-07-23 13:45 UTC
**Duration:** 1 hour 11 minutes
**On-call:** Alice (primary), Bob (secondary)
**Category:** Hallucination cascade

## Summary
Customer-support bot returned hallucinated product specs for ~2 hours.
Triggered by a deploy that changed the prompt template.
Resolved by rolling back the prompt to the previous version.

## Impact
- Affected users: ~10,000
- Affected requests: ~15,000
- Cost of incident: $0 direct; potential brand impact: ~$50K (estimated)
- SLO burn: 5% of monthly error budget consumed

## Timeline (UTC)
- 12:34: Alert fired (correctness score < 0.7)
- 12:35: On-call acknowledged
- 12:42: Category identified (prompt regression)
- 12:50: Rollback initiated
- 13:00: Quality score returned to baseline
- 13:45: All systems confirmed recovered

## Root Cause
The CI/CD pipeline merged a prompt template change that was not validated
against the eval suite. The new template introduced a typo that caused the
LLM to misread the system prompt, leading to fabricated product specs.

## Resolution
Rolled back the prompt template to the previous version. Added the new
template to the eval suite as a regression test.

## Action Items
- [ ] Add prompt-template validation to CI pipeline (Owner: Carol, Due: 2026-07-30)
- [ ] Add regression test for the failed template (Owner: Dave, Due: 2026-07-25)
- [ ] Review other recent prompt template deploys (Owner: Alice, Due: 2026-07-26)
```

---

## 7. When to Page the Team

Page if:
- SEV-1 or SEV-2 alert fired
- Auto-mitigation didn't resolve in 5 minutes
- Customer reports align with the alert (not just one customer)
- Data breach suspected
- Cost impact > $1,000
- Quality degradation affects >10% of traffic

Don't page if:
- SEV-3 alert that auto-resolved in <5 min
- Test environment
- Internal tool with no user impact

---

## 8. Antipatterns

### 8.1 Antipattern 1: Investigating before acknowledging

```python
# ❌ Acknowledge the alert first, then investigate
print("Hmm, interesting. Let me check the logs...")  # page still paged

# ✅ Acknowledge first; investigation comes second
page.acknowledge()  # stops the paging
# THEN investigate
```

### 8.2 Antipattern 2: Trying to fix without diagnosing

```python
# ❌ Random fix attempts
model = "gpt-4o-turbo"  # maybe this fixes it?
prompt = "..."  # maybe a different prompt?
restart_service()  # maybe that works?

# ✅ Diagnose first, then apply the targeted fix
# 1. Look at traces (Note 03 patterns)
# 2. Identify the category (Note 01)
# 3. Apply the right rollback (Note 03 section 3)
```

### 8.3 Antipattern 3: Skipping status communication

```python
# ❌ Investigating silently for 30 minutes
# Users / stakeholders don't know what's happening

# ✅ First status update within 5 minutes, updates every 15 minutes
```

### 8.4 Antipattern 4: Not paging when you should

```python
# ❌ "I can handle this"
# 2 hours later: incident still ongoing, no help, no one knows

# ✅ Page when SEV-1/SEV-2; page even if you think you can handle it
# Senior engineers can opt out if not needed; better to page unnecessarily than miss
```

### 8.5 Antipattern 5: Skipping the rollback to find the "perfect" fix

```python
# ❌ Investigation continues; users still affected
# 1 hour later: rollback might be 5 minutes but you spent 60 investigating

# ✅ Rollback first, investigate later
# Rollback restores service; postmortem investigates root cause
```

---

## 🎯 Key Takeaways

- The OODA loop for AI: Observe (alert), Orient (traces), Decide (rollback), Act (apply + page).
- The 5-minute diagnostic checklist: ack → category → scope → trigger → mitigate.
- Rollback levels: prompt → model → deploy → kill switch. Apply the smallest first.
- Status communication: first update < 5 min, updates every 15 min, all-hands final update.
- The incident note is the source of truth for the postmortem.
- Page when SEV-1/SEV-2; even if you can handle it, others may need to know.
- Avoid investigating before acking, random fixes, skipping status updates, not paging, and skipping rollback.

## References

- Google SRE Book — Handling the Incident — [sre.google/sre-book/managing-incidents](https://sre.google/sre-book/managing-incidents/)
- Atlassian Incident Handbook — [atlassian.com/incident-management](https://www.atlassian.com/incident-management)
- PagerDuty Incident Response — [response.pagerduty.com](https://response.pagerduty.com/)
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/01 - AI Incident Taxonomy - What Can Break|Note 01 — Taxonomy]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/02 - Detection - Alerts, Metrics, and Anomaly Detection|Note 02 — Detection]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/04 - Resolution Patterns and Resilience Engineering|Note 04 — Resolution Patterns]]
- [[09 - MLOps y Produccion/39 - Production Incident Response for AI Systems/05 - Capstone - Production Incident Response Simulation|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability|LangFuse Deep Dive]] — traces
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — drift traces
- [[06 - Large Language Models/15 - LLM Security and Guardrails|LLM Security]] — prompt injection triage
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — failover triage