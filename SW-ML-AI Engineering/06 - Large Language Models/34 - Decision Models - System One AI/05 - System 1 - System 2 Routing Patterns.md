# 🧠 05 - System 1 / System 2 Routing Patterns

Sending every support message to an LLM agent is accurate and ruinously expensive at scale; sending every message through a classifier is cheap and quietly wrong on the hard cases. The engineering answer is a **cascade**: a fast, calibrated decision model handles what it is confident about, and only the uncertain, risky, or out-of-distribution cases are escalated to a slower, smarter System 2. The design questions are quantitative — where to put the threshold, what the cascade costs, what it does to your p95 — and they all depend on calibrated probabilities.

## 🎯 Learning Objectives
- Frame System 1 / System 2 as **selective prediction** and **model cascades**
- Choose thresholds with a **risk–coverage curve** and an explicit **cost model**
- Add **asymmetric safety nets**, **multi-intent margins**, and **OOD detection** as escalation triggers
- Predict the effect of escalation on **latency percentiles** (mixture distributions)
- Design a bounded **System 2 agent** and decide when it is *not* worth having
- Close the loop: System 2 outcomes and human feedback **improve System 1** over time

## Introduction

Cascades are an old idea with a new cost structure. The Viola–Jones face detector (2001) used a cascade of increasingly expensive classifiers so that most image regions were rejected cheaply. In the LLM era, *FrugalGPT* (Chen, Zaharia & Zou, 2023) showed that cascading from cheaper to more expensive LLMs, stopping when a scorer judged an answer reliable, could match the best model's performance at a fraction of the cost. Decision models make the first stage dramatically cheaper and — crucially — give it **calibrated confidence**, which is exactly the signal a cascade needs to decide when to stop.

In P2, System 1 is Laya (multilingual, on CPU) answering four typed questions per message; System 2 is a LangGraph agent with tools (customer history, P1's fraud decisions, P3's knowledge base) running a small local LLM. The policy between them is a few lines of code; choosing its parameters is the real work, and it is all measurable on a labeled golden set.

---

## 1. The Problem and Why This Solution Exists

### Neither extreme works

| Strategy | Cost per 1k messages | Latency | Failure mode |
|---|---|---|---|
| LLM agent for everything | High (tokens × every message) | Seconds | Cost and latency scale with traffic |
| Classifier for everything | ~0 | Milliseconds | Silent errors on negations, multi-intent, OOD |
| **Cascade (S1 → S2 when uncertain)** | Low (tokens × escalated fraction) | ms for most, seconds for few | Requires calibrated confidence and good thresholds |

### Selective prediction

Geifman and El-Yaniv (*Selective Classification for Deep Neural Networks*, 2017) formalized the idea of a classifier that may **abstain**: for a confidence function $\kappa(x)$ and threshold $\tau$, predict when $\kappa(x) \ge \tau$, otherwise abstain. Two quantities trade off:

$$
\text{coverage}(\tau) = P\big(\kappa(X) \ge \tau\big)
\qquad
\text{risk}(\tau) = P\big(\hat{Y} \ne Y \;\big|\; \kappa(X) \ge \tau\big)
$$

In a cascade, "abstain" means **escalate to System 2** rather than giving up.

---

## 2. Conceptual Deep Dive

### 2.1 Choosing the threshold with a cost model

Let $c_1$ and $c_2$ be the per-message costs of System 1 and System 2, $e_1(\tau)$ the error rate of System 1 on the messages it keeps, $e_2$ the error rate of System 2 on escalated messages, and $\lambda$ the cost of an error (in the same units). The expected cost per message is:

$$
\mathbb{E}[C](\tau) = c_1 + \big(1 - \text{cov}(\tau)\big)\,c_2 + \lambda\Big[\text{cov}(\tau)\,e_1(\tau) + \big(1 - \text{cov}(\tau)\big)\,e_2\Big]
$$

Two practical formulations:

- **Constraint form:** maximize coverage subject to System 1 precision ≥ target (e.g., 97%) — easy to explain to stakeholders.
- **Cost form:** minimize $\mathbb{E}[C]$ — makes the error cost explicit.

Both require **calibrated** confidence; otherwise, the risk–coverage curve measured offline won't hold online ([[01 - How Decision Models Work|calibration]]).

### 2.2 Escalation triggers beyond "low confidence"

| Trigger | Rule (illustrative) | Why |
|---|---|---|
| **Low confidence** | $\max_k p_k < \tau$ | Classic selective prediction |
| **Multi-intent** | $p_{(1)} - p_{(2)} < m$ (small margin between top two) | Message plausibly needs two teams |
| **Safety net (asymmetric)** | $P(\text{urgency} \ge \text{high}) > \tau_u$ with $\tau_u$ **low** (e.g., 0.15) | Missing an urgent case costs far more than an extra escalation |
| **Out-of-distribution** | High normalized entropy, unknown language, very short/very long message | Decision models fail confidently OOD |
| **Business rules** | Customer tier, amount mentioned, regulatory keywords | Policy, not model |

The safety net is deliberately **not symmetric**: it trades many cheap false escalations for very few missed urgent cases — exactly the trade an expected-cost model with a large $\lambda$ for missed urgency recommends.

### 2.3 What escalation does to latency percentiles

Overall latency is a **mixture**: with escalation rate $\varepsilon$,

$$
P(L > x) = (1 - \varepsilon)\,P(L_1 > x) + \varepsilon\,P(L_2 > x)
$$

If System 2 takes seconds and System 1 milliseconds, then once $\varepsilon$ **exceeds** 5%, the **overall p95 is essentially System 2's latency** (at exactly 5% it sits on the boundary between the two). Two consequences: report latency **per path** (System 1 p95 and System 2 p95) rather than one blended number, and treat the escalation rate as an SLO-relevant metric — a drift that pushes $\varepsilon$ from 4% to 8% moves the blended p95 by orders of magnitude.

### 2.4 Designing System 2

System 2 is not "the LLM, but with more prompt". It should be a **bounded agent** ([[../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/00 - Welcome to LangGraph Deep Patterns|LangGraph Deep Patterns]]):

- **Inputs:** the message, System 1's full distributions, and the escalation reason.
- **Tools that add information System 1 lacks:** customer history, the fraud decision for a disputed payment (P1), knowledge-base lookups (P3).
- **Structured output** validated against a schema; **max steps** and **timeouts**; a deterministic **fallback** (human triage queue).
- **Evaluation on escalated cases only:** System 2 must beat System 1 *on the cases it receives*. If it doesn't, the cheaper design is "escalate to a human" — measure before keeping the agent.

```mermaid
graph TD
    M[Message] --> S1[System 1: Laya<br/>route · urgency · needs_human · needs_kb]
    S1 --> P{Policy}
    P -->|safety net: P(urgent) > 0.15| S2
    P -->|margin < m or conf < τ or OOD| S2[System 2: LangGraph agent<br/>tools: history · P1 · P3]
    P -->|confident| D[Dispatch]
    S2 -->|structured decision| D
    S2 -->|timeout / invalid| H[Human triage]
    D --> F[(Outcomes + human feedback)]
    F -->|retrain + recalibrate| S1
```

### 2.5 The improvement loop

Every escalated case that System 2 (or a human) resolves becomes a **labeled example** in exactly the region where System 1 is weakest. Periodically fine-tuning and recalibrating System 1 on these examples shifts the risk–coverage curve: coverage rises at the same precision, and the escalation rate — and cost — fall. This is the most persuasive chart in P2's README: **coverage over time**.

---

## 3. Production Reality

### Monitoring the cascade

| Metric | Why |
|---|---|
| Escalation rate (total and by reason) | Cost and blended latency driver; drift detector |
| System 1 precision on kept cases (from feedback) | Threshold still valid? |
| System 2 accuracy on escalations; fallback rate | Is the agent earning its cost? |
| Latency p95 **per path** | Honest SLOs |
| Cost per 1k messages (tokens × escalations) | The business case |
| Confidence distribution / OOD rate | Early warning of distribution shift |

These are exactly the panels of P2's **R4 · System 2** dashboard ([[../../09 - MLOps y Produccion/45 - Grafana and Latency Engineering for ML Systems/05 - LLM Observability Dashboards|LLM Observability Dashboards]]).

### Recalibrate when the world changes

A new product launch, a new language, or a new fraud wave changes the input distribution; calibration fitted last month may no longer hold, and the threshold's promised precision silently degrades. Recompute ECE on recent labeled feedback on a schedule and alert when it drifts.

Caso real: FrugalGPT's experiments showed that a learned cascade over LLM APIs could match the best individual model's accuracy at a large fraction lower cost — the same economic logic P2 applies with a decision model as the first stage, whose per-message cost is close to zero.

Caso real: LLM gateways increasingly put a fast classifier in front of model selection ("auto-routing"): simple requests go to a small, cheap model; complex ones to a frontier model. LiteLLM's benchmark of a decision model as that router (≈127 ms median vs ≈688 ms for Claude Haiku as a classifier) is the System 1 / System 2 pattern applied to model choice ([[../19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]]).

---

## 4. Code in Practice

### The policy

```python
from dataclasses import dataclass

@dataclass
class Policy:
    tau: float = 0.80          # min calibrated confidence for the route
    margin: float = 0.15       # min gap between top-2 routes
    tau_urgent: float = 0.15   # asymmetric safety net on P(urgency >= high)
    max_entropy: float = 0.85  # normalized entropy above this → OOD

def decide(answers: dict, lang_supported: bool, pol: Policy) -> tuple[str, str]:
    route_p = answers["route"]["probabilities"]
    ranked = sorted(route_p.items(), key=lambda kv: kv[1], reverse=True)
    (top, p1), (_, p2) = ranked[0], ranked[1]
    urgency = answers["urgency"]["probabilities"]
    p_urgent = urgency.get("2", 0) + urgency.get("3", 0)
    if p_urgent > pol.tau_urgent:
        return "system2", "urgency_safety_net"
    if not lang_supported or normalized_entropy(route_p) > pol.max_entropy:
        return "system2", "ood"
    if p1 - p2 < pol.margin:
        return "system2", "multi_intent"
    if p1 < pol.tau:
        return "system2", "low_confidence"
    return top, "system1"
```

### Picking τ from the golden set

```python
def risk_coverage(rows, taus):   # rows: (calibrated_conf, correct) for System 1 on the validation set
    out = []
    for t in taus:
        kept = [c for conf, c in rows if conf >= t]
        cov = len(kept) / len(rows)
        prec = sum(kept) / len(kept) if kept else 1.0
        out.append((t, cov, prec))
    return out

best = max((r for r in risk_coverage(val_rows, [i / 100 for i in range(50, 100)]) if r[2] >= 0.97),
           key=lambda r: r[1])                  # max coverage subject to precision ≥ 97%
```

### ❌/✅ Reporting latency

```text
❌ "Router p95 = 2.4 s"          (blended: dominated by the 7% of escalated messages)
✅ "System 1 p95 = 180 ms on 93% of messages; System 2 p95 = 2.9 s on 7%; escalation rate monitored"
```

### 📦 Compression code: thresholds, cost, and the latency mixture

```python
# 📦 Compression code: risk–coverage, expected cost per message, blended vs per-path p95
# Covers: selective prediction with calibrated confidence, cost model, mixture percentiles
import random
import statistics

random.seed(21)
rows = []                                   # (calibrated confidence, System 1 correct?) — calibrated by construction
for _ in range(20_000):
    conf = random.betavariate(5, 1.5)
    rows.append((conf, random.random() < conf))

C1, C2, LAMBDA, E2 = 0.00001, 0.004, 0.05, 0.08     # USD per message S1/S2, cost of an error, S2 error rate
print(" tau  coverage  S1-precision  E[cost]/1k msgs")
for tau in (0.0, 0.6, 0.75, 0.85, 0.95):
    kept = [c for conf, c in rows if conf >= tau]
    cov = len(kept) / len(rows)
    e1 = 1 - (sum(kept) / len(kept) if kept else 1)
    cost = C1 + (1 - cov) * C2 + LAMBDA * (cov * e1 + (1 - cov) * E2)
    print(f"{tau:4.2f}  {cov:8.1%}  {1 - e1:12.1%}  ${cost * 1000:8.2f}")

def p95(xs): return statistics.quantiles(xs, n=20)[18]
for eps in (0.02, 0.05, 0.10):
    lat = [random.lognormvariate(-1.9, 0.3) if random.random() > eps else random.lognormvariate(1.0, 0.3)
           for _ in range(20_000)]              # S1 ~150 ms, S2 ~2.7 s (seconds)
    print(f"escalation {eps:4.0%}: blended p95 = {p95(lat) * 1000:7.0f} ms")
# ¡Sorpresa! 2% → 5% barely moves the blended p95; 10% multiplies it ~8× — past 5% the p95 IS System 2's latency.
```

---

## 🎯 Key Takeaways
- System 1 / System 2 is a **cascade** built on **selective prediction**: keep confident cases, escalate the rest.
- Pick thresholds from a **risk–coverage curve** (max coverage at a precision target) or an explicit **cost model** — both need **calibrated** confidence.
- Escalate on more than low confidence: **multi-intent margins**, **asymmetric safety nets** for high-cost errors, **OOD** signals, business rules.
- Latency is a **mixture**: above 5% escalation, blended p95 ≈ System 2's latency — report **per path** and monitor escalation rate.
- System 2 must be a **bounded agent** that beats System 1 **on escalated cases**; otherwise route to humans.
- Feed System 2 and human outcomes back to **retrain and recalibrate** System 1 — coverage up, cost down over time.

## References
- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks* (NeurIPS 2017)
- Chen, Zaharia & Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* (2023)
- Viola & Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features* (CVPR 2001)
- Daniel Kahneman, *Thinking, Fast and Slow* (2011)
- LiteLLM blog — *JEV classifier: 5.43× as fast as Haiku* (auto-router benchmark)
- [[04 - DIY - Classification Head on a Small Qwen|Previous: DIY Classification Head]] · [[00 - Welcome to Decision Models|Course welcome]]
