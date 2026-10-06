# 🧱 02 - Grafana Dashboards as Code

A dashboard built by clicking in the UI is a production artifact with no code review, no history, and no reproducibility — it lives in one container's database until someone deletes the volume. For a portfolio project it is worse: whoever clones your repo sees an empty Grafana. Dashboards as code fix all of that: datasources and dashboards are **provisioned** from files in Git, load automatically with `docker compose up`, and can be generated programmatically so they stay consistent across services.

## 🎯 Learning Objectives
- Provision **datasources and dashboards** from files with stable UIDs and folders
- Design dashboards around **questions** (health, where time goes, model behavior) rather than metrics
- Use **variables**, `$__rate_interval`, **heatmaps** of histogram buckets, and **annotations**
- Link metrics to traces with **exemplars** and to business data with a **PostgreSQL** datasource
- Generate dashboards programmatically (JSON from Python, Grafonnet/Jsonnet, Foundation SDK, Terraform)
- Ship P1's D1–D5 dashboards so they load on first `docker compose up`

## Introduction

Grafana renders queries against datasources (Prometheus, Tempo, Loki, PostgreSQL…) into panels arranged on dashboards. Everything about a dashboard — panels, queries, layout, variables, links — is a JSON document. Grafana can load such documents and datasource definitions at startup through **provisioning**: YAML files in `/etc/grafana/provisioning/` point at directories of JSON dashboards, which Grafana watches and keeps in sync.

That makes dashboards ordinary code: reviewed in pull requests, versioned with the service they observe, and reproducible on any machine. The second half of the discipline is **design**. A good dashboard answers one question at a glance and lets you drill down; a bad one shows forty graphs and no conclusion. The P1 plan defines five dashboards, each named after its question — *Is it healthy? Where does the time go? What happens inside Kafka/Flink? Is the model still working? Can the explainer keep up?*

---

## 1. The Problem and Why This Solution Exists

### Click-ops dashboards decay

| Problem | Consequence |
|---|---|
| Not in Git | No review, no history, no rollback; lost with the container volume |
| Hand-copied panels across services | Inconsistent queries, buckets, and units; drift |
| Hard-coded datasource names/IDs | Break when moved between environments |
| One giant dashboard | Nobody can tell in 5 seconds whether things are OK |

### Questions first

The design rule used throughout this course: **one dashboard per question; the top-left panel answers it**. Everything else on the dashboard exists to explain *why* the answer is what it is.

---

## 2. Conceptual Deep Dive

### 2.1 Provisioning

```text
observability/grafana/
├── provisioning/
│   ├── datasources/datasources.yaml      # Prometheus, PostgreSQL (and Tempo in 'full')
│   └── dashboards/dashboards.yaml        # provider: where to find JSON files, folder name
└── dashboards/
    ├── d1-overview.json
    ├── d2-latency-breakdown.json
    ├── d3-kafka-flink.json
    ├── d4-model-business.json
    └── d5-explainer-llm.json
```

Datasources get **fixed UIDs** in YAML; dashboards reference datasources by those UIDs. Dashboards themselves have stable `uid` values so links between them never break. With `allowUiUpdates: false`, edits must go through Git — the UI becomes read-only for provisioned dashboards (export a modified dashboard's JSON and commit it).

### 2.2 Query mechanics worth knowing

**`$__rate_interval`.** A `rate()` window must contain enough samples to be meaningful and must adapt to the dashboard's zoom level. Grafana computes:

$$
\texttt{\$\_\_rate\_interval} = \max\big(\texttt{\$\_\_interval} + T_{\text{scrape}},\; 4 \cdot T_{\text{scrape}}\big)
$$

With a 5 s scrape interval, the window is at least 20 s — never a broken `rate()` over a single sample. Use it in panel queries; use fixed windows (e.g., `[5m]`) in SLO recording rules.

**Latency heatmaps.** A heatmap of histogram buckets shows the **whole distribution over time** — bimodality, slow drifts of the tail, batch-boundary artifacts — that a single p95 line hides:

```promql
sum by (le) (increase(fraud_e2e_latency_seconds_bucket[$__rate_interval]))
```

with the panel set to "Heatmap" and the data format "Time series buckets".

**Variables.** `label_values(fraud_decisions_total, model_version)` populates a dropdown; queries use `{model_version=~"$model_version"}`. Variables make one dashboard serve every model version, stage, or topic.

**Annotations.** Vertical markers for events: deployments, model promotions (`make promote` writes an annotation through the Grafana API or a row in Postgres that an annotation query reads), chaos injections. "p95 jumped right after promotion v8" is the fastest root-cause analysis there is.

### 2.3 Linking signals

- **Exemplars → traces:** enable exemplars on the Prometheus datasource and point them at Tempo; dots on the latency panel open the trace of that exact payment ([[../34 - OpenTelemetry for AI Engineers/03 - OTLP Exporters - Phoenix Tempo Jaeger and Beyond|OTLP exporters]]).
- **Data links:** a table row with `payment_id` links to the API (`/decisions/{id}`) or to a filtered dashboard.
- **PostgreSQL datasource:** business and model-quality panels (confusion by model, champion vs challenger, fraud amount blocked) query the auditor's views directly — metrics systems are poor at joins with late-arriving labels; SQL is good at them.

### 2.4 Dashboard design pattern

```mermaid
graph TD
    Q[Question in the title<br/>"Is the fraud pipeline healthy?"] --> R1[Row 1: answer<br/>SLO status · p95 · throughput · lag]
    R1 --> R2[Row 2: why<br/>p95 by stage · heatmap · batch size]
    R2 --> R3[Row 3: context<br/>decision mix · model version · annotations]
    R3 --> L[Links to deeper dashboards<br/>D2 latency · D3 Kafka/Flink · D4 model]
```

| Dashboard (P1) | Question | Top-left panel |
|---|---|---|
| D1 Overview | Is it healthy? | SLI: % decisions < 80 ms (stat, green ≥ 95%) |
| D2 Latency Breakdown | Where does the time go? | p95 by stage (stacked) |
| D3 Kafka & Flink | Is streaming keeping up? | Consumer lag by group |
| D4 Model & Business | Is the model still working? | Score PSI vs training; recall (from Postgres) |
| D5 Explainer & LLM | Can explanations keep up? | Explanations/s vs REVIEW+BLOCK/s; fallback rate |

---

## 3. Production Reality

### Generating dashboards

Hand-writing JSON for five dashboards is fine; for fifty it isn't. Options:

| Tool | Language | Notes |
|---|---|---|
| Plain Python dict → JSON | Python | Zero dependencies; good for a handful of templated panels (see compression code) |
| Grafonnet | Jsonnet | Long-standing library for Grafana JSON |
| Grafana Foundation SDK | Go, TypeScript, Python, … | Typed builders generated from Grafana's schemas |
| Terraform `grafana` provider | HCL | Manage dashboards, folders, alerts, datasources as infrastructure ([[../../10 - Cloud, Infra y Backend/23 - Infrastructure as Code/01 - Terraform Fundamentals - HCL, State and Resource Graph|Terraform]]) |

### On a laptop

Grafana itself is light (~150 MB), but browsers rendering many live panels are not. P1's plan keeps Grafana **off during benchmarks** and on for analysis and demos; dashboards are provisioned, so turning it on is instant. Anonymous viewer access (`GF_AUTH_ANONYMOUS_ENABLED=true`) avoids login friction locally — never in a shared deployment.

Caso real: platform teams that standardize "golden signal" dashboards generate them from code for every service (same panel layout, same buckets, same SLO stat in the top-left). Responders can then read *any* service's dashboard without learning its layout — consistency is the feature.

Caso real: for a portfolio, provisioned dashboards are what turn "trust me, I monitored it" into "clone, `docker compose up`, open localhost:3000" — the dashboards appear populated as soon as the generator starts, which is exactly what a reviewer will try.

---

## 4. Code in Practice

### Provisioning files

```yaml
# provisioning/datasources/datasources.yaml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prom
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    jsonData: { timeInterval: 5s }          # scrape interval → correct $__rate_interval
  - name: Audit
    uid: pg-audit
    type: grafana-postgresql-datasource       # type name differs in older Grafana versions ("postgres")
    url: postgres:5432
    user: grafana_ro
    secureJsonData: { password: ${GRAFANA_PG_PASSWORD} }
    jsonData: { database: fraud, sslmode: disable }
```

```yaml
# provisioning/dashboards/dashboards.yaml
apiVersion: 1
providers:
  - name: fraud
    folder: Fraud Platform
    type: file
    allowUiUpdates: false                    # edits go through Git
    options: { path: /var/lib/grafana/dashboards }
```

```yaml
# docker-compose.yml (excerpt)
grafana:
  image: grafana/grafana:latest              # pin a version in the real project
  environment:
    GF_AUTH_ANONYMOUS_ENABLED: "true"
    GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
  volumes:
    - ./observability/grafana/provisioning:/etc/grafana/provisioning:ro
    - ./observability/grafana/dashboards:/var/lib/grafana/dashboards:ro
  ports: ["3000:3000"]
  profiles: ["ops", "full"]                  # off in the 'core' benchmark profile
```

### ❌/✅ Panel queries

```promql
# ❌ Fixed tiny window: breaks when zoomed out, noisy when zoomed in
rate(fraud_decisions_total[10s])

# ✅ Adapts to zoom and scrape interval
sum by (decision) (rate(fraud_decisions_total[$__rate_interval]))
```

### 📦 Compression code: generate and validate a dashboard as code

```python
# 📦 Compression code: build P1's D1 dashboard JSON from Python and validate it
# Covers: stable uid, datasource by uid, SLO stat, p95-by-stage, heatmap, layout sanity checks
import json

DS = {"type": "prometheus", "uid": "prom"}

def panel(pid, title, kind, expr, x, y, w=12, h=8, **extra):
    return {"id": pid, "type": kind, "title": title, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"refId": "A", "expr": expr, "datasource": DS}], **extra}

sli = ('sum(rate(fraud_e2e_latency_seconds_bucket{le="0.08"}[$__rate_interval]))'
       ' / sum(rate(fraud_e2e_latency_seconds_count[$__rate_interval]))')
dashboard = {
    "uid": "fraud-d1-overview", "title": "D1 · Is the fraud pipeline healthy?", "schemaVersion": 39,
    "time": {"from": "now-30m", "to": "now"}, "refresh": "10s",
    "panels": [
        panel(1, "SLI: decisions < 80 ms", "stat", sli, 0, 0, 6, 6,
              fieldConfig={"defaults": {"unit": "percentunit", "thresholds": {"mode": "absolute", "steps": [
                  {"color": "red", "value": None}, {"color": "green", "value": 0.95}]}}}),
        panel(2, "Throughput (decisions/s)", "stat", "sum(rate(fraud_decisions_total[$__rate_interval]))", 6, 0, 6, 6),
        panel(3, "p95 by stage", "timeseries",
              "histogram_quantile(0.95, sum by (le, stage) (rate(fraud_stage_latency_seconds_bucket[$__rate_interval])))",
              12, 0, 12, 6, fieldConfig={"defaults": {"unit": "s"}}),
        panel(4, "End-to-end latency distribution", "heatmap",
              "sum by (le) (increase(fraud_e2e_latency_seconds_bucket[$__rate_interval]))", 0, 6, 24, 8),
    ],
}

def overlaps(a, b):
    ga, gb = a["gridPos"], b["gridPos"]
    return not (ga["x"] + ga["w"] <= gb["x"] or gb["x"] + gb["w"] <= ga["x"] or
                ga["y"] + ga["h"] <= gb["y"] or gb["y"] + gb["h"] <= ga["y"])

ids = [p["id"] for p in dashboard["panels"]]
assert len(ids) == len(set(ids)), "duplicate panel ids"
assert all(p["gridPos"]["x"] + p["gridPos"]["w"] <= 24 for p in dashboard["panels"]), "panel exceeds 24 columns"
assert not any(overlaps(a, b) for i, a in enumerate(dashboard["panels"]) for b in dashboard["panels"][i + 1:])
print(json.dumps(dashboard)[:120] + "…")
print(f"OK: {len(ids)} panels, uid={dashboard['uid']} — write to observability/grafana/dashboards/d1-overview.json")
# ¡Sorpresa! A 10-line validator catches the layout/ID bugs that otherwise only show up as a broken dashboard in the UI.
```

---

## 🎯 Key Takeaways
- **Provision** datasources and dashboards from files; stable **UIDs**; `allowUiUpdates: false` keeps Git as the source of truth.
- Design **one dashboard per question**, answer it in the top-left panel, explain it below.
- Use `$__rate_interval` in panels; it equals $\max(\$\_\_interval + T_{scrape}, 4T_{scrape})$.
- **Heatmaps** of histogram buckets reveal distribution shapes a p95 line hides.
- Link signals: **exemplars** to traces, **annotations** for deploys/promotions, **PostgreSQL** for label-joined business metrics.
- Generate dashboards with code (Python, Grafonnet, Foundation SDK, Terraform) and validate them in CI.

## References
- Grafana documentation — Provisioning, dashboard JSON model, variables, `$__rate_interval`, heatmaps, exemplars, annotations, PostgreSQL datasource
- Grafana Foundation SDK · Grafonnet · Terraform Grafana provider
- [[01 - Prometheus Metrics Design for ML|Previous: Prometheus Metrics Design]] · [[03 - SLOs, Alerting and Burn Rates|Next: SLOs, Alerting and Burn Rates]]
