# 📐 dbt for ML Data Transformation

## Introduction

dbt (data build tool) is the standard for analytics engineering — but it matters for ML engineers because feature data doesn't appear from nowhere. Before you can train a model, someone (often you) must transform raw event tables into clean, aggregated, feature-rich tables. dbt brings software engineering practices to SQL transformations: version control, testing, documentation, and lineage tracking.

For ML engineers working with data warehouses (BigQuery, Snowflake, Redshift), dbt is the bridge between "the data team's tables" and "the feature table my model needs." Understanding dbt means understanding how production feature data is built, tested, and maintained.

---

## 1. 🧠 What dbt Does (and Doesn't Do)

### dbt Does

- Transform data using **SQL + Jinja templating** — write `SELECT` statements that become tables/views
- **Test data quality** with built-in tests (unique, not_null, accepted_values, relationships)
- **Document** data lineage and column definitions automatically
- **Version control** transformations alongside application code
- **Incremental processing** — only process new data, not full table rebuilds

### dbt Does NOT Do

- Extract or load data (it assumes data is already in your warehouse)
- Stream processing (batch-oriented on warehouse SQL)
- Model training (it transforms data, not trains models)
- Orchestration (you need Airflow/Prefect/Dagster to schedule dbt runs)

### The dbt Philosophy

```
┌──────────────────────────────────────────────────────────────┐
│                    dbt TRANSFORMATION FLOW                    │
│                                                              │
│  Raw Data (source) → Base Models → Intermediate → Feature Tables
│                                                              │
│  models/                                                     │
│  ├── staging/           # 1:1 with source tables (light clean) │
│  │   ├── stg_users.sql                                       │
│  │   └── stg_events.sql                                     │
│  ├── intermediate/      # Complex joins, aggregations        │
│  │   └── int_user_sessions.sql                               │
│  └── marts/             # Business/ML-ready tables           │
│      └── ml_features/                                        │
│          ├── user_features.sql    ← What your model consumes │
│          └── session_features.sql                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. ⚙️ dbt Core Concepts

### Models: SQL SELECT Statements

```sql
-- models/marts/ml_features/user_features.sql
{{
  config(
    materialized='table',
    partition_by={'field': 'dt', 'data_type': 'date'}
  )
}}

WITH user_events AS (
  SELECT
    user_id,
    DATE(ts) AS dt,
    COUNT(*) AS event_count,
    COUNT(DISTINCT session_id) AS session_count,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_count,
    AVG(amount) AS avg_amount,
    SUM(amount) AS total_amount,
    MAX(amount) AS max_amount
  FROM {{ ref('stg_events') }}  -- Reference to upstream model
  WHERE ts >= CURRENT_DATE - INTERVAL '30 days'
  GROUP BY user_id, DATE(ts)
)

SELECT * FROM user_events
```

### Tests: Data Quality Guardrails

```yaml
# models/marts/ml_features/schema.yml
version: 2

models:
  - name: user_features
    description: "Aggregated user-level ML features per day"
    columns:
      - name: user_id
        tests:
          - not_null
          - unique:
              config:
                where: "dt = CURRENT_DATE()"
      - name: event_count
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
      - name: avg_amount
        tests:
          - not_null
      - name: purchase_count
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "purchase_count <= event_count"
```

### Jinja Templating for Dynamic SQL

```sql
-- macros/feature_agg.sql
{% macro rolling_aggregate(column, window_days=7) %}
  AVG({{ column }}) OVER (
    PARTITION BY user_id
    ORDER BY dt
    ROWS BETWEEN {{ window_days - 1 }} PRECEDING AND CURRENT ROW
  ) AS {{ column }}_rolling_{{ window_days }}d
{% endmacro %}

-- Used in models:
SELECT
  user_id, dt,
  {{ rolling_aggregate('amount', 7) }},
  {{ rolling_aggregate('amount', 30) }}
FROM {{ ref('user_events') }}
```

---

## 3. 🔄 Incremental Processing for ML Features

Feature tables grow daily. Without incremental processing, every dbt run rebuilds the entire table from scratch:

```sql
-- models/marts/ml_features/user_features.sql
{{
  config(
    materialized='incremental',
    unique_key=['user_id', 'dt'],
    on_schema_change='append_new_columns'
  )
}}

SELECT
  user_id,
  DATE(ts) AS dt,
  COUNT(*) AS event_count,
  SUM(amount) AS total_amount
FROM {{ ref('stg_events') }}

{% if is_incremental() %}
  -- Only process yesterday's new data
  WHERE DATE(ts) = CURRENT_DATE - 1
{% endif %}

GROUP BY user_id, DATE(ts)
```

| Materialization | When to Use |
|---|---|
| **table** | Small datasets (<1M rows), rebuilt every run |
| **view** | Always up-to-date, no storage cost, slower queries |
| **incremental** | Large datasets — only process new/changed records |
| **ephemeral** | CTE-like, not stored — used as building block for other models |

---

## 4. 📊 dbt for ML: The Feature Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              dbt ML FEATURE PIPELINE                          │
│                                                              │
│  Sources (BigQuery / Snowflake)                              │
│  ├── raw.orders                                               │
│  ├── raw.users                                                │
│  └── raw.events                                               │
│       │                                                      │
│       ▼  dbt run (daily)                                     │
│                                                              │
│  Staging                                                     │
│  ├── stg_orders (cleaned, typed)                             │
│  ├── stg_users (PII masked)                                  │
│  └── stg_events (deduplicated)                               │
│       │                                                      │
│       ▼                                                      │
│  Feature Marts                                               │
│  ├── user_features (30-day rolling aggregations)             │
│  ├── product_features (category embeddings)                  │
│  └── session_features (clickstream patterns)                 │
│       │                                                      │
│       ▼  dbt test (runs after every `dbt run`)               │
│                                                              │
│  Quality Gates:                                              │
│  ├── user_id NOT NULL                                        │
│  ├── event_count >= 0                                        │
│  ├── avg_amount BETWEEN 0 AND 10000                          │
│  └── row_count today >= row_count yesterday * 0.5            │
│       │                                                      │
│       ▼  Export / Train                                      │
│                                                              │
│  ML Training Pipeline reads from feature marts               │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 🔗 dbt + ML Ecosystem

| Integration | Pattern |
|---|---|
| **dbt → BigQuery ML** | dbt builds features → BigQuery ML `CREATE MODEL` on the feature table |
| **dbt → Spark** | dbt runs on warehouse → `dbt seed` exports to S3 → Spark reads Parquet |
| **dbt → MLflow** | `dbt run` completes → Airflow task triggers MLflow training run → tags dbt run ID |
| **dbt → Great Expectations** | dbt tests handle SQL-level validation; Great Expectations for distribution-level expectations |
| **dbt → Feature Store** | dbt output tables serve as offline feature store for Vertex AI/Databricks Feature Store |

---

## ⚠️ Considerations

- **dbt is SQL-only:** Custom feature transformations requiring NumPy/PyTorch must happen outside dbt. Use dbt for SQL-solvable features, Python for the rest.
- **Warehouse compute cost:** Every `dbt run` executes SQL on your warehouse (BigQuery/Snowflake). Incremental models and proper partitioning are not optional — they're cost-control measures.
- **Schema evolution:** Adding a column to a source table doesn't automatically propagate to downstream dbt models. Use `on_schema_change='append_new_columns'` and monitor schema drift.

---

## 💡 Tips

- **Use dbt's `{{ ref() }}` for automatic DAG generation:** dbt builds the execution order by tracing `ref()` calls — no manual DAG definition needed.
- **Set up CI tests for dbt models:** Run `dbt test` in CI/CD on PRs that modify models, just like you run unit tests on Python code.
- **Document features with dbt docs:** `dbt docs generate` creates a searchable data catalog — your team can discover available features without asking you.

---

## References

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt + BigQuery ML](https://docs.getdbt.com/guides/bigquery)
- [dbt Best Practices](https://docs.getdbt.com/best-practices)
