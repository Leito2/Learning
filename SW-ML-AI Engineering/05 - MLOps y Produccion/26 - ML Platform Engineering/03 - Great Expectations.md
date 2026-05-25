# ✅ Great Expectations: Data Quality for ML

## Introduction

Great Expectations (GX) is the open-source standard for data quality validation — the equivalent of unit tests for data. It defines "Expectations" (declarative assertions about data) that run against datasets and produce validation reports. For ML engineers, GX answers the question "is my training data good enough?" before you spend compute on training.

---

## Core Concepts

### Expectations

An Expectation is a declarative statement about what data should look like:

```python
import great_expectations as gx

# Connect to data
context = gx.get_context()
validator = context.sources.pandas_default.read_csv("features.csv")

# Define expectations
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=10000)
validator.expect_column_mean_to_be_between("age", min_value=18, max_value=90)
validator.expect_column_distinct_values_to_contain_set(
    "event_type", value_set=["click", "purchase", "view"]
)
validator.expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")

# Validate
results = validator.validate()
```

### Expectation Types

| Category | Examples | ML Use |
|---|---|---|
| **Completeness** | `not_null`, `no_empty_strings` | Catch missing features before training |
| **Range** | `values_between`, `mean_between` | Detect feature drift |
| **Set membership** | `distinct_values_contain_set` | Validate categorical feature values |
| **Schema** | `columns_match_ordered_list` | Catch schema changes in feature pipeline |
| **Distribution** | `kl_divergence_between` | Detect distribution shift vs training data |
| **Custom** | Any SQL/Python condition | Domain-specific quality checks |

### Data Docs

GX auto-generates HTML documentation showing every expectation and whether it passed:

```
┌─────────────────────────────────────────────────┐
│  Data Docs — fraud_features_2024-12-01          │
│                                                 │
│  ✅ user_id: not_null (100% pass)               │
│  ✅ amount: between 0-10000 (99.8% pass)        │
│  ❌ event_type: set membership FAILED           │
│     Unexpected values found: ['refund', 'swap'] │
│                                                 │
│  Overall: 23/24 expectations passed             │
└─────────────────────────────────────────────────┘
```

---

## ML Pipeline Integration

```
Data Source → GX Validation → {Pass: Train, Fail: Alert + Block}
```

- **CI/CD:** Run GX checks on PRs that modify feature pipelines — block merge if expectations fail.
- **Scheduled:** Run GX on daily feature refresh — alert if data quality degrades.
- **Pre-training:** Run GX on training dataset — abort training if expectations not met (saves compute).

---

## ⚠️ Considerations

- **Expectation maintenance:** As models evolve, valid value ranges change. Review expectations quarterly.
- **False positives at scale:** An expectation "event_count > 0" will always have 0.01% failures due to system glitches. Use percentage-based thresholds, not absolute.

---

## References

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
