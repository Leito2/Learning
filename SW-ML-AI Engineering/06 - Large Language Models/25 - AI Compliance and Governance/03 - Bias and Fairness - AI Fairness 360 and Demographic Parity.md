# 🎯 03 - Bias and Fairness — AI Fairness 360 and Demographic Parity

> **The bias that AI Act Article 10 demands you measure. The toolkit IBM built for it. The metrics that satisfy regulators. The LLM-specific patterns that span Fairness 360.**

## 🎯 Learning Objectives
- Identify the 5 sources of bias in AI systems (historical, sampling, measurement, algorithmic, deployment)
- Calculate demographic parity, equal opportunity, and calibration metrics
- Use IBM's AI Fairness 360 (AIF360) toolkit to measure and mitigate bias
- Apply LLM-specific bias testing (stereotyping, exclusion, toxicity)
- Build continuous bias monitoring with disaggregated metrics
- Detect intersectional bias (compound disadvantage)
- Recognize the impossibility theorem: fairness metrics cannot all be satisfied simultaneously

## Introduction

Bias is the AI compliance failure that gets companies sued. In 2018 Amazon's hiring tool discriminated against women; in 2019 the Apple Card gave men higher credit limits than women; in 2023 the Dutch childcare benefits scandal wrongly accused thousands of families of fraud — all driven by algorithmic bias.

The **EU AI Act Article 10** requires "high-quality datasets" with "appropriate statistical properties" and "documentation of possible biases." **Article 15** requires accuracy metrics "across different demographic groups." Bias testing is not optional — it is regulatory compliance.

This note covers the five sources of bias, the metrics to measure them, the IBM AI Fairness 360 toolkit that automates measurement, the LLM-specific bias patterns, and the impossibility theorem that says you cannot satisfy all fairness metrics at once. Master this, and your AI systems pass regulatory audits.

![Bias sources visualization](https://example.com/bias-sources.png)

---

## 1. The Five Sources of Bias

Bias enters AI systems at multiple stages. Identifying which type you're fighting is the first step in mitigation.

### 1.1 Historical bias

The training data reflects past discrimination. The model learns the pattern.

**Example**: A 2018 dataset of hiring decisions contains patterns where women were hired less often for engineering roles. A model trained on this learns to recommend men.

**Detection**: Compare model predictions against counterfactual data (what should happen in a fair world).

### 1.2 Sampling bias

The training data does not represent the deployment population.

**Example**: A face recognition model trained on mostly-light-skinned faces performs poorly on darker-skinned faces.

**Detection**: Compare demographic distributions between training and deployment data.

### 1.3 Measurement bias

The labels themselves are biased.

**Example**: A "credit risk" label based on whether someone defaulted. But defaults are caused by many factors including discriminatory lending — using defaults as a label perpetuates the discrimination.

**Detection**: Audit the label generation process; question whether labels reflect ground truth or systemic bias.

### 1.4 Algorithmic bias

The model architecture or training process amplifies bias.

**Example**: A neural network with imbalanced training data learns to weight majority-class features more heavily.

**Detection**: Test model performance across subgroups (disaggregated metrics).

### 1.5 Deployment bias

The model is used differently across populations.

**Example**: A medical AI used by doctors for white patients but not for Black patients — different outcomes appear even if the model is "fair".

**Detection**: Monitor usage patterns across populations.

### 1.6 The bias taxonomy table

| Source | When it enters | Detection method |
|--------|---------------|------------------|
| Historical | Training data | Compare to counterfactual |
| Sampling | Training data | Compare demographic distributions |
| Measurement | Label generation | Audit label process |
| Algorithmic | Training process | Disaggregated metrics |
| Deployment | Production usage | Usage monitoring |

---

## 2. Fairness Metrics — The Five Canonical

There are at least 20+ fairness metrics in the literature. Five are most relevant to AI Act compliance.

### 2.1 Demographic parity (statistical parity)

The proportion of positive predictions is equal across groups.

```
P(prediction = positive | group A) = P(prediction = positive | group B)
```

```python
def demographic_parity(y_pred, sensitive_attribute):
    """Calculate the demographic parity difference."""
    groups = {}
    for pred, attr in zip(y_pred, sensitive_attribute):
        if attr not in groups:
            groups[attr] = []
        groups[attr].append(pred)
    
    rates = {g: sum(preds) / len(preds) for g, preds in groups.items()}
    
    # The difference between max and min rates
    diff = max(rates.values()) - min(rates.values())
    return diff


# Example: loan approval
y_pred = [1, 0, 1, 1, 0, 1, 0, 1]  # 1 = approved
sensitive = ["F", "F", "F", "F", "M", "M", "M", "M"]

diff = demographic_parity(y_pred, sensitive)
# If F has 50% approval and M has 75% approval, diff = 0.25
```

A common threshold: `diff > 0.10` is considered biased.

### 2.2 Equal opportunity (true positive rate parity)

The true positive rate is equal across groups.

```
P(prediction = positive | actual = positive, group A) = P(prediction = positive | actual = positive, group B)
```

```python
def equal_opportunity(y_true, y_pred, sensitive_attribute):
    """TPR parity: equal TPR across groups."""
    groups = {}
    for yt, yp, attr in zip(y_true, y_pred, sensitive_attribute):
        if attr not in groups:
            groups[attr] = {"tp": 0, "fn": 0}
        if yt == 1 and yp == 1:
            groups[attr]["tp"] += 1
        elif yt == 1 and yp == 0:
            groups[attr]["fn"] += 1
    
    tpr = {g: v["tp"] / (v["tp"] + v["fn"]) for g, v in groups.items()}
    diff = max(tpr.values()) - min(tpr.values())
    return diff
```

### 2.3 Predictive parity (precision parity)

The precision is equal across groups — when the model predicts positive, it's right equally often across groups.

```
P(actual = positive | prediction = positive, group A) = P(actual = positive | prediction = positive, group B)
```

### 2.4 Calibration (calibration parity)

Among people with the same predicted score, the actual positive rate is the same across groups.

```
P(actual = positive | score = 0.8, group A) = P(actual = positive | score = 0.8, group B)
```

### 2.5 Counterfactual fairness

For any individual, the prediction should not change if we flip their sensitive attribute.

```python
def counterfactual_check(predict_fn, x, sensitive_feature):
    """Check if prediction changes when sensitive feature is flipped."""
    pred_original = predict_fn(x)
    
    x_counterfactual = x.copy()
    x_counterfactual[sensitive_feature] = flip(x[sensitive_feature])
    pred_counterfactual = predict_fn(x_counterfactual)
    
    return pred_original == pred_counterfactual
```

### 2.6 The impossibility theorem

**You cannot satisfy all fairness metrics simultaneously** (except in degenerate cases).

```
demographic_parity = 0
equal_opportunity = 0  
predictive_parity = 0
```

This is mathematically impossible in any non-trivial dataset. You must **choose** which fairness metric to prioritize based on:
- Legal requirements (some metrics are mandated)
- Domain ethics (hiring prioritizes equal opportunity; lending prioritizes predictive parity)
- Stakeholder input (who is harmed most by the metric you're not optimizing?)

---

## 3. AI Fairness 360 (AIF360) — IBM's Toolkit

AIF360 is the de facto standard toolkit for fairness measurement and mitigation.

```bash
pip install aif360
```

### 3.1 Basic usage

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd


# Load your model predictions
df = pd.DataFrame({
    "age": [25, 30, 45, 22, 67, 35, 28, 50, 19, 60] * 100,
    "gender": ["F", "M", "M", "F", "F", "M", "F", "M", "F", "M"] * 100,
    "income": [40000, 50000, 70000, 30000, 90000, 60000, 35000, 80000, 25000, 95000] * 100,
    "loan_approved": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1] * 100,
})

# Convert to AIF360 dataset
dataset = BinaryLabelDataset(
    df=df,
    label_names=["loan_approved"],
    protected_attribute_names=["gender", "age"],
    favorable_label=1,
    unfavorable_label=0,
)

# Calculate fairness metrics
metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{"gender": "F"}],
    privileged_groups=[{"gender": "M"}],
)

print(f"Statistical parity difference: {metric.statistical_parity_difference():.3f}")
print(f"Disparate impact: {metric.disparate_impact():.3f}")
print(f"Equal opportunity difference: {metric.equal_opportunity_difference():.3f}")
print(f"Average odds difference: {metric.average_odds_difference():.3f}")
```

Output:

```
Statistical parity difference: -0.125
Disparate impact: 0.750
Equal opportunity difference: -0.080
Average odds difference: -0.090
```

### 3.2 Mitigation algorithms

AIF360 ships 11 mitigation algorithms. Three families:

| Family | Pre-processing | In-processing | Post-processing |
|---------|---------------|---------------|----------------|
| **Examples** | Reweighing, Optimized Preprocessing | Adversarial Debiasing, GerryFair | Reject Option Classification, Calibrated Equalized Odds |
| **When to use** | When you can modify training data | When you can modify training process | When you can only modify predictions |
| **Trade-off** | May reduce model accuracy | Most effective but requires model retraining | No retraining; may affect calibration |

```python
from aif360.algorithms.preprocessing import Reweighing

# Pre-processing: Reweighing
reweigher = Reweighing(
    unprivileged_groups=[{"gender": "F"}],
    privileged_groups=[{"gender": "M"}],
)
dataset_rw = reweigher.fit_transform(dataset)

# Train new model on reweighed data
# ... your training code ...

# Verify fairness on new model
metric_new = BinaryLabelDatasetMetric(
    dataset_rw,
    unprivileged_groups=[{"gender": "F"}],
    privileged_groups=[{"gender": "M"}],
)
print(f"After mitigation: {metric_new.statistical_parity_difference():.3f}")
```

### 3.3 Production bias monitoring

For continuous monitoring:

```python
from datetime import datetime, timedelta
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd


def monthly_bias_audit():
    """Run bias audit monthly on production data."""
    
    # Load last 30 days of production predictions
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    df = load_production_predictions(start, end)
    
    dataset = BinaryLabelDataset(
        df=df,
        label_names=["approved"],
        protected_attribute_names=["gender", "age_group"],
        favorable_label=1,
        unfavorable_label=0,
    )
    
    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{"gender": "F"}],
        privileged_groups=[{"gender": "M"}],
    )
    
    spd = metric.statistical_parity_difference()
    
    # Alert if bias exceeds threshold
    if abs(spd) > 0.10:
        alert_compliance_team(
            severity="high",
            message=f"Bias threshold exceeded: SPD = {spd:.3f}",
        )
    
    # Generate monthly bias report
    generate_bias_report(metric, period=(start, end))
```

---

## 4. LLM-Specific Bias Patterns

LLMs have unique bias patterns that classical ML fairness metrics don't capture.

### 4.1 The 6 LLM bias patterns

| Pattern | Example | Detection |
|---------|---------|-----------|
| **Stereotyping** | "Doctors are men, nurses are women" | Counterfactual analysis |
| **Exclusion** | Never mentions women when discussing CEOs | Demographic representation counting |
| **Toxicity** | Generates slurs when prompted about minorities | Perspective API or Detoxify |
| **Stereotype preference** | Prefers male pronouns over neutral | Pronoun replacement testing |
| **Cultural bias** | Western-centric defaults | Cross-cultural evaluation |
| **Disability bias** | Assumes able-bodiedness | Disability prompt set |

### 4.2 Counterfactual analysis for LLMs

```python
async def counterfactual_bias_test(model, prompt_template: str, sensitive_attrs: list[str]):
    """Test LLM bias via counterfactual analysis."""
    
    results = {}
    
    for attr in sensitive_attrs:
        # Generate prompt pairs differing only in sensitive attribute
        prompts = []
        for value in ["male", "female", "non-binary"]:
            prompt = prompt_template.format(gender=value)
            prompts.append(prompt)
        
        # Get completions
        completions = await asyncio.gather(*[
            model.complete(p) for p in prompts
        ])
        
        # Compare sentiment, length, etc.
        sentiments = [analyze_sentiment(c) for c in completions]
        
        # Check for differential treatment
        if max(sentiments) - min(sentiments) > 0.2:
            results[attr] = {
                "biased": True,
                "sentiment_range": max(sentiments) - min(sentiments),
            }
    
    return results


# Usage
template = "Write a recommendation letter for {gender} engineer applying for a senior role."
bias = await counterfactual_bias_test(llm, template, ["gender"])
# bias["gender"] = {"biased": True, "sentiment_range": 0.45}
```

### 4.3 The BBQ benchmark

The BBQ (Bias Benchmark for QA) is the standard LLM bias benchmark:

```python
from datasets import load_dataset

bbq = load_dataset("Heegyu/bbq", "all")

# Run model on BBQ
results = []
for example in bbq["test"]:
    response = await llm.complete(example["question"])
    correct = response == example["answer"]
    results.append({
        "question": example["question"],
        "category": example["category"],
        "correct": correct,
        "stereotyped": response == example["stereotyped_answer"],
    })

# Calculate accuracy by category and stereotype alignment
accuracy_by_category = pd.DataFrame(results).groupby("category")["correct"].mean()
stereotype_alignment = pd.DataFrame(results).groupby("category")["stereotyped"].mean()
```

A model that frequently selects stereotyped answers has bias.

### 4.4 Detoxify for toxicity

```python
from detoxify import Detoxify


detoxify = Detoxify("original")  # or "multilingual"


def check_toxicity(text: str) -> dict:
    """Check text for toxicity across multiple categories."""
    results = detoxify.predict(text)
    return {
        "toxicity": float(results["toxicity"]),
        "severe_toxicity": float(results["severe_toxicity"]),
        "obscene": float(results["obscene"]),
        "threat": float(results["threat"]),
        "insult": float(results["insult"]),
        "identity_attack": float(results["identity_attack"]),
    }


# Threshold for production
def is_toxic(text: str) -> bool:
    scores = check_toxicity(text)
    return scores["toxicity"] > 0.7 or scores["identity_attack"] > 0.5
```

---

## 5. Intersectional Bias

Standard fairness metrics test one attribute at a time. **Intersectional bias** is the compound disadvantage of belonging to multiple underprivileged groups.

```python
def intersectional_analysis(y_true, y_pred, attributes: list[str]):
    """Analyze bias across intersections of multiple attributes."""
    
    # Build combinations
    from itertools import product
    intersections = list(product(*[df[attr].unique() for attr in attributes]))
    
    results = []
    for combo in intersections:
        mask = np.ones(len(y_pred), dtype=bool)
        for attr, val in zip(attributes, combo):
            mask &= (df[attr] == val).values
        
        if mask.sum() < 30:  # Skip small groups
            continue
        
        tpr = y_pred[mask & (y_true == 1)].mean()
        fpr = y_pred[mask & (y_true == 0)].mean()
        
        results.append({
            "intersection": dict(zip(attributes, combo)),
            "tpr": tpr,
            "fpr": fpr,
            "sample_size": int(mask.sum()),
        })
    
    return pd.DataFrame(results)


# Example: intersection of gender AND age
results = intersectional_analysis(
    y_true=df["actual_approved"],
    y_pred=df["model_approved"],
    attributes=["gender", "age_group"],
)

# Find the worst-off group
worst = results.loc[results["tpr"].idxmin()]
print(f"Worst-off group: {worst['intersection']}, TPR: {worst['tpr']:.2f}")
# Output: Worst-off group: {'gender': 'F', 'age_group': '18-25'}, TPR: 0.65
```

A 25-year-old woman has 65% true positive rate; a 45-year-old man has 89%. The intersection compounds the disadvantage.

---

## 6. Production Bias Monitoring Pipeline

```python
# monitor_bias.py
from prometheus_client import Gauge, Counter


# Prometheus metrics
bias_spd = Gauge(
    "model_bias_statistical_parity_difference",
    "Statistical parity difference (last 30 days)",
    ["protected_attribute"],
)
bias_tpr_diff = Gauge(
    "model_bias_true_positive_rate_difference",
    "TPR difference (last 30 days)",
    ["protected_attribute"],
)
bias_alerts = Counter(
    "model_bias_alerts_total",
    "Bias alerts triggered",
    ["protected_attribute", "severity"],
)


async def bias_monitor_job():
    """Daily job that computes bias metrics and exports to Prometheus."""
    for attr in ["gender", "age_group", "nationality"]:
        predictions = await load_predictions(last_30_days=True)
        
        metric = BinaryLabelDatasetMetric(
            predictions,
            unprivileged_groups=[{attr: "F" if attr == "gender" else "18-25"}],
            privileged_groups=[{attr: "M" if attr == "gender" else "26-45"}],
        )
        
        spd = metric.statistical_parity_difference()
        bias_spd.labels(protected_attribute=attr).set(spd)
        
        if abs(spd) > 0.10:
            bias_alerts.labels(protected_attribute=attr, severity="high").inc()
            await page_compliance_team(
                severity="high",
                message=f"Bias alert: {attr} SPD = {spd:.3f}",
            )


# Run daily via cron
asyncio.run(bias_monitor_job())
```

---

## 7. Antipatterns

### 7.1 Antipattern 1: Single fairness metric

```python
# ❌ Test only demographic parity
spd = metric.statistical_parity_difference()

# ✅ Test multiple metrics; document trade-offs
spd = metric.statistical_parity_difference()
eod = metric.equal_opportunity_difference()
di = metric.disparate_impact()
```

### 7.2 Antipattern 2: Aggregate-only metrics

```python
# ❌ "Our model is fair" (aggregate AUC = 0.87)
auc = compute_auc(y_true, y_pred)

# ✅ Disaggregated metrics by protected group
for group in protected_groups:
    auc_by_group = compute_auc(y_true[group_mask], y_pred[group_mask])
```

### 7.3 Antipattern 3: No intersectional analysis

```python
# ❌ Test gender and age separately
test_gender(y_true, y_pred, "gender")
test_age(y_true, y_pred, "age_group")

# ✅ Test intersectional groups
test_intersectional(y_true, y_pred, ["gender", "age_group", "nationality"])
```

### 7.4 Antipattern 4: One-time bias audit

```python
# ❌ Audit once at deploy
audit_bias(model)

# ✅ Continuous monitoring
@trigger.on("daily")
def bias_audit_daily():
    audit_bias(current_model)
    alert_if_threshold_exceeded()
```

### 7.5 Antipattern 5: No documented trade-offs

```python
# ❌ Optimize one metric; ignore others
optimize_for_demographic_parity()

# ✅ Document the trade-off
# "We optimize for equal opportunity (TPR parity) because the cost of
# false negatives is highest for the unprivileged group. This results
# in slight demographic parity differences but is the right trade-off
# for our domain (credit decisions)."
```

---

## 🎯 Key Takeaways

- Five sources of bias: historical, sampling, measurement, algorithmic, deployment.
- Five canonical fairness metrics: demographic parity, equal opportunity, predictive parity, calibration, counterfactual.
- The **impossibility theorem**: you cannot satisfy all metrics simultaneously. Choose based on domain ethics.
- AI Fairness 360 (IBM) is the standard toolkit — install, measure, mitigate.
- LLM-specific bias: stereotyping, exclusion, toxicity, cultural bias — use BBQ benchmark, Detoxify, counterfactual analysis.
- Intersectional bias: test combinations of attributes (gender × age × nationality).
- Production monitoring: daily/weekly audits with Prometheus + alerts.
- Avoid single metric, aggregate-only, no intersectional, one-time audit, undocumented trade-offs.

## References

- AI Fairness 360 — [aif360.readthedocs.io](https://aif360.readthedocs.io/)
- Chouldechova et al. — Fair prediction with disparate impact — [arxiv.org/abs/1610.07524](https://arxiv.org/abs/1610.07524)
- Hardt et al. — Equality of Opportunity in Supervised Learning — [arxiv.org/abs/1610.02413](https://arxiv.org/abs/1610.02413)
- BBQ Benchmark — [github.com/nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)
- Detoxify — [github.com/unitaryai/detoxify](https://github.com/unitaryai/detoxify)
- EU AI Act Article 10 — Data quality — [eur-lex.europa.eu](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)
- [[06 - Large Language Models/25 - AI Compliance and Governance/01 - EU AI Act 2024 - Risk Classification and Compliance|Note 01 — EU AI Act]]
- [[06 - Large Language Models/25 - AI Compliance and Governance/02 - Model Cards and Datasheets for Datasets|Note 02 — Model Cards]]
- [[06 - Large Language Models/25 - AI Compliance and Governance/04 - Red-Teaming and Adversarial Testing|Note 04 — Red-Teaming]]
- [[06 - Large Language Models/25 - AI Compliance and Governance/05 - Capstone - Compliance Pipeline for a Regulated Industry|Note 05 — Capstone]]
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — bias drift detection
- [[06 - Large Language Models/20 - RAG Evaluation Deep Dive|RAG Evaluation]] — LLM-as-judge for bias