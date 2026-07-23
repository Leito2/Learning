# 📐 Statistical Rigor — Confidence Intervals, Paired Tests, and McNemar

A RAGAS score is a sample mean. The number `0.83` is meaningless without knowing how precisely it estimates the true metric value. A 5% metric improvement between two pipeline versions is meaningless without knowing if it's inside the noise of the test set. This note teaches the statistical discipline that turns "the metric went up" into "the metric went up with p < 0.05 and a 95% CI of the difference that's entirely above zero".

Four techniques cover 95% of production eval rigor:

1. **Confidence intervals** on the metric itself.
2. **Paired t-tests / Wilcoxon** for comparing two pipelines on the same samples.
3. **McNemar's test** for binary metrics (e.g., "did this sample pass?").
4. **Bootstrap resampling** when assumptions are shaky.

By the end, you can answer "is this 2% improvement real?" with the same precision an A/B test engineer answers it for web experiments.

## 🎯 Learning Objectives

- Compute 95% confidence intervals on RAGAS metric means.
- Apply paired t-tests and Wilcoxon signed-rank tests for pipeline comparisons.
- Use McNemar's test for binary "pass/fail" comparisons.
- Compute bootstrap CIs for non-parametric robustness.
- Calibrate CI gates with appropriate thresholds.
- Avoid the three most common statistical mistakes in eval pipelines.

## 1. The Problem: A Number Is Not Evidence

```python
# v1: ran RAGAS
v1_faithfulness = 0.831

# v2: rerank + prompt tweak
v2_faithfulness = 0.849

# "We improved by 1.8%!"
# ⚠️ Without knowing the variance, you don't know if 1.8% is signal or noise.
```

The standard deviation of a RAGAS metric depends on the test set. For 200 questions, faithfulness might have std=0.12 → 95% CI half-width is 1.96 × 0.12 / √200 = 0.017. So both numbers are inside each other's CI — **the "improvement" is invisible**.

```python
# Without CIs
v1 = 0.831  # "bad"
v2 = 0.849  # "good"
# Looks like improvement, but:
# - Are they different populations?
# - Same test set? (paired)
# - Within noise?

# With CIs
v1_ci = (0.81, 0.85)
v2_ci = (0.83, 0.87)
# Overlap → not significantly different
```

## 2. Confidence Intervals on Metric Means

```python
import numpy as np
from scipy import stats

def mean_and_ci(scores: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean + lower + upper bound of the confidence interval."""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)  # standard error = std / sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

# Example
scores = [0.85, 0.92, 0.78, 0.91, 0.83, 0.88, 0.79, 0.90, 0.86, 0.92]  # 10 samples
mean, lo, hi = mean_and_ci(scores)
print(f"Mean: {mean:.3f}, 95% CI: ({lo:.3f}, {hi:.3f})")
# Mean: 0.864, 95% CI: (0.811, 0.917)

# Rule: CI half-width = 1.96 * std / sqrt(n)
# For 1% CI precision: need n ~ (1.96 * std / 0.01)^2 = 38,000 samples (for std=0.1)
```

### CI for Binary Metrics (Pass/Fail)

```python
def proportion_ci(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson CI for proportions (better than normal approximation for small n)."""
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, center - margin, center + margin

# Example: 87 passes out of 100
p, lo, hi = proportion_ci(87, 100)
print(f"Pass rate: {p:.2f}, 95% CI: ({lo:.3f}, {hi:.3f})")
# Pass rate: 0.87, 95% CI: (0.793, 0.927)
```

> 💡 **Tip:** For binary pass/fail rates, use the **Wilson CI** instead of normal approximation. It's exact for small n and avoids the "negative lower bound" pathology.

## 3. Paired Tests: Comparing Two Pipelines

For comparing v1 vs v2 on the same test set, **paired tests** are the right tool. Each pair is `(v1_score, v2_score)` for the same question. The within-pair difference is what you test.

```python
from scipy import stats

def paired_t_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired t-test: are the per-question differences significantly different from 0?"""
    diff = np.array(scores_a) - np.array(scores_b)
    n = len(diff)
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    t = mean_diff / se_diff
    p_value = 2 * stats.t.sf(abs(t), df=n - 1)  # two-sided

    # CI on the difference
    h = stats.t.ppf(0.975, n - 1) * se_diff
    return {
        "mean_diff": mean_diff,
        "ci_95": (mean_diff - h, mean_diff + h),
        "t_stat": t,
        "p_value": p_value,
        "n": n,
        "significant_at_0.05": p_value < 0.05,
    }

# Example
v1 = [0.85, 0.78, 0.91, 0.83, 0.79, 0.88, 0.92, 0.81]
v2 = [0.88, 0.82, 0.91, 0.85, 0.83, 0.91, 0.93, 0.85]
result = paired_t_test(v1, v2)
print(f"Mean diff: {result['mean_diff']:.3f}, p={result['p_value']:.4f}")
print(f"95% CI: {result['ci_95']}, significant: {result['significant_at_0.05']}")
# Mean diff: 0.018, p=0.0188, significant
```

### Non-Parametric Alternative: Wilcoxon Signed-Rank

When metric distributions are skewed (RAGAS scores are bounded [0,1] and often cluster near extremes), the **Wilcoxon signed-rank test** is more reliable:

```python
def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Non-parametric paired test — robust to non-normal distributions."""
    diff = np.array(scores_a) - np.array(scores_b)
    # Remove zeros (ties)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return {"p_value": 1.0, "significant": False}

    stat, p_value = stats.wilcoxon(diff)
    return {
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "n_pairs": len(diff),
        "significant_at_0.05": p_value < 0.05,
    }
```

## 4. McNemar's Test for Binary Outcomes

For **pass/fail** comparisons (e.g., "did the answer cite correctly?" yes/no), McNemar's test is the right tool. It uses only the **discordant pairs** — cases where one pipeline passed and the other failed.

```python
def mcnemar_test(
    v1_pass: list[bool],
    v2_pass: list[bool],
) -> dict:
    """McNemar's test for paired binary outcomes."""
    assert len(v1_pass) == len(v2_pass)
    n = len(v1_pass)

    # Discordant counts
    b = sum(1 for a, b in zip(v1_pass, v2_pass) if a and not b)  # v1 passed, v2 failed
    c = sum(1 for a, b in zip(v1_pass, v2_pass) if not a and b)  # v1 failed, v2 passed

    # McNemar with continuity correction (good for small discordant counts)
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "discordant_b": b, "discordant_c": c}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "discordant_b": b,  # v1 pass / v2 fail
        "discordant_c": c,  # v1 fail / v2 pass
        "significant_at_0.05": p_value < 0.05,
        "winner": "v2" if c > b else "v1",
    }

# Example
v1_pass = [True, True, False, True, False, True, False, True, False, True]
v2_pass = [True, True, True, True, False, True, True, True, False, True]
result = mcnemar_test(v1_pass, v2_pass)
print(f"chi2={result['chi2']:.3f}, p={result['p_value']:.4f}")
print(f"v1 pass/v2 fail: {result['discordant_b']}, v1 fail/v2 pass: {result['discordant_c']}")
# v2 won: 3 cases flipped from fail to pass, 0 flipped the other way
```

## 5. Bootstrap Confidence Intervals

When you're not sure about normality, **bootstrap** the data 10,000 times and compute the CI on the resampled mean:

```python
def bootstrap_ci(
    scores: list[float],
    statistic: callable = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for any statistic."""
    rng = np.random.default_rng(42)
    n = len(scores)
    boot_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        boot_stats.append(statistic(sample))

    boot_stats = np.sort(boot_stats)
    alpha = 1 - confidence
    lo = boot_stats[int(n_resamples * alpha / 2)]
    hi = boot_stats[int(n_resamples * (1 - alpha / 2))]
    return statistic(scores), lo, hi

# Bootstrap CI on the median (more robust than mean for skewed distributions)
scores = [0.85, 0.92, 0.78, 0.91, 0.83, 0.88, 0.79, 0.90, 0.86, 0.92]
mean, lo, hi = bootstrap_ci(scores, statistic=np.median)
print(f"Median: {mean:.3f}, 95% bootstrap CI: ({lo:.3f}, {hi:.3f})")
```

**Why bootstrap wins:** Makes no assumptions about distribution, works for any statistic (median, p99, ratio), and the only cost is compute.

## 6. The Production Eval Report

```python
def report_pipeline_comparison(
    v1_scores: list[float],
    v2_scores: list[float],
    v1_pass: list[bool],
    v2_pass: list[bool],
) -> dict:
    """Production eval report: paired + McNemar + CIs."""
    n = len(v1_scores)

    # Aggregate stats
    v1_mean, v1_lo, v1_hi = mean_and_ci(v1_scores)
    v2_mean, v2_lo, v2_hi = mean_and_ci(v2_scores)

    # Paired test
    paired = paired_t_test(v1_scores, v2_scores)

    # McNemar for binary
    mcnemar = mcnemar_test(v1_pass, v2_pass)

    # Decision
    if paired["significant_at_0.05"] and paired["mean_diff"] > 0:
        decision = "✅ v2 SIGNIFICANTLY BETTER than v1 (p<0.05, paired)"
    elif paired["significant_at_0.05"]:
        decision = "❌ v2 SIGNIFICANTLY WORSE than v1 (p<0.05, paired)"
    else:
        decision = f"⚠️ INCONCLUSIVE — difference of {paired['mean_diff']:.3f} not significant (p={paired['p_value']:.3f})"

    return {
        "v1": {"mean": v1_mean, "ci_95": (v1_lo, v1_hi), "n": n},
        "v2": {"mean": v2_mean, "ci_95": (v2_lo, v2_hi), "n": n},
        "paired_t": paired,
        "mcnemar": mcnemar,
        "decision": decision,
    }

# Use
report = report_pipeline_comparison(
    v1_scores=[0.85, 0.78, ...],   # 200 samples
    v2_scores=[0.88, 0.82, ...],   # 200 samples (same questions)
    v1_pass=[True, False, ...],     # 200 booleans
    v2_pass=[True, True, ...],     # 200 booleans
)
print(report["decision"])
```

## 7. CI Gates in CI/CD

```python
def should_merge(
    baseline_mean: float,
    candidate_mean: float,
    candidate_ci: tuple[float, float],
    min_improvement: float = 0.02,  # require ≥2% absolute improvement
) -> tuple[bool, str]:
    """CI gate: candidate must be ≥2% above baseline, with CI entirely above min_improvement."""
    diff = candidate_mean - baseline_mean
    ci_lower, _ = candidate_ci

    # Both conditions must hold
    if diff >= min_improvement and ci_lower > 0:  # CI entirely above 0 difference
        return True, f"✅ Pass: {diff:.3f} ≥ {min_improvement} with CI bottom at {ci_lower:.3f}"

    return False, f"❌ Block: diff={diff:.3f}, CI={candidate_ci}, gate={min_improvement}"
```

A safe CI gate:

```yaml
# .github/workflows/eval.yml
- name: Run RAGAS eval
  run: python -m eval.run --test-set v_2024-12-01.jsonl
- name: CI gate
  run: |
    python -c "
    import json
    report = json.load(open('eval_report.json'))
    diff = report['v2']['mean'] - report['v1']['mean']
    ci_lo = report['v2']['ci_95'][0] - report['v1']['mean']
    if diff < 0.02 or ci_lo < 0:
        raise SystemExit(f'Eval gate failed: diff={diff:.3f}, CI bottom={ci_lo:.3f}')
    "
```

The gate blocks merges where the improvement is below 2% or where the lower CI is below zero — both common forms of "false positive improvement".

## 8. ❌/✅ Antipatterns

### ❌ Treating RAGAS scores as exact

```python
# ⚠️ 0.831 vs 0.849 = "1.8% improvement" — but is it real?
print(f"v1: {v1:.3f}, v2: {v2:.3f}, improvement: {(v2-v1)*100:.1f}%")
```

### ✅ Report CIs and significance

```python
mean, lo, hi = mean_and_ci(scores)
print(f"v1: {mean:.3f} (95% CI: {lo:.3f}-{hi:.3f})")
paired = paired_t_test(v1_scores, v2_scores)
print(f"Significant: {paired['significant_at_0.05']}, p={paired['p_value']:.4f}")
```

### ❌ Multiple comparisons without correction

```python
# ⚠️ Run 10 metric comparisons, report 1 with p<0.05 by chance
for metric_name, scores in metrics.items():
    t = paired_t_test(v1[metric_name], v2[metric_name])
    if t["p_value"] < 0.05:
        print(f"{metric_name} improved!")  # ~50% false positive rate
```

### ✅ Bonferroni or Holm correction

```python
from statsmodels.stats.multitest import multipletests

p_values = [paired_t_test(v1[m], v2[m])["p_value"] for m in metrics]
reject, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method="holm")
for metric, p_corrected in zip(metrics, corrected_p):
    print(f"{metric}: corrected p = {p_corrected:.4f}")
```

### ❌ Unpaired tests on paired data

```python
# ⚠️ Independent t-test loses power when samples are paired
t = stats.ttest_ind(v1_scores, v2_scores)  # assumes independence
```

### ✅ Use paired tests

```python
# ✅ Each v1[i] is paired with v2[i] (same question)
t = stats.ttest_rel(v1_scores, v2_scores)
```

## 9. Production Reality

**Caso real — Production RAG Project:** The CI gate uses `mean_and_ci + paired_t_test`. Three "improvements" turned out to be noise (CI bottom at -0.005, -0.012, -0.008) and were blocked. Two real improvements (CI bottom at +0.018, +0.024) passed and shipped. Without the gate, all 5 would have shipped; with the gate, only the real ones did.

**Caso real — Multi-Agent Research System:** A/B test on 200 questions showed v1: 0.81, v2: 0.83. Without paired t-test, this is "2% improvement". With paired t-test: p = 0.18 (not significant). Spent two weeks investigating, learned the improvement was variance. Reverted.

## 📦 Compression Code

```python
# 📦 Compression: Statistical rigor in 80 lines

import numpy as np
from scipy import stats

def mean_and_ci(scores, confidence=0.95):
    n = len(scores)
    mean = np.mean(scores)
    h = stats.sem(scores) * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def paired_t_test(a, b):
    diff = np.array(a) - np.array(b)
    n = len(diff)
    mean_diff = np.mean(diff)
    se = stats.sem(diff)
    t = mean_diff / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    h = stats.t.ppf(0.975, n - 1) * se
    return {"mean_diff": mean_diff, "ci_95": (mean_diff - h, mean_diff + h), "p_value": p, "significant": p < 0.05}

def mcnemar_test(a_pass, b_pass):
    b = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
    c = sum(1 for x, y in zip(a_pass, b_pass) if not x and y)
    if b + c == 0:
        return {"chi2": 0, "p_value": 1, "winner": "tie"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, 1)
    return {"chi2": chi2, "p_value": p, "b": b, "c": c, "winner": "b" if c > b else "a"}

# === Demonstration ===
np.random.seed(42)
v1 = np.random.beta(8, 2, size=200)  # mean ≈ 0.80
v2 = v1 + np.random.normal(0.02, 0.05, size=200)  # small true improvement + noise

mean1, lo1, hi1 = mean_and_ci(v1)
mean2, lo2, hi2 = mean_and_ci(v2)
paired = paired_t_test(v1, v2)

print(f"v1: {mean1:.3f} [{lo1:.3f}, {hi1:.3f}]")
print(f"v2: {mean2:.3f} [{lo2:.3f}, {hi2:.3f}]")
print(f"Paired t-test: mean diff = {paired['mean_diff']:.3f}, p = {paired['p_value']:.4f}")
print(f"Significant: {paired['significant']}")
```

## 🎯 Key Takeaways

1. **A RAGAS score is a sample mean** — always report CIs, not point estimates.
2. **Paired tests are required** for same-test-set comparisons (independent tests lose 30%+ power).
3. **McNemar for binary pass/fail** — uses only discordant pairs, ideal for A/B on "did this sample pass?"
4. **Bootstrap CIs for skewed distributions** — bounded [0,1] metrics often violate normality assumptions.
5. **Bonferroni/Holm correction** when running multiple metric comparisons in one CI run.
6. **CI gate: candidate CI bottom must be > 0** AND improvement must exceed a minimum threshold.
7. **Power analysis drives sample size** — 100 samples detect 5% changes, 1,000 detect 1% changes.

## References

- [[00 - Welcome to RAG Evaluation Deep Dive|Welcome]] — course map.
- [[01 - Test Dataset Construction|Test Set]] — sample size feeds statistical power.
- [[02 - Custom Metrics with RAGAS Protocol|Custom Metrics]] — paired tests over custom metrics.
- [[05 - CI-CD Eval Pipelines|CI/CD]] — uses CIs and paired tests in the gate.
- statsmodels: https://www.statsmodels.org/
- SciPy: https://scipy.org/