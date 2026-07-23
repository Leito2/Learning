# 🎯 03 - Datasets, Evaluations and Prompt Management

> **The three pillars that turn LangFuse from a tracer into an evaluation platform. Run reproducible LLM-as-Judge evals, version prompts like Git, ship experiments with confidence.**

## 🎯 Learning Objectives
- Build versioned evaluation datasets with ground-truth answers and expected outputs
- Run reproducible LLM-as-Judge evaluations with statistical significance
- Manage prompts as versioned artifacts with A/B testing and rollback
- Combine evaluation runs into experiments for comparison across prompt versions
- Use the `langfuse.evaluate()` API for batch evaluation against datasets
- Wire dataset items from production traces to bootstrap new evaluation sets

## Introduction

Tracing tells you **what happened**. Evaluations tell you **how good it was**. Prompts tell you **what code path produced the result**. LangFuse's three pillars — Datasets, Evaluations, and Prompt Management — are the operational substrate for any team that treats LLM quality as a first-class engineering concern rather than a vibes-based judgment call.

A dataset is a versioned collection of `(input, expected_output)` pairs, optionally tagged with metadata. Every prompt-version release can be evaluated against the dataset; the result is a reproducible quality score. LangFuse runs the evaluation against historic data, attaches scores to each dataset item, and surfaces them in the UI for comparison.

The prompt registry stores versioned prompt templates with variables and configuration. Prompts can be promoted to "production" status, tagged as "deprecated", or rolled back to a previous version. Each trace shows which prompt version produced the output — full lineage from code to behavior.

The evaluation engine ties everything together: it runs a target function (typically an LLM call with the chosen prompt) against every item in a dataset, scores each result via LLM-as-Judge or deterministic metrics, and aggregates scores into experiment-level statistics. Statistical significance is computed across runs so you know if prompt v2 is actually better than v1 or just noise.

This is the operational layer for [[06 - Large Language Models/20 - RAG Evaluation Deep Dive]] (RAGAS-style evaluation) and [[06 - Large Language Models/21 - DSPy and Prompt Compilation]] (compiled optimization). Where those courses teach the algorithmic patterns, this one teaches the **plumbing** that turns a one-off eval script into a continuous evaluation pipeline.

![LangFuse datasets view](https://langfuse.com/_next/image?url=%2Fstatic%2Fimages%2Fdocs%2Fdatasets.png&w=1920&q=75)

---

## 1. Datasets — Versioned Evaluation Inputs

### 1.1 The Dataset data model

```
Dataset
├── name (string, e.g. "rag_eval_v1")
├── description (markdown)
├── metadata (json: tags, owner, etc.)
└── DatasetItem (N)
    ├── input (json: question, context, etc.)
    ├── expected_output (json: ground truth)
    └── metadata (json: source, difficulty, etc.)
```

Datasets are immutable after creation but can be augmented with new items. Each `DatasetItem` is itself versioned — once added, its content does not change. To "version" a dataset, create a new dataset with a different name (`rag_eval_v1` → `rag_eval_v2`) and copy the items you want to preserve.

### 1.2 Creating a dataset programmatically

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create the dataset
dataset = langfuse.create_dataset(
    name="rag_eval_v1",
    description="RAG evaluation set for production tenant ABC. 100 questions with ground-truth answers.",
    metadata={
        "owner": "team-rag",
        "creation_date": "2026-07-23",
        "language": "en",
    },
)

# Add items
items = [
    {
        "input": {"question": "What is the capital of France?"},
        "expected_output": {"answer": "Paris"},
        "metadata": {"difficulty": "easy", "category": "geography"},
    },
    {
        "input": {"question": "What did the author say about pruning?"},
        "expected_output": {
            "answer": "Pruning reduces LLM memory by 30-50% with minimal accuracy loss.",
            "citation": "Chapter 4, Section 3"
        },
        "metadata": {"difficulty": "hard", "category": "rag"},
    },
    # ... 98 more items
]

for item in items:
    langfuse.create_dataset_item(
        dataset_name="rag_eval_v1",
        **item,
    )
```

Each item can be retrieved later, scored, or used as an evaluation target. Datasets are also accessible via the LangFuse UI under Datasets → rag_eval_v1.

### 1.3 Importing from production traces

The recommended pattern: capture real production traces, then bootstrap a dataset from the most representative ones:

```python
# Pull high-quality traces from the last week
traces = langfuse.fetch_traces(
    tags=["production", "rag"],
    from_timestamp="2026-07-15T00:00:00Z",
    limit=500,
    fields={"scores.user_feedback": {"operator": ">", "value": 0.8}},  # high-rated only
)

# Create dataset items from the traces
dataset = langfuse.create_dataset(name="rag_eval_v2")
for trace in traces.data:
    langfuse.create_dataset_item(
        dataset_name="rag_eval_v2",
        input={"question": trace.input["question"]},
        expected_output={"answer": trace.output["answer"]},  # user-approved
        metadata={"trace_id": trace.id, "source": "production"},
    )
```

This pattern creates evaluation sets that reflect real user behavior rather than synthetic test cases. Re-running evaluations against the same dataset tracks quality drift as the system evolves.

### 1.4 CSV upload via UI

For one-off imports, the LangFuse UI supports CSV upload. The format:

```csv
input,expected_output,metadata.difficulty,metadata.category
"What is the capital of France?","Paris",easy,geography
"What did the author say about pruning?","Pruning reduces LLM memory...",hard,rag
```

CSV columns are auto-mapped to `input.*`, `expected_output.*`, `metadata.*` based on the dot-prefix convention.

---

## 2. Evaluations — LLM-as-Judge and Custom Metrics

### 2.1 The `langfuse.evaluate()` API

The simplest way to run an evaluation is via the high-level API:

```python
from langfuse import Langfuse
from langfuse.evaluation import evaluate

langfuse = Langfuse()

# The target function: takes a dataset item, returns the model's output
def target(item):
    # 1. Get the prompt
    prompt = langfuse.get_prompt("qa_system_prompt")
    compiled_prompt = prompt.compile(question=item.input["question"])
    
    # 2. Call LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": compiled_prompt},
            {"role": "user", "content": item.input["question"]},
        ],
    )
    return response.choices[0].message.content

# The evaluation function: takes the dataset item + target output, returns scores
def evaluator(item, output):
    # Use an LLM-as-Judge
    judge_prompt = f"""
    Question: {item.input['question']}
    Expected answer: {item.expected_output['answer']}
    Model answer: {output}
    
    Rate the model answer on:
    - correctness (0-1): Is it factually correct?
    - relevance (0-1): Does it answer the question?
    - style (0-1): Is it concise and clear?
    
    Output JSON only.
    """
    
    judge_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )
    scores = json.loads(judge_response.choices[0].message.content)
    return [
        {"name": "correctness", "value": scores["correctness"]},
        {"name": "relevance", "value": scores["relevance"]},
        {"name": "style", "value": scores["style"]},
    ]

# Run the evaluation
experiment = evaluate(
    dataset_name="rag_eval_v1",
    target=target,
    evaluators=[evaluator],
    experiment_name="qa_prompt_v2",
    metadata={"prompt_version": "v2.0.0", "model": "gpt-4o-mini"},
)

print(f"Mean correctness: {experiment.scores['correctness'].mean():.2f}")
print(f"Mean relevance: {experiment.scores['relevance'].mean():.2f}")
```

`evaluate()` returns an `Experiment` object with aggregated scores per metric. Each item's scores are attached to the trace in LangFuse, so you can drill down per-item in the UI.

### 2.2 Deterministic evaluators

For non-LLM metrics (exact match, regex, length, keyword presence), write Python directly:

```python
def deterministic_evaluator(item, output):
    expected = item.expected_output["answer"].lower()
    actual = output.lower()
    
    return [
        {"name": "exact_match", "value": int(expected == actual)},
        {"name": "keyword_overlap", "value": len(set(expected.split()) & set(actual.split())) / len(expected.split())},
        {"name": "length_ratio", "value": min(len(output) / max(len(expected), 1), 1.0)},
    ]
```

Use deterministic evaluators for objective metrics (JSON validity, schema conformance) and LLM-as-Judge for subjective metrics (helpfulness, tone, factual accuracy). Pair them in a single `evaluators=[...]` list for combined scoring.

### 2.3 Custom remote evaluators

For evaluators that need extra context (e.g. a human-in-the-loop interface), use a remote evaluator hosted on your own server:

```python
@observe(as_type="evaluator")
def custom_evaluator(item, output):
    # Call your own evaluator service
    response = requests.post(
        "https://my-evaluator-service.internal/score",
        json={"input": item.input, "output": output, "expected": item.expected_output},
    )
    return [
        {"name": "custom_score", "value": response.json()["score"]},
    ]
```

The decorator makes the evaluator itself a traceable LangFuse span — useful for debugging why a score was what it was.

### 2.4 Statistical rigor

`evaluate()` computes mean, std, min, max per metric, but for statistical significance across runs, use the experiment comparison API:

```python
# Compare two experiment runs
comparison = langfuse.compare_experiments(
    experiment_names=["qa_prompt_v1", "qa_prompt_v2"],
    dataset_name="rag_eval_v1",
    metrics=["correctness", "relevance"],
    statistical_test="welch_t_test",  # or "mann_whitney_u"
)
print(comparison.summary)
# Output:
# correctness: v2 0.85 ± 0.08 vs v1 0.78 ± 0.11 (p=0.003, significant)
# relevance:   v2 0.91 ± 0.05 vs v1 0.88 ± 0.07 (p=0.142, not significant)
```

Welch's t-test for normally-distributed scores; Mann-Whitney U for non-normal. p<0.05 means the difference is statistically significant; p≥0.05 means you cannot conclude which version is better.

This rigor is exactly what the [[06 - Large Language Models/20 - RAG Evaluation Deep Dive]] course recommends. Without statistical tests, teams ship "improvements" that are actually noise.

---

## 3. Prompt Management — Versioned Prompts with A/B Testing

### 3.1 Creating a prompt

```python
prompt = langfuse.create_prompt(
    name="qa_system_prompt",
    prompt="You are a helpful assistant. Answer the user's question concisely.\n\nQuestion: {{question}}",
    config={"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 200},
    labels=["production"],
)
```

This creates prompt v1. Each call to `langfuse.get_prompt("qa_system_prompt")` returns the prompt with the "production" label (or the latest version if no label).

### 3.2 Versioning and rollback

```python
# Create v2
langfuse.create_prompt(
    name="qa_system_prompt",
    prompt="You are an expert assistant. Cite sources when possible.\n\nQuestion: {{question}}",
    config={"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 300},
    labels=["staging"],  # not yet production
)

# Promote v2 to production (and demote v1)
langfuse.update_prompt_labels(
    name="qa_system_prompt",
    version=2,
    labels=["production"],
)
# v1 is now unlabeled; v2 is "production"

# Rollback: re-label v1
langfuse.update_prompt_labels(
    name="qa_system_prompt",
    version=1,
    labels=["production"],
)
```

Prompts are immutable per version. Once created, version 1's content cannot change. New versions are append-only; labels point to which version is "active". This is Git-like behavior with zero risk of accidentally modifying a prompt that production traffic depends on.

### 3.3 Compiling and using prompts

```python
# In the production code path
prompt = langfuse.get_prompt("qa_system_prompt")
compiled = prompt.compile(question="What is the capital of France?")

response = openai_client.chat.completions.create(
    model=prompt.config["model"],  # model comes from the prompt config
    temperature=prompt.config["temperature"],
    messages=[
        {"role": "system", "content": compiled},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
```

The trace captures the prompt version. Compare two runs of the same trace to see exactly what prompt was active.

### 3.4 A/B testing with labels

For true A/B testing, use two labels and route traffic randomly:

```python
import random

def get_prompt_with_ab_test():
    if random.random() < 0.5:
        return langfuse.get_prompt("qa_system_prompt", label="production")  # v2
    else:
        return langfuse.get_prompt("qa_system_prompt", label="experiment_a")  # v3
```

The trace records which label was used. After running for a week, query the dataset for traces with each label and compare scores — if v3 wins with statistical significance, promote it to "production".

### 3.5 Link to source control

For full audit trail, link each prompt version to a Git commit:

```python
langfuse.create_prompt(
    name="qa_system_prompt",
    prompt="...",
    config={...},
    metadata={"git_commit": "abc123def", "git_repo": "github.com/acme/prompts"},
)
```

The UI shows the commit SHA next to each prompt version. Click it to jump to the commit in GitHub. This is the **prompt-as-code integration** from the cutting-edge section of the welcome note.

---

## 4. The Operational Loop — Experiments Across Versions

The full workflow:

```
1. Pull historic dataset (or create new)
2. Develop prompt v_new locally
3. Upload v_new to LangFuse with label "staging"
4. Run evaluation: evaluate(dataset_name, target, evaluators, experiment_name="v_new")
5. Compare v_new vs current production via compare_experiments()
6. If significant improvement: update_prompt_labels(v_new → "production")
7. If regression: keep v_old as "production", iterate
```

This loop is the **continuous evaluation** pattern. Every prompt change is statistically justified before shipping. Every prompt change is reversible with one API call.

Caso real: A customer-support chatbot team runs this loop weekly. Each Friday, the team uploads a candidate prompt, runs the eval against the historic dataset of 1000 production questions, and compares scores. The promotion rate is ~30% — most prompts regress on edge cases and require iteration. Without this loop, the team would ship every prompt change blindly and discover regressions in production.

---

## 5. Annotation Queues for Human Feedback

For subjective metrics, route production traces to human reviewers:

```python
# In the production code path, attach low-confidence traces to a queue
if response.confidence < 0.6:
    langfuse.score(
        trace_id=trace.id,
        name="needs_review",
        value=1,
        metadata={"queue": "human_review", "reason": "low_model_confidence"},
    )
```

Reviewers see the queue in the LangFuse UI under Annotation → Queues. They rate the response on a 1-5 scale; the score attaches to the trace. Over time, the dataset of human-rated traces becomes ground truth for LLM-as-Judge calibration.

The pattern is covered in detail in [[09 - MLOps y Produccion/35 - LangSmith Deep Dive/05 - Annotation Queues and Human Feedback|Annotation Queues in LangSmith]] — the same UI exists in LangFuse with the same API surface.

---

## 6. Antipatterns

### 6.1 Antipattern 1: Evaluating with the same model as the target

```python
# ❌ Bias: target is gpt-4o-mini; judge is also gpt-4o-mini. Judge will reward its own style.
target = lambda item: gpt_4o_mini(item)
evaluator = lambda item, output: gpt_4o_mini_judge(item, output)

# ✅ Use a different (and stronger) model as judge
target = lambda item: gpt_4o_mini(item)  # small model
evaluator = lambda item, output: claude_3_5_sonnet_judge(item, output)  # larger, different model
```

LLM-as-Judge has known biases (verbosity, position, self-style preference). Using a different, stronger model as judge reduces but does not eliminate them. The [[06 - Large Language Models/20 - RAG Evaluation Deep Dive]] course covers the calibration patterns.

### 6.2 Antipattern 2: Treating dataset v1 as the ground truth forever

```python
# ❌ Static ground truth ignores evolving user needs
dataset = langfuse.get_dataset("rag_eval_v1")  # 100 items from 6 months ago

# ✅ Re-curate quarterly; add 10-20% new items each cycle
new_items = curate_from_production_traces(since="2026-04-01")
langfuse.create_dataset(name="rag_eval_v2")
# Copy old + add new
```

User behavior changes; evaluation datasets must too. A 6-month-old eval set may not represent current usage patterns.

### 6.3 Antipattern 3: Using `langfuse.evaluate()` without naming the experiment

```python
# ❌ Anonymous runs clutter the UI; cannot compare later
evaluate(dataset_name="rag_eval_v1", target=target, evaluators=[evaluator])

# ✅ Always name the experiment
evaluate(dataset_name="rag_eval_v1", target=target, evaluators=[evaluator], experiment_name="v2_baseline_2026-07-23")
```

### 6.4 Antipattern 4: Setting `temperature=0` for LLM-as-Judge

```python
# ❌ Greedy judge is brittle; small prompt changes → different scores
judge_response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,  # rigid
    messages=[...judge_prompt...],
)

# ✅ Use temperature=0.3-0.5 for judge; average across 3 samples for stability
scores = []
for _ in range(3):
    judge_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[...judge_prompt...],
    )
    scores.append(parse_score(judge_response))
mean_score = sum(scores) / len(scores)
```

LLM-as-Judge variance is non-trivial. Three samples averaged reduces variance by √3 and gives more reliable scores.

### 6.5 Antipattern 5: Treating `compare_experiments()` as definitive

```python
# ❌ Single evaluation run is noisy
comparison = compare_experiments(["v1", "v2"])
if comparison.p_value < 0.05:
    deploy_v2()  # risky if dataset is small

# ✅ Multiple runs + larger dataset
for _ in range(3):
    comparison = compare_experiments(["v1", "v2"])
# Look at consistency of p-values across runs
```

A single comparison is one statistical test. Multiple runs reduce false positives from random variance.

---

## 🎯 Key Takeaways

- Datasets are versioned `(input, expected_output)` collections; import from production traces or upload CSV.
- `langfuse.evaluate()` runs target × evaluators against a dataset and returns aggregated scores.
- LLM-as-Judge uses a different, stronger model than the target; deterministic evaluators handle objective metrics.
- `compare_experiments()` with Welch's t-test or Mann-Whitney U gives statistical significance for A/B testing.
- Prompt registry stores versioned templates with labels (production, staging); rollback is one API call.
- Link prompt versions to Git commits for full audit trail.
- The operational loop: pull dataset → upload candidate → evaluate → compare → promote if significant.
- Avoid same-model bias, stale ground truth, anonymous experiments, greedy judges, and single-run comparisons.

## References

- LangFuse Datasets docs — [langfuse.com/docs/evaluation/dataset](https://langfuse.com/docs/evaluation/dataset)
- LangFuse Evaluation docs — [langfuse.com/docs/evaluation/evaluation](https://langfuse.com/docs/evaluation/evaluation)
- LangFuse Prompt Management — [langfuse.com/docs/prompts](https://langfuse.com/docs/prompts)
- [[06 - Large Language Models/20 - RAG Evaluation Deep Dive|RAG Evaluation Deep Dive]] — LLM-as-Judge statistical rigor
- [[06 - Large Language Models/21 - DSPy and Prompt Compilation|DSPy and Prompt Compilation]] — compiled optimization partner
- [[09 - MLOps y Produccion/35 - LangSmith Deep Dive|LangSmith Deep Dive]] — SaaS counterpart with same primitives
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/01 - LangFuse Fundamentals - Architecture and Core Primitives|Note 01 — Fundamentals]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/04 - Online Evaluators and Production Patterns|Note 04 — Online Evaluators]]
- [[09 - MLOps y Produccion/36 - LangFuse - Open-Source LLM Observability/05 - Capstone - Self-Hosted LangFuse for Multi-Provider RAG|Note 05 — Capstone]]
