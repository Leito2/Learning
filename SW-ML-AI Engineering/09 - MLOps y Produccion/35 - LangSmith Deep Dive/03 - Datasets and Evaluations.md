# 📊 Datasets and Evaluations — Versioned Golden Sets

The [[../../../06 - Large Language Models/20 - RAG Evaluation Deep Dive/01 - Test Dataset Construction - Synthetic Human Hybrid.md|06/20/01]] course taught you to construct test sets with synthetic + human-validated examples. That work ends with a JSONL file on disk. LangSmith's **Datasets** are the production-grade upgrade: versioned test sets with **first-class integration with evaluators**, **dataset schemas**, and **comparison runs** that show the diff between model versions.

A LangSmith dataset is more than a JSONL file. It has:

- **Versioned snapshots** — every upload creates a new version; old versions are preserved.
- **Example schema** — inputs and outputs are typed; the dataset is queryable by field.
- **Direct evaluator integration** — run `evaluate()` on a dataset with one function call.
- **Compare runs** — compare eval results across model versions on the same dataset.
- **Search and filter** — find examples by tag, source, or metadata.

This note covers dataset creation, the evaluator API, comparison runs, and the patterns that make evaluation a **first-class part of the development workflow** (not an afterthought).

## 🎯 Learning Objectives

- Create **versioned datasets** with the LangSmith SDK.
- Define **example schemas** with inputs and outputs.
- Run **offline evaluations** with custom evaluators.
- Compare eval runs across **model versions**.
- Use **dataset splits** (train/val/test).
- Avoid the four most common dataset pitfalls.

## 1. Dataset Creation

```python
from langsmith import Client

client = Client()

# Create a dataset
dataset = client.create_dataset(
    dataset_name="rag-golden-eval-v1",
    description="Golden test set for the Production RAG system. 200 questions, human-validated.",
)

# Add examples
examples = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris", "citations": [0]},
    },
    {
        "inputs": {"question": "Who wrote the Theory of Relativity?"},
        "outputs": {"answer": "Albert Einstein", "citations": [2]},
    },
    # ... 200 examples
]

client.create_examples(dataset_id=dataset.id, examples=examples)
```

Each example has **inputs** (what the system receives) and **outputs** (what the system should produce). Examples are immutable per version; updating creates a new version.

## 2. Versioning and Updates

```python
# Create v2 (immutable — old v1 is preserved)
dataset_v2 = client.create_dataset(
    dataset_name="rag-golden-eval-v2",
    description="v2: added 50 questions, removed 5 outdated.",
)

# Copy v1 examples + new ones
v1_examples = client.read_dataset(dataset_name="rag-golden-eval-v1").examples
new_examples = [...]
all_v2 = list(v1_examples) + new_examples
client.create_examples(dataset_id=dataset_v2.id, examples=all_v2)
```

**Versioning pattern**: `dataset_name="rag-golden-eval-v{VERSION}"`. Never overwrite; always create a new version.

## 3. Running Evaluations

```python
from langsmith.evaluation import evaluate

def rag_pipeline(inputs: dict) -> dict:
    """Your RAG system."""
    return {"answer": llm.invoke(inputs["question"])}

# Custom evaluator
def answer_correctness(run, example) -> dict:
    pred = run.outputs.get("answer", "").lower()
    truth = example.outputs.get("answer", "").lower()
    return {"key": "correctness", "score": 1.0 if truth in pred else 0.0}

# Run the eval
results = evaluate(
    rag_pipeline,
    data="rag-golden-eval-v3",
    evaluators=[answer_correctness],
    experiment_prefix="rag-gpt-4o-mini-v2",
)
```

The result is an **experiment run** in LangSmith — a single click in the UI shows the per-example scores, average metrics, and failure modes.

## 4. Multiple Evaluators

```python
def faithfulness(run, example) -> dict:
    """LLM-as-judge: is the answer faithful to the retrieved context?"""
    prompt = f"Is this answer faithful?\nAnswer: {run.outputs['answer']}\nContext: {run.outputs['context']}"
    response = judge_llm.invoke(prompt)
    return {"key": "faithfulness", "score": float(response)}

def citation_accuracy(run, example) -> dict:
    """Did the system cite the right passages?"""
    pred_citations = set(run.outputs.get("citations", []))
    truth_citations = set(example.outputs.get("citations", []))
    if not truth_citations:
        return {"key": "citation_accuracy", "score": 0.0}
    return {"key": "citation_accuracy", "score": len(pred_citations & truth_citations) / len(truth_citations)}

results = evaluate(
    rag_pipeline,
    data="rag-golden-eval-v3",
    evaluators=[answer_correctness, faithfulness, citation_accuracy],
)
```

The result tracks all three metrics per example.

## 5. Comparing Runs

```python
# Run v1 with gpt-4o-mini
results_v1 = evaluate(
    rag_pipeline_v1,
    data="rag-golden-eval-v3",
    evaluators=[answer_correctness, faithfulness],
    experiment_prefix="rag-v1-gpt-4o-mini",
)

# Run v2 with gpt-4o
results_v2 = evaluate(
    rag_pipeline_v2,  # same code, different model
    data="rag-golden-eval-v3",
    evaluators=[answer_correctness, faithfulness],
    experiment_prefix="rag-v1-gpt-4o",
)

# Compare in LangSmith UI: side-by-side metrics
# Or via SDK:
comparison = client.compare_runs([results_v1, results_v2])
```

LangSmith shows the **side-by-side** view in the UI: average metrics, per-example diff (where one succeeded and the other failed), and statistical tests (paired t-test, McNemar's).

## 6. Dataset Splits (Train / Val / Test)

```python
# Create separate datasets for train/val/test
client.create_dataset("rag-train-v1", description="Training set for optimizer")
client.create_dataset("rag-val-v1", description="Validation set for optimizer")
client.create_dataset("rag-test-v1", description="Held-out test set")

# Or split a single dataset
dataset = client.read_dataset(dataset_name="rag-golden-eval-v3")
splits = dataset.split(
    train_pct=0.6,
    val_pct=0.2,
    test_pct=0.2,
    seed=42,
)
# Returns 3 sub-datasets
```

The optimizer (e.g., MIPRO) uses `rag-train-v1` for bootstrapping and `rag-val-v1` for candidate evaluation. `rag-test-v1` is held out for final accuracy.

## 7. ❌/✅ Antipatterns

### ❌ One dataset, multiple versions mixed

```python
# ⚠️ Examples from v1 + v2 + v3 in same dataset — version drift
client.create_examples(dataset_id=dataset.id, examples=[v1_examples, v2_examples, v3_examples])
```

### ✅ One dataset per version

```python
# ✅ Clear version boundaries
client.create_examples(dataset_id=dataset_v3.id, examples=v3_examples_only)
```

### ❌ Evaluator without statistical rigor

```python
# ⚠️ Comparing 1 example at a time
def evaluator_wrong(run, example):
    return {"key": "score", "score": 1.0 if correct else 0.0}
```

### ✅ Statistical comparison

```python
# ✅ Use paired t-tests via LangSmith's compare_runs API
# LangSmith applies McNemar's test for binary outcomes
```

### ❌ No metadata on examples

```python
# ⚠️ Can't filter by difficulty or topic
client.create_examples(dataset_id, [{"inputs": {"q": q}, "outputs": {"a": a}}])
```

### ✅ Metadata for filterability

```python
# ✅ Filter by difficulty, topic, source
client.create_examples(dataset_id, [{
    "inputs": {"question": q},
    "outputs": {"answer": a},
    "metadata": {"difficulty": "easy", "topic": "geography", "source": "synthetic"},
}])
```

### ❌ Optimizing on test set

```python
# ⚠️ Test set leaks to optimizer
results = evaluate(pipeline, data="rag-test-v1", evaluators=...)
# Now the optimizer is biased toward test performance
```

### ✅ Trainset + valset, held-out test

```python
# ✅ Optimizer uses train + val; test is final eval only
results = evaluate(pipeline, data="rag-test-v1", evaluators=...)
# (after optimizer runs on train/val)
```

## 8. Production Reality

**Caso real — Production RAG Project:** Three datasets maintained: `rag-train-v1` (60 examples, for MIPRO compilation), `rag-val-v1` (20 examples, for optimizer candidates), `rag-test-v1` (20 examples, held-out for CI gate). Every PR triggers `evaluate()` on `rag-test-v1`; the gate blocks merges where correctness drops >5%.

**Caso real — Multi-Agent Research System:** Dataset versioning tied to Git commits (`rag-golden-eval-v{commit_hash}`). Every model upgrade triggers a new dataset version + new eval run. Comparison view in LangSmith UI shows the **per-example diff** between model versions.

## 📦 Compression Code

```python
# 📦 Compression: Datasets + Evaluations in 30 lines

from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. Create dataset
dataset = client.create_dataset(
    dataset_name="rag-golden-eval-v1",
    description="Golden test set",
)

# 2. Add examples
client.create_examples(dataset_id=dataset.id, examples=[
    {"inputs": {"question": "Capital of France?"}, "outputs": {"answer": "Paris"}},
])

# 3. Define evaluator
def correctness(run, example) -> dict:
    pred = run.outputs["answer"].lower()
    truth = example.outputs["answer"].lower()
    return {"key": "correctness", "score": 1.0 if truth in pred else 0.0}

# 4. Run eval
results = evaluate(
    lambda inputs: {"answer": llm.invoke(inputs["question"])},
    data="rag-golden-eval-v1",
    evaluators=[correctness],
    experiment_prefix="rag-v1",
)

print(f"Average correctness: {results.aggregate['correctness']:.3f}")
```

## 🎯 Key Takeaways

1. **Datasets are versioned** — never overwrite; always create a new version.
2. **Example schema**: `inputs` + `outputs` + optional `metadata`.
3. **`evaluate()` runs the pipeline on every example** — one call, full coverage.
4. **Multiple evaluators per run** — correctness, faithfulness, citation accuracy.
5. **Compare runs in UI** — side-by-side metrics, per-example diff, statistical tests.
6. **Splits: train/val/test** — never optimize on test.
7. **Metadata for filterability** — difficulty, topic, source.

## References

- [[00 - Welcome to LangSmith|Welcome]] — course map.
- [[01 - LangSmith Core|Core primitives]] — runs, traces.
- [[02 - Auto-Instrumentation for LLM SDKs|Auto-Instrumentation]] — the pipeline to instrument.
- [[04 - Online Evaluators|Online Evals]] — production-time scoring.
- [[../../../06 - Large Language Models/20 - RAG Evaluation Deep Dive/01 - Test Dataset Construction - Synthetic Human Hybrid.md|RAG Test Set Construction]] — building the dataset.
- LangSmith datasets: https://docs.smith.langchain.com/evaluation