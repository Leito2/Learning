# 🛠️ Custom Metrics with the RAGAS Protocol

The four built-in RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) cover the most common quality dimensions, but production RAG systems need more. Domain-specific faithfulness (does the answer avoid making claims about competitors?), citation accuracy (does each claim cite the right passage?), instruction adherence (does the answer follow the user's formatting instructions?), and many others are not in the box. This note teaches how to write **custom RAGAS metrics** that integrate with the `evaluate(...)` API and benefit from the same parallelism, retry, and reporting infrastructure as built-ins.

The RAGAS Protocol for a custom metric is small: extend `Metric` or `SingleTurnMetric`, implement `single_turn_score(...)` (or `multi_turn_score` for conversations), and return a `MetricResult` (value + optional reason). The judge prompt is a Jinja2 template that RAGAS renders per-sample. By the end of this note you will have written a citation accuracy metric, a domain faithfulness metric, and a JSON-schema-adherence metric — all production-deployable.

## 🎯 Learning Objectives

- Implement custom metrics by extending the RAGAS `Metric` and `SingleTurnMetric` classes.
- Use `SingleTurnSample` to declare inputs and `MetricOutput` to return results.
- Write judge prompts as Jinja2 templates with proper prompt-engineering discipline.
- Compose built-in and custom metrics in a single `evaluate(...)` call.
- Persist custom metric reasons for debugging low scores.
- Avoid the three most common custom-metric pitfalls.

## 1. The `SingleTurnSample` and `Metric` Contracts

```python
from ragas.metrics import Metric, SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.result import MetricResult
from pydantic import BaseModel

class Metric(Generic[T]):
    """Base interface for any RAGAS metric (single or multi-turn)."""
    name: str
    allowed_values: list[str]  # e.g., ["answer_correct"] for classification metrics

class SingleTurnMetric(Metric, ABC):
    """A metric that scores one Q&A at a time."""

    @abstractmethod
    async def single_turn_score(self, sample: SingleTurnSample) -> MetricResult: ...

class MetricResult(BaseModel):
    """The output of a metric evaluation."""
    value: float                                  # the score (0-1 by convention)
    reason: str | None = None                      # optional explanation (for LLM-judged)
    metadata: dict[str, Any] | None = None        # extra structured info
```

The contract is **5 lines**. A custom metric is a class with a name, an async `single_turn_score` method, and a returned `MetricResult`. The framework handles parallelism, retries, and aggregation.

## 2. The Simplest Custom Metric: Citation Accuracy

The most common custom metric in production RAG is **citation accuracy**: for each claim in the answer, does the cited passage actually support it?

```python
from ragas.metrics import SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.result import MetricResult
from ragas.prompt import PydanticPrompt
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json

# === Step 1: Define the LLM output schema ===
class CitationCheck(BaseModel):
    claim: str
    citation: str  # e.g., "[1]", "[2]"
    is_supported: bool
    explanation: str

class CitationAccuracyOutput(BaseModel):
    checks: list[CitationCheck] = Field(min_length=1)

# === Step 2: Define the judge prompt as a Jinja2 template ===
class CitationPrompt(PydanticPrompt[CitationAccuracyOutput]):
    instruction = """You are a citation accuracy evaluator. Given an answer and its retrieved context, verify each claim's cited passage supports it.

Rules:
- A citation is correct if the cited passage contains or directly entails the claim
- A citation is wrong if the passage contradicts or doesn't address the claim
- Output one check per cited claim
- Use exact citation numbers from the answer (e.g., [1], [2])
"""
    input_model = None
    output_model = CitationAccuracyOutput

    def to_string(self, sample: SingleTurnSample) -> str:
        return self.format(
            question=sample.user_input,
            answer=sample.response,
            contexts="\n".join(f"[{i+1}] {c}" for i, c in enumerate(sample.retrieved_contexts)),
        )

# === Step 3: The metric class ===
class CitationAccuracy(SingleTurnMetric):
    name = "citation_accuracy"

    def __init__(self, llm):
        self.llm = LangchainLLMWrapper(llm)
        self.prompt = CitationPrompt()

    async def single_turn_score(self, sample: SingleTurnSample) -> MetricResult:
        # 1. Extract claims with citations from the answer
        # 2. For each, ask the LLM to verify
        prompt_text = self.prompt.to_string(sample)
        response = await self.llm.agenerate(prompt_text, response_model=CitationAccuracyOutput)

        # 3. Aggregate: % of supported claims
        checks = response.checks
        n_supported = sum(1 for c in checks if c.is_supported)
        score = n_supported / len(checks) if checks else 1.0

        return MetricResult(
            value=score,
            reason=f"{n_supported}/{len(checks)} citations supported",
            metadata={"checks": [c.model_dump() for c in checks]},
        )
```

**Usage:**

```python
from ragas import EvaluationDataset, evaluate

llm = ChatOpenAI(model="gpt-4o-mini")
metric = CitationAccuracy(llm)

dataset = EvaluationDataset.from_list(test_set)
results = evaluate(dataset, metrics=[metric])
print(results["citation_accuracy"])  # [0.85, 1.0, 0.67, ...]
```

## 3. Domain-Specific Faithfulness

Your portfolio's StayBot should never claim a property has amenities it doesn't. The built-in `faithfulness` checks "is this answer supported by retrieved docs?" — but you want "is this answer supported AND does it avoid making claims about features the docs don't mention?"

```python
class DomainFaithfulness(SingleTurnMetric):
    """Faithfulness + no fabricated features."""
    name = "domain_faithfulness"

    def __init__(self, llm, forbidden_topics: list[str] = None):
        self.llm = LangchainLLMWrapper(llm)
        self.forbidden = forbidden_topics or []
        self.prompt = DomainFaithfulnessPrompt()

    async def single_turn_score(self, sample: SingleTurnSample) -> MetricResult:
        prompt_text = self.prompt.to_string(
            sample=sample,
            forbidden=", ".join(self.forbidden),
        )
        response = await self.llm.agenerate(
            prompt_text,
            response_model=DomainFaithfulnessOutput,
        )

        # Combine: base faithfulness + no forbidden topics
        base_score = sum(1 for c in response.checks if c.supported) / len(response.checks)
        forbidden_violations = sum(1 for c in response.checks if c.topic in self.forbidden)
        penalty = forbidden_violations / len(response.checks) if response.checks else 0

        return MetricResult(
            value=max(0.0, base_score - penalty),
            reason=f"{base_score:.2f} base - {penalty:.2f} forbidden = {base_score - penalty:.2f}",
            metadata={"violations": forbidden_violations},
        )

# Usage for StayBot
metric = DomainFaithfulness(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    forbidden_topics=["smoking allowed", "pet-friendly", "wheelchair accessible"],
)
# A response claiming "this is a smoking-friendly property" loses 1/N for that forbidden claim
```

## 4. JSON Schema Adherence

For function-calling RAG systems, the answer must conform to a JSON schema. A custom metric enforces this:

```python
from jsonschema import validate, ValidationError

class JSONSchemaAdherence(SingleTurnMetric):
    """Checks if the response is valid JSON matching a schema."""
    name = "json_schema_adherence"

    def __init__(self, schema: dict):
        self.schema = schema

    async def single_turn_score(self, sample: SingleTurnSample) -> MetricResult:
        # Extract JSON from response (handles ```json ... ``` blocks)
        json_str = extract_json(sample.response)

        try:
            parsed = json.loads(json_str)
            validate(instance=parsed, schema=self.schema)
            return MetricResult(value=1.0, reason="Valid JSON matching schema")
        except json.JSONDecodeError as e:
            return MetricResult(value=0.0, reason=f"JSON parse error: {e}")
        except ValidationError as e:
            return MetricResult(value=0.0, reason=f"Schema violation: {e.message}")

def extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown fences
        lines = text.split("\n")
        return "\n".join(l for l in lines if not l.startswith("```"))
    return text

# Usage
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "sources"],
}
metric = JSONSchemaAdherence(schema=schema)
```

## 5. Prompt Engineering for the Judge

The judge prompt is the metric. A bad prompt yields noise. Three rules:

### Rule 1: Explicit Rubric with Anchor Examples

```python
JUDGE_PROMPT = """Rate the faithfulness of the answer on a 0-1 scale.

0.0 - The answer has NO support in the context (pure hallucination)
0.25 - The answer contradicts the context
0.5 - The answer partially matches (some claims supported, some not)
0.75 - The answer is mostly supported with minor unsupported additions
1.0 - Every claim is supported by the context

Examples:
- Context: "X is true." Answer: "X is true and Y is true."  → 0.5 (Y unsupported)
- Context: "X is true." Answer: "X is true." → 1.0
- Context: "X is true." Answer: "Y is true." → 0.0

Context: {{ context }}
Answer: {{ answer }}

Output: <single number 0-1>"""
```

### Rule 2: Structured Output (Pydantic) Over Free-Form

```python
class FaithfulnessVerdict(BaseModel):
    supported_claims: list[str]
    unsupported_claims: list[str]
    reasoning: str
    score: float

# Better than: free-form text that you parse with regex
```

### Rule 3: Few-Shot Examples for Edge Cases

```python
PROMPT_WITH_EXAMPLES = """...""...\n\nEdge cases to consider:
- Answer that says "I don't know" → score 1.0 (honest)
- Answer that quotes the question → score 0.5 (not informative)
- Answer that contradicts the context but cites it → score 0.0 (misleading)
- Answer with no citable claim (subjective) → score 1.0 (not evaluable)
"""
```

## 6. Composing Built-in and Custom Metrics

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

llm = ChatOpenAI(model="gpt-4o-mini")

# Built-in + custom in one call
results = evaluate(
    dataset,
    metrics=[
        faithfulness,              # built-in
        answer_relevancy,          # built-in
        CitationAccuracy(llm),     # custom
        DomainFaithfulness(llm, forbidden_topics=["smoking"]),  # custom
        JSONSchemaAdherence(schema),  # custom
    ],
    raise_exceptions=False,  # don't fail on one bad sample
)

# Per-metric aggregation
for metric_name in ["faithfulness", "answer_relevancy", "citation_accuracy", "domain_faithfulness", "json_schema_adherence"]:
    scores = [s for s in results[metric_name] if s is not None]
    print(f"{metric_name}: {np.mean(scores):.3f} (n={len(scores)})")
```

## 7. Persisting Reasons for Debugging

```python
class DebuggingCitationAccuracy(CitationAccuracy):
    """Same as CitationAccuracy but always saves per-sample reasons."""

    async def single_turn_score(self, sample: SingleTurnSample) -> MetricResult:
        result = await super().single_turn_score(sample)
        # Save to a JSONL file for offline debugging
        with open("eval_failures.jsonl", "a") as f:
            if result.value < 0.8:  # log low scores
                f.write(json.dumps({
                    "sample_id": id(sample),
                    "question": sample.user_input,
                    "answer": sample.response,
                    "score": result.value,
                    "reason": result.reason,
                    "metadata": result.metadata,
                }) + "\n")
        return result
```

When RAGAS reports "citation_accuracy: 0.71", open `eval_failures.jsonl` and find the 30% of samples that failed. Inspect the `metadata["checks"]` to see which claim was unsupported.

## 8. ❌/✅ Antipatterns

### ❌ Custom metric without prompt engineering

```python
# ⚠️ Generic prompt → noisy scores
class BadMetric(SingleTurnMetric):
    async def single_turn_score(self, sample):
        response = await llm.agenerate(f"Rate this: {sample.response}")
        return MetricResult(value=float(response))  # what scale? what meaning?
```

### ✅ Explicit rubric + anchors + examples

```python
class GoodMetric(SingleTurnMetric):
    async def single_turn_score(self, sample):
        # Rubric with 4 anchor examples
        # Pydantic output schema
        # Per-sample reasoning for debugging
        ...
```

### ❌ Mixing judge models within a single metric

```python
# ⚠️ Different judges = different distributions = incomparable
class MixedJudgeMetric(SingleTurnMetric):
    async def single_turn_score(self, sample):
        if len(sample.user_input) < 50:
            score = await cheap_llm.score(sample)
        else:
            score = await expensive_llm.score(sample)
        return MetricResult(value=score)
```

### ✅ Single judge per metric, varying by metric

```python
class CitationAccuracyGPT4o(SingleTurnMetric):  # expensive judge
    pass

class FastFaithfulnessGpt4oMini(SingleTurnMetric):  # cheap judge
    pass
```

### ❌ Metric that depends on state (non-deterministic)

```python
# ⚠️ Random seed inside the metric → non-reproducible
class RandomMetric(SingleTurnMetric):
    async def single_turn_score(self, sample):
        base = await judge.score(sample)
        return MetricResult(value=base + random.random() * 0.1)  # ⚠️ noise
```

### ✅ Deterministic judge + bounded randomization

```python
class BootstrapMetric(SingleTurnMetric):
    """Deterministic judge, but with bootstrap CIs."""
    async def single_turn_score(self, sample):
        base = await judge.score(sample)
        # Bootstrap CI is computed in post-processing, not in the metric
        return MetricResult(value=base)
```

## 9. Production Reality

**Caso real — Production RAG Project:** The portfolio project uses 4 built-in + 3 custom metrics:
- Built-in: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Custom: `CitationAccuracy(gpt-4o-mini)`, `JSONSchemaAdherence(schema)`, `NoCompetingBrands(forbidden=["competitor-X"])`

The `NoCompetingBrands` metric caught 14 PRs that introduced competitor mentions in 6 months — metric-as-policy-enforcement.

**Caso real — StayBot:** `DomainFaithfulness(forbidden_topics=["smoking", "pet-friendly", "wheelchair accessible"])` enforces a marketing-team constraint at eval time. The metric outputs a 0.0-1.0 score; if it drops below 0.85, CI blocks the merge.

## 📦 Compression Code

```python
# 📦 Compression: Custom RAGAS metrics in 80 lines

from ragas.metrics import SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.result import MetricResult
from ragas.prompt import PydanticPrompt
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json

class SimpleVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str

class SimplePrompt(PydanticPrompt[SimpleVerdict]):
    instruction = """Rate the answer 0-1 based on context support.
0 = unsupported, 1 = fully supported.
Context: {{ context }}
Answer: {{ answer }}
Output JSON with score and reasoning."""
    output_model = SimpleVerdict

    def to_string(self, sample: SingleTurnSample) -> str:
        return self.format(
            context="\n".join(f"[{i+1}] {c}" for i, c in enumerate(sample.retrieved_contexts)),
            answer=sample.response,
        )

class SimpleSupportMetric(SingleTurnMetric):
    """A simple custom metric: is the answer supported by retrieved context?"""
    name = "simple_support"

    def __init__(self, llm):
        self.llm = LangchainLLMWrapper(llm)
        self.prompt = SimplePrompt()

    async def single_turn_score(self, sample: SingleTurnSample) -> MetricResult:
        prompt_text = self.prompt.to_string(sample)
        verdict = await self.llm.agenerate(prompt_text, response_model=SimpleVerdict)
        return MetricResult(
            value=verdict.score,
            reason=verdict.reasoning,
        )

# === Usage ===
from ragas import evaluate, EvaluationDataset

llm = ChatOpenAI(model="gpt-4o-mini")
metric = SimpleSupportMetric(llm)

test_set = [
    SingleTurnSample(
        user_input="What is X?",
        response="X is Y.",
        retrieved_contexts=["X is Y."],
    ),
]
dataset = EvaluationDataset.from_list(test_set)
result = evaluate(dataset, metrics=[metric])
print(result["simple_support"])  # [0.95]
```

## 🎯 Key Takeaways

1. **The protocol is small** — extend `SingleTurnMetric`, implement `single_turn_score`, return `MetricResult`.
2. **The judge prompt is the metric** — invest in rubrics, anchor examples, structured outputs.
3. **Use `PydanticPrompt`** to get structured output instead of regex-parsing free-form text.
4. **Composition is the value** — mix built-in (general quality) + custom (domain policy) metrics.
5. **Persist `reason` and `metadata`** — the score is useless without the explanation.
6. **Deterministic judge per metric** — don't mix models within one metric.
7. **Domain faithfulness > generic faithfulness** — the `forbidden_topics` pattern is the production move.

## References

- [[00 - Welcome to RAG Evaluation Deep Dive|Welcome]] — course map.
- [[01 - Test Dataset Construction|Test Set]] — the input to metrics.
- [[03 - Statistical Rigor|Statistical Rigor]] — when 0.71 is and isn't significant.
- [[04 - LLM-as-Judge Bias|Judge Bias]] — your judge prompt has biases; this note explains how.
- RAGAS custom metrics: https://docs.ragas.io/en/stable/concepts/metrics/index_custom
- Pydantic: https://docs.pydantic.dev/