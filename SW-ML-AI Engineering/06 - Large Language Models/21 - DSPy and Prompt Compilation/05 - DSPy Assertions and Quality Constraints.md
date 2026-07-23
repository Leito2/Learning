# 🛡️ DSPy Assertions and Quality Constraints

Your compiled DSPy program runs and produces an answer — but is it **valid**? Does it cite a real passage? Does the JSON output parse correctly? Is the answer on-topic? Without assertions, a malformed output silently propagates downstream, causing LLM calls to fail, agents to retry, and users to see broken responses.

DSPy Assertions (introduced in v2.4) let you express **hard constraints** on outputs and **automatic retry-on-failure** with self-correction. The compiled program now produces outputs that **must** satisfy your constraints: regex matches, JSON schema validity, length bounds, citation format, topic relevance. Failed assertions trigger a re-prompt with the error message, asking the LM to fix the violation.

This note covers the assertion API (`dspy.Assert`, `dspy.Suggest`, `dspy.Confirm`), the retry mechanism, and the production patterns for quality-constrained agents. By the end you can guarantee that your DSPy-compiled agents never produce malformed outputs — without sacrificing the optimizer's automatic prompt tuning.

## 🎯 Learning Objectives

- Distinguish between **`Assert`** (hard constraint) and **`Suggest`** (soft recommendation).
- Use **`Confirm`** for post-prediction validation with retries.
- Write **custom validators** for domain-specific constraints.
- Combine assertions with **re-prompt strategies** for self-correction.
- Integrate assertions with the **optimizer** (assertions are part of the metric).
- Avoid the four most common assertion pitfalls.

## 1. The Assertion Hierarchy

```python
import dspy

# Hard: the program fails if the constraint is violated
dspy.Assert(condition, "error message")

# Soft: recommendation; program may continue if violated
dspy.Suggest(condition, "suggestion message")

# Post-prediction: validate the output and retry if invalid
dspy.Confirm(predicate, "validation message")
```

| Type | Behavior on violation | Use when |
|------|----------------------|----------|
| `Assert` | Raises exception; fails the run | Production-critical constraints |
| `Suggest` | Logs warning; program continues | Best-practice hints |
| `Confirm` | Triggers re-prompt with error | Output validation with self-correction |

## 2. The `Assert` and `Suggest` Pattern

```python
import dspy

class AnswerWithCitation(dspy.Signature):
    """Answer with mandatory citation."""
    context: list[str] = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: list[int] = dspy.OutputField()


class RAGWithAssertions(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AnswerWithCitation)

    def forward(self, query, context):
        # ... (initial attempt)
        result = self.generate(query=query, context=context)

        # Hard assertion: must have at least 1 citation
        dspy.Assert(
            len(result.citations) >= 1,
            "Answer must include at least one citation. Cite the passages that support your answer.",
        )

        # Soft suggestion: answer should be concise
        dspy.Suggest(
            len(result.answer.split()) < 100,
            "Try to keep the answer under 100 words.",
        )

        # Hard assertion: cited passages must exist in context
        dspy.Assert(
            all(0 <= c < len(context) for c in result.citations),
            "Citations must be valid indices into the provided context.",
        )

        return result
```

When an `Assert` fails, the run raises `dspy.AssertionError`. **By default, this is fatal** — the program stops. To enable self-correction, wrap the program in `dspy.Retry`.

## 3. `dspy.Confirm` — Validation with Self-Correction

`Confirm` runs **after** the prediction, validates the output, and triggers a re-prompt if the validation fails:

```python
class RAGWithConfirm(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AnswerWithCitation)

    def forward(self, query, context):
        # Define the validator
        def is_valid(prediction) -> bool:
            return (
                len(prediction.citations) >= 1
                and all(0 <= c < len(context) for c in prediction.citations)
            )

        # Generate with self-correction
        result = dspy.Confirm(
            self.generate(query=query, context=context),
            is_valid,
            "Please include at least one citation and ensure all citation indices are valid.",
            max_retries=3,
        )
        return result
```

`dspy.Confirm` retries up to `max_retries` times, passing the error message back to the LM. After max retries, it either returns the best attempt or raises.

## 4. Custom Validators

```python
from typing import Any

def validate_citation_format(prediction: dspy.Prediction) -> bool:
    """Custom validator: citations must be 1-indexed, not 0-indexed."""
    return all(c >= 1 for c in prediction.citations)


def validate_no_competitor_mention(prediction: dspy.Prediction, forbidden: list[str]) -> bool:
    """Custom validator: no forbidden brand mentions."""
    text = prediction.answer.lower()
    return not any(brand.lower() in text for brand in forbidden)


def validate_json_schema(prediction: dspy.Prediction, schema: dict) -> bool:
    """Custom validator: prediction matches a JSON schema."""
    import jsonschema
    try:
        jsonschema.validate(instance=prediction.dict(), schema=schema)
        return True
    except jsonschema.ValidationError:
        return False


# Use in your module
class ValidatedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AnswerWithCitation)

    def forward(self, query, context, forbidden_brands):
        result = dspy.Confirm(
            self.generate(query=query, context=context),
            lambda p: (
                validate_citation_format(p)
                and validate_no_competitor_mention(p, forbidden_brands)
            ),
            "Output must have valid citations and not mention forbidden brands.",
            max_retries=2,
        )
        return result
```

## 5. Asserting on Reasoning

For `ChainOfThought` modules, you can also assert on the reasoning:

```python
class MathReasoning(dspy.Signature):
    question: str = dspy.InputField()
    answer: int = dspy.OutputField()

class ValidatedMath(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(MathReasoning)

    def forward(self, question):
        result = self.solve(question=question)

        # The reasoning must mention arithmetic
        dspy.Assert(
            any(op in result.reasoning for op in ["+", "-", "*", "/", "multiply", "add"]),
            "Reasoning must show the arithmetic.",
        )

        # The answer must be an integer
        dspy.Assert(
            isinstance(result.answer, int),
            "Answer must be an integer.",
        )

        return result
```

## 6. Retry Strategies

```python
# Plain: fail on assertion error
program = RAGWithAssertions()

# With retry: up to 3 retries on failure
program = dspy.Retry(
    RAGWithAssertions(),
    max_retries=3,
)

# With advice: include error message in retry prompt
program = dspy.Retry(
    RAGWithAssertions(),
    max_retries=3,
    provide_traceback=True,  # include the assertion error in retry prompt
)
```

`dspy.Retry` wraps any DSPy module and automatically retries on `AssertionError`. The retry prompt includes the assertion's error message, asking the LM to fix the violation.

## 7. Assertions as Part of the Metric

When compiling, include assertions in the metric:

```python
def metric_with_assertions(example, prediction, trace=None):
    # Standard answer-correctness score
    answer_score = float(example.answer.lower() in prediction.answer.lower())

    # Bonus for satisfying assertions
    if hasattr(prediction, 'citations') and len(prediction.citations) >= 1:
        citation_score = 1.0
    else:
        citation_score = 0.0

    return 0.7 * answer_score + 0.3 * citation_score
```

The optimizer finds prompts that **reliably satisfy** the assertions, not just prompts that pass the metric by luck.

## 8. ❌/✅ Antipatterns

### ❌ Assertions that are too strict

```python
# ⚠️ Constant failures; optimizer never finds valid prompts
dspy.Assert(len(result.answer) < 50, "Answer must be <50 chars")
# LLM might output "I don't know" for complex questions
```

### ✅ Calibrated assertions

```python
# ✅ Strict but achievable
dspy.Assert(len(result.answer) < 500, "Answer must be <500 chars")  # reasonable
```

### ❌ Skipping assertions in production

```python
# ⚠️ Outputs may be malformed, JSON parse fails, downstream crashes
result = self.generate(...)  # no validation
```

### ✅ Always validate in production

```python
# ✅ Cheap insurance against malformed outputs
result = dspy.Confirm(self.generate(...), is_valid, max_retries=2)
```

### ❌ Asserting on unobservable state

```python
# ⚠️ Can't verify without external call
dspy.Assert(prediction.is_correct, "Answer must be correct")  # circular
```

### ✅ Assert on observable structure

```python
# ✅ Validates the shape, not the truth
dspy.Assert(
    len(prediction.citations) >= 1
    and all(0 <= c < len(context) for c in prediction.citations),
    "Citations must be valid indices.",
)
```

### ❌ Asserting without self-correction

```python
# ⚠️ Assert fails → run dies
result = self.generate(...)
dspy.Assert(...)
```

### ✅ Wrap in `dspy.Retry` for self-correction

```python
# ✅ Fail → retry with error message → eventually succeed
program = dspy.Retry(ValidatedRAG(), max_retries=3)
```

## 9. Production Reality

**Caso real — Production RAG Project:** Compiled RAG with `Confirm` validates citations: every citation must be a valid index into the retrieved context. The compiled prompts learned to **always cite at least one passage** because the optimizer's metric rewards it. **98% of compiled outputs have valid citations** (vs 71% with hand-tuned prompts).

**Caso real — StayBot:** Used `Assert` to enforce that property descriptions never mention forbidden topics (smoking-friendly, pet-friendly without verification). The compiled agent learned to avoid these phrases entirely — the optimizer found prompts that produce compliant outputs consistently.

## 📦 Compression Code

```python
# 📦 Compression: Assertions + Retry in 50 lines

import dspy

class RAGSignature(dspy.Signature):
    """Answer with citations."""
    context: list[str] = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: list[int] = dspy.OutputField()


class ValidatedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, query, context):
        def is_valid(p):
            return (
                len(p.citations) >= 1
                and all(0 <= c < len(context) for c in p.citations)
                and len(p.answer) < 1000
            )

        return dspy.Confirm(
            self.generate(query=query, context=context),
            is_valid,
            "Output must have valid citations (indices into context) and be <1000 chars.",
            max_retries=3,
        )


# Wrap in Retry for self-correction
program = dspy.Retry(ValidatedRAG(), max_retries=3)

# Use
result = program(query="What is X?", context=["X is Y."])
print(result.answer)
print(result.citations)
```

## 🎯 Key Takeaways

1. **Three assertion types** — `Assert` (hard), `Suggest` (soft), `Confirm` (validation + retry).
2. **Wrap in `dspy.Retry`** — failed assertions trigger re-prompt with error message.
3. **Custom validators for domain rules** — JSON schema, no-competitor-mention, citation format.
4. **Assertions in the metric** — optimizer finds prompts that reliably satisfy constraints.
5. **Calibrate strictness** — too strict = constant failures; too loose = no value.
6. **Assert on structure, not truth** — length, format, valid indices, not "is this correct?".
7. **Production always validates** — `Confirm` is cheap insurance against malformed outputs.

## References

- [[00 - Welcome to DSPy and Prompt Compilation|Welcome]] — course map.
- [[01 - Signatures and Modules|Signatures]] — the building blocks.
- [[02 - Optimizers|Optimizers]] — the compiler (uses assertions in metric).
- [[04 - DSPy + LangGraph Integration|LangGraph integration]] — assertions in agents.
- DSPy Assertions: https://dspy.ai/learn/constraints/