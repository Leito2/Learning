# 🎯 04 - LMQL — A Query Language for LLMs

> **A declarative SQL-like DSL for LLM programs. When you want the cleanest semantics for typed prompts with hard guarantees, LMQL is the most expressive option.**

## 🎯 Learning Objectives
- Write LMQL queries with the `[model]` annotation block and typed variables
- Use `assert` for post-generation validation and automatic retries
- Distribute LMQL programs via `lmql serve` as a FastAPI-compatible endpoint
- Choose LMQL for declarative, type-safe prompts where the grammar is complex but the logic is straightforward
- Recognize LMQL's maintenance and adoption status (active but smaller ecosystem than Instructor/Outlines)
- Understand when LMQL beats raw Python (typed control flow, declarative constraints) and when it loses (DSL friction, debugging difficulty)

## Introduction

LMQL (Language Model Query Language) is the most declarative of the four libraries covered in this course. Instead of an imperative Python program (Instructor), an FSA-over-vocabulary constraint (Outlines), or a token-level control template (Guidance), LMQL is a **SQL-like DSL** that expresses an LLM program as a single annotated string with type declarations, variable bindings, and assertion blocks. The result compiles to a token sequence that satisfies the types and constraints automatically.

```python
import lmql

@lmql.query
def summarize(text: str) -> str:
    """lmql
    argmax
        "Summarize the following text in one sentence: '{text}'\n"
        "Summary: [SUMMARY]"
    from
        "openai/gpt-4o-mini"
    where
        SUMMARY: str = STOPS_AT("\n")
    """

print(summarize("LMQL is a declarative query language for LLMs that..."))
```

That snippet declares a typed query, runs it against GPT-4o-mini, and returns a string. The `STOPS_AT("\n")` constraint is enforced at generation time. The `argmax` directive asks for the highest-probability decoding (vs sampling). The whole thing is one annotated string.

The library was developed at ETH Zürich by Luca Beurer-Kellner and colleagues, with the goal of bringing **type safety, formal semantics, and declarative syntax** to LLM programming. It supports Transformers, llama.cpp, and OpenAI-compatible backends; it can be served as a standalone HTTP endpoint via `lmql serve`; and it integrates with DSPy for compiled optimization (covered in [[06 - Large Language Models/21 - DSPy and Prompt Compilation]]).

LMQL's bet is that **declarative beats imperative** for typed prompts. The DSL is harder to learn than a Python library, but the resulting programs are more readable, more reproducible, and easier to debug post-hoc — every variable has a declared type, every constraint is a first-class construct, and every assertion is logged with a counterexample.

![LMQL syntax highlighting and architecture](https://lmql.ai/_/logo.svg)

---

## 1. The LMQL Syntax

A LMQL query has four parts:

```python
@lmql.query
def my_query(input: str) -> str:
    """lmql
    <decoding-directive>           # argmax, sample, beam, etc.
        <prompt-template>          # f-string-style with [VAR] placeholders
        "Some literal text [VAR]"  # generation markers
    from
        "<provider>/<model>"       # model specification
    where
        VAR: type = <constraint>   # typed variable declarations
    """
```

### 1.1 The `[VAR]` placeholder

`[VAR]` marks a generation site. The model emits tokens until the variable's constraint is satisfied. Variables can be referenced downstream in the prompt and in constraints.

```python
@lmql.query
def greet(name: str) -> str:
    """lmql
    argmax
        "Hello, [NAME]! How are you today?"
        "[GREETING]"
    from
        "openai/gpt-4o-mini"
    where
        NAME: str = STOPS_AT("!")
        GREETING: str = STOPS_AT(QUESTION)
    """
```

`NAME` is captured up to the first `!`. `GREETING` is captured up to the `?` (the `QUESTION` token). Both are typed strings.

### 1.2 The `where` block — typed variables and constraints

The `where` block declares variables and their constraints:

```python
@lmql.query
def classify(text: str) -> str:
    """lmql
    argmax
        "Classify the sentiment of: {text}\n"
        "Sentiment: [SENTIMENT]"
    from
        "openai/gpt-4o-mini"
    where
        SENTIMENT: Literal["positive", "negative", "neutral"] = STOPS_AT("\n")
    """
```

`SENTIMENT` is a `Literal` type — only one of the three strings is valid. The constraint is enforced at generation time via token masking (similar to Outlines).

### 1.3 The `from` block — model selection

```python
from
    "openai/gpt-4o-mini"          # OpenAI
    "anthropic/claude-3-5-sonnet" # Anthropic
    "hf/local"                     # Local Transformers
    "llamacpp/local"               # llama.cpp
    "mock"                          # Mock for tests
```

LMQL supports the same multi-provider pattern as Instructor (via its `lmql serve` proxy), with the trade-off that constraint enforcement varies by backend.

### 1.4 Decoding directives

| Directive | Behavior |
|-----------|----------|
| `argmax` | Greedy decoding (highest-probability token at each step) |
| `sample` | Sampling with default temperature |
| `sample(temperature=0.7)` | Sampling with custom temperature |
| `beam(n=4)` | Beam search with n beams |
| `var(beam_n=4, temperature=0.7)` | Combined search |

`argmax` is the standard for deterministic structured outputs. `sample` is the standard for creative generation. `beam` is rare but useful for tasks where multiple plausible outputs exist (e.g. summarization).

---

## 2. The `assert` Block — Post-Generation Validation with Automatic Retry

LMQL distinguishes **constraints** (enforced at generation time) from **assertions** (validated after generation). Assertions trigger automatic retries when they fail:

```python
@lmql.query
def safe_summary(text: str) -> str:
    """lmql
    argmax
        "Summarize: {text}\n"
        "Summary: [SUMMARY]"
    from
        "openai/gpt-4o-mini"
    where
        SUMMARY: str = STOPS_AT("\n")
    assert
        len(SUMMARY) < 100, "Summary must be under 100 characters"
        "leaked" not in SUMMARY.lower(), "Summary must not contain 'leaked'"
    """
```

If the summary is too long or contains the word "leaked", LMQL re-samples (up to a configurable limit) and tries again. The assertion error messages are passed back to the model as context — the same "validation as feedback" pattern that Instructor uses, but expressed declaratively.

### 2.1 `assert` vs `where`

| Construct | When evaluated | When violated |
|-----------|:--------------:|---------------|
| `where` (constraint) | At generation time | Token mask; model cannot emit invalid |
| `assert` | After generation | Retry with feedback; up to N attempts |

Use `where` for shape constraints (type, range, regex) that the model can satisfy at every step. Use `assert` for semantic constraints (length, content, cross-field) that need the full output to evaluate.

### 2.2 Custom distribution operators

LMQL supports `dist` for distribution-level constraints:

```python
where
    CLASS: Literal["A", "B", "C"] = DIST([0.5, 0.3, 0.2])
```

This says the model should produce `A` 50% of the time, `B` 30%, `C` 20% — useful for synthetic data generation, balanced sampling, and stress-testing classifiers.

---

## 3. Case Studies

### 3.1 Case real 1: ETH Zürich research pipeline (legal contract analysis)

ETH's NLP group used LMQL to build a legal contract clause extractor. Each clause type (parties, dates, amounts, obligations) had its own typed LMQL query with custom assertions:

```python
@lmql.query
def extract_parties(contract: str) -> List[Tuple[str, str]]:
    """lmql
    argmax
        "Extract all parties from this contract: {contract}\n"
        "Party: [P1_NAME], Role: [P1_ROLE]\n"
        "Party: [P2_NAME], Role: [P2_ROLE]\n"
        "Party: [P3_NAME], Role: [P3_ROLE]\n"
    from
        "openai/gpt-4o-mini"
    where
        P1_NAME: str = STOPS_AT(",")
        P1_ROLE: Literal["Buyer", "Seller", "Lender", "Borrower", "Guarantor", "Other"] = STOPS_AT("\n")
        P2_NAME: str = STOPS_AT(",")
        P2_ROLE: Literal["Buyer", "Seller", "Lender", "Borrower", "Guarantor", "Other"] = STOPS_AT("\n")
        P3_NAME: str = STOPS_AT(",")
        P3_ROLE: Literal["Buyer", "Seller", "Lender", "Borrower", "Guarantor", "Other"] = STOPS_AT("\n")
    assert
        P1_NAME != P2_NAME, "Duplicate party name"
    """
```

The declarative syntax made the queries self-documenting — auditors could read the LMQL source and verify the extraction logic without understanding the underlying Python. The `assert` block caught cross-party duplicates that pure JSON Schema could not.

### 3.2 Case real 2: Synthetic data generation for classifier training

Generating balanced training data with LMQL:

```python
@lmql.query
def synthetic_support_ticket(category: str, count: int = 1) -> str:
    """lmql
    sample(temperature=0.9)
        f"Generate a realistic customer support ticket for category '{category}'.\n"
        "Subject: [SUBJECT]\n"
        "Body: [BODY]\n"
    from
        "openai/gpt-4o-mini"
    where
        SUBJECT: str = STOPS_AT("\n")
        BODY: str = STOPS_AT(END)
    assert
        len(SUBJECT) > 10 and len(SUBJECT) < 100
        len(BODY) > 50 and len(BODY) < 500
        any(category.lower() in BODY.lower() for category in [category])
    """
```

Run this 1000 times with `category` cycling through `["billing", "technical", "account", "shipping"]` and you get a balanced synthetic dataset for training a downstream ticket classifier. The `assert` block ensures every ticket has the right length and category keywords.

---

## 4. Serving LMQL via `lmql serve`

LMQL ships with a built-in HTTP server that exposes queries as FastAPI-compatible endpoints:

```bash
# Save queries to a file
cat > queries.lmql <<EOF
@lmql.query
def summarize(text: str) -> str:
    """lmql
    argmax
        "Summarize: {text}\n"
        "Summary: [SUMMARY]"
    from
        "openai/gpt-4o-mini"
    where
        SUMMARY: str = STOPS_AT("\n")
    """
EOF

# Start the server
lmql serve queries.lmql --port 8000
```

Then call from any HTTP client:

```python
import requests

resp = requests.post(
    "http://localhost:8000/summarize",
    json={"text": "LMQL is a declarative query language..."},
)
print(resp.json()["SUMMARY"])
```

`lmql serve` integrates with FastAPI middleware, OpenTelemetry traces from [[09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers]], and DSPy compilation pipelines from [[06 - Large Language Models/21 - DSPy and Prompt Compilation]]. For production, you can wrap LMQL queries inside a larger FastAPI app and route via LiteLLM from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]].

💡 **Tip:** `lmql serve` is great for small deployments, but for high-throughput production, the recommended pattern is to import LMQL queries as Python functions and run them in your existing FastAPI app. The server overhead is non-trivial compared to direct function calls.

---

## 5. Backends and Constraint Enforcement

| Backend | `where` (constraint) | `assert` (post-check) | Multi-provider |
|---------|:--------------------:|:---------------------:|:--------------:|
| `openai/<model>` | ✅ Via API parameters | ✅ | ✅ |
| `anthropic/<model>` | ✅ Limited | ✅ | ✅ |
| `hf/<model>` (Transformers) | ✅ Full (Outlines-style FSA) | ✅ | ❌ (local only) |
| `llamacpp/<model>` | ✅ Full (GBNF grammar) | ✅ | ❌ (local only) |
| `mock` | ✅ Returns canned data | ✅ | — |

The full constraint expressivity is only available with self-hosted backends (Transformers, llama.cpp). For SaaS providers, `where` falls back to type checking + post-hoc validation — similar to Instructor's strategy.

---

## 6. Comparison — LMQL vs Instructor vs Outlines vs Guidance

| Property | LMQL | Instructor | Outlines | Guidance |
|----------|:----:|:----------:|:--------:|:--------:|
| **Paradigm** | 5 (DSL with constraints) | 3 (validation loop) | 4 (FSA constraint) | 4 (control flow + generation) |
| **Syntax** | Custom DSL (SQL-like) | Python decorators | Python library | Python decorators + templates |
| **Constraints at generation** | ✅ (where) | ❌ (post-hoc) | ✅ | ✅ |
| **Post-validation with retry** | ✅ (assert) | ✅ (max_retries) | N/A | N/A |
| **Multi-step in one call** | ✅ | ❌ | ❌ | ✅ |
| **Type system** | Strong (Python types in `where`) | Pydantic | JSON Schema only | Limited |
| **Multi-provider** | ✅ Via lmql serve | ✅ Native | Limited | Limited |
| **Server deployment** | `lmql serve` | Standard FastAPI | Via vLLM/TGI | Custom |
| **DSPy integration** | ✅ Direct | ✅ Via custom signatures | ⚠️ Limited | ⚠️ Limited |
| **Open source** | Apache 2.0 | MIT | Apache 2.0 | MIT |
| **Maintenance (2026)** | ⚠️ Active but smaller ecosystem | ✅ Mature | ✅ Mature | ✅ Active |
| **Learning curve** | High (DSL) | Low | Low | Medium |

---

## 7. When to Choose LMQL

Use LMQL when:
- The schema is **complex but the logic is straightforward** (declarative wins).
- You want **type-safe prompts** that are auditable by non-engineers (legal, finance, compliance).
- You need **distribution-level constraints** (`DIST(...)`) that other libraries cannot express.
- You are building a **research pipeline** that will be compiled with DSPy.

Do not use LMQL when:
- The team is not willing to learn a DSL (Instructor wins on familiarity).
- The schema is simple and stable (Outlines is faster and simpler).
- You need maximum constraint enforcement with minimum overhead (Outlines with vLLM is the throughput king).
- You need provider-agnostic code with broad community support (Instructor's ecosystem is largest).

In practice, LMQL is the **right tool for research and compliance**, less common in pure production code. The capstone (Note 05) uses Instructor because of its multi-provider ergonomics and mature ecosystem; LMQL is mentioned as the alternative for typed declarative pipelines.

---

## 8. Antipatterns

### 8.1 Antipattern 1: Using LMQL for what Instructor does in 3 lines

```python
# ❌ Overkill: simple JSON extraction with no control flow
@lmql.query
def extract_person(text: str) -> dict:
    """lmql
    argmax
        "Extract: [PERSON]"
    from
        "openai/gpt-4o-mini"
    where
        PERSON: JsonSchema = Person
    """
```

```python
# ✅ Correct: use Instructor for trivial schemas
import instructor
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = instructor_client.chat.completions.create(
    response_model=Person,
    messages=[{"role": "user", "content": text}],
)
```

### 8.2 Antipattern 2: Trying to use complex `where` constraints on SaaS providers

```python
# ❌ Fallback: complex regex constraints on OpenAI backend silently degrade
where
    EMAIL: str = REGEX(r"[a-z]+@[a-z]+\.[a-z]+")  # only enforced locally
```

For OpenAI/Anthropic backends, stick to simple types and use `assert` for semantic validation.

### 8.3 Antipattern 3: Deeply nested templates

```python
# ❌ Unreadable: nested templates with cross-references
@lmql.query
def nested(...) -> ...:
    """lmql
    argmax
        "Section 1: [S1]"
        "  Sub 1.1: [S11] (referring to '{S1}')"
        "  Sub 1.2: [S12] (referring to '{S11}')"
        "Section 2: [S2] (referring to '{S1}')"
    ...
    """
```

LMQL is great for declarative templates but breaks down at deep nesting. Split into multiple smaller queries or use Guidance for multi-step programs.

### 8.4 Antipattern 4: Treating `assert` like a Python `assert`

```python
# ❌ Python mindset: `assert` removes the response if it fails
# In LMQL, `assert` triggers retry — heavy under the hood

assert
    len(SUMMARY) > 0  # This will trigger retry if empty
```

If a constraint fails 95% of the time, your retries will burn tokens. Move truly required constraints into `where` (enforced at generation time) so the model never emits the invalid output in the first place.

---

## 9. LMQL + DSPy

LMQL integrates with DSPy for compiled optimization:

```python
import dspy

class SummarizeSignature(dspy.Signature):
    """Summarize the given text."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

# Wrap the LMQL query as a DSPy module
class SummarizeLMQL(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lmql_query = summarize  # the LMQL query from above
    
    def forward(self, text):
        result = self.lmql_query(text=text)
        return dspy.Prediction(summary=result)
```

DSPy can then compile this module against a labeled dataset, optimizing the prompt and constraint parameters. This is the bridge between declarative LMQL and the compilation-based optimization of DSPy — covered in depth in [[06 - Large Language Models/21 - DSPy and Prompt Compilation]].

Caso real: A customer-support automation team used LMQL+DSPy to compile a typed intent classifier. The pre-compiled LMQL query achieved 87% accuracy; after DSPy optimization (over 500 labeled examples), the same compiled query reached 94% — without changing the LMQL source. The optimization happened at the prompt-instruction and constraint-string level, leaving the schema and type declarations untouched.

---

## 🎯 Key Takeaways

- LMQL is a SQL-like DSL for typed LLM programs — declarative, type-safe, auditable.
- `[VAR]` markers declare generation sites; `where` block declares types and constraints; `assert` block validates after generation.
- Constraints are enforced at generation time when running locally (Transformers, llama.cpp) and post-hoc when using SaaS providers.
- `DIST(...)` enables distribution-level constraints for synthetic data generation.
- `lmql serve` exposes queries as HTTP endpoints; alternatively, import as Python functions.
- LMQL integrates with DSPy for compiled optimization of typed prompts.
- Best for: research pipelines, compliance/audit use cases, synthetic data generation, declarative type-safe prompts.
- Avoid: trivial schemas (use Instructor), complex SaaS-provider constraints, deeply nested templates, treating `assert` as Python `assert`.

## References

- LMQL docs — [lmql.ai](https://lmql.ai)
- LMQL GitHub — [github.com/eth-cscs/lmql](https://github.com/eth-cscs/lmql)
- LMQL paper — Beurer-Kellner et al., 2023, "Prompting Is Programming: A Query Language for Large Language Models"
- [[06 - Large Language Models/21 - DSPy and Prompt Compilation|DSPy and Prompt Compilation]] — compiled optimization partner
- [[06 - Large Language Models/22 - Instructor and Structured Generation/01 - Instructor - Pydantic-Native Structured Outputs|Note 01 — Instructor]] — multi-provider alternative
- [[06 - Large Language Models/22 - Instructor and Structured Generation/02 - Outlines - Constrained Decoding at the Token Level|Note 02 — Outlines]] — fastest constrained decoding
- [[06 - Large Language Models/22 - Instructor and Structured Generation/03 - Guidance - Token-Level Control and Prompt Programming|Note 03 — Guidance]] — token-level control flow
- [[06 - Large Language Models/22 - Instructor and Structured Generation/05 - Capstone - Production Structured Extraction Service|Note 05 — Capstone]] — production synthesis using Instructor
