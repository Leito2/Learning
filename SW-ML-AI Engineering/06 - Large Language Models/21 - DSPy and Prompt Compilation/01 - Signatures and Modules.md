# 📝 Signatures and Modules — DSPy's Declarative Surface

DSPy's surface API is two primitives: **Signatures** (declarative input/output schemas) and **Modules** (prompt patterns that implement them). Master these two and you can express any LLM workflow — RAG, multi-hop reasoning, agent loops, eval pipelines — in 5-10 lines per step. The optimizer (covered in [[02 - Optimizers - BootstrapFewShot MIPRO and COPRO|note 02]]) then compiles the program automatically.

A **Signature** is the contract between your code and the LLM: it says "given these inputs, produce these outputs". You declare it as a class with typed fields. DSPy uses the field names, types, and docstrings to **synthesize prompts automatically** — no hand-written prompt strings.

A **Module** is a prompt pattern: `Predict` (basic), `ChainOfThought` (think step-by-step), `ReAct` (reason + act loop), `MultiChainComparison` (compare multiple completions). You wrap a Signature in a Module to get a compiled prompt pattern.

By the end of this note you can express any LLM call as a 5-line DSPy program and reason about which module fits which task.

## 🎯 Learning Objectives

- Declare **Signatures** with `InputField` and `OutputField`.
- Use the four built-in **Modules**: `Predict`, `ChainOfThought`, `ReAct`, `MultiChainComparison`.
- Compose modules into multi-step programs.
- Distinguish between inline and class-based signatures.
- Configure type hints and descriptions for optimal compilation.
- Avoid the four most common signature and module pitfalls.

## 1. The Signature Primitive

```python
import dspy

# Inline signature
generate_answer = dspy.Predict("question, context -> answer")

# Class-based signature (preferred — type-safe)
class GenerateAnswer(dspy.Signature):
    """Answer a question using the provided context."""
    question: str = dspy.InputField(desc="The user's question")
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    answer: str = dspy.OutputField(desc="Concise answer grounded in context")

generate = dspy.Predict(GenerateAnswer)
result = generate(
    question="What is the capital of France?",
    context=["Paris is the capital of France."],
)
print(result.answer)
# "Paris"
```

### Anatomy of a Signature

```python
class MySignature(dspy.Signature):
    """One-line docstring — DSPy uses this as the system prompt."""
    input_field_1: type = dspy.InputField(desc="What goes in — be specific")
    input_field_2: type = dspy.InputField(desc="Another input")
    output_field_1: type = dspy.OutputField(desc="What comes out — specify format")

    # Optional: advanced settings
    class Config:
        # Custom instructions for the LM
        instructions = "Be concise, accurate, and cite sources."
```

Three components:
1. **Docstring** — the system prompt. Be specific about the task and constraints.
2. **`InputField`s** — typed inputs with `desc=` for guidance.
3. **`OutputField`s** — typed outputs with `desc=` specifying format.

### Supported Types

```python
class TypedSignature(dspy.Signature):
    question: str = dspy.InputField()                          # str
    options: list[str] = dspy.InputField()                     # list[str]
    metadata: dict = dspy.InputField()                         # dict
    optional_input: str | None = dspy.InputField()             # Optional
    count: int = dspy.InputField()                             # int
    probability: float = dspy.InputField()                     # float

    answer: str = dspy.OutputField()
    citations: list[int] = dspy.OutputField()                  # list[int]
    confidence: float = dspy.OutputField()
```

### Custom Output Types

```python
from pydantic import BaseModel, Field

class Citation(BaseModel):
    source_id: int
    quote: str = Field(max_length=200)

class WithStructuredOutput(dspy.Signature):
    """Answer with structured citations."""
    context: list[str] = dspy.InputField()
    query: str = dspy.InputField()
    citations: list[Citation] = dspy.OutputField(desc="List of citations, each with source_id and quote")
```

DSPy sends the Pydantic schema as JSON to the LM, which returns JSON matching the schema. The output is validated against the schema before returning.

## 2. The Four Core Modules

### `Predict` — Basic Generation

```python
class BasicQA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

basic = dspy.Predict(BasicQA)
result = basic(question="What is 2+2?")
# result.answer = "4"
```

`Predict` is the simplest module: it generates the output directly. **Use it for simple tasks where chain-of-thought doesn't help.**

### `ChainOfThought` — Reasoning Before Answering

```python
class ReasonedQA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

reasoned = dspy.ChainOfThought(ReasonedQA)
result = reasoned(question="What is 12 * 15?")
# result.reasoning = "12 * 15 = 12 * 10 + 12 * 5 = 120 + 60 = 180"
# result.answer = "180"
```

`ChainOfThought` adds a `reasoning` field to the output — the LLM thinks step-by-step before answering. **Use it for math, logic, multi-step reasoning.**

Research shows CoT improves accuracy by **30-50%** on math/reasoning tasks. The cost: 1-3× more tokens (the reasoning chain).

### `ReAct` — Reasoning + Tool Use

```python
class SearchThenAnswer(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

def search(query: str) -> list[str]:
    """Search the web for the given query."""
    # ... (your search implementation)
    return [r["content"] for r in tavily.search(query)]

agent = dspy.ReAct(
    SearchThenAnswer,
    tools=[search],
    max_iters=5,
)
result = agent(question="What is the population of Tokyo?")
# ReAct loop:
# Thought: I need to search for "Tokyo population"
# Action: search("Tokyo population")
# Observation: ["Tokyo has 13.96M residents..."]
# Thought: I have enough to answer
# Answer: "13.96M"
```

`ReAct` is an agent loop: **thought → action → observation → repeat**. **Use it for tasks that need external tools or multi-step research.**

### `MultiChainComparison` — Multiple CoTs, Pick Best

```python
class Compare(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Generate 3 CoT completions, pick the one that best matches the question
mcc = dspy.MultiChainComparison(Compare, M=3)
result = mcc(question="What is the meaning of life?")
# Internally: 3x ChainOfThought, then pick best
```

`MultiChainComparison` is **majority voting with chain-of-thought**. Use it for tasks where multiple attempts improve accuracy (classification, factual QA). Cost: 3× the LLM calls.

## 3. Composing Modules

```python
class RAG(dspy.Module):
    """Multi-stage RAG: rewrite query, retrieve, generate."""

    def __init__(self, retriever):
        super().__init__()

        # Stage 1: Query rewriting
        class RewriteQuery(dspy.Signature):
            """Rewrite the query for vector search."""
            original_query: str = dspy.InputField()
            rewritten_query: str = dspy.OutputField()
        self.rewrite = dspy.ChainOfThought(RewriteQuery)

        # Stage 2: Generation
        class Generate(dspy.Signature):
            """Answer using retrieved context."""
            context: list[str] = dspy.InputField()
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()
        self.generate = dspy.ChainOfThought(Generate)

        self.retriever = retriever  # not a DSPy module — a callable

    def forward(self, query: str) -> dspy.Prediction:
        with dspy.context(retrievers=[self.retriever]):
            # Stage 1: Rewrite
            rewritten = self.rewrite(original_query=query).rewritten_query

            # Stage 2: Retrieve (non-DSPy call)
            results = self.retriever.search(rewritten, top_k=10)
            context = [r.content for r in results]

            # Stage 3: Generate
            return self.generate(query=query, context=context)
```

`dspy.Module` is a regular Python class. `forward()` orchestrates the steps. Each step can be a DSPy Module, a plain function call, or a mix.

## 4. Configuring the LM

```python
import os

# OpenAI
gpt4o = dspy.LM("openai/gpt-4o", api_key=os.environ["OPENAI_API_KEY"], max_tokens=2000)

# Anthropic
claude = dspy.LM("anthropic/claude-3-5-sonnet-20241022", api_key=os.environ["ANTHROPIC_API_KEY"])

# Local via Ollama
local = dspy.LM("ollama/llama3.1", api_base="http://localhost:11434")

# Configure globally
dspy.configure(lm=gpt4o)

# Or per-call
with dspy.context(lm=claude):
    result = generate(query="...")
```

### Per-Stage LM

```python
class MultiLM(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteQuery)  # cheap model
        self.generate = dspy.ChainOfThought(Generate)  # expensive model

    def forward(self, query):
        with dspy.context(lm=gpt4o_mini):  # cheap LM for rewrite
            rewritten = self.rewrite(original_query=query).rewritten_query
        # Generate uses the default LM
        return self.generate(rewritten_query=rewritten)
```

## 5. The Synthesis: from Signature to Module to Program

```python
# 1. Define signature
class QA(dspy.Signature):
    context: list[str] = dspy.InputField(desc="Retrieved passages")
    query: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Faithful, concise answer")

# 2. Wrap in module
qa_module = dspy.ChainOfThought(QA)

# 3. Use directly
result = qa_module(context=ctx, query=q)

# 4. Or wrap in custom module
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(QA)

    def forward(self, query, context):
        return self.qa(context=context, query=query)
```

## 6. ❌/✅ Antipatterns

### ❌ Manually-formatted prompts in DSPy

```python
# ⚠️ Defeats the purpose — DSPy compiles the prompt, not you
prompt = f"Answer this question: {q}"
result = lm.invoke(prompt)  # not using DSPy
```

### ✅ Use signatures + modules

```python
result = generate(query=q, context=ctx)  # DSPy compiles
```

### ❌ Vague descriptions

```python
class VagueSig(dspy.Signature):
    """QA."""
    question: str = dspy.InputField()  # no description
    answer: str = dspy.OutputField()   # no description
```

### ✅ Specific, formatted-output descriptions

```python
class SpecificSig(dspy.Signature):
    """Answer a factual question. Cite your sources by passage ID. Be concise."""
    question: str = dspy.InputField(desc="User's natural-language question")
    context: list[str] = dspy.InputField(desc="Top-10 retrieved passages")
    answer: str = dspy.OutputField(desc="1-3 sentence answer grounded in context")
    citations: list[int] = dspy.OutputField(desc="Passage IDs that support the answer")
```

### ❌ ChainOfThought for simple tasks

```python
# ⚠️ 3× tokens, no accuracy gain
result = dspy.ChainOfThought(SimpleTextRewrite).forward(text="...")
```

### ✅ Use Predict for simple generation

```python
result = dspy.Predict(SimpleTextRewrite).forward(text="...")
```

### ❌ ReAct without tools

```python
# ⚠️ ReAct without tools just generates text
agent = dspy.ReAct(QA, tools=[])  # pointless
```

### ✅ ReAct only when tools are needed

```python
agent = dspy.ReAct(
    SearchThenAnswer,
    tools=[web_search, calc, database],  # real tools
)
```

### ❌ Missing type annotations on fields

```python
# ⚠️ DSPy can't validate; LM doesn't know format
class UntypedSig(dspy.Signature):
    question = dspy.InputField()  # no type
    answer = dspy.OutputField()
```

### ✅ Type-annotate every field

```python
class TypedSig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

## 7. Production Reality

**Caso real — Production RAG Project:** The portfolio's RAG system uses DSPy with a `RAG` module wrapping `ChainOfThought(QA)`. The compiled program consistently outperforms the previous hand-tuned prompt by **12% on the RAGAS faithfulness metric** (78% → 90%). The compiled prompt discovered via MIPRO is qualitatively better than what the team had hand-tuned.

**Caso real — Multi-Agent Research System:** The research node is a DSPy `ReAct` module with `web_search` and `arxiv_search` tools. After compilation with MIPRO, the tool-selection accuracy improves from 64% to 81%. **No hand-written prompt achieves this**; it's the optimizer finding the right few-shot examples and tool descriptions.

## 📦 Compression Code

```python
# 📦 Compression: Signatures + Modules in 50 lines

import dspy
import os

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
dspy.configure(lm=lm)


# 1. Inline signatures for simple tasks
basic_qa = dspy.Predict("question -> answer")
result = basic_qa(question="What is 2+2?")

# 2. Class signatures for type safety
class ReasonedQA(dspy.Signature):
    """Answer with step-by-step reasoning."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
reasoned_qa = dspy.ChainOfThought(ReasonedQA)

# 3. Custom module composition
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ReasonedQA)

    def forward(self, query: str, context: list[str]):
        return self.generate(query=query, context=context)

rag = RAG()
result = rag(query="What is X?", context=["X is Y."])
print(result.reasoning)  # "I should look at the context..."
print(result.answer)     # "Y"
```

## 🎯 Key Takeaways

1. **Signatures are contracts** — `InputField`, `OutputField`, docstring, and types.
2. **Specific descriptions beat clever ones** — the LM reads your `desc=` strings.
3. **Four modules** — `Predict` (basic), `ChainOfThought` (reasoning), `ReAct` (agent), `MultiChainComparison` (vote).
4. **Compose modules in `dspy.Module`** — each `forward()` orchestrates steps.
5. **Use Predict for simple tasks** — ChainOfThought costs 1-3× tokens.
6. **Type-annotate every field** — DSPy validates outputs to Pydantic / types.
7. **Per-stage LM selection** — cheap LM for rewriting, expensive LM for generation.

## References

- [[00 - Welcome to DSPy and Prompt Compilation|Welcome]] — course map.
- [[02 - Optimizers - BootstrapFewShot MIPRO and COPRO|Optimizers]] — compile these signatures.
- [[03 - DSPy for RAG - Retrieval Modules and Signatures|RAG]] — retriever modules.
- DSPy signatures: https://dspy.ai/learn/program/signatures/
- DSPy modules: https://dspy.ai/learn/program/modules/