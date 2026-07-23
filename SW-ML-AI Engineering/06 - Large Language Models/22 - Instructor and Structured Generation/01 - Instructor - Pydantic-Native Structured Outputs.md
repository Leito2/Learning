# 🎯 01 - Instructor — Pydantic-Native Structured Outputs

> **The orchestration layer for LLM validation. Pairs 1:1 with your Pydantic v2 skills from [[03 - Advanced Python/06 - Pydantic Deep Dive]].**

## 🎯 Learning Objectives
- Wire `instructor.Instructor` into OpenAI, Anthropic, Groq, Cohere, Mistral, and local Ollama in 3 lines
- Pass a Pydantic `BaseModel` as `response_model=` and let Instructor handle retries automatically
- Stream partial objects with `partial=True` and validate each chunk as the model emits tokens
- Use multi-modal fields (`Image`, `Audio`, `PDF`) as first-class schema members
- Build async batch extraction pipelines with `asyncio.gather` and bounded concurrency
- Avoid the five most common Instructor pitfalls (mode mismatch, retry storm, token leakage, partial-validation ambiguity, cost blow-up)

## Introduction

Instructor is the bridge between the LLM provider's chat-completion API and the typed world your application actually wants. It started in 2023 as a 200-line patch over the OpenAI Python SDK that wrapped `client.chat.completions.create` to accept a Pydantic model instead of `messages`. Two years later it is the de facto validation layer for typed LLM applications — used by the rankers at SGLang's benchmarks, by the LLM-as-Judge evaluators in DeepEval, by the tool schemas in LangChain agents, and by every production codebase that treats the model as a typed function rather than a string generator.

The library's bet is simple: **declarative validation beats imperative parsing**. Instead of asking the model "give me JSON" and then defending against every malformed response with manual `try/except`, you declare what you want as a Pydantic model and let Instructor ferry the model output through that model as a validator. When validation fails, Instructor automatically re-sends the conversation with the error message appended to the prompt, up to a configurable number of retries. When validation succeeds, you get a fully-typed `BaseModel` instance — not a string, not a dict, not a partial parse.

By the end of this note you will have written a typed extraction pipeline that retries on validation failure, streams partial objects to a UI, processes PDFs as schema fields, and runs batch extraction with bounded concurrency. You will also understand why Instructor chose function-calling under the hood and how to switch between modes for providers that lack first-class tool support.

![Instructor architecture: Pydantic model → validation loop → typed output](https://python.use-instructor.com/_/image-01.png)

---

## 1. Why Instructor Exists

### 1.1 The naive pipeline and its five failure modes

Consider the simplest possible structured extraction:

```python
from openai import OpenAI
import json

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract name, age, and role as JSON."},
        {"role": "user", "content": "Maria is a 28 year old ML engineer."},
    ],
    response_format={"type": "json_object"},
)
data = json.loads(resp.choices[0].message.content)
```

This looks correct and will succeed in dev. In production it fails in five distinct ways:

| Failure | Cause | Frequency |
|---------|-------|:---------:|
| **No schema** | `response_format={"type": "json_object"}` is JSON-mode but not schema-aware. The model can return `{"name": "Maria", "age": "twenty-eight"}` — valid JSON, wrong type | ~10-15% |
| **Missing required field** | Model drops a key. `data["role"]` raises `KeyError` | ~3-7% |
| **Hallucinated field** | Model invents `data["salary"]`. Downstream code reads it without checking | ~2-4% |
| **Truncated JSON** | Token limit cutoff mid-object. `json.loads` raises | ~1-3% |
| **Off-schema types** | Boolean encoded as `"true"`, list as comma-separated string, nested objects flattened | ~5-10% |

For a one-shot script this is annoying. For a production pipeline feeding a database or downstream service, it is a defect source. ❌ Every team that built "just call GPT and parse JSON" reaches the same conclusion: schema enforcement has to live **inside** the validation loop, not outside it.

### 1.2 The Instructor fix

Instructor moves the schema into the request and feeds validation errors back to the model:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int
    role: str

person = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Maria is a 28 year old ML engineer."},
    ],
    response_model=Person,
)
print(person.name, person.age, person.role)  # Maria 28 ML engineer — typed
```

Three lines replace twenty lines of `try/except` and string parsing. When `Person` validation fails (e.g. age not parseable as `int`), Instructor re-sends the prompt with the `ValidationError` appended and tries again — by default up to 3 times. The model sees its own mistake and corrects it on retry. In practice the success rate jumps from ~85% (naive JSON mode) to ~99.5% with `response_model=Person` because the model has the schema in its context and the validation errors as a feedback signal.

💡 **Tip:** This "validation error as feedback" pattern is the same one DSPy uses for prompt optimization in [[06 - Large Language Models/21 - DSPy and Prompt Compilation]] and the same one your LangGraph state reducers use in [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns]]. It is the canonical loop for any system that treats an LLM as a typed function.

### 1.3 Case real: Replit's bug-bot

Replit's internal agent ([source: Jason Liu, Instructor creator, 2024](https://python.use-instructor.com)) runs hundreds of LLM calls per session to classify user code into categories like `bug-fix-needed`, `feature-request`, `question`. Before Instructor the bot would occasionally misclassify requests as strings ("I think this is a feature request") instead of enum values — silently corrupting routing. After Instructor with `response_model=Category` (an enum), the validation loop catches and retries all such cases. The bot's classification accuracy went from ~88% to ~99% without any prompt changes — **schema enforcement and retry feedback did all the work**.

---

## 2. Anatomy of an Instructor Call

### 2.1 The patch — `instructor.from_openai(...)`

`instructor.from_<provider>()` returns a wrapped client that intercepts `chat.completions.create` and adds the `response_model` parameter. The patching is non-invasive: the returned client still supports every argument the underlying SDK accepts. You can mix Pydantic calls and plain chat calls on the same client.

```python
import instructor
from openai import AsyncOpenAI

# Sync client
sync_client = instructor.from_openai(OpenAI())

# Async client
async_client = instructor.from_openai(AsyncOpenAI())

# Anthropic
import anthropic
anth_client = instructor.from_anthropic(anthropic.Anthropic())

# Cohere
import cohere
cohere_client = instructor.from_cohere(cohere.Client())

# Groq (uses OpenAI-compatible API under the hood)
groq_client = instructor.from_openai(
    OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY),
    model="llama-3.3-70b-versatile",
    mode=instructor.Mode.JSON,  # Groq doesn't yet support function calling for all models
)

# Self-hosted Ollama / vLLM
ollama_client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)

# Generic OpenAI-compatible endpoint via LiteLLM
from litellm import completion
litellm_client = instructor.from_litellm(completion)
```

💡 **Tip:** If you already use LiteLLM as your multi-provider transport (you do — see [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]]), `instructor.from_litellm(completion)` is the cleanest way to combine them. Instructor handles validation; LiteLLM handles routing across 100+ providers.

### 2.2 Mode selection

Instructor uses different transport strategies depending on what the provider supports. The `Mode` enum controls this:

| Mode | Transport | Providers |
|------|-----------|-----------|
| `Mode.TOOLS` | Function calling / tool use | OpenAI, Anthropic, Gemini, Groq (Llama 3.3), Mistral |
| `Mode.JSON` | JSON mode + parse | All OpenAI-compatible (Ollama, vLLM, TGI) |
| `Mode.MD_JSON` | Markdown JSON code blocks | Older models without JSON mode |
| `Mode.JSON_SCHEMA` | Native JSON Schema constraint | OpenAI 2024+, Gemini 1.5+ |
| `Mode.PARALLEL_TOOLS` | Multiple tool calls in one turn | OpenAI parallel tool calling |
| `Mode.MOCK_JSON` | Returns deterministic mock data | Tests |
| `Mode.LLAMA3_JSON` | Llama-3 specific JSON prompting | Ollama Llama 3 |
| `Mode.GEMINI_TOOLS` | Gemini native function calling | Gemini |

Default mode is inferred from the provider. For OpenAI it is `TOOLS`. For Anthropic it is `TOOLS`. For Ollama it is `JSON` (because Ollama does not yet support function calling for most models). For `LiteLLM`, default is `JSON` unless overridden.

```python
# Force JSON mode for an OpenAI-compatible endpoint that lacks function calling
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1"),
    mode=instructor.Mode.JSON,
)

# Use JSON Schema mode for OpenAI 2024+ models
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON_SCHEMA)
```

❌ **Common antipattern:** switching modes mid-pipeline without confirming the target model supports them. Mode mismatch is the single most common cause of `InstructorError` in production. Always pin the mode for the model you deploy.

### 2.3 Retries on validation failure

Instructor re-sends the conversation automatically when the response fails to validate. The retry count and back-off are configurable:

```python
from instructor import Retry

person = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Maria is 28 years old."}],
    response_model=Person,
    max_retries=4,  # default is 3
)
```

Each retry appends the `ValidationError` to the messages. The model sees e.g. `ValidationError: age must be int, got "twenty-eight"`. This is **hot in-context learning**: the model now knows exactly what shape to produce on the next attempt.

You can also pass a custom retry policy:

```python
from instructor import Retry
import openai

def retry_on_rate_limit(error):
    return isinstance(error, openai.RateLimitError)

person = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}],
    response_model=Person,
    max_retries=4,
    retry=retry_on_rate_limit,  # custom retry for transient provider errors
)
```

💡 **Tip:** The `Retry` callable runs **before** the `max_retries` counter. A typical pattern is `retry on RateLimitError with exponential backoff, but raise immediately on ValidationError after max_retries`. The default `Retry()` raises on any error other than `ValidationError`. Override it if you want retries across error families.

---

## 3. Streaming Partial Objects

For real-time UIs (voice agents, IDE plugins, chat surfaces), waiting for the full completion is unacceptable. Instructor's `partial=True` streams validated partial objects as the model emits tokens:

```python
from instructor import Partial[]
import instructor
from openai import OpenAI

class Person(BaseModel):
    name: str
    age: int
    role: str

client = instructor.from_openai(OpenAI())

stream = client.chat.completions.create_partial(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Maria is 28 and works on LLMs."}],
    response_model=Person,
)

for partial in stream:
    print(partial)
    # Person(name='Maria', age=0, role='')
    # Person(name='Maria', age=0, role='')
    # Person(name='Maria', age=28, role='')
    # Person(name='Maria', age=28, role='LLM engineer')
```

Each `partial` is a fully-typed `Person` instance with whatever fields have been determined so far. The frontend can render `partial.age` as soon as it arrives — typically 100-300ms after the request starts, vs the full completion latency. For a UI showing 100 tokens of streaming text, this can mean **5-10× faster time-to-first-value**.

⚠️ **Watch out:** partial objects can be in invalid intermediate states. `partial.age` might be `0` (the Pydantic default) and then `28` (the real value), but `age=0` between the two is not a valid representation of a real-world person. Treat partial objects as **progressive displays**, not as committed data. Commit only on the final non-partial object.

```python
# Async streaming
async_client = instructor.from_openai(AsyncOpenAI())

stream = await async_client.chat.completions.create_partial(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Maria is 28 and works on LLMs."}],
    response_model=Person,
)

final_person: Person | None = None
async for partial in stream:
    final_person = partial  # last chunk is the final validated object
    # Or: stream to UI: await websocket.send_json(partial.model_dump())
```

### 3.1 Case real: Realtime IDE auto-completion

Cursor-style code assistants (think Copilot Workspace) cannot wait 800ms for a full completion — they need to stream code as it generates. By treating the completion as a structured `CodeEdit` schema (`{language, before, after, explanation}`) with `partial=True`, the IDE can render the `before`/`after` diff in real-time while the `explanation` field streams last. Validation on each partial ensures the syntax tree never goes off-schema during the render.

---

## 4. Multi-Modal Structured Outputs

Production RAG and document processing require structured extraction from images, audio, and PDFs. Instructor treats these as schema fields:

```python
import instructor
from instructor import Image, Audio
from openai import OpenAI
from pydantic import BaseModel

class Receipt(BaseModel):
    store: str
    total: float
    items: list[str]
    receipt_image: Image  # Instructor handles base64 encoding + provider upload

client = instructor.from_openai(OpenAI())

receipt = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                "Extract structured fields from this receipt.",
                Image.from_url("https://example.com/receipt.jpg"),
            ],
        }
    ],
    response_model=Receipt,
)
```

`Image`, `Audio`, and `PDF` accept `from_url`, `from_path`, or `from_base64` constructors. Under the hood Instructor handles encoding and provider-specific upload (e.g. Anthropic requires multipart, OpenAI accepts base64 in content array). The schema *contains* the multi-modal data — not just a reference — which means the structured object is self-contained and reproducible.

⚠️ **Watch out:** Multi-modal field tokens count toward your input bill. A 4 MB receipt image is ~1000 input tokens after Claude encoding, ~1700 for OpenAI. For high-throughput pipelines, pre-process the image with OCR and send the text instead.

💡 **Tip:** Even when you only need the text, passing the image alongside structured text often improves extraction quality by 10-30%. The model can verify ambiguous fields ("is this a 5 or an S?") against visual context.

### 4.1 Capstone preview: PDF extraction

The capstone (Note 05) demonstrates a multi-modal PDF extraction pipeline:

```python
class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float
    confidence: float  # Instructor's automatic self-reported confidence

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float

invoices = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            "Extract every invoice from this PDF with line-item granularity.",
            PDF.from_path("/uploads/invoice-batch.pdf"),
        ],
    }],
    response_model=list[Invoice],
    max_retries=4,
)
```

That single call returns `list[Invoice]` — fully typed, validated, with `confidence` self-reported by the model. Compare with the naive pipeline: download PDF → chunk it → call LLM per chunk → manually merge line items → manually validate totals → manually re-OCR ambiguous fields. Instructor collapses this to one call.

---

## 5. Async Batch Extraction

For high-throughput extraction (10K+ documents per hour), synchronous calls leave GPU time on the table. Instructor exposes async clients with the same surface:

```python
import asyncio
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(AsyncOpenAI())

async def extract_one(doc: str) -> Person:
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": doc}],
        response_model=Person,
    )

# Bounded concurrency with a semaphore
sem = asyncio.Semaphore(20)  # max 20 concurrent requests

async def extract_with_limit(doc: str) -> Person:
    async with sem:
        return await extract_one(doc)

docs = ["..." for _ in range(1000)]
people = await asyncio.gather(*(extract_with_limit(d) for d in docs))
```

`asyncio.Semaphore(N)` caps the concurrency to whatever your rate limit allows (20 for OpenAI tier 1, 100 for tier 4+). `asyncio.gather` distributes the work. For 1000 documents at 20 concurrency, total wall time is `~50 × single_call_latency` — typically 15-30 seconds for `gpt-4o-mini`.

💡 **Tip:** For pipelines that cross the OpenAI rate-limit boundary, wrap the semaphore in an exponential backoff loop:

```python
import backoff

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
async def extract_with_retry(doc: str) -> Person:
    async with sem:
        return await extract_one(doc)
```

This pairs with the retry policy from [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM]] — same backoff pattern, different library.

---

## 6. Production Anti-Patterns

### 6.1 Antipattern 1: Mode mismatch

```python
# ❌ Production bug: Ollama model doesn't support function calling but Mode is TOOLS
client = instructor.from_openai(OpenAI(base_url="http://localhost:11434/v1"))
client.chat.completions.create(model="llama3.1", messages=..., response_model=Person)

# ✅ Correct: explicitly set Mode.JSON for non-function-calling endpoints
client = instructor.from_openai(OpenAI(base_url="http://localhost:11434/v1"), mode=instructor.Mode.JSON)
```

### 6.2 Antipattern 2: Unbounded retry loops

```python
# ❌ Production bug: max_retries=100 on a schema the model fundamentally cannot satisfy
person = client.chat.completions.create(..., response_model=Person, max_retries=100)

# ✅ Correct: cap at 3-5 and add a fallback strategy
try:
    person = client.chat.completions.create(..., response_model=Person, max_retries=4)
except ValidationError as e:
    person = fallback_extract(content)  # manual regex or null-fields model
```

Unbounded retries bill the customer, balloon latency, and rarely succeed where 3-5 retries do not. Cap the retries and have a fallback strategy for the residual failures.

### 6.3 Antipattern 3: Logging raw responses

```python
# ❌ Compliance bug: logging the raw response can leak PII
logger.info(resp.choices[0].message.content)

# ✅ Correct: log the validated Pydantic model dump or specific fields only
logger.info(person.model_dump_json(exclude={"raw_pii_field"}))
```

Instructor's `response_model` is the natural scrubber — log the validated fields, not the raw response. For HIPAA / GDPR pipelines, pair with [[06 - Large Language Models/15 - LLM Security and Guardrails]] for PII redaction before logging.

### 6.4 Antipattern 4: Partial-validation ambiguity

```python
# ❌ Frontend bug: rendering partial objects as committed data
async for partial in stream:
    save_to_db(partial)  # saves invalid intermediate states

# ✅ Correct: accumulate, save only the final object
final = None
async for partial in stream:
    final = partial
save_to_db(final)
```

### 6.5 Antipattern 5: Token leakage via validators

```python
# ❌ Surprise: validators that use external services leak the model output to third parties
@validator("name")
def check_name_in_crm(cls, v):
    requests.get(f"https://crm.example.com/lookup?name={v}")  # leaks PII
    return v

# ✅ Correct: keep validators self-contained; move external lookups to a separate post-processing step
```

---

## 7. Validation as Feedback — The Deeper Pattern

The Instructor loop (request → validate → re-send on failure) is one instance of a general pattern: **treat validation errors as additional prompt context**. This same pattern powers:

| System | How it uses validation feedback |
|--------|---------------------------------|
| **Instructor** | Re-sends messages with `ValidationError` appended |
| **DSPy** | `assert` statements + `Suggest` rewrite prompts |
| **Outlines** | Constrained generation — the model *cannot* produce invalid tokens |
| **LMQL** | `assert` blocks in the DSL |
| **LangGraph** | State reducers with typed validation; failed nodes re-route |
| **SGLang `regex`** | Provider-side token constraint |

Every production agent framework in your portfolio uses some form of this loop. Instructor is the explicit Python expression of it; the others are specialized implementations for specific constraints (grammar, type, token-level).

Caso real: The `Automated LLM Evaluation Suite` in your portfolio uses Instructor under the hood for LLM-as-Judge. Each judge call has `response_model=Judgment` (an enum + reasoning), `max_retries=2`, and feeds `ValidationError` back to the judge model. This is why your suite's inter-annotator agreement is high even with smaller judge models — the schema enforces the format, and the retries enforce the rubric.

---

## 🎯 Key Takeaways

- Instructor is the validation-orchestration layer for LLM APIs; Pydantic v2 is the schema layer.
- `response_model=YourBaseModel` replaces manual JSON parsing with declarative validation; `max_retries` feeds `ValidationError` back to the model for self-correction.
- `Mode.TOOLS` (function calling) is the default and most reliable; `Mode.JSON` is the fallback for non-function-calling endpoints.
- Streaming `partial=True` enables 5-10× faster time-to-first-value at the cost of accepting intermediate invalid states.
- Multi-modal fields (`Image`, `Audio`, `PDF`) are first-class schema members, simplifying PDF extraction pipelines.
- Async + `asyncio.Semaphore` + `backoff` is the standard pattern for high-throughput batch extraction.
- Avoid mode mismatch, unbounded retries, raw-response logging, partial-validation ambiguity, and validator side effects.
- The "validation as feedback" loop is the same pattern DSPy, LangGraph, Outlines, and LMQL all specialize.

## References

- Instructor docs — [python.use-instructor.com](https://python.use-instructor.com)
- [[03 - Advanced Python/06 - Pydantic Deep Dive|Pydantic Deep Dive]] — foundation for `response_model`
- [[06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM|LLM Gateway Patterns]] — multi-provider transport (`instructor.from_litellm`)
- [[06 - Large Language Models/20 - RAG Evaluation Deep Dive|RAG Evaluation Deep Dive]] — LLM-as-Judge with structured outputs
- [[06 - Large Language Models/21 - DSPy and Prompt Compilation|DSPy and Prompt Compilation]] — alternative compilation-based approach
- [[07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks|Production Agent Frameworks]] — smolagents, PydanticAI internally use structured outputs
- [[07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns|LangGraph Deep Patterns]] — state validation with Pydantic
- [[09 - MLOps y Produccion/31 - Evidently AI and Phoenix|Evidently AI and Phoenix]] — trace structured outputs as JSON attributes
- [[10 - Cloud, Infra y Backend/31 - FastAPI for ML|FastAPI for ML]] — the capstone service uses FastAPI streaming
- Anthropic structured outputs docs — [docs.anthropic.com/en/docs/build-with-claude/structured-outputs](https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs)
- OpenAI structured outputs docs — [platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs)
