# 🤗 Generation, Decoding, and Structured Output

## 🎯 Learning Objectives

- Understand the autoregressive loop inside `model.generate()` and how it wraps the forward pass.
- Configure `GenerationConfig` parameters (temperature, top_k, top_p, repetition_penalty) with theoretical justification.
- Implement custom `LogitsProcessor`, `LogitsWarper`, and `StoppingCriteria` for fine-grained inference control.
- Apply beam search, constrained decoding, and structured output techniques for production use cases.

## Introduction

Training teaches a model the distribution of language; generation samples from that distribution to produce coherent text. The gap between a trained model and a useful product is bridged by decoding strategies. Greedy decoding often produces repetitive, generic output, while naive random sampling yields incoherent gibberish. Modern systems balance diversity and quality through temperature scaling, nucleus sampling, and constrained beam search.

This note demystifies `model.generate()`, the Swiss Army knife of text generation in `transformers`. We trace how a single token prediction expands into a full sequence, how `GenerationConfig` shapes the probability distribution at each step, and how custom processors enforce hard constraints like JSON schemas or banned phrases. These techniques connect to inference optimization in [[06 - Large Language Models/13 - vLLM Deep Dive|vLLM Deep Dive]] and structured API design in [[10 - FastAPI|FastAPI]].

---

## 1. Generation Internals and Decoding Strategies

Autoregressive generation treats text as a Markov chain with long-range dependencies. At each timestep $t$, the model outputs a conditional probability distribution over the vocabulary $\mathcal{V}$:

$$P(x_t \mid x_{<t}) = \text{softmax}\left(\frac{\ell_t}{T}\right)$$

where $\ell_t \in \mathbb{R}^{|\mathcal{V}|}$ are the logits from the final LM head and $T$ is the temperature. The decoder's job is to select $x_t$ from this distribution according to a strategy that maximizes fluency, diversity, or task-specific correctness.

### Decoding Strategies

**Greedy decoding** selects $x_t = \arg\max P(x_t)$ at every step. It is deterministic and fast but prone to repetitive loops and locally optimal but globally poor sequences.

**Beam search** maintains $k$ partial hypotheses $\{h^{(1)}, \dots, h^{(k)}\}$. At each step, it expands all $k \times |\mathcal{V}|$ candidates and prunes back to $k$ by cumulative log-probability:

$$h^{(i)}_{1:t} = \arg\operatorname{top-}k \sum_{j=1}^{t} \log P(x_j \mid x_{<j})$$

Beam search approximates maximum-likelihood decoding but can still produce bland, high-probability text.

**Temperature scaling** controls the sharpness of the distribution:

$$P_T(x_t \mid x_{<t}) = \frac{\exp(\ell_t / T)}{\sum_{v} \exp(\ell_v / T)}$$

$T < 1$ sharpens (more deterministic), $T > 1$ flattens (more random). At $T \to 0$, sampling approaches greedy.

**Top-k sampling** restricts candidates to the $k$ most likely tokens, preventing bizarre low-probability choices.

**Top-p (nucleus) sampling** dynamically selects the smallest set $\mathcal{V}_p$ whose cumulative probability exceeds $p$:

$$\mathcal{V}_p = \min\left\{ \mathcal{V}' \subseteq \mathcal{V} : \sum_{v \in \mathcal{V}'} P(v) \geq p \right\}$$

This adapts the candidate pool to the model's confidence at each step.

**Repetition penalty** divides logits of previously generated tokens by a factor $\rho > 1$:

$$\ell_t'[i] = \begin{cases} \ell_t[i] / \rho & i \in \mathcal{G} \\ \ell_t[i] & i \notin \mathcal{G} \end{cases}$$

where $\mathcal{G}$ is the set of token IDs generated so far. This breaks repetition loops without hard masking.

### GenerationConfig and model.generate()

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

gen_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

inputs = tokenizer("The future of AI is", return_tensors="pt")

outputs = model.generate(**inputs, generation_config=gen_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Manual Generation Loop

The internal loop that `generate()` abstracts — useful for understanding:

```python
past_key_values = None
input_ids = inputs["input_ids"]

for _ in range(20):
    out = model(input_ids, past_key_values=past_key_values, use_cache=True)
    logits = out.logits[:, -1, :]
    past_key_values = out.past_key_values

    probs = torch.softmax(logits / 0.8, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    input_ids = torch.cat([input_ids, next_token], dim=-1)

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
```

The `past_key_values` cache avoids recomputing the full attention for the prefix at each step — a quadratic saving. Without it, generating $N$ tokens from a sequence of length $L$ would cost $O(N \cdot (L+N)^2)$ instead of $O(L^2 + N \cdot (L+N))$.

During each step, the logits tensor has shape $[B, |\mathcal{V}|]$, where $B$ is the batch size (typically 1 for generation). A single forward pass with cached keys/values costs only $O(L + N)$ per layer instead of $O((L+N)^2)$ — the difference between 100ms and multiple seconds at 4096 tokens.

✅ **Antipattern: Conflicting sampling parameters**
```python
# ❌ Temperature=2.0 + top_k=1: flatten then argmax destroys sampling
GenerationConfig(temperature=2.0, top_k=1, do_sample=True)

# ✅ Use temperature and top_p together for smooth control
GenerationConfig(temperature=0.8, top_p=0.9, do_sample=True)
```

> **Caso real: Perplexity.ai** uses a hybrid decoding stack: nucleus sampling ($p=0.9$, $T=0.7$) for general queries, but switches to greedy + repetition penalty for fact-retrieval prompts where hallucination must be minimized. Their serving layer dynamically selects the `GenerationConfig` based on prompt intent classification.

⚠️ **Temperature and top_k/top_p interaction**: Setting `temperature=2.0` with `top_k=1` is contradictory. The flattened distribution is immediately discarded, silently disabling sampling diversity.

⚠️ **KV-cache + attention mask mismatch**: When using `past_key_values`, the attention mask must be extended by one position each step. Forgetting causes position embedding misalignment. Prefer `model.generate()` over manual loops.

💡 **Prompt engineering**: Generation quality is highly sensitive to prompt formatting. Test at least 3 prompt templates per use case before tuning generation parameters.

---

## 2. Structured Output and Constrained Generation

Unconstrained generation is insufficient for production systems that must emit valid JSON, SQL, or domain-specific languages. The naive approach — generate freely, then validate — is brittle because a single syntax error invalidates the entire output. Instead, we **constrain the probability distribution** so invalid tokens receive zero probability at every step.

### Custom LogitsProcessor

A `LogitsProcessor` is called after the forward pass but before sampling. It receives the current logits and can mask any token by setting its logit to $-\infty$:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch

class JSONLogitsProcessor(LogitsProcessor):
    """Enforces valid JSON structure by masking invalid tokens."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = "start"

    def __call__(self, input_ids, scores):
        last_token = self.tokenizer.decode(input_ids[0, -1:])
        self._update_state(last_token)
        valid_tokens = self._valid_tokens()
        mask = torch.full_like(scores, float("-inf"))
        for token_id in valid_tokens:
            mask[:, token_id] = scores[:, token_id]
        return mask

    def _update_state(self, token):
        if "{" in token:
            self.state = "in_object"
        elif "}" in token:
            self.state = "start"

    def _valid_tokens(self):
        """Return valid token IDs for current parser state."""
        if self.state == "start":
            return self.tokenizer.encode("{", add_special_tokens=False)
        return list(range(self.tokenizer.vocab_size))

class LengthStoppingCriteria(StoppingCriteria):
    """Stop after generating N tokens."""
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.max_tokens

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Generate JSON:", return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=LogitsProcessorList([JSONLogitsProcessor(tokenizer)]),
    stopping_criteria=StoppingCriteriaList([LengthStoppingCriteria(50)]),
    do_sample=True,
    temperature=0.7
)
```

### Constrained Beam Search

Beam search maintains $k$ hypotheses and at each step expands all $k \times |\mathcal{V}|$ candidates before pruning. This is effective for tasks where global coherence matters more than diversity (e.g., translation, summarization).

The `force_words_ids` parameter guarantees specific phrases appear in the output by biasing the beam toward sequences containing those tokens:

```python
phrase = "artificial intelligence"
force_ids = [tokenizer.encode(phrase, add_special_tokens=False)]

outputs = model.generate(
    **inputs,
    num_beams=4,
    force_words_ids=force_ids,
    max_new_tokens=30
)
```

### Speculative Decoding

Speculative decoding accelerates inference without changing the output distribution. A small **draft** model predicts $k$ tokens ahead; the large **target** model verifies all $k$ in a single forward pass. Tokens that match the target distribution are accepted; the first mismatch is corrected.

```python
assistant = AutoModelForCausalLM.from_pretrained("gpt2")
# In practice: use a distilled variant or smaller layer count

outputs = model.generate(
    **inputs,
    assistant_model=assistant,
    max_new_tokens=50
)
```

The acceptance rate $\alpha$ depends on how well the draft matches the target. The expected speedup is:

$$\mathbb{E}[\text{speedup}] = \frac{1}{1 - \alpha} \cdot \frac{c_{\text{target}}}{c_{\text{target}} + c_{\text{draft}}}$$

where $c_{\text{target}}$ and $c_{\text{draft}}$ are the per-token costs. In practice, $\alpha \approx 0.7-0.9$ yields 1.5-3x throughput on latency-sensitive workloads like chat serving.

✅ **Antipattern: Overly aggressive masking**
```python
# ❌ Masking too many tokens → dead end, no valid token has probability > 0
class AggressiveMask(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, :] = float("-inf")  # All tokens masked!
        return scores

# ✅ Always leave at least one valid path
class SafeMask(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, valid_ids] = scores[:, valid_ids]
        scores[:, invalid_ids] = float("-inf")
        # Fallback: if all masked, unmask at least EOS
        if torch.all(scores == float("-inf")):
            scores[:, tokenizer.eos_token_id] = 0
        return scores
```

> **Caso real: Outlines.dev** (a structured generation library built on `transformers`) compiles regex and JSON schemas into finite state machines implemented as `LogitsProcessor` objects. When generating API responses, the processor masks any token that would violate the schema, guaranteeing 100% syntactic validity without post-hoc parsing or retry loops.

⚠️ **Assistant model distribution mismatch**: If the draft model's distribution diverges significantly from the target, many drafts are rejected and speedup vanishes. The assistant should be a distilled or pruned variant of the same model family, ideally sharing the same tokenizer and embedding space.

💡 **Limit speculative batches to 3-5 tokens**: Larger draft sizes have diminishing acceptance rates and waste compute on rejected tokens. Monitor the `acceptance_rate` metric in production to tune draft length.

✅ **Antipattern: Generate then validate**
```python
# ❌ Generate freely, hope for valid output, retry on failure
text = generate(prompt)
while not is_valid_json(text):
    text = generate(prompt)

# ✅ Constrain during generation via LogitsProcessor
text = generate(prompt, logits_processor=[JSONProcessor()])
# Output is guaranteed valid on first attempt
```

## 🎯 Key Takeaways

- `model.generate()` is an autoregressive loop applying `LogitsProcessorList`, sampling, and `StoppingCriteria` at every step.
- Temperature, top_k, and top_p jointly shape the sampling distribution; they interact non-trivially.
- `repetition_penalty` is a simple but effective heuristic to break degenerate repetition loops.
- Custom `LogitsProcessor` subclasses enable hard constraints (grammars, schemas, banned tokens) at the token level.
- `force_words_ids` provides constrained beam search for guaranteed phrase inclusion.
- Speculative decoding (`assistant_model`) speeds up inference by verifying draft tokens without altering the output distribution.
- Structured generation (constrain-then-sample) is superior to generate-then-validate for production systems.
- Overly aggressive logit masking can create dead-end states; always leave a valid fallback path.

## References

- Hugging Face Generation Docs: [https://huggingface.co/docs/transformers/main_classes/text_generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- Holtzman et al., "The Curious Case of Neural Text Degeneration", ICLR 2020.
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
- Related Vault: [[03 - Trainer, TrainingArguments, and Distributed Training]]
- Related Vault: [[06 - Large Language Models/13 - vLLM Deep Dive|vLLM Deep Dive]]
- Related Vault: [[10 - FastAPI]]

## Código de compresión

```python
"""
Production generation script with custom processors and speculative decoding.
"""
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
    StoppingCriteriaList,
    StoppingCriteria
)
import torch

MODEL = "gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty=1.2, window=50):
        self.penalty = penalty
        self.window = window

    def __call__(self, input_ids, scores):
        for seq in input_ids:
            recent = set(seq[-self.window:].tolist())
            for idx in recent:
                scores[0, idx] /= self.penalty
        return scores

class LengthStopCriteria(StoppingCriteria):
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.max_tokens

inputs = tokenizer("In the year 2050, AI will", return_tensors="pt")

logits_proc = LogitsProcessorList([
    RepetitionPenaltyLogitsProcessor(penalty=1.15)
])
stop_criteria = StoppingCriteriaList([
    LengthStopCriteria(max_tokens=60)
])

outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_p=0.92,
    logits_processor=logits_proc,
    stopping_criteria=stop_criteria,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

force_phrase = [
    tokenizer.encode("transform the world", add_special_tokens=False)
]
outputs_beam = model.generate(
    **inputs,
    num_beams=5,
    force_words_ids=force_phrase,
    max_new_tokens=40,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs_beam[0], skip_special_tokens=True))
```
