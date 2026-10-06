# 🔧 04 - DIY — Classification Head on a Small Qwen

The fastest way to stop treating decision models as magic is to build one. Take a small open LLM, remove the part that predicts the next word from a 150,000-token vocabulary, bolt on a layer that outputs a handful of scores, and train only that layer. What you get decides in one forward pass, returns real probabilities, and fits on a 4 GB GPU — and building it teaches you exactly what Laya, Kev, and Jev are doing differently.

## 🎯 Learning Objectives
- Explain the anatomy of a causal LM: **embeddings → transformer backbone → LM head**
- Replace the LM head with a **classification head** and choose the right **pooling** for a decoder
- Choose between a **fixed-label head** and an **option-scoring head** that accepts arbitrary options
- Train with a **frozen backbone** (head only) or **LoRA**, and estimate memory for a 4 GB GPU
- Quantize for CPU inference and **calibrate** the resulting probabilities
- Understand why production decision models need far more than this tutorial (data, calibration training, evaluation)

## Introduction

A decoder-only language model such as Qwen is a stack of transformer blocks (the **backbone**) followed by a linear **LM head** that maps the final hidden state $\mathbf{h} \in \mathbb{R}^d$ to logits over the vocabulary $V$. Generation repeatedly applies the head to the last position and samples a token. The backbone is where language understanding lives; the head is just a projection.

A decision model needs scores over a **small, task-defined** set of answers. So the surgery is simple: keep the backbone, discard the LM head, and add a small head that maps $\mathbf{h}$ to the answers. This is the technique popularized in a Towards Data Science walkthrough (*How to Make Your Own JEV Model from an Open LLM*, 2026), which froze a Qwen2.5-Coder-1.5B backbone, trained a 2-way linear head on a few dozen examples, and quantized the backbone from 6.17 GB (fp32) to 0.93 GB (int8) — and, in spirit, what decoder reproductions like **Kev** do at scale with a trained, calibrated head on Qwen3.5.

This note builds the same idea with **Qwen3-0.6B** (to leave room on a 4 GB GPU), shows a version that scores **arbitrary options** instead of a fixed label set, and is honest about the distance between a tutorial head and a production decision model.

---

## 1. The Problem and Why This Solution Exists

### Why not just prompt the LLM?

Prompting a 0.6B model to answer "disputes/cards/security/general" works poorly and slowly: small models follow instructions loosely, generate extra tokens, and give no usable probabilities ([[01 - How Decision Models Work|how decision models work]]). Reading the next-token log-probabilities of each option (a logit reader) is better, but uncalibrated and sensitive to wording. Training a head uses labeled data to turn the backbone's representations into exactly the scores you need.

### Why not a classic encoder classifier?

You can — a fine-tuned ModernBERT classifier is a strong baseline (and is what Laya builds on). The DIY decoder head is interesting because it reuses a model family you may already run (Qwen), inherits a long context and multilingual pretraining, and mirrors how decoder-based decision models work.

---

## 2. Conceptual Deep Dive

### 2.1 Anatomy and the swap

$$
\underbrace{\mathbf{x}_{1:n}}_{\text{tokens}} \xrightarrow{\text{embed}} \xrightarrow{\text{backbone (L blocks)}} \mathbf{H} \in \mathbb{R}^{n \times d}
\quad\Rightarrow\quad
\begin{cases}
\text{LM head: } \mathbf{W}_{V} \mathbf{h}_n \in \mathbb{R}^{|V|} & \text{(generation)}\\[2pt]
\text{decision head: } \mathbf{W}_{K} \mathbf{h}_n + \mathbf{b} \in \mathbb{R}^{K} & \text{(K answers)}
\end{cases}
$$

For Qwen3-0.6B, $d$ is on the order of 1,000 and $|V| \approx 150{,}000$: the LM head alone is ~150M parameters. A 4-way decision head is ~4,000 parameters.

### 2.2 Pooling in a causal model

In a causal (decoder-only) model, position $i$ only attends to positions $\le i$. The **last non-padding token** is therefore the only position that has seen the whole input — that is the representation to pool. Mean pooling over all positions mixes in states that haven't seen the end of the text. With batched inputs, either pad on the **left** (so the last position is always real) or gather the hidden state at each sequence's last real index from the attention mask.

### 2.3 Fixed-label head vs option-scoring head

| Design | Input | Head | Generalizes to new options? |
|---|---|---|---|
| **Fixed-label** | `state + question` | $\mathbf{W}_K \mathbf{h}$ → K logits | ❌ retrain for new labels |
| **Option-scoring** | `state + question + option_k`, one sequence per option | $\mathbf{w}^\top \mathbf{h}^{(k)}$ → 1 logit per option; softmax over options | ✅ options are inputs |

Option scoring is how decision models answer questions they were not specifically trained on: the model learns "how well does this option fit this state and question", and options are just text. The cost is $K$ forward passes per question — mitigated by batching and, in production systems, by **sharing the prefix** (state + question) across options via KV caching.

$$
p_k = \frac{\exp(\mathbf{w}^\top \mathbf{h}^{(k)} / T)}{\sum_{j=1}^{K} \exp(\mathbf{w}^\top \mathbf{h}^{(j)} / T)}
$$

### 2.4 Training regimes and memory on a 4 GB GPU

| Regime | Trainable params | Memory (fp16 weights, rough) | Quality |
|---|---|---|---|
| **Frozen backbone, head only** | ~$10^3$–$10^4$ | ~1.2 GB weights + activations (no backbone grads) | Good when backbone features already separate the classes |
| **LoRA on attention projections + head** | ~1–5M | weights + small optimizer state + activations (gradient checkpointing helps) | Better adaptation; fits 4 GB for 0.6B with short sequences |
| Full fine-tune | 600M | weights + grads + Adam states ≈ 16 bytes/param ≈ 10 GB | ❌ on 4 GB |

The AdamW rule of thumb — about 16 bytes per trainable parameter for fp32 master weights, gradients, and two moment estimates — is why full fine-tuning is out and **head-only or LoRA** are the realistic options locally.

### 2.5 From trained head to usable decision model

A trained head gives probabilities, but not necessarily **calibrated** ones. Fit a temperature on a validation set (note 01) and report ECE. Then compress for CPU: dynamic int8 quantization of the backbone's linear layers (as in the TDS walkthrough), or export to ONNX and quantize with ONNX Runtime.

```mermaid
graph LR
    Q[Qwen3-0.6B<br/>backbone] -->|last-token hidden| H[Decision head<br/>Linear d→K or d→1]
    D[Labeled decisions] --> TR[Train head (± LoRA)<br/>cross-entropy]
    TR --> H
    H --> CAL[Fit temperature<br/>on validation]
    CAL --> QZ[int8 quantize / ONNX]
    QZ --> API[decide(state, questions)]
```

---

## 3. Production Reality

### What the tutorial skips

| Production requirement | Tutorial | Real decision model |
|---|---|---|
| Training data | Dozens of examples, one task | Hundreds of thousands of typed questions across many tasks (Laya: ~30k for fine-tuning; Jev: undisclosed, large) |
| Generalization to new questions | One fixed question | Option scoring + instruction diversity |
| Calibration | Raw softmax | Proper-scoring-rule training (RLCD-style) + temperature |
| Evaluation | A handful of test cases | Held-out tasks, ECE, robustness (negations, OOD languages) |
| Efficiency | One sequence per option | Prefix sharing, batching, quantization |

That gap is why P2 **uses Laya** for System 1 and treats the DIY head as a learning exercise and an optional baseline in its harness — not as the production model.

### When DIY is the right call

A fixed set of labels that will not change, plenty of labeled data, and a model family already in your stack: a head on a small backbone (or a fine-tuned encoder classifier) can match a general decision model on that one task, with full control and no new dependency.

Caso real: the TDS walkthrough applied the technique to a narrow code-review check — whether a function's name matches its body — trained on a few dozen labeled functions, and used the quantized model as a fast pre-commit gate that flagged 10 of 27 test functions with high confidence. Narrow, fixed-label tasks with clear signal are where DIY heads shine.

Caso real: Kev's approach — Qwen3.5 backbones (0.8B/4B/9B) trained to answer the Jev request format — is the "industrial" version of the same surgery, adding the data scale, multi-task training, and calibration work that turn a head into a general decision model.

---

## 4. Code in Practice

### Head swap on a real checkpoint (frozen backbone)

```python
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

NAME = "Qwen/Qwen3-0.6B"                     # small enough for a 4 GB GPU in fp16
tok = AutoTokenizer.from_pretrained(NAME)
tok.padding_side = "left"                    # last position is always a real token
backbone = AutoModel.from_pretrained(NAME, torch_dtype=torch.float16).to("cuda")   # no LM head loaded
for p in backbone.parameters():
    p.requires_grad = False                  # freeze: only the head learns

ROUTES = ["disputes", "cards", "security", "general"]

class DecisionHead(nn.Module):
    def __init__(self, hidden: int, k: int):
        super().__init__()
        self.score = nn.Linear(hidden, k)
    def forward(self, last_hidden):          # [batch, hidden]
        return self.score(last_hidden.float())

head = DecisionHead(backbone.config.hidden_size, len(ROUTES)).to("cuda")

def encode(texts: list[str]):
    prompts = [f"Message: {t}\nQuestion: which team should handle it?\nAnswer:" for t in texts]
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
    with torch.no_grad():
        out = backbone(**batch).last_hidden_state
    return out[:, -1, :]                     # left padding → last position = last real token

opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
def train_step(texts, labels):
    logits = head(encode(texts))
    loss = loss_fn(logits, torch.tensor(labels, device="cuda"))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

⚠️ **Warning:** check the exact model ID, chat template behavior, and hidden size for the checkpoint you download; small Qwen releases are frequent. Truncate inputs (`max_length`) — memory for activations grows with sequence length even when the backbone is frozen.

### ❌/✅ Pooling in a decoder

```python
# ❌ Right padding + taking position -1: for short sequences, position -1 is a PAD token's state
tok.padding_side = "right"; vec = out[:, -1, :]

# ❌ Mean pooling in a causal model: early positions never saw the end of the message
vec = out.mean(dim=1)

# ✅ Left padding (or gather at the last real index from the attention mask)
tok.padding_side = "left"; vec = out[:, -1, :]
```

### 📦 Compression code: the surgery on a tiny Qwen3 (no download needed)

```python
# 📦 Compression code: replace the LM head of a (tiny, randomly initialized) Qwen3 with a decision head
# Covers: backbone vs LM head size, last-token pooling, frozen backbone, head-only training, temperature
# pip install torch transformers   (CPU is fine)
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model

torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
backbone = Qwen3Model(cfg).eval()
for p in backbone.parameters():
    p.requires_grad = False
lm_head_params = cfg.hidden_size * cfg.vocab_size
head = nn.Linear(cfg.hidden_size, 2)
print(f"LM head would have {lm_head_params:,} params; decision head has {sum(p.numel() for p in head.parameters()):,}")

def make_batch(n=64, length=12):              # label 1 iff "keyword" token 7 appears (toy task)
    x = torch.randint(10, cfg.vocab_size, (n, length))
    y = torch.randint(0, 2, (n,))
    pos = torch.randint(0, length, (n,))
    x[y == 1, pos[y == 1]] = 7
    return x, y

def features(x):
    with torch.no_grad():
        return backbone(input_ids=x).last_hidden_state[:, -1, :]   # causal model → last token saw everything

opt, loss_fn = torch.optim.AdamW(head.parameters(), lr=1e-2), nn.CrossEntropyLoss()
for step in range(300):
    x, y = make_batch()
    loss = loss_fn(head(features(x)), y)
    opt.zero_grad(); loss.backward(); opt.step()

x, y = make_batch(512)
with torch.no_grad():
    logits = head(features(x))
acc = (logits.argmax(1) == y).float().mean().item()
T = min((t / 10 for t in range(5, 40)), key=lambda t: loss_fn(logits / t, y).item())
print(f"head-only accuracy on toy task = {acc:.2f} | fitted temperature T = {T:.1f}")
# ¡Sorpresa! A RANDOM frozen backbone gives a linear head only weak signal (~0.67 vs 0.50 chance): the head
# can only read what the backbone already encodes — which is why the surgery needs a PRETRAINED backbone.
```

---

## 🎯 Key Takeaways
- A causal LM = embeddings → **backbone** → **LM head**; a decision model keeps the backbone and replaces the head.
- Pool the **last real token** in decoders (left padding or mask-based gather), never a pad or a mean.
- **Fixed-label heads** are simple; **option-scoring heads** accept arbitrary options as text (one sequence per option, prefix-shareable).
- On 4 GB: train **head-only or LoRA** on a small backbone (Qwen3-0.6B); full fine-tuning needs ~16 bytes/param.
- Calibrate with temperature and quantize (int8/ONNX) for CPU.
- The tutorial head is a learning tool; production decision models add **data scale, calibration training, and robust evaluation** — which is why P2 uses Laya.

## References
- *How to Make Your Own JEV Model from an Open LLM* (Towards Data Science, 2026)
- Qwen3 technical report and model cards (Qwen team, 2025)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022)
- Hugging Face Transformers — `AutoModel`, `AutoModelForSequenceClassification` (which implements last-token pooling for decoders)
- [[03 - Laya in Practice|Previous: Laya in Practice]] · [[05 - System 1 - System 2 Routing Patterns|Next: System 1 / System 2 Routing]]
