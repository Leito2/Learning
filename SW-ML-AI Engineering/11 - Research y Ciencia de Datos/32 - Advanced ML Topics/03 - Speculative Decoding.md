# ⚡ Speculative Decoding

## Introduction

Speculative decoding accelerates LLM inference by 2-3x without any quality degradation — it's a pure performance optimization. The key insight: instead of generating one token at a time with the large model, use a small, fast "draft" model to propose multiple tokens, then have the large model verify them in parallel. If the draft tokens match what the large model would have generated, you accept them — getting multiple tokens for the cost of one forward pass.

For production LLM serving where latency is the primary cost driver, speculative decoding is the most impactful single optimization you can deploy.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│              SPECULATIVE DECODING (K=4)                       │
│                                                              │
│  Step 1: Draft Model proposes K tokens:                      │
│  Draft (fast, 0.1B params): "I" "love" "machine" "learning"  │
│                                                              │
│  Step 2: Target Model verifies in ONE forward pass:          │
│  Target (slow, 70B params): forward(["I","love","machine","learning"]) │
│  → Output: logits for each position                          │
│                                                              │
│  Step 3: Accept/Reject:                                      │
│  Position 1: Draft prob "I" = 0.95, Target prob "I" = 0.92 → ACCEPT  │
│  Position 2: Draft prob "love" = 0.85, Target prob "love" = 0.88 → ACCEPT │
│  Position 3: Draft prob "machine" = 0.7, Target prob "learning" = 0.9 → REJECT │
│                                                              │
│  Step 4: Re-sample from target at rejection point:           │
│  Accept: "I", "love"                                         │
│  Reject: "machine" → Re-sample from target: "learning"       │
│  Result: 3 tokens for 1 forward pass (3x speedup)            │
└──────────────────────────────────────────────────────────────┘
```

### The Mathematical Guarantee

The acceptance probability for each draft token is:

$$
P(\text{accept } x_i) = \min\left(1, \frac{p_{\text{target}}(x_i)}{q_{\text{draft}}(x_i)}\right)
$$

This ensures the output distribution is IDENTICAL to what the target model would produce without speculation — zero quality degradation, guaranteed.

---

## Key Design Decisions

| Decision | Options | Impact |
|---|---|---|
| **Draft model** | Small version of same architecture (Llama-3-70B → Llama-3-1B), or distilled model | Higher draft acceptance = higher speedup |
| **Lookahead K** | 3-8 tokens | Larger K → higher potential speedup, but more wasted computation on rejections |
| **Tree attention** | Verify multiple draft paths simultaneously | Increases acceptance rate without more forward passes |
| **Medusa heads** | Add auxiliary prediction heads instead of separate draft model | No draft model needed — heads added to target model |

---

## Production Impact

| Scenario | Without Speculative Decoding | With Speculative Decoding |
|---|---|---|
| **Llama-3-70B** | 15 tokens/sec on A100 | 35 tokens/sec (2.3x) |
| **Mixtral 8x7B** | 25 tokens/sec on A100 | 50 tokens/sec (2x) |
| **Llama-3-8B (Medusa)** | 60 tokens/sec | 120 tokens/sec (2x) |

---

## References

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML, 2023)
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework" (2024)
- [Hugging Face: Assisted Generation](https://huggingface.co/blog/assisted-generation)
