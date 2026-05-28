# 🏷️ Speculative Decoding 2.0 — From Draft Models to Eagle and Multi-Token Prediction

## 🎯 Learning Objectives
- Explain why HBM bandwidth, not compute, is the bottleneck in autoregressive LLM decoding
- Compare vanilla speculative decoding (draft model), Eagle (feature-level speculation), and Multi-Token Prediction (MTP)
- Derive the acceptance probability and expected tokens-per-iteration for draft-verify pipelines
- Analyze the latency-throughput trade-offs between draft model methods and MTP training-based methods
- Deploy speculative decoding in production with TensorRT-LLM and vLLM backends

## Introduction

**Autoregressive decoding is bottlenecked by memory bandwidth — reading 140GB of weights from HBM for every single token you generate.** This is not a compute problem. It's a data movement problem. A 70B parameter model at FP16 occupies ~140GB. At 3.35 TB/s HBM bandwidth (H100), the theoretical maximum throughput is 3.35 TB/s ÷ 140 GB = ~24 forward passes per second — meaning 24 tokens/second absolute maximum, regardless of how many FLOPS your GPU can do. In practice, overhead brings this down to 15-20 tok/s. Speculative decoding attacks this bottleneck by generating K > 1 tokens per forward pass, amortizing the weight-read cost across multiple output tokens.

The name "speculative decoding" comes from the CPU architecture concept of *branch prediction* — the processor speculatively executes instructions down a predicted branch path, and rolls back if the prediction was wrong. Similarly, a draft model speculatively generates multiple tokens, and the target model verifies them. If a token is rejected, only the rejected token needs regeneration; everything before it stays valid. The seminal papers are Leviathan et al. (2023) from Google and Chen et al. (2023) for the "Fast Inference from Transformers via Speculative Decoding" formulation.

Before speculative decoding, the production story for LLM inference was grim: every token cost one full model forward pass. At 20 tok/s, generating a 500-token response takes 25 seconds. Adding more GPUs (tensor parallelism) helps with model loading but does NOT increase tokens-per-second for a single request — it only increases throughput across requests. For latency-sensitive applications like chatbots, coding assistants, and real-time translation, you were fundamentally capped by physics: the speed of HBM. This matters for [[06/09 - Sistemas de LLMs en Producción]] because latency directly drives user satisfaction metrics, and speculative decoding is one of the few techniques that directly reduces per-request wall-clock time.

---

## 1. The HBM Bandwidth Wall

To understand speculative decoding, you must first internalize why autoregressive decoding is memory-bandwidth-bound. Consider a single forward pass of a transformer decoder:

**Compute (FLOPs):** For a single token (batch size = 1, sequence length = 1), the FLOPs are:
$$F_{\text{decode}} = L \times (4d^2 + 2 \cdot n_{\text{kv}} \cdot d) \approx 4 L d^2$$

For Llama-70B ($L=80, d=8192$): $F_{\text{decode}} \approx 4 \times 80 \times 8192^2 \approx 2.15 \times 10^{10}$ FLOPs = 21.5 GFLOPs.

**Memory movement (bytes):** To compute one forward pass, you must read every weight from HBM:
$$M_{\text{decode}} = 2N \text{ bytes} = 2 \times 70 \times 10^9 = 140 \text{ GB}$$ (factor of 2 for FP16)

**Arithmetic intensity:** The ratio of FLOPs to bytes:
$$I = \frac{F_{\text{decode}}}{M_{\text{decode}}} = \frac{2.15 \times 10^{10}}{140 \times 10^9} \approx 0.15 \frac{\text{FLOP}}{\text{byte}}$$

An H100 has an ops:byte ratio of ~989 FP16 TFLOPS / 3.35 TB/s = 295 FLOPs/byte — meaning the GPU is capable of 295 FLOPs for every byte it reads, but this workload only uses 0.15. **The GPU compute units are 99.95% idle.** This is the memory bandwidth wall.

**The key insight:** If you process K tokens in one forward pass (by concatenating K tokens as a batch), the weight reads stay the same (you read weights once), and the FLOPs scale sub-linearly (matrix multiplication is more efficient with larger batches). The effective arithmetic intensity becomes:

$$I_K = \frac{K \times F_{\text{decode}}}{M_{\text{decode}}} = K \times 0.15 \frac{\text{FLOP}}{\text{byte}}$$

At $K=5$: $I_5 = 0.75$ FLOPs/byte. Still far below 295, but 5× better. At $K=100$: $I_{100} = 15$, which starts to meaningfully utilize the GPU. **Speculative decoding achieves K > 1 by having the draft model propose K tokens, then the target model verifies all K in a single batched forward pass.**

![HBM bandwidth hierarchy](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Computer_memory_hierarchy.svg/640px-Computer_memory_hierarchy.svg.png)

## 2. Vanilla Speculative Decoding

The classic formulation uses two models: a small *draft model* $M_d$ (e.g., 1B params) and a large *target model* $M_t$ (e.g., 70B params).

**Algorithm per step:**
1. **Draft:** $M_d$ autoregressively generates K tokens: $x_1, x_2, ..., x_K$. This is fast because $M_d$ is small.
2. **Verify:** $M_t$ processes all K drafted tokens in a SINGLE forward pass. For each position $i \in \{1, ..., K\}$, $M_t$ computes the probability distribution $p_t(x_i | x_{<i})$ over the vocabulary.
3. **Accept/Reject:** For each draft token $x_i$, accept it with probability:

$$\alpha_i = \min\left(1, \frac{p_t(x_i | x_{<i})}{p_d(x_i | x_{<i})}\right)$$

4. **If rejected at position j:** Resample from the adjusted distribution:

$$p_{\text{resample}}(x) = \text{norm}\left(\max(0, p_t(x) - p_d(x))\right)$$

Then discard tokens after position j and start a new draft sequence from the accepted prefix.

**Why this works (proof sketch):** The acceptance probability ensures that the distribution of accepted tokens matches EXACTLY the distribution that $M_t$ would produce if run autoregressively. This is a rejection sampling algorithm. The proof relies on the fact that at each position, the probability of accepting token $x$ given that we proposed it is:
$$P(\text{accept } x | \text{proposed } x) = \min(1, p_t(x)/p_d(x))$$

Multiplying by the proposal probability: $P(\text{propose } x \land \text{accept } x) = p_d(x) \cdot \min(1, p_t(x)/p_d(x)) = \min(p_d(x), p_t(x))$. The corrected resampling adds the difference, ensuring the total probability of outputting any $x$ equals $p_t(x)$.

**Expected acceptance length:** The expected number of accepted tokens per draft cycle is:
$$\mathbb{E}[k] = \sum_{i=1}^{K} \prod_{j=1}^{i-1} \alpha_j$$ If the draft and target disagree significantly, $\alpha$ values are low, $k$ stays small, and you get minimal speedup. If they agree closely ($p_d \approx p_t$), $\alpha \approx 1$ for all positions, and $k \approx K$, giving a speedup of roughly:

$$S = \frac{K \cdot T_{\text{target}}}{T_{\text{draft}} + T_{\text{target}}}$$

where $T_{\text{draft}} = K \times t_{\text{draft}}$ (time for K draft forward passes) and $T_{\text{target}}$ is the single target model forward pass.

**Practical limitations of vanilla speculative decoding:**
- Need to load TWO models in GPU memory (draft + target), increasing total VRAM usage
- Draft model may have a different tokenizer — tokenizer mismatch causes systematic errors at boundaries
- The draft autoregressive forward passes themselves are memory-bound (just for a smaller model)
- Draft model quality directly caps speedup: a weak draft (low alignment with target) may achieve $k < 2$ average, yielding negligible speedup

💡 A draft model doesn't need to be the same architecture as the target. You can use a transformer for the target and an RNN (Mamba/SSM) for the draft — SSMs are faster per token but weaker at long-range dependencies, making them ideal draft candidates for casual conversation where local coherence dominates.

## 3. Eagle: Feature-Level Speculation

**Etymology:** "Eagle" stands for "Extrapolation Algorithm for Greater Language-model Efficiency." The name evokes speed and precision — the eagle strikes faster than its prey (the vanilla draft approach).

Eagle eliminates the separate draft model entirely. Instead of running a second model to propose tokens, **Eagle uses the target model's own intermediate hidden states to predict future tokens.** The key insight: at layer $l$ of the target model, the hidden state $\mathbf{h}^{(l)}_t$ at position $t$ already contains rich contextual information. A lightweight prediction head can map this to the output token — and with a slight time-shift, predict tokens *ahead* of the current position.

**Architecture:**

1. During the target model's forward pass at step $t$, extract the hidden state $\mathbf{h}^{(l)}_t$ from a middle layer $l$
2. Feed $\mathbf{h}^{(l)}_t$ through a small prediction network (typically 1-2 transformer layers + LM head) to predict the feature representation for position $t+1$
3. From the predicted feature, produce a token probability distribution $p_d(x_{t+1})$
4. Repeat to unroll K steps: predict feature at $t+1$, then feed that into the same prediction network to predict $t+2$, etc.

Formally, the prediction module $f_\theta$ maps:
$$\hat{\mathbf{h}}^{(l)}_{t+k} = f_\theta(\hat{\mathbf{h}}^{(l)}_{t+k-1}), \quad p_d(x_{t+k}) = \text{softmax}(\mathbf{W}_{\text{lm}} \cdot \hat{\mathbf{h}}^{(l)}_{t+k})$$

The verification step uses the SAME target model: compute the true distribution $p_t(x_{t+k})$ from a forward pass through all layers, starting from the predicted features or a known prefix.

**Advantages over vanilla speculative decoding:**
- **No second model in memory.** The prediction module is tiny (~50-100M parameters), a fraction of the target model
- **No tokenizer mismatch.** Uses the target model's own vocabulary and embeddings directly
- **Feature-level information** is richer than token-level: a hidden state at layer $l$ encodes syntactic structure and local semantics that a token ID alone cannot capture
- **The prediction module can be trained end-to-end** to minimize the KL divergence between draft and target distributions, maximizing the acceptance rate $\alpha$

**Memory overhead of Eagle:** The prediction module adds ~100M parameters. For a 70B model, this is a 0.14% parameter increase — negligible in practice.

**Eagle-2:** A follow-up work that uses a tree-structured speculation strategy. Instead of predicting a single token at each future step, predict a small tree of candidate tokens. The target model verifies the tree in one batched forward pass using tree attention masking. This increases expected acceptance length $k$ by 15-20% over linear speculation.

⚠️ Eagle requires modifying the model architecture (adding prediction heads) and fine-tuning. This is not a drop-in optimization like vanilla speculative decoding. You need access to model training infrastructure and the ability to run forward-backward passes on the target model.

## 4. Multi-Token Prediction (MTP)

**Etymology:** "Multi-Token Prediction" is descriptive — the model is trained to predict multiple future tokens simultaneously rather than just the next single token.

MTP takes a different approach than draft-model-based methods: instead of using the inference-time dynamic of draft-then-verify, **MTP bakes multi-token generation directly into the model's training objective.**

**Architecture:** The model has $K$ auxiliary prediction heads. The main transformer trunk processes the input sequence normally. At the final layer, rather than a single LM head, there are $K$ independent heads:

$$\text{Head}_1: p(x_{t+1} | x_{\leq t}) \quad \text{(standard next-token)}$$
$$\text{Head}_2: p(x_{t+2} | x_{\leq t}) \quad \text{(2-tokens-ahead)}$$
$$\text{Head}_3: p(x_{t+3} | x_{\leq t}) \quad \text{(3-tokens-ahead)}$$
$$...$$
$$\text{Head}_K: p(x_{t+K} | x_{\leq t}) \quad \text{(K-tokens-ahead)}$$

**Training:** The training objective combines losses across all heads:
$$\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{K} \lambda_k \cdot \mathcal{L}_{\text{CE}}\left(\text{Head}_k(x_{\leq t}), x_{t+k}\right)$$

where $\lambda_k$ are decay weights (typically $\lambda_k = \gamma^{k-1}$ with $\gamma \in [0.5, 0.9]$ since predicting further-ahead tokens is harder and contributes less useful signal).

**At inference time:** Feed the input through the trunk ONCE. All K heads generate their predictions in parallel. Accept tokens sequentially using the same rejection sampling as speculative decoding ($\alpha = \min(1, p_{\text{head1}}(x)/p_{\text{headK}}(x))$), or greedily accept the top prediction from each head with a verification pass.

**Key difference from draft model approaches:** MTP uses the SAME hidden representation for all predictions. The trunk computes the representation once; all heads operate on that same representation. This means:
- **No extra forward passes** at inference time (unlike the draft model's K autoregressive passes)
- **No second model** in memory
- The representation quality is that of the FULL target model, not a small draft — leading to higher acceptance rates
- Training overhead: K output heads add $K \times d \times V$ parameters (embedding dimension × vocab size), which for $K=4$, $d=8192$, $V=128K$ is approximately $4 \times 8192 \times 128000 \approx 4.2$B parameters — substantial but acceptable for very large models

**Meta's MTP implementation for code:** Meta trained a 7B model with 4 prediction heads specifically for code generation. The result: 3× throughput for code completion with <0.5% accuracy degradation on HumanEval. Code is an ideal MTP domain because tokens are highly predictable from context — once you see `def sort_list(arr):`, the next few tokens follow a narrow distribution.

**Comparison table:**

| Technique | Memory Overhead | Speedup | Accuracy Loss | Requires Training | Production Maturity |
|-----------|----------------|---------|---------------|-------------------|---------------------|
| Vanilla SD | 2nd model (~1-3B) | 2-3× | 0% (exact match) | No | High (vLLM, TRT-LLM) |
| Eagle | ~100M params | 2.5-3.5× | <0.1% | Yes | Medium |
| MTP | ~4.2B params (K=4) | 2-4× | <0.5% | Yes (full pretrain) | Low-Medium |
| Medusa | ~100M params | 2-3× | <0.2% | Yes (fine-tune) | Medium |

## 5. Production Reality

**Caso real: NVIDIA TensorRT-LLM** — NVIDIA's production inference stack (TensorRT-LLM) ships with first-class speculative decoding support. On H100 GPUs with Llama-70B using a 3B draft model, they achieve 2.5× speedup across a range of workloads (summarization, chat, code generation). The key engineering contributions: (1) draft and target models share the same KV cache block manager to minimize memory fragmentation, (2) the acceptance/rejection computation is fused into a single CUDA kernel to minimize kernel launch overhead, (3) dynamic draft length — the draft generates until it hits an end-of-sequence token or reaches a configurable max length, adapting to each query.

**Caso real: Meta's Multi-Token Prediction for Code Generation** — Meta trained a 7B code model with 4 prediction heads and achieved 3× throughput for code completion in production. The key insight: code has very low per-token entropy compared to natural language. Once you've typed `import numpy as`, `np` is almost certain — making MTP's parallel prediction heads correct >90% of the time. This domain-specific predictability makes speculative decoding disproportionately effective for code assistants (GitHub Copilot, Cursor, etc.).

**Caso real: DeepSeek-V2/3 MTP deployment** — DeepSeek-V3 uses MTP as part of its inference optimization alongside [[07 - KV Cache Compression - Multi-Head Latent Attention]]. The combination of MLA (reducing KV cache memory) and MTP (increasing tokens-per-forward-pass) creates a compound acceleration: less time per token × more tokens per iteration ≈ 4-5× overall throughput improvement over naive Llama-70B inference at equivalent model quality.

---

## ❌/✅ Comparison: Draft Model vs Eagle

```python
# ❌ Vanilla speculative decoding: two models, tokenizer mismatch risk
def vanilla_speculative_decode(target, draft, prompt, K=5):
    draft_tokens = draft.generate(prompt, max_new_tokens=K)  # May use different tokenizer!
    target_logits = target.forward(draft_tokens)  # Verify all K in parallel
    # ⚠️ If tokenizers differ, draft tokens map to wrong target tokens
    return verify_and_correct(draft_tokens, target_logits)

# ✅ Eagle: single model, feature-level speculation, no tokenizer mismatch
def eagle_speculative_decode(model, eagle_head, prompt, K=5):
    h = model.encode(prompt)           # Forward pass through trunk
    draft_feats = [h[-1]]              # Start from last hidden state
    for _ in range(K):
        next_feat = eagle_head(draft_feats[-1])
        draft_feats.append(next_feat)  # Unroll in feature space
    draft_tokens = [model.lm_head(f) for f in draft_feats]
    # 💡 Eagle reuses the same tokenizer as target — no mismatch
    # ¡Sorpresa! Feature-space prediction is more stable than token-space
    target_logits = model.verify_features(draft_feats)
    return verify_and_correct(draft_tokens, target_logits)
```

## 6. Code in Practice — Acceptance Testing with Rejection Sampling

```python
import torch
import torch.nn.functional as F
import numpy as np

def speculative_decode_step(target_model, draft_model, tokenizer,
                             prompt_ids, K=5, device="cuda"):
    """One step of speculative decoding: draft K tokens, verify, accept/reject."""
    # Step 1: Draft model generates K tokens autoregressively
    draft_ids = prompt_ids.clone()
    for _ in range(K):
        with torch.no_grad():
            draft_logits = draft_model(draft_ids).logits[:, -1, :]  # [1, V]
        next_token = torch.multinomial(F.softmax(draft_logits / 0.6, dim=-1), 1)
        draft_ids = torch.cat([draft_ids, next_token], dim=1)
    draft_tokens = draft_ids[0, -K:]  # The K new tokens

    # Step 2: Target model verifies all K in ONE forward pass
    with torch.no_grad():
        target_logits = target_model(draft_ids).logits
        # target_logits[:, -(K+1):-1, :] → distributions for each drafted position

    # Step 3: Acceptance testing with rejection sampling
    accepted_count = 0
    for i in range(K):
        pos = prompt_ids.shape[1] + i  # position in target sequence
        draft_token = draft_tokens[i].item()

        # Probability under draft model
        p_draft = F.softmax(draft_logits[i], dim=-1)[0, draft_token].item()

        # Probability under target model
        # ⚠️ The target distribution is at position [pos] in the logits
        target_dist = F.softmax(
            target_logits[0, prompt_ids.shape[1] + i - 1, :], dim=-1
        )
        p_target = target_dist[draft_token].item()

        # Acceptance probability
        alpha = min(1.0, p_target / (p_draft + 1e-10))  # ¡Sorpresa! epsilon prevents NaN

        if np.random.random() < alpha:
            accepted_count += 1
        else:
            # Rejected: resample from adjusted distribution
            adjusted = torch.clamp(target_dist - F.softmax(draft_logits[i], dim=-1)[0], min=0)
            resampled_token = torch.multinomial(adjusted / adjusted.sum(), 1).item()
            # 💡 Resampling guarantees output distribution matches target model exactly
            draft_tokens[i] = resampled_token
            break  # All subsequent tokens are stale

    new_ids = torch.cat([prompt_ids, draft_tokens[:accepted_count+1].unsqueeze(0)], dim=1)
    return new_ids, accepted_count + 1  # +1 for the resampled/broken token
```

## 🎯 Key Takeaways
- **Autoregressive decoding is bottlenecked by HBM-to-SRAM bandwidth, not compute** — a 70B model achieves 0.15 FLOPs/byte on hardware capable of 295 FLOPs/byte, leaving GPU compute 99.95% idle
- **Speculative decoding amortizes weight reads by generating K tokens per forward pass** — draft proposes, target verifies in parallel, acceptance/rejection preserves exact output distribution
- **Eagle eliminates the second model by predicting future hidden states from the target model's own intermediate representations** — feature-level speculation is richer than token-level and adds only ~100M parameters
- **Multi-Token Prediction trains auxiliary heads to predict multiple future tokens from a single hidden representation** — no extra forward passes needed, but requires modified training with K output heads
- **The acceptance probability $\alpha = \min(1, p_t/p_d)$ mathematically guarantees the output distribution matches the target model exactly** — speculative decoding is lossless in the information-theoretic sense
- **Domain matters: code generation benefits disproportionately from speculative decoding because tokens have low entropy** — MTP achieves 3× speedup on code vs 2× on natural language

## 🔬 Código de Compresión

```python
"""Mock speculative decoding: draft K tokens, verify in parallel, accept/reject."""
import torch, numpy as np
from torch.nn.functional import softmax

def mock_speculative_decode(prompt_len, K=5, vocab=32000):
    """Simulates acceptance testing logic. Real impl needs model forward passes."""
    # Mock draft/target distributions (real impl: actual model logits)
    torch.manual_seed(42)
    draft_probs = softmax(torch.randn(K, vocab), dim=-1)
    target_probs = softmax(torch.randn(K, vocab), dim=-1)

    draft_tokens = torch.multinomial(draft_probs, 1).squeeze()  # [K]
    accepted = 0

    for i in range(K):
        p_d = draft_probs[i, draft_tokens[i]].item()
        p_t = target_probs[i, draft_tokens[i]].item()
        alpha = min(1.0, p_t / (p_d + 1e-8))  # ¡Sorpresa! Small epsilon blocks NaN

        if np.random.random() < alpha:
            accepted += 1
        else:
            # Resample: max(0, target - draft), then normalize
            adj = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            resample = torch.multinomial(adj / adj.sum(), 1).item()
            draft_tokens[i] = resample
            break  # ⚠️ Tokens beyond rejection point are discarded

    speedup = (accepted + 1) / 1  # tokens generated per target forward pass
    return accepted + 1, speedup

tokens, speedup = mock_speculative_decode(100, K=5)
print(f"Accepted {tokens} tokens in 1 target pass → {speedup:.1f}x speedup")
```

## References
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." *arXiv:2211.17192*
- Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." *arXiv:2302.01318*
- Li, Y., et al. (2024). "Eagle: Speculative Decoding Requires Rethinking Feature Uncertainty." *arXiv:2401.15077*
- Gloeckle, F., et al. (2024). "Better & Faster Large Language Models via Multi-token Prediction." *arXiv:2404.19737*
- Xia, H., et al. (2024). "Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation." NVIDIA Technical Blog
- NVIDIA TensorRT-LLM Speculative Decoding Documentation
- [[06/09 - Sistemas de LLMs en Producción]]
- [[06/13 - vLLM and Advanced RAG]]
- [[06/17-03 - SGLang (current course)]]
- [[05/03 - Deep Learning con PyTorch]]
