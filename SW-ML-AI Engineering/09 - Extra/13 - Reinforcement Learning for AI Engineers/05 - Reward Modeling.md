# 🎯 Reward Modeling and Preference Optimization

## Introduction

The reward model is the bottleneck in RLHF. If the reward model doesn't accurately capture human preferences, no amount of PPO optimization will produce a truly aligned model — you'll just optimize a flawed proxy faster. Reward modeling is where the human signal enters the RLHF pipeline, and getting it right is more art than science: it requires careful data collection, calibration against biases, and iterative refinement as the policy improves.

This module covers the complete reward modeling lifecycle: collecting preference data, training Bradley-Terry models, handling annotation noise, debiasing common reward model failure modes, and the emerging paradigm of preference optimization (DPO, KTO) that bypasses reward models entirely.

---

## 1. 📊 Collecting Human Preference Data

### The Comparison Interface

Human labelers are shown a prompt and two responses (A and B) and asked to choose which is better — or indicate tie/equally good/equally bad:

```
┌──────────────────────────────────────────────────┐
│ PROMPT: "Explain gravity to a 5-year-old"         │
│                                                  │
│ RESPONSE A:                                      │
│ "Gravity is like a giant invisible hand that     │
│ pulls everything down to Earth. When you jump,    │
│ gravity pulls you back. The bigger something is,  │
│ the stronger its gravity pull."                   │
│                                                  │
│ RESPONSE B:                                      │
│ "Gravity is a fundamental force described by     │
│ Newton's law of universal gravitation:            │
│ F = G(m₁m₂)/r². It is the curvature of           │
│ spacetime caused by mass-energy."                 │
│                                                  │
│ Which is better for a 5-year-old?                 │
│ ○ A is much better  ○ A is better  ○ Tie         │
│ ○ B is better  ○ B is much better                │
└──────────────────────────────────────────────────┘
```

### Preference Annotation Scales

| Scale | Options | Best For |
|---|---|---|
| **Binary** | A > B, B > A | Simple, high throughput |
| **5-point Likert** | A >> B, A > B, Tie, B > A, B >> A | Captures preference strength |
| **Multi-dimensional** | Rate helpfulness, harmlessness, honesty separately | Training multi-objective RMs |
| **Ranking** | Rank 3+ responses in order | Efficient data collection (more signal per prompt) |

### The Data Collection Loop

```
1. Generate responses from SFT model for diverse prompts
2. Send (prompt, response_A, response_B) to human labelers
3. Aggregate judgments (multiple labelers per comparison)
4. Filter low-agreement comparisons (inter-annotator < threshold)
5. Train RM on cleaned preference pairs
6. Use RM to train PPO policy
7. Generate new responses from PPO policy
8. Send new responses for human comparison (RM may be stale)
9. Retrain RM on expanded dataset
10. Repeat
```

---

## 2. 🔬 Reward Model Architecture and Training

### Architecture Options

| Architecture | Description | When to Use |
|---|---|---|
| **Full LLM + linear head** | Base model (e.g., Llama-7B) with a scalar projection on final hidden state | Default choice — best accuracy |
| **Shared backbone** | Policy and RM share base model weights (only head differs) | Reduces memory, faster inference |
| **Ensemble of RMs** | Train multiple RMs with different seeds, average scores | Reduces reward hacking, better calibration |
| **Multi-head RM** | Separate heads for helpfulness, harmlessness, honesty | Multi-objective alignment |
| **Process-based RM** | Score each step in chain-of-thought, not just final output | Math reasoning, code generation |

### Training Recipe

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset of (prompt, chosen_response, rejected_response) triples."""

    def __init__(self, prompts, chosen, rejected):
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.chosen[idx], self.rejected[idx]


def train_reward_model(rm, dataloader, optimizer, device, epochs=3):
    """Train reward model on preference pairs."""

    rm.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for prompts, chosen, rejected in dataloader:
            prompts, chosen, rejected = (
                prompts.to(device), chosen.to(device), rejected.to(device)
            )

            # Get rewards for both responses
            r_chosen = rm(prompts, chosen)       # Shape: (batch,)
            r_rejected = rm(prompts, rejected)   # Shape: (batch,)

            # Bradley-Terry loss: maximize r_chosen - r_rejected
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()

        accuracy = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
              f"Accuracy={accuracy:.3f}")
```

### Reward Model Evaluation Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **Pairwise accuracy** | % of comparisons where RM ranks chosen > rejected | > 70% (human agreement ~70-75%) |
| **Calibration** | Do RM scores correlate with human preference strength? | High Pearson correlation |
| **Per-prompt accuracy** | Can RM distinguish quality within the same prompt? | > 65% |
| **OOD detection** | Can RM identify when it's uncertain? | Low confidence on unfamiliar inputs |
| **Ensemble disagreement** | Do diverse RMs agree? If not, reward is unreliable | Low variance |

---

## 3. ⚠️ Common Reward Model Failure Modes

### Mode 1: Length Bias

RMs systematically prefer longer responses, even when shorter responses are equally informative. This creates a perverse incentive for PPO to generate verbose, repetitive text.

**Solution:** Include response length as a feature during RM training, or normalize rewards by length. OpenAI includes a length penalty in the PPO reward to counteract this.

### Mode 2: Confidence / Flattery Bias

RMs prefer responses that sound confident and agreeable, even when wrong. An incorrect answer stated confidently often outscores a correct answer stated tentatively.

**Solution:** Include "correctness" as a separate reward dimension. For factual prompts, use ground-truth evaluation as an additional reward signal.

### Mode 3: Position Bias

Labelers tend to prefer the first response they read (or the last). The presentation order affects judgments.

**Solution:** Randomize response order for each comparison. Compute inter-annotator agreement separately for A-first vs B-first presentations.

### Mode 4: Sycophancy

RMs reward responses that agree with the user's stated beliefs, even when those beliefs are factually incorrect.

**Solution:** Include "honesty" prompts where the user states a false belief and the assistant must politely correct them. Train RM to prefer honest corrections over sycophantic agreement.

### Mode 5: Distributional Collapse

As PPO optimizes against the RM, the policy moves into regions where the RM was never trained. The RM's scores become increasingly unreliable.

**Solution:** Retrain the RM on PPO-generated responses every 5-10 PPO iterations (online RLHF). This keeps the RM calibrated on the current policy distribution.

---

## 4. 🔄 Direct Preference Optimization (DPO)

DPO eliminates the reward model entirely by reparameterizing the Bradley-Terry model:

### The DPO Loss

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

The policy itself becomes the implicit reward model: $r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.

### DPO vs RLHF

| Aspect | RLHF (PPO + RM) | DPO |
|---|---|---|
| **Components** | 3 models (SFT, RM, PPO policy) | 2 models (SFT, DPO policy) |
| **Training pipeline** | Sequential: RM training → PPO | Single stage: optimize on preference pairs |
| **Iterative refinement** | Online RLHF: retrain RM on policy outputs | Offline: static preference dataset |
| **Reward hacking risk** | Yes (RM misspecification) | Lower (no separate RM to exploit) |
| **Compute** | High (RM + PPO + generation) | Lower (single loss, no generation during training) |
| **Flexibility** | Can add multi-objective rewards, constraints | Simpler but less customizable |
| **Performance** | Gold standard (carefully tuned) | Competitive on benchmarks |

### When to Use Each

```
Use RLHF (PPO + RM) when:
  └── You need iterative refinement with human-in-the-loop
  └── You want multi-objective alignment (safety + quality + honesty)
  └── You have the infrastructure for online training

Use DPO when:
  └── You have a static, high-quality preference dataset
  └── You want a simpler training pipeline (fewer models to maintain)
  └── You're resource-constrained and can't run online PPO
```

---

## 5. 🔬 Advanced Preference Optimization

### KTO (Kahneman-Tversky Optimization)

KTO aligns from binary (good/bad) feedback per example — no pairwise comparisons needed. This is more natural in production (users provide thumbs up/down, not comparisons between alternatives).

### Iterative DPO

Train DPO → generate new responses → get human feedback on those responses → retrain DPO. This mimics the iterative refinement of RLHF while keeping the simplicity of DPO.

### Reward Model Ensembling

```
┌──────────────────────────────────────────────────────────┐
│              REWARD MODEL ENSEMBLE                        │
│                                                          │
│  RM₁ (trained on seed 1) ──┐                             │
│  RM₂ (trained on seed 2) ──┼── Average ──▶ Final Reward  │
│  RM₃ (trained on seed 3) ──┘                             │
│                                                          │
│  Bonus: Variance across RMs = uncertainty estimate       │
│  High variance → reward is unreliable → weight it less   │
└──────────────────────────────────────────────────────────┘
```

---

## 6. 🌍 Production Reward Model Deployments

| Organization | RM Approach | Scale |
|---|---|---|
| **OpenAI** | Ensemble of RMs + multi-dimensional scoring | Hundreds of thousands of comparisons |
| **Anthropic** | Constitutional AI + RM for harmlessness | AI-generated critiques replace human labels |
| **Google** | Multi-objective RM (quality + safety + groundedness) | Used across Gemini family |
| **Meta** | DPO on community preference data for Llama | Open-source preference dataset |
| **Hugging Face** | Community DPO datasets (UltraFeedback, OpenAssistant) | Democratizing alignment |

---

## ⚠️ Pitfalls

- **Using the same model as policy and RM initializer:** Starting RMs from the same checkpoint as the policy creates correlation — the RM may rate the policy's outputs highly simply because they share the same pre-training biases. Initialize RM from a different model family or checkpoint.
- **Imbalanced preference data:** If 80% of comparisons in your dataset prefer response A, the RM learns to always predict A. Balance your dataset — exactly 50% should have A as the winner.
- **Comparison quality decreases with response divergence:** Labelers find it harder to judge when responses are very different in style. Prefer comparisons between similar-length, similar-style responses for cleaner signal.

---

## 💡 Tips

- **Use multiple labelers per comparison:** Inter-annotator agreement is 60-70% for complex prompts. 3-5 labelers per comparison and using majority vote reduces noise significantly.
- **Include "gold" comparisons for quality control:** Intersperse comparisons with known answers among the real ones. If labelers fail gold comparisons, flag their data for review.
- **Log reward model scores alongside PPO metrics:** Track RM score distribution over training. If scores shoot up while human eval stays flat, your RM is being hacked.
- **Calibrate RMs with human eval periodically:** Every N PPO steps, run a human evaluation batch and compare RM scores to human preferences. Calibration drift = retrain the RM.

---

## 📦 Compression Code

```python
import torch.nn.functional as F

def dpo_loss(pi_logps_chosen, pi_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """Direct Preference Optimization loss."""
    pi_logratios = pi_logps_chosen - pi_logps_rejected
    ref_logratios = ref_logps_chosen - ref_logps_rejected
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()
```

---

## ✅ Knowledge Check

1. **Why is reward model training harder than it seems?** — Human preferences are noisy (inter-annotator agreement ~65%), RMs exhibit systematic biases (length, confidence, position), and the policy's output distribution drifts beyond the RM's training distribution (distributional shift).

2. **How does DPO eliminate the reward model?** — DPO shows that the optimal policy under the Bradley-Terry model can be expressed in terms of policy log-ratios to a reference model. The policy itself becomes the implicit reward model, removing the need for a separate RM.

3. **What is the length bias problem in reward models?** — RMs systematically prefer longer responses because length correlates with perceived informativeness in training data. Without correction, PPO optimizes for verbosity, not quality.

4. **Why use an ensemble of reward models instead of a single RM?** — Individual RMs have blind spots and biases. Ensemble averaging reduces variance, and disagreement between ensemble members signals uncertain regions where the reward signal is unreliable.

---

## 🎯 Key Takeaways

- Reward models are the human signal bottleneck in RLHF — their quality determines the ceiling of alignment.
- Bradley-Terry models train on pairwise preferences: $\mathcal{L} = -\log \sigma(r_w - r_l)$.
- RMs suffer from systematic biases: length, confidence, position, and sycophancy — each requires targeted debiasing.
- DPO eliminates the RM entirely by making the policy itself the implicit reward model — simpler, competitive performance.
- Iterative refinement (online RLHF or iterative DPO) is essential: retrain on policy-generated outputs to prevent distributional collapse.

---

## References

- Stiennon et al., "Learning to summarize with human feedback" (NeurIPS, 2020)
- Bai et al., "Training a Helpful and Harmless Assistant with RLHF" (arXiv, 2022)
- Rafailov et al., "DPO: Your Language Model is Secretly a Reward Model" (NeurIPS, 2023)
- Cheng et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (arXiv, 2024)
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (arXiv, 2023)
