# 🧬 Synthetic Data Generation

## Introduction

Synthetic data generation uses LLMs or generative models to create training data when real data is scarce, expensive, or privacy-constrained. For fine-tuning models on domain-specific tasks where labeled data doesn't exist, synthetic generation is often the only viable path. It is also a core technique for instruction-tuning datasets (Self-Instruct, Alpaca) and for creating evaluation benchmarks without expensive human annotation.

---

## Core Techniques

### 1. Self-Instruct (Prompt-Based)

Generate (instruction, input, output) triples by prompting a strong LLM:

```
Prompt: "Generate 5 diverse instructions for a customer support assistant.
         For each, write a helpful response."

Output:
  1. Instruction: "How do I reset my password?"
     Response: "To reset your password, go to Settings → Account → ..."
  2. Instruction: "I was charged twice for my subscription."
     Response: "I understand your concern. Let me check..."
  ...
```

### 2. Distillation from Stronger Models

Use GPT-4/Claude to generate training data for fine-tuning smaller models like Llama-3-8B:

```
For each seed prompt:
  1. Send to GPT-4 → get high-quality response
  2. (seed_prompt, GPT-4_response) becomes training data for Llama-3-8B
```

### 3. Back-Translation and Paraphrasing

Given a small set of real examples, generate variants through paraphrasing:

```
Original: "What is the refund policy for digital products?"
Paraphrased: "Can I get my money back if I bought a digital item?"
Paraphrased: "Does your digital product refund policy differ from physical items?"
```

### 4. Agent-Roleplay for Multi-Turn Data

For conversation and agent training, simulate multi-turn interactions between an LLM playing the user and another playing the assistant.

---

## Quality Control

| Technique | How | When |
|---|---|---|
| **Deduplication** | Remove near-duplicate examples (MinHash, embeddings) | Always — synthetic data is repetitive |
| **Diversity filtering** | Cluster embeddings, sample evenly from clusters | When generation skews toward common patterns |
| **Quality scoring** | Score examples with reward model or LLM-as-judge | Filter out low-quality synthetic examples |
| **Human verification** | Spot-check 1-5% of generated data | Before using for production fine-tuning |
| **Contamination check** | Verify generated data doesn't leak test set information | When using strong LLMs for generation |

---

## ⚠️ Pitfalls

- **Model collapse:** Training on synthetic data from the same model amplifies biases and reduces diversity. Always mix synthetic with real data.
- **Hallucination propagation:** If GPT-4 hallucinates an answer and you train Llama on it, Llama learns the hallucination. Verify factual claims in synthetic data.

---

## References

- Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (2023)
- Gunasekar et al., "Textbooks Are All You Need" (Phi-1, 2023)
- [Synthetic Data in ML (Google Research)](https://research.google/pubs/synthetic-data/)
