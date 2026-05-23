# 🎓 ML Interview Preparation

## Introduction

ML engineering interviews test a unique combination of skills: system design for ML, theoretical depth, coding proficiency, and behavioral alignment. This note maps the interview landscape and provides a structured preparation plan targeted at AI/ML Engineer roles.

---

## The ML Interview Landscape

| Round | Duration | What They Test | Weight |
|---|---|---|---|
| **ML System Design** | 45-60 min | Design an end-to-end ML system (recommendation, fraud, search) | 30% |
| **ML Theory/Depth** | 45 min | Deep understanding of algorithms, loss functions, evaluation metrics | 25% |
| **Coding** | 45 min | Data structures, algorithms, Python fluency, ML coding from scratch | 25% |
| **Behavioral** | 30 min | Teamwork, ownership, conflict resolution, project impact | 15% |
| **Take-home** | 4-8 hours | Build and document a small ML system | Varies |

---

## ML System Design Framework

Use the **4D Framework** for any ML system design question:

1. **Define** — Clarify the problem, users, success metrics, and constraints
2. **Data** — What data exists? How to collect, label, and validate? Data pipeline design
3. **Design** — Model architecture, loss function, evaluation metrics, offline/online experiments
4. **Deploy** — Serving architecture, monitoring, retraining, A/B testing, failure modes

### Common Questions

- "Design YouTube's video recommendation system"
- "Build a fraud detection system for credit card transactions"
- "Design the ML system behind Google Photos search"
- "How would you detect and fix training-serving skew?"

---

## ML Theory: Depth Areas

| Topic | Go Deep On |
|---|---|
| **Loss Functions** | Cross-entropy, focal loss, contrastive loss, triplet loss — when to use each |
| **Evaluation** | AUC-ROC vs PR-AUC, offline vs online metrics, counterfactual evaluation |
| **Regularization** | L1 vs L2, dropout, batch norm, data augmentation, early stopping |
| **Optimization** | SGD, Adam, AdamW, learning rate schedules, gradient clipping |
| **Transformers** | Attention mechanism, positional encoding, KV cache, pre-training objectives |
| **Bias/Variance** | Diagnose from learning curves, how to address each |
| **Distribution Shift** | Covariate shift, label shift, concept drift — detect and mitigate |

---

## Preparation Plan (4 Weeks)

### Week 1: ML Theory + Coding
- Read ML system design case studies (YouTube, Uber, Airbnb blogs)
- Practice coding: LeetCode Medium, implement ML algorithms from scratch
- Review fundamentals: loss functions, evaluation metrics, regularization

### Week 2: System Design
- Practice 10+ ML system design questions with the 4D framework
- Study production architectures: feature stores, model registries, serving
- Read "Machine Learning System Design" (Chip Huyen, Stanford)

### Week 3: Deep Dive Your Projects
- Prepare STAR stories for 3 projects
- Be ready to discuss: trade-offs, failures, what you'd do differently
- Quantify impact: "Improved accuracy by X%, reduced latency by Y ms"

### Week 4: Mock Interviews
- Mock ML system design with a peer or platform (Interviewing.io, Pramp)
- Mock coding with time pressure
- Review and refine all materials

---

## Key Behavioral Questions

- "Tell me about a ML project that failed. What did you learn?"
- "How do you handle disagreement about model metrics with stakeholders?"
- "Describe a time you had to ship a model with imperfect data."
- "How do you prioritize between model accuracy and engineering effort?"

---

## References

- Chip Huyen, "Designing Machine Learning Systems" (O'Reilly, 2022)
- [ML System Design Primer (GitHub)](https://github.com/chiphuyen/machine-learning-system-design)
- [Stanford MLSys Seminar](https://mlsys.stanford.edu/)
