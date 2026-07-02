# 🎯 Welcome to ML System Design Interviews

## 🎯 Learning Objectives

- Decode the **three interview formats** (45-min on-site, 60-min virtual, take-home) and what each one tests
- Master the **interview scoring rubric** used at FAANG, OpenAI, Anthropic, and AI-first startups so you can self-grade
- Pick the **right problem depth** for your seniority: junior = API, mid = system, senior = ML-specific
- Build a **portfolio of 8 canonical ML problems** that recur across companies (search, dispatch, timeline, ETA, recs)
- Practice the **CLEAR framework** (Clarify → Locate → Estimate → Architect → Refine) in every problem

---

## Why This Course Exists

The 2026 ML Engineer interview has three rounds that catch every candidate off-guard: **system design for ML** (DoorDash dispatch, Airbnb search ranking, Twitter timeline), **ML system design** (LLM serving, RAG, fine-tuning pipelines), and **coding in production** (feature pipelines, model evaluation, A/B test analysis). The first round is the one most candidates fail — it requires a different mental model than coding interviews and a different vocabulary than SWE system design. This course is the systematic preparation for that round.

If you have already studied [[../32 - System Design for ML/00 - Welcome to System Design for ML|System Design for ML]] (06/32), you know the SWE-flavored patterns: microservices, queues, caches, sharding. This course adds the **ML-specific layer**: feature stores, online/offline skew, model retraining cadence, exploration vs exploitation, multi-objective optimization, and the data feedback loop that turns every prediction into training data for the next iteration. The combination of SWE system design + ML-specific concerns is what makes an ML System Design interview.

For your **portfolio projects**, the same problem statements appear in your interview prep. The **LLM Edge Gateway** in Go/Fiber is a small-scale version of a system design interview problem; the **Multi-Agent Research System** is a multi-agent RAG architecture question; the **StayBot** Airbnb agent is a property-search ranking problem. The interview asks you to design a system; you already built one. The course teaches you to articulate the design decisions you already made implicitly.

---

## Course Map

| #  | Note                                                                  | Focus                                                | Family        |
|----|-----------------------------------------------------------------------|------------------------------------------------------|---------------|
| 00 | [[00 - Welcome to ML System Design Interviews\|Welcome]] (this)       | Format decoder, scoring rubric, depth selection      | —             |
| 01 | [[01 - The CLEAR Framework - 5-Step Method\|CLEAR Framework]]          | Clarify, Locate, Estimate, Architect, Refine          | Meta-method   |
| 02 | [[02 - Problem 1 - Airbnb Search Ranking\|Airbnb Search]]              | Query understanding, listing embeddings, two-tower    | Search        |
| 03 | [[03 - Problem 2 - DoorDash Dispatch\|DoorDash Dispatch]]              | Batching, online assignment, hot-path inference      | Dispatch      |
| 04 | [[04 - Problem 3 - Twitter-X Timeline\|Twitter/X Timeline]]            | Multi-objective ranking, heavy ranker, diversity      | Feed          |
| 05 | [[05 - Problem 4 - Uber ETA Prediction\|Uber ETA]]                    | Spatio-temporal features, GBT + NN ensemble          | Prediction    |
| 06 | [[06 - Problem 5 - Netflix Recommendations\|Netflix Recs]]             | Two-stage retrieval + ranking, A/B infra             | Recs          |
| 07 | [[07 - Problem 6 - YouTube Recommendations\|YouTube Recs]]             | Multi-modal embeddings, deep candidate gen           | Recs          |
| 08 | [[08 - Problem 7 - TikTok For You Page\|TikTok FYP]]                  | Short-form signals, retention prediction             | Feed          |
| 09 | [[09 - Problem 8 - Spotify Discover Weekly\|Spotify Discover]]         | Collaborative filtering + audio NLP                  | Recs          |
| 10 | [[10 - Capstone - Design Your Own ML System\|Capstone]]                | Apply CLEAR end-to-end, evaluate, present tradeoffs  | Meta-method   |

---

## Prerequisites

| Concept | Expected Knowledge | Review If Needed |
|---------|-------------------|------------------|
| System design for SWE | Microservices, queues, caches, sharding | [[../32 - System Design for ML/00 - Welcome to System Design for ML\|System Design for ML]] |
| ML fundamentals | Supervised, unsupervised, eval metrics | [[../../05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida\|DL con PyTorch]] |
| Feature stores | Online/offline skew, point-in-time joins | [[../../09 - MLOps y Produccion/19 - Feature Engineering y Feature Stores/00 - Bienvenida\|Feature Engineering y Feature Stores]] |
| A/B testing | Experiment design, statistical significance | [[../../05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/04 - Training Strategies\|Training Strategies]] |
| Vector search | Embeddings, ANN, hybrid retrieval | [[../../10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search/00 - Welcome to Vector Databases and Semantic Search\|Vector DBs]] |
| Streaming | Kafka, Pub/Sub, real-time features | [[../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/00 - Welcome to LLM Gateway Patterns and LiteLLM\|LLM Gateway]] |

> 💡 **Tip**: This course is **interview prep**, not architecture theory. Each problem is sized for a 45-60 minute whiteboard discussion, not a 6-month engineering project. The depth is "what would you draw on the whiteboard and explain in 5 minutes per section."

---

## Project Outcome

By the end of this course you will have **8 whiteboard-ready ML system designs** (one per canonical problem), a personal **scoring rubric** for self-grading your mock interviews, and a **capstone design** for a 9th system (your choice) that you can present in a real interview. Every problem is structured so that you can reproduce the design from memory in under 10 minutes.

---

## 📦 Compression Code

```python
# COURSE_MAP: ML System Design Interviews
# 11 notes, ~4,400 lines, English
# Framework: CLEAR = Clarify -> Locate -> Estimate -> Architect -> Refine
# 8 canonical problems: search (Airbnb), dispatch (DoorDash), feed (Twitter/TikTok),
#                       prediction (Uber ETA), recs (Netflix/YouTube/Spotify)
# Per-problem template: Problem Statement -> Clarifying Questions -> Back-of-Envelope
#                      -> Architecture (Mermaid) -> ML Component -> System Component
#                      -> Tradeoffs -> Production Reality -> Compression Code
# Cross-links: 06/32 System Design, 09/21 Monitoreo, 09/27 Feast, 10/30 WebSockets,
#              10/33 Vector DBs, 10/23 IaC
# Use case: interview prep for ML Engineer / Applied Scientist roles
```

## 🎯 Key Takeaways

- **ML system design ≠ SWE system design** — the ML-specific layer (features, retraining, exploration, data feedback) is what trips up most candidates
- **8 canonical problems cover the field** — search, dispatch, feed, prediction, recs; if you can design 8, you can design 80
- **CLEAR is the framework** — Clarify, Locate, Estimate, Architect, Refine; 5 steps, 5 minutes each
- **Every problem follows the same template** — muscle memory from 8 reps > intellectual insight from 1
- **The capstone is yours** — pick a 9th system (yours, a real one) and present it as the closing argument

## References

- Alex Xu, *System Design Interview* Vol 1 & 2
- Alex Xu, *Machine Learning System Design Interview*
- *Acing the System Design Interview* (Zhiyong Tan)
- ML System Design interviews on YouTube (Stanford CS329H, Tecton, Cornell Tech)
- Hiring rubrics: levels.fyi, interviewing.io, FAANG engineering blogs
