# 00 — Welcome to Harness Engineering and SDD

**Welcome.** This course maps the emerging discipline of AI agent control systems — the infrastructure that transforms stochastic LLMs into reliable, auditable software engineering teammates.

## The Problem

In 2025, AI coding tools reached production capability. But teams quickly discovered a pattern: agents perform brilliantly on isolated tasks yet wreak havoc in real codebases. They rewrite the wrong files. They forget decisions made two turns ago. They drift from architecture to exploration without warning.

**The missing layer is not a better model. It is deterministic control infrastructure around the model.**

## Course Map

This course builds the control plane from first principles, moving bottom-up through five layers of engineering:

| # | Note | Core Question |
|---|------|---------------|
| 00 | **Welcome** (this) | Why does AI development need control infrastructure? |
| 01 | [[01 - Context Engineering - The Physics of AI Attention]] | What is the MEDIUM we're engineering within? |
| 02 | [[02 - Harness Engineering - Directing AI Force]] | How do we BUILD the control structures? |
| 03 | [[03 - Specification-Driven Development - The Workflow Inside the Harness]] | What PROTOCOL runs inside the harness? |
| 04 | [[04 - File Architecture - Organizing Harness Infrastructure]] | WHERE does everything live on disk? |
| 05 | [[05 - Multi-Agent Orchestration and End-to-End Workflow]] | How does the COMPLETE system operate in motion? |

## The Core Insight

```
Chat-driven development = fragile, untraceable, unpredictable
Harness-driven development  = auditable, reproducible, safe
```

The difference is not the model. It's whether the model operates inside a structured control system that manages context, enforces phases, and records every decision as a file.

## Prerequisites

- **Software engineering fundamentals**: You understand code review, CI/CD, testing, and the difference between specification and implementation.
- **AI coding tools familiarity**: You've used Claude, ChatGPT, or Copilot on a real project. You've experienced the "agent went rogue" moment.
- **Command-line comfort**: Harnesses live in bash scripts and file trees, not GUIs.

## Key References

These companion notes provide foundations:
- [[../13 - Go Engineering/13 - Go Engineering]] — Language-level harness patterns
- [[../07 - AI Agents/07 - AI Agents]] — Agent architecture fundamentals
- [[../09 - MLOps/09 - MLOps]] — Production ML infrastructure patterns

## Navigating This Course

Read linearly. Each note assumes the previous. The stack is:

```
Context Engineering (physics) → Harness Engineering (structure) → SDD (protocol) → File Architecture (layout) → Orchestration (motion)
```

**Every layer constrains without eliminating.** The goal is never to remove AI creativity — it's to channel it safely.

Let's begin with the physics: [[01 - Context Engineering - The Physics of AI Attention]].

---

*Course compiled from Vercel D0, Gentle framework, Claude Code, and production harness engineering patterns. May 2026.*
