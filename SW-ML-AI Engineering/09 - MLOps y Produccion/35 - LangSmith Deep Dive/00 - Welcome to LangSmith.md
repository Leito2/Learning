# 🪶 Welcome to LangSmith

You just deployed a LangGraph agent to production. A user reports "the answer is wrong." You open your dashboard and see **a trace tree**: `chat_handler → research → fact_audit → synthesis`. Each node has its inputs, outputs, latency, and the exact prompt sent to the LLM. You click the failing trace, compare it to a successful trace, and identify the bad prompt in 5 minutes. **That's LangSmith.**

LangSmith (LangChain's commercial observability platform) is the most-used LLM observability backend in production. Where [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers.md|09/34 OpenTelemetry]] is the **vendor-neutral standard**, LangSmith is the **LLM-specific opinionated platform**. It ships:

- **Trace explorer** with prompts, completions, and tool calls visible inline.
- **Datasets** for golden test sets, versioning, and management.
- **Online evaluators** that run on every production trace.
- **Annotation queues** for human feedback workflows.
- **A/B testing** with statistical comparison built in.

This is the second observability course in the vault. After [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers.md|09/34]], you understand the standard. After this one, you understand the most-used LLM-specific platform. **The two are complementary**: LangSmith projects can export to OTel, and OTel traces can be enriched with LangChain-specific metadata.

## 🎯 Learning Objectives

- Master **LangSmith primitives**: traces, runs, projects, datasets, evaluators.
- Use **auto-instrumentation** for OpenAI, Anthropic, LangChain, LangGraph.
- Build **datasets** with versioning for evaluation.
- Run **offline evaluations** on golden test sets.
- Configure **online evaluators** that run on production traces.
- Set up **annotation queues** for human-in-the-loop feedback.
- Deploy with **production patterns**: sampling, costs, PII.

## Course Map

| # | Note | Core concept | Closes gap |
|:-:|------|--------------|------------|
| 00 | [[00 - Welcome to LangSmith\|You are here]] | Why LangSmith for LLM observability | Course map |
| 01 | [[01 - LangSmith Core - Traces Runs Projects\|Core Primitives]] | Traces, runs, projects, datasets | Gap #1 |
| 02 | [[02 - Auto-Instrumentation for LLM SDKs\|Auto-Instrumentation]] | OpenAI, Anthropic, LangChain, LangGraph | Gap #2 |
| 03 | [[03 - Datasets and Evaluations\|Datasets]] | Test set management, versioning | Gap #3 |
| 04 | [[04 - Online Evaluators and LLM-as-Judge\|Online Evals]] | Production-time scoring | Gap #4 |
| 05 | [[05 - Annotation Queues and Human Feedback\|Annotation Queues]] | Human-in-the-loop feedback | Gap #5 |
| 06 | [[06 - Production Patterns - Sampling Costs PII\|Production]] | Sampling, costs, PII redaction | Gap #6 |
| 07 | [[07 - Capstone - Production RAG with LangSmith\|Capstone]] | Production LangSmith-wired RAG | Integration |

## Why LangSmith vs Phoenix

Both are popular LLM observability backends. The choice is usually ecosystem fit:

| Aspect | LangSmith | Phoenix (Arize) |
|--------|-----------|-----------------|
| **Ecosystem** | LangChain, LangGraph native | Framework-agnostic, OTel-native |
| **Pricing** | Per-trace + storage | Per-trace + storage |
| **Self-host** | ❌ Closed source | ✅ Open source |
| **Datasets** | ✅ First-class | Basic |
| **Online evals** | ✅ Built-in | DIY |
| **Annotation queues** | ✅ Built-in | DIY |
| **OTel export** | ✅ Recent feature | Native |

**Choose LangSmith** if you're all-in on LangChain/LangGraph and want the best-in-class developer experience for LLM evaluation. **Choose Phoenix** if you want self-hosting, OTel-native, or framework-agnostic.

> 💡 **Tip:** LangSmith now exports to OTel (since mid-2024). You can use LangSmith for LLM-specific features (datasets, evaluators, annotation queues) while keeping OTel as your primary observability backend.

## Prerequisites

- **Python 3.10+** with `pip install langsmith langchain langchain-openai`.
- **LangChain or LangGraph basics** ([[../../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/00 - Welcome to LangGraph Deep Patterns.md|07/18]]).
- **OpenTelemetry basics** (recommended, not required — [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers.md|09/34]]).
- **LLM API keys** (OpenAI or Anthropic).

## How to Read This Course

1. **Notes 01-02** are the foundation: primitives + auto-instrumentation.
2. **Notes 03-04** are the LLM-specific value: datasets + evaluators.
3. **Notes 05-06** are production patterns: human feedback + sampling.
4. **Note 07** is the capstone integration.

## 📦 Compression Code

```python
# 📦 Welcome - LangSmith in 15 lines
import os
from langsmith import traceable
from langchain_openai import ChatOpenAI

# 1. Set environment variables
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-chatbot"

# 2. Use any LangChain component — auto-traced
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is LangSmith?")

# 3. Custom functions — wrap with @traceable
@traceable
def my_custom_step(query: str) -> str:
    return f"Processed: {query}"

# 4. View in LangSmith UI at https://smith.langchain.com
```

That's it. Every LangChain call, every `@traceable` function, every custom span is in LangSmith. **No SDK integration work beyond setting the API key.**

## References

- [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers.md|OpenTelemetry]] — the vendor-neutral alternative.
- [[../../../09 - MLOps y Produccion/31 - Evidently AI and Phoenix/00 - Welcome to Evidently AI and Phoenix.md|Phoenix]] — the OSS alternative.
- [[../../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/00 - Welcome to LangGraph Deep Patterns.md|LangGraph Deep Patterns]] — LangSmith integrates with LangGraph natively.
- LangSmith docs: https://docs.smith.langchain.com/
- LangSmith pricing: https://smith.langchain.com/#pricing