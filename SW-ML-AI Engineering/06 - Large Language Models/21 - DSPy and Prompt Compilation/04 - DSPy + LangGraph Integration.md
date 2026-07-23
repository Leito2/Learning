# 🔗 DSPy + LangGraph Integration

Your [[../../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/00 - Welcome to LangGraph Deep Patterns.md|LangGraph Deep Patterns]] course taught you to build state machines with `StateGraph`, persistence via `PostgresSaver`, and human-in-the-loop with `interrupt()`. Your DSPy course taught you to compile prompts automatically. Now: **put them together**. A DSPy-compiled module inside a LangGraph node gives you state management + compiled prompts. **The best of both worlds: structured orchestration + auto-optimized LLM calls.**

This note covers the integration patterns: DSPy modules as LangGraph nodes, the `thread_id` propagation story, optimization for graph-level metrics, and a capstone that combines a multi-agent LangGraph system with DSPy-compiled agents.

By the end you can take any existing LangGraph agent and replace its hand-written prompts with compiled DSPy programs, with the optimizer finding the best prompts at each node.

## 🎯 Learning Objectives

- Wrap a **DSPy module** as a LangGraph node.
- Propagate `thread_id` and `baggage` from LangGraph state into DSPy.
- Compile a **graph-level metric** (success across multiple nodes).
- Build a **multi-agent LangGraph system** where each agent is DSPy-compiled.
- Apply the [[../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/05 - Human-in-the-Loop with interrupt() and Command.md|HITL]] pattern with DSPy prompts.
- Avoid the four most common integration pitfalls.

## 1. The Integration Architecture

```mermaid
flowchart LR
    subgraph LangGraph["LangGraph StateGraph"]
        Start[START] --> Rewrite[Rewrite<br/>DSPy module]
        Rewrite --> Retrieve[Retrieve<br/>dspy.Retrieve]
        Retrieve --> Rerank[Rerank<br/>DSPy module]
        Rerank --> Generate[Generate<br/>DSPy module]
        Generate --> Interrupt{HITL?<br/>interrupt()}
        Interrupt -->|yes| Human[Human Review]
        Interrupt -->|no| End[END]
        Human --> End
    end
```

Each DSPy module is a node. The state flows between nodes via the `TypedDict`. The `interrupt()` pauses for human review at the checkpoint.

## 2. The DSPy Module as a LangGraph Node

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
import dspy

class State(TypedDict):
    query: str
    rewritten_query: str
    context: list[str]
    findings: list[str]
    confidence: float
    draft: str
    final_answer: str
    thread_id: str

# DSPy signatures
class RewriteSignature(dspy.Signature):
    """Rewrite query for vector search."""
    original_query: str = dspy.InputField()
    rewritten_query: str = dspy.OutputField()

class GenerateSignature(dspy.Signature):
    """Answer using context. Cite sources."""
    context: list[str] = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Faithful, concise answer")
    citations: list[int] = dspy.OutputField()


# DSPy modules
class RewriteModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought(RewriteSignature)

    def forward(self, original_query):
        return self.rewrite(original_query=original_query)


class GenerateModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateSignature)

    def forward(self, query, context):
        return self.generate(query=query, context=context)


# Compile each module separately (or compile together — see note 04 section 5)
compiled_rewrite = compile_with_mipro(RewriteModule(), trainset)
compiled_generate = compile_with_mipro(GenerateModule(), trainset)


# LangGraph node functions — wrap DSPy modules
def rewrite_node(state: State) -> dict:
    """LangGraph node: invoke compiled DSPy rewrite module."""
    from opentelemetry import baggage
    from opentelemetry.context import attach
    ctx = baggage.set_baggage("thread_id", state["thread_id"])
    token = attach(ctx)
    try:
        result = compiled_rewrite(original_query=state["query"])
        return {"rewritten_query": result.rewritten_query}
    finally:
        from opentelemetry.context import detach
        detach(token)


def retrieve_node(state: State) -> dict:
    """LangGraph node: Qdrant retrieval."""
    embedding = embed_query(state["rewritten_query"])
    passages = qdrant_search(embedding, top_k=10)
    return {"context": [p.content for p in passages]}


def rerank_node(state: State) -> dict:
    """LangGraph node: bge-reranker."""
    top_passages = bge_rerank(state["rewritten_query"], state["context"], top_k=3)
    return {"context": top_passages}


def generate_node(state: State) -> dict:
    """LangGraph node: invoke compiled DSPy generation module."""
    result = compiled_generate(query=state["rewritten_query"], context=state["context"])
    return {
        "findings": state.get("findings", []) + [result.answer],
        "draft": result.answer,
    }


# Build the graph
graph = StateGraph(State)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "rewrite")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "generate")
graph.add_edge("generate", END)

app = graph.compile(checkpointer=PostgresSaver(...))
```

## 3. Thread ID Propagation

`thread_id` from LangGraph's `config["configurable"]["thread_id"]` flows into DSPy via OpenTelemetry baggage:

```python
def make_node(dspy_module):
    """Factory: wrap a DSPy module as a LangGraph node with baggage propagation."""
    def node(state: State) -> dict:
        from opentelemetry import baggage
        from opentelemetry.context import attach, detach

        ctx = baggage.set_baggage("thread_id", state.get("thread_id", "default"))
        ctx = baggage.set_baggage("user_id", state.get("user_id", ""))
        token = attach(ctx)

        try:
            # Run the compiled DSPy module
            result = dspy_module(**{k: v for k, v in state.items() if k in expected_inputs})
            return result_to_state_update(result)
        finally:
            detach(token)
    return node
```

Every DSPy call inside the node inherits `thread_id` baggage, which propagates to all LLM calls via Phoenix ([[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/00 - Welcome to OpenTelemetry for AI Engineers.md|09/34]]) traces.

## 4. Human-in-the-Loop with DSPy Prompts

```python
from langgraph.types import interrupt, Command

def human_review_node(state: State) -> dict:
    """HITL: pause for review if confidence is low."""
    from dspy import LM

    # Quick LLM-as-judge for confidence (could be DSPy-compiled too)
    judge = dspy.ChainOfThought("answer -> confidence: float, needs_review: bool")
    judgment = judge(answer=state["draft"])

    if judgment.needs_review or state.get("confidence", 1.0) < 0.6:
        # Pause for human review
        human_input = interrupt({
            "question": "Approve this answer?",
            "draft": state["draft"],
            "options": ["approve", "reject", "modify"],
        })

        return {
            "final_answer": human_input.get("answer", state["draft"]),
            "human_approved": human_input.get("choice") == "approve",
        }

    return {"final_answer": state["draft"], "human_approved": True}
```

## 5. Graph-Level Optimization

You can compile **the whole graph** with a single metric:

```python
def graph_metric(example, prediction, trace=None) -> float:
    """Evaluate the entire graph on one example."""
    # The trace contains all the spans from the graph run
    final_answer = prediction.final_answer
    expected_answer = example.answer
    return float(expected_answer.lower() in final_answer.lower())


class CompiledGraph(dspy.Module):
    """A LangGraph app wrapped as a DSPy module."""

    def __init__(self, app):
        super().__init__()
        self.app = app

    def forward(self, query, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke({"query": query, "thread_id": thread_id}, config)
        return dspy.Prediction(final_answer=result["final_answer"])


# Compile
graph_module = CompiledGraph(app)
trainset = [...]  # (query, expected_answer) pairs

from dspy.teleprompt import MIPRO
compiled_graph = MIPRO(metric=graph_metric, auto="medium").compile(
    graph_module,
    trainset=trainset,
)

# Use
result = compiled_graph(query="What is X?", thread_id="u-42")
print(result.final_answer)
```

**MIPRO tunes the prompts in `rewrite_node` and `generate_node` based on the graph-level outcome.** This is the production-grade compilation: the optimizer sees the full pipeline's success, not just one node.

## 6. Per-Agent Compilation

For multi-agent graphs, compile each agent separately:

```python
# Compile the research agent
research_agent = DSPyRAG(retriever)
compiled_research = MIPRO(metric=research_metric).compile(research_agent, trainset)

# Compile the synthesis agent
synthesis_agent = SynthesisDSPy()
compiled_synthesis = MIPRO(metric=synthesis_metric).compile(synthesis_agent, trainset)

# Compile the fact-audit agent
audit_agent = AuditDSPy()
compiled_audit = MIPRO(metric=audit_metric).compile(audit_agent, trainset)

# Build the multi-agent LangGraph using compiled modules
graph.add_node("research", lambda state: compiled_research(query=state["query"], context=...))
graph.add_node("audit", lambda state: compiled_audit(draft=state["draft"]))
graph.add_node("synthesis", lambda state: compiled_synthesis(findings=state["findings"]))
```

Each agent has its own optimized prompts. **The graph orchestrates; DSPy optimizes.**

## 7. ❌/✅ Antipatterns

### ❌ Compile without LangGraph context

```python
# ⚠️ Compiler doesn't see the graph structure
compiled = MIPRO(...).compile(RAG(), trainset)
# The compiled prompts work in isolation but fail in the graph
```

### ✅ Compile with graph-aware metric

```python
# ✅ Metric reflects the full graph outcome
def graph_metric(example, prediction, trace=None):
    return float(example.answer.lower() in prediction.final_answer.lower())

compiled = MIPRO(metric=graph_metric).compile(CompiledGraph(app), trainset)
```

### ❌ Drop the state in the node function

```python
# ⚠️ Returns empty dict — state doesn't update
def node(state):
    result = dspy_module(query=state["query"])
    # forgot to return state update
```

### ✅ Return state update dict

```python
# ✅ Returns the partial state update
def node(state):
    result = dspy_module(query=state["query"])
    return {"rewritten_query": result.rewritten_query}
```

### ❌ DSPy module with side effects

```python
# ⚠️ Optimizer can't replay side-effecting modules
class BadDSPyModule(dspy.Module):
    def forward(self, query):
        send_metric_to_prometheus(query)  # side effect
        return self.generate(query=query)
```

### ✅ Pure DSPy module

```python
# ✅ No side effects; optimizer can replay
class GoodDSPyModule(dspy.Module):
    def forward(self, query):
        return self.generate(query=query)
```

### ❌ Compiling too many modules

```python
# ⚠️ Compiler scales poorly with module count
compiled_graph = MIPRO(...).compile(GraphWith12DSPyModules(), trainset)
```

### ✅ Compile critical modules, leave simple ones alone

```python
# ✅ Optimize the high-value modules
compiled_rewrite = MIPRO(...).compile(RewriteModule(), trainset)  # optimize
compiled_generate = MIPRO(...).compile(GenerateModule(), trainset)  # optimize
simple_retrieve = PlainRetrieve()  # not worth compiling
```

## 8. Production Reality

**Caso real — Production RAG Project:** The chat service is a LangGraph state machine where the generation node is a DSPy-compiled module. The MIPRO compile took 2 hours and $15; the resulting faithfulness score (90%) is 12% better than the hand-tuned baseline. The `thread_id` propagates from LangGraph through DSPy into Phoenix traces.

**Caso real — Multi-Agent Research System:** Three agents (research, fact-audit, synthesis), each compiled separately with MIPRO using its own metric (research = retrieval accuracy, audit = factual precision, synthesis = answer quality). The graph-level metric (final answer F1) is monitored in CI; when it drops, a re-compile is triggered.

## 📦 Compression Code

```python
# 📦 Compression: DSPy + LangGraph in 70 lines

import dspy
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. DSPy module
class Generate(dspy.Signature):
    context: list[str] = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAGNode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(Generate)

    def forward(self, query, context):
        return self.generate(query=query, context=context)


# 2. LangGraph state
class State(TypedDict):
    query: str
    context: list[str]
    answer: str
    thread_id: str


# 3. LangGraph node (wraps DSPy)
compiled = compile_dspy(RAGNode(), trainset)  # MIPRO

def rag_node(state: State) -> dict:
    from opentelemetry import baggage
    from opentelemetry.context import attach, detach
    ctx = baggage.set_baggage("thread_id", state.get("thread_id", ""))
    token = attach(ctx)
    try:
        result = compiled(query=state["query"], context=state["context"])
        return {"answer": result.answer}
    finally:
        detach(token)


# 4. Graph
graph = StateGraph(State)
graph.add_node("rag", rag_node)
graph.add_edge(START, "rag")
graph.add_edge("rag", END)
app = graph.compile()
```

## 🎯 Key Takeaways

1. **DSPy modules as LangGraph nodes** — wrap `dspy.Module.forward` in a node function.
2. **Thread ID via baggage** — propagates from LangGraph state to DSPy to OTel.
3. **Compile each module separately** — agent-specific metrics, MIPRO per agent.
4. **Graph-level metric for end-to-end** — compiler sees the full pipeline outcome.
5. **HITL via `interrupt()`** — works with DSPy prompts seamlessly.
6. **Pure DSPy modules** — no side effects; optimizer can replay.
7. **Return state updates** — node function must return the partial `TypedDict`.

## References

- [[00 - Welcome to DSPy and Prompt Compilation|Welcome]] — course map.
- [[01 - Signatures and Modules|Signatures]] — the building blocks.
- [[02 - Optimizers|Optimizers]] — the compiler.
- [[03 - DSPy for RAG|RAG]] — retrieval module compilation.
- [[../../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/00 - Welcome to LangGraph Deep Patterns.md|LangGraph Deep Patterns]] — the graph primitive.
- [[../../../07 - AI Agents y Agentic Systems/18 - LangGraph Deep Patterns/05 - Human-in-the-Loop with interrupt() and Command.md|HITL]] — interrupt() integration.
- [[../../../09 - MLOps y Produccion/34 - OpenTelemetry for AI Engineers/04 - OTel for LangGraph and Agent Frameworks.md|OTel for LangGraph]] — thread_id propagation.
- DSPy + LangGraph: https://dspy.ai/tutorials/langgraph/