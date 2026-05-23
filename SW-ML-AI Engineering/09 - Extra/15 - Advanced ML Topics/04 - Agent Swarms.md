# 🐝 Agent Swarms

## Introduction

Agent swarms distribute tasks across multiple lightweight AI agents that operate in parallel, communicate minimally, and converge on a solution through simple coordination rules. Unlike monolithic agents that reason through entire chains of thought, swarm architectures decompose complex tasks into independent subtasks that multiple agents attack simultaneously — inspired by biological systems (ant colonies, bee swarms).

For AI engineers building multi-agent systems, swarm patterns offer an alternative to heavy orchestration frameworks (LangGraph, CrewAI) when the task can be parallelized, the coordination protocol is simple, and cost is a concern.

---

## Key Frameworks

### OpenAI Swarm (Experimental)

OpenAI's lightweight multi-agent framework based on two primitives:

| Primitive | What It Does |
|---|---|
| **Agent** | Has instructions, tools, and a model. Can hand off to another agent. |
| **Handoff** | Agent transfers conversation to another agent, passing context |

```python
from swarm import Swarm, Agent

client = Swarm()

def refund_tool(order_id): return f"Refunded order {order_id}"
def cancel_tool(order_id): return f"Cancelled order {order_id}"

refund_agent = Agent(name="Refund Agent", instructions="Handle refunds.", functions=[refund_tool])
cancel_agent = Agent(name="Cancel Agent", instructions="Handle cancellations.", functions=[cancel_tool])
triage_agent = Agent(name="Triage", instructions="Route to refund or cancel.", functions=[refund_agent, cancel_agent])

response = client.run(agent=triage_agent, messages=[{"role": "user", "content": "Cancel order #456"}])
# triage_agent detects "cancel" → hands off to cancel_agent → cancel_agent calls cancel_tool
```

### Microsoft Magentic-One

A multi-agent system where a central Orchestrator agent directs specialized agents:

```
Orchestrator (Plans and delegates)
├── WebSurfer (Browser-based agent)
├── FileSurfer (Local file operations)
├── Coder (Code generation and execution)
└── ComputerTerminal (CLI operations)
```

---

## Swarm vs Orchestrated Agents

| Aspect | Swarm (OpenAI Swarm) | Orchestrated (LangGraph) |
|---|---|---|
| **Control flow** | Handoff-based routing | Explicit DAG graph |
| **Parallelism** | Natural (agents run independently) | Must be explicitly defined |
| **State management** | Minimal (conversation context) | Sophisticated (checkpoints, branching) |
| **Best for** | Task routing, simple multi-agent | Complex workflows, conditional logic |
| **Complexity** | Low (2 primitives) | Medium-high (graph API) |

---

## ⚠️ Considerations

- **Swarm doesn't guarantee correctness:** Agents hand off independently — no central orchestrator validates the overall solution. For safety-critical tasks, use orchestrated agents.
- **Context window accumulates:** Each handoff passes full conversation history. For multi-hour swarm sessions, implement context pruning.
- **Cost multiplies with agents:** N agents running in parallel = N× API calls. Use small models for routing agents, large models only for reasoning agents.

---

## References

- [OpenAI Swarm (GitHub)](https://github.com/openai/swarm)
- [Microsoft Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one/)
- "Swarm Intelligence: From Natural to Artificial Systems" (Bonabeau et al.)
