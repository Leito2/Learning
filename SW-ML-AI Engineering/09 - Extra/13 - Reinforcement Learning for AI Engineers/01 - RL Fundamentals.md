# 🧠 RL Fundamentals: MDPs, Value Functions, and Policies

## Introduction

Reinforcement Learning solves a fundamentally different problem from supervised learning. In supervised learning, you have a dataset of (input, correct_output) pairs and the model learns to map between them. In RL, there is no correct output — only a reward signal that arrives after a sequence of actions, often delayed, sparse, and stochastic. The agent must discover which actions lead to reward through trial and error in an unknown environment.

This module builds the mathematical foundation of RL: Markov Decision Processes that formalize the environment, value functions that quantify the long-term desirability of states, and policies that map states to actions. These concepts are the vocabulary of every RL algorithm from DQN to PPO to RLHF.

---

## 1. 🧠 The Agent-Environment Loop

RL frames learning as an interaction between an agent and an environment over discrete time steps:

```mermaid
graph LR
    A[Agent<br/>Policy π] -->|Action a_t| B[Environment]
    B -->|State s_{t+1}| A
    B -->|Reward r_t| A

    style A fill:#ccffcc
    style B fill:#e0e0ff
```

At each time step $t$:
1. Agent observes state $s_t$
2. Agent selects action $a_t$ according to its policy $\pi(a_t | s_t)$
3. Environment transitions to state $s_{t+1}$ and emits reward $r_t$
4. Agent updates its policy based on experience $(s_t, a_t, r_t, s_{t+1})$

The agent's goal: maximize the cumulative discounted reward (the **return**):

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

Where $\gamma \in [0, 1]$ is the discount factor. $\gamma = 0$ means the agent is myopic (only cares about immediate reward). $\gamma \approx 1$ means the agent is far-sighted (values future reward almost as much as immediate).

---

## 2. 📐 Markov Decision Processes (MDPs)

An MDP is the formal mathematical framework for RL. It is defined by the tuple:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

| Symbol | Name | Meaning | Example (Chatbot) |
|---|---|---|---|
| $\mathcal{S}$ | State space | All possible situations the agent can be in | Conversation history + user intent |
| $\mathcal{A}$ | Action space | All possible actions the agent can take | Next token from vocabulary |
| $\mathcal{P}$ | Transition function | $P(s' \mid s, a)$ — probability of landing in $s'$ | LLM output distribution |
| $\mathcal{R}$ | Reward function | $R(s, a, s')$ — immediate reward | Human thumbs up/down |
| $\gamma$ | Discount factor | How much to value future rewards | 0.95-0.99 typical |

### The Markov Property

The future depends only on the present state and action, not on the history:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)
$$

This is the **Markov property** — the state $s_t$ is a sufficient statistic of the entire history. In practice, this means your state representation must encode everything the agent needs to make optimal decisions.

```
┌──────────────────────────────────────────────────────────────┐
│                MDP TRAJECTORY                                 │
│                                                              │
│  s₀ ──a₀──▶ s₁ ──a₁──▶ s₂ ──a₂──▶ ... ──▶ s_T (terminal)   │
│        r₀         r₁         r₂                               │
│                                                              │
│  Return G₀ = r₀ + γr₁ + γ²r₂ + ... + γ^T r_T               │
└──────────────────────────────────────────────────────────────┘
```

### Episodic vs Continuing Tasks

| Type | Termination | Return | Example |
|---|---|---|---|
| **Episodic** | Has terminal state | Finite sum | Game of chess, dialogue session |
| **Continuing** | No terminal state | Infinite sum (requires γ < 1) | Robot navigation, stock trading |

---

## 3. 🔢 Value Functions: V(s) and Q(s, a)

Value functions quantify "how good" a state or state-action pair is — not by immediate reward, but by expected future return.

### State Value Function V(s)

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \mid S_t = s \right]
$$

$V^\pi(s)$ answers: "If I start in state $s$ and follow policy $\pi$, what total reward do I expect?"

### Action-Value Function Q(s, a)

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]
$$

$Q^\pi(s, a)$ answers: "If I take action $a$ in state $s$, then follow policy $\pi$, what total reward do I expect?"

### The Bellman Equation

The Bellman equation decomposes the value function into immediate reward plus discounted future value — this recursive relationship is the computational heart of RL:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]
$$

Intuitively: the value of a state is the average over all actions (weighted by policy) of the immediate reward plus the discounted value of the resulting state.

For Q-values:

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]
$$

### Optimal Value Functions

The optimal policy $\pi^*$ achieves the maximum value from every state:

$$
V^*(s) = \max_\pi V^\pi(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
$$

This is the **Bellman optimality equation**. If you know $V^*$ or $Q^*$, optimal behavior is trivial: always choose $\argmax_a Q^*(s, a)$.

---

## 4. 📈 Policies: From Stochastic to Deterministic

A policy $\pi$ defines the agent's behavior — how it maps states to actions:

| Policy Type | Definition | Use Case |
|---|---|---|
| **Stochastic** | $\pi(a \mid s)$ — probability distribution over actions | Exploration, RLHF, continuous action spaces |
| **Deterministic** | $a = \mu(s)$ — single action per state | Exploitation after learning, DDPG |
| **Greedy** | $a = \argmax_a Q(s, a)$ — always pick best known action | Exploitation, evaluation |
| **ε-greedy** | With probability ε: random action; else: greedy | Balancing exploration vs exploitation |
| **Softmax/Boltzmann** | $\pi(a \mid s) \propto \exp(Q(s, a) / \tau)$ | Temperature-controlled exploration |

### Exploration vs Exploitation

This is the central dilemma of RL. The optimal policy requires knowing Q*(s,a) for ALL actions, but you can only learn about actions you take:

```
┌──────────────────────────────────────────────────────┐
│            EXPLORATION-EXPLOITATION TRADE-OFF         │
│                                                      │
│  Exploitation: Take the action with highest known Q  │
│  ─ Risk: Missing a better action you've never tried  │
│                                                      │
│  Exploration: Try an action with uncertain Q         │
│  ─ Risk: Losing immediate reward                     │
│                                                      │
│  ε-greedy solves this with a simple rule:            │
│  ┌────────────────────────────────────────────┐     │
│  │ ε = 1.0 at start  →  ε → 0.01 over time   │     │
│  │ Explore aggressively → Exploit increasingly │     │
│  └────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

---

## 5. 💻 A Minimal RL Agent

```python
import numpy as np
from collections import defaultdict

class BanditAgent:
    """Epsilon-greedy agent for multi-armed bandit (simplest RL problem)."""

    def __init__(self, n_actions, epsilon=0.1, alpha=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = defaultdict(float)   # Q(a) = estimated value of action a
        self.N = defaultdict(int)     # N(a) = how many times action a was taken

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return max(range(self.n_actions), key=lambda a: self.Q[a])  # Exploit

    def update(self, action, reward):
        self.N[action] += 1
        # Incremental mean update (sample average)
        self.Q[action] += self.alpha * (reward - self.Q[action])

# Usage
agent = BanditAgent(n_actions=5, epsilon=0.1, alpha=0.1)
true_rewards = [0.5, 0.7, 0.1, 0.9, 0.3]  # Unknown to agent

for episode in range(1000):
    action = agent.select_action()
    reward = np.random.binomial(1, true_rewards[action])  # Stochastic reward
    agent.update(action, reward)

print("Estimated action values:", [f"{agent.Q[a]:.3f}" for a in range(5)])
print("True action values:     ", true_rewards)
# Agent learns: action 3 (true mean 0.9) has the highest Q
```

---

## 6. 🌍 Real-World RL Foundations

| Company | RL Problem | MDP Components |
|---|---|---|
| **OpenAI (ChatGPT)** | Generate helpful, harmless responses | State = conversation, Action = next token, Reward = human preference |
| **DeepMind (AlphaFold)** | Protein folding as sequential decision | State = partial structure, Action = torsion angle, Reward = energy |
| **YouTube** | Video recommendation | State = user history, Action = recommended video, Reward = watch time |
| **Tesla (FSD)** | Autonomous driving | State = sensor fusion, Action = steering/accel, Reward = safety + progress |
| **Amazon** | Warehouse robot routing | State = position + inventory, Action = movement, Reward = efficiency |

---

## ⚠️ Pitfalls

- **Credit assignment problem:** A reward at time $t$ may be the result of an action at time $t-k$. Standard RL blames/rewards all actions equally. This slows learning in long-horizon tasks.
- **Reward hacking:** Agents optimize the reward function, not the intended objective. If your reward function has loopholes, the agent WILL find them.
- **State representation matters:** If your state doesn't encode critical information (e.g., velocity for a driving agent), the MDP becomes partially observable and standard RL fails.
- **Discount factor is a hyperparameter, not a detail:** $\gamma = 0.99$ vs $\gamma = 0.9$ fundamentally changes what the agent optimizes. Lower $\gamma$ makes the agent myopic.

---

## 💡 Tips

- **Start with bandits before full MDPs:** Multi-armed bandits are RL without state — just action → reward. They're the sanity check for your policy and exploration logic.
- **Log every component in the Bellman equation during debugging:** Log $R(s,a,s')$, $V(s')$, and the TD error $R + \gamma V(s') - V(s)$. When one term dominates, you've found your bug.
- **Use $\gamma < 1$ even in episodic tasks:** A discount factor provides a soft horizon that stabilizes training. $\gamma = 0.99$ is a safe default.

---

## 📦 Compression Code

```python
import numpy as np

class SimpleQLearning:
    """Tabular Q-Learning for discrete state/action spaces."""

    def __init__(self, n_states, n_actions, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

# Grid world example
env = np.zeros((5, 5))  # 5x5 grid, 25 states
env[4, 4] = 1.0          # Goal state reward
agent = SimpleQLearning(n_states=25, n_actions=4)

for ep in range(500):
    state = 0  # Top-left start
    for step in range(100):
        action = agent.act(state)
        # Simulate transition (simplified)
        next_state = min(24, state + [1, -1, 5, -5][action])
        reward = env[next_state // 5, next_state % 5]
        done = (next_state == 24)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

print("Q-table converged. Best action from state 0:", agent.act(0))
```

---

## ✅ Knowledge Check

1. **What is the difference between $V(s)$ and $Q(s, a)$?** — $V(s)$ is the expected return from state $s$ following policy $\pi$. $Q(s, a)$ is the expected return from taking action $a$ in state $s$ and then following $\pi$. Q-values are action-conditional; V-values average over the policy.

2. **Why does $\gamma < 1$ matter even in episodic tasks?** — A discount factor provides a soft horizon that stabilizes learning by reducing the variance of distant returns and ensuring the infinite sum converges. It also encodes a preference for sooner rewards.

3. **What problem does ε-greedy solve?** — The exploration-exploitation dilemma. Pure greed gets stuck in local optima (never tries potentially better actions). Pure random never exploits what it learns. ε-greedy balances both.

4. **What does the Markov property require of state representations?** — The state must contain all information relevant to future decisions. Adding history to the state is valid if the raw observation isn't Markov (e.g., including past 4 frames in Atari games).

---

## 🎯 Key Takeaways

- MDPs formalize RL: states, actions, transitions, rewards, and a discount factor.
- Value functions $V(s)$ and $Q(s,a)$ predict expected future return — the Bellman equation decomposes them recursively.
- Policies map states to actions; the exploration-exploitation trade-off is managed by ε-greedy, softmax, or more advanced strategies.
- Q-learning is the simplest off-policy RL algorithm: learn Q*(s,a) from any experience by bootstrapping from max Q in the next state.
- All modern RL — DQN, PPO, RLHF — builds on these MDP and value function foundations.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018), Chapters 1-6
- [OpenAI Spinning Up: RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [DeepMind Lecture Series: Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
