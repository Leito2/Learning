# ⚡ PPO and Policy Gradient Methods

## Introduction

DQN works beautifully for discrete action spaces — but many real-world problems demand continuous actions: how many degrees to turn a steering wheel, what temperature and top_p to use in LLM generation, how much to bid in an ad auction. Policy gradient methods solve this by directly optimizing the policy parameters $\theta$ to maximize expected return, without ever computing Q-values.

PPO (Proximal Policy Optimization) emerged as the default policy gradient algorithm for both research and production. It is the engine behind RLHF (training ChatGPT to be helpful and harmless), game-playing agents (OpenAI Five), and robotic control systems. This module covers the policy gradient theorem, the variance reduction problem, and why PPO's clipped objective is the pragmatic sweet spot between simplicity and stability.

---

## 1. 🧠 Why Policy Gradients?

Value-based methods (DQN) learn $Q(s,a)$ implicitly and derive the policy: $\pi(s) = \argmax_a Q(s,a)$. Policy gradient methods parameterize the policy directly: $\pi_\theta(a \mid s)$ and optimize $\theta$ to maximize expected return.

### Value-Based vs Policy-Based

| Aspect | Value-Based (DQN) | Policy-Based (PPO) |
|---|---|---|
| **Output** | Q-values → derived policy | Direct policy: $\pi_\theta(a \mid s)$ |
| **Action space** | Discrete only | Discrete + continuous |
| **Policy type** | Deterministic (argmax Q) | Stochastic (distribution over actions) |
| **Convergence** | Oscillatory (greedy on noisy Q) | Smooth (direct policy optimization) |
| **Exploration** | ε-greedy (external) | Entropy bonus (internal) |
| **Stability** | Sensitive to hyperparameters | More stable (clipped updates) |
| **Sample efficiency** | High (replay buffer, off-policy) | Lower (on-policy) |
| **Use in RLHF** | Not used | PPO is the standard |

---

## 2. 📐 The Policy Gradient Theorem

The objective: maximize expected return from the start state:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t R_t \right]
$$

Where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ is a trajectory sampled from $\pi_\theta$.

The Policy Gradient Theorem gives us the gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

**Intuition:** Increase the log-probability of actions that led to high returns (positive $G_t$), decrease for low returns. The log-probability gradient $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ points in the direction that makes action $a_t$ more likely under the current policy.

### The REINFORCE Algorithm

The simplest policy gradient algorithm — Monte Carlo sampling of complete trajectories:

```python
# REINFORCE pseudo-code
for episode in range(N):
    tau = collect_trajectory(pi_theta)    # Run policy until terminal
    G = compute_returns(tau)              # G_t = sum gamma^k * r_{t+k}
    loss = -sum(log_prob(a_t|s_t) * G_t)  # Negative for gradient ascent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

REINFORCE has high variance — the return $G_t$ varies wildly between episodes. This is why raw REINFORCE is a teaching tool, not a production algorithm.

---

## 3. ⚡ Variance Reduction: Actor-Critic and GAE

### The Actor-Critic Architecture

Instead of using the raw return $G_t$ (high variance), use the **advantage function** $A(s, a)$:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

The advantage answers: "how much better is action $a$ compared to the average action in state $s$?"

```mermaid
graph TB
    subgraph "Actor-Critic Architecture"
        A[State s] --> B[Actor Network π_θ<br/>Outputs action distribution]
        A --> C[Critic Network V_φ<br/>Outputs state value]
        B --> D[Sample action a ~ π_θ(a|s)]
        D --> E[Environment]
        E --> F[Reward r, Next state s']
        C --> G[Compute advantage:<br/>A = r + γV(s') - V(s)]
        G --> H[Update Actor:<br/>θ ← θ + α∇log π(a|s)·A]
        F --> I[Update Critic:<br/>φ ← φ - β∇(V(s) - G)²]
    end
```

### Generalized Advantage Estimation (GAE)

GAE computes a multi-step advantage estimate that balances bias and variance:

$$
\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

| Parameter | $\lambda = 0$ | $\lambda = 1$ |
|---|---|---|
| Advantage = | TD(0): one-step TD error | Monte Carlo: full return minus baseline |
| Bias | High (biased by V function) | Low (uses actual returns) |
| Variance | Low (one reward step) | High (full trajectory) |
| Best for | When your value function is accurate | When your value function is poor |

Typical GAE $\lambda = 0.95$ — gives most of the bias reduction with most of the variance control.

---

## 4. 🔒 PPO: The Clipped Objective

PPO introduced a simple but powerful idea: don't let the policy change too much in a single update. This prevents the catastrophic collapse that plagued earlier policy gradient methods (TRPO was the predecessor — mathematically elegant but computationally complex).

### The PPO-Clip Objective

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio between the new and old policy.

### How the Clip Works

```
┌──────────────────────────────────────────────────────────────┐
│                     PPO CLIP MECHANISM                        │
│                                                              │
│  r(θ) = π_new(a|s) / π_old(a|s)   ← Probability ratio        │
│                                                              │
│  When Advantage > 0 (action was GOOD):                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Increase probability, but CLIP at 1 + ε              │   │
│  │ L = min(r·A, clip(r, 1-ε, 1+ε)·A)                   │   │
│  │ → If r > 1+ε: L = (1+ε)·A  (clip stops overshoot)   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  When Advantage < 0 (action was BAD):                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Decrease probability, but CLIP at 1 - ε              │   │
│  │ L = min(r·A, clip(r, 1-ε, 1+ε)·A)                   │   │
│  │ → If r < 1-ε: L = (1-ε)·A  (clip stops overshoot)   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

The `min()` with the clipped version means PPO is **pessimistic**: it ignores changes that would make the objective look better if those changes come from an excessive probability ratio. This conservatism is what makes PPO stable.

### PPO Hyperparameters

| Parameter | Typical Value | Effect |
|---|---|---|
| **$\epsilon$ (clip range)** | 0.1 — 0.3 | Smaller = more conservative updates |
| **GAE $\lambda$** | 0.95 | Higher = lower bias, higher variance |
| **$\gamma$ (discount)** | 0.99 | Standard RL discount |
| **Epochs per batch** | 3 — 10 | More epochs = more policy updates per batch |
| **Mini-batch size** | 64 — 256 | Larger = more stable gradients |
| **Entropy coefficient** | 0.01 | Higher = more exploration |
| **Value loss coefficient** | 0.5 | Weight between actor and critic loss |

---

## 5. 💻 PPO Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    """Shared backbone with separate actor and critic heads."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: outputs mean of Gaussian for continuous actions
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic: outputs scalar value V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        action_mean = torch.tanh(self.actor_mean(features))  # Bound to [-1, 1]
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        value = self.critic(features)
        return action_mean, action_std, value

    def act(self, state, deterministic=False):
        with torch.no_grad():
            mean, std, value = self.forward(state.unsqueeze(0))
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
        return action.squeeze(0).numpy(), log_prob.item(), value.item()


class PPOBuffer:
    """Stores trajectories for on-policy PPO updates."""

    def __init__(self, gamma=0.99, lam=0.95):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []
        self.gamma, self.lam = gamma, lam

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value):
        """Compute GAE advantages and returns."""
        rewards = self.rewards + [0]  # Pad for last step
        values = self.values + [last_value]
        dones = self.dones + [False]

        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def get_batch(self):
        return (np.array(self.states), np.array(self.actions),
                np.array(self.log_probs))

    def clear(self):
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()


def ppo_update(ac, optimizer, buffer, clip_eps=0.2, epochs=10, batch_size=64,
               entropy_coef=0.01, value_coef=0.5):
    """PPO update with clipped objective."""

    states, actions, old_log_probs = buffer.get_batch()
    advantages, returns = buffer.compute_gae(last_value=0)

    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)

    # Normalize advantages (stabilizes training)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(
        states, actions, old_log_probs, advantages, returns
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(epochs):
        for s_batch, a_batch, lp_batch, adv_batch, ret_batch in loader:
            mean, std, value = ac(s_batch)

            # Current log probs
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(a_batch).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # Probability ratio
            ratio = torch.exp(new_log_probs - lp_batch)

            # PPO clipped objective
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(value.squeeze(), ret_batch)

            # Combined loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)
```

---

## 6. 🎯 PPO vs Alternatives

| Algorithm | Year | Key Idea | Best For |
|---|---|---|---|
| **REINFORCE** | 1992 | Monte Carlo policy gradients | Teaching tool, simple environments |
| **A2C/A3C** | 2016 | Actor-critic with advantage, async | Discrete games, moderate complexity |
| **TRPO** | 2015 | Trust region (KL divergence constraint) | When stability is critical |
| **PPO** | 2017 | Clipped objective | Default choice — stability + simplicity |
| **SAC** | 2018 | Maximum entropy + off-policy | Continuous control, robotics |
| **DDPG/TD3** | 2015/2018 | DPG + off-policy + twin critics | Deterministic continuous control |

**Why PPO won:** TRPO achieves better theoretical guarantees but requires computing the Hessian of KL divergence (complex, fragile). PPO approximates the trust region with a simple clip operation that is trivial to implement and tune. In practice, PPO matches or exceeds TRPO performance on most benchmarks.

---

## 7. 🌍 PPO in Production

| Application | Company | PPO Detail |
|---|---|---|
| **ChatGPT alignment** | OpenAI | PPO with KL penalty to reference model prevents reward hacking |
| **Claude alignment** | Anthropic | PPO + constitutional AI for harmlessness training |
| **OpenAI Five (Dota 2)** | OpenAI | PPO with 256K CPUs and 128K CPU cores, self-play |
| **Dexterous robotic hand** | OpenAI | PPO with domain randomization for sim-to-real transfer |
| **YouTube Shorts recommendation** | Google | PPO for content ranking with user satisfaction reward |
| **Trade execution** | JPMorgan | PPO for optimal order slicing in dark pools |

---

## ⚠️ Pitfalls

- **On-policy data is sample-inefficient:** PPO discards data after each update. For problems where data collection is expensive, consider off-policy methods (SAC).
- **Entropy collapse:** Without an entropy bonus (or with too small a coefficient), the policy can converge to a deterministic strategy too early, never exploring alternatives. Monitor entropy — if it drops to near zero in early training, increase the coefficient.
- **Reward scale sensitivity:** PPO is sensitive to reward magnitude. Advantages that range from -1000 to +1000 cause gradient spikes. Normalize advantages or clip rewards to [-1, 1].
- **Hyperparameter coupling:** PPO's performance depends on the interaction of clip range, learning rate, and epochs per batch. Grid search over one parameter at a time is insufficient — these parameters are coupled.

---

## 💡 Tips

- **Use advantage normalization:** `advantages = (advantages - mean) / (std + 1e-8)` — this single line often improves PPO performance more than any hyperparameter tune.
- **Start with PPO defaults from stable-baselines3/RLHF papers:** ε=0.2, λ=0.95, γ=0.99, 4 epochs, 64 minibatches. These "just work" for most problems.
- **Early stopping on KL divergence:** In RLHF, stop PPO updates if KL(π_θ || π_ref) exceeds a threshold. This prevents the policy from diverging too far from the supervised fine-tuned model.

---

## 📦 Compression Code

```python
import torch
import gymnasium as gym
from torch.optim import Adam

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ac = ActorCritic(state_dim, action_dim)
optimizer = Adam(ac.parameters(), lr=3e-4)
buffer = PPOBuffer()

for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(200):
        action, log_prob, value = ac.act(torch.FloatTensor(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.store(state, action, log_prob, reward, value, terminated or truncated)
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break

    p_loss, v_loss, ent = ppo_update(ac, optimizer, buffer)
    buffer.clear()

    if episode % 100 == 0:
        print(f"Ep {episode:4d} | R: {total_reward:7.1f} | "
              f"P_loss: {p_loss:.3f} | V_loss: {v_loss:.3f} | Ent: {ent:.3f}")
```

---

## ✅ Knowledge Check

1. **Why do policy gradient methods have high variance?** — The return $G_t$ sums many stochastic rewards across an entire trajectory. Two trajectories from the same policy can have wildly different returns due to environment stochasticity and action sampling noise.

2. **How does GAE reduce variance over simple TD(0)?** — GAE uses a weighted sum of multi-step TD errors (λ controls the bias-variance trade-off). λ=1 gives Monte Carlo returns (low bias, high variance); λ=0 gives TD(0) (high bias, low variance); λ=0.95 gets most of the variance reduction with most of the bias reduction.

3. **What does PPO's clip do when advantage is positive and the policy wants to increase action probability sharply?** — The clip caps the probability ratio at 1+ε. If the new policy would be > (1+ε)× the old policy's probability, the objective ignores the "extra" improvement — preventing an excessively large update that could destabilize training.

4. **Why is PPO the standard for RLHF rather than DQN?** — RLHF requires a stochastic policy (the LLM outputs a probability distribution over tokens) and must constrain policy updates to not diverge from the reference model. PPO's clipped objective naturally handles both — DQN is deterministic and offers no update constraint mechanism.

---

## 🎯 Key Takeaways

- Policy gradient methods optimize the policy directly via $\nabla_\theta \log \pi_\theta(a \mid s) \cdot A(s,a)$.
- Actor-Critic architecture reduces variance: actor outputs actions, critic evaluates them via advantage $A(s,a) = Q(s,a) - V(s)$.
- GAE provides a controlled bias-variance trade-off through the λ parameter (λ=0.95 standard).
- PPO's clipped objective `min(r·A, clip(r, 1-ε, 1+ε)·A)` prevents destructive policy updates with simple, implementable math.
- PPO is the RLHF workhorse because it supports stochastic policies and naturally constrains divergence from a reference model.

---

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv, 2017)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (ICLR, 2016)
- Schulman et al., "Trust Region Policy Optimization" (ICML, 2015)
- Ouyang et al., "Training language models to follow instructions with human feedback" (NeurIPS, 2022)
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
