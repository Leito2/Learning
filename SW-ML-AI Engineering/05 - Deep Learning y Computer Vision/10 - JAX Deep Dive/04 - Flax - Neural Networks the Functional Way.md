# 🏷️ Flax — Neural Networks the Functional Way

## 🎯 Learning Objectives
- Understand the **philosophical split** between Flax and PyTorch: parameters OUTSIDE the module
- Master the **init/apply separation** — why defining a network takes two steps
- Build common architectures using **Flax Linen** (`nn.Module`, `@nn.compact`, `nn.Dense`, `nn.Conv`)
- Manage training state with **TrainState** — the functional alternative to `model.train()`
- Navigate the **Flax vs Haiku vs Equinox** landscape

## Introduction

If JAX is not a deep learning framework, then what do you use to build neural networks? **Flax** is the answer from Google Research — a neural network library for JAX that embodies the functional paradigm at every level. The core insight that separates Flax from PyTorch's `nn.Module` is elegantly simple: **the neural network is a pure function of parameters and input**. In PyTorch, `self.weight` lives inside the module as mutable state; in Flax, weights are passed *into* the module on every call and the module itself carries no state. This isn't just an implementation detail — it's a design philosophy that makes JAX's transformations work seamlessly.

The practical consequence is the **init/apply split**. Defining a Flax module produces an `init` function (which generates initial parameters) and an `apply` function (which runs the forward pass given parameters). At first this feels like unnecessary ceremony — why can't I just call `model(x)`? The answer reveals itself when you try to `jit` your model, `pmap` it across 8 TPU cores, or differentiate through it with `grad`. Because parameters are explicit arguments, JAX's transformations see them as just another input — the compiler can optimize the entire forward+backward pass as one fused XLA kernel. In PyTorch, `nn.Module` parameters are hidden state that the JIT compiler can't reason about, which limits optimization scope.

This note covers Flax Linen (the standard, Google-maintained API), with context on Haiku (DeepMind's alternative) and Equinox (the newer, more Pythonic approach). We'll build a 2-layer MLP, train it with Optax, see how `TrainState` bundles parameters + optimizer state, and understand the module hierarchy that scales to massive architectures like PaLM and Gemini. By the end, you'll see the init/apply split not as ceremony but as the design choice that enables JAX's composability.

---

## 1. Flax vs PyTorch nn.Module — The Philosophical Split

### 1.1 Where Parameters Live

```python
# ❌ PyTorch: parameters ARE the module (self-contained, stateful)
import torch.nn as nn
class PyTorchMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)    # Parameters stored HERE
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = PyTorchMLP(784, 256, 10)    # Parameters created and stored inside
output = model(x)                   # Output magically uses internal parameters

# ✅ Flax: parameters are EXTERNAL — module is stateless
import flax.linen as nn
class FlaxMLP(nn.Module):
    hidden: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)     # Parameters auto-created by @compact
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x

model = FlaxMLP(hidden=256, out_dim=10)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))  # params extracted!
output = model.apply(params, x)  # Pass params EXTERNALLY — stateless module
```

The analogy: PyTorch modules are **stateful objects** ($\text{model} : \text{Input} \to \text{Output}$). Flax modules are **parametrized functions** ($\text{model} : \text{Params} \times \text{Input} \to \text{Output}$). The module itself carries zero state — it's a pure function specification.

### 1.2 Why the Split Enables JAX Transformations

Because parameters are explicit arguments:

```python
# jit the entire forward pass (parameters included)
@jax.jit
def predict(params, x):
    return model.apply(params, x)

# grad through the model takes params as first argument
@jax.grad
def loss_fn(params, x, y):
    logits = model.apply(params, x)
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=-1))

# vmap over batch (model sees single examples, vmap handles batching)
batch_predict = jax.vmap(lambda p, x: model.apply(p, x), in_axes=(None, 0))
```

In PyTorch, you can't `torch.jit.script(jax.grad(f))` — the stateful module breaks the functional contract. In Flax, the stateful module breaks the functional contract. In Flax, parameters are just data — JAX can optimize them, parallelize over them, and differentiate through them as it would any other array.

> **💡 Tip:** Think of `model.apply(params, x)` as the functional equivalent of `model(x)` in PyTorch. The `apply` method takes parameters as the first positional argument (after `self`) by convention, making it the default target for `jax.grad`.

---

## 2. Flax Linen — The Module System

### 2.1 Basic Module Definition

```python
import flax.linen as nn
import jax.numpy as jnp
import jax

class SimpleMLP(nn.Module):
    """A clean 2-layer MLP in Flax Linen."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x
```

The `@nn.compact` decorator is Flax's "define-by-run" style: sub-modules and variables are created on the fly during the first call (tracing), and their names are auto-generated based on the call order. Without `@compact`, you'd need to manually create and track sub-modules in a `setup()` method:

```python
# Alternative: explicit setup() — more verbose but clearer
class ExplicitMLP(nn.Module):
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x
```

> **💡 Tip:** Use `@nn.compact` for research and rapid prototyping (less boilerplate). Use `setup()` for production code where parameter naming matters (e.g., checkpoint compatibility, weight loading from PyTorch checkpoints).

### 2.2 The init/apply Split

```python
# Step 1: Define the module
model = SimpleMLP(hidden_dim=256, output_dim=10)

# Step 2: Initialize — trace the module with dummy data to create parameters
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 784))
variables = model.init(key, dummy_input)

# ⚠️ The output of init is a FLAX-specific structure with separate 'params' and 'batch_stats' etc.
params = variables['params']
print(jax.tree_map(lambda x: x.shape, params))
# Output: {'Dense_0': {'kernel': (784, 256), 'bias': (256,)},
#          'Dense_1': {'kernel': (256, 10), 'bias': (10,)}}

# Step 3: Apply — run the forward pass with trained parameters
x_real = jnp.ones((32, 784))
logits = model.apply({'params': params}, x_real)
```

### 2.3 Common Layers in Flax Linen

```python
import flax.linen as nn

# Dense / Linear
nn.Dense(features=128, use_bias=True, kernel_init=nn.initializers.lecun_normal())

# Convolution
nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

# Attention
nn.MultiHeadDotProductAttention(
    num_heads=8, qkv_features=512, out_features=256,
    dropout_rate=0.1, deterministic=False
)

# Normalization
nn.LayerNorm()
nn.BatchNorm(use_running_average=False, momentum=0.9, axis_name='batch')
nn.GroupNorm(num_groups=32)

# Dropout (requires PRNG key!)
nn.Dropout(rate=0.5, deterministic=False)

# Embedding
nn.Embed(num_embeddings=10000, features=512)

# Recurrent
nn.GRUCell(features=128)
nn.LSTMCell(features=128)
nn.OptimizedLSTMCell(features=128)  # CuDNN-optimized on GPU

# Pooling
nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
```

### 2.4 Convolutional Network Example

```python
class SimpleCNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
```

> **¡Sorpresa!** `nn.Dropout` requires a `deterministic` argument AND a PRNG key. In Flax, the key is passed through the `rngs` argument in `model.apply({'params': params}, x, rngs={'dropout': dropout_key})`. Unlike PyTorch where `model.eval()` switches off dropout globally, Flax makes dropout/deterministic an explicit parameter — this is functional purity applied to stochastic layers.

---

## 3. TrainState — Managing State Functionally

### 3.1 What TrainState Bundles

`TrainState` is Flax's abstraction for bundling everything that changes during training into one PyTree:

```python
from flax.training import train_state
import optax

class TrainState(train_state.TrainState):
    """Custom TrainState — add metrics, EMA params, etc."""
    batch_stats: Any = None  # For BatchNorm running statistics

def create_train_state(key, model, input_shape, learning_rate=1e-3):
    params = model.init(key, jnp.ones(input_shape))['params']
    tx = optax.adamw(learning_rate, weight_decay=1e-4)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
```

The `TrainState` contains:
- `state.params`: the model parameters (PyTree)
- `state.opt_state`: the optimizer's internal state (Adam's m/v buffers, etc.)
- `state.apply_fn`: the model's forward function (for inference)
- `state.tx`: the optimizer (for applying gradients)
- `state.step`: the current training step counter

### 3.2 The Functional Training Step

```python
@jax.jit
def train_step(state, batch, dropout_key):
    """Single training step — pure function: state + batch → (new_state, metrics)"""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['image'],
            rngs={'dropout': dropout_key}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)  # Returns new TrainState — no mutation!
    accuracy = (logits.argmax(axis=-1) == batch['label']).mean()
    return state, {'loss': loss, 'accuracy': accuracy}
```

Notice: `state.apply_gradients(grads=grads)` returns a **new TrainState object**. The original `state` is not modified — this is functional purity in action.

> **¡Sorpresa!** Even `state.apply_gradients` is a pure function. The optimizer's internal state (Adam's m and v buffers) is **returned** as part of the new TrainState, not mutated in-place. This means you can keep checkpoints of arbitrary training steps for analysis without worrying about side effects.

### 3.3 Handling BatchNorm and Other Stateful Layers

BatchNorm maintains running mean/variance, which is technically mutable state. Flax stores this in a separate variable collection:

```python
class CNNWithBN(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return x

# Initialize returns params AND batch_stats
variables = model.init(key, x, train=True)
params = variables['params']
batch_stats = variables['batch_stats']

# Forward pass returns new batch_stats during training
output, new_batch_stats = model.apply(
    {'params': params, 'batch_stats': batch_stats},
    x, train=True,
    mutable=['batch_stats']  # ⚠️ Must mark batch_stats as mutable!
)
```

> **⚠️ Warning:** For BatchNorm in Flax, you must explicitly manage `batch_stats` in your TrainState and update it after every training step. This is more verbose than PyTorch's automatic handling but makes state flow explicit.

---

## 4. The Flax / Haiku / Equinox Landscape

### 4.1 Flax Linen (Google)

- **Maintained by**: Google Research
- **Design philosophy**: Explicit parameters, init/apply split, `@nn.compact` for convenience
- **Strengths**: Most mature, best documentation, used in production by Google (PaLM, Gemini), excellent TPU support
- **Weaknesses**: init/apply split is verbose for simple experiments, `@compact` magic can be confusing
- **Best for**: Production models, TPU training, research that needs to scale

### 4.2 Haiku (DeepMind)

- **Maintained by**: DeepMind
- **Design philosophy**: Parameter state is implicitly managed by `hk.transform`, more PyTorch-like
- **Key difference from Flax**: `hk.transform(f)` converts a stateful function into a pure one automatically
- **Strengths**: Feels more natural coming from PyTorch, DeepMind's research ecosystem
- **Weaknesses**: Simpler API means less control, less third-party tooling
- **Best for**: Quick experimentation, PyTorch migrants who want minimal friction

### 4.3 Equinox (Community / Patrick Kidger)

- **Maintained by**: Community (Patrick Kidger at Google)
- **Design philosophy**: Modules ARE PyTrees — no init/apply split, just `__call__`
- **Key difference**: An Equinox module is directly a PyTree; parameters are fields of the module (like PyTorch) but the module is immutable (like JAX)
- **Strengths**: Feels exactly like PyTorch, zero boilerplate, excellent for research
- **Weaknesses**: Newer, smaller ecosystem, fewer pre-built layers
- **Best for**: Research code, small-to-medium scale projects, PyTorch-like ergonomics

```python
# Equinox example — looks almost like PyTorch!
import equinox as eqx

class EQMLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(784, 256, key=k1)
        self.fc2 = eqx.nn.Linear(256, 10, key=k2)

    def __call__(self, x):
        x = jax.nn.relu(self.fc1(x))
        return self.fc2(x)

# Parameters are fields — no init/apply split!
# But the module is immutable — you use jax.tree_map to "update" it
```

---

## 5. Building a Complete Model in Flax

### 5.1 The Full Pattern: Define → Init → Train → Evaluate

```python
import flax.linen as nn
import jax, jax.numpy as jnp, optax
from flax.training import train_state

class Classifier(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.3, deterministic=not train)(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

# Create state
rng = jax.random.PRNGKey(0)
model = Classifier(num_classes=10)
params = model.init(rng, jnp.ones((1, 784)), train=False)['params']
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(1e-3),
)

# Training — jit the whole step
@jax.jit
def step(state, batch, key):
    def loss_fn(p):
        logits = state.apply_fn({'params': p}, batch['x'], train=True,
                                 rngs={'dropout': key})
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']))
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
```

> **Caso real: Google's PaLM-540B** is implemented entirely in Flax Linen. The model definition uses `@nn.compact` with custom attention blocks, and the training infrastructure passes parameters across 6144 TPU cores using `jax.pmap` with custom sharding. The functional design means the model code doesn't change between single-GPU debugging and 6144-TPU training — only the outermost `pmap` call differs.

---

## 🎯 Key Takeaways
- **Flax modules are pure functions**: parameters are passed in externally via `model.apply(params, x)`. The module itself is stateless — this is the enabling design choice for `jit`, `vmap`, `grad`, and `pmap`.
- **The init/apply split** is not ceremony — it separates parameter creation from forward computation, letting JAX see parameters as data for optimization.
- **`@nn.compact`** auto-generates parameter names during the first trace, eliminating boilerplate for research code. Use `setup()` for production when explicit naming matters.
- **TrainState** bundles parameters, optimizer state, and step counter into a single PyTree managed by pure functional updates — no `model.train()`, no hidden state.
- **BatchNorm and Dropout** require explicit PRNG keys and manual state tracking. This is the price of functional purity; it's verbose but transparent.
- **Flax vs Haiku vs Equinox**: Flax is the production-grade option (Google's choice), Haiku is DeepMind's more PyTorch-like alternative, Equinox is the elegant newcomer with zero boilerplate.
- The functional design means your model code scales from a single GPU to thousands of TPU cores **without changes** — only the outermost `pmap` call differs.

## 📦 Código de Compresión

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# 1. Define the model (stateless specification)
class SimpleNN(nn.Module):
    """Two-layer MLP — just a function specification."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# 2. Initialize (create parameters)
model = SimpleNN()
key = jax.random.PRNGKey(42)
params = model.init(key, jnp.ones((1, 784)))['params']

# 3. TrainState (bundle params + optimizer)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(1e-3),
)

# 4. Training step (pure function, jit-compiled)
@jax.jit
def train_step(state, x, y):
    def loss_fn(p):
        logits = state.apply_fn({'params': p}, x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

# 5. Run one step
x_batch = jnp.ones((32, 784))
y_batch = jnp.ones(32, dtype=jnp.int32)
new_state, loss = train_step(state, x_batch, y_batch)
print(f"Loss: {loss:.4f}")
print("¡Sorpresa! Model is stateless — only params carry state. No model.train() needed.")
```

## References
- Flax Documentation: https://flax.readthedocs.io/
- Heek et al. (2023). "Flax: A neural network library and ecosystem for JAX." GitHub.
- DeepMind (2023). "Using Haiku + JAX for Research."
- Kidger & Garcia (2021). "Equinox: neural networks in JAX via callable PyTrees and filtered transformations." *NeurIPS Workshop*.
- Chowdhery et al. (2022). "PaLM: Scaling Language Modeling with Pathways." *arXiv:2204.02311*.
- [[05/03 - Deep Learning con PyTorch]]
- [[05/09 - Deep Learning with TensorFlow]]
- [[07/32 - Advanced ML Topics]]
- [[06/16 - HuggingFace Transformers Deep Dive]]
