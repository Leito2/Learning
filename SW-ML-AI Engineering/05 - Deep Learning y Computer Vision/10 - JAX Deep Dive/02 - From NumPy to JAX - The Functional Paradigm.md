# 🏷️ From NumPy to JAX — The Functional Paradigm

## 🎯 Learning Objectives
- Master the **NumPy facade** in JAX — 90%+ API compatibility with critical behavioral differences
- Internalize **immutability** as a feature, not a limitation: why `x[i] = v` is forbidden
- Understand the **PRNG key system** and why global random state is incompatible with functional purity
- Grasp the **PyTree abstraction** — how JAX represents structured parameters across any nested container
- Recognize **functional purity** as the enabler of `jit`, `vmap`, `pmap`, and composability

## Introduction

Open a JAX notebook, type `import jax.numpy as jnp`, and write `jnp.array([1, 2, 3])`. It feels exactly like NumPy. You'll write `jnp.dot`, `jnp.sum`, `jnp.sin`, `jnp.reshape`, `jnp.concatenate` — the API is so close that experienced NumPy users are productive in minutes. This is intentional: Google designed `jax.numpy` as a **drop-in facade** over the NumPy API to minimize the learning curve. But beneath this familiar surface, **JAX is not NumPy**. Every operation builds a computation graph destined for XLA; every array is immutable; every random number requires an explicit key. The gap between "looks like NumPy" and "thinks like Haskell" is where most JAX beginners stumble — and where this note focuses.

The mental model shift is profound: in NumPy/PyTorch, arrays are **mutable containers of numbers** that you manipulate step by step. In JAX, arrays are **immutable representations of mathematical values** in a pure function. This isn't a quirk — it's the prerequisite for XLA to fuse operations, for `jit` to compile entire functions, for `vmap` to reason about batching semantics, and for `pmap` to safely distribute computation. Every time you try `x[0] = 5` and JAX refuses, you're encountering the boundary between imperative programming (do this, then that) and functional programming (define a transformation of values). Crossing this boundary is the single most important skill in JAX mastery.

This note is about building that mental model. We'll systematically work through the NumPy-to-JAX transition, the immutability constraint and the `.at[...].set(...)` syntax that replaces in-place updates, the explicit PRNG key system that replaces global random state, the PyTree abstraction that unifies parameter management, and the practical patterns for writing correct, efficient JAX code. If Note 01 was about the four transformations, this note is about the **philosophy that makes them possible**.

---

## 1. The NumPy Facade — Surface Familiarity, Deep Difference

### 1.1 What Works (Most of It)

```python
import jax.numpy as jnp

# These are all identical to NumPy:
a = jnp.array([1, 2, 3, 4])
b = jnp.zeros((3, 3))
c = jnp.ones((5,))
d = jnp.arange(10.0)
e = jnp.linspace(0, 1, 100)
f = jnp.eye(4)
g = jnp.stack([a, a * 2])
h = jnp.concatenate([a, a * 3])

# Math operations: identical
result = jnp.dot(a, a.T)
result = jnp.sin(a) + jnp.cos(a)
result = jnp.sum(a, axis=0)
result = jnp.mean(a, keepdims=True)
result = jnp.where(a > 2, a, 0)

# Linear algebra: identical
U, S, Vh = jnp.linalg.svd(jnp.eye(4))

# Indexing (almost identical — see caveats below)
subset = a[1:3]
bool_mask = a[a > 2]
```

The `jax.numpy` module implements **approximately 95%** of the NumPy API. Functions like `jnp.einsum`, `jnp.matmul`, `jnp.fft`, and `jnp.linalg` all work with near-identical signatures.

### 1.2 What's Different — The Critical Gaps

**1. In-place mutation is forbidden:**

```python
# ✅ NumPy / PyTorch
import numpy as np
x = np.array([1, 2, 3])
x[0] = 10                     # Works — x is now [10, 2, 3]
x.sort()                      # Works — x sorted in-place
x += 1                        # Works — increments in-place

# ❌ JAX equivalents ALL fail
x = jnp.array([1, 2, 3])
x[0] = 10                     # ❌ TypeError: JAX arrays are immutable!
x.sort()                      # ❌ AttributeError: no sort method
x += 1                        # ❌ TypeError: arrays are immutable!
```

**2. Out-of-bounds indexing:**

```python
# NumPy: clips or errors depending on mode
np.array([1, 2, 3])[5]        # IndexError: out of bounds

# JAX: silently returns 0 for out-of-bounds (by default, like NumPy indexing with mode='clip')
# This behavior can cause silent bugs — use jnp.ndarray.at with mode='drop' or 'promise_in_bounds'
```

> **⚠️ Warning:** JAX's default out-of-bounds behavior differs from NumPy in edge cases with scatter/gather operations. Always test boundary conditions when porting NumPy code to JAX.

**3. Type promotion differences:**

```python
# NumPy: type promotion follows NumPy rules
np.array([1, 2, 3]) + 1.5     # → float64

# JAX: follows JAX type promotion (slightly different for bfloat16, mixed precision)
# JAX defaults to float32 on GPU, NumPy defaults to float64 on CPU
```

> **💡 Tip:** Enable `jax.config.update("jax_enable_x64", True)` at the start of your script to get NumPy-compatible float64 precision. Without this, JAX defaults to float32 on GPU/TPU for performance.

### 1.3 The `jax.numpy` Implementation Architecture

When you call `jnp.sum(x)`, the execution path is:

```
jnp.sum(x) → jax.lax.reduce_sum_p (JAX primitive) → XLA Reduce op → GPU kernel
```

The `jax.numpy` module is a thin wrapper over `jax.lax` (the low-level XLA operations library). `jax.numpy` provides the familiar NumPy API; `jax.lax` provides the XLA-native operations. Understanding this split helps when you need lower-level control (e.g., custom fused operations via `jax.lax`).

> **¡Sorpresa!** `jnp.sum(x)` may produce **slightly different results** on CPU vs GPU vs TPU because XLA uses different reduction algorithms (tree reduction on GPU, sequential on CPU). The difference is typically within floating-point epsilon, but for exact reproducibility across devices, use `jax.lax.precision` parameter.

---

## 2. Immutability — The Constraint That Enables Everything

### 2.1 Why Immutability?

In PyTorch, you can write:

```python
x = torch.randn(100)
x[5] = 0.0                # Mutation
x.add_(1.0)               # In-place addition
x.relu_()                 # In-place ReLU
```

Why does JAX forbid this? Because XLA needs to reason about the **entire computation** to perform optimizations. If `x` could change at any moment, the compiler can't fuse operations — there might be a dependency it can't see. Immutability guarantees that the output of any operation depends **only** on its inputs, which makes the computation graph a pure DAG (Directed Acyclic Graph) amenable to aggressive optimization.

Mathematically: every JAX array represents a value $v \in \mathbb{R}^{d_1 \times d_2 \times \cdots}$ that **cannot change**. An operation $\text{op}(v, w)$ produces a **new** value $u$, leaving $v$ and $w$ unchanged:

$$u = f(v, w), \quad v \text{ unchanged}, \quad w \text{ unchanged}$$

### 2.2 The `.at[...].set(...)` Replacement

To "update" an array, JAX provides the functional `.at` syntax that returns a **new** array:

```python
import jax.numpy as jnp

# ❌ PyTorch / NumPy
x = np.array([1, 2, 3, 4, 5])
x[2] = 99
x[0:2] = [10, 20]

# ✅ JAX functional equivalent
x = jnp.array([1, 2, 3, 4, 5])
x = x.at[2].set(99)              # ⚠️ THE OVERWRITTEN ARRAY IS RETURNED
x = x.at[0:2].set(jnp.array([10, 20]))
x = x.at[x > 3].set(0)           # Boolean mask indexing works!
x = x.at[1].add(5)               # Atomic add: x[1] += 5
x = x.at[3].multiply(2)          # Atomic multiply: x[3] *= 2
x = x.at[4].min(3)               # Element-wise min
x = x.at[4].max(0)               # Element-wise max
```

> **⚠️ Warning:** The `.at[...].set(...)` syntax **returns a new array**. Forgetting to assign the result (`x = x.at[...].set(...)`) silently discards the update. This is one of the most common JAX bugs for ex-PyTorch users.

**Performance note:** Inside a `@jax.jit`-compiled function, `.at[...].set(...)` is compiled by XLA into an efficient scatter operation — it does NOT copy the entire array. Outside `jit`, it does create a copy (but JAX arrays are typically small enough that this is negligible).

### 2.3 Full Comparison: In-Place vs Functional

```python
# ❌ PyTorch-style in-place mutation (stateful)
import torch
def pytorch_update(params, i, value):
    params[i] = value          # Mutates params — side effect!
    return params              # Same object, mutated

# ✅ JAX functional update (pure)
import jax.numpy as jnp
def jax_update(params, i, value):
    return params.at[i].set(value)  # Returns NEW array — no side effects

# Test immutability
x = jnp.array([1.0, 2.0, 3.0])
y = jax_update(x, 0, 10.0)
print(id(x) == id(y))          # ¡Sorpresa! False — different objects
print(x)                       # [1.0, 2.0, 3.0] — unchanged!
print(y)                       # [10.0, 2.0, 3.0] — new array
```

> **💡 Tip:** For updating model parameters during training, you'll use `optax.apply_updates(params, updates)` which handles this functionally. You rarely need `.at[...].set(...)` for training — it's more common in data preprocessing and custom layer implementations.

---

## 3. PRNG Keys — Functional Randomness

### 3.1 The Problem with Global Random State

Most frameworks use a global random state:

```python
# ❌ NumPy/PyTorch: global random state
import numpy as np
np.random.seed(42)
a = np.random.randn(100)     # Draws from global state
b = np.random.randn(100)     # State advances invisibly
# Problem 1: Non-reproducible in parallel
# Problem 2: Can't be jit-compiled (side effect on global state)
# Problem 3: Thread-unsafe
```

In JAX, randomness is a **resource** managed through explicit PRNG keys. Every random operation consumes a key and produces a new key:

```python
import jax
import jax.numpy as jnp

# ✅ JAX: explicit PRNG keys
key = jax.random.PRNGKey(42)          # Seed the generator
key, subkey1 = jax.random.split(key)  # Split into two independent keys
key, subkey2 = jax.random.split(key)
```

This is pure: `split(key)` is a deterministic function. Given the same input key, it always produces the same two output keys. There is no global state — everything is explicit.

### 3.2 The Key-Splitting Pattern

The standard pattern for every JAX program:

```python
def create_and_initialize(key):
    # Split for model init and data generation
    key_params, key_data, key_noise = jax.random.split(key, num=3)

    params = init_network(key_params, input_shape)
    data = jax.random.normal(key_data, (1000, 784))
    noise = jax.random.uniform(key_noise, (1000, 10))

    return params, data, noise, key  # Return unused subkeys
```

> **⚠️ Warning:** Never reuse the same PRNG key for multiple random operations! Two calls with the same key produce **identical** "random" outputs. If you call `jax.random.normal(key, (10,))` twice with the same key, you get the same tensor both times — because the function is pure and deterministic. **Always split before using.**

### 3.3 PyTorch Global RNG vs JAX Key Splitting — Side by Side

```python
# ❌ PyTorch: global RNG — implicit, non-composable
import torch
torch.manual_seed(42)
w1 = torch.randn(64, 10)     # State advances
w2 = torch.randn(10, 1)      # State advances
# Question: was the same seed used? Depends on call order.
# In parallel contexts: nondeterministic.

# ✅ JAX: explicit keys — pure, composable, deterministic
import jax; import jax.random as jr
root_key = jr.PRNGKey(42)
k1, k2 = jr.split(root_key)
w1 = jr.normal(k1, (64, 10))  # Explicitly uses k1
w2 = jr.normal(k2, (10, 1))   # Explicitly uses k2
# Same root_key → always same w1, w2. Parallel-safe by construction.
```

> **¡Sorpresa!** `jax.random.PRNGKey(42)` produces a **key array** (shape `(2,)`, dtype `uint32`), not an RNG object. The key is literally a 2-element array of integers that you can save, inspect, and serialize. `print(key)` outputs `[0 42]` — it's just data.

### 3.4 Keys in vmap and pmap Contexts

```python
# vmap: each batch element gets its own key
keys = jax.random.split(key, num=1000)
x = jax.vmap(jax.random.normal)(keys, ...)

# In training: fold key through scan
@jax.jit
def train_step(state, batch):
    key, subkey = jax.random.split(state.key)
    # Use subkey for dropout, data augmentation, etc.
    new_state = state.replace(key=key, ...)
    return new_state, loss
```

> **Caso real: Google DeepMind's training infrastructure** carries a PRNG key through every training step for all stochastic operations (dropout, data augmentation, label smoothing). This guarantees full reproducibility even in distributed settings — every TPU core gets its own key chain derived from a root key, so the same experiment produces identical results on every run.

---

## 4. PyTrees — JAX's Universal Data Structure

### 4.1 What is a PyTree?

A **PyTree** is any nested structure of Python containers (dicts, lists, tuples, namedtuples, dataclasses, custom classes registered via `jax.tree_util`) whose leaves are JAX arrays. PyTrees are JAX's way of representing structured parameters without imposing a specific container format.

```python
# All of these are valid PyTrees:
params_dict = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array(0.5)}
params_list = [jnp.array([1.0, 2.0]), jnp.array(0.5)]
params_tuple = (jnp.array([1.0, 2.0]), jnp.array(0.5))
params_nested = {
    'layer1': {'w': jnp.eye(10), 'b': jnp.zeros(10)},
    'layer2': {'w': jnp.eye(5), 'b': jnp.zeros(5)},
}

# All PyTrees can be traversed with identical APIs:
flat, tree_def = jax.tree_util.tree_flatten(params_nested)
# → flat: [eye(10), zeros(10), eye(5), zeros(5)]
# → tree_def: recipe to reconstruct the nested structure
```

### 4.2 PyTree Operations

```python
import jax

# Map a function over every leaf in a PyTree
def add_one(x):
    return x + 1.0

new_params = jax.tree_util.tree_map(add_one, params_nested)
# Every array in params_nested has 1.0 added

# Sum two PyTrees element-wise (matching structure required)
combined = jax.tree_util.tree_map(lambda a, b: a + b, params_v1, params_v2)

# Flatten and unflatten for serialization or checkpointing
leaves, treedef = jax.tree_util.tree_flatten(params_nested)
# Save leaves to disk, reconstruct with treedef later
```

### 4.3 Why PyTrees Matter for Deep Learning

**Model parameters are PyTrees.** When you define a Flax module, its parameters are automatically structured as a PyTree mirroring the module hierarchy. `jax.grad` automatically traverses input PyTrees — if your loss function takes a parameter dict, `grad(loss_fn)(params)` returns a dict with the same structure containing gradients at each leaf.

```python
params = {'w': jnp.array([1.0, 2.0, 3.0]), 'b': jnp.array(0.5)}

def loss(params, x, y):
    return jnp.sum((jnp.dot(x, params['w']) + params['b'] - y) ** 2)

grads = jax.grad(loss)(params, x_batch, y_batch)
# grads = {'w': jnp.array([...]), 'b': jnp.array([...])}  # Same structure!
```

> **¡Sorpresa!** The PyTree abstraction extends to optimizer states, metrics, and any nested container. `optax` optimizers accept and produce PyTrees. `jax.tree_map` works on any registered container. This unified abstraction is why JAX code is so concise — you never write custom parameter-unpacking logic.

---

## 5. Functional Patterns in Practice

### 5.1 The `lax.scan` Pattern — Functional Loops

When you need to iterate inside a jitted function, use `jax.lax.scan` (the functional equivalent of a for loop):

```python
import jax.lax as lax

def body_fn(carry, x):
    # carry: state that persists across iterations
    # x: current element
    new_carry = carry + x
    return new_carry, new_carry  # (new_carry, output_for_this_step)

initial = 0.0
xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
final_carry, outputs = lax.scan(body_fn, initial, xs)
# final_carry = 15.0, outputs = [1.0, 3.0, 6.0, 10.0, 15.0]
```

### 5.2 The Functional Training Loop Pattern

```python
# Functional state update: no mutation, no side effects
import jax, optax

def create_train_state(key, model, learning_rate):
    params = model.init(key, jnp.ones((1, 784)))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        return -jnp.mean(jnp.sum(batch['label'] * jax.nn.log_softmax(logits), axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

> **💡 Tip:** The functional pattern propagates through the entire training pipeline: data loading produces batches, training step transforms state → new state, evaluation transforms state → metrics. Every component is a pure function of its inputs. This is what makes the entire training script compilable via `@jax.jit`.

---

## 🎯 Key Takeaways
- **`jax.numpy` looks like NumPy but behaves like a functional language**: 95% API compatibility hides fundamental differences in mutability, randomness, and execution model.
- **Immutability is JAX's most important constraint**: `x[i] = v` is forbidden; use `x.at[i].set(v)`. This restriction enables XLA's aggressive optimizations (fusion, dead code elimination, parallelization).
- **PRNG keys replace global random state**: every random operation consumes a key and produces a new one. `jax.random.split(key)` generates independent subkeys. Never reuse keys — it produces identical "random" outputs.
- **PyTrees unify structured data**: parameters, optimizer states, and metrics are all nested containers of arrays traversed automatically by `tree_map`, `grad`, and `vmap`.
- **Functional purity enables composability**: no global state, no side effects, deterministic outputs for given inputs. This is what allows `jit`, `vmap`, `grad`, and `pmap` to compose arbitrarily without bugs.
- **`lax.scan` replaces for loops** inside jit-compiled functions; `lax.cond` replaces if/else. These maintain functional purity while expressing iteration and branching.
- The mental shift from "mutate this tensor" to "produce this value" is the single most important skill for JAX mastery. Once internalized, the rest follows naturally.

## 📦 Código de Compresión

```python
import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util

# DEMO 1: Immutability — the .at pattern
def demo_immutability():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Original x: {x}")
    x_new = x.at[2].set(99.0).at[0:2].set(jnp.array([10.0, 20.0]))
    print(f"x_new (updated copy): {x_new}")
    print(f"x still intact: {x}")
    print(f"¡Sorpresa! id(x)={id(x)}, id(x_new)={id(x_new)} — different objects")

# DEMO 2: PRNG keys — explicit, pure, splittable
def demo_prng():
    key = random.PRNGKey(42)
    print(f"Root key: {key}")  # [0 42] — just data!
    k1, k2, k3 = random.split(key, num=3)
    a = random.normal(k1, (3,))
    b = random.normal(k2, (3,))
    c = random.normal(k1, (3,))  # Same key → same output!
    print(f"a (from k1): {a}")
    print(f"b (from k2): {b}")
    print(f"c (from k1, same as a): {c}")  # ¡Sorpresa! a == c

# DEMO 3: PyTrees — automatic traversal
def demo_pytrees():
    params = {'w': jnp.eye(3), 'b': jnp.zeros(3)}
    leaves, treedef = tree_util.tree_flatten(params)
    print(f"Flattened leaves: {len(leaves)} arrays, shapes: {[l.shape for l in leaves]}")
    doubled = tree_util.tree_map(lambda x: x * 2, params)
    print(f"PyTree shape preserved: {type(doubled)}, keys: {list(doubled.keys())}")

demo_immutability()
demo_prng()
demo_pytrees()
```

## References
- JAX Documentation — Sharp Bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
- JAX PRNG Design: https://jax.readthedocs.io/en/latest/jep/263-prng.html
- JAX PyTree documentation: https://jax.readthedocs.io/en/latest/pytrees.html
- Frostig, Johnson, Leary (2018). "Compiling ML Programs via High-Level Tracing."
- [[05/03 - Deep Learning con PyTorch]]
- [[05/09 - Deep Learning with TensorFlow]]
- [[04/01 - Matemáticas para ML]]
- [[07/32 - Advanced ML Topics]]
