# 🏷️ Training Loops with Optax and End-to-End JAX

## 🎯 Learning Objectives
- Master **Optax** — JAX's composable optimizer library where optimizers are pure functions
- Design **explicit training loops** in JAX: full control, no magic, maximum flexibility
- **jit-compile the entire training step** for 1.5–2× speedup over eager execution
- Implement **gradient accumulation, mixed precision, and learning rate schedules**
- Export and checkpoint models with **Orbax** for production deployment

## Introduction

JAX does not have a built-in training loop API. There is no `model.fit()`, no `Trainer` class, no callback system. This is not an oversight — it is a deliberate design choice that reflects JAX's philosophy: **you write the loop**. PyTorch Lightning and HuggingFace Trainer abstract the training loop away; JAX hands you `grad`, `jit`, and `vmap` and says "build what you need." This is simultaneously liberating (you control every aspect) and demanding (you must write correct, efficient loops yourself). But the payoff is real: a well-written JAX training loop runs **1.5–2× faster** than equivalent PyTorch because the entire step — forward, backward, optimizer update — is compiled into a single fused XLA kernel.

Optax fills the optimizer role. It's a library of composable gradient transformations: chain clipping, weight decay, and SGD/Adam/AdamW into a single pure function. An Optax optimizer is not an object with state — it's a function `(grads, opt_state, params) → (updates, new_opt_state)`. You call it, get updates, apply them to parameters, carry the new optimizer state to the next step. There is no optimizer object mutating internal buffers; everything is explicit and functional.

This note covers the complete training loop: Optax fundamentals, the jit-compiled training step, learning rate scheduling, gradient accumulation for large effective batch sizes, mixed-precision training, checkpointing with Orbax, and the patterns that scale from a single GPU to a TPU pod. We'll also cover the debugging workflow — `jax.disable_jit()`, `jax.debug.print()`, and how to profile your compiled loops. By the end, you'll be able to write production-grade training loops that are fast, correct, and clean.

---

## 1. Optax — Composable Gradient Transformations

### 1.1 Optimizers as Pure Functions

In PyTorch, `optimizer.step()` mutates parameters and internal state. In Optax, the optimizer is a pure function that computes updates from gradients:

$$\theta_{t+1} = \theta_t - \eta \cdot \text{opt}(\nabla \mathcal{L}(\theta_t), \text{opt_state}_t)$$

Mathematically, the optimizer is a function $\text{opt} : \mathbb{R}^n \times \text{OptState} \to \mathbb{R}^n \times \text{OptState}$ that takes gradients and optimizer state, producing updates and new optimizer state:

$$\text{opt}(g_t, s_t) = (u_t, s_{t+1}), \quad \theta_{t+1} = \theta_t - u_t$$

```python
import optax
import jax.numpy as jnp
import jax

# Create optimizer — it's a GradientTransformation (a pair of init + update functions)
optimizer = optax.adam(learning_rate=1e-3)

# opt_state tracks Adam's m (first moment) and v (second moment) estimates
params = jnp.array([1.0, 2.0, 3.0])
opt_state = optimizer.init(params)

# Compute gradients (using jax.grad)
loss = lambda p: jnp.sum(p ** 2)
grads = jax.grad(loss)(params)  # [2.0, 4.0, 6.0]

# Get parameter updates — the optimizer is a PURE function
updates, new_opt_state = optimizer.update(grads, opt_state, params)

# Apply updates (also a pure function)
new_params = optax.apply_updates(params, updates)
print(f"Params: {new_params}")  # Updated parameters
```

> **¡Sorpresa!** `optax.apply_updates` is a simple helper: `params - updates`. You can write `params - lr * grads` yourself, but `apply_updates` handles PyTrees and sign conventions correctly for all optimizers (some decompose the update differently).

### 1.2 Composing Optimizer Components

Optax's killer feature is **chaining**: combine gradient transformations into a pipeline.

```python
# Chain multiple transformations: clip gradients → apply weight decay → AdamW
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),       # 1. Clip gradients
    optax.add_decayed_weights(weight_decay=1e-4),  # 2. Add weight decay
    optax.adamw(learning_rate=1e-3),               # 3. AdamW update
)

# Or: gradient clipping + SGD with momentum
optimizer = optax.chain(
    optax.clip(max_delta=0.1),                     # Clip per-parameter
    optax.sgd(learning_rate=0.01, momentum=0.9),
)
```

The transformations apply in order, left to right. `optax.chain(clip, decay, adamw)` means: first clip gradients, then add weight decay contributions, then compute AdamW update from the result.

### 1.3 Common Optimizer Components

```python
# Gradient clipping
optax.clip(max_delta=0.1)                         # Per-parameter clipping
optax.clip_by_global_norm(max_norm=1.0)           # Global norm clipping
optax.clip_by_block_rms(threshold=1.0)            # Per-block RMS clipping

# Weight regularization
optax.add_decayed_weights(weight_decay=0.01)       # L2 weight decay (decoupled)
optax.add_noise(eta=1e-4, gamma=0.55, seed=0)     # Gradient noise (helps generalization)

# Optimizers
optax.sgd(learning_rate, momentum=0.9, nesterov=True)
optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
optax.adamw(learning_rate, b1=0.9, b2=0.999, weight_decay=0.01)
optax.adagrad(learning_rate)
optax.rmsprop(learning_rate, decay=0.9)
optax.lion(learning_rate, b1=0.9, b2=0.99)        # Lion optimizer (from Symbolic Discovery)
optax.lamb(learning_rate)                          # Layer-wise Adaptive Moments
optax.lars(learning_rate)                          # Layer-wise Adaptive Rate Scaling

# Learning rate schedules (compose with optimizer via inject_hyperparams)
optax.cosine_decay_schedule(init_value=1e-3, decay_steps=10000)
optax.warmup_cosine_decay_schedule(init_value=0, peak_value=1e-3,
                                    warmup_steps=1000, decay_steps=10000)
optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.96)
optax.piecewise_constant_schedule(init_value=1e-3,
                                   boundaries_and_scales={5000: 0.1, 15000: 0.1})
```

### 1.4 Injecting Schedules into Optimizers

```python
# Method 1: optax.inject_hyperparams — most general
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
)
optimizer = optax.inject_hyperparams(optimizer, static_args='learning_rate')

# Method 2: optax.adam(learning_rate=schedule) — simpler for single schedules
optimizer = optax.adam(learning_rate=schedule)
```

> **💡 Tip:** `optax.inject_hyperparams` is the canonical way to make hyperparameters differentiable or schedulable. It wraps the optimizer's `init` and `update` functions so that `learning_rate` is part of the optimizer state and can be updated per-step.

---

## 2. The jit-Compiled Training Step

### 2.1 The Basic Training Loop

```python
import jax, optax
import jax.numpy as jnp
from flax.training import train_state

def create_train_state(key, model, input_shape, learning_rate):
    params = model.init(key, jnp.ones(input_shape))['params']
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch):
    """Compiled training step — pure function, no side effects."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = (logits.argmax(axis=-1) == batch['y']).mean()
    return state, {'loss': loss, 'accuracy': accuracy}

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        state, metrics = train_step(state, batch)
        # First iteration: JIT compiles (~2-5 seconds)
        # Subsequent iterations: cached compiled code (~microseconds)
```

### 2.2 What Gets Compiled?

When you `@jax.jit` the training step, the **entire function** is compiled:

```
train_step = jit(forward → loss → grad → optimizer.update → apply_updates)
```

XLA fuses all these into a single GPU kernel. There are no intermediate memory allocations for `logits`, `loss`, or `grads` — the compiler allocates exactly the memory needed, reuses buffers, and eliminates dead writes. This is why JAX training loops are fast:

$$\text{JAX train\_step} = \text{XLA}( \text{forward} \circ \text{loss} \circ \text{grad} \circ \text{update} )$$

vs PyTorch eager:

$$\text{PyTorch train\_step} = \text{forward}() \to \text{loss}() \to \text{backward}() \to \text{step}() \quad (\text{4 separate kernel sequences})$$

> **Caso real: DeepMind benchmarks show** that a jit-compiled JAX training loop runs 1.5–2× faster than equivalent PyTorch eager on a single GPU for transformer training. On TPU pods with pmap, the speedup grows to 3–5× because JAX's SPMD model eliminates the Python overhead of coordinating multiple devices.

### 2.3 Debugging JIT-Compiled Loops

During development, you can't `print` inside a jitted function (or rather, `print` executes during tracing, not runtime). Options:

```python
# 1. jax.debug.print — works inside jit!
@jax.jit
def step(state, batch):
    loss = compute_loss(state.params, batch)
    jax.debug.print("Loss at step {}: {}", state.step, loss)  # Prints at runtime!
    return loss

# 2. jax.disable_jit — temporarily disable compilation for debugging
with jax.disable_jit():
    loss = step(state, batch)  # Runs in eager mode — can use breakpoints, print, etc.

# 3. jax.make_jaxpr — inspect the compiled computation
jaxpr = jax.make_jaxpr(train_step)(state, batch)
print(jaxpr)  # Shows the traced computation before XLA compilation
```

> **💡 Tip:** Always develop your training loop with `jax.disable_jit()` first. Get the logic right, then add `@jax.jit` for performance. The `disable_jit` context manager is your best friend during debugging.

---

## 3. Advanced Training Patterns

### 3.1 Gradient Accumulation

When your GPU memory can't fit a full batch, accumulate gradients across micro-batches:

```python
@jax.jit
def accumulate_gradients_step(state, batch, accumulated_grads, rng):
    """Compute gradients for one micro-batch and accumulate."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'],
                                 rngs={'dropout': rng})
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Accumulate: sum gradients across micro-batches
    accumulated_grads = jax.tree_map(
        lambda acc, g: acc + g, accumulated_grads, grads
    )
    return accumulated_grads, loss

# In training loop:
accumulated = jax.tree_map(jnp.zeros_like, state.params)
for micro_batch in micro_batches:
    accumulated, loss = accumulate_gradients_step(state, micro_batch, accumulated, key)

# After accumulation step: apply gradients once
grads = jax.tree_map(lambda g: g / num_micro_batches, accumulated)
state = state.apply_gradients(grads=grads)
```

### 3.2 Mixed Precision Training

Use `jmp` (JAX Mixed Precision) for FP16/BF16 training:

```python
import jmp

# Configure mixed precision policy
policy = jmp.Policy(
    param_dtype=jnp.float32,       # Parameters stay in float32
    compute_dtype=jnp.bfloat16,   # Computations in bfloat16
    output_dtype=jnp.float32,      # Outputs back to float32
)

@jax.jit
def mixed_precision_step(state, batch):
    # Cast params to bfloat16 for computation, keep master copy in float32
    params_f16 = jmp.cast_to_compute(state.params)

    def loss_fn(mp_params):
        logits = state.apply_fn({'params': mp_params}, batch['x'])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']).mean()

    loss, grads = jax.value_and_grad(loss_fn)(params_f16)
    # Gradients are in compute dtype — cast back to float32 for optimizer
    grads = jmp.cast_to_param(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

Note: as of 2024, `jmp` is being superseded by JAX's native mixed precision support. Check the latest JAX docs for the recommended approach.

> **⚠️ Warning:** Mixed precision requires careful handling of loss scaling for FP16 (to prevent gradient underflow). BF16 doesn't need loss scaling because it preserves the exponent range of FP32. For TPU training, always use BF16.

### 3.3 Multi-GPU Training with pmap

```python
# Single-GPU training step
@jax.jit
def train_step(state, batch):
    ...

# Multi-GPU: just add pmap!
devices = jax.devices()
n_devices = len(devices)

@jax.pmap
def parallel_train_step(state, batch):
    # Same train_step logic, but runs on each device
    def loss_fn(params):
        ...
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Sync gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    state = state.apply_gradients(grads=grads)
    return state, loss

# Reshape batch for pmap: (total_batch,) → (n_devices, per_device_batch)
batch = jax.tree_map(
    lambda x: x.reshape(n_devices, -1, *x.shape[1:]), batch
)
state, loss = parallel_train_step(state, batch)
```

The model code doesn't change! Only the outermost `pmap` call and the batch reshaping differ. This is the functional paradigm's superpower — the model is unaware of parallelism.

---

## 4. Checkpointing and Export with Orbax

### 4.1 Saving and Loading with Orbax

```python
import orbax.checkpoint as ocp

# Create checkpoint manager
checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
options = ocp.CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics['accuracy'])

# Save checkpoint
checkpointer.save('/path/to/checkpoints/step_0', state)

# Restore checkpoint
restored_state = checkpointer.restore(
    '/path/to/checkpoints/step_0',
    target=state  # ⚠️ Must provide target to reconstruct PyTree structure
)
```

For production training loops:

```python
checkpoint_manager = ocp.CheckpointManager(
    '/path/to/checkpoints',
    options=ocp.CheckpointManagerOptions(
        max_to_keep=5,
        save_interval_steps=1000,
    ),
)

# In training loop:
for step, batch in enumerate(dataloader):
    state, metrics = train_step(state, batch)
    if step % 1000 == 0:
        checkpoint_manager.save(step, state, metrics=metrics)
```

### 4.2 Exporting for Serving

```python
# Export to TensorFlow SavedModel (via jax2tf)
import jax2tf

@jax.jit
def serving_fn(params, x):
    return model.apply({'params': params}, x, train=False)

# Convert to TF function
tf_serving_fn = jax2tf.convert(serving_fn, with_gradient=False)

# Export to TFLite for mobile/edge
import jax2tf as jax2tf
converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_serving_fn.get_concrete_function(...)])
tflite_model = converter.convert()
```

> **💡 Tip:** For production serving, consider ONNX export or JAX-native serving via `jax.jit(serving_fn)` in a Python server. The jit-compiled function is already optimized — no need for TensorFlow. See [[09/20 - Deployment y Serving]] for complete deployment patterns.

---

## 5. Profiling and Performance Optimization

### 5.1 The JAX Profiler

```python
# Start profiling
jax.profiler.start_trace('/tmp/jax_trace')

# Run a few steps
for i in range(10):
    state, metrics = train_step(state, batch)

# Stop profiling
jax.profiler.stop_trace()

# View with TensorBoard
# $ tensorboard --logdir=/tmp/jax_trace
```

### 5.2 Common Performance Tips

```python
# 1. Prefer jnp operations over Python loops inside jit
@jax.jit
def bad_loop(x):
    for i in range(100):
        x = x + 0.01 * jnp.sin(x)   # ❌ JAX unrolls Python loops
    return x

@jax.jit
def good_scan(x):
    return jax.lax.fori_loop(0, 100, lambda i, v: v + 0.01 * jnp.sin(v), x)

# 2. Avoid repeated jit calls with varying static shapes
for size in [32, 64, 128, 256]:
    x = jnp.ones((size, 784))
    f(x)  # ❌ New compilation for each size!

# Instead: pad to fixed size or use variable-length batching

# 3. Use jax.lax.map instead of vmap for sequential execution
#     when vmap memory explodes (large intermediate tensors)

# 4. Enable async dispatch for overlapping computation and data loading
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'gpu')
```

> **Caso real: Google trains PaLM-540B using 6144 TPU v4 chips** with a JAX training loop that is fully jit-compiled. The entire training step — forward pass through 118 transformer layers, backward pass, gradient all-reduce across 6144 chips, and optimizer update — is compiled into **one XLA module**. The compilation takes ~15 minutes on the first step, but subsequent steps run at peak TPU throughput with zero Python overhead. No PyTorch training loop can achieve this because eager execution introduces per-step Python latencies that become the bottleneck at scale.

---

## 🎯 Key Takeaways
- **Optax optimizers are pure functions**: `(grads, opt_state, params) → (updates, new_opt_state)`. Compose them with `optax.chain()` — clip, decay, optimize in sequence.
- **Write the training loop yourself**: JAX gives you `grad`, `jit`, `vmap` and lets you compose them. No abstract `Trainer` class — you build exactly what you need.
- **jit-compile the entire training step**: forward, backward, gradient sync, and optimizer update become one fused XLA kernel. This eliminates intermediate memory and yields 1.5–2× speedup.
- **Gradient accumulation** simulates large batch sizes by summing gradients across micro-batches in JAX. Since gradients are explicit arrays, this is trivial.
- **Learning rate schedules** are first-class in Optax: cosine decay, warmup, piecewise constant. Inject them via `optax.inject_hyperparams`.
- **Checkpointing with Orbax** supports PyTree structures natively — save/restore works on `TrainState` objects directly.
- **Multi-GPU training requires only `jax.pmap`** — the model code doesn't change. The functional design makes scaling transparent.

## 📦 Código de Compresión

```python
import jax, jax.numpy as jnp, optax
from flax.training import train_state
import flax.linen as nn

# Model
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x); x = nn.relu(x); x = nn.Dense(10)(x)
        return x

# State
model = SimpleNN()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))['params']
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params,
    tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=1e-4)),
)

# Training step — fully jit-compiled
@jax.jit
def step(state, x, y):
    def loss_fn(p):
        logits = state.apply_fn({'params': p}, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

# Run
x_b, y_b = jnp.ones((32, 784)), jnp.zeros(32, dtype=jnp.int32)
new_state, loss = step(state, x_b, y_b)
print(f"Loss: {loss:.4f}")
print("¡Sorpresa! One fused kernel: forward + backward + optimizer update = 1 XLA call.")
```

## References
- Optax Documentation: https://optax.readthedocs.io/
- Hessel et al. (2020). "Optax: composable gradient transformation and optimisation, in JAX!"
- Orbax Documentation: https://orbax.readthedocs.io/
- JAX Profiling Guide: https://jax.readthedocs.io/en/latest/profiling.html
- Chowdhery et al. (2022). "PaLM: Scaling Language Modeling with Pathways." *arXiv:2204.02311*.
- [[05/03 - Deep Learning con PyTorch]]
- [[05/09 - Deep Learning with TensorFlow]]
- [[09/20 - Deployment y Serving]]
- [[09/24 - Weights and Biases]]
