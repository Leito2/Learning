# 🚀 Training at Scale

## 🎯 Learning Objectives

- Distribute training across GPUs, TPUs, and multiple hosts using `tf.distribute` strategies
- Understand the mechanics of `tf.function`, AutoGraph, tracing, and retracing
- Accelerate kernels with XLA compilation and mixed precision at scale
- Save and restore distributed training state with `tf.train.Checkpoint`
- Compare TensorFlow distribution semantics with PyTorch DistributedDataParallel

## Introduction

Training modern models with billions of parameters requires computation beyond any single device. TensorFlow addresses this through `tf.distribute`, which abstracts all-reduce communication, variable placement, and gradient synchronization behind a strategy object.

Distribution alone is insufficient: Python-level loops are too slow. TensorFlow solves this with `tf.function`, which traces Python into optimizable graphs, and XLA, which compiles them into fused device code. This note connects to [[01 - tf.keras Architectures]] and [[02 - tf.data and TFRecord Pipelines]], and bridges to [[09 - MLOps y Produccion]] and [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome]]. For PyTorch users, strategy scope replaces `DistributedDataParallel` wrapping, and `tf.function` replaces `torch.compile`.

---

## 1. Distribution Strategies

Data parallelism replicates the model across devices, splits the global batch into local micro-batches, computes gradients independently, then synchronizes them before updating weights.

- **`MirroredStrategy`**: Single-host, multi-GPU. Gradients are all-reduced via NCCL; optimizer steps are identical on all replicas.
- **`MultiWorkerMirroredStrategy`**: Multi-machine with gRPC inter-machine communication; requires `TF_CONFIG`.
- **`TPUStrategy`**: Targets Cloud TPU and TPU Pods with high-bandwidth ICI; expects `tf.data.Dataset` inputs.
- **`ParameterServerStrategy`**: For models too large for one worker, sharding variables across parameter servers (common in large recommender systems).

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# MirroredStrategy: automatic replication on all visible GPUs.
# strategy.scope() ensures variables and layers are created once per replica.
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = keras.Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(10)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

# Global batch size must scale with replica count.
# Each replica processes batch_size / num_replicas examples.
GLOBAL_BATCH_SIZE = 64 * strategy.num_replicas_in_sync

# MultiWorkerMirroredStrategy: requires TF_CONFIG in the environment.
# {"cluster": {"worker": ["host1:port", "host2:port"]}, "task": {"type": "worker", "index": 0}}
# strategy = tf.distribute.MultiWorkerMirroredStrategy()

# TPUStrategy: connects to a Cloud TPU via a gRPC resolver.
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://...")
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)

# ParameterServerStrategy: for models with large embedding tables.
# strategy = tf.distribute.experimental.ParameterServerStrategy()
```

⚠️ Creating model variables outside `strategy.scope()` places them on the CPU, causing expensive cross-device copies during every forward pass and often triggering OOM on the host.

💡 "Scope your variables, scale your batch, reduce your gradients."

**Caso real**: Google Search trains ranking models on TPUs using `TPUStrategy`, where the global batch size reaches millions of examples to stabilize sparse feature gradients. The entire model definition lives inside `strategy.scope()` to guarantee correct device placement.

❌ **Wrong**: Defining the model outside `strategy.scope()` and then wrapping it inside.
✅ **Right**: Create the model and optimizer entirely within `with strategy.scope():` so Keras places variables on each replica correctly.

```python
# Wrong way
model = keras.Sequential([layers.Dense(10)])
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model.compile(...)  # Variables already on CPU!

# Right way
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.Sequential([layers.Dense(10)])
    model.compile(...)
```

---

## 2. Graph Compilation with tf.function

Eager execution is great for debugging but suffers from Python dispatch latency, inability to fuse kernels, and lack of whole-program optimization.

`tf.function` converts Python into a callable TensorFlow graph. On first invocation, TensorFlow **traces** the function in eager mode, caches the graph, and reuses it for identical input signatures. **AutoGraph** converts Python control flow (`if`, `for`, `while`) into graph-compatible ops. Retracing happens on signature changes and is expensive; in distributed training, it can stall all replicas.

```python
import tensorflow as tf

@tf.function
def dense_layer_sum(x, w, b):
    # AutoGraph converts this Python if into a tf.cond.
    if tf.reduce_mean(x) > 0.0:
        x = x * 2.0
    return tf.matmul(x, w) + b

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 128], dtype=tf.float32),
    tf.TensorSpec(shape=[128, 64], dtype=tf.float32),
    tf.TensorSpec(shape=[64], dtype=tf.float32),
])
def stable_dense(x, w, b):
    return tf.nn.relu(tf.matmul(x, w) + b)

# Python side effects run only at trace time; use tf.print in graph.
@tf.function
def side_effect_demo(x):
    tf.print("Graph print:", x)
    return x + 1
```

The key latency numbers: eager mode processes each op through the Python runtime (~1µs per op dispatch overhead). Graph mode dispatches once per entire function call. For a model with thousands of ops, `tf.function` easily provides 3-5x speedup.

```python
# Benchmark eager vs graph mode
@tf.function
def graph_fn(x):
    return tf.matmul(x, x) + tf.reduce_sum(x)

def eager_fn(x):
    return tf.matmul(x, x) + tf.reduce_sum(x)

x = tf.random.normal([1000, 1000])
# graph_fn will be ~3-5x faster on repeated calls
```

⚠️ Retracing on every batch due to varying sequence lengths destroys throughput. Use `None` dimensions in `input_signature` to accept variable sizes without triggering a new trace.

💡 "Trace once, run many. Pin your signatures, watch your shapes."

**Caso real**: Uber's Michelangelo platform exports all production models via `tf.function` tracing into SavedModel, ensuring inference graphs run identically in training and serving environments. The `input_signature` parameter guarantees that no retracing happens during serving, keeping latency predictable.

❌ **Wrong**: Using Python `print()` for debug logging inside `@tf.function` — it prints only at trace time, not on every call.
✅ **Right**: Use `tf.print()` inside `@tf.function` for graph-compatible logging that executes every call.

---

## 3. XLA and Mixed Precision at Scale

TensorFlow normally executes operations through individual kernels, each reading and writing memory. **XLA (Accelerated Linear Algebra)** compiles clusters of ops into a single optimized kernel, fusing convolutions with biases and activations while eliminating intermediate buffers.

Mixed precision is required to saturate modern accelerators. Tensor Cores and TPUs perform `bfloat16`/`float16` matmul up to 16x faster than `float32`, but small gradients can underflow. `LossScaleOptimizer` multiplies loss before backprop, divides gradients after, and skips steps with Inf/NaN while adjusting scale dynamically.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# XLA Compilation: jit_compile=True passes the graph through XLA for kernel fusion.
@tf.function(jit_compile=True)
def xla_train_step(x, y, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Mixed Precision with Distribution
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, dtype="float32")
    ])
    base_opt = keras.optimizers.Adam(1e-3)
    optimizer = keras.mixed_precision.LossScaleOptimizer(base_opt)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True
    )

print(optimizer.dynamic_scale)
```

The loss scale mechanism works as follows: multiply the loss by a large constant (e.g., 1024) before backprop, compute gradients in float16 (which would otherwise underflow), then divide the gradients by the same constant before applying. If any gradient contains Inf or NaN, the step is skipped and the scale factor is halved.

⚠️ Enabling XLA on CPU rarely helps and can hurt performance because CPU kernels are already heavily optimized in Eigen. XLA shines on GPU and TPU.

💡 "XLA fuses ops, mixed precision doubles speed, loss scale guards the small."

**Caso real**: OpenAI trains GPT-class models with mixed precision on NVIDIA clusters; loss scaling prevents divergence during the early phase when activation gradients are small. The combination of XLA + mixed precision gives ~2x throughput over FP32 on A100 GPUs.

❌ **Wrong**: Enabling `mixed_float16` but keeping the output layer in float16, causing NaN loss values in cross-entropy.
✅ **Right**: Keep the final logits layer in `dtype="float32"` so the softmax cross-entropy computation has full numerical precision.

---

## 4. Checkpointing and Advanced Distribution

Distributed training can run for days or weeks; hardware failures and preemptions are inevitable. Checkpointing persists model weights, optimizer state, and global step to resume exactly where training left off.

`tf.train.Checkpoint` uses **deferred restoration**: register objects by name, save values to disk, and restore lazily by matching names and shapes. This lets you change the Python object graph and still restore compatible weights.

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(10)])
optimizer = keras.optimizers.Adam()
step = tf.Variable(0, dtype=tf.int64)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
manager = tf.train.CheckpointManager(checkpoint, directory="./checkpoints", max_to_keep=3)

if int(step) % 1000 == 0:
    save_path = manager.save()
    print(f"Saved checkpoint at {save_path}")

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print("Restored.")
```

The `CheckpointManager` handles disk bookkeeping: it keeps only the `max_to_keep` most recent checkpoints, deleting older ones automatically. This prevents filling your disk during long training runs. On restore, you call `manager.latest_checkpoint` to find the most recent file and `checkpoint.restore()` to load all matching variables.

For massive models, `tf.experimental.dtensor` introduces a **device mesh** abstraction where tensors are sharded across a multi-dimensional grid of devices — TensorFlow's answer to FSDP for models with hundreds of billions of parameters.

```python
# DTensor conceptual snippet
from tensorflow.experimental import dtensor
mesh = dtensor.create_mesh([("x", 2), ("y", 4)], devices=dtensor.local_devices())
layout = dtensor.Layout([dtensor.UNSHARDED, "x"], mesh)
```

⚠️ Saving only `model.save_weights()` and not the optimizer state forces you to restart training from a cold optimizer, which often causes loss spikes and convergence issues after resume.

💡 "Save the model, save the opt, save the step — resume with no regret."

**Caso real**: Meta AI uses checkpointing aggressively for OPT and LLaMA training runs, with async checkpoint writes to NFS so the training step is not blocked by disk I/O. Their `CheckpointManager` keeps up to 20 checkpoints spread across training to allow rolling back to any recent state after a hardware fault.

❌ **Wrong**: Periodically saving only `model.save_weights("ckpt.h5")` in a loop, overwriting the same file.
✅ **Right**: Use `tf.train.CheckpointManager` with `max_to_keep=5` to preserve a sliding window of checkpoints with optimizer state and training step.

---

## Compression Code

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.mixed_precision.set_global_policy("mixed_float16")

strategy = tf.distribute.MirroredStrategy()
print(f"Replicas: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = keras.Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(10, dtype="float32")
    ])
    base_opt = keras.optimizers.Adam(1e-3)
    optimizer = keras.mixed_precision.LossScaleOptimizer(base_opt)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, "./ckpts", max_to_keep=3)
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print("Restored.")

def make_dataset(batch_size):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([1024, 128]), tf.random.uniform([1024], maxval=10, dtype=tf.int32))
    )
    return ds.shuffle(256).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(64 * strategy.num_replicas_in_sync)

@tf.function(jit_compile=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def distributed_train_step(x, y):
    per_replica_losses = strategy.run(train_step, args=(x, y))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

for step, (x, y) in enumerate(train_ds):
    loss = distributed_train_step(x, y)
    if step % 100 == 0:
        print(f"Step {step}, loss: {loss:.4f}")
        manager.save()
```

## 🎯 Key Takeaways

- `MirroredStrategy` is the default for single-host multi-GPU; use `TPUStrategy` for TPU pods
- Always create models and optimizers inside `strategy.scope()` for correct device placement
- `@tf.function` traces Python into graphs; specify `input_signature` to avoid expensive retracing
- XLA (`jit_compile=True`) fuses kernels and often yields 10-30% speedups on GPU/TPU
- Mixed precision requires `LossScaleOptimizer` to prevent gradient underflow in `float16`
- `tf.train.Checkpoint` + `CheckpointManager` robustly saves and resumes distributed training state

## References

- TensorFlow distributed training guide: https://www.tensorflow.org/guide/distributed_training
- tf.function and AutoGraph: https://www.tensorflow.org/guide/function
- XLA overview: https://www.tensorflow.org/xla
- Mixed precision: https://www.tensorflow.org/guide/mixed_precision
- PyTorch comparison: see [[05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida]]
- Distributed infrastructure: see [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome]]
