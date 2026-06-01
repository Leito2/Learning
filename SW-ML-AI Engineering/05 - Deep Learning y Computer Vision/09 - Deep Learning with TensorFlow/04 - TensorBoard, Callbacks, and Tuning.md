# 🏷️ 04 - TensorBoard, Callbacks, and Tuning

## 🎯 Learning Objectives
- Master the Keras `Callback` base class and its lifecycle hooks.
- Configure built-in callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`, etc.) for robust training pipelines.
- Build custom callbacks to inject arbitrary logic into the training loop.
- Use TensorBoard to visualize scalars, images, histograms, and computation graphs.
- Profile TensorFlow programs with `tf.profiler` and interpret results in TensorBoard.
- Automate hyperparameter search with KerasTuner (`BayesianOptimization`, `Hyperband`, `RandomSearch`).
- Design reusable `HyperModel` classes and understand `Oracle` and `Tuner` internals.

## Introduction
Training deep neural networks is rarely a fire-and-forget operation. Without intervention, models can overfit, learning rates can stall, or training can waste epochs on a dead-end configuration. Keras callbacks are the canonical mechanism for injecting control logic into the training loop without rewriting the fit routine. They bridge the gap between research experimentation and production reliability, enabling automated checkpointing, adaptive learning rates, and early termination. TensorBoard extends this by providing a time-travel debugger for your model: every weight histogram, gradient norm, and prediction image can be visualized across epochs, turning black-box optimization into an inspectable process. Finally, KerasTuner transforms the art of hyperparameter selection into a reproducible, scalable search process, replacing manual grid-search notebooks with Bayesian optimization and population-based training strategies.

This note connects to model monitoring in [[09 - MLOps y Produccion]], distributed training infrastructure in [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome]], and system design patterns in [[10 - Cloud, Infra y Backend/32 - System Design for ML/02 - System Design for ML]]. It also complements the PyTorch training ecosystem covered in [[05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida]].

---

## 1. Callback Architecture

The callback pattern originates from event-driven programming: an object registers interest in lifecycle events and executes code when those events fire. In Keras, the `fit()` loop is a closed system that iterates over epochs and batches. Without callbacks, the only way to alter behavior mid-flight would be to fork the framework itself. The callback API exposes hooks (`on_epoch_begin`, `on_epoch_end`, `on_train_batch_begin`, `on_train_batch_end`, `on_train_begin`, `on_train_end`, and their test/predict counterparts) that the training engine calls at predetermined points.

Early deep learning frameworks like Caffe required external shell scripts to snapshot models periodically. Keras unified this into a first-class Python API, making training pipelines composable and testable. The design motivation is separation of concerns: the training loop handles forward/backward passes, while callbacks handle everything orthogonal to gradient descent—checkpointing, logging, learning rate scheduling, and early stopping. This decoupling means you can stack callbacks like middleware in a web framework, each performing a distinct responsibility without interfering with others.

```python
import tensorflow as tf
from tensorflow import keras

class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best_loss = float('inf')
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = float(tf.timestamp() - self._start)
        self.epoch_times.append(elapsed)
        current = logs.get('val_loss')
        if current is not None and current < self.best_loss:
            self.best_loss = current
            print(f"Epoch {epoch}: new best val_loss {current:.4f} ({elapsed:.1f}s)")

    def on_train_end(self, logs=None):
        avg = sum(self.epoch_times) / len(self.epoch_times)
        print(f"Done. Best val_loss: {self.best_loss:.4f}, avg epoch: {avg:.1f}s")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras', monitor='val_accuracy', save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7
    ),
    keras.callbacks.TerminateOnNaN(),
    MyCallback()
]

model.fit(x_train, y_train, validation_split=0.2, epochs=100, callbacks=callbacks)
```

⚠️ Setting `restore_best_weights=True` in `EarlyStopping` without `ModelCheckpoint` means you lose the exact epoch snapshot if your process crashes after training ends — the callback only restores weights in memory. Always pair them so disk and memory are synchronized.

⚠️ Using `TerminateOnNaN()` without identifying the root cause (exploding gradients, bad data normalization) turns a training failure into a silent early exit. Add `tf.debugging.check_numerics` inside your loss function during debugging to catch NaN origins before they propagate.

The relationship between patience values is critical. `ReduceLROnPlateau` should have lower patience than `EarlyStopping` so the LR has a chance to escape plateaus before training is terminated entirely. A typical configuration is `ReduceLROnPlateau(patience=3)` + `EarlyStopping(patience=6)`.

<antipattern>
❌ Stacking callbacks carelessly:
```python
callbacks = [ReduceLROnPlateau(...), EarlyStopping(...), ModelCheckpoint(...)]
# LR may reduce AFTER the early stop decision has already been made
```
✅ Order by dependency: checkpoint first (capture the best state), then LR reduction (adapt learning rate), then early stopping (decide whether to continue):
```python
callbacks = [ModelCheckpoint(...), ReduceLROnPlateau(...), EarlyStopping(...)]
```
</antipattern>

**Caso real — Spotify:** Music recommendation models use `EarlyStopping` + `ModelCheckpoint` to cut training when validation NDCG plateaus, saving ~30% compute per training run across hundreds of model variants. **Caso real — Waymo:** Custom callbacks log per-step LIDAR loss to TensorBoard, enabling real-time anomaly detection during autonomous-driving model training.

---

## 2. TensorBoard for Visualization

Before TensorBoard, practitioners wrote ad-hoc Matplotlib scripts that were slow, non-interactive, and impossible to compare across runs. TensorBoard centralizes visualization by consuming timestamped event files written during training, enabling scalar tracking, image inspection, histogram analysis, and graph visualization in a single interface.

The underlying mechanism is the `tf.summary` API, which serializes tensors into protocol buffers and appends them to a log directory. TensorBoard watches this directory and updates its web UI in real time. In production ML systems, this log directory becomes the artifact storage for experiment tracking, integrated with MLflow or Vertex AI.

```python
import tensorflow as tf
from datetime import datetime

# Option 1: TensorBoard callback (auto-logs scalars, histograms, graphs)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='batch',
    profile_batch='500,520'
)

model.fit(dataset, epochs=10, callbacks=[tensorboard_callback])

# Option 2: Manual summary writing (for custom training loops)
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    for step, (x, y) in enumerate(train_dataset):
        loss = train_step(x, y)
        tf.summary.scalar('train_loss', loss, step=step)
        if step % 100 == 0:
            preds = model(x[:4])
            tf.summary.image('predictions', preds, step=step, max_outputs=4)

# Option 3: Logging hyperparameters for comparison across runs
with writer.as_default():
    tf.summary.hparams({
        'learning_rate': 1e-3, 'dropout': 0.5, 'optimizer': 'adam',
    }, metrics={'val_accuracy': 0.92})
```

The `summary` API supports multiple data types:
- `tf.summary.scalar(name, data, step)` — loss, accuracy, learning rate
- `tf.summary.image(name, data, step, max_outputs)` — model predictions, input samples
- `tf.summary.histogram(name, data, step)` — weight distributions, gradient norms
- `tf.summary.text(name, data, step)` — hyperparameter configs, evaluation reports
- `tf.summary.mesh(name, vertices, faces, step)` — 3D point clouds (e.g., for LIDAR data)

⚠️ Enabling `histogram_freq=1` on every layer for large models writes gigabytes of event data and slows training by 20-40%. Profile one layer class at a time or reduce frequency to every 10 epochs.

💡 Use `update_freq='epoch'` for stable plots; per-batch logging is only useful for small datasets or debugging. When comparing runs, create a parent directory structure: `logs/fit/run1`, `logs/fit/run2` — then launch `tensorboard --logdir=logs/fit` to see all runs overlaid.

<antipattern>
❌ Forgetting to set `update_freq` on large datasets (1M+ batches) yields millions of data points and unreadable scalar plots.
✅ Set `update_freq='epoch'` to get one clean point per epoch; use per-batch logging only during initial debugging.
</antipattern>

**Caso real — DeepMind:** TensorBoard histograms of AlphaFold attention maps revealed structural patterns in protein folding attention heads, directly informing architectural improvements in subsequent model versions.

---

## 3. Profiling Training Performance

The `tf.profiler` captures hardware-level traces—GPU kernel execution, memory allocation, data pipeline stalls—turning performance optimization from guesswork into data-driven bottleneck removal. The TensorBoard Profile dashboard visualizes a timeline of ops, showing exactly where each step's time is spent.

```python
# Via TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, profile_batch='200,220'
)

# Manual context-manager profiling
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

tf.profiler.experimental.start(log_dir)
for step, (x, y) in enumerate(train_dataset):
    loss = train_step(x, y)
    if step >= 200:
        break
tf.profiler.experimental.stop()
```

💡 Common profiler insights and remediations:
- *Input pipeline idle time >50%* → increase `prefetch` buffer, parallelize `map` with `num_parallel_calls=tf.data.AUTOTUNE`, pre-serialize data to TFRecords
- *GPU kernel launch overhead* → increase batch size, enable XLA (`tf.function(jit_compile=True)`)
- *All-reduce bottleneck* → use gradient compression or switch to Ring All-Reduce in `tf.distribute`
- *Memory bandwidth saturation* → reduce model size, enable mixed precision (float16)

The profiler also computes a summary with actionable recommendations:

```python
# Programmatic profiler analysis
from tensorflow.python.profiler import profiler_client

# Profile a running server
result = profiler_client.monitor('localhost:6006', duration_ms=2000)
print(result)  # includes step time breakdown and bottleneck analysis
```

**Caso real — Google Brain:** Profiling TPU pods identified an all-reduce bottleneck where 30% of step time was spent synchronizing gradients. Implementing gradient compression reduced step time by 18% across a 512-core TPU slice.

![TensorFlow Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/512px-Tensorflow_logo.svg.png)

---

## 4. Hyperparameter Optimization with KerasTuner

Hyperparameter optimization (HPO) is the problem of finding the configuration of knobs that maximize model performance on a validation set. Unlike model weights, hyperparameters are non-differentiable discrete or continuous choices. Manual grid search scales exponentially with dimensionality — for $d$ hyperparameters each taking $v$ values, the cost is $O(v^d)$. Random search is more efficient but still wastes budget on unpromising regions.

KerasTuner separates HPO into the `Oracle` (search strategy) and the `Tuner` (execution engine). The Oracle decides which hyperparameter values to try next based on prior observations, while the Tuner handles training, evaluation, and result reporting. Three strategies are available:

| Strategy | Approach | Best for |
|----------|----------|----------|
| `RandomSearch` | Uniform random sampling | Quick baselines, many hyperparameters |
| `BayesianOptimization` | Gaussian Process surrogate model | Expensive trials (<500), continuous hparams |
| `Hyperband` | Adaptive resource allocation (Successive Halving) | Expensive trials, large search spaces |

**Bayesian Optimization** builds a probabilistic model of the objective: $f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$. The acquisition function, Expected Improvement $\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f^*, 0)]$, balances exploration (high uncertainty) and exploitation (high predicted performance). Each trial updates the posterior, converging to the global optimum in $O(d^2)$ trials rather than $O(v^d)$.

**Hyperband** extends Successive Halving: allocate a small budget to all configurations, retain the top $1/\eta$, multiply budget by $\eta$, repeat. With $\eta=3$, each round triples the epochs for the top third of configurations. This means poor models are killed early, and only promising ones receive the full budget.

```python
import keras_tuner as kt

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(28, 28)))
        model.add(keras.layers.Flatten())

        units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units, activation='relu'))

        if hp.Boolean('use_dropout'):
            model.add(keras.layers.Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))

        model.add(keras.layers.Dense(10, activation='softmax'))
        hp_opt = hp.Choice('optimizer', ['adam', 'sgd'])
        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        opt = (keras.optimizers.Adam(learning_rate=lr) if hp_opt == 'adam'
               else keras.optimizers.SGD(learning_rate=lr, momentum=0.9))
        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args, **kwargs,
            callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        )

# Hyperband: adaptive budget allocation
tuner = kt.Hyperband(
    MyHyperModel(), objective='val_accuracy', max_epochs=40, factor=3,
    directory='my_dir', project_name='intro_to_kt', overwrite=True
)
tuner.search(x_train, y_train, epochs=10, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best units: {best_hps.get('units')}")
print(f"Best lr: {best_hps.get('lr'):.4f}")
print(f"Best dropout: {best_hps.get('dropout_rate', 'N/A')}")

best_model = tuner.hypermodel.build(best_hps)
best_model.fit(x_train, y_train, epochs=50, validation_split=0.2)
```

⚠️ Defining conditional hyperparameters (e.g., `dropout_rate` when `use_dropout=False`) and sampling them unconditionally wastes Oracle trials on irrelevant regions — the Oracle models a relationship that doesn't exist. Use `hp.conditional_scope()` or structure your search space so conditional parameters only appear under their conditions.

⚠️ Setting Hyperband `max_epochs` too low prevents the algorithm from distinguishing good from bad configurations during early rungs. Ensure `max_epochs` is at least 3x the expected convergence epoch count, and use `EarlyStopping` inside `fit()` to cap individual trials.

<antipattern>
❌ Grid search over 5 hyperparameters with 10 values each = 100,000 trials, most wasted on bad regions:
```python
# Each trial gets full epochs regardless of quality
for lr in [1e-4, 1e-3, 1e-2, ...]:
    for units in [32, 64, 128, ...]:
        train_and_eval(lr, units, epochs=50)  # all get 50 epochs
```
✅ Hyperband with `factor=3` evaluates configurations at increasing budgets, discarding poor ones early:
```python
# Round 1: 81 configs × 1 epoch → keep top 27
# Round 2: 27 configs × 3 epochs → keep top 9
# Round 3: 9 configs × 9 epochs → keep top 3
# Round 4: 3 configs × 27 epochs → best 1
```
</antipattern>

**Caso real — Airbnb:** KerasTuner `Hyperband` on listing ranking models found an architecture with +2.3% NDCG in 1/10th the compute of grid search. **Caso real — Spotify:** Bayesian optimization for playlist embedding dimensions reduced embedding table memory by 40% with no quality loss.

---

## 🎯 Key Takeaways
- Keras callbacks are event-driven middleware for the training loop; compose them to handle orthogonal concerns like checkpointing and LR scheduling.
- Always pair `EarlyStopping` with `ModelCheckpoint` to ensure both memory and disk reflect the best model state.
- `ReduceLROnPlateau` patience should be lower than `EarlyStopping` patience so the optimiser can escape plateaus before training is terminated.
- TensorBoard turns training into an inspectable process; use `tf.summary` APIs for custom loops and the callback for standard `fit()` workflows.
- Profiling with `tf.profiler` is essential for production pipelines where data-loading or kernel-launch overhead can dominate step time.
- KerasTuner separates search strategy (`Oracle`) from execution (`Tuner`), enabling scalable HPO with Bayesian optimization or adaptive bandit methods.
- Define search spaces carefully: conditional hyperparameters should avoid exploring irrelevant regions to maximise Oracle efficiency.
- In PyTorch, `torch.utils.tensorboard` provides similar scalar/image logging but lacks native integration with the training loop; practitioners often use Weights & Biases instead.

## References
- [Keras Callbacks API](https://keras.io/api/callbacks/)
- [TensorBoard: TensorFlow Visualization Toolkit](https://www.tensorflow.org/tensorboard)
- [tf.profiler Guide](https://www.tensorflow.org/guide/profiler)
- [KerasTuner Documentation](https://keras.io/keras_tuner/)
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research*.

## Código de compresión

```python
"""
Compression: TensorBoard, Callbacks, and Tuning
Demonstrates custom callbacks, TensorBoard logging, and KerasTuner Hyperband.
Run: python compression_04.py
"""
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from datetime import datetime

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class TimingCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._start = tf.timestamp()
    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = float(tf.timestamp() - self._start)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    TimingCallback(),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True, monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
]

class TunedMLP(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hp.Int('units', 64, 256, step=64), activation='relu'),
            keras.layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)),
            keras.layers.Dense(10, activation='softmax'),
        ])
        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

tuner = kt.Hyperband(TunedMLP(), objective='val_accuracy', max_epochs=20, factor=3,
                     directory='kt_dir', project_name='compression', overwrite=True)
tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=callbacks)

best = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best)
history = model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=callbacks)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Best hyperparameters: units={best.get('units')}, "
      f"dropout={best.get('dropout'):.2f}, lr={best.get('lr'):.2e}")
```
