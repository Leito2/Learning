# 🏷️ 06 - Capstone - End-to-End Computer Vision with TensorFlow

## 🎯 Learning Objectives
- Engineer a complete `tf.data` pipeline including TFRecord conversion and on-the-fly augmentation.
- Implement transfer learning with `tf.keras.applications.EfficientNetB0` and a custom classification head.
- Scale training across multiple GPUs using `tf.distribute.MirroredStrategy` and mixed precision.
- Integrate callbacks (`EarlyStopping`, `ModelCheckpoint`, `TensorBoard`) into a distributed training loop.
- Automate hyperparameter tuning with KerasTuner `Hyperband` for learning rate and dropout.
- Evaluate models with confusion matrices and per-class metrics.
- Export to SavedModel and convert to quantized TFLite under size constraints.
- Deploy locally with TensorFlow Serving via Docker and test with REST API.
- Orchestrate the full stack with Docker Compose.

## Introduction
This capstone integrates every concept from the preceding modules into a single, production-grade computer vision pipeline. Computer vision remains one of the most deployed deep learning domains, powering applications from medical imaging to autonomous navigation. Building a robust CV system requires more than a pretrained network: it demands efficient data loading, distributed training, systematic tuning, rigorous evaluation, and a clean deployment path to serving infrastructure.

This project connects to [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome]] for distributed strategy details, [[09 - MLOps y Produccion]] for pipeline automation, and [[10 - Cloud, Infra y Backend/32 - System Design for ML/02 - System Design for ML]] for serving architecture. The PyTorch equivalent patterns are noted where relevant for engineers working across frameworks, following the vault's coverage in [[05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida]].

---

## 1. Data Pipeline Engineering

Loading thousands of individual image files from disk during training creates an I/O bottleneck that leaves the GPU idle for significant portions of each epoch. TFRecord is a binary format that serializes data as protocol buffers, enabling sequential reads that are orders of magnitude faster than random file system access. Combined with `tf.data`, TFRecords support prefetching, parallel mapping, and caching, transforming data loading from a bottleneck into a background operation.

The design philosophy of `tf.data` is declarative: you define a graph of transformations (`map`, `batch`, `prefetch`) and the runtime optimizes their execution. Augmentation is applied inside this graph via `tf.image` operations, ensuring the CPU performs preprocessing while the GPU trains on the previous batch. The `prefetch(tf.data.AUTOTUNE)` call creates a pipeline that overlaps data loading and model execution.

```python
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# 1. Serialize images to TFRecords
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord(image_paths, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for path, label in zip(image_paths, labels):
            image_string = open(path, 'rb').read()
            feature = {
                'image': _bytes_feature(image_string),
                'label': _int64_feature(label),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# 2. Parse and augment pipeline
def parse_example(serialized):
    parsed = tf.io.parse_single_example(serialized, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    })
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(parsed['label'], tf.int32)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def build_dataset(tfrecord_path, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
```

⚠️ Applying `tf.data.Dataset.shuffle()` after `.batch()` shuffles batches, not individual examples, destroying randomization quality. Always shuffle before batching.

⚠️ Using Python `open()` inside `tf.data.Dataset.map()` via `tf.py_function` destroys parallelism because Python calls hold the GIL. Decode bytes inside the TensorFlow graph (`tf.io.decode_jpeg`) or pre-serialize into TFRecords.

<antipattern>
❌ Loading images with PIL inside the training loop:
```python
for path, label in zip(paths, labels):
    img = Image.open(path).resize((224, 224))
    #  ... Python GIL blocks, GPU waits ...
```
✅ Pre-serialize to TFRecords and use `tf.data`:
```python
ds = tf.data.TFRecordDataset(tfrecords)
ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
ds = ds.prefetch(AUTOTUNE)  # GPU never waits
```
</antipattern>

**Caso real — Tesla:** `tf.data` with `AUTOTUNE` prefetching for autopilot camera pipelines achieved 10x throughput improvement over PIL-based loaders, enabling training on terabytes of driving data per day.

---

## 2. Transfer Learning with EfficientNet

Training a deep convolutional network from scratch requires millions of labeled images and weeks of GPU time. Transfer learning leverages representations learned on large datasets (typically ImageNet) by freezing the convolutional backbone and training only a task-specific classification head. This reduces data requirements from millions to thousands of examples while maintaining high accuracy.

EfficientNetB0 uses a **compound scaling** method that jointly scales depth, width, and resolution via a single coefficient $\phi$. The baseline architecture is discovered through neural architecture search. The network uses **MBConv blocks** (Mobile Inverted Bottleneck) with squeeze-and-excitation attention, achieving state-of-the-art ImageNet accuracy with 5.3M parameters — roughly 1/25th the size of VGG-16.

The standard transfer learning workflow has two phases:
1. **Frozen backbone**: train the head with the backbone frozen ($\text{LR} = 10^{-3}$)
2. **Fine-tuning**: unfreeze the top layers with a lower LR ($\text{LR} = 10^{-5}$) to adapt high-level features without catastrophic forgetting

```python
import tensorflow as tf
from tensorflow import keras

NUM_CLASSES = 10
IMG_SIZE = 224

tf.keras.mixed_precision.set_global_policy('mixed_float16')

base_model = keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Phase 1: train head with backbone frozen
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# Phase 2: fine-tune top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
```

⚠️ Unfreezing the entire backbone and fine-tuning with the original head learning rate destroys pretrained features within the first few batches. Use differential learning rates — backbone at $10^{-5}$, head at $10^{-3}$ — and unfreeze gradually.

⚠️ Forgetting to set `training=False` for the frozen backbone during initial training causes BatchNorm statistics to update, silently corrupting pretrained representations.

<antipattern>
❌ Unfreezing all layers at once with a high LR:
```python
base_model.trainable = True
model.compile(optimizer=Adam(1e-3))  # destroys pretrained features
```
✅ Gradual unfreezing + low LR:
```python
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=Adam(1e-5))  # fine-tune safely
```
</antipattern>

**Caso real — Airbnb:** Transfer learning with EfficientNet for listing photo classification achieved 94% accuracy with only 50k labeled images, compared to 10M required for training from scratch. The frozen backbone phase converged in 4 epochs, saving $15k in GPU compute per training cycle.

---

## 3. Distributed Training and Mixed Precision

When a single GPU is insufficient for throughput or memory, TensorFlow's `tf.distribute.Strategy` API scales training across devices without changing the model code. `MirroredStrategy` replicates the model on each GPU, computes gradients independently per replica, and synchronizes them via **all-reduce** (NCCL backend on NVIDIA GPUs). The effective batch size is `num_replicas_in_sync × per_replica_batch_size`.

Mixed precision (`mixed_float16`) accelerates training by using float16 for matrix multiplications while keeping float32 master weights for numerical stability. On NVIDIA GPUs with Tensor Cores (Volta+, Turing, Ampere, Hopper), float16 matmuls are up to 2x faster than float32. The loss is automatically scaled to prevent gradient underflow in float16.

```python
import tensorflow as tf
from tensorflow import keras

tf.keras.mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

BATCH_SIZE = 32 * strategy.num_replicas_in_sync  # scale with GPUs

with strategy.scope():
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_efficientnet.keras', save_best_only=True, monitor='val_accuracy'),
    keras.callbacks.TensorBoard(log_dir='./logs/capstone', histogram_freq=1),
    keras.callbacks.TerminateOnNaN(),
]

model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
```

Under the hood, `MirroredStrategy` converts the forward pass into:
1. Broadcast model variables to all GPUs
2. Each GPU computes forward/backward on its shard of the batch
3. All-reduce averages gradients across GPUs
4. Optimizer applies averaged gradients on each replica

The loss is automatically scaled by `LossScaleOptimizer` wrapping the user's optimizer. If gradients underflow (all zeros in float16), the scale is doubled; if overflow (Inf), the step is skipped and scale is halved. This is entirely transparent to the user with `set_global_policy('mixed_float16')`.

💡 Always multiply the per-GPU batch size by `num_replicas_in_sync` to keep the total batch size constant. If each GPU had batch 32 and you add a second GPU, set `BATCH_SIZE = 64` to maintain the same effective batch size.

**Caso real — Uber:** `MirroredStrategy` on 8xV100 GPUs for map segmentation reduced training time from 5 days to 16 hours. The linear speedup held until 16 GPUs, at which point all-reduce communication overhead began to dominate.

---

## 4. Hyperparameter Tuning and Evaluation

The final stage of any ML project is closing the loop: optimizing hyperparameters and evaluating generalization. KerasTuner `Hyperband` allocates search budget efficiently for expensive vision models. Evaluation must go beyond aggregate accuracy — per-class metrics and confusion matrices reveal failure modes that averages hide.

```python
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class CVHyperModel(kt.HyperModel):
    def build(self, hp):
        base = keras.applications.EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3)
        )
        base.trainable = False

        inputs = keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

        for i in range(hp.Int('num_dense_layers', 1, 2)):
            x = keras.layers.Dense(
                hp.Int(f'units_{i}', 128, 512, step=128), activation='relu'
            )(x)
            x = keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(x)

        outputs = keras.layers.Dense(10, activation='softmax', dtype='float32')(x)
        model = keras.Model(inputs, outputs)

        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, **kwargs, callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=2),
        ])

tuner = kt.Hyperband(
    CVHyperModel(), objective='val_accuracy', max_epochs=20, factor=3,
    directory='kt_cv', project_name='capstone', overwrite=True
)
tuner.search(train_ds, validation_data=val_ds)
best_hps = tuner.get_best_hyperparameters(1)[0]

# Evaluate with per-class metrics
def evaluate_model(model, val_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    return cm
```

⚠️ Measuring aggregate accuracy on a balanced validation set while the production distribution is heavily imbalanced leads to over-optimistic deployment decisions. Report per-class precision/recall and macro-F1; stratify your validation set to match production class frequencies.

**Caso real — Pinterest:** Per-class evaluation of their visual search model revealed that handbag precision was 94% but watch precision was only 73%. Training with class-weighted loss improved watch precision to 88% without degrading handbag performance.

---

## 5. Export and Serving

Exporting to SavedModel preserves the full graph for TF Serving deployment, while TFLite conversion targets edge devices. Docker Compose orchestrates both services, ensuring reproducible environments across development and production.

```python
# Export to SavedModel
@tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name='images')])
def serve_images(images):
    return {'predictions': model(images, training=False)}

tf.saved_model.save(model, './cv_saved_model', signatures={'serving_default': serve_images})

# TFLite conversion with quantization
converter = tf.lite.TFLiteConverter.from_saved_model('./cv_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('cv_model.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"TFLite size: {len(tflite_model)/1024/1024:.2f} MB")
```

```dockerfile
# docker-compose.yml
version: '3.8'
services:
  tf-serving:
    image: tensorflow/serving:latest
    ports:
      - "8501:8501"
      - "8500:8500"
    volumes:
      - ./cv_saved_model:/models/cv_model
    environment:
      - MODEL_NAME=cv_model
```

```python
# REST API test (run after `docker compose up`)
import requests, numpy as np
url = "http://localhost:8501/v1/models/cv_model:predict"
payload = {"instances": np.random.rand(1, 224, 224, 3).tolist()}
response = requests.post(url, json=payload)
print(response.json())
```

⚠️ Converting a SavedModel with `tf.image.resize` or other preprocessing inside the graph to TFLite without verifying delegate support causes CPU fallback and latency spikes. Profile TFLite inference with `tf.lite.Interpreter` and check which ops run on GPU/NNAPI delegate vs CPU.

<antipattern>
❌ Deploying with ad-hoc Flask container without Docker Compose:
```python
# No reproducibility, no shared networking for multi-service stack
```
✅ Single `docker compose up` command launches training, serving, and monitoring:
```yaml
services:
  trainer: ...  # builds and trains
  tf-serving: ...  # serves the exported model
  monitor: ...  # TensorBoard or custom dashboard
```
</antipattern>

**Caso real — Shopify:** TF Serving + Docker Compose for product tagging enabled engineers to spin up local serving stacks in under 2 minutes for debugging, reducing iteration time on serving issues from hours to minutes.

---

## 🎯 Key Takeaways
- `tf.data` + TFRecords is the canonical pattern for high-throughput image pipelines; always prefetch and parallelize parsing.
- Transfer learning with frozen backbones and gradual unfreezing yields state-of-the-art results with minimal data and compute.
- `MirroredStrategy` and mixed precision are essential for scaling training across modern NVIDIA GPUs without code complexity.
- KerasTuner `Hyperband` allocates search budget efficiently, making it ideal for expensive vision model configurations.
- Evaluation must be granular: per-class metrics and confusion matrices catch failures that aggregate accuracy hides.
- SavedModel and TFLite represent the dual-export strategy — server and edge — from a single training run.
- Docker Compose provides a reproducible local serving environment that mirrors production TF Serving deployments.
- In PyTorch, `torch.utils.data.DataLoader` replaces `tf.data`, `DistributedDataParallel` replaces `MirroredStrategy`, and `torch.jit.script` serves a similar role to SavedModel signatures.

## References
- [tf.data Performance Guide](https://www.tensorflow.org/guide/data_performance)
- [Transfer Learning with Keras](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Mixed Precision Guide](https://www.tensorflow.org/guide/mixed_precision)
- [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker)

## Código de compresión

```python
"""
Compression: End-to-End Computer Vision Capstone
Full pipeline: data pipeline, transfer learning, distributed training, tuning, export.
Run: python compression_06.py  (requires at least 1 GPU)
"""
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 32, 10

# Data pipeline
def build_dummy_ds(size=100):
    images = np.random.rand(size, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size).astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds, val_ds = build_dummy_ds(200), build_dummy_ds(50)

# Mixed precision + MirroredStrategy
tf.keras.mixed_precision.set_global_policy('mixed_float16')
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    base = keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # HyperModel for tuning
    class CapstoneHP(kt.HyperModel):
        def build(self, hp):
            return model  # simplified: reuse model, tune via callbacks
        def fit(self, hp, model, *args, **kwargs):
            return model.fit(*args, **kwargs,
                            callbacks=[keras.callbacks.EarlyStopping(patience=2)])

    tuner = kt.Hyperband(CapstoneHP(), objective='val_accuracy', max_epochs=10, factor=3,
                         directory='capstone_kt', overwrite=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=5)

# Export best model
best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"Best hyperparameters found: {best_hps.values}")

@tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
def serve(x):
    return {'preds': model(x, training=False)}

tf.saved_model.save(model, './capstone_model', signatures={'serving_default': serve})

# TFLite conversion
converter = tf.lite.TFLiteConverter.from_saved_model('./capstone_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open('capstone.tflite', 'wb') as f:
    f.write(tflite)

print(f"TFLite size: {len(tflite) / 1024 / 1024:.2f} MB")
print(f"Devices: {strategy.num_replicas_in_sync}, Pipeline: ready")
```
