# 🏷️ 05 - TensorFlow Production Ecosystem

## 🎯 Learning Objectives
- Understand the SavedModel format, its directory structure, and signature definitions.
- Save and load models using `tf.saved_model` APIs with explicit signatures.
- Deploy models with TensorFlow Serving via REST and gRPC APIs.
- Convert trained models to TensorFlow Lite with post-training and quantization-aware techniques.
- Export Keras models to TensorFlow.js for browser inference.
- Apply `tf.Transform` for consistent preprocessing across training and serving.
- Map TFX pipeline concepts to the broader MLOps lifecycle.

## Introduction
A trained model in a Jupyter notebook has zero business value until it serves predictions in production. The gap between research artifact and production system is where most ML projects fail. TensorFlow's production ecosystem provides a standardized, language-agnostic serialization format (SavedModel), a high-performance serving system (TF Serving), and lightweight runtimes for edge and mobile (TFLite). Together, these tools form a continuum from data-center GPU clusters to microcontroller inference, all sharing the same core graph representation.

This note is the bridge between model training and the MLOps practices covered in [[09 - MLOps y Produccion]], the distributed infrastructure topics in [[10 - Cloud, Infra y Backend/29 - Distributed ML Infrastructure/00 - Welcome]], and the system design principles in [[10 - Cloud, Infra y Backend/32 - System Design for ML/02 - System Design for ML]]. While [[05 - Deep Learning y Computer Vision/03 - Deep Learning con PyTorch/00 - Bienvenida]] covers PyTorch deployment via TorchScript and ONNX, this note focuses on the TensorFlow-native path.

---

## 1. SavedModel Format and Signatures

Before SavedModel, TensorFlow used checkpoints — raw variable dumps that required the original Python code to reconstruct the graph. This was fragile: a refactor in layer naming or a version mismatch between training and serving code could render a multi-day training run unloadable. SavedModel solves this by bundling the complete TensorFlow program: the graph definition, variable values, and metadata about how to invoke the model.

The design is inspired by language-agnostic serialization formats like Protocol Buffers and ONNX. A SavedModel directory contains:

```
my_saved_model/
├── saved_model.pb          # serialized MetaGraphDef (graph + signatures)
├── variables/
│   ├── variables.data-00000-of-00001   # sharded weight values
│   └── variables.index                 # weight name-to-shard mapping
└── assets/                             # vocabulary files, lookup tables (optional)
```

The critical abstraction is the **signature**: a named set of input and output tensors that defines a callable interface. This allows a model trained in Python to be loaded in C++, Java, or Go without any source code dependency. Standard signatures include `serving_default`, `predict`, `classify`, and `regress`, each enforcing a specific contract between producer and consumer.

A signature maps input tensor names to `TensorInfo` protos that encode dtype, shape, and tensor name. When a client sends a request, TF Serving uses the signature to map JSON/gRPC field names to the correct graph inputs, execute the graph, and map outputs back to named fields.

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define an explicit serving signature
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 10], dtype=tf.float32, name='inputs')
])
def serve_fn(inputs):
    return {'predictions': model(inputs, training=False)}

tf.saved_model.save(
    model,
    export_dir='./my_saved_model',
    signatures={'serving_default': serve_fn}
)

# Load and inspect in a different process (no original Python code needed)
loaded = tf.saved_model.load('./my_saved_model')
print(list(loaded.signatures.keys()))  # ['serving_default']

infer = loaded.signatures['serving_default']
print(infer.structured_input_signature)
print(infer.structured_outputs)

# Direct inference
sample = tf.constant([[1.0]*10], dtype=tf.float32)
result = infer(inputs=sample)
print(result['predictions'])
```

⚠️ Saving a Keras model with `model.save('path.keras')` inside a training script that has custom layers or loss functions, then loading it in a different Python environment where those classes are undefined, raises `ValueError: Unknown layer`. Use `tf.saved_model.save()` for serving exports (no code dependency) and `model.save('path.keras')` only when the identical Python environment is guaranteed.

⚠️ Forgetting `training=False` inside a `@tf.function` serving signature causes dropout and batch normalization to behave incorrectly at inference time. Always pass `training=False` explicitly in serving functions.

<antipattern>
❌ Using a single `@tf.function` that mixes training and serving logic:
```python
@tf.function
def predict(x, training=True):  # default True is dangerous
    return model(x, training=training)
```
✅ Create separate serving-dedicated signatures with `training=False` hardcoded:
```python
@tf.function(input_signature=[TensorSpec([None, 10], tf.float32)])
def serving(x):
    return model(x, training=False)  # never train mode
```
</antipattern>

**Caso real — Uber:** Signature versioning for ranking models enabled rollback to a previous signature within seconds during incidents, preventing cascading failures across downstream services that depend on specific output field names.

---

## 2. TensorFlow Serving

TensorFlow Serving (TFS) is a flexible, high-performance serving system for machine learning models designed for production environments. It was open-sourced by Google to solve the problem of deploying and versioning models at scale. Unlike Flask or FastAPI wrappers around `model.predict()`, TFS is written in C++ and optimized for throughput, supporting batching, model versioning, and A/B testing out of the box.

The architecture separates the **model lifecycle manager** (loads/unloads versions) from the **inference core** (executes graphs). Client communication happens via REST (port 8501, easy debugging) or gRPC (port 8500, higher performance). TFS discovers models by monitoring a file system path, enabling seamless hot-swapping: dropping a new SavedModel into a versioned subdirectory makes TFS automatically route traffic to it.

The batching layer is a critical performance feature. Individual requests are queued and combined into a single batch, amortizing GPU kernel launch overhead across multiple inferences. The `max_batch_size` and `batch_timeout_micros` parameters control the trade-off between throughput (large batches) and latency (short timeouts).

```bash
# Start TensorFlow Serving with Docker
docker run -p 8501:8501 -p 8500:8500 \
  --mount type=bind,source=/path/to/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving:latest

# REST API prediction (language-agnostic, debuggable)
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}'
```

```python
# gRPC client (lower latency, streaming support)
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'
request.inputs['inputs'].CopyFrom(
    tf.make_tensor_proto([[1.0]*10], shape=[1, 10], dtype=tf.float32)
)

response = stub.Predict(request, timeout=10.0)
print(response.outputs['predictions'])
```

Model versioning follows a simple directory convention. TFS serves the latest version by default, but clients can pin specific versions:

```
/models/my_model/
├── 1/                     # version 1 (served)
├── 2/                     # version 2 (auto-discovered, served)
└── 3/                     # version 3 (being loaded, soon served)
```

Clients can specify `versions/1` or `labels/stable` in the URL to pin behavior. A/B testing is achieved by running multiple TFS instances with different version configurations behind a load balancer.

⚠️ Using the REST API under high load without batching causes GPU underutilization and latency spikes. Enable `batching_parameters_config` in TFS or switch to gRPC with client-side batching.

⚠️ Exposing TFS directly to the public internet without authentication or rate limiting is a security risk. Place TFS behind a reverse proxy (Envoy/Nginx) or API gateway.

<antipattern>
❌ Wrapping `model.predict()` in a Flask server:
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return jsonify(model.predict(data['instances']).tolist())
# Single-threaded, no batching, Python GIL bottleneck
```
✅ Using TFS:
```bash
# C++ inference engine, automatic batching, version management
docker run tensorflow/serving --model_name=my_model
```
</antipattern>

**Caso real — LinkedIn:** gRPC + model versioning for feed ranking enabled zero-downtime model updates with canary deployments. New versions received 5% of traffic initially, gradually ramping as confidence increased. **Caso real — Airbnb:** TFS with batching for listing ranking achieved 3x throughput improvement over unbatched Flask serving.

---

## 3. TensorFlow Lite and Edge Deployment

Not all inference happens in data centers. Mobile apps, browsers, microcontrollers, and IoT devices demand models that are small, fast, and hardware-agnostic. TensorFlow Lite (TFLite) addresses this by converting SavedModels into a flatbuffer format optimized for on-device execution, with interpreters written in C++, Java, and Python.

The conversion pipeline supports three quantization modes, each offering a different accuracy/size trade-off:

| Method | Weight type | Activation type | Size reduction | Accuracy impact |
|--------|------------|-----------------|----------------|-----------------|
| Dynamic range | INT8 | FP32 | ~4x | Minimal |
| Float16 | FP16 | FP32 | ~2x | Negligible |
| Full integer | INT8 | INT8 | ~4x | Small (+calibration) |

Full integer quantization requires a **representative dataset** (typically 100-500 training samples) for calibration. The converter measures the dynamic range of each activation tensor and maps it to INT8, minimizing quantization error. For models with sensitive outputs (softmax probabilities, regression targets), **quantization-aware training** (QAT) simulates quantization noise during training, yielding significantly better accuracy at INT8 than post-training quantization.

```python
import tensorflow as tf

# Dynamic range quantization (weights → INT8, activations → FP32)
converter = tf.lite.TFLiteConverter.from_saved_model('./my_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model_dynamic.tflite', 'wb') as f:
    f.write(tflite_model)

# Full integer quantization (requires representative dataset)
converter = tf.lite.TFLiteConverter.from_saved_model('./my_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for _ in range(100):
        data = tf.random.normal([1, 10])
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_int8_model = converter.convert()
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_int8_model)

# On-device inference with TFLite Interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], sample.astype(np.float32))
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

For browser deployment, TensorFlow.js provides a converter that translates SavedModels or Keras H5 files into the TF.js Layers or Graph model format:

```bash
# Converter CLI (installed via pip install tensorflowjs)
tensorflowjs_converter \
  --input_format=keras \
  --output_format=tfjs_graph_model \
  ./model.keras \
  ./tfjs_model
```

⚠️ Applying post-training quantization to a model with sensitive outputs without calibration can cause severe accuracy drops. Always evaluate the quantized model on a held-out test set; use QAT if accuracy degradation exceeds 1%.

⚠️ Using unsupported TensorFlow ops in a model destined for TFLite causes conversion failures. Run `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]` to fail fast, or use `SELECT_TF_OPS` as a fallback with increased binary size.

<antipattern>
❌ Training without quantization awareness, then applying aggressive full-integer quantization post-hoc:
```python
# Accuracy drops from 92% → 74% on a fine-grained classification task
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
```
✅ Use quantization-aware training for targets requiring INT8:
```python
# tfmot.quantization.keras.quantize_model(model)
# Then convert with INT8 ops; accuracy stays at ~91%
```
</antipattern>

**Caso real — Google Lens:** TFLite on Android with the NNAPI delegate achieves real-time object recognition at 30 FPS on mid-range phones. The model goes through full-integer quantization with calibration on 200 representative images, reducing size from 25MB to 6.5MB with <0.5% accuracy loss.

---

## 4. TF Transform and TFX Pipeline Orchestration

A common source of production model failure is **training-serving skew**: preprocessing logic (normalization, tokenization, feature crosses) implemented differently during training and inference. TF Transform (`tft`) addresses this by compiling transformations into the TensorFlow graph, ensuring the exact same computation runs in both phases.

`tft.scale_to_z_score(x)` computes mean and variance over the entire training dataset (not per-batch), embeds these statistics as constants in the graph, and applies the same normalization during serving. This eliminates the common bug where training uses `dataset-wide StandardScaler` while serving computes per-request normalization.

```python
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    x = inputs['x']
    x_normalized = tft.scale_to_z_score(x)      # dataset-wide stats
    x_bucketed = tft.bucketize(x, num_buckets=5)  # quantile-based
    x_cross = tft.cross([x_bucketed, inputs['cat']], hash_bucket_size=100)
    return {
        'x_normalized': x_normalized,
        'x_crossed': x_cross,
        'cat': inputs['cat'],  # pass through
    }
```

TFX (TensorFlow Extended) orchestrates the end-to-end ML pipeline, from raw data to deployed model. Its stages form a DAG:

```
ExampleGen → StatisticsGen → SchemaGen → Transform → Trainer → Evaluator → Pusher → TF Serving
```

- **ExampleGen**: ingests data, splits into train/eval, converts to TFRecords
- **StatisticsGen**: computes descriptive statistics (mean, min, max, quantiles)
- **SchemaGen**: infers a schema (dtypes, domains, constraints) automatically
- **Transform**: applies `tft` preprocessing, saving the transform graph
- **Trainer**: trains the model using the transformed data
- **Tuner**: performs HPO (optional stage, connects to KerasTuner)
- **Evaluator**: validates model against thresholds (fairness, slice-based metrics)
- **Pusher**: exports the validated SavedModel to a serving destination

The critical insight is that `Transform` and `Trainer` share the `preprocessing_fn`, ensuring zero skew. The transform graph is bundled with the SavedModel so TF Serving applies preprocessing automatically.

💡 TFX uses Apache Beam for distributed data processing, meaning the pipeline scales from a single machine to a large cluster with the same code. For smaller teams, `tfx` can run in local mode (`LocalDagRunner`) before migrating to cloud (`BeamDagRunner`).

**Caso real — Twitter:** TFX pipelines for timeline ranking automated retraining and deployment, reducing model staleness from days to hours. The `Evaluator` stage caught two regressions (a feature gap bug and a label leak) before they reached production, saving an estimated 12 engineer-days per incident.

---

## 🎯 Key Takeaways
- SavedModel is the canonical TensorFlow serialization format: language-agnostic, signature-based, and self-contained; prefer it over `model.save('path.keras')` for serving.
- Always define explicit `SignatureDef` objects so serving systems know exact input/output shapes and names.
- TensorFlow Serving provides production-grade model serving with versioning, batching, and hot-reloading; prefer gRPC for latency-sensitive workloads.
- TFLite enables edge deployment through flatbuffer conversion and quantization; choose dynamic-range, float16, or full-integer based on hardware constraints and accuracy requirements.
- Quantization-aware training (QAT) is essential when post-training quantization degrades accuracy beyond acceptable thresholds.
- TF Transform embeds preprocessing into the graph, eliminating training-serving skew — a leading cause of production model degradation.
- TFX pipelines standardize the path from raw data to served model, integrating validation, transformation, tuning, and deployment into a reproducible DAG.
- In PyTorch, `torch.jit.script` traces models for C++ deployment but lacks the standardized directory format of SavedModel; ONNX serves as the cross-framework bridge instead.

## References
- [TensorFlow SavedModel Guide](https://www.tensorflow.org/guide/saved_model)
- [TensorFlow Serving Documentation](https://www.tensorflow.org/tfx/serving/serving_config)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [TensorFlow.js Converter](https://www.tensorflow.org/js/guide/conversion)
- [TF Transform Documentation](https://www.tensorflow.org/tfx/transform/get_started)
- [TFX Pipeline Overview](https://www.tensorflow.org/tfx/guide)

## Código de compresión

```python
"""
Compression: TensorFlow Production Ecosystem
Save, serve, convert, and deploy a Keras model end-to-end.
Run: python compression_05.py
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Train a small model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Simulate training with random data
x_dummy = np.random.rand(1000, 10).astype(np.float32)
y_dummy = keras.utils.to_categorical(np.random.randint(0, 10, 1000), 10)
model.fit(x_dummy, y_dummy, epochs=5, verbose=0)

# SaveModel with explicit signature
@tf.function(input_signature=[tf.TensorSpec([None, 10], tf.float32, name='inputs')])
def serve(inputs):
    return {'outputs': model(inputs, training=False)}

tf.saved_model.save(model, './saved_model', signatures={'serving_default': serve})

# Verify
loaded = tf.saved_model.load('./saved_model')
print("Signatures:", list(loaded.signatures.keys()))

# TFLite conversion (dynamic range quantization)
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# TFLite inference test
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
sample = np.random.rand(1, 10).astype(np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], sample)
interpreter.invoke()
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print(f"TFLite output shape: {output.shape}, sum: {output.sum():.2f}")
print(f"SavedModel size: {os.path.getsize('./saved_model/saved_model.pb') / 1024:.1f} KB")
print(f"TFLite size: {len(tflite_model) / 1024:.1f} KB")
```
